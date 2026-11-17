"""GPU/TPU full-step M7-JAX throughput, memory, and parity benchmark.

Measures forward (`jax.jit`) and reverse-mode (`jax.jit(jax.value_and_grad)`)
wall-clock throughput, XLA static buffer allocations, runtime peak HBM usage,
and numerical parity across AoS (`step_all`) and SoA (`step_all_soa`) kernels
at GCM column-grid resolutions (default T63L47 = 47 x 18,432 = 866,304 cells).

Each timed configuration executes in an isolated subprocess so JAX's `x64`
mode and `device.memory_stats()['peak_bytes_in_use']` are uncontaminated
across runs, and out-of-memory failures on unchunked baselines are recorded
cleanly without aborting the suite.

Example usage:
    .venv/bin/python -m benchmarks.benchmark_accelerator \
        --baseline-src /path/to/baseline/src \
        --output docs/evidence/m7-accelerator-benchmark-gpu.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KAPPA_LUT_PATH = ROOT / "reference" / "input" / "lut_kappa.nc"

RESOLUTIONS = {
    "T63L47": (47, 18432),    # 866,304 cells (standard ECHAM/JCM grid)
    "T106L95": (95, 51200),   # 4,864,000 cells (high-resolution stress test)
    "smoke": (8, 512),        # 4,096 cells (local CPU verification)
}


def build_realistic_grid_np(nlev: int, ncol: int, dtype_name: str, seed: int = 42):
    """Deterministic multi-regime atmospheric aerosol state on `(nlev, ncol)`."""
    rng = np.random.default_rng(seed)
    dt = np.float64 if dtype_name == "float64" else np.float32
    shape = (nlev, ncol)

    # Vertical pressure [Pa] and temperature [K] profiles with horizontal wave structure.
    sigma = np.linspace(0.02, 0.99, nlev)
    lon_wave = np.sin(np.linspace(0.0, 4.0 * np.pi, ncol))
    lat_wave = np.cos(np.linspace(-0.5 * np.pi, 0.5 * np.pi, ncol))
    p = (1.013e5 * sigma[:, None] * (1.0 + 0.03 * lon_wave[None, :])).astype(dt)
    t_surf = 260.0 + 38.0 * np.maximum(lat_wave, 0.0)
    t = np.clip(
        t_surf[None, :] - 65.0 * (1.0 - sigma[:, None]) + 3.0 * lon_wave[None, :],
        195.0,
        310.0,
    ).astype(dt)

    rh = np.clip(
        0.55 + 0.35 * np.sin(3.0 * sigma[:, None] + lon_wave[None, :]),
        0.05,
        0.98,
    ).astype(dt)
    # Include clear, partly cloudy, and overcast cells.
    cc_raw = 0.45 + 0.65 * np.cos(2.5 * sigma[:, None] - 1.7 * lon_wave[None, :])
    cloud_cover = np.clip(cc_raw, 0.0, 1.0).astype(dt)

    gas = (1.0e7 * (0.2 + 8.0 * rng.uniform(0.0, 1.0, shape))).astype(dt)
    production = (2.0e4 * rng.uniform(0.0, 1.5, shape)).astype(dt)

    # 18 native mass components: 4 SO4 [molec cm-3], 14 non-SO4 [ug m-3].
    typical_mass = np.array(
        [
            4.0e7, 2.5e8, 6.0e8, 8.0e7,          # SO4: NS, KS, AS, CS
            0.08, 0.15, 0.02, 0.12,               # BC:  KS, AS, CS, KI
            0.30, 0.60, 0.08, 0.35,               # OC:  KS, AS, CS, KI
            0.40, 3.50,                           # SS:  AS, CS
            0.25, 2.00, 0.60, 4.00,               # DU:  AS, CS, AI, CI
        ],
        dtype=np.float64,
    )
    typical_num = np.array(
        [450.0, 600.0, 180.0, 1.2, 220.0, 0.8, 0.15],
        dtype=np.float64,
    )

    vertical_decay = (0.15 + 0.85 * sigma)[:, None]
    mass_aos = np.empty((*shape, 18), dtype=dt)
    for k in range(18):
        mass_aos[..., k] = (
            typical_mass[k] * vertical_decay * rng.uniform(0.1, 2.5, shape)
        ).astype(dt)
    num_aos = np.empty((*shape, 7), dtype=dt)
    for m in range(7):
        scale_v = (1.2 - 0.8 * sigma)[:, None] if m == 0 else vertical_decay
        num_aos[..., m] = (
            typical_num[m] * scale_v * rng.uniform(0.1, 2.5, shape)
        ).astype(dt)

    ipr = (2.0 + 18.0 * (1.0 - sigma[:, None]) * (0.4 + 0.6 * np.abs(lat_wave)[None, :])).astype(dt)
    voc = (5.0e6 * vertical_decay * rng.uniform(0.1, 2.0, shape)).astype(dt)
    forest = np.clip(0.5 + 0.5 * lon_wave, 0.0, 1.0)[None, :].repeat(nlev, axis=0).astype(dt)
    pbl_mask = (sigma[:, None] >= 0.80) & (np.ones(shape, dtype=bool))

    return {
        "pressure": p,
        "temperature": t,
        "relative_humidity": rh,
        "gas": gas,
        "mass": mass_aos,
        "number": num_aos,
        "production": production,
        "timestep": dt(720.0),
        "cloud_cover": cloud_cover,
        "ion_pair_rate": ipr,
        "monoterpene_oxidation_rate": voc,
        "ho2_plus_ro2_rate": (0.6 * voc).astype(dt),
        "forest_fraction": forest,
        "in_pbl": pbl_mask,
    }


def build_kazil_table_np(dtype_name: str):
    """Authentic-shape (40, 40, 40, 20, 40) Kazil-Lovejoy table (~205 MB f32)."""
    dt = np.float64 if dtype_name == "float64" else np.float32
    t_ax = np.linspace(180.0, 320.0, 40, dtype=dt)
    rh_ax = np.linspace(0.5, 99.5, 40, dtype=dt)
    h2so4_ax = np.geomspace(1.0e5, 1.0e9, 40).astype(dt)
    ion_ax = np.linspace(0.0, 40.0, 20, dtype=dt)
    cs_ax = np.geomspace(1.0e-5, 1.0e-1, 40).astype(dt)
    # Smooth physical-range log10 formation rate [-12, 3].
    log_pfr = (
        -8.0
        - 0.03 * (t_ax[:, None, None, None, None] - 240.0)
        + 0.02 * rh_ax[None, :, None, None, None]
        + 1.5 * np.log10(h2so4_ax.astype(np.float64) / 1.0e7).astype(dt)[None, None, :, None, None]
        + 0.05 * ion_ax[None, None, None, :, None]
        - 0.4 * np.log10(cs_ax.astype(np.float64) / 1.0e-3).astype(dt)[None, None, None, None, :]
    ).astype(dt)
    return t_ax, rh_ax, h2so4_ax, ion_ax, cs_ax, log_pfr


def run_worker(spec: dict) -> dict:
    """Execute one isolated benchmark configuration inside a fresh process."""
    src_dir = spec["src_dir"]
    sys.path.insert(0, src_dir)

    import jax
    import jax.numpy as jnp

    dtype_name = spec["dtype"]
    jax.config.update("jax_enable_x64", dtype_name == "float64")
    j_dtype = jnp.float64 if dtype_name == "float64" else jnp.float32

    from m7_jax.model import FullState, step_all
    from m7_jax.nucleation import KazilLovejoyTable
    from m7_jax.properties import KappaTable, load_kappa_table

    has_soa = False
    try:
        from m7_jax.model import SoAState, step_all_soa
        from m7_jax.nucleation import prepare_kazil_lovejoy_table
        from m7_jax.properties import prepare_kappa_table
        has_soa = True
    except ImportError:
        pass

    nlev, ncol = spec["nlev"], spec["ncol"]
    ncells = nlev * ncol
    raw = build_realistic_grid_np(nlev, ncol, dtype_name)

    kappa_raw = load_kappa_table(KAPPA_LUT_PATH)
    nsnucl = spec["nucleation_scheme"]
    nnucl2 = 1 if nsnucl == 2 else 0

    variant = spec["variant"]
    chunk_size = spec.get("chunk_size")
    mode = spec["mode"]  # "forward" or "grad"

    dev = jax.devices()[0]
    try:
        if variant == "aos_step_all":
            if chunk_size is not None:
                raise ValueError("aos_step_all does not support chunk_size")
            kappa = KappaTable(*(jnp.asarray(x, dtype=j_dtype) for x in kappa_raw))
            kazil = None
            if nsnucl == 2:
                axes_and_pfr = build_kazil_table_np(dtype_name)
                kazil = KazilLovejoyTable(*(jnp.asarray(a, dtype=j_dtype) for a in axes_and_pfr))

            dev_inputs = {k: jax.device_put(jnp.asarray(v)) for k, v in raw.items()}

            def forward_fn(gas, mass, number):
                res = step_all(
                    FullState(mass, number, gas),
                    dev_inputs["temperature"],
                    dev_inputs["pressure"],
                    dev_inputs["relative_humidity"],
                    dev_inputs["timestep"],
                    production=dev_inputs["production"],
                    cloud_fraction=dev_inputs["cloud_cover"],
                    table=kappa,
                    forest_fraction=dev_inputs["forest_fraction"] if nnucl2 == 1 else j_dtype(0.0),
                    in_pbl=dev_inputs["in_pbl"] if nnucl2 == 1 else False,
                    nucleation_scheme=nsnucl,
                    organic_scheme=nnucl2,
                    ion_pair_rate=dev_inputs["ion_pair_rate"] if nsnucl == 2 else j_dtype(0.0),
                    kazil_lovejoy_table=kazil,
                )
                return res.state.gas, res.state.mass, res.state.number, res.properties.wet_radius

            arg_gas = dev_inputs["gas"]
            arg_mass = dev_inputs["mass"]
            arg_num = dev_inputs["number"]

            if mode == "forward":
                compiled_fn = jax.jit(forward_fn)
                call_args = (arg_gas, arg_mass, arg_num)
            else:
                def loss_fn(gas, mass, number):
                    g, m, n, rw = forward_fn(gas, mass, number)
                    return (
                        jnp.sum(g * j_dtype(1e-8))
                        + jnp.sum(m * j_dtype(1e-8))
                        + jnp.sum(n * j_dtype(1e-3))
                        + jnp.sum(rw * j_dtype(1e4))
                    )
                compiled_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))
                call_args = (arg_gas, arg_mass, arg_num)

        elif variant in ("soa_full", "soa_jcm"):
            if not has_soa:
                return {"status": "unsupported", "reason": "step_all_soa not present in target src"}
            kappa_prep = prepare_kappa_table(kappa_raw, dtype=j_dtype)
            kazil_prep = None
            if nsnucl == 2:
                axes_and_pfr = build_kazil_table_np(dtype_name)
                kt = KazilLovejoyTable(*(jnp.asarray(a, dtype=j_dtype) for a in axes_and_pfr))
                kazil_prep = prepare_kazil_lovejoy_table(kt, dtype=j_dtype)

            compute_valid = variant == "soa_full"
            compute_water = variant == "soa_full"

            mass_t = tuple(
                jax.device_put(jnp.asarray(raw["mass"][..., k])) for k in range(18)
            )
            num_t = tuple(
                jax.device_put(jnp.asarray(raw["number"][..., m])) for m in range(7)
            )
            dev_scalars = {
                k: jax.device_put(jnp.asarray(v))
                for k, v in raw.items()
                if k not in ("mass", "number")
            }

            def forward_soa(gas, mass_tuple, num_tuple):
                res = step_all_soa(
                    SoAState(mass_tuple, num_tuple, gas),
                    dev_scalars["temperature"],
                    dev_scalars["pressure"],
                    dev_scalars["relative_humidity"],
                    dev_scalars["timestep"],
                    production=dev_scalars["production"],
                    cloud_fraction=dev_scalars["cloud_cover"],
                    table=kappa_prep,
                    forest_fraction=dev_scalars["forest_fraction"] if nnucl2 == 1 else j_dtype(0.0),
                    in_pbl=dev_scalars["in_pbl"] if nnucl2 == 1 else False,
                    nucleation_scheme=nsnucl,
                    organic_scheme=nnucl2,
                    ion_pair_rate=dev_scalars["ion_pair_rate"] if nsnucl == 2 else j_dtype(0.0),
                    kazil_lovejoy_table=kazil_prep,
                    compute_valid=compute_valid,
                    compute_water=compute_water,
                    chunk_size=chunk_size,
                )
                return res.state.gas, res.state.mass, res.state.number, res.properties.wet_radius

            arg_gas = dev_scalars["gas"]
            if mode == "forward":
                compiled_fn = jax.jit(forward_soa)
                call_args = (arg_gas, mass_t, num_t)
            else:
                def loss_soa(gas, mass_tuple, num_tuple):
                    g, m_tup, n_tup, rw_tup = forward_soa(gas, mass_tuple, num_tuple)
                    return (
                        jnp.sum(g * j_dtype(1e-8))
                        + sum(jnp.sum(mk * j_dtype(1e-8)) for mk in m_tup)
                        + sum(jnp.sum(nm * j_dtype(1e-3)) for nm in n_tup)
                        + sum(jnp.sum(rwm * j_dtype(1e4)) for rwm in rw_tup)
                    )
                compiled_fn = jax.jit(jax.value_and_grad(loss_soa, argnums=(0, 1, 2)))
                call_args = (arg_gas, mass_t, num_t)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Lower & static memory analysis.
        t0 = time.perf_counter()
        lowered = compiled_fn.lower(*call_args)
        compiled = lowered.compile()
        compile_only_s = time.perf_counter() - t0

        mem_info = {}
        try:
            ma = compiled.memory_analysis()
            if ma is not None:
                mem_info = {
                    "temp_mb": round(ma.temp_size_in_bytes / (1024 * 1024), 2),
                    "argument_mb": round(ma.argument_size_in_bytes / (1024 * 1024), 2),
                    "output_mb": round(ma.output_size_in_bytes / (1024 * 1024), 2),
                    "code_kb": round(ma.generated_code_size_in_bytes / 1024, 2),
                }
        except Exception:
            pass

        # Execute first call + warmup.
        t_first = time.perf_counter()
        out = compiled(*call_args)
        jax.block_until_ready(out)
        first_exec_s = time.perf_counter() - t_first

        # Optional numerical output dump for parity verification.
        if spec.get("save_npz"):
            if mode == "forward":
                g_out, m_out, n_out, rw_out = out
                if isinstance(m_out, tuple):
                    m_np = np.stack([np.asarray(x) for x in m_out], axis=-1)
                    n_np = np.stack([np.asarray(x) for x in n_out], axis=-1)
                    rw_np = np.stack([np.asarray(x) for x in rw_out], axis=-1)
                else:
                    m_np, n_np, rw_np = np.asarray(m_out), np.asarray(n_out), np.asarray(rw_out)
                np.savez(
                    spec["save_npz"],
                    gas=np.asarray(g_out),
                    mass=m_np,
                    number=n_np,
                    wet_radius=rw_np,
                )
            else:
                val, (g_gas, g_mass, g_num) = out
                if isinstance(g_mass, tuple):
                    gm_np = np.stack([np.asarray(x) for x in g_mass], axis=-1)
                    gn_np = np.stack([np.asarray(x) for x in g_num], axis=-1)
                else:
                    gm_np, gn_np = np.asarray(g_mass), np.asarray(g_num)
                np.savez(
                    spec["save_npz"],
                    loss=np.asarray(val),
                    grad_gas=np.asarray(g_gas),
                    grad_mass=gm_np,
                    grad_number=gn_np,
                )

        warmup = spec.get("warmup", 4)
        for _ in range(max(0, warmup - 1)):
            jax.block_until_ready(compiled(*call_args))

        iters = spec.get("iters", 15)
        step_ms = []
        for _ in range(iters):
            ts = time.perf_counter()
            res_iter = compiled(*call_args)
            jax.block_until_ready(res_iter)
            step_ms.append((time.perf_counter() - ts) * 1000.0)

        runtime_mem = {}
        try:
            ms = dev.memory_stats()
            if ms:
                runtime_mem = {
                    "peak_hbm_mb": round(ms.get("peak_bytes_in_use", 0) / (1024 * 1024), 2),
                    "current_hbm_mb": round(ms.get("bytes_in_use", 0) / (1024 * 1024), 2),
                }
        except Exception:
            pass

        arr = np.asarray(step_ms, dtype=np.float64)
        med_ms = float(np.median(arr))
        q25, q75 = np.percentile(arr, [25, 75])
        return {
            "status": "ok",
            "device": str(dev),
            "device_kind": getattr(dev, "device_kind", str(dev)),
            "platform": dev.platform,
            "nlev": nlev,
            "ncol": ncol,
            "ncells": ncells,
            "dtype": dtype_name,
            "mode": mode,
            "nucleation_scheme": nsnucl,
            "variant": variant,
            "chunk_size": chunk_size,
            "compile_seconds": round(compile_only_s, 4),
            "first_exec_seconds": round(first_exec_s, 4),
            "median_ms": round(med_ms, 4),
            "iqr_ms": round(float(q75 - q25), 4),
            "min_ms": round(float(arr.min()), 4),
            "mean_ms": round(float(arr.mean()), 4),
            "throughput_mcells_per_s": round((ncells / 1.0e6) / (med_ms / 1000.0), 4),
            "xla_memory": mem_info,
            "runtime_memory": runtime_mem,
        }
    except Exception as exc:
        msg = str(exc)
        is_oom = "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg or "out of memory" in msg
        return {
            "status": "oom" if is_oom else "error",
            "nlev": nlev,
            "ncol": ncol,
            "ncells": ncells,
            "dtype": dtype_name,
            "mode": mode,
            "nucleation_scheme": nsnucl,
            "variant": variant,
            "chunk_size": chunk_size,
            "error": msg[:600],
        }


def invoke_worker_subprocess(spec: dict, timeout_s: int = 600) -> dict:
    """Spawn a clean Python process for `spec` and return its JSON payload."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-spec",
        json.dumps(spec),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("WORKER_JSON:"):
            return json.loads(line[len("WORKER_JSON:"):])
    return {
        "status": "crash",
        "returncode": proc.returncode,
        "stderr": proc.stderr[-800:],
        "stdout": proc.stdout[-400:],
        **{k: spec.get(k) for k in ("dtype", "mode", "nucleation_scheme", "variant", "chunk_size")},
    }


def compare_npz(ref_path: Path, cand_path: Path) -> dict:
    """Compute quantile and scale-normalized error metrics between two `.npz` dumps."""
    out = {}
    with np.load(ref_path) as ref, np.load(cand_path) as cand:
        for key in ref.files:
            r = np.asarray(ref[key], dtype=np.float64)
            c = np.asarray(cand[key], dtype=np.float64)
            diff = np.abs(c - r)
            scale = max(float(np.abs(r).max()), 1.0e-30)
            rel = np.divide(diff, np.maximum(np.abs(r), 1.0e-20 * scale))
            med_r, p99_r, p999_r = np.quantile(rel, [0.5, 0.99, 0.999])
            out[key] = {
                "max_abs_error": float(diff.max()),
                "max_norm_error": float(diff.max() / scale),
                "median_rel_error": float(med_r),
                "p99_rel_error": float(p99_r),
                "p999_rel_error": float(p999_r),
                "max_rel_error": float(rel.max()),
            }
    return out


def source_hashes() -> dict[str, str]:
    files = (
        "benchmarks/benchmark_accelerator.py",
        "src/m7_jax/constants.py",
        "src/m7_jax/properties.py",
        "src/m7_jax/condensation.py",
        "src/m7_jax/nucleation.py",
        "src/m7_jax/coagulation.py",
        "src/m7_jax/redistribution.py",
        "src/m7_jax/interface.py",
        "src/m7_jax/model.py",
    )
    return {
        rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        for rel in files
        if (ROOT / rel).exists()
    }


def probe_accelerator() -> dict:
    """Query JAX backend/device metadata in a short-lived subprocess.

    On Cloud TPU, opening `libtpu` in the parent orchestrator process holds an
    exclusive lock on `/dev/vfio/*`, blocking child worker subprocesses. Probing
    in a transient child process releases the device lock immediately.
    """
    code = (
        "import json, jax; ds = jax.devices(); "
        "print('PROBE:' + json.dumps({"
        "'jax_version': jax.__version__, "
        "'backend': ds[0].platform, "
        "'devices': [str(d) for d in ds], "
        "'device_kinds': [getattr(d, 'device_kind', str(d)) for d in ds]"
        "}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("PROBE:"):
            return json.loads(line[len("PROBE:"):])
    raise RuntimeError(f"Accelerator probe failed: {proc.stderr}")


def run_suite(args) -> dict:
    opt_src = str((ROOT / "src").resolve())
    base_src = str(args.baseline_src.resolve()) if args.baseline_src else None
    accel = probe_accelerator()
    backend = accel["backend"]

    dtypes = args.dtypes.split(",") if args.dtypes else (
        ["float32", "float64"] if backend != "tpu" else ["float32"]
    )
    resolutions = args.resolutions.split(",")
    chunk_sizes = [None if x == "none" else int(x) for x in args.chunk_sizes.split(",")]

    parity_records = []
    if base_src is not None and not args.skip_parity:
        with tempfile.TemporaryDirectory(prefix="m7-accel-parity-") as tmp:
            tmp_dir = Path(tmp)
            for dt_name in dtypes:
                for mode in ("forward", "grad"):
                    ref_npz = tmp_dir / f"ref_{dt_name}_{mode}.npz"
                    ref_res = invoke_worker_subprocess({
                        "src_dir": base_src,
                        "nlev": 16,
                        "ncol": 256,
                        "dtype": dt_name,
                        "mode": mode,
                        "nucleation_scheme": 1,
                        "variant": "aos_step_all",
                        "chunk_size": None,
                        "warmup": 1,
                        "iters": 1,
                        "save_npz": str(ref_npz),
                    })
                    if ref_res.get("status") != "ok":
                        parity_records.append({"dtype": dt_name, "mode": mode, "error": ref_res})
                        continue
                    for variant, cs in (
                        ("aos_step_all", None),
                        ("soa_full", None),
                        ("soa_jcm", None),
                        ("soa_jcm", 1000),
                    ):
                        cand_npz = tmp_dir / f"cand_{dt_name}_{mode}_{variant}_{cs}.npz"
                        cand_res = invoke_worker_subprocess({
                            "src_dir": opt_src,
                            "nlev": 16,
                            "ncol": 256,
                            "dtype": dt_name,
                            "mode": mode,
                            "nucleation_scheme": 1,
                            "variant": variant,
                            "chunk_size": cs,
                            "warmup": 1,
                            "iters": 1,
                            "save_npz": str(cand_npz),
                        })
                        if cand_res.get("status") == "ok":
                            parity_records.append({
                                "dtype": dt_name,
                                "mode": mode,
                                "variant": variant,
                                "chunk_size": cs,
                                "errors_vs_baseline": compare_npz(ref_npz, cand_npz),
                            })

    timing_records = []
    for res_name in resolutions:
        nlev, ncol = RESOLUTIONS[res_name]
        for dt_name in dtypes:
            for nsnucl in (1, 2):
                modes = ("forward", "grad") if nsnucl == 1 else ("forward",)
                for mode in modes:
                    # 1. Baseline step_all (if provided)
                    base_med = None
                    if base_src is not None:
                        b_rec = invoke_worker_subprocess({
                            "src_dir": base_src,
                            "nlev": nlev,
                            "ncol": ncol,
                            "dtype": dt_name,
                            "mode": mode,
                            "nucleation_scheme": nsnucl,
                            "variant": "aos_step_all",
                            "chunk_size": None,
                            "warmup": args.warmup,
                            "iters": args.iters,
                        })
                        b_rec["codebase"] = "baseline"
                        b_rec["resolution"] = res_name
                        if b_rec.get("status") == "ok":
                            base_med = b_rec["median_ms"]
                            b_rec["speedup_vs_baseline"] = 1.0
                        timing_records.append(b_rec)
                        print(f"[{res_name} {dt_name} nsnucl={nsnucl} {mode}] baseline aos: {b_rec.get('median_ms', b_rec.get('status'))} ms", flush=True)

                    # 2. Optimized backward-compatible AoS step_all (unchunked)
                    o_aos = invoke_worker_subprocess({
                        "src_dir": opt_src,
                        "nlev": nlev,
                        "ncol": ncol,
                        "dtype": dt_name,
                        "mode": mode,
                        "nucleation_scheme": nsnucl,
                        "variant": "aos_step_all",
                        "chunk_size": None,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    })
                    o_aos["codebase"] = "optimized"
                    o_aos["resolution"] = res_name
                    if o_aos.get("status") == "ok" and base_med is not None:
                        o_aos["speedup_vs_baseline"] = round(base_med / o_aos["median_ms"], 4)
                    timing_records.append(o_aos)
                    print(f"[{res_name} {dt_name} nsnucl={nsnucl} {mode}] opt aos: {o_aos.get('median_ms', o_aos.get('status'))} ms (speedup={o_aos.get('speedup_vs_baseline')})", flush=True)

                    # 3. Optimized SoA full (compute_valid=True, compute_water=True) unchunked
                    o_soa_full = invoke_worker_subprocess({
                        "src_dir": opt_src,
                        "nlev": nlev,
                        "ncol": ncol,
                        "dtype": dt_name,
                        "mode": mode,
                        "nucleation_scheme": nsnucl,
                        "variant": "soa_full",
                        "chunk_size": None,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    })
                    o_soa_full["codebase"] = "optimized"
                    o_soa_full["resolution"] = res_name
                    if o_soa_full.get("status") == "ok" and base_med is not None:
                        o_soa_full["speedup_vs_baseline"] = round(base_med / o_soa_full["median_ms"], 4)
                    timing_records.append(o_soa_full)
                    print(f"[{res_name} {dt_name} nsnucl={nsnucl} {mode}] opt soa_full: {o_soa_full.get('median_ms', o_soa_full.get('status'))} ms (speedup={o_soa_full.get('speedup_vs_baseline')})", flush=True)

                    # 4. Optimized SoA JCM mode across chunk_sizes
                    for cs in chunk_sizes:
                        o_soa = invoke_worker_subprocess({
                            "src_dir": opt_src,
                            "nlev": nlev,
                            "ncol": ncol,
                            "dtype": dt_name,
                            "mode": mode,
                            "nucleation_scheme": nsnucl,
                            "variant": "soa_jcm",
                            "chunk_size": cs,
                            "warmup": args.warmup,
                            "iters": args.iters,
                        })
                        o_soa["codebase"] = "optimized"
                        o_soa["resolution"] = res_name
                        if o_soa.get("status") == "ok" and base_med is not None:
                            o_soa["speedup_vs_baseline"] = round(base_med / o_soa["median_ms"], 4)
                        timing_records.append(o_soa)
                        print(f"[{res_name} {dt_name} nsnucl={nsnucl} {mode}] opt soa_jcm chunk={cs}: {o_soa.get('median_ms', o_soa.get('status'))} ms (speedup={o_soa.get('speedup_vs_baseline')})", flush=True)

    return {
        "scope": "Full-step M7-JAX AoS vs SoA accelerator benchmark (forward + reverse-mode AD)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": accel["jax_version"],
        "numpy_version": np.__version__,
        "backend": backend,
        "devices": accel["devices"],
        "device_kinds": accel["device_kinds"],
        "source_sha256": source_hashes(),
        "kappa_lut_sha256": hashlib.sha256(KAPPA_LUT_PATH.read_bytes()).hexdigest(),
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "parity_checks": parity_records,
        "results": timing_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-src", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resolutions", type=str, default="T63L47")
    parser.add_argument("--dtypes", type=str, default=None)
    parser.add_argument("--chunk-sizes", type=str, default="none,16384,32768")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    if args.worker_spec is not None:
        res = run_worker(json.loads(args.worker_spec))
        print("WORKER_JSON:" + json.dumps(res), flush=True)
        return

    payload = run_suite(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
