"""Reproducible native/JAX M7 process timings after explicit parity gates.

Run from the repository root with its virtual environment:
python -m benchmarks.benchmark_m7 --output docs/evidence/m7-process-benchmark.json
"""
import argparse
from functools import partial
import hashlib
import json
import platform
from pathlib import Path
import tempfile
import time

import jax
import numpy as np

from m7_jax.coagulation import coaset, dnum
from m7_jax.condensation import condensation_sink, produce_and_condense
from m7_jax.nucleation import nucleate
from tools.m7_oracle import build_oracle, run_oracle

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ('coaset', 'condensation_sink', 'produce_and_condense', 'nucleate', 'dnum')


def cases(count):
    """Nonempty soluble sulfate cases, with common inputs for both backends."""
    number = np.broadcast_to([1000., 200., 30., 1., 0., 0., 0.], (count, 7)).copy()
    scale = np.linspace(1., 1.5, count)
    number *= scale[:, None]
    radius = np.broadcast_to([2e-7, 2e-6, 1e-5, 1e-4, 0., 0., 0.], (count, 7)).copy()
    sulfate = (number[:, :4] * (4*np.pi/3) * radius[:, :4]**3
               * np.exp(4.5*np.log([1.59, 1.59, 1.59, 2.])**2)
               * 1.841 * 6.02214179e23 / 96.0631)
    delta = np.zeros((count, 4))
    delta[:, 0] = 20.
    return dict(number=number, radius=radius, density=np.full_like(number, 1.841),
                sulfate=sulfate, sulfate_delta=delta, nucleated_number=np.full(count, 5.),
                temperature=np.linspace(240., 250., count), pressure=np.full(count, 101325.),
                gas=np.linspace(1e8, 2e8, count), production=np.full(count, 1e5),
                relative_humidity=np.full(count, .5), cloud_fraction=np.zeros(count),
                timestep=np.ones(count), scheme=np.ones(count, dtype=int))


def kernel_arguments(operation, data):
    if operation == 'coaset':
        return coaset, [data[key] for key in ('number', 'temperature', 'pressure', 'radius', 'density')]
    if operation == 'condensation_sink':
        return condensation_sink, [data[key] for key in ('number', 'radius', 'temperature', 'pressure')]
    if operation == 'produce_and_condense':
        return produce_and_condense, [data[key] for key in ('gas', 'production', 'sulfate', 'total_sink', 'modal_sink', 'cloud_fraction', 'timestep')]
    if operation == 'nucleate':
        function = partial(nucleate, scheme=1, initial_number=data['nucleated_number'],
                           initial_sulfate=data['sulfate_delta'][:, 0])
        return function, [data[key] for key in ('gas', 'temperature', 'relative_humidity', 'total_sink', 'cloud_fraction', 'timestep')]
    if operation == 'dnum':
        return dnum, [data[key] for key in ('sulfate', 'number', 'temperature', 'pressure', 'radius', 'density', 'nucleated_number', 'sulfate_delta', 'timestep')]
    raise ValueError(operation)


def output_fields(operation, actual):
    if operation == 'coaset':
        return {'coefficients': np.asarray(actual)}
    if operation == 'condensation_sink':
        return {'total_sink': np.asarray(actual.total), 'modal_sink': np.asarray(actual.modal)}
    if operation == 'produce_and_condense':
        return {key: np.asarray(getattr(actual, key)) for key in actual._fields}
    if operation == 'nucleate':
        return {'gas': np.asarray(actual.gas), 'nucleated_number': np.asarray(actual.number),
                'sulfate_delta': np.pad(np.asarray(actual.sulfate)[:, None], ((0, 0), (0, 3)))}
    return dict(zip(('sulfate', 'number', 'sulfate_delta'), map(np.asarray, actual)))


def array_digest(data):
    digest = hashlib.sha256()
    for key, value in sorted(data.items()):
        value = np.asarray(value)
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def benchmark(*, repetitions=2000, trials=3, batch_size=64):
    if any(type(value) is not int or value < 1 for value in (repetitions, trials, batch_size)):
        raise ValueError('repetitions, trials and batch_size must be positive integers')
    jax.config.update('jax_enable_x64', True)
    records = []
    diagnostics = []
    with tempfile.TemporaryDirectory(prefix='m7-process-benchmark-') as temporary:
        checked, checked_provenance = build_oracle(Path(temporary)/'checked', checked=True)
        optimized, optimized_provenance = build_oracle(Path(temporary)/'optimized', checked=False)
        for count in dict.fromkeys((1, batch_size)):
            data = cases(count)
            sink, _ = run_oracle(checked, data, operation='condensation_sink')
            data.update(total_sink=sink['total_sink'], modal_sink=sink['modal_sink'])
            if count == 1:
                small_reference, _ = run_oracle(checked, data, operation='dnum')
                function, args = kernel_arguments('dnum', data)
                small_actual = output_fields('dnum', jax.jit(function)(*jax.device_put(args)))
                fields = {}
                for key, value in small_actual.items():
                    error = np.abs(value-small_reference[key])
                    relative = np.divide(error, np.abs(small_reference[key]),
                        out=np.zeros_like(error), where=small_reference[key] != 0.)
                    fields[key] = {'max_absolute_error': float(error.max()),
                        'max_relative_error_nonzero': float(relative.max()),
                        'passed': bool(np.all(error <= 3e-12*np.abs(small_reference[key])))}
                diagnostics.append({'operation': 'dnum', 'cases': count, 'timestep_s': 1.,
                    'case_sha256': array_digest(data), 'relative_tolerance': 3e-12,
                    'absolute_tolerance': 0., 'passed': all(row['passed'] for row in fields.values()),
                    'fields': fields,
                    'interpretation': 'Small sulfate transfer deltas amplify cancellation in total-number minus new-number. This one-second case is not accepted at the strict relative gate. Final number and sulfate can agree more closely; no tolerance was relaxed.'})
            for operation in OPERATIONS:
                operation_data = {**data, 'timestep': np.full(count, 600.)} if operation == 'dnum' else data
                reference, _ = run_oracle(checked, operation_data, operation=operation)
                fast, _ = run_oracle(optimized, operation_data, operation=operation)
                for key in reference:
                    np.testing.assert_array_equal(fast[key], reference[key])
                function, args = kernel_arguments(operation, operation_data)
                args = jax.device_put(args)
                compiled = jax.jit(function)
                started = time.perf_counter()
                actual = compiled(*args)
                jax.block_until_ready(actual)
                cold_seconds = time.perf_counter() - started
                fields = output_fields(operation, actual)
                tolerance = {'nucleate': 1e-11, 'dnum': 3e-12}.get(operation, 3e-13)
                errors = {}
                for key, value in fields.items():
                    expected = reference[key]
                    np.testing.assert_allclose(value, expected, rtol=tolerance, atol=0.)
                    error = np.abs(value - expected)
                    relative = np.divide(error, np.abs(expected), out=np.zeros_like(error), where=expected != 0.)
                    errors[key] = {'max_absolute_error': float(error.max()),
                                   'max_relative_error_nonzero': float(relative.max()),
                                   'nonzero_at_reference_zero': int(np.count_nonzero(value[expected == 0.]))}
                native_times, jax_times = [], []
                for _ in range(trials):
                    _, timing = run_oracle(optimized, operation_data, operation=operation, repetitions=repetitions)
                    native_times.append(timing['native_cpu_seconds'] / repetitions)
                    started = time.perf_counter()
                    for _ in range(repetitions):
                        jax.block_until_ready(compiled(*args))
                    jax_times.append((time.perf_counter() - started) / repetitions)
                native_median, jax_median = float(np.median(native_times)), float(np.median(jax_times))
                records.append({'operation': operation, 'cases': count, 'case_sha256': array_digest(operation_data),
                    'timestep_s': float(operation_data['timestep'][0]),
                    'accepted': True, 'checked_optimized_exact_all_fields': True,
                    'jax_native_rtol': tolerance, 'jax_native_atol': 0., 'errors': errors,
                    'repetitions': repetitions, 'trials': trials,
                    'jax_compile_and_first_execution_seconds': cold_seconds,
                    'native_cpu_seconds_per_batch_trials': native_times,
                    'jax_wall_seconds_per_batch_trials': jax_times,
                    'native_cpu_seconds_per_batch_median': native_median,
                    'jax_wall_seconds_per_batch_median': jax_median,
                    'native_cpu_over_jax_wall_ratio': native_median / jax_median})
    files = ('benchmarks/benchmark_m7.py', 'tools/m7_oracle.py', 'src/m7_jax/constants.py',
             'src/m7_jax/coagulation.py', 'src/m7_jax/condensation.py', 'src/m7_jax/nucleation.py')
    return {'accepted': all(row['accepted'] for row in records), 'scope': 'Independent M7 sulfate process calls; no full-step or trajectory performance claim',
        'platform': platform.platform(), 'python_version': platform.python_version(),
        'jax_version': jax.__version__, 'numpy_version': np.__version__, 'devices': [str(x) for x in jax.devices()],
        'dtype': 'float64', 'benchmark_source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        'checked_provenance': checked_provenance, 'optimized_provenance': optimized_provenance,
        'timing_limits': ['Native uses process CPU_TIME; JAX uses synchronized wall time, so ratios are descriptive, not identical-clock throughput measurements.',
            'Native includes case state reset and a 107-field output copy; excludes startup, table loading, input/output serialization and compilation.',
            'JAX inputs already reside on device; every returned output is synchronized for each call. JIT compilation/first execution are recorded separately.',
            'The native dnum wrapper computes average properties to populate native per-particle mass; JAX computes the required average within its sulfate integrator.',
            'Native loops over independent one-cell calls; JAX evaluates the batch together. These include different dispatch/setup overheads.'],
        'untimed_small_step_diagnostics': diagnostics,
        'dnum_timestep_choice': '600 seconds follows the independently tested larger-transfer regime; the separate one-second strict-gate failure is retained above.',
        'results': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2000)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    result = benchmark(repetitions=args.repetitions, trials=args.trials, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'accepted': result['accepted'], 'output': str(args.output),
        'ratios': {f"{row['operation']}-{row['cases']}": row['native_cpu_over_jax_wall_ratio'] for row in result['results']}}, indent=2))


if __name__ == '__main__':
    main()
