"""M7 coagulation in native CGS units.

The 16 collision kernels retain all seven M7 modes. ``dnum``/``delcoa`` are the
soluble-sulfate-only branch of ``m7_dnum``/``m7_delcoa`` (insoluble modes must
have zero number). ``dnum_all`` is the full multi-species port, including both
sections of ``m7_delcoa`` (soluble modes 1-4 and the insoluble Aitken mode 5)
and the insoluble-to-soluble ageing of ``m7_concoag``/``m7_coat`` (coating of
modes 5/6/7 -- KI/AI/CI -- on ``so4_coating_threshold`` sulfate monolayers).
Dtype-generic: every function computes in whichever float dtype its inputs
carry (float64 under ``jax_enable_x64``, float32 without it -- #1017 W1 task
3); the ``EPS_DP``/``ZMIN_COAG`` thresholds are fixed native literals
(``EPSILON(1._dp)``-derived and the native unsuffixed ``1e-15``
respectively) and deliberately do NOT track the working dtype. Below float64
the native nearly-equal-totals subtractions of ``m7_delcoa``/``m7_concoag``
switch to algebraically identical cancellation-free forms
(:mod:`m7_jax.precision`); float64 keeps the native forms bit for bit.
Leading dimensions describe independent cells.
"""
import math

import jax
import jax.numpy as jnp

from . import _ad, precision
from .constants import (BK, CMIN_AERNL, CMINRAD, CMINRHO, COAG_MODE_PAIRS, PI,
    SIGMA, ZMIN_COAG, EPS_DP)

# TODO(lead): move to constants.py's registry once it grows a multi-species
# section. These are native mo_ham_m7ctl.F90 values not yet exposed there:
# so4_coating_threshold (line ~257), the H2SO4 molecule cross-section used by
# m7_coat (line 3868), and cmr2ras (mo_ham_m7ctl.F90:433,
# cmr2ras(jclass)=EXP(1.0*LOG(sigma(jclass))**2) -- note the exponent of 1,
# not 1.5 as for CMR2RAM in constants.py).
SO4_COATING_THRESHOLD = 1.0  # monolayers, mo_ham_m7ctl.F90:257
CSURF_MOLEC = 2.39e-15  # cm^2, cross-section of one H2SO4 molecule, m7_coat
CMR2RAS = tuple(math.exp(math.log(s) ** 2) for s in SIGMA)

# 0-based column indices into the 18-component ``mass``/``mass_delta`` state,
# in HAM aerocomp order (task spec): SO4 NS KS AS CS; BC KS AS CS KI;
# OC KS AS CS KI; SS AS CS; DU AS CS AI CI.
ISO4NS, ISO4KS, ISO4AS, ISO4CS = 0, 1, 2, 3
IBCKS, IBCAS, IBCCS, IBCKI = 4, 5, 6, 7
IOCKS, IOCAS, IOCCS, IOCKI = 8, 9, 10, 11
ISSAS, ISSCS = 12, 13
IDUAS, IDUCS, IDUAI, IDUCI = 14, 15, 16, 17

# 0-based mode indices (nclass=7): NS KS AS CS KI AI CI.
_NS, _KS, _AS, _CS, _KI, _AI, _CI = 0, 1, 2, 3, 4, 5, 6

# mo_ham_m7.F90's final aggregation (section 4, lines 4679-4702) adds
# pa4delt into paerml for every (species, mode) pair with a valid aerocomp
# index EXCEPT the nucleation-mode sulfate slot (already folded into the
# mode-1 overwrite via za4sum) and sea salt (no branch exists for id_ss: SS
# has no insoluble companion mode and never receives a delcoa/concoag delta).
# A static tuple: ``_dnum_all_soa`` adds the delta only where it is 1.
_SECTION4_MASK = (0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1.)


def _coaset_soa(number7, temperature, pressure, radius7, density7):
    """Tuple-of-arrays implementation of ``m7_coaset`` returning 16 collision kernels [cm³/s]."""
    dtype = number7[0].dtype
    temperature = jnp.asarray(temperature, dtype=dtype)
    pressure = jnp.asarray(pressure, dtype=dtype)
    free_path = 2.251e-3 * temperature / pressure
    viscosity = 1.827e-4 * (temperature / 291.15)**0.74
    thermal = (BK / (6.0 * PI)) * temperature / viscosity
    vel_num = (8.0 * BK / PI) * temperature
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=dtype))
    four_pi_over_3 = 4.0 * PI / 3.0

    act = []
    r = []
    vsq = []
    diff = []
    jump_sq = []
    for m in range(7):
        act_m = (number7[m] > CMIN_AERNL) & (radius7[m] > CMINRAD) & (density7[m] > CMINRHO)
        r_m = jnp.where(act_m, radius7[m], 1.0)
        rho_m = jnp.where(act_m, density7[m], 1.0)
        vol_m = four_pi_over_3 * r_m**3
        vsq_m = _ad.div(vel_num, vol_m * rho_m)
        kn_m = 0.5 * free_path / r_m
        diff_m = thermal * (1.0 + kn_m * 2.492 + kn_m * 0.84 * jnp.exp(-0.43 / kn_m)) / r_m
        dist_m = (8.0 / PI) * diff_m / jnp.sqrt(vsq_m)
        diam_m = 2.0 * r_m
        jump_m = (((diam_m + dist_m)**3 - jnp.sqrt((diam_m * diam_m + dist_m * dist_m)**3))
                  / (6.0 * r_m * dist_m) - diam_m) * sqrt2
        act.append(act_m)
        r.append(r_m)
        vsq.append(vsq_m)
        diff.append(diff_m)
        jump_sq.append(jump_m * jump_m)

    kernels = []
    for i, j in COAG_MODE_PAIRS:
        average_radius = EPS_DP + 0.5 * (r[i] + r[j])
        velocity = jnp.sqrt(vsq[i] + vsq[j])
        average_diffusion = 0.5 * (diff[i] + diff[j])
        jump_pair = _ad.sqrt(jump_sq[i] + jump_sq[j])
        continuum = (16.0 * PI) * average_diffusion * average_radius
        resistance = (4.0 * average_diffusion / (velocity * average_radius)
                      + average_radius / (average_radius + 0.5 * jump_pair))
        kernels.append(jnp.where(act[i] & act[j], continuum / resistance, 0.0))
    return tuple(kernels)


def coaset(number, temperature, pressure, radius, density):
    """Return the 16 native M7 collision coefficients [cm³/s].

    ``number``, ``radius`` and ``density`` have trailing dimension seven,
    in cm⁻³, cm and g/cm³ respectively. Temperature [K] and pressure [Pa]
    broadcast over the leading cell dimensions. Inactive modes yield zero.
    """
    # Anchor the working dtype on the three state arrays (not temperature/
    # pressure, which a caller may pass as bare Python scalars) so the whole
    # function computes in float32 when they are float32, float64 when they
    # are float64 -- #1017 W1 task 3.
    dtype = jnp.result_type(number, radius, density)
    arrays = [jnp.asarray(x, dtype=dtype) for x in (number, radius, density)]
    if any(not value.ndim or value.shape[-1] != 7 for value in arrays):
        raise ValueError("number, radius and density need seven trailing modes")
    number, radius, density = jnp.broadcast_arrays(*arrays)
    number7 = tuple(number[..., m] for m in range(7))
    radius7 = tuple(radius[..., m] for m in range(7))
    density7 = tuple(density[..., m] for m in range(7))
    kernels16 = _coaset_soa(number7, temperature, pressure, radius7, density7)
    return jnp.stack(kernels16, axis=-1)


def _mixed_particles(zaerni, zaerns, ztotn, a0dt, *, robust):
    """Mixed soluble+insoluble particles of m7_delcoa 2.2.3 (``zanytnm``).

    Native (mo_ham_m7.F90:4379-4389), with c = a*dt, T = ztotn, S = zaerns,
    I = zaerni::

        zanytnm = 2T/(2+cT) - [4I/((2+cT)(2+cS)) + 4S/((2+cT)(2+cI))]

    (the I-term only when I > 1), then ``MIN(zanytnm, zaerni)``. The
    bracket nearly equals ``2T/(2+cT)`` -- both are ~T while the net is
    ~c*S*I -- so in float32 the result is pure rounding noise and goes
    negative (4.4e5 NS particles on 52 KI give a negative count that drives
    KS sulfate and number negative). With ``robust`` the three terms are
    collected over one denominator, which cancels the totals exactly in real
    arithmetic::

        I > 1:   zanytnm = [cS/(2+cS)] * [2I/(2+cI)] * [(4+cT)/(2+cT)]
        I <= 1:  zanytnm = 2I/(2+cI)

    Every factor is a ratio of non-negative terms, so the result is
    non-negative and accurate to a few roundings, and no intermediate can
    overflow float32. ``robust`` is :func:`precision.cancellation_free` of
    the working dtype.
    """
    if robust:
        pair = 2.0 * zaerni / (2.0 + a0dt * zaerni)
        soluble = a0dt * zaerns / (2.0 + a0dt * zaerns)
        total = (4.0 + a0dt * ztotn) / (2.0 + a0dt * ztotn)
        zanytnm = jnp.where(zaerni > 1.0, soluble * pair * total, pair)
    else:
        denom_totn = 2.0 + a0dt * ztotn
        zanytni = jnp.where(zaerni > 1.0,
            4.0 * zaerni / (denom_totn * (2.0 + a0dt * (ztotn - zaerni))), 0.0)
        zanytnt = 2.0 * ztotn / denom_totn
        zanytns = 4.0 * zaerns / (denom_totn * (2.0 + a0dt * (ztotn - zaerns)))
        zanytnm = zanytnt - (zanytni + zanytns)
    return jnp.minimum(zanytnm, zaerni)


def _decay(total, initial, a, b, formed, timestep, *, robust=False):
    """Native analytical number solution and intra/inter-mode loss split.

    The four branches are m7_delcoa's 1.2.1/2.2.1a (intra-modal only),
    1.2.2/2.2.1b (intra + inter), 2.2.2a (nucleation + both) and 2.2.2b
    (nucleation + intra-modal only), mo_ham_m7.F90:4101-4366.

    ``robust`` (:func:`precision.cancellation_free` of the working dtype)
    selects forms that are identical in real arithmetic but avoid the native
    cancellations, each of which in float32 loses every digit:

    - ``1 - exp(-x)`` is ``-expm1(-x)``. For a weakly scavenged mode
      (``b*dt`` ~ 1e-7) the native difference is quantised to float32's
      spacing near 1 and wrong by up to 100%.
    - Without a source, the particles lost (``zatot = zansum - zaernt``, the
      difference of two nearly equal totals when little coagulates) are
      computed directly: ``n0 * n0*a*dt / (1 + n0*a*dt)`` intra-only, and
      ``n0 * G*(n0*a + b) / (n0*a*G + b)`` with ``G = 1 - exp(-b*dt)``,
      which is ``n0 - b*n0*E/(n0*a*G + b)`` with ``n0`` cancelled
      analytically. Both are non-negative by construction.
    - The nucleating solution ``(R*(1-f3)/(1+f3) - b) / (2a)`` subtracts
      ``b`` from a quantity that tends to ``b`` when ``2a*n`` is small next
      to ``b`` (a strongly scavenged fresh nucleation mode): it can come out
      negative in float32. Multiplying through by ``(R + f2)`` and using
      ``R - b = 4ac/(R + b)`` gives the same solution as a ratio of sums of
      non-negative terms,
      ``[n0*(2*E*R + G*(R - b)) + 2*c*G] / [(1 + E)*R + G*f2]``
      with ``E = exp(-R*dt)``, ``G = 1 - E``.
    - The intra-only nucleating denominator ``zbot = zf1 - ze1`` (both near 1
      when ``n0 >> R1`` and ``2*a*R1*dt`` is small) is
      ``(zf1 - 1) + (1 - ze1) = 2*R1/(n0 - R1) - expm1(-2*a*R1*dt)``. Its sign,
      which selects the native ``zbot < zmin_coag`` fallback, is that of the
      native expression wherever the native value is not rounding noise.

    With a source, ``zatot = zansum - zaernt`` stays a difference in both
    forms: ``zansum`` (``n0`` plus the particles formed this step) is itself
    one rounding of a sum, so the difference is accurate to that rounding;
    it is never amplified (it is only split by ratios and multiplied by the
    mode's own mean mass); and it is non-negative through the native
    ``MIN(zaernt, zansum)``. A difference-free form would need the
    fresh-particle loss ``x*(1 + exp(-x)) - 2*(1 - exp(-x))``, which is not
    expressible without cancellation in closed form.
    """
    inter = b >= ZMIN_COAG
    safe_a = jnp.where(a != 0, a, 1.0)
    no_source, no_source_loss = _no_source_solution(total, a, b, timestep, robust=robust)
    source = formed / timestep
    radical = _ad.sqrt(b * b + 4.0 * a * source)
    f2 = 2.0 * a * initial + b
    r1 = _ad.sqrt(_ad.div(source, safe_a))
    exp_intra = jnp.exp(-r1 * 2.0 * a * timestep)
    offset = jnp.where(initial != r1, initial - r1, 1.0)
    ratio = _ad.div(initial + r1, offset)
    if robust:
        radical_exp = jnp.exp(-radical * timestep)
        radical_growth = -jnp.expm1(-radical * timestep)        # 1 - exp(-R dt)
        radical_excess = _ad.div(4.0 * a * source, jnp.where(inter, radical + b, 1.0))   # R - b
        top = initial * (2.0 * radical_exp * radical + radical_growth * radical_excess) \
            + 2.0 * source * radical_growth
        with_inter = _ad.div(top, jnp.where(inter, (1.0 + radical_exp) * radical + radical_growth * f2, 1.0))
        bottom = _ad.div(2.0 * r1, offset) - jnp.expm1(-r1 * 2.0 * a * timestep)       # ratio - exp_intra
    else:
        denominator = radical + f2
        f3 = _ad.div(jnp.exp(-radical * timestep) * (radical - f2), jnp.where(denominator != 0, denominator, 1.0))
        # 1 + f3 > 0 wherever this branch is selected (b >= zmin_coag makes
        # R > 0 and |f3| < 1); it is exactly 0 in the masked cells with no
        # source and b = 0, whose infinity would make the masked branch's
        # zero cotangent NaN. The guard leaves every selected value unchanged.
        f3_plus = 1.0 + f3
        with_inter = _ad.div((radical * _ad.div(1.0 - f3, jnp.where(f3_plus != 0, f3_plus, 1.0)) - b) / 2.0, safe_a)
        # Preserve the native solution, including its signed denominator test.
        bottom = ratio - exp_intra
    with_intra = jnp.where(bottom < ZMIN_COAG, total,
                          _ad.div(r1 * (ratio + exp_intra), jnp.where(bottom != 0, bottom, 1.0)))
    # The native formula is singular at initial == sqrt(source/a). That case
    # is outside the validated native domain; no alternate science is added.
    with_intra = jnp.where((source >= 1.0) & (initial == r1), jnp.nan, with_intra)
    new = jnp.where(source >= 1.0, jnp.minimum(jnp.where(inter, with_inter, with_intra), total), no_source)
    total_loss = total - new
    if robust:
        total_loss = jnp.where(source >= 1.0, total_loss, no_source_loss)
    return (new, *_split_loss(total, a, b, inter, total_loss))


def _no_source_solution(total, a, b, timestep, *, robust):
    """Native decay without a particle source (m7_delcoa 1.2.1/1.2.2 and
    2.2.1a/b): ``(new, total_loss)``, ``total_loss`` in the form ``robust``
    selects (see :func:`_decay`). ``b=None`` statically drops the inter-modal
    branch for a mode with no inter-modal sink; that is what ``b = 0`` gives
    anyway, since it never passes ``b >= zmin_coag``.
    """
    intra_bottom = 1.0 + total * a * timestep
    intra_new = _ad.div(total, intra_bottom)
    intra_loss = _ad.div(total * (total * a * timestep), intra_bottom) if robust else None
    if b is None:
        return intra_new, intra_loss if robust else total - intra_new
    inter = b >= ZMIN_COAG
    safe_b = jnp.where(inter, b, 1.0)
    exponential = jnp.exp(-b * timestep)
    if robust:
        growth = -jnp.expm1(-safe_b * timestep)                  # 1 - exp(-b dt)
        inter_bottom = total * a * growth + safe_b
        new = jnp.where(inter, jnp.minimum(_ad.div(safe_b * total * exponential, inter_bottom), total), intra_new)
        total_loss = jnp.where(inter, _ad.div(total * growth * (total * a + safe_b), inter_bottom), intra_loss)
        return new, total_loss
    new = jnp.where(
        inter,
        jnp.minimum(_ad.div(safe_b * total * exponential, total * a * (1.0 - exponential) + safe_b), total),
        intra_new,
    )
    return new, total - new


def _split_loss(total, a, b, inter, total_loss):
    """Split a mode's particle loss into ``(intra, inter)``-modal parts by the native rate ratio."""
    intra_rate = total * total * a
    inter_rate = total * b
    rate = intra_rate + inter_rate
    safe_rate = jnp.where(rate != 0, rate, 1.0)
    intra_loss = jnp.where(inter, _ad.div(total_loss * intra_rate, safe_rate), total_loss)
    inter_loss = jnp.where(inter, _ad.div(total_loss * inter_rate, safe_rate), 0.0)
    return intra_loss, inter_loss


def _decay_no_source(total, a, b, timestep, *, robust=False):
    """:func:`_decay` with ``formed == 0`` statically (modes KS, KI): no nucleating branches."""
    new, total_loss = _no_source_solution(total, a, b, timestep, robust=robust)
    return (new, *_split_loss(total, a, b, b >= ZMIN_COAG, total_loss))


def _decay_intra_only(total, a, timestep, *, robust=False):
    """:func:`_decay` with ``formed == 0`` and ``b == 0`` statically (mode AS): ``(new, intra_loss)``."""
    return _no_source_solution(total, a, None, timestep, robust=robust)


def delcoa(sulfate, number, sulfate_delta, nucleated_number, a, b, fraction1, fraction2, timestep, *,
           cancellation_free=None):
    """Integrate pure-sulfate M7 coagulation, returning mass, number, delta.

    Sulfate and delta have four trailing soluble modes [molecules/cm³];
    number has seven [cm⁻³], with the three insoluble entries zero. ``a`` and
    ``b`` have four entries [cm³/s and s⁻¹]; fractions have six entries in the
    native destination-minus-two order. Only delta[..., 0] is a supported
    incoming source (nucleated sulfate); nucleated_number is cm⁻³ per step.
    ``timestep`` is positive seconds. Ordered transfers match the Fortran.
    ``cancellation_free`` (static ``None``/``True``/``False``) selects the
    decay's expression forms, see :func:`precision.cancellation_free`; this
    API always computes in float64, so ``None`` means the native forms.
    """
    sulfate, sulfate_delta = jnp.broadcast_arrays(
        jnp.asarray(sulfate, dtype=jnp.float64), jnp.asarray(sulfate_delta, dtype=jnp.float64))
    number = jnp.asarray(number, dtype=jnp.float64)
    nucleated_number = jnp.asarray(nucleated_number, dtype=jnp.float64)
    timestep = jnp.asarray(timestep, dtype=jnp.float64)
    a, b, fraction1, fraction2 = (jnp.asarray(value, dtype=jnp.float64)
                                for value in (a, b, fraction1, fraction2))
    for value, size in ((sulfate, 4), (number, 7), (a, 4), (b, 4), (fraction1, 6), (fraction2, 6)):
        if not value.ndim or value.shape[-1] != size:
            raise ValueError(f"Expected trailing mode dimension {size}")
    robust = precision.cancellation_free(jnp.float64, cancellation_free)
    for mode in range(4):
        formed = nucleated_number if mode == 0 else jnp.zeros_like(nucleated_number)
        total = number[..., mode] + formed
        mass = sulfate[..., mode] + (sulfate_delta[..., 0] if mode == 0 else 0.0)
        sufficient = total > CMIN_AERNL
        active = sufficient & ((a[..., mode] >= ZMIN_COAG) | (b[..., mode] >= ZMIN_COAG))
        average = mass / jnp.where(sufficient, total, 1.0)
        new, intra, inter = _decay(total, number[..., mode], a[..., mode], b[..., mode], formed, timestep,
                                   robust=robust)
        updated_mass = jnp.where(active, (new + intra) * average, mass if mode == 0 else sulfate[..., mode])
        updated_number = jnp.where(active, new, total if mode == 0 else number[..., mode])
        sulfate = sulfate.at[..., mode].set(jnp.where(sufficient, updated_mass, sulfate[..., mode]))
        number = number.at[..., mode].set(jnp.where(sufficient, updated_number, number[..., mode]))
        if mode == 0:
            for target in range(1, 4):
                transfer = fraction1[..., target - 1] * inter * average
                transfer += fraction1[..., target + 2] * inter * average
                sulfate_delta = sulfate_delta.at[..., target].set(jnp.where(active, transfer, sulfate_delta[..., target]))
        elif mode == 1:
            for target in (2, 3):
                transfer = fraction2[..., target - 1] * inter * average
                transfer2 = fraction2[..., target + 2] * inter * average
                value = sulfate_delta[..., target] + transfer
                value = value + transfer2
                sulfate_delta = sulfate_delta.at[..., target].set(jnp.where(active, value, sulfate_delta[..., target]))
            value = sulfate_delta[..., 1] + fraction2[..., 3] * inter * average
            sulfate_delta = sulfate_delta.at[..., 1].set(jnp.where(active, value, sulfate_delta[..., 1]))
    sulfate = sulfate.at[..., 1:4].add(sulfate_delta[..., 1:4])
    return sulfate, number, sulfate_delta


def dnum(sulfate, number, temperature, pressure, radius, density,
         nucleated_number, sulfate_delta, timestep, *, coagulation=True, cancellation_free=None):
    """Pure-sulfate ``m7_dnum`` with native units and trailing mode axes.

    Insoluble numbers must be zero; coating and mixed-species aging are outside
    this API. ``coagulation`` and ``cancellation_free`` (see :func:`delcoa`)
    are static switches when compiling with JAX.
    See :func:`coaset` and :func:`delcoa` for the input and return contracts.
    """
    number = jnp.asarray(number, dtype=jnp.float64)
    if not number.ndim or number.shape[-1] != 7:
        raise ValueError("number needs seven trailing modes")
    shape = number.shape[:-1]
    a, b = jnp.zeros(shape + (4,), dtype=jnp.float64), jnp.zeros(shape + (4,), dtype=jnp.float64)
    fraction1 = jnp.zeros(shape + (6,), dtype=jnp.float64)
    fraction2 = jnp.zeros(shape + (6,), dtype=jnp.float64)
    if coagulation:
        kernels = coaset(number, temperature, pressure, radius, density)
        a = a.at[..., :3].set(kernels[..., jnp.asarray([0, 7, 13])] / 2.0)
        fraction1 = kernels[..., 1:7] * number[..., 1:7]
        fraction2 = fraction2.at[..., 1:3].set(kernels[..., 8:10] * number[..., 2:4])
        fraction2 = fraction2.at[..., 4:6].set(kernels[..., 11:13] * number[..., 5:7])
        # Ordered sums follow the native statement order.
        b1 = fraction1[..., 0]
        for index in range(1, 6):
            b1 = b1 + fraction1[..., index]
        b2 = fraction2[..., 1]
        for index in range(2, 6):
            b2 = b2 + fraction2[..., index]
        b = b.at[..., 0].set(b1).at[..., 1].set(b2)
        fraction1 = fraction1 / jnp.where(b1 > 0, b1, 1.0)[..., None]
        fraction2 = fraction2 / jnp.where(b2 > 0, b2, 1.0)[..., None]
    return delcoa(sulfate, number, sulfate_delta, nucleated_number, a, b, fraction1, fraction2, timestep,
                  cancellation_free=cancellation_free)


def _coat_soa(r4, r5, r6):
    """Tuple-of-arrays implementation of ``m7_coat`` for insoluble modes ``(KI, AI, CI)``."""
    out = []
    for r_m, factor in ((r4, CMR2RAS[4]), (r5, CMR2RAS[5]), (r6, CMR2RAS[6])):
        # Keep the barrier: removing it (with those in properties) changed the
        # outputs' rounding and slowed the step (see properties._soluble_mode_sums).
        fac = jax.lax.optimization_barrier(jnp.asarray(factor, dtype=r_m.dtype))
        ras = r_m * fac
        surface = 4.0 * PI * ras * ras
        out.append(surface / CSURF_MOLEC * SO4_COATING_THRESHOLD)
    return tuple(out)


def _coat(radius):
    """Native ``m7_coat`` (mo_ham_m7.F90:3830-3891): sulfate molecules needed
    to coat the mean particle of the insoluble modes (KI, AI, CI; trailing
    indices 4, 5, 6 of ``radius``) with ``so4_coating_threshold`` monolayers.
    Returns an array with trailing dimension 3, ordered (KI, AI, CI).
    """
    radius = jnp.asarray(radius)
    ras = radius[..., 4:7] * jnp.asarray(CMR2RAS[4:7], dtype=radius.dtype)
    surface = 4.0 * PI * ras * ras
    return surface / CSURF_MOLEC * SO4_COATING_THRESHOLD


def _dnum_all_soa(mass18, number7, temperature, pressure, radius7, density7,
                  insoluble_sulfate3, nucleated_number, mass_delta18, mean_mass18,
                  timestep, *, coagulation=True, cancellation_free=None):
    """Tuple-of-arrays implementation of full multi-species ``m7_dnum``.

    Returns ``(mass18, number7, delta18)``; under ``jit`` an unused ``delta18``
    is dead code and is eliminated, so callers simply discard it.

    Statement order is deliberate: a reordering that is exact in the forward
    pass (shared helpers, loops over modes) made XLA fuse the backward pass
    worse (float32 ``value_and_grad`` +17% on an A100). Re-benchmark the
    gradient before restructuring.
    """
    dtype = mass18[0].dtype
    nucleated_number = jnp.asarray(nucleated_number, dtype=dtype)
    timestep = jnp.asarray(timestep, dtype=dtype)
    eps = EPS_DP
    robust = precision.cancellation_free(dtype, cancellation_free)

    mass = list(mass18)
    number = list(number7)
    delta = list(mass_delta18)
    crtcst0, crtcst1, crtcst2 = _coat_soa(radius7[4], radius7[5], radius7[6])

    if coagulation:
        # --- Coagulation coefficients (m7_dnum, mo_ham_m7.F90:2628-2769): ----
        # za(1,2,3,5) = zcom(1,8,14,16)/2 (unimodal); za(4,6,7) = 0 (CS/AI/CI
        # never self-coagulate here -- no kernel pair is defined for them). zb
        # sums the inter-modal fractions for modes 1, 2 and 5; zb(3,4,6,7) = 0.
        # Only the structurally non-zero fractions are formed: pbfract2's
        # KS-to-KI target is m7_dnum's "pointless calculation"
        # (mo_ham_m7.F90:2678-2685) and pbfract5's KI-to-KI target is zero, so
        # their native no-op adds are omitted.
        k = _coaset_soa(number7, temperature, pressure, radius7, density7)
        a0 = k[0] / 2.0
        a1 = k[7] / 2.0
        a2 = k[13] / 2.0
        a4 = k[15] / 2.0

        f1 = [k[idx] * number[idx] for idx in range(1, 7)]
        f2_1 = k[8] * number[2]
        f2_2 = k[9] * number[3]
        f2_4 = k[11] * number[5]
        f2_5 = k[12] * number[6]
        f5_1 = k[10] * number[1]
        f5_2 = k[14] * number[2]

        b1 = f1[0]
        for idx in range(1, 6):
            b1 = b1 + f1[idx]
        b2 = f2_1 + f2_2 + f2_4 + f2_5
        b5 = f5_1 + f5_2

        safe_b1 = jnp.where(b1 > 0, b1, 1.0)
        safe_b2 = jnp.where(b2 > 0, b2, 1.0)
        safe_b5 = jnp.where(b5 > 0, b5, 1.0)
        pbf1 = [_ad.div(x, safe_b1) for x in f1]
        pbf2_1 = _ad.div(f2_1, safe_b2)
        pbf2_2 = _ad.div(f2_2, safe_b2)
        pbf2_4 = _ad.div(f2_4, safe_b2)
        pbf2_5 = _ad.div(f2_5, safe_b2)
        pbf5_1 = _ad.div(f5_1, safe_b5)
        pbf5_2 = _ad.div(f5_2, safe_b5)

        # --- Section 1: insoluble Aitken mode KI (m7_delcoa:4072-4190). -----
        # Dust-only insoluble modes (AI, CI) do nothing here; only KI
        # coagulates (intra-modal, plus inter-modal loss to KS/AS via pbfract5).
        number4 = number[_KI]
        sufficient4 = number4 > CMIN_AERNL
        gate4 = sufficient4 & ((a4 >= ZMIN_COAG) | (b5 >= ZMIN_COAG))
        new4, intra4, inter4 = _decay_no_source(number4, a4, b5, timestep, robust=robust)
        zanli4 = jnp.where(gate4, inter4, 0.0)
        zansq4 = jnp.where(gate4, intra4, 0.0)
        zaernt4 = jnp.where(gate4, new4, number4)
        ki_factor = zaernt4 + zansq4  # pbfract5[0] is structurally zero
        mass[IBCKI] = jnp.where(gate4, ki_factor * mean_mass18[IBCKI] * 1e12, mass[IBCKI])
        mass[IOCKI] = jnp.where(gate4, ki_factor * mean_mass18[IOCKI] * 1e12, mass[IOCKI])
        number[_KI] = jnp.where(gate4, zaernt4, number[_KI])
        delta[IBCKS] = jnp.where(gate4, pbf5_1 * zanli4 * mean_mass18[IBCKI] * 1e12, delta[IBCKS])
        delta[IOCKS] = jnp.where(gate4, pbf5_1 * zanli4 * mean_mass18[IOCKI] * 1e12, delta[IOCKS])
        delta[IBCAS] = jnp.where(gate4, pbf5_2 * zanli4 * mean_mass18[IBCKI] * 1e12, delta[IBCAS])
        delta[IOCAS] = jnp.where(gate4, pbf5_2 * zanli4 * mean_mass18[IOCKI] * 1e12, delta[IOCAS])

        # --- Section 2: soluble modes NS, KS, AS (CS has a=b=formed=0). -----
        zansum0 = number[0] + nucleated_number
        za4sum0 = mass[0] + delta[ISO4NS]
        suff0 = zansum0 > CMIN_AERNL
        gate0 = suff0 & ((a0 >= ZMIN_COAG) | (b1 >= ZMIN_COAG))
        avg0 = _ad.div(za4sum0, jnp.where(suff0, zansum0, 1.0))
        new0, intra0, inter0 = _decay(
            zansum0, number[0], a0, b1, nucleated_number, timestep, robust=robust
        )
        za4av0 = jnp.where(suff0, avg0, 0.0)
        zaernt0 = jnp.where(gate0, new0, zansum0)
        zansq0 = jnp.where(gate0, intra0, 0.0)
        zanli0 = jnp.where(gate0, inter0, 0.0)

        zansum1 = number[1]
        za4sum1 = mass[1]
        suff1 = zansum1 > CMIN_AERNL
        gate1 = suff1 & ((a1 >= ZMIN_COAG) | (b2 >= ZMIN_COAG))
        avg1 = _ad.div(za4sum1, jnp.where(suff1, zansum1, 1.0))
        new1, intra1, inter1 = _decay_no_source(zansum1, a1, b2, timestep, robust=robust)
        za4av1 = jnp.where(suff1, avg1, 0.0)
        zaernt1 = jnp.where(gate1, new1, zansum1)
        zansq1 = jnp.where(gate1, intra1, 0.0)
        zanli1 = jnp.where(gate1, inter1, 0.0)

        zansum2 = number[2]
        za4sum2 = mass[2]
        suff2 = zansum2 > CMIN_AERNL
        gate2 = suff2 & (a2 >= ZMIN_COAG)
        avg2 = _ad.div(za4sum2, jnp.where(suff2, zansum2, 1.0))
        new2, intra2 = _decay_intra_only(zansum2, a2, timestep, robust=robust)
        za4av2 = jnp.where(suff2, avg2, 0.0)
        zaernt2 = jnp.where(gate2, new2, zansum2)
        zansq2 = jnp.where(gate2, intra2, 0.0)

        # --- 2.2.3: insoluble/soluble mixed coagulation correction, mode NS
        # only (m7_delcoa:4368-4452). Boosts the fraction of NS's inter-modal
        # loss (pbfract1) that lands on KI/AI/CI's soluble homes, at the expense
        # of NS's own intra-modal loss. The Fortran also reassigns
        # zanytni = zaerni - zanytnm and zanytns = zaernt (m7_delcoa:4390-4391)
        # but neither value is read before its next assignment: dead code. ---
        zaerni = number[_KI] + number[_AI] + number[_CI]
        zaerns = zansum0 + number[_KS]
        ztotn = zaerns + zaerni
        mix_gate = (zanli0 > 0.0) & (zaerns > zaerni) & (zaerni > eps)
        zanytnm = _mixed_particles(zaerni, zaerns, ztotn, a0 * timestep, robust=robust)
        safe_zaerni = jnp.where(zaerni > eps, zaerni, 1.0)
        has_sulfate = za4av0 > 0.0
        safe_za4av0 = jnp.where(za4av0 != 0.0, za4av0, 1.0)
        amloss5 = _ad.div(number[_KI], safe_zaerni) * zanytnm * crtcst0
        amloss6 = _ad.div(number[_AI], safe_zaerni) * zanytnm * crtcst1
        amloss7 = _ad.div(number[_CI], safe_zaerni) * zanytnm * crtcst2
        anloss5 = jnp.where(has_sulfate, _ad.div(amloss5, safe_za4av0), 0.0)
        anloss6 = jnp.where(has_sulfate, _ad.div(amloss6, safe_za4av0), 0.0)
        anloss7 = jnp.where(has_sulfate, _ad.div(amloss7, safe_za4av0), 0.0)
        ztloss = jnp.minimum(anloss5 + anloss6 + anloss7, zansq0 * 0.95)
        denom_mix = zanli0 + ztloss
        safe_denom_mix = jnp.where(denom_mix != 0.0, denom_mix, 1.0)
        zbfofac = jnp.where(mix_gate, _ad.div(zanli0, safe_denom_mix), 1.0)
        zbfnfac = jnp.where(mix_gate, _ad.div(ztloss, safe_denom_mix), 0.0)
        p1_s = [
            pbf1[0] * zbfofac,
            pbf1[1] * zbfofac,
            pbf1[2] * zbfofac,
            pbf1[3] * zbfofac + _ad.div(zbfnfac * number[_KI], safe_zaerni),
            pbf1[4] * zbfofac + _ad.div(zbfnfac * number[_AI], safe_zaerni),
            pbf1[5] * zbfofac + _ad.div(zbfnfac * number[_CI], safe_zaerni),
        ]
        # Ordered sum matching mo_ham_m7.F90:4425-4436 DO kmod=1,nclass.
        zbftot = p1_s[0]
        for idx in range(1, 6):
            zbftot = zbftot + p1_s[idx]
        has_zbftot = jnp.abs(zbftot) > eps
        safe_zbftot = jnp.where(has_zbftot, zbftot, 1.0)
        pbf1 = [
            jnp.where(mix_gate, jnp.where(has_zbftot, _ad.div(x, safe_zbftot), 0.0), orig)
            for x, orig in zip(p1_s, pbf1)
        ]
        zanli0 = jnp.where(mix_gate, zanli0 + ztloss, zanli0)
        zansq0 = jnp.where(mix_gate, zansq0 - ztloss, zansq0)

        # --- 2.3: write surviving sulfate/number/BC/OC back for NS, KS, AS. -
        so4_0 = jnp.where(gate0, (zaernt0 + zansq0) * za4av0, za4sum0)
        num_0 = jnp.where(gate0, zaernt0, zansum0)
        mass[0] = jnp.where(suff0, so4_0, mass[0])
        number[0] = jnp.where(suff0, num_0, number[0])

        surv1 = zaernt1 + zansq1
        mass[1] = jnp.where(gate1, surv1 * za4av1, mass[1])
        number[1] = jnp.where(gate1, zaernt1, number[1])
        surv1_ug = surv1 * 1e12
        mass[IBCKS] = jnp.where(gate1, surv1_ug * mean_mass18[IBCKS], mass[IBCKS])
        mass[IOCKS] = jnp.where(gate1, surv1_ug * mean_mass18[IOCKS], mass[IOCKS])

        surv2 = zaernt2 + zansq2
        mass[2] = jnp.where(gate2, surv2 * za4av2, mass[2])
        number[2] = jnp.where(gate2, zaernt2, number[2])
        surv2_ug = surv2 * 1e12
        mass[IBCAS] = jnp.where(gate2, surv2_ug * mean_mass18[IBCAS], mass[IBCAS])
        mass[IOCAS] = jnp.where(gate2, surv2_ug * mean_mass18[IOCAS], mass[IOCAS])

        # --- 2.4: distribute inter-modal transfers (m7_delcoa:4494-4647).
        # Mode NS's targets overwrite (nothing else has written them yet); mode
        # KS's targets add on top, matching the native "=" then "+=" order. --
        ns_ks = pbf1[0] * zanli0 * za4av0 + pbf1[3] * zanli0 * za4av0
        ns_as = pbf1[1] * zanli0 * za4av0 + pbf1[4] * zanli0 * za4av0
        ns_cs = pbf1[2] * zanli0 * za4av0 + pbf1[5] * zanli0 * za4av0
        delta[ISO4KS] = jnp.where(gate0, ns_ks, delta[ISO4KS])
        delta[ISO4AS] = jnp.where(gate0, ns_as, delta[ISO4AS])
        delta[ISO4CS] = jnp.where(gate0, ns_cs, delta[ISO4CS])

        zbcmass = mean_mass18[IBCKS] * zanli1 * 1e12
        zocmass = mean_mass18[IOCKS] * zanli1 * 1e12
        delta[ISO4AS] = delta[ISO4AS] + jnp.where(gate1, pbf2_1 * zanli1 * za4av1 + pbf2_4 * zanli1 * za4av1, 0.0)
        delta[IBCAS] = delta[IBCAS] + jnp.where(gate1, pbf2_1 * zbcmass + pbf2_4 * zbcmass, 0.0)
        delta[IOCAS] = delta[IOCAS] + jnp.where(gate1, pbf2_1 * zocmass + pbf2_4 * zocmass, 0.0)
        delta[ISO4CS] = delta[ISO4CS] + jnp.where(gate1, pbf2_2 * zanli1 * za4av1 + pbf2_5 * zanli1 * za4av1, 0.0)
        delta[IBCCS] = delta[IBCCS] + jnp.where(gate1, pbf2_2 * zbcmass + pbf2_5 * zbcmass, 0.0)
        delta[IOCCS] = delta[IOCCS] + jnp.where(gate1, pbf2_2 * zocmass + pbf2_5 * zocmass, 0.0)

        zso4_5 = zanli0 * pbf1[3] * za4av0 + insoluble_sulfate3[0]
        zso4_6 = zanli0 * pbf1[4] * za4av0 + zanli1 * pbf2_4 * za4av1 + insoluble_sulfate3[1]
        zso4_7 = zanli0 * pbf1[5] * za4av0 + zanli1 * pbf2_5 * za4av1 + insoluble_sulfate3[2]
        basis_5 = jnp.where(gate4, ki_factor, number[_KI])
    else:
        zansum0 = number[0] + nucleated_number
        za4sum0 = mass[0] + delta[ISO4NS]
        suff0 = zansum0 > CMIN_AERNL
        mass[0] = jnp.where(suff0, za4sum0, mass[0])
        number[0] = jnp.where(suff0, zansum0, number[0])
        zso4_5, zso4_6, zso4_7 = insoluble_sulfate3
        basis_5 = number[_KI]

    # --- m7_concoag: insoluble-to-soluble ageing (mo_ham_m7.F90:3579-3827).
    # Runs unconditionally (not gated on ``coagulation``), exactly as the
    # native m7_delcoa calls it outside any lscoag check. --------------------
    safe_c0 = jnp.where(crtcst0 > eps, crtcst0, 1.0)
    safe_c1 = jnp.where(crtcst1 > eps, crtcst1, 1.0)
    safe_c2 = jnp.where(crtcst2 > eps, crtcst2, 1.0)
    gate5 = (number[_KI] >= CMIN_AERNL) & (crtcst0 > eps)
    gate6 = (number[_AI] >= CMIN_AERNL) & (crtcst1 > eps)
    gate7 = (number[_CI] >= CMIN_AERNL) & (crtcst2 > eps)
    zcrit_5 = jnp.where(gate5, jnp.minimum(number[_KI], _ad.div(zso4_5, safe_c0)), 0.0)
    zcrit_6 = jnp.where(gate6, jnp.minimum(number[_AI], _ad.div(zso4_6, safe_c1)), 0.0)
    zcrit_7 = jnp.where(gate7, jnp.minimum(number[_CI], _ad.div(zso4_7, safe_c2)), 0.0)

    # Particle-equivalents (in units of the mean particle ``pttn``) each
    # insoluble mode holds before ageing: section 1's ``ki_factor`` where KI
    # coagulated (its mass was just set to ki_factor*pttn*1e12), otherwise the
    # mode's number, since pttn = mass/number (m7_averageproperties).
    basis_6, basis_7 = number[_AI], number[_CI]
    # ``+ (-zcrit)`` is deliberate: ``- zcrit`` is bit-identical but fused
    # ~6% slower in float32 value_and_grad on an A100.
    number[_KS] = number[_KS] + zcrit_5
    number[_KI] = number[_KI] + (-zcrit_5)
    number[_AS] = number[_AS] + zcrit_6
    number[_AI] = number[_AI] + (-zcrit_6)
    number[_CS] = number[_CS] + zcrit_7
    number[_CI] = number[_CI] + (-zcrit_7)

    def _age_soa(index, basis, zcrit):
        # Native: paerml = paerml - zcrit*1e12*pttn (mo_ham_m7.F90:3777-3822).
        # When every particle ages (zcrit = number) that subtracts two equal
        # totals and leaves rounding noise of either sign. The robust form is
        # the same quantity factored, (basis - zcrit)*pttn*1e12: zcrit =
        # MIN(number, ...) <= basis, so it is exactly zero for full ageing and
        # non-negative otherwise. Where pttn = 0 (mass below cmin_aerml) the
        # native update subtracts nothing, so neither does it.
        if not robust:
            return mass[index] + (-zcrit * 1e12 * mean_mass18[index])
        factored = (basis - zcrit) * mean_mass18[index] * 1e12
        return jnp.where(mean_mass18[index] > 0.0, factored, mass[index])

    delta[ISO4KS] = delta[ISO4KS] + insoluble_sulfate3[0]
    mass[IBCKI] = _age_soa(IBCKI, basis_5, zcrit_5)
    mass[IOCKI] = _age_soa(IOCKI, basis_5, zcrit_5)
    delta[IBCKS] = delta[IBCKS] + zcrit_5 * 1e12 * mean_mass18[IBCKI]
    delta[IOCKS] = delta[IOCKS] + zcrit_5 * 1e12 * mean_mass18[IOCKI]

    delta[ISO4AS] = delta[ISO4AS] + insoluble_sulfate3[1]
    mass[IDUAI] = _age_soa(IDUAI, basis_6, zcrit_6)
    delta[IDUAS] = delta[IDUAS] + zcrit_6 * 1e12 * mean_mass18[IDUAI]

    delta[ISO4CS] = delta[ISO4CS] + insoluble_sulfate3[2]
    mass[IDUCI] = _age_soa(IDUCI, basis_7, zcrit_7)
    delta[IDUCS] = delta[IDUCS] + zcrit_7 * 1e12 * mean_mass18[IDUCI]

    # --- Section 4: final aggregation (mo_ham_m7.F90:4674-4702). ------------
    for jn in range(18):
        if _SECTION4_MASK[jn]:
            mass[jn] = mass[jn] + delta[jn]
    return tuple(mass), tuple(number), tuple(delta)


def dnum_all(mass, number, temperature, pressure, radius, density,
             insoluble_sulfate, nucleated_number, mass_delta, mean_mass,
             timestep, *, coagulation=True, cancellation_free=None):
    """Full multi-species ``m7_dnum``: ``m7_delcoa`` (both sections) then
    ``m7_concoag``/``m7_coat`` ageing. Inputs/outputs mirror the oracle's
    ``dnum`` operation; ``mean_mass`` is ``pttn`` (molecules/particle for SO4,
    grams/particle for BC/OC/DU), an input here as in the native call.

    ``mass``/``mass_delta`` have 18 trailing components in HAM aerocomp order
    (see module docstring); ``number``/``radius``/``density`` have 7 trailing
    modes (NS KS AS CS KI AI CI); ``insoluble_sulfate`` has 3, ordered
    (KI, AI, CI) -- the native ``zso4_5``/``zso4_6``/``zso4_7``.
    ``coagulation`` is a static switch; ``m7_concoag`` always runs (the
    native ``m7_delcoa`` calls it unconditionally), so insoluble ageing driven
    purely by condensed ``insoluble_sulfate`` still proceeds with coagulation
    disabled, exactly as in the Fortran.

    ``cancellation_free`` (static ``None``/``True``/``False``) chooses between
    the native and the cancellation-free expression forms; ``None`` follows
    the working dtype (native in float64). See :func:`precision.cancellation_free`.
    """
    mass = jnp.asarray(mass)
    dtype = mass.dtype
    mass_delta = jnp.asarray(mass_delta, dtype=dtype)
    number = jnp.asarray(number, dtype=dtype)
    radius = jnp.asarray(radius, dtype=dtype)
    density = jnp.asarray(density, dtype=dtype)
    mean_mass = jnp.asarray(mean_mass, dtype=dtype)
    insoluble_sulfate = jnp.asarray(insoluble_sulfate, dtype=dtype)
    for value, size, name in ((mass, 18, "mass"), (mass_delta, 18, "mass_delta"),
                               (number, 7, "number"), (radius, 7, "radius"),
                               (density, 7, "density"), (mean_mass, 18, "mean_mass"),
                               (insoluble_sulfate, 3, "insoluble_sulfate")):
        if not value.ndim or value.shape[-1] != size:
            raise ValueError(f"{name} needs trailing dimension {size}")

    mass18 = tuple(mass[..., jn] for jn in range(18))
    number7 = tuple(number[..., m] for m in range(7))
    radius7 = tuple(radius[..., m] for m in range(7))
    density7 = tuple(density[..., m] for m in range(7))
    insoluble3 = tuple(insoluble_sulfate[..., m] for m in range(3))
    delta18 = tuple(mass_delta[..., jn] for jn in range(18))
    mean18 = tuple(mean_mass[..., jn] for jn in range(18))
    out_mass18, out_number7, out_delta18 = _dnum_all_soa(
        mass18, number7, temperature, pressure, radius7, density7,
        insoluble3, nucleated_number, delta18, mean18, timestep,
        coagulation=coagulation, cancellation_free=cancellation_free,
    )
    return (
        jnp.stack(out_mass18, axis=-1),
        jnp.stack(out_number7, axis=-1),
        jnp.stack(out_delta18, axis=-1),
    )
