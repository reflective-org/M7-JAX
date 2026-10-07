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

import jax.numpy as jnp

from . import precision
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
# A plain tuple, not a module-level jnp array: a jnp array built once at
# import time freezes its dtype to whichever `jax_enable_x64` setting was
# live at FIRST import of this module, which does not track later per-call
# dtype changes (#1017 W1 task 3) -- built fresh, at the input's dtype, in
# `dnum_all` below instead.
_SECTION4_MASK = (0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1.)


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
    temperature = jnp.asarray(temperature, dtype=dtype)[..., None]
    pressure = jnp.asarray(pressure, dtype=dtype)[..., None]
    active = (number > CMIN_AERNL) & (radius > CMINRAD) & (density > CMINRHO)
    # Benign arguments keep unselected empty-mode arithmetic finite.
    r = jnp.where(active, radius, 1.0)
    rho = jnp.where(active, density, 1.0)
    free_path = 2.251e-3 * temperature / pressure
    viscosity = 1.827e-4 * (temperature / 291.15)**0.74
    thermal = (BK / (6.0 * PI)) * temperature / viscosity
    volume = (4.0 * PI / 3.0) * r**3
    velocity_squared = (8.0 * BK / PI) * temperature / (volume * rho)
    knudsen = 0.5 * free_path / r
    diffusion = thermal * (1.0 + knudsen * 2.492 + knudsen * 0.84 * jnp.exp(-0.43 / knudsen)) / r
    distance = (8.0 / PI) * diffusion / jnp.sqrt(velocity_squared)
    diameter = 2.0 * r
    jump = (((diameter + distance)**3 - jnp.sqrt((diameter * diameter + distance * distance)**3))
            / (6.0 * r * distance) - diameter) * jnp.sqrt(2.0)
    first, second = jnp.asarray(COAG_MODE_PAIRS).T
    average_radius = EPS_DP + 0.5 * (r[..., first] + r[..., second])
    velocity = jnp.sqrt(velocity_squared[..., first] + velocity_squared[..., second])
    average_diffusion = 0.5 * (diffusion[..., first] + diffusion[..., second])
    jump_pair = jnp.sqrt(jump[..., first]**2 + jump[..., second]**2)
    continuum = (16.0 * PI) * average_diffusion * average_radius
    resistance = (4.0 * average_diffusion / (velocity * average_radius)
                  + average_radius / (average_radius + 0.5 * jump_pair))
    return jnp.where(active[..., first] & active[..., second], continuum / resistance, 0.0)


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
    safe_b = jnp.where(inter, b, 1.0)
    safe_a = jnp.where(a != 0, a, 1.0)
    exponential = jnp.exp(-b * timestep)
    source = formed / timestep
    radical = jnp.sqrt(b * b + 4.0 * a * source)
    f2 = 2.0 * a * initial + b
    r1 = jnp.sqrt(source / safe_a)
    exp_intra = jnp.exp(-r1 * 2.0 * a * timestep)
    offset = jnp.where(initial != r1, initial - r1, 1.0)
    ratio = (initial + r1) / offset
    if robust:
        growth = -jnp.expm1(-safe_b * timestep)                  # 1 - exp(-b dt)
        inter_bottom = total * a * growth + safe_b
        intra_bottom = 1.0 + total * a * timestep
        no_source = jnp.where(inter, jnp.minimum(safe_b * total * exponential / inter_bottom, total),
                              total / intra_bottom)
        no_source_loss = jnp.where(inter, total * growth * (total * a + safe_b) / inter_bottom,
                                   total * (total * a * timestep) / intra_bottom)
        radical_exp = jnp.exp(-radical * timestep)
        radical_growth = -jnp.expm1(-radical * timestep)        # 1 - exp(-R dt)
        radical_excess = 4.0 * a * source / jnp.where(inter, radical + b, 1.0)   # R - b
        top = initial * (2.0 * radical_exp * radical + radical_growth * radical_excess) \
            + 2.0 * source * radical_growth
        with_inter = top / jnp.where(inter, (1.0 + radical_exp) * radical + radical_growth * f2, 1.0)
        bottom = 2.0 * r1 / offset - jnp.expm1(-r1 * 2.0 * a * timestep)       # ratio - exp_intra
    else:
        no_source = jnp.where(
            inter,
            jnp.minimum(safe_b * total * exponential / (total * a * (1.0 - exponential) + safe_b), total),
            total / (1.0 + total * a * timestep),
        )
        denominator = radical + f2
        f3 = jnp.exp(-radical * timestep) * (radical - f2) / jnp.where(denominator != 0, denominator, 1.0)
        with_inter = (radical * ((1.0 - f3) / (1.0 + f3)) - b) / 2.0 / safe_a
        # Preserve the native solution, including its signed denominator test.
        bottom = ratio - exp_intra
    with_intra = jnp.where(bottom < ZMIN_COAG, total,
                          r1 * (ratio + exp_intra) / jnp.where(bottom != 0, bottom, 1.0))
    # The native formula is singular at initial == sqrt(source/a). That case
    # is outside the validated native domain; no alternate science is added.
    with_intra = jnp.where((source >= 1.0) & (initial == r1), jnp.nan, with_intra)
    new = jnp.where(source >= 1.0, jnp.minimum(jnp.where(inter, with_inter, with_intra), total), no_source)
    total_loss = total - new
    if robust:
        total_loss = jnp.where(source >= 1.0, total_loss, no_source_loss)
    intra_rate = total * total * a
    inter_rate = total * b
    rate = intra_rate + inter_rate
    safe_rate = jnp.where(rate != 0, rate, 1.0)
    inter_loss = jnp.where(inter, total_loss * inter_rate / safe_rate, 0.0)
    intra_loss = jnp.where(inter, total_loss * intra_rate / safe_rate, total_loss)
    return new, intra_loss, inter_loss


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


def _coat(radius):
    """Native ``m7_coat`` (mo_ham_m7.F90:3830-3891): sulfate molecules needed
    to coat the mean particle of the insoluble modes (KI, AI, CI; trailing
    indices 4, 5, 6 of ``radius``) with ``so4_coating_threshold`` monolayers.
    Returns an array with trailing dimension 3, ordered (KI, AI, CI).
    """
    ras = radius[..., 4:7] * jnp.asarray(CMR2RAS[4:7])
    surface = 4.0 * PI * ras * ras
    return surface / CSURF_MOLEC * SO4_COATING_THRESHOLD


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
    # Anchor the working dtype on `mass` (the step_all chain's primary state
    # array) so this whole function computes in float32 when the caller
    # passes float32, float64 when it passes float64 -- #1017 W1 task 3.
    mass = jnp.asarray(mass)
    dtype = mass.dtype
    mass_delta = jnp.asarray(mass_delta, dtype=dtype)
    number = jnp.asarray(number, dtype=dtype)
    mean_mass = jnp.asarray(mean_mass, dtype=dtype)
    insoluble_sulfate = jnp.asarray(insoluble_sulfate, dtype=dtype)
    nucleated_number = jnp.asarray(nucleated_number, dtype=dtype)
    timestep = jnp.asarray(timestep, dtype=dtype)
    for value, size, name in ((mass, 18, "mass"), (mass_delta, 18, "mass_delta"),
                               (number, 7, "number"), (mean_mass, 18, "mean_mass"),
                               (insoluble_sulfate, 3, "insoluble_sulfate")):
        if not value.ndim or value.shape[-1] != size:
            raise ValueError(f"{name} needs trailing dimension {size}")
    shape = number.shape[:-1]
    eps = EPS_DP
    robust = precision.cancellation_free(dtype, cancellation_free)

    # --- Coagulation coefficients (m7_dnum, mo_ham_m7.F90:2628-2769): ------
    # za(1,2,3,5)=zcom(1,8,14,16)/2 (unimodal); za(4,6,7)=0 (CS/AI/CI never
    # self-coagulate here -- no kernel pair is defined for them). zb sums the
    # inter-modal fractions for modes 1, 2 and 5; zb(3,4,6,7)=0.
    a = jnp.zeros(shape + (7,), dtype=dtype)
    b = jnp.zeros(shape + (7,), dtype=dtype)
    pbfract1 = jnp.zeros(shape + (6,), dtype=dtype)
    pbfract2 = jnp.zeros(shape + (6,), dtype=dtype)
    pbfract5 = jnp.zeros(shape + (3,), dtype=dtype)
    if coagulation:
        kernels = coaset(number, temperature, pressure, radius, density)
        a = a.at[..., 0].set(kernels[..., 0] / 2.0)
        a = a.at[..., 1].set(kernels[..., 7] / 2.0)
        a = a.at[..., 2].set(kernels[..., 13] / 2.0)
        a = a.at[..., 4].set(kernels[..., 15] / 2.0)
        fraction1 = kernels[..., 1:7] * number[..., 1:7]
        fraction2 = jnp.zeros(shape + (6,), dtype=dtype)
        fraction2 = fraction2.at[..., 1:3].set(kernels[..., 8:10] * number[..., 2:4])
        fraction2 = fraction2.at[..., 4:6].set(kernels[..., 11:13] * number[..., 5:7])
        fraction5 = jnp.zeros(shape + (3,), dtype=dtype)
        fraction5 = fraction5.at[..., 1].set(kernels[..., 10] * number[..., 1])
        fraction5 = fraction5.at[..., 2].set(kernels[..., 14] * number[..., 2])
        b1 = fraction1[..., 0]
        for index in range(1, 6):
            b1 = b1 + fraction1[..., index]
        b2 = fraction2[..., 1]
        for index in range(2, 6):
            b2 = b2 + fraction2[..., index]
        b5 = fraction5[..., 1] + fraction5[..., 2]
        b = b.at[..., 0].set(b1).at[..., 1].set(b2).at[..., 4].set(b5)
        pbfract1 = fraction1 / jnp.where(b1 > 0, b1, 1.0)[..., None]
        pbfract2 = fraction2 / jnp.where(b2 > 0, b2, 1.0)[..., None]
        pbfract5 = fraction5 / jnp.where(b5 > 0, b5, 1.0)[..., None]

    # --- Section 1: insoluble Aitken mode KI (m7_delcoa:4072-4190). --------
    # Dust-only insoluble modes (AI, CI) do nothing here; only KI coagulates
    # (intra-modal, plus inter-modal loss to KS/AS via pbfract5).
    number4 = number[..., _KI]
    sufficient4 = number4 > CMIN_AERNL
    active4 = sufficient4 & ((a[..., _KI] >= ZMIN_COAG) | (b[..., _KI] >= ZMIN_COAG))
    new4, intra4, inter4 = _decay(number4, number4, a[..., _KI], b[..., _KI],
                                   jnp.zeros_like(number4), timestep, robust=robust)
    gate4 = sufficient4 & active4
    zanli4 = jnp.where(gate4, inter4, 0.0)
    zansq4 = jnp.where(gate4, intra4, 0.0)
    zaernt4 = jnp.where(gate4, new4, number4)
    ki_factor = zaernt4 + zansq4 + pbfract5[..., 0] * zanli4
    mass = mass.at[..., IBCKI].set(jnp.where(gate4, ki_factor * mean_mass[..., IBCKI] * 1e12, mass[..., IBCKI]))
    mass = mass.at[..., IOCKI].set(jnp.where(gate4, ki_factor * mean_mass[..., IOCKI] * 1e12, mass[..., IOCKI]))
    number = number.at[..., _KI].set(jnp.where(gate4, zaernt4 + pbfract5[..., 0] * zanli4, number[..., _KI]))
    mass_delta = mass_delta.at[..., IBCKS].set(jnp.where(
        gate4, pbfract5[..., 1] * zanli4 * mean_mass[..., IBCKI] * 1e12, mass_delta[..., IBCKS]))
    mass_delta = mass_delta.at[..., IOCKS].set(jnp.where(
        gate4, pbfract5[..., 1] * zanli4 * mean_mass[..., IOCKI] * 1e12, mass_delta[..., IOCKS]))
    mass_delta = mass_delta.at[..., IBCAS].set(jnp.where(
        gate4, pbfract5[..., 2] * zanli4 * mean_mass[..., IBCKI] * 1e12, mass_delta[..., IBCAS]))
    mass_delta = mass_delta.at[..., IOCAS].set(jnp.where(
        gate4, pbfract5[..., 2] * zanli4 * mean_mass[..., IOCKI] * 1e12, mass_delta[..., IOCAS]))

    # --- Section 2: soluble modes NS, KS, AS, CS (m7_delcoa:4194-4648). ----
    # Raw per-mode decay, before the insoluble/soluble mixed-coagulation
    # correction (2.2.3) that only ever touches mode NS.
    za4av, zaernt, zansq, zanli, gate, za4sum_list, zansum_list = [], [], [], [], [], [], []
    for mode in range(4):
        formed = nucleated_number if mode == 0 else jnp.zeros_like(nucleated_number)
        zansum = number[..., mode] + formed
        za4sum = mass[..., mode] + (mass_delta[..., ISO4NS] if mode == 0 else 0.0)
        sufficient = zansum > CMIN_AERNL
        active = sufficient & ((a[..., mode] >= ZMIN_COAG) | (b[..., mode] >= ZMIN_COAG))
        average = za4sum / jnp.where(sufficient, zansum, 1.0)
        new, intra, inter = _decay(zansum, number[..., mode], a[..., mode], b[..., mode], formed, timestep,
                                   robust=robust)
        za4av.append(jnp.where(sufficient, average, 0.0))
        zaernt.append(jnp.where(active, new, zansum))
        zansq.append(jnp.where(active, intra, 0.0))
        zanli.append(jnp.where(active, inter, 0.0))
        gate.append(sufficient & active)
        za4sum_list.append(za4sum)
        zansum_list.append(zansum)

    # --- 2.2.3: insoluble/soluble mixed coagulation correction, mode NS ----
    # only (m7_delcoa:4368-4452). Boosts the fraction of NS's inter-modal
    # loss (pbfract1) that lands on KI/AI/CI's soluble homes, at the expense
    # of NS's own intra-modal loss, proportional to how much of the
    # coexisting insoluble population gets "used up" by the mixed-mode
    # analytical solution below.
    zaerni = number[..., _KI] + number[..., _AI] + number[..., _CI]
    zaerns = zansum_list[0] + number[..., _KS]
    ztotn = zaerns + zaerni
    mix_gate = (zanli[0] > 0.0) & (zaerns > zaerni) & (zaerni > eps)
    zanytnm = _mixed_particles(zaerni, zaerns, ztotn, a[..., 0] * timestep, robust=robust)
    # The Fortran also reassigns zanytni=zaerni-zanytnm and zanytns=zaernt
    # (m7_delcoa:4390-4391) but neither reassigned value is read again
    # before the next use of those names; dropped here as dead code with no
    # numerical effect, not a bug.
    safe_zaerni = jnp.where(zaerni > eps, zaerni, 1.0)
    crtcst = _coat(radius)
    has_sulfate = za4av[0] > 0.0
    safe_za4av0 = jnp.where(za4av[0] != 0.0, za4av[0], 1.0)
    amloss5 = number[..., _KI] / safe_zaerni * zanytnm * crtcst[..., 0]
    amloss6 = number[..., _AI] / safe_zaerni * zanytnm * crtcst[..., 1]
    amloss7 = number[..., _CI] / safe_zaerni * zanytnm * crtcst[..., 2]
    anloss5 = jnp.where(has_sulfate, amloss5 / safe_za4av0, 0.0)
    anloss6 = jnp.where(has_sulfate, amloss6 / safe_za4av0, 0.0)
    anloss7 = jnp.where(has_sulfate, amloss7 / safe_za4av0, 0.0)
    ztloss = jnp.minimum(anloss5 + anloss6 + anloss7, zansq[0] * 0.95)
    denom_mix = zanli[0] + ztloss
    safe_denom_mix = jnp.where(denom_mix != 0.0, denom_mix, 1.0)
    zbfofac = jnp.where(mix_gate, zanli[0] / safe_denom_mix, 1.0)
    zbfnfac = jnp.where(mix_gate, ztloss / safe_denom_mix, 0.0)
    pbfract1_scaled = pbfract1 * zbfofac[..., None]
    pbfract1_scaled = pbfract1_scaled.at[..., 3].add(zbfnfac * number[..., _KI] / safe_zaerni)
    pbfract1_scaled = pbfract1_scaled.at[..., 4].add(zbfnfac * number[..., _AI] / safe_zaerni)
    pbfract1_scaled = pbfract1_scaled.at[..., 5].add(zbfnfac * number[..., _CI] / safe_zaerni)
    zbftot = jnp.sum(pbfract1_scaled, axis=-1)
    safe_zbftot = jnp.where(jnp.abs(zbftot) > eps, zbftot, 1.0)
    pbfract1_renorm = jnp.where(jnp.abs(zbftot)[..., None] > eps, pbfract1_scaled / safe_zbftot[..., None], 0.0)
    pbfract1 = jnp.where(mix_gate[..., None], pbfract1_renorm, pbfract1)
    zanli[0] = jnp.where(mix_gate, zanli[0] + ztloss, zanli[0])
    zansq[0] = jnp.where(mix_gate, zansq[0] - ztloss, zansq[0])

    # --- Write the surviving sulfate/number/BC/OC back for NS,KS,AS,CS. ---
    # (m7_delcoa:4460-4473, 2.3.1/2.3.2.)
    bc_by_mode = {1: IBCKS, 2: IBCAS, 3: IBCCS}
    oc_by_mode = {1: IOCKS, 2: IOCAS, 3: IOCCS}
    for mode in range(4):
        so4_value = jnp.where(gate[mode], (zaernt[mode] + zansq[mode]) * za4av[mode], za4sum_list[mode])
        number_value = jnp.where(gate[mode], zaernt[mode], zansum_list[mode])
        mass = mass.at[..., mode].set(jnp.where(jnp.asarray(zansum_list[mode]) > CMIN_AERNL, so4_value, mass[..., mode]))
        number = number.at[..., mode].set(jnp.where(jnp.asarray(zansum_list[mode]) > CMIN_AERNL, number_value, number[..., mode]))
        if mode in bc_by_mode:
            survivors = (zaernt[mode] + zansq[mode]) * 1e12
            mass = mass.at[..., bc_by_mode[mode]].set(
                jnp.where(gate[mode], survivors * mean_mass[..., bc_by_mode[mode]], mass[..., bc_by_mode[mode]]))
            mass = mass.at[..., oc_by_mode[mode]].set(
                jnp.where(gate[mode], survivors * mean_mass[..., oc_by_mode[mode]], mass[..., oc_by_mode[mode]]))

    # --- 2.4: distribute inter-modal transfers (m7_delcoa:4494-4647). -----
    # Mode NS's targets overwrite (nothing else has written them yet); mode
    # KS's targets add on top, matching the native "=" then "+=" sequence.
    za4av0, za4av1 = za4av[0], za4av[1]
    zanli0, zanli1 = zanli[0], zanli[1]
    ns_ks = pbfract1[..., 0] * zanli0 * za4av0 + pbfract1[..., 3] * zanli0 * za4av0
    ns_as = pbfract1[..., 1] * zanli0 * za4av0 + pbfract1[..., 4] * zanli0 * za4av0
    ns_cs = pbfract1[..., 2] * zanli0 * za4av0 + pbfract1[..., 5] * zanli0 * za4av0
    mass_delta = mass_delta.at[..., ISO4KS].set(jnp.where(gate[0], ns_ks, mass_delta[..., ISO4KS]))
    mass_delta = mass_delta.at[..., ISO4AS].set(jnp.where(gate[0], ns_as, mass_delta[..., ISO4AS]))
    mass_delta = mass_delta.at[..., ISO4CS].set(jnp.where(gate[0], ns_cs, mass_delta[..., ISO4CS]))

    zbcmass = mean_mass[..., IBCKS] * zanli1 * 1e12
    zocmass = mean_mass[..., IOCKS] * zanli1 * 1e12

    def _add_ks(index, value):
        nonlocal mass_delta
        mass_delta = mass_delta.at[..., index].add(jnp.where(gate[1], value, 0.0))

    _add_ks(ISO4AS, pbfract2[..., 1] * zanli1 * za4av1 + pbfract2[..., 4] * zanli1 * za4av1)
    _add_ks(IBCAS, pbfract2[..., 1] * zbcmass + pbfract2[..., 4] * zbcmass)
    _add_ks(IOCAS, pbfract2[..., 1] * zocmass + pbfract2[..., 4] * zocmass)
    _add_ks(ISO4CS, pbfract2[..., 2] * zanli1 * za4av1 + pbfract2[..., 5] * zanli1 * za4av1)
    _add_ks(IBCCS, pbfract2[..., 2] * zbcmass + pbfract2[..., 5] * zbcmass)
    _add_ks(IOCCS, pbfract2[..., 2] * zocmass + pbfract2[..., 5] * zocmass)
    # pbfract2[...,3] (the KS-to-KI target) is structurally zero (m7_dnum's
    # "pointless calculation" comment, mo_ham_m7.F90:2678-2685); kept as an
    # explicit no-op add for fidelity with the native statement sequence.
    _add_ks(ISO4KS, pbfract2[..., 3] * zanli1 * za4av1)
    _add_ks(IBCKS, pbfract2[..., 3] * zbcmass)
    _add_ks(IOCKS, pbfract2[..., 3] * zocmass)

    # --- m7_concoag: insoluble-to-soluble ageing (mo_ham_m7.F90:3579-3827).
    # Runs unconditionally (not gated on ``coagulation``), exactly as the
    # native m7_delcoa calls it outside any lscoag check.
    zso4_5 = zanli0 * pbfract1[..., 3] * za4av0 + insoluble_sulfate[..., 0]
    zso4_6 = (zanli0 * pbfract1[..., 4] * za4av0 + zanli1 * pbfract2[..., 4] * za4av1
              + insoluble_sulfate[..., 1])
    zso4_7 = (zanli0 * pbfract1[..., 5] * za4av0 + zanli1 * pbfract2[..., 5] * za4av1
              + insoluble_sulfate[..., 2])
    safe_crtcst = jnp.where(crtcst > eps, crtcst, 1.0)
    gate5 = (number[..., _KI] >= CMIN_AERNL) & (crtcst[..., 0] > eps)
    gate6 = (number[..., _AI] >= CMIN_AERNL) & (crtcst[..., 1] > eps)
    gate7 = (number[..., _CI] >= CMIN_AERNL) & (crtcst[..., 2] > eps)
    zcrit_5 = jnp.where(gate5, jnp.minimum(number[..., _KI], zso4_5 / safe_crtcst[..., 0]), 0.0)
    zcrit_6 = jnp.where(gate6, jnp.minimum(number[..., _AI], zso4_6 / safe_crtcst[..., 1]), 0.0)
    zcrit_7 = jnp.where(gate7, jnp.minimum(number[..., _CI], zso4_7 / safe_crtcst[..., 2]), 0.0)

    # Particle-equivalents (in units of the mean particle ``pttn``) each
    # insoluble mode holds before ageing: section 1's ``ki_factor`` where KI
    # coagulated (its mass was just set to ki_factor*pttn*1e12), otherwise the
    # mode's number, since pttn = mass/number (m7_averageproperties).
    basis_5 = jnp.where(gate4, ki_factor, number[..., _KI])
    basis_6, basis_7 = number[..., _AI], number[..., _CI]

    number = number.at[..., _KS].add(zcrit_5).at[..., _KI].add(-zcrit_5)
    number = number.at[..., _AS].add(zcrit_6).at[..., _AI].add(-zcrit_6)
    number = number.at[..., _CS].add(zcrit_7).at[..., _CI].add(-zcrit_7)

    def _age(index, basis, zcrit):
        # Native: paerml = paerml - zcrit*1e12*pttn (mo_ham_m7.F90:3777-3822).
        # When every particle ages (zcrit = number) that subtracts two equal
        # totals and leaves rounding noise of either sign, in float64 as in
        # float32 (whose noise is ~5e8 times larger). The
        # robust form is the same quantity factored, (basis - zcrit)*pttn*1e12:
        # zcrit = MIN(number, ...) <= basis, so it is exactly zero for full
        # ageing and non-negative otherwise. Where pttn = 0 (mass below
        # cmin_aerml) the native update subtracts nothing, so neither does it.
        if not robust:
            return mass.at[..., index].add(-zcrit * 1e12 * mean_mass[..., index])
        factored = (basis - zcrit) * mean_mass[..., index] * 1e12
        return mass.at[..., index].set(jnp.where(mean_mass[..., index] > 0.0, factored, mass[..., index]))

    mass_delta = mass_delta.at[..., ISO4KS].add(insoluble_sulfate[..., 0])
    mass = _age(IBCKI, basis_5, zcrit_5)
    mass = _age(IOCKI, basis_5, zcrit_5)
    mass_delta = mass_delta.at[..., IBCKS].add(zcrit_5 * 1e12 * mean_mass[..., IBCKI])
    mass_delta = mass_delta.at[..., IOCKS].add(zcrit_5 * 1e12 * mean_mass[..., IOCKI])

    mass_delta = mass_delta.at[..., ISO4AS].add(insoluble_sulfate[..., 1])
    mass = _age(IDUAI, basis_6, zcrit_6)
    mass_delta = mass_delta.at[..., IDUAS].add(zcrit_6 * 1e12 * mean_mass[..., IDUAI])

    mass_delta = mass_delta.at[..., ISO4CS].add(insoluble_sulfate[..., 2])
    mass = _age(IDUCI, basis_7, zcrit_7)
    mass_delta = mass_delta.at[..., IDUCS].add(zcrit_7 * 1e12 * mean_mass[..., IDUCI])

    # --- Section 4: final aggregation (mo_ham_m7.F90:4674-4702). ----------
    mass = mass + mass_delta * jnp.asarray(_SECTION4_MASK, dtype=dtype)
    return mass, number, mass_delta
