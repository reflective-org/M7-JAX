"""Working-dtype selection of M7's cancellation-free expression forms."""
import jax.numpy as jnp


def cancellation_free(dtype):
    """Whether the working ``dtype`` selects M7's cancellation-free forms.

    Several native M7 expressions obtain a small net quantity as the
    difference of two nearly equal totals. In float64 the result keeps
    enough digits to reproduce the compiled r7492 Fortran, which evaluates
    the same expressions in double precision, and the native forms are used
    so the float64 path stays bit-for-bit what the parity suite validates. In
    any narrower dtype the same subtractions lose every significant digit
    and can change sign (negative particle counts and masses), so each such
    site switches to an algebraically identical form that computes the net
    quantity directly. These are precision adaptations, not deviations:
    the two forms are equal in real arithmetic, and the choice is made from
    the static working dtype, never from the data. ``docs/model.md``
    ("Float32 precision adaptations") lists every site.
    """
    return jnp.finfo(dtype).bits < 64
