"""Working-dtype selection of M7's cancellation-free expression forms."""
import jax.numpy as jnp
import numpy as np


def cancellation_free(dtype, choice=None):
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

    ``choice`` is the caller's static ``cancellation_free`` option of the
    public entry points: ``None`` (default) applies the dtype rule above;
    ``True`` selects the cancellation-free forms in any dtype (they are also
    the more accurate float64 evaluation wherever the native form cancels,
    see ``docs/validation.md``); ``False`` selects the native forms in any
    dtype (the compiled-Fortran expressions, also in float32).
    """
    if choice is None:
        return jnp.finfo(dtype).bits < 64
    if not isinstance(choice, (bool, np.bool_)):
        raise TypeError(f'cancellation_free must be None, True or False (static), got {choice!r}')
    return bool(choice)
