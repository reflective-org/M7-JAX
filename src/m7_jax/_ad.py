"""Reverse-mode-safe elementary operations with bit-identical forward values.

M7's native expressions are evaluated in every cell and masked with
``jnp.where``, and its gates (``cmin_aernl``, ``cminvol``, ...) admit values
far below one. Two derivative rules then overflow even where the function
itself is perfectly well behaved:

- ``d(x/y)/dy = -x/y**2``: ``y**2`` underflows (to zero in float32 for
  ``|y| < 1e-19``), so a quotient with a small but valid denominator gets an
  infinite partial derivative;
- ``d(x**(1/3))/dx`` is infinite at ``x = 0``.

A masked branch's zero cotangent times such an infinity is NaN, and an
unmasked one is infinite, either way poisoning every input gradient. The
helpers below return exactly the same forward value as the plain operation
(bit for bit) and differentiate it without the overflowing intermediate.
"""
import jax
import jax.numpy as jnp


@jax.custom_jvp
def div(x, y):
    """``x / y``, differentiated as ``(dx - (x/y)*dy)/y`` (no ``y**2``)."""
    return x / y


@div.defjvp
def _div_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    quotient = x / y
    return quotient, jnp.broadcast_to((dx - quotient * dy) / y, quotient.shape)


def cbrt_volume(x):
    """``x**(1/3)`` for a non-negative volume, with a zero derivative at 0.

    The value is ``x**(1/3)`` bit for bit: 0 at 0, NaN for a negative or NaN
    argument. Only the masked zero's derivative changes, from infinity to 0.
    """
    positive = x > 0
    root = jnp.where(positive, x, 1.0) ** (1 / 3)
    return jnp.where(positive, root, jnp.where(x == 0, jnp.zeros_like(root), jnp.nan))


def sqrt(x):
    """``sqrt(x)`` with a finite reverse-mode derivative at ``x = 0``.

    The forward value is ``jnp.sqrt(x)`` bit for bit: zero (of either sign)
    returns itself, and every other value, NaN and negatives included, goes
    through ``jnp.sqrt`` unchanged. The derivative ``1/(2 sqrt(x))`` is
    infinite at 0. Masked branches meet exactly 0, for example the analytical
    decay's nucleating radicals without a source, or the coagulation jump
    distance of two empty modes, and their zero cotangent times that infinity
    is NaN. The double ``where`` gives the zero a zero derivative instead.
    """
    zero = x == 0
    return jnp.where(zero, x, jnp.sqrt(jnp.where(zero, 1.0, x)))


def with_gradient_of(value_fn, gradient_fn):
    """``value_fn``'s value, differentiated as ``gradient_fn``.

    For two forms of the same function in real arithmetic -- a native
    expression whose partial derivatives are huge and cancelling (and
    overflow in float32), and an algebraically identical form whose are not
    -- this returns ``value_fn``'s value bit for bit and ``gradient_fn``'s
    derivative, which is the derivative of the same function.
    """
    @jax.custom_jvp
    def f(*args):
        return value_fn(*args)

    @f.defjvp
    def _jvp(primals, tangents):
        return value_fn(*primals), jax.jvp(gradient_fn, primals, tangents)[1]

    return f
