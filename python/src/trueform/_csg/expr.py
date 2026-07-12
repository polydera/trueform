"""
CSG boolean expressions over operand indices.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

_OR = -1
_AND = -2
_SUB = -3
_NOT = -4


def _as_expr(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return op(value)
    raise TypeError(
        f"expected tf.op(i) expression or operand index, got {type(value)!r}"
    )


class Expr:
    """
    A boolean expression tree over operand indices.

    Build leaves with ``tf.op(i)`` and combine with operators:
    ``|`` (union), ``&`` (intersection), ``-`` (difference),
    ``~`` (complement). Plain integers are accepted as operands:
    ``tf.op(0) - 1`` is the difference of operand 0 and operand 1.

    Examples
    --------
    >>> carved = tf.op(0) - tf.op(1) - tf.op(2)
    >>> shared = tf.op(0) & tf.op(1)
    >>> outside = ~(tf.op(0) | tf.op(1))
    """

    __slots__ = ("_program",)

    def __init__(self, program):
        self._program = program

    def program(self):
        """The flat postfix program consumed by the native layer."""
        return list(self._program)

    def _combine(self, other, code, reflected=False):
        other = _as_expr(other)
        left, right = (other, self) if reflected else (self, other)
        return Expr(left._program + right._program + [code])

    def __or__(self, other):
        return self._combine(other, _OR)

    def __ror__(self, other):
        return self._combine(other, _OR, reflected=True)

    def __and__(self, other):
        return self._combine(other, _AND)

    def __rand__(self, other):
        return self._combine(other, _AND, reflected=True)

    def __sub__(self, other):
        return self._combine(other, _SUB)

    def __rsub__(self, other):
        return self._combine(other, _SUB, reflected=True)

    def __invert__(self):
        return Expr(self._program + [_NOT])

    def __repr__(self):
        return f"tf.Expr(program={self._program})"


def op(operand_id):
    """
    A leaf expression selecting operand ``operand_id``.

    Parameters
    ----------
    operand_id : int
        Index of the operand in the ``CsgGraph``'s form list.

    Returns
    -------
    Expr
    """
    i = int(operand_id)
    if i < 0:
        raise ValueError(f"operand index must be non-negative, got {i}")
    return Expr([i])
