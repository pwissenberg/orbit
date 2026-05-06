"""Compatibility shim for numpy + nptyping (used by pecanpy/SPACE).

Must be imported BEFORE any SPACE or pecanpy imports.
"""

import numpy as np

_ALIASES = {
    "bool8": "bool_",
    "object0": "object_",
    "int0": "intp",
    "uint0": "uintp",
    "void0": "void",
    "bytes0": "bytes_",
    "str0": "str_",
    "float_": "float64",
    "complex_": "complex128",
    "longfloat": "longdouble",
    "singlecomplex": "complex64",
    "cfloat": "complex128",
    "longcomplex": "clongdouble",
    "clongfloat": "clongdouble",
    "string_": "bytes_",
    "unicode_": "str_",
    "uint": "uint64",
}

for _attr, _replacement in _ALIASES.items():
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(np, _replacement))
