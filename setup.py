# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Build hook: refuse to build a package without the kappa lookup table.

``src/m7_jax/data/lut_kappa.nc`` is a link to ``reference/input/lut_kappa.nc``.
Without ``reference/`` (a subtree or vendored copy, ``git archive`` of part of
the tree, a checkout without symlink support) the link dangles, and setuptools
would otherwise build a wheel without the table and without a warning.
Everything else is configured in ``pyproject.toml``.
"""
from pathlib import Path

KAPPA_TABLE = Path('src') / 'm7_jax' / 'data' / 'lut_kappa.nc'


def check_kappa_table(root):
    """Raise ``SystemExit`` unless ``root`` holds a readable NetCDF kappa table."""
    path = Path(root) / KAPPA_TABLE
    if not path.is_file():
        raise SystemExit(f'{path} is missing or a dangling link to reference/input/lut_kappa.nc; '
                         'refusing to build m7-jax without the kappa lookup table.')
    with path.open('rb') as table:
        if table.read(3) != b'CDF':
            raise SystemExit(f'{path} is not a NetCDF file; refusing to build m7-jax without the kappa table.')


if __name__ == '__main__':
    from setuptools import setup

    check_kappa_table(Path(__file__).resolve().parent)
    setup()
