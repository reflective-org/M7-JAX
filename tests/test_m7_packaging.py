# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""The kappa lookup ships as package data, complete, and its absence is loud.

``src/m7_jax/data/lut_kappa.nc`` is a symlink to
``reference/input/lut_kappa.nc``. When the link dangles (a build from a copy
without ``reference/input``) or is checked out as a text placeholder,
setuptools still builds a wheel, just without the table, so the wheel test
below and the runtime check in ``default_kappa_table_path`` guard it.
"""
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from m7_jax import properties

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TABLE = ROOT / 'reference' / 'input' / 'lut_kappa.nc'


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def test_default_kappa_table_path_is_the_reference_table():
    path = properties.default_kappa_table_path()
    assert path.is_file()
    assert path.stat().st_size == REFERENCE_TABLE.stat().st_size == 45_450_972
    assert _sha256(path.read_bytes()) == _sha256(REFERENCE_TABLE.read_bytes())
    table = properties.load_kappa_table(path)
    assert table.growth.shape == (98, 17, 62, 55)


@pytest.mark.parametrize('broken', ['dangling-link', 'text-placeholder'])
def test_missing_package_data_raises_a_clear_error(tmp_path, monkeypatch, broken):
    package = tmp_path / 'm7_jax'
    (package / 'data').mkdir(parents=True)
    link = package / 'data' / 'lut_kappa.nc'
    if broken == 'dangling-link':
        link.symlink_to('../../../reference/input/lut_kappa.nc')
    else:  # what git writes where symlinks are not supported
        link.write_text('../../../reference/input/lut_kappa.nc')
    monkeypatch.setattr(properties, '__file__', str(package / 'properties.py'))
    with pytest.raises(FileNotFoundError, match='package data m7_jax/data/lut_kappa.nc'):
        properties.default_kappa_table_path()


def _setuptools_skip_reason():
    if importlib.util.find_spec('setuptools') is None:
        return ('setuptools is not installed in this environment, so '
                '`pip wheel --no-build-isolation` cannot build the package offline')
    import setuptools
    if int(setuptools.__version__.split('.')[0]) < 77:
        return f'pyproject.toml needs setuptools>=77, found {setuptools.__version__}'
    return None


@pytest.mark.skipif(_setuptools_skip_reason() is not None, reason=str(_setuptools_skip_reason()))
def test_wheel_contains_the_full_kappa_table(tmp_path):
    """Build a wheel (about 45 MB, a few seconds) and check the table inside.

    The build runs in a copy of the files the wheel is made from, symlink
    preserved, so it leaves no build/ or egg-info in the checkout."""
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('pyproject.toml', 'LICENSE', 'NOTICE'):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / 'LICENSES', source / 'LICENSES')
    shutil.copytree(ROOT / 'src', source / 'src', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    shutil.copytree(ROOT / 'reference' / 'input', source / 'reference' / 'input')
    assert (source / 'src' / 'm7_jax' / 'data' / 'lut_kappa.nc').is_symlink()

    wheels = tmp_path / 'wheels'
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                    '--no-index', '-q', '-w', str(wheels), str(source)],
                   check=True, capture_output=True, text=True, timeout=300)
    (wheel,) = wheels.glob('m7_jax-*.whl')
    with zipfile.ZipFile(wheel) as archive:
        data = archive.read('m7_jax/data/lut_kappa.nc')
    assert len(data) == REFERENCE_TABLE.stat().st_size
    assert _sha256(data) == _sha256(REFERENCE_TABLE.read_bytes())
