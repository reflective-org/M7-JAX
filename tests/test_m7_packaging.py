# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""The kappa lookup ships as package data, complete, and its absence is loud.

``src/m7_jax/data/lut_kappa.nc`` is a symlink to
``reference/input/lut_kappa.nc``. When the link dangles (a copy without
``reference/input``) or is checked out as a text placeholder, ``setup.py``
refuses to build, and ``default_kappa_table_path`` raises at run time for a
source tree used without installing.
"""
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from m7_jax import properties
from setup import KAPPA_TABLE, check_kappa_table

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TABLE = ROOT / 'reference' / 'input' / 'lut_kappa.nc'


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _setuptools_skip_reason():
    try:
        import setuptools
    except ImportError:
        return 'setuptools is not installed, so `pip wheel --no-build-isolation` cannot build offline'
    if int(setuptools.__version__.split('.')[0]) < 77:
        return f'building m7-jax needs setuptools>=77 (license-files), found {setuptools.__version__}'
    return None


needs_setuptools = pytest.mark.skipif(_setuptools_skip_reason() is not None,
                                      reason=str(_setuptools_skip_reason()))


# --- Run time: the default path is the reference table, or a clear error. -----

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


# --- Build time: the wheel carries the table, or the build fails. -------------

def _project_copy(tmp_path, with_reference):
    """The packaging inputs, with the table link as git checks it out."""
    root = tmp_path / 'project'
    root.mkdir()
    for name in ('pyproject.toml', 'setup.py', 'README.md', 'LICENSE', 'NOTICE'):
        shutil.copy2(ROOT / name, root / name)
    shutil.copytree(ROOT / 'LICENSES', root / 'LICENSES')
    shutil.copytree(ROOT / 'src', root / 'src', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    if with_reference:
        (root / 'reference' / 'input').mkdir(parents=True)
        (root / 'reference' / 'input' / 'lut_kappa.nc').symlink_to(REFERENCE_TABLE)
    return root


def _build_wheel(root, out):
    return subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                           '--no-index', '-w', str(out), str(root)],
                          capture_output=True, text=True, timeout=600)


@needs_setuptools
def test_wheel_ships_the_full_kappa_table(tmp_path):
    """Build a wheel (about 42 MB, a few seconds) from a copy of the packaging
    inputs, so the checkout gets no build/ or egg-info."""
    root = _project_copy(tmp_path, with_reference=True)
    result = _build_wheel(root, tmp_path / 'dist')
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = (tmp_path / 'dist').glob('m7_jax-*.whl')
    with zipfile.ZipFile(wheel) as archive:
        data = archive.read('m7_jax/data/lut_kappa.nc')
    assert len(data) == REFERENCE_TABLE.stat().st_size
    assert _sha256(data) == _sha256(REFERENCE_TABLE.read_bytes())


@needs_setuptools
def test_build_refuses_a_dangling_kappa_table_link(tmp_path):
    root = _project_copy(tmp_path, with_reference=False)
    assert (root / KAPPA_TABLE).is_symlink() and not (root / KAPPA_TABLE).exists()
    with pytest.raises(SystemExit, match='dangling'):
        check_kappa_table(root)
    result = _build_wheel(root, tmp_path / 'dist')
    assert result.returncode != 0
    assert 'dangling' in result.stdout + result.stderr
    assert not list((tmp_path / 'dist').glob('*.whl'))
