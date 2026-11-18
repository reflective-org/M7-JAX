# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""The wheel carries the kappa lookup table, or the build fails loudly."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from setup import KAPPA_TABLE, check_kappa_table

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TABLE = ROOT / 'reference' / 'input' / 'lut_kappa.nc'

setuptools = pytest.importorskip('setuptools')
if int(setuptools.__version__.split('.')[0]) < 77:
    pytest.skip('building m7-jax needs setuptools>=77 (license-files)', allow_module_level=True)


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


def test_wheel_ships_the_kappa_table(tmp_path):
    root = _project_copy(tmp_path, with_reference=True)
    result = _build_wheel(root, tmp_path / 'dist')
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = (tmp_path / 'dist').glob('m7_jax-*.whl')
    with zipfile.ZipFile(wheel) as archive:
        table = archive.getinfo('m7_jax/data/lut_kappa.nc')
    assert table.file_size == REFERENCE_TABLE.stat().st_size


def test_build_refuses_a_dangling_kappa_table_link(tmp_path):
    root = _project_copy(tmp_path, with_reference=False)
    assert (root / KAPPA_TABLE).is_symlink() and not (root / KAPPA_TABLE).exists()
    with pytest.raises(SystemExit, match='dangling'):
        check_kappa_table(root)
    result = _build_wheel(root, tmp_path / 'dist')
    assert result.returncode != 0
    assert 'dangling' in result.stdout + result.stderr
    assert not list((tmp_path / 'dist').glob('*.whl'))
