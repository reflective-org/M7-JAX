"""Observe every complete native M7 call without altering scientific routines.

The driver retains each case's state for ``steps`` consecutive calls under fixed
forcing. Checkpoints are copied after the original whole M7 routine returns and
written outside the measured loop. This is a trajectory observer, not a timing
benchmark. The synthetic registry and exclusions are those of ``m7_oracle``.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from tools.m7_oracle import DRIVER, build_oracle, run_oracle


def _observing_driver():
    """Move the existing final-state copy inside the loop and retain all copies."""
    copy_start = DRIVER.index(' output(:,i)=0\n')
    copy_end = DRIVER.index('\n enddo\nenddo\ncall cpu_time(finish)', copy_start)
    copy_block = DRIVER[copy_start:copy_end]
    driver = DRIVER[:copy_start] + DRIVER[copy_end:]
    anchor = ' end select\n enddo\n'
    if driver.count(anchor) != 1:
        raise ValueError('Native driver observation anchor changed')
    driver = driver.replace(anchor, ' end select\n' + copy_block +
                            '\n trajectory(:,step,i)=output(:,i)\n enddo\n', 1)
    driver = driver.replace('real(dp),allocatable :: input(:,:), output(:,:)',
                            'real(dp),allocatable :: input(:,:), output(:,:), trajectory(:,:,:)')
    driver = driver.replace('allocate(input(88,nc),output(107,nc))',
                            'allocate(input(88,nc),output(107,nc),trajectory(107,steps,nc))')
    driver = driver.replace('call cpu_time(finish)\n', '''call cpu_time(finish)
open(unit=92,file='trajectory.bin',status='replace',access='stream',form='unformatted')
write(92) trajectory
close(92)
''', 1)
    return driver


def build_trajectory(directory, checked=True):
    """Build original native sources plus a copy-only per-call observer."""
    base, provenance = build_oracle(directory, checked=checked)
    directory = base.parent
    driver_path = directory / 'trajectory_driver.F90'
    driver_path.write_text(_observing_driver())
    files = [name for name in provenance['generated_source_sha256'] if name != 'driver.F90']
    executable = directory / 'm7_trajectory'
    compiler = shutil.which('gfortran')
    start = time.perf_counter()
    result = subprocess.run(
        [compiler, *provenance['flags'], *files, driver_path.name, '-o', str(executable)],
        cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    provenance = dict(provenance)
    provenance['trajectory_compile_seconds'] = time.perf_counter() - start
    provenance['trajectory_driver_sha256'] = hashlib.sha256(driver_path.read_bytes()).hexdigest()
    provenance['observation'] = (
        'Existing output assignments moved inside the driver step loop; a binary64 '
        'checkpoint is copied after every original whole M7 call. No process source changes.')
    (directory / 'trajectory_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return executable, provenance


def run_trajectory(executable, cases, steps=1000):
    """Return ``(outputs, timing)`` with outputs shaped (steps, cases, ...).

    ``cases`` follows ``m7_oracle.run_oracle``. The first checkpoint is after
    call one (there is no duplicated initial-state checkpoint). All calls use
    the same forcing and options, while number, mass, gas and geometry persist.
    """
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer')
    executable = Path(executable).resolve()
    final, timing = run_oracle(executable, cases, operation='full', steps=int(steps))
    count = timing['cases']
    raw = np.fromfile(executable.parent / 'trajectory.bin', dtype='=f8')
    if raw.size != 107 * steps * count:
        raise ValueError('Native trajectory checkpoint size does not match requested dimensions')
    raw = raw.reshape(count, steps, 107).transpose(1, 0, 2)
    # Decode in the same stable insertion order as the base oracle's wire format.
    outputs = {}
    offset = 0
    for field, value in final.items():
        width = value.shape[-1] if value.ndim == 2 else 1
        data = raw[..., offset:offset + width]
        outputs[field] = data.copy() if width > 1 else data[..., 0].copy()
        offset += width
        np.testing.assert_array_equal(outputs[field][-1], value,
                                      err_msg=f'Native {field} final checkpoint disagrees with final output')
    if offset != 107:
        raise ValueError('Base oracle output layout changed')
    timing = dict(timing)
    timing['observation'] = 'Every post-call state; checkpoint copies included in native CPU measurement'
    return outputs, timing
