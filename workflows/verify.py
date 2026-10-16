#!/usr/bin/env python3
"""One read-only verification entry point for CF-State.

Default: offline source-provenance + numerical unit tests (safe for CI).
--full: additionally check local frozen status files and replay Atlanta,
Phase 7, and Phase 8 frozen results. Never runs a generating audit.

Run from any working directory:
    python workflows/verify.py
    python workflows/verify.py --full
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CODE_CHECKS = (
    ("Offline numerical + original script SHA256 tests",
     (sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")),
)

FROZEN_CHECKS = (
    ("Clinical frozen chain including new patients",
     (sys.executable, "workflows/clinical.py", "--check", "--include-optional")),
    ("Perturbation frozen evidence (excluding optional retrospective benchmark)",
     (sys.executable, "workflows/perturbations.py", "--check")),
    ("Atlanta 04k4 frozen numerical regression",
     (sys.executable, "workflows/projection_regression.py", "--check")),
    ("Phase 7 07h four-specimen numerical regression",
     (sys.executable, "workflows/followup_regression.py", "--check")),
    ("Phase 8 08g 17-donor metal-axis numerical regression",
     (sys.executable, "workflows/metal_axis_regression.py", "--check")),
)


def execute(full: bool, keep_going: bool = False) -> int:
    checks = CODE_CHECKS + FROZEN_CHECKS if full else CODE_CHECKS
    env = os.environ.copy()
    # Limit thread-dependent changes to floating point summation/order.
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    n_pass = 0
    failed = []
    print("CF-STATE | %s VERIFICATION" % ("FULL LOCAL" if full else "OFFLINE CODE"),
          flush=True)
    print("Repository:", ROOT, flush=True)
    print("Generating audits are NEVER executed.", flush=True)
    for i, (label, cmd) in enumerate(checks, 1):
        print("\n%s/%s | %s" % (i, len(checks), label), flush=True)
        print("$ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if result.returncode == 0:
            print("PASS:", label, flush=True)
            n_pass += 1
        else:
            print("FAIL:", label, "(exit %s)" % result.returncode, flush=True)
            failed.append(label)
            if not keep_going:
                break
    print("\nSUMMARY: %s/%s checks passed" % (n_pass, len(checks)), flush=True)
    if failed:
        print("Failed:", "; ".join(failed), flush=True)
        print("Frozen status/numerical failure is NOT permission to overwrite archived outputs.",
              flush=True)
        return 1
    if n_pass < len(checks):
        return 1
    print("All selected checks passed; source provenance retained.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Also validate local frozen data and results")
    parser.add_argument("--keep-going", action="store_true",
                        help="Run remaining checks after a failure")
    args = parser.parse_args()
    return execute(args.full, args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
