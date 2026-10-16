"""Protect original CF-State audit source hashes during refactoring.

The frozen inventory was captured before adding the new reusable package.
Moving/changing historic generator scripts would invalidate output provenance.
"""
from __future__ import annotations

import csv
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'docs/CODE_INDEX.csv'


class HistoricalSourceProvenance(unittest.TestCase):
    def test_original_script_hashes_remain_unchanged(self):
        with INDEX.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 249)
        failures = []
        for row in rows:
            relative = row['path']
            expected = row['sha256']
            path = ROOT / relative
            if not path.is_file():
                failures.append(relative + ' MISSING')
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(relative + ' SHA256_CHANGED')
        self.assertFalse(failures, 'Frozen generator script changed:\n' + '\n'.join(failures[:30]))


if __name__ == '__main__':
    unittest.main()
