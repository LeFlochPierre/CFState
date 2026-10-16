"""Shared, read-only frozen-output checks for CF-State workflow entry points.

This module verifies filesystem contracts, not the underlying biological claims.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Stage:
    key: str
    name: str
    script: str
    files: tuple[str, ...]
    status: tuple[str, str] | None = None
    depends: tuple[str, ...] = ()
    optional: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def validate_definition(stages: Sequence[Stage]) -> None:
    keys = [stage.key for stage in stages]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate stage key')
    earlier = set()
    for stage in stages:
        if any(dep not in earlier for dep in stage.depends):
            raise ValueError('Stages not in dependency order: ' + stage.key)
        earlier.add(stage.key)
        for relative in (stage.script, *stage.files, *( (stage.status[0],) if stage.status else () )):
            p = Path(relative)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('Unsafe path in stage ' + stage.key + ': ' + relative)


def check_stages(root: Path, stages: Sequence[Stage],
                 include_optional: bool = False,
                 hashes: bool = False) -> list[dict]:
    validate_definition(stages)
    rows: list[dict] = []
    for stage in stages:
        if stage.optional and not include_optional:
            continue
        issues: list[str] = []
        found: list[dict] = []
        for relative in (stage.script, *stage.files):
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                issues.append('missing or empty: ' + relative)
                continue
            item = {'path': relative, 'bytes': path.stat().st_size}
            if hashes:
                item['sha256'] = sha256(path)
            found.append(item)
        if stage.status:
            relative, expected = stage.status
            path = root / relative
            if not path.is_file():
                issues.append('missing status: ' + relative)
            else:
                try:
                    if path.suffix.lower() == '.json':
                        actual = json.loads(path.read_text(encoding='utf-8'))['status']
                    else:
                        actual = path.read_text(encoding='utf-8').strip()
                    if actual != expected:
                        issues.append('unexpected status: ' + relative + ' (' + repr(actual) + ')')
                    elif hashes:
                        found.append({'path': relative, 'sha256': sha256(path),
                                      'bytes': path.stat().st_size})
                except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
                    issues.append('invalid status: ' + relative + ' (' + str(exc) + ')')
        rows.append({'key': stage.key, 'name': stage.name,
                     'depends': list(stage.depends), 'ok': not issues,
                     'issues': issues, 'checked': found})
    return rows


def run_cli(stages: Sequence[Stage], title: str, caution: str,
            self_test=None) -> int:
    """Common command-line interface for the frozen clinical/evidence manifests."""
    import argparse

    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--show', action='store_true',
                        help='Print dependency order without opening frozen data')
    parser.add_argument('--check', action='store_true',
                        help='Check local file and frozen status contracts (default)')
    parser.add_argument('--include-optional', action='store_true',
                        help='Also check historical/optional stages')
    parser.add_argument('--hash', action='store_true',
                        help='SHA256-hash verified files without modifying them')
    parser.add_argument('--json', action='store_true',
                        help='Return stage checks as machine-readable JSON')
    parser.add_argument('--self-test', action='store_true',
                        help='Test the checker using disposable synthetic fixtures')
    args = parser.parse_args()

    if args.self_test:
        if self_test is None:
            parser.error('no synthetic self-test available')
        self_test()
        return 0

    if args.show:
        validate_definition(stages)
        for index, stage in enumerate(stages, 1):
            suffix = ' (optional)' if stage.optional else ''
            print(f'{index:02d}. {stage.key}: {stage.name}{suffix}')
            print('    source:', stage.script)
            print('    depends:', ', '.join(stage.depends) or 'none')
        return 0

    root = args.root.resolve()
    rows = check_stages(root, stages, args.include_optional, args.hash)
    if args.json:
        print(json.dumps({'root': str(root), 'stages': rows}, indent=2))
    else:
        print(title)
        print('Root:', root)
        for row in rows:
            print('[%s] %s: %s' %
                  ('PASS' if row['ok'] else 'FAIL', row['key'], row['name']))
            for issue in row['issues']:
                print('    -', issue)
        print('Checked stages: %s/%s PASS' %
              (sum(row['ok'] for row in rows), len(rows)))
        print(caution)
        print('No frozen inputs, scientific outputs, or archived audits modified.')
    return 0 if all(row['ok'] for row in rows) else 1
