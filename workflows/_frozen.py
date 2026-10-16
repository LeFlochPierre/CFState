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
