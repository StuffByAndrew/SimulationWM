"""Submitted-project ZIP decoding and safe extraction (shared by the evaluator, the candidate worker and predictor workers)."""
from __future__ import annotations
import base64
import io
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
import xml.etree.ElementTree as ET

MAX_ZIP = 8 * 1024 * 1024
MAX_EXPANDED = 32 * 1024 * 1024
MAX_FILES = 256


def decode_submission(value: str) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_ZIP * 4 // 3 + 8):
        raise ValueError('submission exceeds base64 size limit')
    try:
        data = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError('invalid base64 submission') from exc
    if len(data) > MAX_ZIP:
        raise ValueError('submission exceeds ZIP size limit')
    return data


def _relative(value):
    if not value or '\\' in value or ':' in value or '\x00' in value:
        raise ValueError('unsafe project path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or value.startswith('/'):
        raise ValueError('unsafe project path')
    return path


def extract_submission(data: bytes, destination: Path, entrypoint='scene.xml'):
    if len(data) > MAX_ZIP:
        raise ValueError('submission exceeds ZIP size limit')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_EXPANDED:
            raise ValueError('expanded submission exceeds limits')
        seen = set()
        for info in infos:
            path = _relative(info.filename)
            if str(path) in seen:
                raise ValueError('duplicate ZIP path')
            seen.add(str(path))
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError('special files and symlinks are forbidden')
            target = destination.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    scene = destination.joinpath(*_relative(entrypoint).parts)
    if not scene.is_file():
        raise ValueError('missing scene entrypoint')
    # Every XML file is checked as MJCF, and so is the entrypoint unless it is a predictor program (src/sim2sim/predictor.py)
    # or a low-rank parameter answer (JSON, metric `lowrank`).
    pending = [scene] * (scene.suffix not in ('.py', '.json')) + [p for p in destination.rglob('*') if p.is_file() and p.suffix.lower() == '.xml']
    checked = set()
    while pending:
        xml = pending.pop()
        if xml in checked:
            continue
        checked.add(xml)
        raw = xml.read_bytes()
        if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
            raise ValueError('XML entities are forbidden')
        root = ET.fromstring(raw)
        for element in root.iter():
            if element.tag in ('plugin', 'extension'):
                raise ValueError('MJCF plugins are forbidden')
            for key in ('file', 'meshdir', 'texturedir', 'assetdir'):
                value = element.get(key)
                if value:
                    rel = _relative(value)
                    resolved = (xml.parent / str(rel)).resolve()
                    if not resolved.is_relative_to(destination.resolve()):
                        raise ValueError('external MJCF reference')
                    if element.tag == 'include' and key == 'file':
                        pending.append(resolved)
    return scene
