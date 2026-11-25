"""Compact a finished run's audit from one file per blob into audit/pack: one file per stream (events, rgb, depth,
state, other) plus a gzip index. Blob bytes and digests are unchanged; sim2sim.audit reads either layout.

Streams keep blobs in first-appearance (event) order; the index records each blob's first event, which ties an image or
state to its event's time. rgb/depth are PNGs and stay raw; events, states and other blobs (wire bytes, projects) are
zlib-compressed in ~4 MiB chunks. Loose blob files are removed only after the packed audit verifies end to end with
read_audit (hash chain, every attachment's size and digest) and every packed blob, referenced or not, re-hashes to its
name. A run named by a live Slurm job, still preparing, or with an unsealed audit touched in the last 2 h is refused.

    python -m sim2sim.compact RUN_DIR [RUN_DIR ...]      (one JSON line per run)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audit import PACK_COLUMNS, PACK_STREAMS, AuditFailure, PackedBlobs, _digest, _safe_file, read_audit

STREAM_OF = {'rgb': 'rgb', 'wrist_rgb': 'rgb', 'url': 'rgb', 'image_url': 'rgb', 'depth': 'depth', 'wrist_depth': 'depth',
             'state': 'state', 'before_state': 'state', 'after_state': 'state', 'beforestate': 'state', 'afterstate': 'state'}
CHUNK = 4 << 20
QUIET_SECONDS = 2 * 3600
THREADS = 32   # loose blob reads and unlinks are NFS-latency bound


def live_run_ids():
    """Text of every live Slurm job of this user except the caller's own (its command line names its run IDs)."""
    own = os.environ.get('SLURM_JOB_ID')
    jobs = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-t', 'R,PD,S,CG', '-o', '%i'],
                          capture_output=True, text=True, check=True).stdout.split()
    return '\n'.join(subprocess.run(['scontrol', 'show', 'job', '-o', job], capture_output=True, text=True).stdout
                     for job in jobs if job != own)


def _is_digest(name):
    return len(name) == 64 and all(c in '0123456789abcdef' for c in name)


def _blob_order(root, pool):
    """[(digest, stream, first seq)] for every blob in event order, then loose blobs no event references (seq 0)."""
    blobs = root / 'blobs'
    pointers = [json.loads(line) for line in _safe_file(root, 'events.jsonl').open('rb')]
    envelopes = pool.map(lambda p: json.loads((blobs / p['event_sha256']).read_bytes()), pointers)
    order, seen = [], set()
    for pointer, envelope in zip(pointers, envelopes):
        for digest, stream in [(pointer['event_sha256'], 'events')] + [
                (item['sha256'], STREAM_OF.get(str(item['path'][-1]), 'other')) for item in envelope['attachments']]:
            if digest not in seen:
                seen.add(digest)
                order.append((digest, stream, pointer['seq']))
    order += [(name, 'other', 0) for name in sorted(os.listdir(blobs)) if _is_digest(name) and name not in seen]
    return order


def _write_pack(root, order, pool, staging):
    """Write the streams and index into staging; every loose blob is re-hashed as it is packed."""
    staging.mkdir()
    files = {name: (staging / f'{name}.pack').open('wb') for name in PACK_STREAMS}
    buffers = {name: [] for name in PACK_STREAMS}   # pending (digest, data, seq) per stream
    rows = []

    def flush(stream):
        if not buffers[stream]:
            return
        raw = b''.join(data for _, data, _ in buffers[stream])
        packed = zlib.compress(raw, 6) if PACK_STREAMS[stream] == 'zlib' else raw
        start = files[stream].tell()
        files[stream].write(packed)
        offset = 0
        for digest, data, seq in buffers[stream]:
            rows.append((digest, stream, start, len(packed), offset, len(data), seq))
            offset += len(data)
        buffers[stream].clear()

    def read(entry):
        data = (root / 'blobs' / entry[0]).read_bytes()
        if _digest(data) != entry[0]:
            raise AuditFailure(f'loose blob {entry[0]} does not match its digest; not compacting')
        return data

    pending = {name: 0 for name in PACK_STREAMS}
    for start in range(0, len(order), 4096):
        batch = order[start:start + 4096]
        for (digest, stream, seq), data in zip(batch, pool.map(read, batch)):
            buffers[stream].append((digest, data, seq))
            pending[stream] += len(data)
            if pending[stream] >= CHUNK:
                flush(stream)
                pending[stream] = 0
    for stream in PACK_STREAMS:
        flush(stream)
    for handle in files.values():
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    with gzip.open(staging / 'index.tsv.gz', 'wt', encoding='ascii') as handle:
        handle.write('\t'.join(PACK_COLUMNS) + '\n')
        handle.writelines('\t'.join(map(str, row)) + '\n' for row in rows)
    with open(staging / 'index.tsv.gz', 'rb') as handle:
        os.fsync(handle.fileno())


def compact(run, live=''):
    """Compact one run; returns a summary dict. Raises on anything unsafe or unverifiable (nothing is removed then)."""
    started = time.time()
    run = Path(run).resolve()
    root = run / 'audit'
    status = json.loads((run / 'status.json').read_text())
    manifest = json.loads(_safe_file(root, 'manifest.json').read_bytes())
    if status.get('status') in ('preparing', 'prepared'):
        raise AuditFailure(f"run is {status.get('status')}")
    if run.name in live:
        raise AuditFailure('a live Slurm job names this run')
    if not manifest.get('sealed') and time.time() - (root / 'events.jsonl').stat().st_mtime < QUIET_SECONDS:
        raise AuditFailure('unsealed audit was written in the last 2 h')
    blobs, pack, staging = root / 'blobs', root / 'pack', root / 'pack.tmp'
    loose_before = sum(1 for name in os.listdir(blobs) if _is_digest(name))
    with ThreadPoolExecutor(THREADS) as pool:
        if not pack.exists():
            if loose_before == 0:
                return {'run': run.name, 'status': 'nothing to compact'}
            if staging.exists():
                shutil.rmtree(staging)   # this tool's own unfinished output
            _write_pack(root, _blob_order(root, pool), pool, staging)
            staging.rename(pack)
        for _ in read_audit(run, budget=False):   # packed audit must verify end to end before anything is removed
            pass
        packs = PackedBlobs(root)
        try:
            for digest, (_, _, _, _, size) in packs.index.items():
                if _digest(packs.get(digest, size)) != digest:
                    raise AuditFailure(f'packed blob {digest} does not match its digest')
            loose = [name for name in os.listdir(blobs) if _is_digest(name)]
            missing = [name for name in loose if name not in packs.index]
            if missing:
                raise AuditFailure(f'{len(missing)} loose blobs are not in the pack; not removing anything')
            list(pool.map(lambda name: (blobs / name).unlink(), loose))
            blob_bytes = sum(entry[4] for entry in packs.index.values())
        finally:
            packs.close()
    pack_bytes = sum(path.stat().st_size for path in pack.iterdir())
    return {'run': run.name, 'status': 'compacted', 'loose_files_removed': loose_before, 'pack_files': len(list(pack.iterdir())),
            'blob_bytes': blob_bytes, 'pack_bytes': pack_bytes, 'seconds': round(time.time() - started, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    live, failed = live_run_ids(), 0
    for run in args.runs:
        try:
            print(json.dumps(compact(run, live)), flush=True)
        except Exception as exc:   # report every run; a refused or failed run keeps all its loose blobs
            failed += 1
            print(json.dumps({'run': run.name, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
