"""Lightweight researcher job monitor, meant to stay up on a login node (`scripts/serve-monitor.sh`).

Shows running strict jobs and how far into their wall budget their runs are, recent failures with their reasons, the
Slurm CPU/RAM utilisation of every strict job (persisted, backfilled from Slurm history, with a suggested request size),
API spend per key (daily and lifetime, from the cost ledger) and storage headroom. Stdlib only; one refresh thread;
binds localhost. Researcher-only: it reads spend and private run status, so never expose it to agents.
"""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = (Path('/n/holylabs/LABS/qianqian_lab/Lab/aejxie/sim2sim-experiments'),
         Path('/n/netscratch/qianqian_lab/Lab/aejxie/sim2sim-experiments'), REPO / 'experiments')
LEDGER = REPO / 'experiments' / 'cost-ledger'
JOB_NAME = 'sim2sim-strict'
HISTORY_START = dt.date(2026, 10, 25)   # sacct rejects wide windows, so history is fetched one day at a time
KEY_CAPS = {'GEMINI_API_KEY_BYCOST': 15000.0}   # lifetime cap (AGENTS.md)
FIELDS = ('JobID', 'JobName', 'State', 'ExitCode', 'Elapsed', 'AllocCPUS', 'ReqMem', 'MaxRSS', 'TotalCPU', 'Start',
          'End', 'Partition', 'NodeList', 'Timelimit', 'SubmitLine')
FINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')


def seconds(text):
    """sacct durations: [D-]HH:MM:SS[.fff] or MM:SS.fff; empty is 0."""
    if not text or text in ('INVALID', 'UNLIMITED', 'Partition_Limit'):
        return 0.0
    days, _, rest = text.rpartition('-')
    parts = [float(p) for p in rest.split(':')]
    parts = [0.0] * (3 - len(parts)) + parts
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def nbytes(text, cpus=1):
    """sacct sizes such as 733968K, 1.5G or 48G (a trailing c means per CPU); empty is None."""
    if not text:
        return None
    per_cpu = text[-1] == 'c'
    text = text.rstrip('cn')
    scale = 1024 ** ('KMGTP'.index(text[-1]) + 1) if text[-1] in 'KMGTP' else 1
    return float(text.rstrip('KMGTP')) * scale * (cpus if per_cpu else 1)


def _role(name):
    if name == 'batch':
        return 'controller'
    match = re.match(r'sim2sim-([a-z]+)-[0-9a-f]+$', name)
    return match.group(1) if match else name


def parse_sacct(text):
    """{job id: job} for strict jobs, each with its steps, from `sacct -n -P -o FIELDS` output."""
    jobs, steps = {}, []
    for line in text.splitlines():
        row = dict(zip(FIELDS, line.split('|', len(FIELDS) - 1)))
        if len(row) != len(FIELDS):
            continue
        if '.' in row['JobID']:
            steps.append(row)
        else:
            jobs[row['JobID']] = row
    jobs = {key: job for key, job in jobs.items() if job['JobName'] == JOB_NAME}
    for job in jobs.values():
        submit = job['SubmitLine'].split()
        job['runs'] = submit[submit.index('scripts/experiment-strict.slurm') + 1:] if 'scripts/experiment-strict.slurm' in submit else []
        job['steps'] = []
    for step in steps:
        parent = jobs.get(step['JobID'].split('.')[0])
        if parent is not None and not step['JobID'].endswith('.extern'):
            parent['steps'].append(step)
    return jobs


def _stamp(text):
    return None if text in ('', 'Unknown', 'None') else dt.datetime.fromisoformat(text).timestamp()


def usage(job, live=None):
    """CPU and memory use of one job. `live` maps running step IDs to sstat (rss bytes, cpu seconds)."""
    alloc, elapsed = int(job['AllocCPUS'] or 0), seconds(job['Elapsed'])
    roles, events, cpu = {}, [], 0.0
    for step in job['steps']:
        rss, step_cpu = nbytes(step['MaxRSS']), seconds(step['TotalCPU'])
        if live and step['JobID'] in live:
            rss, step_cpu = live[step['JobID']]
        cpu += step_cpu
        role = roles.setdefault(_role(step['JobName']), {'steps': 0, 'cpu_s': 0.0, 'busy_s': 0.0, 'max_rss': 0.0, 'alloc': 0})
        role.update(steps=role['steps'] + 1, cpu_s=role['cpu_s'] + step_cpu, busy_s=role['busy_s'] + seconds(step['Elapsed']),
                    max_rss=max(role['max_rss'], rss or 0.0), alloc=max(role['alloc'], int(step['AllocCPUS'] or 0)))
        start, end = _stamp(step['Start']), _stamp(step['End']) or time.time()
        # The batch step nominally holds the whole allocation; only container steps reserve CPUs of their own.
        step_cpus = 0 if step['JobName'] == 'batch' else int(step['AllocCPUS'] or 0)
        if start:
            events += [(start, 1, rss or 0.0, step_cpus), (end, 0, -(rss or 0.0), -step_cpus)]
    peak = current = 0.0
    peak_cpus = cpus = 0
    for _, _, change, cpu_change in sorted(events):   # releases (0) before starts (1) at equal times
        current, cpus = current + change, cpus + cpu_change
        peak, peak_cpus = max(peak, current), max(peak_cpus, cpus)
    requested = nbytes(job['ReqMem'], alloc)
    return {'alloc_cpus': alloc, 'elapsed_s': elapsed, 'cpu_s': cpu, 'cores_used': cpu / elapsed if elapsed else None,
            'cpu_efficiency': cpu / (elapsed * alloc) if elapsed and alloc else None, 'req_mem': requested,
            'peak_mem_bound': peak or None, 'mem_efficiency': peak / requested if peak and requested else None,
            'peak_step_cpus': peak_cpus, 'roles': roles}


def run_command(args, timeout=120):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args[:2])} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def sacct(start, end=None):
    args = ['sacct', '-u', os.environ.get('USER', ''), '-n', '-P', '-S', start] + (['-E', end] if end else []) + ['-o', ','.join(FIELDS)]
    return parse_sacct(run_command(args))


def sstat(job_id):
    """{step id: (max rss bytes, cpu seconds so far)} for a running job's live steps."""
    live = {}
    for line in run_command(['sstat', '-n', '-P', '-a', '-j', job_id, '-o', 'JobID,MaxRSS,AveCPU'], timeout=30).splitlines():
        step, rss, cpu = (line.split('|') + ['', ''])[:3]
        if not step.endswith('.extern'):
            live[step] = (nbytes(rss), seconds(cpu))
    return live


def run_dir(run_id):
    found = [root / 'runs' / run_id for root in ROOTS if (root / 'runs' / run_id / 'status.json').is_file()]
    return found[0] if len(found) == 1 else None


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_status(run_id, job_ended=False):
    """Researcher summary of one run's status.json (no scores). A run still 'running' after its job ended was killed
    before the controller could record why (e.g. the job hit a storage quota or was cancelled)."""
    directory = run_dir(run_id)
    status = _read_json(directory / 'status.json') if directory else None
    if status is None:
        return {'id': run_id, 'status': 'missing', 'root': None}
    result = status.get('result') or {}
    started = status.get('started_at')
    state = status.get('status')
    if job_ended and state == 'running':
        state = 'killed (status never finalised)'
    return {'id': run_id, 'root': directory.parent.parent.name if directory.parent.parent != REPO / 'experiments' else 'home',
            'status': state, 'reason': status.get('reason') or result.get('reason'),
            'error': (status.get('error') or '')[:400], 'stage': status.get('stage'), 'started_at': started,
            'wall_limit_s': status.get('wall_limit_seconds'), 'wall_elapsed_s': status.get('wall_elapsed')
            or (time.time() - dt.datetime.fromisoformat(started).timestamp() if started and status.get('status') == 'running' else None)}


def log_hint(job_id):
    """Last error-looking line of the strict job's log."""
    path = REPO / 'experiments' / 'validation' / f'experiment-strict-{job_id}.log'
    try:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            lines = handle.read().decode(errors='replace').splitlines()
    except OSError:
        return None
    hits = [line for line in lines if re.search(r'Error|Traceback|error:|Exception|Killed|CANCELLED|DUE TO', line)]
    return hits[-1][:300] if hits else None


def spend_by_run():
    """[(run id, pool, cost, updated date)] from the cost ledger."""
    rows = []
    for path in LEDGER.glob('*/*.json'):
        entry = _read_json(path)
        if entry is not None:
            rows.append((entry['run_id'], entry['pool'], float(entry['cost_usd']), entry.get('updated_at', '')[:10]))
    return rows


def pool_day(pool, updated):
    match = re.search(r'(\d{8})$', pool)
    return f'{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}' if match else updated


class Monitor:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.state_dir / 'usage.json'
        self.store = _read_json(self.store_path) or {'jobs': {}, 'run_keys': {}, 'backfilled_through': None}
        self.snapshots_path = self.state_dir / 'spend.jsonl'
        self.snapshot = {'updated_at': None, 'error': 'starting'}
        self.storage, self.storage_at = [], 0.0
        self.lock = threading.Lock()

    def _save(self):
        tmp = self.store_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.store))
        tmp.replace(self.store_path)

    def record(self, jobs, live=None):
        for job_id, job in jobs.items():
            final = job['State'].split()[0] in FINAL
            if _stamp(job['Start']) is None:   # cancelled before it started: its runs stay prepared and are requeued
                continue
            if final or (live is not None and job['State'] == 'RUNNING'):
                entry = {'runs': job['runs'], 'state': job['State'], 'exit': job['ExitCode'], 'partition': job['Partition'],
                         'node': job['NodeList'], 'start': job['Start'], 'end': job['End'], 'timelimit': job['Timelimit'],
                         'usage': usage(job, live.get(job_id) if live else None)}
                if final:
                    self.store['jobs'][job_id] = entry
                job['record'] = entry

    def backfill(self):
        day = dt.date.fromisoformat(self.store['backfilled_through']) if self.store['backfilled_through'] else HISTORY_START
        today = dt.date.today()
        while day < today - dt.timedelta(days=1):
            self.record(sacct(day.isoformat(), (day + dt.timedelta(days=1)).isoformat()))
            day += dt.timedelta(days=1)
            self.store['backfilled_through'] = day.isoformat()
            self._save()

    def key_of(self, run_id):
        if run_id not in self.store['run_keys']:
            directory = run_dir(run_id)
            config = _read_json(directory / 'config-private.json') if directory else None
            if config is None:
                return 'unattributed (run dir not found)'
            self.store['run_keys'][run_id] = (config.get('inference') or {}).get('api_key_env') or 'none'
        return self.store['run_keys'][run_id]

    def spend(self):
        rows = spend_by_run()
        # A pool belongs to one key by construction; a run whose directory is gone takes its pool's key when every
        # attributed run of that pool agrees, else it stays visibly unattributed.
        pool_keys = {}
        for run_id, pool, _, _ in rows:
            key = self.key_of(run_id)
            if not key.startswith('unattributed'):
                pool_keys.setdefault(pool, set()).add(key)
        lifetime, estimated = {}, {}
        for run_id, pool, cost, updated in rows:
            key = self.key_of(run_id)
            if key.startswith('unattributed') and len(pool_keys.get(pool, ())) == 1:
                key = next(iter(pool_keys[pool]))
            lifetime[key] = lifetime.get(key, 0.0) + cost
            day = pool_day(pool, updated)
            estimated.setdefault(day, {})[key] = estimated.setdefault(day, {}).get(key, 0.0) + cost
        totals = {key: round(value, 2) for key, value in lifetime.items()}
        snapshots = [json.loads(line) for line in self.snapshots_path.read_text().splitlines()] if self.snapshots_path.exists() else []
        now = dt.datetime.now().astimezone()
        if not snapshots or snapshots[-1]['keys'] != totals or snapshots[-1]['t'][:10] != now.date().isoformat():
            with self.snapshots_path.open('a') as handle:
                handle.write(json.dumps({'t': now.isoformat(timespec='seconds'), 'keys': totals}) + '\n')
            snapshots.append({'t': now.isoformat(timespec='seconds'), 'keys': totals})
        # Exact daily spend = change in snapshot totals across the day; days before the first snapshot use the
        # pool-day attribution (daily pools are named by their ET day; undated pools by the ledger's last update).
        first_day, daily = snapshots[0]['t'][:10], []
        for offset in range(13, -1, -1):
            day = (now.date() - dt.timedelta(days=offset)).isoformat()
            if day > first_day:
                before = [s['keys'] for s in snapshots if s['t'][:10] < day][-1]
                after = [s['keys'] for s in snapshots if s['t'][:10] <= day][-1]
                daily.append({'day': day, 'source': 'exact', 'keys': {k: round(after.get(k, 0) - before.get(k, 0), 2) for k in after}})
            else:
                daily.append({'day': day, 'source': 'estimated' if day < first_day else 'partial',
                              'keys': {k: round(v, 2) for k, v in estimated.get(day, {}).items()}})
        return {'lifetime': totals, 'caps': KEY_CAPS, 'daily': daily}

    def probe_storage(self):
        rows = []
        home = os.statvfs(Path.home())
        rows.append({'name': 'home', 'path': str(Path.home()), 'free_gb': home.f_bavail * home.f_frsize / 1e9,
                     'total_gb': home.f_blocks * home.f_frsize / 1e9, 'writable': True})
        for name, root in (('lab', ROOTS[0]), ('netscratch', ROOTS[1])):
            probe, ok = root / f'.monitor-probe-{os.getpid()}', 'ok'
            try:   # group quotas are invisible here: 20 files plus 1 MiB must write, or the store is full
                probe.mkdir(exist_ok=True)
                for index in range(20):
                    (probe / str(index)).write_bytes(b'' if index else os.urandom(1 << 20))
            except OSError as exc:
                ok = 'FULL (quota)' if exc.errno == errno.EDQUOT else f'error: {exc.strerror}'
            for child in probe.glob('*'):   # only this probe's own files
                child.unlink()
            if probe.exists():
                probe.rmdir()
            runs = root / 'runs'
            rows.append({'name': name, 'path': str(root), 'writable': ok, 'runs': len(os.listdir(runs)) if runs.is_dir() else 0})
        return rows

    def refresh(self):
        two_days_ago = (dt.date.today() - dt.timedelta(days=2)).isoformat()
        jobs = sacct(two_days_ago)
        queued = run_command(['squeue', '-h', '-u', os.environ.get('USER', ''), '-n', JOB_NAME, '-o', '%i|%T|%P|%r|%M|%l'])
        live = {job_id: sstat(job_id) for job_id, job in jobs.items() if job['State'] == 'RUNNING'}
        self.record(jobs, live)
        self._save()
        running = [{'job': job_id, 'node': job['NodeList'], 'partition': job['Partition'], 'elapsed': job['Elapsed'],
                    'timelimit': job['Timelimit'], 'usage': job['record']['usage'], 'runs': [run_status(r) for r in job['runs']]}
                   for job_id, job in jobs.items() if job['State'] == 'RUNNING']
        pending = [dict(zip(('job', 'state', 'partition', 'reason', 'elapsed', 'timelimit'), line.split('|')))
                   for line in queued.splitlines() if '|PENDING|' in line]
        week_ago = (dt.datetime.now() - dt.timedelta(days=7)).isoformat()
        failures = []
        for job_id, entry in sorted(self.store['jobs'].items(), key=lambda item: item[1]['end'] or '', reverse=True):
            if (entry['end'] or '') < week_ago:
                break
            runs = [run_status(r, job_ended=True) for r in entry['runs']]
            if entry['state'] != 'COMPLETED' or any(r['status'] != 'completed' for r in runs):
                failures.append({'job': job_id, 'state': entry['state'], 'exit': entry['exit'], 'end': entry['end'],
                                 'runs': runs, 'log': log_hint(job_id)})
        if time.time() - self.storage_at > 600:
            self.storage, self.storage_at = self.probe_storage(), time.time()
        finished = [entry for entry in self.store['jobs'].values() if entry['usage']['elapsed_s'] >= 1800 and entry['usage']['cpu_s']]
        with self.lock:
            self.snapshot = {'updated_at': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'error': None,
                             'running': running, 'pending': pending, 'failures': failures[:60], 'spend': self.spend(),
                             'storage': self.storage, 'history': self.history(), 'advice': advice(finished),
                             'backfilled_through': self.store['backfilled_through']}

    def history(self, limit=150):
        rows = sorted(self.store['jobs'].items(), key=lambda item: item[1]['start'] or '', reverse=True)[:limit]
        return [{'job': job_id, **{k: entry[k] for k in ('runs', 'state', 'partition', 'start', 'end')}, 'usage': entry['usage']}
                for job_id, entry in rows]

    def loop(self, interval):
        while True:
            try:
                if (self.store['backfilled_through'] or '') < (dt.date.today() - dt.timedelta(days=1)).isoformat():
                    self.backfill()
                self.refresh()
            except Exception as exc:   # keep serving the last good data, but show the failure on the page
                with self.lock:
                    self.snapshot = dict(self.snapshot, error=f'{type(exc).__name__}: {exc}')
            time.sleep(interval)


def _percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def advice(entries):
    """Request-size suggestion from finished strict jobs of at least 30 min: 1.5x the p95 of what was used."""
    if not entries:
        return None
    per_run = [len(e['runs']) or 1 for e in entries]
    cores = [e['usage']['cores_used'] / n for e, n in zip(entries, per_run)]
    memory = [(e['usage']['peak_mem_bound'] or 0) / n for e, n in zip(entries, per_run)]
    return {'jobs': len(entries), 'median_cpu_efficiency': _percentile([e['usage']['cpu_efficiency'] for e in entries], .5),
            'max_step_cpus_per_run': max(e['usage'].get('peak_step_cpus', 0) / n for e, n in zip(entries, per_run)),
            'p95_cores_per_run': _percentile(cores, .95), 'p95_peak_mem_per_run': _percentile(memory, .95),
            'median_alloc_cpus': _percentile([e['usage']['alloc_cpus'] for e in entries], .5),
            'median_req_mem': _percentile([e['usage']['req_mem'] or 0 for e in entries], .5)}


PAGE = (Path(__file__).with_name('monitor_view.html')).read_text


def serve(monitor, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/state':
                with monitor.lock:
                    body, kind = json.dumps(monitor.snapshot).encode(), 'application/json'
            elif self.path in ('/', '/index.html'):
                body, kind = PAGE().encode(), 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--state-dir', type=Path, default=REPO / 'experiments' / 'monitor')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    monitor = Monitor(args.state_dir)
    threading.Thread(target=monitor.loop, args=(args.interval,), daemon=True).start()
    serve(monitor, args.port)


if __name__ == '__main__':
    main()
