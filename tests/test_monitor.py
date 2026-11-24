from sim2sim.monitor import FIELDS, nbytes, parse_sacct, pool_day, seconds, usage

# Real sacct rows of strict job 63157947 (two runs), trimmed to three steps, in FIELDS order.
ROWS = [
    '63157947|sim2sim-strict|FAILED|1:0|09:59:05|22|48G||28:36.629|2026-11-21T08:00:00|2026-11-21T17:59:05|sapphire|holy8a1|14:00:00|'
    'sbatch --parsable --partition=sapphire --export=ALL scripts/experiment-strict.slurm run-a run-b',
    '63157947.batch|batch|FAILED|1:0|09:59:05|22||733968K|07:41.048|2026-11-21T08:00:00|2026-11-21T17:59:05||holy8a1||',
    '63157947.extern|extern|COMPLETED|0:0|09:59:05|22|||00:00:00|2026-11-21T08:00:00|2026-11-21T17:59:05||holy8a1||',
    '63157947.2|sim2sim-agent-926bda7885784c5d9297594131ace095|COMPLETED|0:0|09:58:25|4||272336K|11:42.876|2026-11-21T08:00:30|2026-11-21T17:58:55||holy8a1||',
    '63157947.8|sim2sim-eval-ece1501f9f8b49fa9b3c2ac7e6c904ee|COMPLETED|0:0|00:03:20|2||669656K|03:19.133|2026-11-21T09:00:00|2026-11-21T09:03:20||holy8a1||',
    '63157948|wrap|COMPLETED|0:0|00:01:00|1|1G||00:01.000|2026-11-21T08:00:00|2026-11-21T08:01:00|test|holy8a2|00:10:00|sbatch --wrap true',
]


def test_sacct_parsing_and_usage_of_a_strict_job():
    assert len(ROWS[0].split('|')) == len(FIELDS)
    assert seconds('1-02:03:04') == 93784 and seconds('28:36.629') == 1716.629 and seconds('') == 0
    assert nbytes('733968K') == 733968 * 1024 and nbytes('48G') == 48 * 2 ** 30 and nbytes('2Gc', 4) == 8 * 2 ** 30 and nbytes('') is None
    assert pool_day('gemini-20261124', '2026-11-25') == '2026-11-24' and pool_day('gemini-bycost', '2026-11-25') == '2026-11-25'
    jobs = parse_sacct('\n'.join(ROWS))
    assert list(jobs) == ['63157947']   # only strict jobs
    job = jobs['63157947']
    assert job['runs'] == ['run-a', 'run-b'] and [s['JobID'] for s in job['steps']] == ['63157947.batch', '63157947.2', '63157947.8']
    u = usage(job)
    assert u['alloc_cpus'] == 22 and abs(u['cpu_s'] - (461.048 + 702.876 + 199.133)) < 1e-6
    assert abs(u['cpu_efficiency'] - u['cpu_s'] / (seconds('09:59:05') * 22)) < 1e-12
    # Concurrent peak: controller + agent + evaluator overlap at 09:00; the batch step's nominal 22 CPUs are not a reservation.
    assert u['peak_mem_bound'] == (733968 + 272336 + 669656) * 1024 and u['peak_step_cpus'] == 6
    assert set(u['roles']) == {'controller', 'agent', 'eval'} and u['roles']['eval']['max_rss'] == 669656 * 1024
