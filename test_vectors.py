"""
Central definitions for all test vectors used in this repo.

Each vector is a list of (read_len, gen_len) pairs:
- benchmark_test_suite: used by `src/test.py` / `main.ipynb` benchmark runs
- plot_test_suite: used by `src/utils/visuals.py` when plotting JSON results
- profiling_test_suite: used by `profiling_main.ipynb` + profiling helpers
"""

# Prefill sweep: vary prompt/input length (read_len) from 64 to 16384 with fixed gen_len=16.
prefill_test_suite = [
    (64, 16),
    (128, 16),
    (256, 16),
    (512, 16),
    (1024, 16),
    (2048, 16),
    (4096, 16),
    (8192, 16),
    (16384, 16),
]

# Decode sweep: keep a long prompt (read_len=512) while varying gen_len from 16 to 256.
decode_test_suite = [
    (512, 16),
    (512, 32),
    (512, 64),
    (512, 128),
    (512, 256),
]

# Used for the main benchmark loop (see `src/test.py`) — combined set of both sweeps.
benchmark_test_suite = [
    *prefill_test_suite,
    # Add extra decode points beyond the shared (512, 16) prefill point
    (512, 32),
    (512, 64),
    (512, 128),
    (512, 256),
]

# Used for plotting comparisons (see `src/utils/visuals.py`)
plot_test_suite = [*benchmark_test_suite]

# Used for profiling traces (see `src/utils/metrics.py` + `profiling_main.ipynb`)
profiling_test_suite = [
    # Keep profiling very lightweight: only a few small points.
    (64, 16),
    (128, 16),
    (256, 32),
]

# Backwards-compatible alias (some code/tests refer to `test_suite`)
test_suite = benchmark_test_suite

