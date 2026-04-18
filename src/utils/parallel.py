from __future__ import annotations

import os


# Default to the benchmark-selected stable throughput settings unless explicitly overridden.
DEFAULT_MAX_PARALLEL_WORKERS = 2
DEFAULT_THREAD_COUNT = 1
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def preferred_thread_count(requested: int | None = None) -> int:
    if requested is not None:
        if requested < 1:
            raise RuntimeError("thread_count must be >= 1.")
        return int(requested)
    return DEFAULT_THREAD_COUNT


def apply_global_thread_env(thread_count: int | None = None) -> int:
    resolved = preferred_thread_count(thread_count)
    for env_var in THREAD_ENV_VARS:
        os.environ[env_var] = str(resolved)
    return resolved


def resolve_max_parallel_workers(requested: int | None = None) -> int:
    if requested is not None:
        if requested < 1:
            raise RuntimeError("max_parallel_workers must be >= 1.")
        return int(requested)
    if DEFAULT_MAX_PARALLEL_WORKERS is not None:
        return int(DEFAULT_MAX_PARALLEL_WORKERS)
    return preferred_thread_count()


def bounded_worker_count(*, max_parallel_workers: int, task_count: int) -> int:
    if task_count <= 1 or max_parallel_workers <= 1:
        return 1
    return max(1, min(max_parallel_workers, task_count))
