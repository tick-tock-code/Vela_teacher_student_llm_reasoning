from __future__ import annotations

import os


DEFAULT_MAX_PARALLEL_WORKERS = 7


def resolve_max_parallel_workers(requested: int | None = None) -> int:
    if requested is not None:
        if requested < 1:
            raise RuntimeError("max_parallel_workers must be >= 1.")
        return int(requested)
    cpu_count = os.cpu_count() or 1
    return max(1, min(DEFAULT_MAX_PARALLEL_WORKERS, cpu_count - 1))


def bounded_worker_count(*, max_parallel_workers: int, task_count: int) -> int:
    if task_count <= 1 or max_parallel_workers <= 1:
        return 1
    return max(1, min(max_parallel_workers, task_count))
