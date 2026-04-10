from __future__ import annotations


def not_implemented_placeholder(function_name: str, details: str) -> None:
    raise NotImplementedError(
        f"{function_name} must be explicitly replaced before this pipeline can run. {details}"
    )
