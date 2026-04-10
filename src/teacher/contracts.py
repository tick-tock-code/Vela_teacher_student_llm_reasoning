from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReasoningTargetContract:
    target_id: str
    train_path: Path
    column_name: str
    source_id_column: str = "founder_uuid"
    test_path: Path | None = None
    scale_min: float = 0.0
    scale_max: float = 1.0
    prediction_mode: str = "regression"


def validate_reasoning_target_contract(contract: ReasoningTargetContract) -> None:
    if not contract.target_id.strip():
        raise RuntimeError("Reasoning target contract requires a non-empty target_id.")
    if not contract.column_name.strip():
        raise RuntimeError(f"Reasoning target '{contract.target_id}' requires a column_name.")
    if contract.prediction_mode != "regression":
        raise RuntimeError(
            f"Reasoning target '{contract.target_id}' uses unsupported prediction_mode "
            f"'{contract.prediction_mode}'. v1 only supports 'regression'."
        )
    if contract.scale_min >= contract.scale_max:
        raise RuntimeError(
            f"Reasoning target '{contract.target_id}' must have scale_min < scale_max."
        )
