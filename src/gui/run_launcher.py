from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from dataclasses import dataclass, replace
from tkinter import scrolledtext, ttk
from typing import Callable

from src.data.targets import load_target_family
from src.pipeline.config import ExperimentConfig, load_experiment_config
from src.pipeline.distillation import run_pipeline
from src.pipeline.mlp_calibration import load_latest_mlp_calibration
from src.pipeline.rf_calibration import load_latest_rf_calibration
from src.pipeline.saved_model_configs import (
    list_saved_bundle_dirs,
    load_bundle_manifest,
    resolve_saved_bundle_path,
)
from src.pipeline.xgb_calibration import load_latest_xgb_calibration
from src.pipeline.run_options import (
    ABLATION_V25_19SET_LINEAR_FEATURE_SET_IDS,
    DEFAULT_CONFIG_PATH,
    RunOverrides,
)
from src.utils.model_ids import (
    XGB_CLASSIFIER_MODEL_KIND,
    XGB_FAMILY_ID,
    XGB_FAMILY_UI_LABEL,
    XGB_REGRESSOR_MODEL_KIND,
)
from src.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class LauncherSelections:
    config_path: str
    run_mode: str
    target_family: str
    heldout_evaluation: bool
    active_feature_banks: list[str] | None
    force_rebuild_intermediary_features: bool
    reasoning_models: list[str] | None
    embedding_model_name: str | None
    reproduction_outer_splits: int | None = None
    reproduction_inner_splits: int | None = None
    distillation_splits: int | None = None
    cv_random_state: int | None = None
    threshold_start: float | None = None
    threshold_stop: float | None = None
    threshold_step: float | None = None
    repeat_cv_with_new_seeds: bool = False
    cv_seed_repeat_count: int | None = None
    distillation_nested_sweep: bool | None = None
    save_reasoning_predictions: bool | None = None
    candidate_feature_sets: list[str] | None = None
    model_families: list[str] | None = None
    output_modes: list[str] | None = None
    model_family_output_modes: dict[str, list[str]] | None = None
    save_model_configs_after_training: bool | None = None
    saved_config_bundle_path: str | None = None
    saved_eval_mode: str | None = None
    saved_eval_combo_ids: list[str] | None = None
    saved_eval_combo_refs: list[str] | None = None
    saved_eval_success_branch_ids: list[str] | None = None
    saved_eval_per_target_best_r2: bool | None = None
    hq_exit_override_mode: str | None = None
    xgb_calibration_estimators: list[int] | None = None
    use_latest_xgb_calibration: bool | None = None
    rf_calibration_min_samples_leaf: list[int] | None = None
    rf_calibration_max_depth: list[int | None] | None = None
    rf_calibration_max_features: list[str | float] | None = None
    use_latest_rf_calibration: bool | None = None
    mlp_calibration_hidden_layer_sizes: list[list[int]] | None = None
    mlp_calibration_alpha: list[float] | None = None
    use_latest_mlp_calibration: bool | None = None
    mlp_hidden_layer_sizes: list[int] | None = None
    mlp_alpha: float | None = None
    xgb_model_param_overrides_by_model_id: dict[str, dict[str, object]] | None = None
    max_parallel_workers: int | None = None


def selections_to_overrides(selections: LauncherSelections) -> RunOverrides:
    return RunOverrides(
        config_path=selections.config_path,
        run_mode=selections.run_mode,
        target_family=selections.target_family,
        heldout_evaluation=selections.heldout_evaluation,
        active_feature_banks=selections.active_feature_banks,
        force_rebuild_intermediary_features=selections.force_rebuild_intermediary_features,
        reasoning_models=selections.reasoning_models,
        embedding_model_name=selections.embedding_model_name,
        repeat_cv_with_new_seeds=selections.repeat_cv_with_new_seeds,
        cv_seed_repeat_count=selections.cv_seed_repeat_count,
        distillation_nested_sweep=selections.distillation_nested_sweep,
        save_reasoning_predictions=selections.save_reasoning_predictions,
        candidate_feature_sets=selections.candidate_feature_sets,
        model_families=selections.model_families,
        output_modes=selections.output_modes,
        model_family_output_modes=selections.model_family_output_modes,
        save_model_configs_after_training=selections.save_model_configs_after_training,
        saved_config_bundle_path=selections.saved_config_bundle_path,
        saved_eval_mode=selections.saved_eval_mode,
        saved_eval_combo_ids=selections.saved_eval_combo_ids,
        saved_eval_combo_refs=selections.saved_eval_combo_refs,
        saved_eval_success_branch_ids=selections.saved_eval_success_branch_ids,
        saved_eval_per_target_best_r2=selections.saved_eval_per_target_best_r2,
        hq_exit_override_mode=selections.hq_exit_override_mode,
        xgb_calibration_estimators=selections.xgb_calibration_estimators,
        use_latest_xgb_calibration=selections.use_latest_xgb_calibration,
        rf_calibration_min_samples_leaf=selections.rf_calibration_min_samples_leaf,
        rf_calibration_max_depth=selections.rf_calibration_max_depth,
        rf_calibration_max_features=selections.rf_calibration_max_features,
        use_latest_rf_calibration=selections.use_latest_rf_calibration,
        mlp_calibration_hidden_layer_sizes=selections.mlp_calibration_hidden_layer_sizes,
        mlp_calibration_alpha=selections.mlp_calibration_alpha,
        use_latest_mlp_calibration=selections.use_latest_mlp_calibration,
        mlp_hidden_layer_sizes=selections.mlp_hidden_layer_sizes,
        mlp_alpha=selections.mlp_alpha,
        xgb_model_param_overrides_by_model_id=selections.xgb_model_param_overrides_by_model_id,
        max_parallel_workers=selections.max_parallel_workers,
    )


def apply_launcher_config_overrides(config: ExperimentConfig, selections: LauncherSelections) -> ExperimentConfig:
    _ = selections
    # CV/threshold override controls are intentionally dormant in the GUI.
    # Launcher runs always use the config-defined CV strategy.
    return config


class RunLauncher(ttk.Frame):
    XGB_DEPTH_TEST_DEPTHS: tuple[int, ...] = (3, 5)
    XGB_DEPTH_TEST_N_ESTIMATORS: int = 320
    XGB_DEPTH_TEST_REPEAT_COUNT: int = 4
    XGB_DEPTH_TEST_MAX_PARALLEL_WORKERS: int = 2
    XGB_DEPTH_TEST_FEATURE_SET_IDS: tuple[str, ...] = (
        "hq_plus_sentence_prose",
        "hq_plus_sentence_bundle",
        "lambda_policies_plus_sentence_prose",
        "lambda_policies_plus_sentence_bundle",
    )
    ABLATION_V25_19SET_LINEAR_FEATURE_SET_IDS: tuple[str, ...] = ABLATION_V25_19SET_LINEAR_FEATURE_SET_IDS
    COMBINATION_FEATURE_SET_IDS: tuple[str, ...] = (
        "hq_plus_sentence_bundle",
        "llm_engineering_plus_sentence_bundle",
        "lambda_policies_plus_sentence_bundle",
        "hq_plus_llm_engineering_plus_sentence_bundle",
        "hq_plus_lambda_policies_plus_sentence_bundle",
        "llm_engineering_plus_lambda_policies_plus_sentence_bundle",
        "hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle",
    )
    COMBINATION_SUCCESS_BASE_COMBO_IDS: tuple[str, ...] = (
        "hq_baseline",
        "llm_engineering",
        "lambda_policies",
        "hq_plus_llm_engineering",
        "hq_plus_lambda_policies",
        "llm_engineering_plus_lambda_policies",
        "hq_plus_llm_engineering_plus_lambda_policies",
    )
    COMBINATION_SUCCESS_OVERRIDE_BRANCHES: tuple[str, ...] = ("with_override", "without_override")
    COMBINATION_SCREENING_DEFAULT_REPEAT_COUNT: int = 4

    def __init__(self, master: tk.Misc | None = None, *, initial_config_path: str = DEFAULT_CONFIG_PATH):
        super().__init__(master, padding=10)
        self.master = master
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._loaded_config: ExperimentConfig | None = None
        self._defaults: dict[str, object] = {}
        self._run_started_monotonic: float | None = None
        self._last_heartbeat_monotonic: float = 0.0
        self._heartbeat_interval_seconds: float = 20.0

        self.config_path_var = tk.StringVar(value=initial_config_path)
        self.defaults_summary_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.output_path_var = tk.StringVar(value="")
        self.target_preview_var = tk.StringVar(value="")
        self.mt_target_preview_var = tk.StringVar(value="")

        self.run_mode_var = tk.StringVar(value="reproduction_mode")
        self.target_family_var = tk.StringVar(value="v25_policies")
        self.heldout_var = tk.BooleanVar(value=False)
        self.force_rebuild_var = tk.BooleanVar(value=False)
        self.embedding_model_var = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")
        self.repeat_cv_var = tk.BooleanVar(value=False)
        self.repeat_count_var = tk.StringVar(value="1")
        self.setup_nested_cv_var = tk.BooleanVar(value=False)
        self.save_predictions_var = tk.BooleanVar(value=True)

        self.mt_target_family_var = tk.StringVar(value="v25_policies")
        self.mt_repeat_cv_var = tk.BooleanVar(value=False)
        self.mt_repeat_count_var = tk.StringVar(value="1")
        self.mt_force_rebuild_var = tk.BooleanVar(value=False)
        self.mt_save_model_configs_var = tk.BooleanVar(value=False)
        self.mt_embedding_model_var = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")
        self.mt_use_latest_xgb_calibration_var = tk.BooleanVar(value=False)
        self.mt_use_latest_rf_calibration_var = tk.BooleanVar(value=False)
        self.mt_use_latest_mlp_calibration_var = tk.BooleanVar(value=False)
        self.mt_mlp_hidden_layers_var = tk.StringVar(value="32")
        self.mt_mlp_alpha_var = tk.StringVar(value="0.1")
        self.mt_calibration_sweep_var = tk.StringVar(value="")
        self.mt_rf_calibration_sweep_var = tk.StringVar(value="")
        self.mt_mlp_calibration_sweep_var = tk.StringVar(value="")
        self.mt_latest_calibration_var = tk.StringVar(value="No calibration loaded")
        self.mt_latest_rf_calibration_var = tk.StringVar(value="No RF calibration loaded")
        self.mt_latest_mlp_calibration_var = tk.StringVar(value="No MLP calibration loaded")
        self.combo_target_family_var = tk.StringVar(value="v25_policies")
        self.combo_repeat_cv_var = tk.BooleanVar(value=True)
        self.combo_repeat_count_var = tk.StringVar(value=str(self.COMBINATION_SCREENING_DEFAULT_REPEAT_COUNT))
        self.combo_save_model_configs_var = tk.BooleanVar(value=True)
        self.combo_force_rebuild_var = tk.BooleanVar(value=False)
        self.combo_bundle_root_var = tk.StringVar(value="data/saved_model_configs")
        self.combo_bundle_selection_var = tk.StringVar(value="")
        self.combo_selected_combo_var = tk.StringVar(value="")
        self.combo_bundle_status_var = tk.StringVar(value="No bundle selected")
        self._combo_combo_choices: list[tuple[str, str]] = []
        self.combo_success_branch_count_var = tk.StringVar(value="No success branches selected")
        self._combo_success_branch_ids_by_index: list[str] = []
        self.saved_bundle_root_var = tk.StringVar(value="data/saved_model_configs")
        self.saved_bundle_selection_var = tk.StringVar(value="")
        self.saved_eval_mode_var = tk.StringVar(value="reasoning_test_metrics")
        self.saved_hq_override_mode_var = tk.StringVar(value="with_override")
        self.saved_eval_best_r2_var = tk.BooleanVar(value=False)
        self.saved_combo_count_var = tk.StringVar(value="No bundle selected")
        self._saved_combo_ids_by_index: list[str] = []
        self.saved_model_pick_bundle_vars: dict[str, tk.StringVar] = {
            "linear_l2": tk.StringVar(value=""),
            XGB_FAMILY_ID: tk.StringVar(value=""),
            "mlp": tk.StringVar(value=""),
        }
        self.saved_model_pick_combo_vars: dict[str, tk.StringVar] = {
            "linear_l2": tk.StringVar(value=""),
            XGB_FAMILY_ID: tk.StringVar(value=""),
            "mlp": tk.StringVar(value=""),
        }
        self._saved_model_pick_combo_choices: dict[str, list[tuple[str, str]]] = {
            "linear_l2": [],
            XGB_FAMILY_ID: [],
            "mlp": [],
        }

        self.feature_bank_vars: dict[str, tk.BooleanVar] = {}
        self.setup_sentence_bundle_var: tk.BooleanVar | None = None
        self.setup_model_vars: dict[str, tk.BooleanVar] = {
            "linear_l2": tk.BooleanVar(value=True),
            "linear_svm": tk.BooleanVar(value=False),
            XGB_FAMILY_ID: tk.BooleanVar(value=False),
        }
        self.mt_feature_set_vars: dict[str, tk.BooleanVar] = {}
        self.mt_model_family_vars: dict[str, tk.BooleanVar] = {
            "linear_l2": tk.BooleanVar(value=True),
            "linear_svm": tk.BooleanVar(value=False),
            XGB_FAMILY_ID: tk.BooleanVar(value=False),
            "mlp": tk.BooleanVar(value=False),
            "elasticnet": tk.BooleanVar(value=False),
            "randomforest": tk.BooleanVar(value=False),
        }
        self.mt_model_family_output_vars: dict[str, dict[str, tk.BooleanVar]] = {
            "linear_l2": {
                "single_target": tk.BooleanVar(value=True),
                "multi_output": tk.BooleanVar(value=False),
            },
            "linear_svm": {
                "single_target": tk.BooleanVar(value=True),
                "multi_output": tk.BooleanVar(value=False),
            },
            XGB_FAMILY_ID: {
                "single_target": tk.BooleanVar(value=True),
                "multi_output": tk.BooleanVar(value=False),
            },
            "mlp": {
                "single_target": tk.BooleanVar(value=False),
                "multi_output": tk.BooleanVar(value=True),
            },
            "elasticnet": {
                "single_target": tk.BooleanVar(value=True),
                "multi_output": tk.BooleanVar(value=False),
            },
            "randomforest": {
                "single_target": tk.BooleanVar(value=True),
                "multi_output": tk.BooleanVar(value=False),
            },
        }

        self._build_ui()
        self._load_config_preview()
        self.after(100, self._poll_queue)

    @staticmethod
    def _attach_mousewheel(canvas: tk.Canvas) -> None:
        def _on_mousewheel(event: tk.Event) -> None:  # type: ignore[type-arg]
            if getattr(event, "delta", 0) == 0:
                return
            canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))

    def _make_scrollable_tab(self, parent: ttk.Frame) -> ttk.Frame:
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas, padding=10)
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_body_configure(_event: tk.Event) -> None:  # type: ignore[type-arg]
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:  # type: ignore[type-arg]
            canvas.itemconfigure(window_id, width=event.width)

        body.bind("<Configure>", _on_body_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._attach_mousewheel(canvas)
        return body

    def _build_ui(self) -> None:
        self.grid(sticky="nsew")
        if self.master is not None:
            self.master.columnconfigure(0, weight=1)
            self.master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        head = ttk.Frame(self)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        head.columnconfigure(1, weight=1)
        ttk.Label(head, text="Config path").grid(row=0, column=0, sticky="w")
        ttk.Entry(head, textvariable=self.config_path_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(head, text="Reload Config", command=self._load_config_preview).grid(row=0, column=2, sticky="e")
        ttk.Label(head, textvariable=self.defaults_summary_var, wraplength=1180, justify="left").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)

        setup_tab_container = ttk.Frame(notebook)
        testing_tab_container = ttk.Frame(notebook)
        combination_tab_container = ttk.Frame(notebook)
        saved_eval_tab_container = ttk.Frame(notebook)
        notebook.add(setup_tab_container, text="Run Setup")
        notebook.add(testing_tab_container, text="Model Testing")
        notebook.add(combination_tab_container, text="Combination Testing")
        notebook.add(saved_eval_tab_container, text="Saved Config Eval")
        self.setup_tab = self._make_scrollable_tab(setup_tab_container)
        self.testing_tab = self._make_scrollable_tab(testing_tab_container)
        self.combination_tab = self._make_scrollable_tab(combination_tab_container)
        self.saved_eval_tab = self._make_scrollable_tab(saved_eval_tab_container)

        self._build_setup_tab()
        self._build_testing_tab()
        self._build_combination_tab()
        self._build_saved_eval_tab()

        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, sticky="ew", pady=(8, 8))
        actions.columnconfigure(11, weight=1)
        ttk.Button(actions, text="Reset Defaults", command=self._reset_defaults).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Run Setup Pipeline", command=self.start_run_setup).grid(row=0, column=1, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run Model Testing", command=self.start_model_testing).grid(row=0, column=2, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run Combination Screening", command=self.start_combination_screening).grid(row=0, column=3, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run Combination Success CV", command=self.start_combination_success_screening).grid(row=0, column=4, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run Combination Success Test", command=self.start_combination_success_test_eval).grid(row=0, column=5, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run Saved Config Eval", command=self.start_saved_config_eval).grid(row=0, column=6, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run XGB Calibration", command=self.start_xgb_calibration).grid(row=0, column=7, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run RF Calibration", command=self.start_rf_calibration).grid(row=0, column=8, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run MLP Calibration", command=self.start_mlp_calibration).grid(row=0, column=9, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run XGB Depth Test", command=self.start_xgb_depth_test).grid(row=0, column=10, padx=(8, 0), sticky="w")
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=11, sticky="w")

        out = ttk.LabelFrame(self, text="Output")
        out.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(out, textvariable=self.output_path_var, wraplength=1180, justify="left").grid(row=0, column=0, sticky="w")

        log_frame = ttk.LabelFrame(self, text="Status Log")
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=16, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def _build_setup_tab(self) -> None:
        self.setup_tab.columnconfigure(1, weight=1)
        ttk.Label(self.setup_tab, text="Mode").grid(row=0, column=0, sticky="w")
        self.setup_mode_combo = ttk.Combobox(
            self.setup_tab,
            textvariable=self.run_mode_var,
            values=("reproduction_mode", "reasoning_distillation_mode"),
            state="readonly",
            width=30,
        )
        self.setup_mode_combo.grid(row=0, column=1, sticky="w")
        self.setup_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_setup_mode())

        ttk.Label(self.setup_tab, text="Target family").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.setup_target_combo = ttk.Combobox(
            self.setup_tab,
            textvariable=self.target_family_var,
            state="readonly",
            width=30,
        )
        self.setup_target_combo.grid(row=1, column=1, sticky="w", pady=(6, 0))
        self.setup_target_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_target_preview())
        ttk.Label(self.setup_tab, textvariable=self.target_preview_var, wraplength=760, justify="left").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        flags = ttk.LabelFrame(self.setup_tab, text="Flags")
        flags.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 8))
        flags.columnconfigure(0, weight=1)
        ttk.Checkbutton(flags, text="Run held-out evaluation", variable=self.heldout_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(flags, text="Force rebuild intermediary feature banks", variable=self.force_rebuild_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Checkbutton(
            flags,
            text="Repeat stratified CV with new random seeds",
            variable=self.repeat_cv_var,
            command=self._on_setup_repeat_toggle,
        ).grid(row=2, column=0, sticky="w")
        ttk.Label(flags, text="Repeat count").grid(row=2, column=1, sticky="e")
        self.setup_repeat_entry = ttk.Entry(flags, textvariable=self.repeat_count_var, width=8)
        self.setup_repeat_entry.grid(row=2, column=2, sticky="w")
        ttk.Checkbutton(
            flags,
            text="Use nested hyperparameter CV (untick to run fixed L2 C=5)",
            variable=self.setup_nested_cv_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(flags, text="Save reasoning predictions to tmp", variable=self.save_predictions_var).grid(
            row=4, column=0, sticky="w"
        )
        ttk.Label(flags, text="Embedding model").grid(row=5, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(flags, textvariable=self.embedding_model_var).grid(row=6, column=0, columnspan=3, sticky="ew")

        features_frame = ttk.LabelFrame(self.setup_tab, text="Feature Banks")
        features_frame.grid(row=4, column=0, sticky="nsew", padx=(0, 8))
        features_frame.columnconfigure(0, weight=1)
        self.setup_features_frame = features_frame

        models_frame = ttk.LabelFrame(self.setup_tab, text="Models (never greyed)")
        models_frame.grid(row=4, column=1, sticky="nsew")
        ttk.Checkbutton(
            models_frame,
            text="Linear L2 (Ridge/LogReg)",
            variable=self.setup_model_vars["linear_l2"],
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            models_frame,
            text="Linear SVM/SVR",
            variable=self.setup_model_vars["linear_svm"],
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            models_frame,
            text=XGB_FAMILY_UI_LABEL,
            variable=self.setup_model_vars[XGB_FAMILY_ID],
        ).grid(row=2, column=0, sticky="w")

        ttk.Label(
            self.setup_tab,
            text="CV defaults are 5-fold outer / 3-fold inner nested when the setup nested toggle is enabled.",
            justify="left",
            wraplength=980,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_testing_tab(self) -> None:
        self.testing_tab.columnconfigure(1, weight=1)
        self.testing_tab.columnconfigure(2, weight=1)
        ttk.Label(self.testing_tab, text="Target family").grid(row=0, column=0, sticky="w")
        self.mt_target_combo = ttk.Combobox(
            self.testing_tab,
            textvariable=self.mt_target_family_var,
            values=("v25_policies", "taste_policies", "v25_and_taste"),
            state="readonly",
            width=28,
        )
        self.mt_target_combo.grid(row=0, column=1, sticky="w")
        self.mt_target_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_mt_target_preview())
        ttk.Label(self.testing_tab, textvariable=self.mt_target_preview_var, wraplength=920, justify="left").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

        candidates = ttk.LabelFrame(self.testing_tab, text="Candidate Feature Sets")
        candidates.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(8, 8))
        self.mt_candidates_frame = candidates

        families = ttk.LabelFrame(self.testing_tab, text="Model Families + Output Modes")
        families.grid(row=2, column=1, columnspan=2, sticky="nsew", pady=(8, 8), padx=(8, 0))
        ttk.Label(families, text="Family").grid(row=0, column=0, sticky="w")
        ttk.Label(families, text="Use").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(families, text="Single").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(families, text="Multi").grid(row=0, column=3, sticky="w", padx=(8, 0))
        for row, (key, label) in enumerate(
            [
                ("linear_l2", "Linear L2 (Ridge/LogReg)"),
                ("linear_svm", "Linear SVM/SVR"),
                (XGB_FAMILY_ID, XGB_FAMILY_UI_LABEL),
                ("mlp", "MLP"),
                ("elasticnet", "ElasticNet"),
                ("randomforest", "RandomForest"),
            ],
            start=1,
        ):
            ttk.Label(families, text=label).grid(row=row, column=0, sticky="w")
            ttk.Checkbutton(families, variable=self.mt_model_family_vars[key]).grid(row=row, column=1, sticky="w", padx=(8, 0))
            ttk.Checkbutton(
                families,
                variable=self.mt_model_family_output_vars[key]["single_target"],
            ).grid(row=row, column=2, sticky="w", padx=(8, 0))
            ttk.Checkbutton(
                families,
                variable=self.mt_model_family_output_vars[key]["multi_output"],
                state="normal" if key == "mlp" else "disabled",
            ).grid(row=row, column=3, sticky="w", padx=(8, 0))

        settings = ttk.LabelFrame(self.testing_tab, text="Settings")
        settings.grid(row=3, column=0, columnspan=3, sticky="ew")
        settings.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            settings,
            text="Repeat stratified CV with new random seeds",
            variable=self.mt_repeat_cv_var,
            command=self._on_testing_repeat_toggle,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(settings, text="Repeat count").grid(row=0, column=1, sticky="e")
        self.mt_repeat_entry = ttk.Entry(settings, textvariable=self.mt_repeat_count_var, width=8)
        self.mt_repeat_entry.grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(settings, text="Save model configs after training", variable=self.mt_save_model_configs_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Checkbutton(
            settings,
            text="Use latest XGB calibration defaults",
            variable=self.mt_use_latest_xgb_calibration_var,
        ).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(
            settings,
            text="Use latest RF calibration defaults",
            variable=self.mt_use_latest_rf_calibration_var,
        ).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(
            settings,
            text="Use latest MLP calibration defaults",
            variable=self.mt_use_latest_mlp_calibration_var,
        ).grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(settings, text="Force rebuild intermediary banks", variable=self.mt_force_rebuild_var).grid(
            row=1, column=2, sticky="w"
        )
        ttk.Label(settings, text="Embedding model").grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(settings, textvariable=self.mt_embedding_model_var).grid(row=4, column=1, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(settings, text="XGB calibration sweep").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Label(settings, textvariable=self.mt_calibration_sweep_var, justify="left").grid(row=5, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(settings, text="RF calibration sweep").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Label(settings, textvariable=self.mt_rf_calibration_sweep_var, justify="left").grid(row=6, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(settings, text="MLP calibration sweep").grid(row=7, column=0, sticky="w", pady=(6, 0))
        ttk.Label(settings, textvariable=self.mt_mlp_calibration_sweep_var, justify="left").grid(row=7, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(settings, text="Latest XGB calibration").grid(row=8, column=0, sticky="w", pady=(4, 0))
        ttk.Label(settings, textvariable=self.mt_latest_calibration_var, justify="left", wraplength=900).grid(
            row=8, column=1, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Label(settings, text="Latest RF calibration").grid(row=9, column=0, sticky="w", pady=(4, 0))
        ttk.Label(settings, textvariable=self.mt_latest_rf_calibration_var, justify="left", wraplength=900).grid(
            row=9, column=1, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Label(settings, text="Latest MLP calibration").grid(row=10, column=0, sticky="w", pady=(4, 0))
        ttk.Label(settings, textvariable=self.mt_latest_mlp_calibration_var, justify="left", wraplength=900).grid(
            row=10, column=1, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Label(
            settings,
            text="Screening is training-only: held-out/test features and labels are not used.",
            justify="left",
        ).grid(row=11, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Button(
            settings,
            text="Apply Ablation Preset (v25, 19-set, Linear L2)",
            command=self._apply_ablation_v25_19set_linear_preset,
        ).grid(row=12, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _build_combination_tab(self) -> None:
        self.combination_tab.columnconfigure(0, weight=1)
        self.combination_tab.columnconfigure(1, weight=1)

        step1 = ttk.LabelFrame(self.combination_tab, text="Step 1: Combination Screening (Train Only)")
        step1.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        step1.columnconfigure(1, weight=1)

        ttk.Label(step1, text="Target family").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            step1,
            textvariable=self.combo_target_family_var,
            values=("v25_policies", "taste_policies", "v25_and_taste"),
            state="readonly",
            width=28,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(step1, text="Locked candidate feature sets").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        ttk.Label(
            step1,
            text="\n".join(self.COMBINATION_FEATURE_SET_IDS),
            justify="left",
        ).grid(row=1, column=1, sticky="w", pady=(8, 0))

        families = ttk.Frame(step1)
        families.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(families, text="Model family (locked)").grid(row=0, column=0, sticky="w")
        ttk.Label(families, text="Linear L2 (single_target)").grid(row=1, column=0, sticky="w")

        ttk.Checkbutton(
            step1,
            text="Repeat stratified CV with new random seeds",
            variable=self.combo_repeat_cv_var,
            command=self._on_combination_repeat_toggle,
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Label(step1, text="Repeat count").grid(row=3, column=1, sticky="w", padx=(180, 0), pady=(8, 0))
        self.combo_repeat_entry = ttk.Entry(step1, textvariable=self.combo_repeat_count_var, width=8)
        self.combo_repeat_entry.grid(row=3, column=1, sticky="w", padx=(260, 0), pady=(8, 0))

        ttk.Checkbutton(
            step1,
            text="Save model configs after training",
            variable=self.combo_save_model_configs_var,
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            step1,
            text="Force rebuild intermediary feature banks",
            variable=self.combo_force_rebuild_var,
        ).grid(row=4, column=1, sticky="w", pady=(8, 0))

        ttk.Button(
            step1,
            text="Run Combination Screening (Train Only)",
            command=self.start_combination_screening,
        ).grid(row=5, column=0, sticky="w", pady=(10, 0))

        ttk.Label(
            step1,
            text=(
                "Step 1 always runs model_testing_mode with held-out disabled and ridge-only "
                "(Linear L2, single_target)."
            ),
            justify="left",
            wraplength=980,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        step2 = ttk.LabelFrame(self.combination_tab, text="Step 2: Success Screening (Train CV Only)")
        step2.grid(row=1, column=0, columnspan=2, sticky="ew")
        step2.columnconfigure(1, weight=1)

        ttk.Label(step2, text="Saved bundle root").grid(row=0, column=0, sticky="w")
        ttk.Entry(step2, textvariable=self.combo_bundle_root_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(step2, text="Refresh Bundles", command=self._refresh_combination_bundle_choices).grid(row=0, column=2, sticky="w")

        ttk.Label(step2, text="Bundle").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.combo_bundle_combo = ttk.Combobox(
            step2,
            textvariable=self.combo_bundle_selection_var,
            state="readonly",
            width=82,
        )
        self.combo_bundle_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        self.combo_bundle_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_combination_combo_choices())

        ttk.Label(step2, text="Selected combo").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.combo_selected_combo_combo = ttk.Combobox(
            step2,
            textvariable=self.combo_selected_combo_var,
            state="readonly",
            width=82,
        )
        self.combo_selected_combo_combo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(8, 0))

        ttk.Label(step2, textvariable=self.combo_bundle_status_var, justify="left").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        ttk.Button(
            step2,
            text="Run Success Screening (Train CV Only)",
            command=self.start_combination_success_screening,
        ).grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Label(
            step2,
            text=(
                "Step 2 uses the selected reasoning-prediction combo and runs success CV across all 7 base "
                "feature combinations with HQ override both ON and OFF."
            ),
            justify="left",
            wraplength=980,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

        step3 = ttk.LabelFrame(self.combination_tab, text="Step 3: Held-Out Test Evaluation (Selected Success Models)")
        step3.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        step3.columnconfigure(0, weight=1)
        step3.rowconfigure(1, weight=1)
        ttk.Label(
            step3,
            text="Select one or multiple success models (branch + HQ override variant) for held-out testing.",
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        self.combo_success_branch_listbox = tk.Listbox(
            step3,
            selectmode="extended",
            exportselection=False,
            height=10,
        )
        self.combo_success_branch_listbox.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.combo_success_branch_listbox.bind(
            "<<ListboxSelect>>",
            lambda _event: self._refresh_combination_success_branch_count(),
        )
        combo_branch_scroll = ttk.Scrollbar(
            step3,
            orient="vertical",
            command=self.combo_success_branch_listbox.yview,
        )
        combo_branch_scroll.grid(row=1, column=1, sticky="ns", pady=(6, 0))
        self.combo_success_branch_listbox.configure(yscrollcommand=combo_branch_scroll.set)
        combo_branch_actions = ttk.Frame(step3)
        combo_branch_actions.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Button(
            combo_branch_actions,
            text="Select All",
            command=self._select_all_combination_success_branches,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            combo_branch_actions,
            text="Clear Selection",
            command=self._clear_combination_success_branches,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(
            step3,
            textvariable=self.combo_success_branch_count_var,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Button(
            step3,
            text="Run Held-Out Test Evaluation (Selected Success Models)",
            command=self.start_combination_success_test_eval,
        ).grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Label(
            step3,
            text=(
                "Step 3 reuses the selected reasoning-prediction combo and evaluates only selected success "
                "model branches on held-out test."
            ),
            justify="left",
            wraplength=980,
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))

    def _build_saved_eval_tab(self) -> None:
        self.saved_eval_tab.columnconfigure(1, weight=1)
        ttk.Label(self.saved_eval_tab, text="Saved bundle root").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.saved_eval_tab, textvariable=self.saved_bundle_root_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 8)
        )
        ttk.Button(
            self.saved_eval_tab,
            text="Refresh Bundles",
            command=self._refresh_saved_bundle_choices,
        ).grid(row=0, column=2, sticky="w")

        ttk.Label(self.saved_eval_tab, text="Bundle").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.saved_bundle_combo = ttk.Combobox(
            self.saved_eval_tab,
            textvariable=self.saved_bundle_selection_var,
            state="readonly",
            width=80,
        )
        self.saved_bundle_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        self.saved_bundle_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_saved_eval_combo_choices())

        ttk.Label(self.saved_eval_tab, text="Evaluation mode").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.saved_eval_mode_combo = ttk.Combobox(
            self.saved_eval_tab,
            textvariable=self.saved_eval_mode_var,
            values=(
                "reasoning_test_metrics",
                "success_with_pred_reasoning",
                "full_transfer_report",
                "combination_transfer_report",
            ),
            state="readonly",
            width=40,
        )
        self.saved_eval_mode_combo.grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(self.saved_eval_tab, text="HQ override mode").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.saved_hq_override_combo = ttk.Combobox(
            self.saved_eval_tab,
            textvariable=self.saved_hq_override_mode_var,
            values=(
                "with_override",
                "both_with_and_without",
                "force_off_all_branches",
                "force_on_all_branches",
                "both_force_off_and_on_all_branches",
            ),
            state="readonly",
            width=40,
        )
        self.saved_hq_override_combo.grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            self.saved_eval_tab,
            text="Build per-target composite (choose best CV R^2 combo per target)",
            variable=self.saved_eval_best_r2_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

        picks_frame = ttk.LabelFrame(self.saved_eval_tab, text="Cross-Run Model Picks (Optional)")
        picks_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        picks_frame.columnconfigure(1, weight=1)
        picks_frame.columnconfigure(2, weight=1)
        ttk.Label(picks_frame, text="Model family").grid(row=0, column=0, sticky="w")
        ttk.Label(picks_frame, text="Bundle").grid(row=0, column=1, sticky="w")
        ttk.Label(picks_frame, text="Combo").grid(row=0, column=2, sticky="w")
        row_specs = [
            ("linear_l2", "Linear L2"),
            (XGB_FAMILY_ID, XGB_FAMILY_UI_LABEL),
            ("mlp", "MLP"),
        ]
        self.saved_model_pick_bundle_combos: dict[str, ttk.Combobox] = {}
        self.saved_model_pick_combo_combos: dict[str, ttk.Combobox] = {}
        for row_index, (family_id, label) in enumerate(row_specs, start=1):
            ttk.Label(picks_frame, text=label).grid(row=row_index, column=0, sticky="w", pady=(6, 0))
            bundle_combo = ttk.Combobox(
                picks_frame,
                textvariable=self.saved_model_pick_bundle_vars[family_id],
                state="readonly",
                width=46,
            )
            bundle_combo.grid(row=row_index, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))
            bundle_combo.bind(
                "<<ComboboxSelected>>",
                lambda _event, model_family=family_id: self._refresh_saved_pick_combo_choices(model_family),
            )
            combo_combo = ttk.Combobox(
                picks_frame,
                textvariable=self.saved_model_pick_combo_vars[family_id],
                state="readonly",
                width=64,
            )
            combo_combo.grid(row=row_index, column=2, sticky="ew", pady=(6, 0))
            self.saved_model_pick_bundle_combos[family_id] = bundle_combo
            self.saved_model_pick_combo_combos[family_id] = combo_combo

        ttk.Label(
            picks_frame,
            text=(
                "Pick combos from different runs, then enable the composite toggle above. "
                "Composite uses the best source CV R^2 per target."
            ),
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Button(
            self.saved_eval_tab,
            text="Lambda Bundle Transfer Report Preset",
            command=self._apply_lambda_bundle_transfer_preset,
        ).grid(row=6, column=0, sticky="w", pady=(10, 0))

        combo_frame = ttk.LabelFrame(self.saved_eval_tab, text="Saved Combos To Evaluate")
        combo_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        combo_frame.columnconfigure(0, weight=1)
        combo_frame.rowconfigure(1, weight=1)
        ttk.Label(
            combo_frame,
            textvariable=self.saved_combo_count_var,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.saved_combo_listbox = tk.Listbox(
            combo_frame,
            selectmode="extended",
            exportselection=False,
            height=10,
        )
        self.saved_combo_listbox.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        combo_scroll = ttk.Scrollbar(combo_frame, orient="vertical", command=self.saved_combo_listbox.yview)
        combo_scroll.grid(row=1, column=1, sticky="ns", pady=(6, 0))
        self.saved_combo_listbox.configure(yscrollcommand=combo_scroll.set)
        combo_actions = ttk.Frame(combo_frame)
        combo_actions.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(combo_actions, text="Select All", command=self._select_all_saved_eval_combos).grid(row=0, column=0, sticky="w")
        ttk.Button(combo_actions, text="Clear Selection", command=self._clear_saved_eval_combos).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Button(
            self.saved_eval_tab,
            text="Run Saved Config Eval",
            command=self.start_saved_config_eval,
        ).grid(row=8, column=0, sticky="w", pady=(12, 0))
        ttk.Label(
            self.saved_eval_tab,
            text=(
                "reasoning_test_metrics: evaluate saved reasoning models on held-out test targets.\n"
                "success_with_pred_reasoning: evaluate HQ/LLM branches using predicted reasoning + fixed L2 C=5.\n"
                "full_transfer_report: lambda-bundle transfer report (reasoning + success + reproduction consistency).\n"
                "combination_transfer_report: single selected combo -> held-out reasoning + 7-way success transfer."
            ),
            justify="left",
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _load_config_preview(self) -> None:
        config = load_experiment_config(self.config_path_var.get().strip())
        self._loaded_config = config
        self.run_mode_var.set(config.defaults.run_mode)
        self.target_family_var.set(config.defaults.target_family)
        self.mt_target_family_var.set(config.defaults.target_family)
        self.setup_target_combo.configure(values=[spec.family_id for spec in config.target_families] + ["v25_and_taste"])
        self.defaults_summary_var.set(
            f"default mode={config.defaults.run_mode} | target={config.defaults.target_family} | "
            f"repro folds={config.reproduction.outer_cv.n_splits}/{config.reproduction.inner_cv.n_splits} | "
            f"distillation folds={config.distillation_cv.n_splits} | model-testing repeats={config.model_testing.screening_repeat_cv_count}"
        )
        self.heldout_var.set(bool(config.defaults.heldout_evaluation))
        self.repeat_cv_var.set(False)
        self.repeat_count_var.set("1")
        self.setup_nested_cv_var.set(False)
        self.mt_repeat_cv_var.set(True)
        self.mt_repeat_count_var.set(str(config.model_testing.screening_repeat_cv_count))
        self.combo_target_family_var.set("v25_policies")
        self.combo_repeat_cv_var.set(True)
        self.combo_repeat_count_var.set(str(self.COMBINATION_SCREENING_DEFAULT_REPEAT_COUNT))
        self.combo_save_model_configs_var.set(bool(config.model_testing.save_model_configs_after_training_default))
        self.combo_force_rebuild_var.set(False)
        self.mt_save_model_configs_var.set(bool(config.model_testing.save_model_configs_after_training_default))
        self.mt_use_latest_xgb_calibration_var.set(bool(config.model_testing.use_latest_xgb_calibration_default))
        self.mt_use_latest_rf_calibration_var.set(bool(config.model_testing.use_latest_rf_calibration_default))
        self.mt_use_latest_mlp_calibration_var.set(bool(config.model_testing.use_latest_mlp_calibration_default))
        self.saved_eval_best_r2_var.set(False)
        default_model_families = set(config.model_testing.default_model_families)
        for family_id, var in self.mt_model_family_vars.items():
            var.set(family_id in default_model_families)
        for family_id, mode_vars in self.mt_model_family_output_vars.items():
            if family_id == "mlp":
                mode_vars["single_target"].set(False)
                mode_vars["multi_output"].set(True)
            else:
                mode_vars["single_target"].set(True)
                mode_vars["multi_output"].set(False)
        self.mt_calibration_sweep_var.set(
            ", ".join(str(value) for value in config.model_testing.xgb_calibration_estimators)
        )
        self.mt_rf_calibration_sweep_var.set(
            "min_leaf="
            + ",".join(str(value) for value in config.model_testing.rf_calibration_min_samples_leaf)
            + " | max_depth="
            + ",".join("None" if value is None else str(value) for value in config.model_testing.rf_calibration_max_depth)
            + " | max_features="
            + ",".join(str(value) for value in config.model_testing.rf_calibration_max_features)
        )
        self.mt_mlp_calibration_sweep_var.set(
            "hidden_layer_sizes="
            + ", ".join(str(tuple(layer)) for layer in config.model_testing.mlp_calibration_hidden_layer_sizes)
            + " | alpha="
            + ",".join(str(value) for value in config.model_testing.mlp_calibration_alpha)
        )
        latest = load_latest_xgb_calibration(config.experiment_id)
        if latest is None:
            self.mt_latest_calibration_var.set("No calibration found")
        else:
            selected = latest.get("selected_n_estimators_by_family", {})
            self.mt_latest_calibration_var.set(
                f"from {latest.get('run_dir', 'unknown')}: {selected}"
            )
        latest_rf = load_latest_rf_calibration(config.experiment_id)
        if latest_rf is None:
            self.mt_latest_rf_calibration_var.set("No RF calibration found")
        else:
            selected_rf = latest_rf.get("selected_params_by_family", {})
            self.mt_latest_rf_calibration_var.set(
                f"from {latest_rf.get('run_dir', 'unknown')}: {selected_rf}"
            )
        latest_mlp = load_latest_mlp_calibration(config.experiment_id)
        if latest_mlp is None:
            self.mt_latest_mlp_calibration_var.set("No MLP calibration found")
        else:
            selected_mlp = latest_mlp.get("selected_params_by_family", {})
            self.mt_latest_mlp_calibration_var.set(
                f"from {latest_mlp.get('run_dir', 'unknown')}: {selected_mlp}"
            )

        for spec in config.intermediary_features:
            if spec.kind.startswith("sentence_transformer") and spec.embedding_model_name:
                self.embedding_model_var.set(spec.embedding_model_name)
                self.mt_embedding_model_var.set(spec.embedding_model_name)
                break
        self._rebuild_feature_bank_controls()
        self._rebuild_testing_candidate_controls()
        self._refresh_target_preview()
        self._refresh_mt_target_preview()
        self._sync_setup_mode()
        self._on_setup_repeat_toggle()
        self._on_testing_repeat_toggle()
        self._on_combination_repeat_toggle()
        self._refresh_saved_bundle_choices()
        self._refresh_combination_bundle_choices()
        self._refresh_combination_success_branch_choices()

    def _reset_defaults(self) -> None:
        if self._loaded_config is None:
            return
        self._load_config_preview()
        self.status_var.set("Reset to defaults")

    def _on_combination_repeat_toggle(self) -> None:
        if not self.combo_repeat_cv_var.get():
            self.combo_repeat_count_var.set("1")
            self.combo_repeat_entry.state(["disabled"])
            return
        try:
            value = int(self.combo_repeat_count_var.get().strip() or "0")
        except ValueError:
            value = 0
        if value < 2:
            default_repeat = self.COMBINATION_SCREENING_DEFAULT_REPEAT_COUNT
            self.combo_repeat_count_var.set(str(default_repeat))
        self.combo_repeat_entry.state(["!disabled"])

    @staticmethod
    def _format_combination_combo_label(combo: dict[str, object]) -> str:
        return (
            f"{combo.get('combo_id', 'unknown')} | "
            f"{combo.get('target_family', 'unknown')} | "
            f"{combo.get('feature_set_id', 'unknown')} | "
            f"{combo.get('model_id', 'unknown')} | "
            f"{combo.get('output_mode', 'unknown')}"
        )

    def _refresh_combination_bundle_choices(self) -> None:
        root_text = self.combo_bundle_root_var.get().strip() or "data/saved_model_configs"
        root_path = resolve_repo_path(root_text)
        if root_text == "data/saved_model_configs":
            bundles = list_saved_bundle_dirs()
        elif root_path.exists():
            bundles = sorted([path for path in root_path.iterdir() if path.is_dir()], key=lambda item: item.name, reverse=True)
        else:
            bundles = []
        values = [str(path) for path in bundles]
        self.combo_bundle_combo.configure(values=values)
        current = self.combo_bundle_selection_var.get().strip()
        if current and current in values:
            self._refresh_combination_combo_choices()
            return
        self.combo_bundle_selection_var.set(values[0] if values else "")
        self._refresh_combination_combo_choices()

    def _refresh_combination_combo_choices(self) -> None:
        bundle_value = self.combo_bundle_selection_var.get().strip()
        self._combo_combo_choices = []
        self.combo_selected_combo_var.set("")
        if not bundle_value:
            self.combo_selected_combo_combo.configure(values=[])
            self.combo_bundle_status_var.set("No bundle selected")
            return
        try:
            bundle_dir, manifest = load_bundle_manifest(bundle_value)
        except Exception as exc:
            self.combo_selected_combo_combo.configure(values=[])
            self.combo_bundle_status_var.set(f"Failed to read bundle manifest: {exc}")
            return
        allowed_sets = set(self.COMBINATION_FEATURE_SET_IDS)
        choices: list[tuple[str, str]] = []
        for item in list(manifest.get("combos", [])):
            combo = dict(item)
            combo_id = str(combo.get("combo_id", "")).strip()
            if not combo_id:
                continue
            if str(combo.get("target_family", "")).strip() != "v25_policies":
                continue
            if str(combo.get("task_kind", "")).strip() != "regression":
                continue
            if str(combo.get("feature_set_id", "")).strip() not in allowed_sets:
                continue
            label = self._format_combination_combo_label(combo)
            choices.append((combo_id, label))
        self._combo_combo_choices = choices
        labels = [label for _, label in choices]
        self.combo_selected_combo_combo.configure(values=labels)
        if labels:
            self.combo_selected_combo_var.set(labels[0])
            self.combo_bundle_status_var.set(f"Loaded {len(labels)} valid v25 combination combos from {bundle_dir}")
        else:
            self.combo_bundle_status_var.set(f"No valid v25 combination combos found in {bundle_dir}")

    def _selected_combination_combo_ref(self) -> str | None:
        bundle_value = self.combo_bundle_selection_var.get().strip()
        label_value = self.combo_selected_combo_var.get().strip()
        if not bundle_value or not label_value:
            return None
        choice_map = {label: combo_id for combo_id, label in self._combo_combo_choices}
        combo_id = choice_map.get(label_value)
        if combo_id is None:
            return None
        return f"{bundle_value}::{combo_id}"

    def _refresh_combination_success_branch_choices(self) -> None:
        if not hasattr(self, "combo_success_branch_listbox"):
            return
        self.combo_success_branch_listbox.delete(0, "end")
        self._combo_success_branch_ids_by_index = []
        for base_combo_id in self.COMBINATION_SUCCESS_BASE_COMBO_IDS:
            for override_branch in self.COMBINATION_SUCCESS_OVERRIDE_BRANCHES:
                branch_id = f"{base_combo_id}__{override_branch}"
                self._combo_success_branch_ids_by_index.append(branch_id)
                self.combo_success_branch_listbox.insert("end", branch_id)
        self._select_all_combination_success_branches()
        self._refresh_combination_success_branch_count()

    def _refresh_combination_success_branch_count(self) -> None:
        if not hasattr(self, "combo_success_branch_listbox"):
            return
        selected_count = len(self.combo_success_branch_listbox.curselection())
        total_count = len(self._combo_success_branch_ids_by_index)
        self.combo_success_branch_count_var.set(
            f"Selected {selected_count}/{total_count} success branches."
        )

    def _select_all_combination_success_branches(self) -> None:
        if not hasattr(self, "combo_success_branch_listbox"):
            return
        self.combo_success_branch_listbox.selection_set(0, "end")
        self._refresh_combination_success_branch_count()

    def _clear_combination_success_branches(self) -> None:
        if not hasattr(self, "combo_success_branch_listbox"):
            return
        self.combo_success_branch_listbox.selection_clear(0, "end")
        self._refresh_combination_success_branch_count()

    def _selected_combination_success_branch_ids(self) -> list[str]:
        if not hasattr(self, "combo_success_branch_listbox"):
            return []
        selected_indices = list(self.combo_success_branch_listbox.curselection())
        if not selected_indices:
            return []
        return [self._combo_success_branch_ids_by_index[index] for index in selected_indices]

    def _refresh_saved_bundle_choices(self) -> None:
        root_text = self.saved_bundle_root_var.get().strip() or "data/saved_model_configs"
        root_path = resolve_repo_path(root_text)
        if root_text == "data/saved_model_configs":
            bundles = list_saved_bundle_dirs()
        elif root_path.exists():
            bundles = sorted([path for path in root_path.iterdir() if path.is_dir()], key=lambda item: item.name, reverse=True)
        else:
            bundles = []
        values = [str(path) for path in bundles]
        self.saved_bundle_combo.configure(values=values)
        for model_family in ("linear_l2", XGB_FAMILY_ID, "mlp"):
            bundle_combo = self.saved_model_pick_bundle_combos.get(model_family)
            if bundle_combo is not None:
                bundle_combo.configure(values=values)
            current_pick = self.saved_model_pick_bundle_vars[model_family].get().strip()
            if current_pick and current_pick in values:
                pass
            else:
                self.saved_model_pick_bundle_vars[model_family].set(values[0] if values else "")
            self._refresh_saved_pick_combo_choices(model_family)
        current = self.saved_bundle_selection_var.get().strip()
        if current and current in values:
            self._refresh_saved_eval_combo_choices()
            return
        self.saved_bundle_selection_var.set(values[0] if values else "")
        self._refresh_saved_eval_combo_choices()

    @staticmethod
    def _format_saved_combo_label(combo: dict[str, object]) -> str:
        return (
            f"{combo.get('combo_id', 'unknown')} | "
            f"{combo.get('target_family', 'unknown')} | "
            f"{combo.get('feature_set_id', 'unknown')} | "
            f"{combo.get('model_id', 'unknown')} | "
            f"{combo.get('output_mode', 'unknown')}"
        )

    @staticmethod
    def _saved_pick_family_match(model_family: str, model_id: str) -> bool:
        model_id_use = str(model_id).strip().lower()
        if model_family == "linear_l2":
            return model_id_use in {"ridge", "logreg_classifier"}
        if model_family == XGB_FAMILY_ID:
            return model_id_use.startswith("xgb")
        if model_family == "mlp":
            return model_id_use.startswith("mlp")
        return False

    @staticmethod
    def _format_saved_pick_combo_label(combo: dict[str, object]) -> str:
        return (
            f"{combo.get('combo_id', 'unknown')} | "
            f"{combo.get('target_family', 'unknown')} | "
            f"{combo.get('feature_set_id', 'unknown')} | "
            f"{combo.get('model_id', 'unknown')} | "
            f"{combo.get('output_mode', 'unknown')}"
        )

    def _refresh_saved_pick_combo_choices(self, model_family: str) -> None:
        if model_family not in self.saved_model_pick_bundle_vars:
            return
        bundle_value = self.saved_model_pick_bundle_vars[model_family].get().strip()
        combo_widget = self.saved_model_pick_combo_combos.get(model_family)
        if combo_widget is None:
            return
        self.saved_model_pick_combo_vars[model_family].set("")
        self._saved_model_pick_combo_choices[model_family] = []
        if not bundle_value:
            combo_widget.configure(values=[])
            return
        try:
            _, manifest = load_bundle_manifest(bundle_value)
        except Exception:
            combo_widget.configure(values=[])
            return
        matched: list[tuple[str, str]] = []
        for item in list(manifest.get("combos", [])):
            combo = dict(item)
            combo_id = str(combo.get("combo_id", "")).strip()
            model_id = str(combo.get("model_id", "")).strip()
            if not combo_id or not self._saved_pick_family_match(model_family, model_id):
                continue
            label = self._format_saved_pick_combo_label(combo)
            matched.append((combo_id, label))
        self._saved_model_pick_combo_choices[model_family] = matched
        values = [label for _, label in matched]
        combo_widget.configure(values=values)
        if values:
            combo_widget.set(values[0])
            self.saved_model_pick_combo_vars[model_family].set(values[0])

    def _selected_saved_eval_combo_refs(self) -> list[str]:
        refs: list[str] = []
        for model_family in ("linear_l2", XGB_FAMILY_ID, "mlp"):
            bundle_value = self.saved_model_pick_bundle_vars[model_family].get().strip()
            label_value = self.saved_model_pick_combo_vars[model_family].get().strip()
            if not bundle_value or not label_value:
                continue
            choice_map = {
                label: combo_id
                for combo_id, label in self._saved_model_pick_combo_choices.get(model_family, [])
            }
            combo_id = choice_map.get(label_value)
            if combo_id is None:
                continue
            refs.append(f"{bundle_value}::{combo_id}")
        return refs

    def _apply_lambda_bundle_transfer_preset(self) -> None:
        self._refresh_saved_bundle_choices()
        bundle_values = list(self.saved_bundle_combo.cget("values"))
        if not bundle_values:
            self._append_log("ERROR: No saved model bundles are available for preset selection.")
            self.status_var.set("Invalid input")
            return

        desired: list[tuple[str, str]] = [
            ("linear_l2", "ridge"),
            (XGB_FAMILY_ID, "xgb3_regressor"),
            ("mlp", "mlp_regressor"),
        ]
        picks: dict[str, tuple[str, str]] = {}
        for family_id, model_id in desired:
            selected: tuple[str, str] | None = None
            for bundle_value in bundle_values:
                try:
                    _, manifest = load_bundle_manifest(bundle_value)
                except Exception:
                    continue
                for item in list(manifest.get("combos", [])):
                    combo = dict(item)
                    if str(combo.get("target_family", "")).strip() != "v25_policies":
                        continue
                    if str(combo.get("feature_set_id", "")).strip() != "lambda_policies_plus_sentence_bundle":
                        continue
                    if str(combo.get("model_id", "")).strip() != model_id:
                        continue
                    combo_id = str(combo.get("combo_id", "")).strip()
                    if not combo_id:
                        continue
                    selected = (bundle_value, combo_id)
                    break
                if selected is not None:
                    break
            if selected is None:
                self._append_log(
                    "ERROR: Could not find required combo for preset: "
                    f"model_id='{model_id}', target_family='v25_policies', "
                    "feature_set='lambda_policies_plus_sentence_bundle'."
                )
                self.status_var.set("Invalid input")
                return
            picks[family_id] = selected

        for family_id, (bundle_value, combo_id) in picks.items():
            self.saved_model_pick_bundle_vars[family_id].set(bundle_value)
            bundle_widget = self.saved_model_pick_bundle_combos.get(family_id)
            if bundle_widget is not None:
                bundle_widget.set(bundle_value)
            self._refresh_saved_pick_combo_choices(family_id)
            label_by_id = {
                combo_id_value: label
                for combo_id_value, label in self._saved_model_pick_combo_choices.get(family_id, [])
            }
            label = label_by_id.get(combo_id)
            if label is None:
                self._append_log(
                    f"ERROR: Preset combo id '{combo_id}' could not be selected in UI for family '{family_id}'."
                )
                self.status_var.set("Invalid input")
                return
            self.saved_model_pick_combo_vars[family_id].set(label)
            combo_widget = self.saved_model_pick_combo_combos.get(family_id)
            if combo_widget is not None:
                combo_widget.set(label)

        first_bundle = next(iter(picks.values()))[0]
        self.saved_bundle_selection_var.set(first_bundle)
        self.saved_eval_mode_var.set("full_transfer_report")
        self.saved_eval_best_r2_var.set(True)
        self.saved_hq_override_mode_var.set("with_override")
        self._refresh_saved_eval_combo_choices()
        self.status_var.set("Preset loaded")
        self._append_log(
            "Applied Lambda Bundle Transfer Report preset with ridge/xgb3_regressor/mlp_regressor cross-run picks."
        )

    def _select_all_saved_eval_combos(self) -> None:
        if not hasattr(self, "saved_combo_listbox"):
            return
        self.saved_combo_listbox.selection_set(0, "end")

    def _clear_saved_eval_combos(self) -> None:
        if not hasattr(self, "saved_combo_listbox"):
            return
        self.saved_combo_listbox.selection_clear(0, "end")

    def _selected_saved_eval_combo_ids(self) -> list[str] | None:
        if not hasattr(self, "saved_combo_listbox"):
            return None
        selected_indices = list(self.saved_combo_listbox.curselection())
        if not selected_indices:
            return []
        return [self._saved_combo_ids_by_index[index] for index in selected_indices]

    def _refresh_saved_eval_combo_choices(self) -> None:
        if not hasattr(self, "saved_combo_listbox"):
            return
        self.saved_combo_listbox.delete(0, "end")
        self._saved_combo_ids_by_index = []
        bundle_value = self.saved_bundle_selection_var.get().strip()
        if not bundle_value:
            self.saved_combo_count_var.set("No bundle selected")
            return
        try:
            bundle_dir, manifest = load_bundle_manifest(bundle_value)
        except Exception as exc:
            self.saved_combo_count_var.set(f"Failed to read bundle manifest: {exc}")
            return
        combos = [dict(item) for item in list(manifest.get("combos", []))]
        if not combos:
            self.saved_combo_count_var.set(f"No combos found in {bundle_dir}")
            return
        for combo in combos:
            combo_id = str(combo.get("combo_id", "")).strip()
            if not combo_id:
                continue
            self._saved_combo_ids_by_index.append(combo_id)
            self.saved_combo_listbox.insert("end", self._format_saved_combo_label(combo))
        self._select_all_saved_eval_combos()
        self.saved_combo_count_var.set(
            f"Loaded {len(self._saved_combo_ids_by_index)} combos from {bundle_dir}"
        )

    def _rebuild_feature_bank_controls(self) -> None:
        for child in self.setup_features_frame.winfo_children():
            child.destroy()
        self.feature_bank_vars.clear()
        self.setup_sentence_bundle_var = None
        if self._loaded_config is None:
            return
        default_selected = {"hq_baseline", "llm_engineering", "lambda_policies", "sentence_prose", "sentence_structured"}
        row = 0
        available_bank_ids: set[str] = set()
        for spec in [*self._loaded_config.repository_feature_banks, *self._loaded_config.intermediary_features]:
            if not spec.enabled:
                continue
            available_bank_ids.add(spec.feature_bank_id)
            value = spec.feature_bank_id in default_selected
            var = tk.BooleanVar(value=value)
            self.feature_bank_vars[spec.feature_bank_id] = var
            ttk.Checkbutton(self.setup_features_frame, text=spec.feature_bank_id, variable=var).grid(
                row=row, column=0, sticky="w"
            )
            row += 1

        if {"sentence_prose", "sentence_structured"}.issubset(available_bank_ids):
            bundle_default = bool(
                self.feature_bank_vars["sentence_prose"].get()
                and self.feature_bank_vars["sentence_structured"].get()
            )
            self.setup_sentence_bundle_var = tk.BooleanVar(value=bundle_default)
            ttk.Checkbutton(
                self.setup_features_frame,
                text="sentence_bundle",
                variable=self.setup_sentence_bundle_var,
            ).grid(row=row, column=0, sticky="w", pady=(6, 0))
            row += 1
            ttk.Label(
                self.setup_features_frame,
                text="bundle = prose + structured",
                justify="left",
            ).grid(row=row, column=0, sticky="w")

    def _rebuild_testing_candidate_controls(self) -> None:
        for child in self.mt_candidates_frame.winfo_children():
            child.destroy()
        self.mt_feature_set_vars.clear()
        if self._loaded_config is None:
            return
        defaults = set(self._loaded_config.model_testing.candidate_feature_sets)
        allowed = set(self.ABLATION_V25_19SET_LINEAR_FEATURE_SET_IDS)
        row = 0
        for spec in self._loaded_config.distillation_feature_sets:
            feature_set_id = spec.feature_set_id
            if feature_set_id not in allowed:
                continue
            value = feature_set_id in defaults
            var = tk.BooleanVar(value=value)
            self.mt_feature_set_vars[feature_set_id] = var
            ttk.Checkbutton(self.mt_candidates_frame, text=feature_set_id, variable=var).grid(
                row=row, column=0, sticky="w"
            )
            row += 1

    def _apply_ablation_v25_19set_linear_preset(self) -> None:
        self.mt_target_family_var.set("v25_policies")
        self.mt_repeat_cv_var.set(False)
        self.mt_repeat_count_var.set("1")
        self.mt_force_rebuild_var.set(False)
        self.mt_save_model_configs_var.set(False)
        self.mt_use_latest_xgb_calibration_var.set(False)
        self.mt_use_latest_rf_calibration_var.set(False)
        self.mt_use_latest_mlp_calibration_var.set(False)

        for family_id, var in self.mt_model_family_vars.items():
            var.set(family_id == "linear_l2")
        for family_id, mode_map in self.mt_model_family_output_vars.items():
            mode_map["single_target"].set(family_id == "linear_l2")
            mode_map["multi_output"].set(False)

        allowed = set(self.ABLATION_V25_19SET_LINEAR_FEATURE_SET_IDS)
        for feature_set_id, var in self.mt_feature_set_vars.items():
            var.set(feature_set_id in allowed)

        self._on_testing_repeat_toggle()
        self._refresh_mt_target_preview()
        self.status_var.set("Applied ablation preset (v25 19-set Linear L2)")

    def _sync_setup_mode(self) -> None:
        is_distillation = self.run_mode_var.get() == "reasoning_distillation_mode"
        state = "!disabled" if is_distillation else "disabled"
        self.setup_target_combo.state([state])
        if is_distillation and self.repeat_cv_var.get():
            self.setup_repeat_entry.state(["!disabled"])
        else:
            self.setup_repeat_entry.state(["disabled"])
        if is_distillation:
            self.target_preview_var.set(self.target_preview_var.get())

    def _on_setup_repeat_toggle(self) -> None:
        if not self.repeat_cv_var.get():
            self.repeat_count_var.set("1")
            self.setup_repeat_entry.state(["disabled"])
        else:
            try:
                value = int(self.repeat_count_var.get().strip() or "0")
            except ValueError:
                value = 0
            if value < 2:
                self.repeat_count_var.set("3")
            self.setup_repeat_entry.state(["!disabled"])
        self._sync_setup_mode()

    def _on_testing_repeat_toggle(self) -> None:
        if not self.mt_repeat_cv_var.get():
            self.mt_repeat_count_var.set("1")
            self.mt_repeat_entry.state(["disabled"])
            return
        try:
            value = int(self.mt_repeat_count_var.get().strip() or "0")
        except ValueError:
            value = 0
        if value < 2:
            default_repeat = self._loaded_config.model_testing.screening_repeat_cv_count if self._loaded_config else 3
            self.mt_repeat_count_var.set(str(default_repeat))
        self.mt_repeat_entry.state(["!disabled"])

    def _refresh_target_preview(self) -> None:
        if self._loaded_config is None:
            return
        family_id = self.target_family_var.get()
        if family_id == "v25_and_taste":
            v25 = next(spec for spec in self._loaded_config.target_families if spec.family_id == "v25_policies")
            taste = next(spec for spec in self._loaded_config.target_families if spec.family_id == "taste_policies")
            self.target_preview_var.set(
                f"v25_and_taste: {len(load_target_family(v25).target_columns)} regression + "
                f"{len(load_target_family(taste).target_columns)} classification targets."
            )
            return
        selected = next((spec for spec in self._loaded_config.target_families if spec.family_id == family_id), None)
        if selected is None:
            self.target_preview_var.set("")
            return
        loaded = load_target_family(selected)
        self.target_preview_var.set(f"{selected.family_id}: {len(loaded.target_columns)} targets ({selected.task_kind}).")

    def _refresh_mt_target_preview(self) -> None:
        if self._loaded_config is None:
            return
        family_id = self.mt_target_family_var.get()
        if family_id == "v25_and_taste":
            self.mt_target_preview_var.set("Sequential run: v25 then taste.")
            return
        selected = next((spec for spec in self._loaded_config.target_families if spec.family_id == family_id), None)
        if selected is None:
            self.mt_target_preview_var.set("")
            return
        loaded = load_target_family(selected)
        self.mt_target_preview_var.set(f"{selected.family_id}: {len(loaded.target_columns)} targets ({selected.task_kind}).")

    @staticmethod
    def _parse_optional_int(value: str) -> int | None:
        value = value.strip()
        return None if not value else int(value)

    def _build_common_kwargs(self) -> dict[str, object]:
        return {
            "config_path": self.config_path_var.get().strip(),
            "reproduction_outer_splits": None,
            "reproduction_inner_splits": None,
            "distillation_splits": None,
            "cv_random_state": None,
            "threshold_start": None,
            "threshold_stop": None,
            "threshold_step": None,
        }

    def _setup_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        selected_target_family = self.target_family_var.get().strip()
        models: list[str] = []
        if self.setup_model_vars["linear_l2"].get():
            if selected_target_family in {"v25_policies", "v25_and_taste"}:
                models.append("ridge")
            if selected_target_family in {"taste_policies", "v25_and_taste"}:
                models.append("logreg_classifier")
        if self.setup_model_vars["linear_svm"].get():
            if selected_target_family in {"v25_policies", "v25_and_taste"}:
                models.append("linear_svr_regressor")
            if selected_target_family in {"taste_policies", "v25_and_taste"}:
                models.append("linear_svm_classifier")
        if self.setup_model_vars[XGB_FAMILY_ID].get():
            if selected_target_family in {"v25_policies", "v25_and_taste"}:
                models.append(XGB_REGRESSOR_MODEL_KIND)
            if selected_target_family in {"taste_policies", "v25_and_taste"}:
                models.append(XGB_CLASSIFIER_MODEL_KIND)
        active_feature_banks = [key for key, var in self.feature_bank_vars.items() if var.get()]
        if self.setup_sentence_bundle_var is not None and self.setup_sentence_bundle_var.get():
            for sentence_bank in ("sentence_prose", "sentence_structured"):
                if sentence_bank not in active_feature_banks:
                    active_feature_banks.append(sentence_bank)
        return LauncherSelections(
            run_mode=self.run_mode_var.get().strip(),
            target_family=selected_target_family,
            heldout_evaluation=bool(self.heldout_var.get()),
            active_feature_banks=active_feature_banks,
            force_rebuild_intermediary_features=bool(self.force_rebuild_var.get()),
            reasoning_models=models,
            embedding_model_name=self.embedding_model_var.get().strip() or None,
            repeat_cv_with_new_seeds=bool(self.repeat_cv_var.get()),
            cv_seed_repeat_count=self._parse_optional_int(self.repeat_count_var.get()),
            distillation_nested_sweep=bool(self.setup_nested_cv_var.get()),
            save_reasoning_predictions=bool(self.save_predictions_var.get()),
            candidate_feature_sets=None,
            model_families=None,
            save_model_configs_after_training=None,
            saved_config_bundle_path=None,
            saved_eval_mode=None,
            hq_exit_override_mode=None,
            **common,
        )

    def _testing_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        model_families = [key for key, var in self.mt_model_family_vars.items() if var.get()]
        model_family_output_modes: dict[str, list[str]] = {}
        for family_id in model_families:
            selected_modes = [
                mode
                for mode, var in self.mt_model_family_output_vars[family_id].items()
                if var.get()
            ]
            model_family_output_modes[family_id] = selected_modes
        output_modes: list[str] = []
        for mode in ("single_target", "multi_output"):
            if any(mode in model_family_output_modes.get(family_id, []) for family_id in model_families):
                output_modes.append(mode)
        return LauncherSelections(
            run_mode="model_testing_mode",
            target_family=self.mt_target_family_var.get().strip(),
            heldout_evaluation=False,
            active_feature_banks=None,
            force_rebuild_intermediary_features=bool(self.mt_force_rebuild_var.get()),
            reasoning_models=None,
            embedding_model_name=self.mt_embedding_model_var.get().strip() or None,
            repeat_cv_with_new_seeds=bool(self.mt_repeat_cv_var.get()),
            cv_seed_repeat_count=self._parse_optional_int(self.mt_repeat_count_var.get()),
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            candidate_feature_sets=[key for key, var in self.mt_feature_set_vars.items() if var.get()],
            model_families=model_families,
            output_modes=output_modes,
            model_family_output_modes=model_family_output_modes,
            save_model_configs_after_training=bool(self.mt_save_model_configs_var.get()),
            saved_config_bundle_path=None,
            saved_eval_mode=None,
            hq_exit_override_mode=None,
            use_latest_xgb_calibration=bool(self.mt_use_latest_xgb_calibration_var.get()),
            use_latest_rf_calibration=bool(self.mt_use_latest_rf_calibration_var.get()),
            use_latest_mlp_calibration=bool(self.mt_use_latest_mlp_calibration_var.get()),
            **common,
        )

    def _combination_screening_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        model_families: list[str] = ["linear_l2"]
        model_family_output_modes: dict[str, list[str]] = {"linear_l2": ["single_target"]}
        output_modes: list[str] = ["single_target"]
        return LauncherSelections(
            run_mode="model_testing_mode",
            target_family=self.combo_target_family_var.get().strip(),
            heldout_evaluation=False,
            active_feature_banks=None,
            force_rebuild_intermediary_features=bool(self.combo_force_rebuild_var.get()),
            reasoning_models=None,
            embedding_model_name=self.mt_embedding_model_var.get().strip() or None,
            repeat_cv_with_new_seeds=bool(self.combo_repeat_cv_var.get()),
            cv_seed_repeat_count=self._parse_optional_int(self.combo_repeat_count_var.get()),
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            candidate_feature_sets=list(self.COMBINATION_FEATURE_SET_IDS),
            model_families=model_families,
            output_modes=output_modes,
            model_family_output_modes=model_family_output_modes,
            save_model_configs_after_training=bool(self.combo_save_model_configs_var.get()),
            saved_config_bundle_path=None,
            saved_eval_mode=None,
            saved_eval_success_branch_ids=None,
            hq_exit_override_mode=None,
            use_latest_xgb_calibration=False,
            use_latest_rf_calibration=False,
            use_latest_mlp_calibration=False,
            **common,
        )

    def _combination_success_screening_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        combo_ref = self._selected_combination_combo_ref()
        return LauncherSelections(
            run_mode="saved_config_evaluation_mode",
            target_family="v25_policies",
            heldout_evaluation=False,
            active_feature_banks=None,
            force_rebuild_intermediary_features=False,
            reasoning_models=None,
            embedding_model_name=None,
            repeat_cv_with_new_seeds=False,
            cv_seed_repeat_count=1,
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            candidate_feature_sets=None,
            model_families=None,
            output_modes=None,
            model_family_output_modes=None,
            save_model_configs_after_training=False,
            saved_config_bundle_path=self.combo_bundle_selection_var.get().strip() or None,
            saved_eval_mode="combination_transfer_report",
            saved_eval_combo_ids=None,
            saved_eval_combo_refs=[combo_ref] if combo_ref else None,
            saved_eval_success_branch_ids=None,
            saved_eval_per_target_best_r2=False,
            hq_exit_override_mode="both_force_off_and_on_all_branches",
            use_latest_xgb_calibration=False,
            use_latest_rf_calibration=False,
            use_latest_mlp_calibration=False,
            **common,
        )

    def _combination_success_test_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        combo_ref = self._selected_combination_combo_ref()
        selected_branch_ids = self._selected_combination_success_branch_ids()
        return LauncherSelections(
            run_mode="saved_config_evaluation_mode",
            target_family="v25_policies",
            heldout_evaluation=True,
            active_feature_banks=None,
            force_rebuild_intermediary_features=False,
            reasoning_models=None,
            embedding_model_name=None,
            repeat_cv_with_new_seeds=False,
            cv_seed_repeat_count=1,
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            candidate_feature_sets=None,
            model_families=None,
            output_modes=None,
            model_family_output_modes=None,
            save_model_configs_after_training=False,
            saved_config_bundle_path=self.combo_bundle_selection_var.get().strip() or None,
            saved_eval_mode="combination_transfer_report",
            saved_eval_combo_ids=None,
            saved_eval_combo_refs=[combo_ref] if combo_ref else None,
            saved_eval_success_branch_ids=selected_branch_ids,
            saved_eval_per_target_best_r2=False,
            hq_exit_override_mode="both_force_off_and_on_all_branches",
            use_latest_xgb_calibration=False,
            use_latest_rf_calibration=False,
            use_latest_mlp_calibration=False,
            **common,
        )

    def _saved_eval_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        bundle_value = self.saved_bundle_selection_var.get().strip()
        mode_value = self.saved_eval_mode_var.get().strip() or None
        use_full_transfer_mode = mode_value == "full_transfer_report"
        full_transfer_repeat_count = (
            self._loaded_config.model_testing.screening_repeat_cv_count
            if (use_full_transfer_mode and self._loaded_config is not None)
            else 1
        )
        use_composite_best_r2 = bool(self.saved_eval_best_r2_var.get())
        selected_combo_ids = None if use_composite_best_r2 else self._selected_saved_eval_combo_ids()
        selected_combo_refs = self._selected_saved_eval_combo_refs() if use_composite_best_r2 else None
        return LauncherSelections(
            run_mode="saved_config_evaluation_mode",
            target_family=self.mt_target_family_var.get().strip(),
            heldout_evaluation=True,
            active_feature_banks=None,
            force_rebuild_intermediary_features=False,
            reasoning_models=None,
            embedding_model_name=None,
            repeat_cv_with_new_seeds=bool(use_full_transfer_mode),
            cv_seed_repeat_count=int(full_transfer_repeat_count),
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            candidate_feature_sets=None,
            model_families=None,
            output_modes=None,
            model_family_output_modes=None,
            save_model_configs_after_training=False,
            saved_config_bundle_path=bundle_value or None,
            saved_eval_mode=mode_value,
            saved_eval_combo_ids=selected_combo_ids,
            saved_eval_combo_refs=selected_combo_refs,
            saved_eval_success_branch_ids=None,
            saved_eval_per_target_best_r2=use_composite_best_r2,
            hq_exit_override_mode=self.saved_hq_override_mode_var.get().strip() or None,
            use_latest_xgb_calibration=False,
            use_latest_rf_calibration=False,
            use_latest_mlp_calibration=False,
            **common,
        )

    def _xgb_calibration_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        sweep = []
        if self._loaded_config is not None:
            sweep = list(self._loaded_config.model_testing.xgb_calibration_estimators)
        return LauncherSelections(
            run_mode="xgb_calibration_mode",
            target_family=self.mt_target_family_var.get().strip(),
            heldout_evaluation=False,
            active_feature_banks=None,
            force_rebuild_intermediary_features=bool(self.mt_force_rebuild_var.get()),
            reasoning_models=None,
            embedding_model_name=self.mt_embedding_model_var.get().strip() or None,
            repeat_cv_with_new_seeds=False,
            cv_seed_repeat_count=1,
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            candidate_feature_sets=[key for key, var in self.mt_feature_set_vars.items() if var.get()],
            model_families=[XGB_FAMILY_ID],
            output_modes=["single_target"],
            save_model_configs_after_training=False,
            saved_config_bundle_path=None,
            saved_eval_mode=None,
            hq_exit_override_mode=None,
            xgb_calibration_estimators=sweep,
            use_latest_xgb_calibration=False,
            use_latest_rf_calibration=False,
            use_latest_mlp_calibration=False,
            **common,
        )

    def _rf_calibration_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        min_leaf = []
        max_depth = []
        max_features = []
        if self._loaded_config is not None:
            min_leaf = list(self._loaded_config.model_testing.rf_calibration_min_samples_leaf)
            max_depth = list(self._loaded_config.model_testing.rf_calibration_max_depth)
            max_features = list(self._loaded_config.model_testing.rf_calibration_max_features)
        return LauncherSelections(
            run_mode="rf_calibration_mode",
            target_family=self.mt_target_family_var.get().strip(),
            heldout_evaluation=False,
            active_feature_banks=None,
            force_rebuild_intermediary_features=bool(self.mt_force_rebuild_var.get()),
            reasoning_models=None,
            embedding_model_name=self.mt_embedding_model_var.get().strip() or None,
            repeat_cv_with_new_seeds=False,
            cv_seed_repeat_count=1,
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            candidate_feature_sets=[key for key, var in self.mt_feature_set_vars.items() if var.get()],
            model_families=["randomforest"],
            output_modes=["single_target"],
            save_model_configs_after_training=False,
            saved_config_bundle_path=None,
            saved_eval_mode=None,
            hq_exit_override_mode=None,
            use_latest_xgb_calibration=False,
            rf_calibration_min_samples_leaf=min_leaf,
            rf_calibration_max_depth=max_depth,
            rf_calibration_max_features=max_features,
            use_latest_rf_calibration=False,
            use_latest_mlp_calibration=False,
            **common,
        )

    def _mlp_calibration_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        hidden_layers: list[list[int]] = []
        alpha_values: list[float] = []
        if self._loaded_config is not None:
            hidden_layers = [list(layer) for layer in self._loaded_config.model_testing.mlp_calibration_hidden_layer_sizes]
            alpha_values = list(self._loaded_config.model_testing.mlp_calibration_alpha)
        return LauncherSelections(
            run_mode="mlp_calibration_mode",
            target_family=self.mt_target_family_var.get().strip(),
            heldout_evaluation=False,
            active_feature_banks=None,
            force_rebuild_intermediary_features=bool(self.mt_force_rebuild_var.get()),
            reasoning_models=None,
            embedding_model_name=self.mt_embedding_model_var.get().strip() or None,
            repeat_cv_with_new_seeds=False,
            cv_seed_repeat_count=1,
            distillation_nested_sweep=False,
            save_reasoning_predictions=False,
            candidate_feature_sets=[key for key, var in self.mt_feature_set_vars.items() if var.get()],
            model_families=["mlp"],
            output_modes=["single_target"],
            save_model_configs_after_training=False,
            saved_config_bundle_path=None,
            saved_eval_mode=None,
            hq_exit_override_mode=None,
            use_latest_xgb_calibration=False,
            use_latest_rf_calibration=False,
            mlp_calibration_hidden_layer_sizes=hidden_layers,
            mlp_calibration_alpha=alpha_values,
            use_latest_mlp_calibration=False,
            **common,
        )

    def _xgb_depth_test_batch_selections(self) -> list[tuple[int, LauncherSelections]]:
        common = self._build_common_kwargs()
        batch: list[tuple[int, LauncherSelections]] = []
        for depth in self.XGB_DEPTH_TEST_DEPTHS:
            selection = LauncherSelections(
                run_mode="model_testing_mode",
                target_family="v25_and_taste",
                heldout_evaluation=False,
                active_feature_banks=None,
                force_rebuild_intermediary_features=False,
                reasoning_models=None,
                embedding_model_name=self.mt_embedding_model_var.get().strip() or None,
                repeat_cv_with_new_seeds=True,
                cv_seed_repeat_count=self.XGB_DEPTH_TEST_REPEAT_COUNT,
                distillation_nested_sweep=False,
                save_reasoning_predictions=False,
                candidate_feature_sets=list(self.XGB_DEPTH_TEST_FEATURE_SET_IDS),
                model_families=[XGB_FAMILY_ID],
                output_modes=["single_target"],
                model_family_output_modes={XGB_FAMILY_ID: ["single_target"]},
                save_model_configs_after_training=False,
                saved_config_bundle_path=None,
                saved_eval_mode=None,
                hq_exit_override_mode=None,
                use_latest_xgb_calibration=False,
                use_latest_rf_calibration=False,
                use_latest_mlp_calibration=False,
                xgb_model_param_overrides_by_model_id={
                    XGB_REGRESSOR_MODEL_KIND: {
                        "n_estimators": self.XGB_DEPTH_TEST_N_ESTIMATORS,
                        "max_depth": depth,
                    },
                    XGB_CLASSIFIER_MODEL_KIND: {
                        "n_estimators": self.XGB_DEPTH_TEST_N_ESTIMATORS,
                        "max_depth": depth,
                    },
                },
                max_parallel_workers=self.XGB_DEPTH_TEST_MAX_PARALLEL_WORKERS,
                **common,
            )
            batch.append((depth, selection))
        return batch

    @staticmethod
    def _run_overrides_once(
        config: ExperimentConfig,
        overrides: RunOverrides,
        logger: Callable[[str], None] | None,
    ) -> Path:
        return run_pipeline(config, overrides, logger=logger)

    @staticmethod
    def _execute_xgb_depth_batch(
        *,
        config: ExperimentConfig,
        batch: list[tuple[int, LauncherSelections]],
        logger: Callable[[str], None] | None = None,
        run_once: Callable[[ExperimentConfig, RunOverrides, Callable[[str], None] | None], Path] | None = None,
    ) -> list[str]:
        run_once_fn = run_once or RunLauncher._run_overrides_once
        total = len(batch)
        completed_runs: list[str] = []
        for index, (depth, selections) in enumerate(batch, start=1):
            if logger is not None:
                logger(f"XGB depth test run {index}/{total}: max_depth={depth} starting.")
            overrides = selections_to_overrides(selections)
            run_dir = run_once_fn(config, overrides, logger)
            if logger is not None:
                logger(f"XGB depth test run {index}/{total}: max_depth={depth} complete. Artifacts: {run_dir}")
            completed_runs.append(f"max_depth={depth}: {run_dir}")
        return completed_runs

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def start_run_setup(self) -> None:
        try:
            selections = self._setup_selections()
            if selections.run_mode == "reasoning_distillation_mode" and not selections.reasoning_models:
                raise ValueError("Select at least one setup model family.")
            if selections.repeat_cv_with_new_seeds and (selections.cv_seed_repeat_count or 0) < 2:
                raise ValueError("Repeat count must be >= 2 when repeat CV is enabled.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting run-setup pipeline.")

    def start_model_testing(self) -> None:
        try:
            selections = self._testing_selections()
            if not selections.candidate_feature_sets:
                raise ValueError("Select at least one candidate feature set.")
            if not selections.model_families:
                raise ValueError("Select at least one model family.")
            if selections.model_family_output_modes is None:
                raise ValueError("Model-family output mapping was not initialized.")
            for family_id in selections.model_families:
                if not selections.model_family_output_modes.get(family_id):
                    raise ValueError(f"Select at least one output type for model family '{family_id}'.")
            if selections.repeat_cv_with_new_seeds and (selections.cv_seed_repeat_count or 0) < 2:
                raise ValueError("Repeat count must be >= 2 when repeat CV is enabled.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting model-testing pipeline.")

    def start_combination_screening(self) -> None:
        try:
            selections = self._combination_screening_selections()
            if selections.repeat_cv_with_new_seeds and (selections.cv_seed_repeat_count or 0) < 2:
                raise ValueError("Repeat count must be >= 2 when repeat CV is enabled.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting combination screening (train-only).")

    def start_combination_success_screening(self) -> None:
        try:
            selections = self._combination_success_screening_selections()
            combo_ref = selections.saved_eval_combo_refs[0] if selections.saved_eval_combo_refs else ""
            if not combo_ref:
                raise ValueError("Select exactly one valid combo from the selected bundle.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting combination success screening (train CV only).")

    def start_combination_success_test_eval(self) -> None:
        try:
            selections = self._combination_success_test_selections()
            combo_ref = selections.saved_eval_combo_refs[0] if selections.saved_eval_combo_refs else ""
            if not combo_ref:
                raise ValueError("Select exactly one valid combo from the selected bundle.")
            selected_branches = selections.saved_eval_success_branch_ids or []
            if not selected_branches:
                raise ValueError("Select at least one success model branch for held-out evaluation.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting combination held-out success test evaluation.")

    def start_saved_config_eval(self) -> None:
        try:
            selections = self._saved_eval_selections()
            is_combination_transfer_mode = selections.saved_eval_mode == "combination_transfer_report"
            if selections.saved_eval_per_target_best_r2:
                combo_refs = list(selections.saved_eval_combo_refs or [])
                if not combo_refs:
                    raise ValueError(
                        "Select at least one model-specific combo (Linear L2, XGB, or MLP) "
                        "when per-target composite is enabled."
                    )
                resolved_refs: list[str] = []
                first_bundle: str | None = None
                for combo_ref in combo_refs:
                    raw = str(combo_ref).strip()
                    if "::" not in raw:
                        raise ValueError(
                            f"Invalid combo ref '{raw}'. Expected format: <bundle_path_or_id>::<combo_id>"
                        )
                    bundle_part, combo_id = raw.split("::", 1)
                    bundle_token = bundle_part.strip()
                    combo_token = combo_id.strip()
                    if not bundle_token or not combo_token:
                        raise ValueError(
                            f"Invalid combo ref '{raw}'. Expected format: <bundle_path_or_id>::<combo_id>"
                        )
                    resolved_bundle = resolve_saved_bundle_path(bundle_token)
                    if not resolved_bundle.exists():
                        raise ValueError(f"Saved bundle path does not exist: {resolved_bundle}")
                    if first_bundle is None:
                        first_bundle = str(resolved_bundle)
                    resolved_refs.append(f"{resolved_bundle}::{combo_token}")
                selections = replace(
                    selections,
                    saved_config_bundle_path=first_bundle,
                    saved_eval_combo_refs=resolved_refs,
                    saved_eval_per_target_best_r2=(False if is_combination_transfer_mode else selections.saved_eval_per_target_best_r2),
                )
            else:
                bundle_raw = (selections.saved_config_bundle_path or "").strip()
                if not bundle_raw:
                    raise ValueError("Select a saved model config bundle.")
                resolved_bundle = resolve_saved_bundle_path(bundle_raw)
                if not resolved_bundle.exists():
                    raise ValueError(f"Saved bundle path does not exist: {resolved_bundle}")
                if selections.saved_eval_combo_ids is not None and len(selections.saved_eval_combo_ids) == 0:
                    raise ValueError("Select at least one saved combo for evaluation.")
                selections = replace(selections, saved_config_bundle_path=str(resolved_bundle))
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting saved-config evaluation pipeline.")

    def start_xgb_calibration(self) -> None:
        try:
            selections = self._xgb_calibration_selections()
            if not selections.candidate_feature_sets:
                raise ValueError("Select at least one candidate feature set.")
            if not selections.xgb_calibration_estimators:
                raise ValueError("Calibration estimator sweep cannot be empty.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting XGB calibration pipeline.")

    def start_rf_calibration(self) -> None:
        try:
            selections = self._rf_calibration_selections()
            if not selections.candidate_feature_sets:
                raise ValueError("Select at least one candidate feature set.")
            if not selections.rf_calibration_min_samples_leaf:
                raise ValueError("RF calibration min_samples_leaf sweep cannot be empty.")
            if not selections.rf_calibration_max_depth:
                raise ValueError("RF calibration max_depth sweep cannot be empty.")
            if not selections.rf_calibration_max_features:
                raise ValueError("RF calibration max_features sweep cannot be empty.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting RF calibration pipeline.")

    def start_mlp_calibration(self) -> None:
        try:
            selections = self._mlp_calibration_selections()
            if not selections.candidate_feature_sets:
                raise ValueError("Select at least one candidate feature set.")
            if not selections.mlp_calibration_hidden_layer_sizes:
                raise ValueError("MLP calibration hidden_layer_sizes sweep cannot be empty.")
            if not selections.mlp_calibration_alpha:
                raise ValueError("MLP calibration alpha sweep cannot be empty.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting MLP calibration pipeline.")

    def start_xgb_depth_test(self) -> None:
        try:
            batch = self._xgb_depth_test_batch_selections()
            if not batch:
                raise ValueError("XGB depth test batch is empty.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_batch_worker(
            batch,
            start_message=(
                "Starting XGB depth test batch (locked preset: "
                "v25_and_taste, HQ/Lambda prose+bundle, depths=3,5, repeats=4, n_estimators=320)."
            ),
        )

    def _start_worker(self, selections: LauncherSelections, start_message: str) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.status_var.set("Run already in progress")
            return
        self.status_var.set("Running")
        self.output_path_var.set("")
        self._append_log(start_message)
        self._run_started_monotonic = time.monotonic()
        self._last_heartbeat_monotonic = self._run_started_monotonic
        overrides = selections_to_overrides(selections)

        def worker() -> None:
            try:
                config = load_experiment_config(overrides.config_path)
                config = apply_launcher_config_overrides(config, selections)
                run_dir = run_pipeline(config, overrides, logger=lambda msg: self.queue.put(("log", msg)))
            except Exception as exc:  # pragma: no cover - UI path
                self.queue.put(("error", str(exc)))
                return
            self.queue.put(("done", str(run_dir)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _start_batch_worker(
        self,
        batch: list[tuple[int, LauncherSelections]],
        *,
        start_message: str,
    ) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.status_var.set("Run already in progress")
            return
        self.status_var.set("Running")
        self.output_path_var.set("")
        self._append_log(start_message)
        self._run_started_monotonic = time.monotonic()
        self._last_heartbeat_monotonic = self._run_started_monotonic

        def worker() -> None:
            try:
                config_path = batch[0][1].config_path
                config = load_experiment_config(config_path)
                completed_runs = self._execute_xgb_depth_batch(
                    config=config,
                    batch=batch,
                    logger=lambda msg: self.queue.put(("log", msg)),
                )
                self.queue.put(("done", "\n".join(completed_runs)))
            except Exception as exc:  # pragma: no cover - UI path
                self.queue.put(("error", str(exc)))
                return

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _poll_queue(self) -> None:
        while True:
            try:
                event_type, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if event_type == "log":
                self._append_log(payload)
            elif event_type == "error":
                self.status_var.set("Failed")
                self._append_log(f"ERROR: {payload}")
                self._run_started_monotonic = None
            elif event_type == "done":
                self.status_var.set("Complete")
                self.output_path_var.set(payload)
                self._append_log(f"Run finished. Artifacts: {payload}")
                self._run_started_monotonic = None
        if self.worker is not None and self.worker.is_alive() and self._run_started_monotonic is not None:
            now = time.monotonic()
            elapsed = int(now - self._run_started_monotonic)
            mins, secs = divmod(elapsed, 60)
            self.status_var.set(f"Running ({mins:02d}:{secs:02d})")
            if (now - self._last_heartbeat_monotonic) >= self._heartbeat_interval_seconds:
                self._append_log(f"Heartbeat: run is still active ({mins:02d}:{secs:02d} elapsed).")
                self._last_heartbeat_monotonic = now
        self.after(100, self._poll_queue)


def launch_app(initial_config_path: str = DEFAULT_CONFIG_PATH) -> None:
    root = tk.Tk()
    root.title("Feature Repository Pipeline Launcher")
    root.geometry("1280x960")
    RunLauncher(root, initial_config_path=initial_config_path)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
