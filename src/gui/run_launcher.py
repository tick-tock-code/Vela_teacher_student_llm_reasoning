from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass, replace
from tkinter import scrolledtext, ttk

from src.data.targets import load_target_family
from src.pipeline.config import ExperimentConfig, load_experiment_config
from src.pipeline.distillation import run_pipeline
from src.pipeline.run_options import DEFAULT_CONFIG_PATH, RunOverrides


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
    run_advanced_models: bool | None = None


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
        run_advanced_models=selections.run_advanced_models,
    )


def apply_launcher_config_overrides(config: ExperimentConfig, selections: LauncherSelections) -> ExperimentConfig:
    random_state = (
        config.distillation_cv.random_state if selections.cv_random_state is None else int(selections.cv_random_state)
    )
    reproduction = replace(
        config.reproduction,
        outer_cv=replace(
            config.reproduction.outer_cv,
            n_splits=(
                config.reproduction.outer_cv.n_splits
                if selections.reproduction_outer_splits is None
                else int(selections.reproduction_outer_splits)
            ),
            random_state=random_state,
        ),
        inner_cv=replace(
            config.reproduction.inner_cv,
            n_splits=(
                config.reproduction.inner_cv.n_splits
                if selections.reproduction_inner_splits is None
                else int(selections.reproduction_inner_splits)
            ),
            random_state=random_state,
        ),
        threshold_grid=replace(
            config.reproduction.threshold_grid,
            start=(
                config.reproduction.threshold_grid.start
                if selections.threshold_start is None
                else float(selections.threshold_start)
            ),
            stop=(
                config.reproduction.threshold_grid.stop
                if selections.threshold_stop is None
                else float(selections.threshold_stop)
            ),
            step=(
                config.reproduction.threshold_grid.step
                if selections.threshold_step is None
                else float(selections.threshold_step)
            ),
        ),
    )
    return replace(
        config,
        reproduction=reproduction,
        distillation_cv=replace(
            config.distillation_cv,
            n_splits=(
                config.distillation_cv.n_splits
                if selections.distillation_splits is None
                else int(selections.distillation_splits)
            ),
            random_state=random_state,
        ),
    )


class RunLauncher(ttk.Frame):
    def __init__(self, master: tk.Misc | None = None, *, initial_config_path: str = DEFAULT_CONFIG_PATH):
        super().__init__(master, padding=10)
        self.master = master
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._loaded_config: ExperimentConfig | None = None
        self._defaults: dict[str, object] = {}

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
        self.nested_cv_var = tk.BooleanVar(value=True)
        self.save_predictions_var = tk.BooleanVar(value=True)
        self.repro_outer_var = tk.StringVar(value="")
        self.repro_inner_var = tk.StringVar(value="")
        self.distill_folds_var = tk.StringVar(value="")
        self.random_state_var = tk.StringVar(value="")
        self.threshold_start_var = tk.StringVar(value="")
        self.threshold_stop_var = tk.StringVar(value="")
        self.threshold_step_var = tk.StringVar(value="")

        self.mt_target_family_var = tk.StringVar(value="v25_policies")
        self.mt_repeat_cv_var = tk.BooleanVar(value=True)
        self.mt_repeat_count_var = tk.StringVar(value="3")
        self.mt_nested_cv_var = tk.BooleanVar(value=False)
        self.mt_force_rebuild_var = tk.BooleanVar(value=False)
        self.mt_run_advanced_var = tk.BooleanVar(value=False)
        self.mt_embedding_model_var = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")

        self.feature_bank_vars: dict[str, tk.BooleanVar] = {}
        self.setup_model_vars: dict[str, tk.BooleanVar] = {
            "linear_l2": tk.BooleanVar(value=True),
            "xgb1": tk.BooleanVar(value=True),
        }
        self.mt_feature_set_vars: dict[str, tk.BooleanVar] = {}
        self.mt_model_family_vars: dict[str, tk.BooleanVar] = {
            "linear_l2": tk.BooleanVar(value=True),
            "xgb1": tk.BooleanVar(value=True),
            "mlp": tk.BooleanVar(value=True),
            "elasticnet": tk.BooleanVar(value=True),
            "randomforest": tk.BooleanVar(value=True),
        }

        self._build_ui()
        self._load_config_preview()
        self.after(100, self._poll_queue)

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

        self.setup_tab = ttk.Frame(notebook, padding=10)
        self.testing_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.setup_tab, text="Run Setup")
        notebook.add(self.testing_tab, text="Model Testing")
        self.setup_tab.columnconfigure(1, weight=1)
        self.testing_tab.columnconfigure(1, weight=1)

        self._build_setup_tab()
        self._build_testing_tab()

        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, sticky="ew", pady=(8, 8))
        actions.columnconfigure(4, weight=1)
        ttk.Button(actions, text="Reset Defaults", command=self._reset_defaults).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Run Setup Pipeline", command=self.start_run_setup).grid(row=0, column=1, padx=(8, 0), sticky="w")
        ttk.Button(actions, text="Run Model Testing", command=self.start_model_testing).grid(row=0, column=2, padx=(8, 0), sticky="w")
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=4, sticky="w")

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
        ttk.Checkbutton(flags, text="Use nested hyperparameter CV", variable=self.nested_cv_var).grid(row=3, column=0, sticky="w")
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
            text="XGB1",
            variable=self.setup_model_vars["xgb1"],
        ).grid(row=1, column=0, sticky="w")

        hyper = ttk.LabelFrame(self.setup_tab, text="CV / Threshold Overrides")
        hyper.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for index in range(4):
            hyper.columnconfigure(index, weight=1)
        ttk.Label(hyper, text="Repro outer").grid(row=0, column=0, sticky="w")
        ttk.Entry(hyper, textvariable=self.repro_outer_var, width=10).grid(row=1, column=0, sticky="w")
        ttk.Label(hyper, text="Repro inner").grid(row=0, column=1, sticky="w")
        ttk.Entry(hyper, textvariable=self.repro_inner_var, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(hyper, text="Distillation folds").grid(row=0, column=2, sticky="w")
        ttk.Entry(hyper, textvariable=self.distill_folds_var, width=10).grid(row=1, column=2, sticky="w")
        ttk.Label(hyper, text="Random state").grid(row=0, column=3, sticky="w")
        ttk.Entry(hyper, textvariable=self.random_state_var, width=10).grid(row=1, column=3, sticky="w")
        ttk.Label(hyper, text="Threshold start").grid(row=2, column=0, sticky="w")
        ttk.Entry(hyper, textvariable=self.threshold_start_var, width=10).grid(row=3, column=0, sticky="w")
        ttk.Label(hyper, text="Threshold stop").grid(row=2, column=1, sticky="w")
        ttk.Entry(hyper, textvariable=self.threshold_stop_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(hyper, text="Threshold step").grid(row=2, column=2, sticky="w")
        ttk.Entry(hyper, textvariable=self.threshold_step_var, width=10).grid(row=3, column=2, sticky="w")

    def _build_testing_tab(self) -> None:
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
        ttk.Label(self.testing_tab, textvariable=self.mt_target_preview_var, wraplength=760, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        candidates = ttk.LabelFrame(self.testing_tab, text="Candidate Feature Sets")
        candidates.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(8, 8))
        self.testing_tab.columnconfigure(0, weight=1)
        self.mt_candidates_frame = candidates

        families = ttk.LabelFrame(self.testing_tab, text="Model Families")
        families.grid(row=2, column=1, sticky="nsew", pady=(8, 8))
        for row, (key, label) in enumerate(
            [
                ("linear_l2", "Linear L2 (Ridge/LogReg)"),
                ("xgb1", "XGB1"),
                ("mlp", "MLP"),
                ("elasticnet", "ElasticNet"),
                ("randomforest", "RandomForest"),
            ]
        ):
            ttk.Checkbutton(families, text=label, variable=self.mt_model_family_vars[key]).grid(row=row, column=0, sticky="w")

        settings = ttk.LabelFrame(self.testing_tab, text="Settings")
        settings.grid(row=3, column=0, columnspan=2, sticky="ew")
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
        ttk.Checkbutton(settings, text="Use nested hyperparameter CV", variable=self.mt_nested_cv_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Checkbutton(settings, text="Run advanced models stage", variable=self.mt_run_advanced_var).grid(
            row=1, column=1, sticky="w"
        )
        ttk.Checkbutton(settings, text="Force rebuild intermediary banks", variable=self.mt_force_rebuild_var).grid(
            row=1, column=2, sticky="w"
        )
        ttk.Label(settings, text="Embedding model").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(settings, textvariable=self.mt_embedding_model_var).grid(row=2, column=1, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(
            settings,
            text="Screening is training-only: held-out/test features and labels are not used.",
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

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
        self.repro_outer_var.set(str(config.reproduction.outer_cv.n_splits))
        self.repro_inner_var.set(str(config.reproduction.inner_cv.n_splits))
        self.distill_folds_var.set(str(config.distillation_cv.n_splits))
        self.random_state_var.set(str(config.distillation_cv.random_state))
        self.threshold_start_var.set(str(config.reproduction.threshold_grid.start))
        self.threshold_stop_var.set(str(config.reproduction.threshold_grid.stop))
        self.threshold_step_var.set(str(config.reproduction.threshold_grid.step))
        self.repeat_cv_var.set(False)
        self.repeat_count_var.set("1")
        self.mt_repeat_cv_var.set(True)
        self.mt_repeat_count_var.set(str(config.model_testing.screening_repeat_cv_count))
        self.mt_run_advanced_var.set(bool(config.model_testing.run_advanced_models_default))

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

    def _reset_defaults(self) -> None:
        if self._loaded_config is None:
            return
        self._load_config_preview()
        self.status_var.set("Reset to defaults")

    def _rebuild_feature_bank_controls(self) -> None:
        for child in self.setup_features_frame.winfo_children():
            child.destroy()
        self.feature_bank_vars.clear()
        if self._loaded_config is None:
            return
        default_selected = {"hq_baseline", "llm_engineering", "lambda_policies", "sentence_prose", "sentence_structured"}
        row = 0
        for spec in [*self._loaded_config.repository_feature_banks, *self._loaded_config.intermediary_features]:
            if not spec.enabled:
                continue
            value = spec.feature_bank_id in default_selected
            var = tk.BooleanVar(value=value)
            self.feature_bank_vars[spec.feature_bank_id] = var
            ttk.Checkbutton(self.setup_features_frame, text=spec.feature_bank_id, variable=var).grid(
                row=row, column=0, sticky="w"
            )
            row += 1

    def _rebuild_testing_candidate_controls(self) -> None:
        for child in self.mt_candidates_frame.winfo_children():
            child.destroy()
        self.mt_feature_set_vars.clear()
        if self._loaded_config is None:
            return
        defaults = set(self._loaded_config.model_testing.candidate_feature_sets)
        allowed = {
            "hq_plus_sentence_bundle",
            "llm_engineering_plus_sentence_bundle",
            "lambda_policies_plus_sentence_bundle",
            "sentence_bundle",
        }
        row = 0
        for spec in self._loaded_config.distillation_feature_sets:
            feature_set_id = spec.feature_set_id
            if feature_set_id not in allowed:
                continue
            value = True if feature_set_id == "sentence_bundle" else (feature_set_id in defaults)
            var = tk.BooleanVar(value=value)
            self.mt_feature_set_vars[feature_set_id] = var
            ttk.Checkbutton(self.mt_candidates_frame, text=feature_set_id, variable=var).grid(
                row=row, column=0, sticky="w"
            )
            row += 1

    def _sync_setup_mode(self) -> None:
        is_distillation = self.run_mode_var.get() == "reasoning_distillation_mode"
        state = "!disabled" if is_distillation else "disabled"
        self.setup_target_combo.state([state])
        self.setup_repeat_entry.state([state if self.repeat_cv_var.get() else "disabled"])
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

    @staticmethod
    def _parse_optional_float(value: str) -> float | None:
        value = value.strip()
        return None if not value else float(value)

    def _build_common_kwargs(self) -> dict[str, object]:
        return {
            "config_path": self.config_path_var.get().strip(),
            "reproduction_outer_splits": self._parse_optional_int(self.repro_outer_var.get()),
            "reproduction_inner_splits": self._parse_optional_int(self.repro_inner_var.get()),
            "distillation_splits": self._parse_optional_int(self.distill_folds_var.get()),
            "cv_random_state": self._parse_optional_int(self.random_state_var.get()),
            "threshold_start": self._parse_optional_float(self.threshold_start_var.get()),
            "threshold_stop": self._parse_optional_float(self.threshold_stop_var.get()),
            "threshold_step": self._parse_optional_float(self.threshold_step_var.get()),
        }

    def _setup_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
        models: list[str] = []
        if self.setup_model_vars["linear_l2"].get():
            models.extend(["ridge", "logreg_classifier"])
        if self.setup_model_vars["xgb1"].get():
            models.extend(["xgb1_regressor", "xgb1_classifier"])
        return LauncherSelections(
            run_mode=self.run_mode_var.get().strip(),
            target_family=self.target_family_var.get().strip(),
            heldout_evaluation=bool(self.heldout_var.get()),
            active_feature_banks=[key for key, var in self.feature_bank_vars.items() if var.get()],
            force_rebuild_intermediary_features=bool(self.force_rebuild_var.get()),
            reasoning_models=models,
            embedding_model_name=self.embedding_model_var.get().strip() or None,
            repeat_cv_with_new_seeds=bool(self.repeat_cv_var.get()),
            cv_seed_repeat_count=self._parse_optional_int(self.repeat_count_var.get()),
            distillation_nested_sweep=bool(self.nested_cv_var.get()),
            save_reasoning_predictions=bool(self.save_predictions_var.get()),
            candidate_feature_sets=None,
            model_families=None,
            run_advanced_models=None,
            **common,
        )

    def _testing_selections(self) -> LauncherSelections:
        common = self._build_common_kwargs()
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
            distillation_nested_sweep=bool(self.mt_nested_cv_var.get()),
            save_reasoning_predictions=False,
            candidate_feature_sets=[key for key, var in self.mt_feature_set_vars.items() if var.get()],
            model_families=[key for key, var in self.mt_model_family_vars.items() if var.get()],
            run_advanced_models=bool(self.mt_run_advanced_var.get()),
            **common,
        )

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
            if selections.repeat_cv_with_new_seeds and (selections.cv_seed_repeat_count or 0) < 2:
                raise ValueError("Repeat count must be >= 2 when repeat CV is enabled.")
        except ValueError as exc:
            self.status_var.set("Invalid input")
            self._append_log(f"ERROR: {exc}")
            return
        self._start_worker(selections, "Starting model-testing pipeline.")

    def _start_worker(self, selections: LauncherSelections, start_message: str) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.status_var.set("Run already in progress")
            return
        self.status_var.set("Running")
        self.output_path_var.set("")
        self._append_log(start_message)
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
            elif event_type == "done":
                self.status_var.set("Complete")
                self.output_path_var.set(payload)
                self._append_log(f"Run finished. Artifacts: {payload}")
        self.after(100, self._poll_queue)


def launch_app(initial_config_path: str = DEFAULT_CONFIG_PATH) -> None:
    root = tk.Tk()
    root.title("Feature Repository Pipeline Launcher")
    root.geometry("1280x960")
    RunLauncher(root, initial_config_path=initial_config_path)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
