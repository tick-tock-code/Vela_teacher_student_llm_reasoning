from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import scrolledtext, ttk

from src.pipeline.config import load_experiment_config
from src.pipeline.distillation import run_reasoning_reconstruction
from src.pipeline.run_options import DEFAULT_CONFIG_PATH, RunOverrides


@dataclass(frozen=True)
class LauncherSelections:
    config_path: str
    run_reasoning_predictions: bool
    run_heldout_reasoning_predictions: bool
    run_success_predictions: bool
    active_intermediary_features: list[str]
    force_rebuild_intermediary_features: bool
    reasoning_models: list[str]
    embedding_model_name: str | None


def selections_to_overrides(selections: LauncherSelections) -> RunOverrides:
    return RunOverrides(
        config_path=selections.config_path,
        run_reasoning_predictions=selections.run_reasoning_predictions,
        run_heldout_reasoning_predictions=selections.run_heldout_reasoning_predictions,
        run_success_predictions=selections.run_success_predictions,
        active_intermediary_features=selections.active_intermediary_features,
        force_rebuild_intermediary_features=selections.force_rebuild_intermediary_features,
        reasoning_models=selections.reasoning_models,
        embedding_model_name=selections.embedding_model_name,
    )


class RunLauncher(ttk.Frame):
    def __init__(self, master: tk.Misc | None = None, *, initial_config_path: str = DEFAULT_CONFIG_PATH):
        super().__init__(master, padding=12)
        self.master = master
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.config_path_var = tk.StringVar(value=initial_config_path)
        self.embedding_model_var = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")
        self.run_reasoning_var = tk.BooleanVar(value=True)
        self.run_heldout_reasoning_var = tk.BooleanVar(value=False)
        self.run_success_var = tk.BooleanVar(value=False)
        self.force_rebuild_var = tk.BooleanVar(value=False)
        self.feature_vars = {
            "mirror": tk.BooleanVar(value=True),
            "sentence_prose": tk.BooleanVar(value=True),
            "sentence_structured": tk.BooleanVar(value=True),
            "llm_engineered": tk.BooleanVar(value=False),
        }
        self.model_vars = {
            "ridge": tk.BooleanVar(value=True),
            "xgb1_regressor": tk.BooleanVar(value=True),
        }
        self.selected_targets_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.output_path_var = tk.StringVar(value="")

        self._build_widgets()
        self._load_config_preview()
        self.after(100, self._poll_queue)

    def _build_widgets(self) -> None:
        self.grid(sticky="nsew")
        if self.master is not None:
            self.master.columnconfigure(0, weight=1)
            self.master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        config_frame = ttk.LabelFrame(self, text="Config")
        config_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        config_frame.columnconfigure(1, weight=1)
        ttk.Label(config_frame, text="Config path").grid(row=0, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.config_path_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 8)
        )
        ttk.Button(config_frame, text="Reload", command=self._load_config_preview).grid(row=0, column=2, sticky="e")

        target_frame = ttk.LabelFrame(self, text="Selected Targets")
        target_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            target_frame,
            textvariable=self.selected_targets_var,
            justify="left",
            wraplength=780,
        ).grid(row=0, column=0, sticky="w")

        feature_frame = ttk.LabelFrame(self, text="Intermediary Features")
        feature_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.feature_controls = {
            "mirror": ttk.Checkbutton(feature_frame, text="Mirror", variable=self.feature_vars["mirror"]),
            "sentence_prose": ttk.Checkbutton(
                feature_frame,
                text="Sentence-transformer prose",
                variable=self.feature_vars["sentence_prose"],
            ),
            "sentence_structured": ttk.Checkbutton(
                feature_frame,
                text="Sentence-transformer structured",
                variable=self.feature_vars["sentence_structured"],
            ),
            "llm_engineered": ttk.Checkbutton(
                feature_frame,
                text="LLM-engineered (inactive)",
                variable=self.feature_vars["llm_engineered"],
                state="disabled",
            ),
        }
        for row_index, key in enumerate(["mirror", "sentence_prose", "sentence_structured", "llm_engineered"]):
            self.feature_controls[key].grid(row=row_index, column=0, sticky="w")

        model_frame = ttk.LabelFrame(self, text="Reasoning Models")
        model_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Checkbutton(model_frame, text="Ridge", variable=self.model_vars["ridge"]).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            model_frame,
            text="XGB1 regressor",
            variable=self.model_vars["xgb1_regressor"],
        ).grid(row=1, column=0, sticky="w")

        options_frame = ttk.LabelFrame(self, text="Run Options")
        options_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        options_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            options_frame,
            text="Run reasoning predictions",
            variable=self.run_reasoning_var,
        ).grid(row=0, column=0, sticky="w", columnspan=2)
        ttk.Checkbutton(
            options_frame,
            text="Run held-out reasoning predictions",
            variable=self.run_heldout_reasoning_var,
        ).grid(row=1, column=0, sticky="w", columnspan=2)
        self.run_success_control = ttk.Checkbutton(
            options_frame,
            text="Run success predictions (inactive)",
            variable=self.run_success_var,
            state="disabled",
        )
        self.run_success_control.grid(row=2, column=0, sticky="w", columnspan=2)
        ttk.Checkbutton(
            options_frame,
            text="Force rebuild intermediary features",
            variable=self.force_rebuild_var,
        ).grid(row=3, column=0, sticky="w", columnspan=2)
        ttk.Label(options_frame, text="Embedding model").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(options_frame, textvariable=self.embedding_model_var).grid(
            row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )

        action_frame = ttk.Frame(self)
        action_frame.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        action_frame.columnconfigure(1, weight=1)
        ttk.Button(action_frame, text="Run", command=self.start_run).grid(row=0, column=0, sticky="w")
        ttk.Label(action_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=(12, 0))

        output_frame = ttk.LabelFrame(self, text="Output")
        output_frame.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(output_frame, textvariable=self.output_path_var, wraplength=780, justify="left").grid(
            row=0, column=0, sticky="w"
        )

        log_frame = ttk.LabelFrame(self, text="Status Log")
        log_frame.grid(row=7, column=0, sticky="nsew")
        self.rowconfigure(7, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=16, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def _load_config_preview(self) -> None:
        config = load_experiment_config(self.config_path_var.get())
        self.selected_targets_var.set(", ".join(item.target_id for item in config.reasoning_targets))
        for spec in config.intermediary_features:
            if spec.kind.startswith("sentence_transformer") and spec.embedding_model_name:
                self.embedding_model_var.set(spec.embedding_model_name)
                break

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def get_selections(self) -> LauncherSelections:
        active_features = [
            feature_id
            for feature_id, variable in self.feature_vars.items()
            if variable.get()
        ]
        active_models = [
            model_id
            for model_id, variable in self.model_vars.items()
            if variable.get()
        ]
        return LauncherSelections(
            config_path=self.config_path_var.get().strip(),
            run_reasoning_predictions=bool(self.run_reasoning_var.get()),
            run_heldout_reasoning_predictions=bool(self.run_heldout_reasoning_var.get()),
            run_success_predictions=bool(self.run_success_var.get()),
            active_intermediary_features=active_features,
            force_rebuild_intermediary_features=bool(self.force_rebuild_var.get()),
            reasoning_models=active_models,
            embedding_model_name=self.embedding_model_var.get().strip() or None,
        )

    def start_run(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.status_var.set("Run already in progress")
            return

        self.status_var.set("Running")
        self.output_path_var.set("")
        self._append_log("Starting reasoning-reconstruction run.")

        selections = self.get_selections()
        overrides = selections_to_overrides(selections)

        def worker() -> None:
            try:
                config = load_experiment_config(overrides.config_path)
                run_dir = run_reasoning_reconstruction(
                    config,
                    overrides,
                    logger=lambda message: self.queue.put(("log", message)),
                )
            except Exception as exc:  # pragma: no cover - exercised by manual UI use
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
    root.title("Reasoning Reconstruction Launcher")
    root.geometry("900x760")
    RunLauncher(root, initial_config_path=initial_config_path)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
