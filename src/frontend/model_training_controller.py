"""Model definition, queue management, training, and import behavior."""

import json
from pathlib import Path

import joblib
from PySide6.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QWidget
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.process import (
    create_model_package,
    make_json_safe,
    unpack_model_package,
)
from src.frontend.unified_model_panel import UnifiedModelPage
from src.frontend.workers import UnifiedModelTrainingWorker
from src.model.model_controller import ModelController


class ModelTrainingControllerMixin:
    """Model workflow behavior mixed into the main application window."""

    def _build_models_page(self):
        # page = QWidget()
        # layout = QVBoxLayout(page)
        # layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(12)

        # self.models_title = section_label("SAVED MODELS")
        # self.models_model = PandasTableModel(pd.DataFrame())
        # self.models_table = table_view(self.models_model)
        # layout.addWidget(self.models_title)
        # layout.addWidget(self.models_table, 1)
        # self.main_stack.addWidget(page)
        self.model_page = QWidget()
        layout = QVBoxLayout(self.model_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.unified_model_page = UnifiedModelPage()
        self.unified_model_page.model_action_requested.connect(self.on_added_model_action)
        self.unified_model_page.queue_add_requested.connect(self.add_models_to_queue)
        self.unified_model_page.queue_remove_requested.connect(self.remove_models_from_queue)
        self.unified_model_page.queue_reordered.connect(self.on_queue_reordered)
        self.unified_model_page.train_queue_requested.connect(self.train_model_queue)
        layout.addWidget(self.unified_model_page, 1)
        self.model_page_scroll = self._wrap_main_page(self.model_page)
        self.main_stack.addWidget(self.model_page_scroll)

    def on_model_tab_changed(self, index):
        del index
        self.refresh_model_page()

    def on_model_category_selected(self, category):
        if str(category).strip().lower() != "semi_supervised":
            return

        if self.project is None or self.working_df is None or self.working_df.empty:
            QMessageBox.warning(
                self,
                "Semi-Supervised Dataset",
                "Import a dataset before configuring a semi-supervised model.",
            )
            return

        label = str(self.project.get("label_column", "") or "").strip()
        if not label or label not in self.working_df.columns:
            QMessageBox.warning(
                self,
                "Semi-Supervised Dataset",
                "Select a valid label column before configuring a semi-supervised model.",
            )
            return

        labels = self.working_df[label]
        missing_mask = labels.isna()
        try:
            blank_mask = labels.astype("string").str.strip().eq("").fillna(False)
            missing_mask = missing_mask | blank_mask
        except Exception:
            pass

        missing_count = int(missing_mask.sum())
        labeled_count = int((~missing_mask).sum())

        if labeled_count == 0:
            QMessageBox.warning(
                self,
                "Semi-Supervised Dataset",
                f"The label column '{label}' has no known labels. Semi-supervised training needs both labeled and unlabeled rows.",
            )
            return

        if missing_count == 0:
            QMessageBox.warning(
                self,
                "Semi-Supervised Dataset",
                f"The label column '{label}' has no missing labels. Semi-supervised training requires both labeled and unlabeled rows. Add missing values to the label column before adding or training this model.",
            )
            return

    def refresh_model_page(self):
        """Synchronize Added Models and Queue using project-backed state."""
        if self.project is None:
            self.unified_model_page.set_added_models([])
            self.unified_model_page.set_queue([])
            self.model_sidebar.set_project_label("")
            return

        ModelController.ensure_project_state(self.project)
        self.model_sidebar.set_project_label(self.project.get("label_column", ""))
        trained_supervised = [
            {
                "id": model.get("display_name", ""),
                "name": model.get("display_name", ""),
                "label": str(model.get("label") or self.project.get("label_column", "")),
                "source": "saved",
            }
            for model in self.project.get("models", [])
            if (
                model.get("category") == "supervised"
                and model.get("display_name")
                and model.get("model") is not None
                and (
                    not str(model.get("label", "")).strip()
                    or str(model.get("label", "")).strip()
                    == str(self.project.get("label_column", "") or "").strip()
                )
                and (
                    not (model.get("input_feature_columns") or model.get("feature_columns"))
                    or self.working_df.empty
                    or all(
                        str(col) in self.working_df.columns
                        for col in (
                            model.get("input_feature_columns")
                            or model.get("feature_columns", [])
                        )
                    )
                )
            )
        ]

        exported_dir = Path("ExportedModels")
        if exported_dir.exists():
            for pkl_path in sorted(exported_dir.glob("*.pkl")):
                try:
                    payload = joblib.load(pkl_path)
                    model_obj, metadata, _ = unpack_model_package(payload)
                except Exception:
                    continue
                if model_obj is None:
                    continue
                if str(metadata.get("category", "")).strip().lower() != "supervised":
                    continue
                feature_columns = [str(col) for col in metadata.get("feature_columns", []) or []]
                input_feature_columns = [
                    str(col)
                    for col in (
                        metadata.get("input_feature_columns")
                        or feature_columns
                    )
                ]
                if input_feature_columns and not self.working_df.empty and any(
                    col not in self.working_df.columns for col in input_feature_columns
                ):
                    continue
                model_label = str(metadata.get("label", "")).strip()
                project_label = str(self.project.get("label_column", "") or "").strip()
                if model_label and project_label and model_label != project_label:
                    continue
                trained_supervised.append(
                    {
                        "id": str(pkl_path.resolve()),
                        "name": str(metadata.get("display_name") or pkl_path.stem),
                        "label": model_label,
                        "source": "exported",
                    }
                )
        self.model_sidebar.set_trained_supervised_models(trained_supervised)
        self.unified_model_page.set_added_models(self.project.get("added_models", []))
        self.unified_model_page.set_queue(ModelController.queue_rows(self.project))

    def add_or_update_model_definition(self, payload):
        if self.project is None:
            QMessageBox.warning(self, "No Project", "Create or open a project before adding models.")
            return

        ModelController.ensure_project_state(self.project)

        if str(payload.get("category", "")).strip().lower() == "semi_supervised":
            label = str(self.project.get("label_column", "") or "").strip()
            if self.working_df is None or self.working_df.empty or not label or label not in self.working_df.columns:
                QMessageBox.warning(
                    self,
                    "Semi-Supervised Dataset",
                    "Semi-supervised models require a dataset with a valid label column.",
                )
                return

            labels = self.working_df[label]
            missing_mask = labels.isna()
            try:
                blank_mask = labels.astype("string").str.strip().eq("").fillna(False)
                missing_mask = missing_mask | blank_mask
            except Exception:
                pass

            labeled_count = int((~missing_mask).sum())
            missing_count = int(missing_mask.sum())

            if labeled_count == 0:
                QMessageBox.warning(
                    self,
                    "Semi-Supervised Dataset",
                    f"The label column '{label}' needs at least one known label before a semi-supervised model can be added.",
                )
                return

            if missing_count == 0:
                QMessageBox.warning(
                    self,
                    "Semi-Supervised Dataset",
                    f"The label column '{label}' needs at least one missing label before a semi-supervised model can be added.",
                )
                return

        added_models = self.project.setdefault("added_models", [])
        existing_names = [item.get("name", "") for item in added_models]

        original_name = payload.get("original_name")
        desired_name = payload.get("name", "").strip()
        if not desired_name:
            QMessageBox.warning(self, "Invalid Name", "Enter a model name.")
            return

        if original_name:
            existing = ModelController.find_added_model(self.project, original_name)
            if existing is None:
                QMessageBox.warning(self, "Model Missing", f"Could not find '{original_name}' to edit.")
                return

            if desired_name != original_name:
                without_original = [name for name in existing_names if name != original_name]
                desired_name = ModelController.unique_name(desired_name, without_original)

            existing.update({
                "name": desired_name,
                "category": payload.get("category", "supervised"),
                "algorithm": payload.get("algorithm", "svm"),
                "label": self.project.get("label_column", ""),
                "common_parameters": payload.get("common_parameters", {}),
                "required_parameters": payload.get("required_parameters", {}),
                "advanced_parameters": payload.get("advanced_parameters", {}),
                "trained": False,
                "externally_added": existing.get("externally_added", False),
                "editable_external": existing.get("editable_external", True),
            })

            self.project["models"] = [
                model for model in self.project.get("models", [])
                if model.get("display_name") not in {original_name, desired_name}
            ]

            queue = self.project.setdefault("model_queue", [])
            self.project["model_queue"] = [desired_name if name == original_name else name for name in queue]
        else:
            unique_name = ModelController.unique_name(desired_name, existing_names)
            entry = ModelController.create_added_model_entry(
                name=unique_name,
                category=payload.get("category", "supervised"),
                algorithm=payload.get("algorithm", "svm"),
                label=str(self.project.get("label_column", "")),
                common_parameters=payload.get("common_parameters", {}),
                required_parameters=payload.get("required_parameters", {}),
                advanced_parameters=payload.get("advanced_parameters", {}),
            )
            entry["externally_added"] = False
            entry["editable_external"] = True
            added_models.append(entry)

        self._set_dirty(True)
        self.model_sidebar.reset_form()
        self.refresh_model_page()

    def on_added_model_action(self, action, name):
        if self.project is None:
            return

        entry = ModelController.find_added_model(self.project, name)
        if entry is None:
            QMessageBox.warning(self, "Model Missing", f"Could not find '{name}'.")
            return

        if action == "inspect":
            saved = next((model for model in self.project.get("models", []) if model.get("display_name") == name), None)
            columns = entry.get("feature_columns", [])
            if not columns and saved:
                columns = saved.get("feature_columns", [])
            if not columns:
                columns = self.project.get("selected_columns", [])

            common_text = "\n".join([f"{k}: {v}" for k, v in entry.get("common_parameters", {}).items()]) or "None"
            required_text = "\n".join([f"{k}: {v}" for k, v in entry.get("required_parameters", {}).items()]) or "None"
            advanced_text = "\n".join([f"{k}: {v}" for k, v in entry.get("advanced_parameters", {}).items()]) or "None"
            inspect_text = (
                f"Name: {entry.get('name', '')}\n"
                f"Category: {entry.get('category', '')}\n"
                f"Algorithm: {entry.get('algorithm', '')}\n"
                f"Label: {entry.get('label', '')}\n"
                f"Trained: {'Yes' if entry.get('trained') else 'No'}\n"
                f"Added Externally: {'Yes' if entry.get('externally_added') else 'No'}\n\n"
                f"Columns:\n{', '.join(columns) if columns else 'None'}\n\n"
                f"Common Parameters:\n{common_text}\n\n"
                f"Required Parameters:\n{required_text}\n\n"
                f"Advanced Parameters:\n{advanced_text}"
            )
            QMessageBox.information(self, "Model Inspect", inspect_text)
            return

        if action == "edit":
            if entry.get("externally_added") and not entry.get("editable_external", False):
                QMessageBox.warning(
                    self,
                    "External Model",
                    "This model was added externally and cannot be edited for this dataset.",
                )
                return
            if entry.get("trained"):
                proceed = QMessageBox.question(
                    self,
                    "Edit Trained Model",
                    "Editing this model will mark it as not trained. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if proceed != QMessageBox.StandardButton.Yes:
                    return
            self.model_sidebar.load_for_edit(entry)
            return

        if action == "duplicate":
            existing_names = [item.get("name", "") for item in self.project.get("added_models", [])]
            duplicate = ModelController.duplicate_entry(entry, existing_names)
            self.project.setdefault("added_models", []).append(duplicate)
            self._set_dirty(True)
            self.refresh_model_page()
            return

        if action == "delete":
            confirm = QMessageBox.question(
                self,
                "Delete Model",
                f"Delete '{name}' from Added Models?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            self.project["added_models"] = [item for item in self.project.get("added_models", []) if item.get("name") != name]
            self.project["model_queue"] = [item for item in self.project.get("model_queue", []) if item != name]
            self.project["models"] = [model for model in self.project.get("models", []) if model.get("display_name") != name]
            self._set_dirty(True)
            self.refresh_model_page()
            return

        if action == "export_json":
            export_dir = Path("ExportedModels")
            export_dir.mkdir(parents=True, exist_ok=True)
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Model Config",
                str(export_dir / f"{name}.json"),
                "JSON Files (*.json)",
            )
            if not path:
                return
            category = entry.get("category", "")
            algorithm = entry.get("algorithm", "")
            common_parameters = ModelController.default_common_parameters()
            common_parameters.update(entry.get("common_parameters", {}))
            required_parameters = ModelController.default_required_parameters(category, algorithm)
            required_parameters.update(entry.get("required_parameters", {}))
            advanced_parameters = ModelController.default_advanced_parameters(category, algorithm)
            advanced_parameters.update(entry.get("advanced_parameters", {}))
            saved = next((model for model in self.project.get("models", []) if model.get("display_name") == name), {})
            evaluation = saved.get("evaluation", {}) if saved else {}

            export_payload = {
                "name": entry.get("name", ""),
                "category": category,
                "algorithm": algorithm,
                "label": entry.get("label", ""),
                "trained": bool(entry.get("trained", False)),
                "externally_added": bool(entry.get("externally_added", False)),
                "editable_external": bool(entry.get("editable_external", True)),
                "common_parameters": common_parameters,
                "required_parameters": required_parameters,
                "advanced_parameters": advanced_parameters,
                "training_parameters": {**required_parameters, **advanced_parameters},
                "feature_columns": saved.get("feature_columns", entry.get("feature_columns", [])),
                "input_feature_columns": saved.get(
                    "input_feature_columns",
                    entry.get("input_feature_columns", []),
                ),
                "preprocessing": saved.get(
                    "preprocessing",
                    entry.get("preprocessing", {}),
                ),
                "metrics": saved.get("metrics", entry.get("metrics", {})),
                "confusion_matrix": evaluation.get("confusion_matrix"),
                "confusion_labels": evaluation.get("confusion_labels"),
                "cluster_summary": evaluation.get("cluster_summary"),
                "cluster_plot_data": evaluation.get("cluster_plot_data"),
                "cluster_plot_components": evaluation.get(
                    "cluster_plot_components"
                ),
                "ssl_progress": evaluation.get("ssl_progress"),
                "ssl_iteration_progress": evaluation.get("ssl_iteration_progress"),
            }
            try:
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(make_json_safe(export_payload), file, indent=2)
            except Exception as error:
                self.show_error("Export Error", error)
            return

        if action == "export_pkl":
            if not entry.get("trained"):
                QMessageBox.warning(self, "Not Trained", "Model must be trained before exporting PKL.")
                return
            saved = next((model for model in self.project.get("models", []) if model.get("display_name") == name), None)
            if saved is None or "model" not in saved:
                QMessageBox.warning(self, "Model Missing", "Trained model artifact was not found.")
                return
            export_dir = Path("ExportedModels")
            export_dir.mkdir(parents=True, exist_ok=True)
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Trained Model",
                str(self.downloads_dir / f"{name}.pkl"),
                "Pickle Files (*.pkl)",
            )
            if not path:
                return
            try:
                # Export one self-contained PKL package
                package = create_model_package(
                    saved,
                    added_entry=entry,
                    label=entry.get("label", self.project.get("label_column", "")),
                )
                joblib.dump(package, path)
            except Exception as error:
                self.show_error("Export Error", error)

    def add_models_to_queue(self, names):
        if self.project is None or not names:
            return
        ModelController.ensure_project_state(self.project)
        queue = self.project.setdefault("model_queue", [])
        skipped_trained = []
        for name in names:
            added = ModelController.find_added_model(self.project, name)
            if added and added.get("trained"):
                skipped_trained.append(name)
                continue
            if name not in queue:
                queue.append(name)
        if skipped_trained:
            QMessageBox.information(
                self,
                "Queue Update",
                "These models were skipped because they are already trained:\n"
                + "\n".join(skipped_trained),
            )
        self._set_dirty(True)
        self.refresh_model_page()

    def remove_models_from_queue(self, names):
        if self.project is None or not names:
            return
        self.project["model_queue"] = [name for name in self.project.get("model_queue", []) if name not in set(names)]
        self._set_dirty(True)
        self.refresh_model_page()

    def on_queue_reordered(self, ordered_names):
        if self.project is None:
            return

        current_queue = list(self.project.get("model_queue", []))
        proposed_queue = [str(name) for name in ordered_names if name]

        # Ignore malformed reorder payloads so queue items are never dropped by
        # a widget drag/drop edge case.
        if not proposed_queue:
            return
        if len(proposed_queue) != len(current_queue):
            return
        if set(proposed_queue) != set(current_queue):
            return

        self.project["model_queue"] = proposed_queue
        self._set_dirty(True)
        self.refresh_model_page()

    def train_model_queue(self):
        if self.project is None:
            QMessageBox.warning(self, "No Project", "Create or open a project before training.")
            return
        ModelController.ensure_project_state(self.project)
        queue = list(self.project.get("model_queue", []))
        if not queue:
            QMessageBox.warning(self, "Empty Queue", "Add at least one model to the queue.")
            return

        if self.working_df.empty:
            QMessageBox.warning(self, "No Working Data", "Training queue requires the working dataframe from Data & Features.")
            return

        decision = QMessageBox.question(
            self,
            "Parallel Processing",
            "Run queue in parallel?\n\nParallel processing may increase CPU usage significantly.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if decision == QMessageBox.StandardButton.Cancel:
            return
        use_parallel = decision == QMessageBox.StandardButton.Yes

        self._queue_parallel = bool(use_parallel)
        self._queue_pending = list(queue)
        self._queue_failures = []
        self._queue_total = len(queue)
        self.unified_model_page.set_training(True)

        if self._queue_parallel:
            for name in list(self._queue_pending):
                self._start_queue_worker(name)
        else:
            self._start_next_serial_worker()

    def _start_next_serial_worker(self):
        if not getattr(self, "_queue_pending", []):
            self._finish_queue_training()
            return
        self._start_queue_worker(self._queue_pending[0])

    def _start_queue_worker(self, name):
        entry = ModelController.find_added_model(self.project, name)
        if entry is None:
            self._queue_failures.append((name, "Model entry was not found."))
            if name in self._queue_pending:
                self._queue_pending.remove(name)
            if self._queue_parallel:
                if not self._queue_pending:
                    self._finish_queue_training()
            else:
                self._start_next_serial_worker()
            return

        label = self.project.get("label_column") or self.label_combo.currentText()
        worker = UnifiedModelTrainingWorker(
            dataframe=self.working_df,
            label_column=label,
            added_model_entry=entry,
            saved_models=self.project.get("models", []),
        )
        worker.signals.finished.connect(self.on_queue_model_trained)
        worker.signals.error.connect(lambda error, n=name: self.on_queue_model_error(n, error))
        self.thread_pool.start(worker)

    def on_queue_model_trained(self, payload):
        name = payload["name"]
        snapshot = self._normalize_evaluation_snapshot(payload.get("result", {}))
        self.project["models"] = [
            model for model in self.project.get("models", [])
            if model.get("display_name") != name
        ]
        self.project.setdefault("models", []).append({
            "display_name": name,
            "category": payload.get("category", ""),
            "algorithm": payload["algorithm"],
            "label": (
                (ModelController.find_added_model(self.project, name) or {}).get("label")
                or self.project.get("label_column", "")
            ),
            "model": payload["trained_model"],
            "parameters": payload["parameters"],
            "metrics": payload.get("metrics", {}),
            "feature_columns": payload.get("feature_columns", []),
            "input_feature_columns": payload.get("input_feature_columns", []),
            "preprocessing": payload.get("preprocessing", {}),
            "evaluation": snapshot,
        })

        entry = ModelController.find_added_model(self.project, name)
        if entry is not None:
            entry["trained"] = True
            entry["feature_columns"] = payload.get("feature_columns", [])
            entry["input_feature_columns"] = payload.get("input_feature_columns", [])
            entry["preprocessing"] = payload.get("preprocessing", {})
            entry["metrics"] = payload.get("metrics", {})
            entry["evaluation"] = snapshot

        self.project["model_queue"] = [item for item in self.project.get("model_queue", []) if item != name]
        if name in self._queue_pending:
            self._queue_pending.remove(name)

        self._set_dirty(True)
        self.refresh_model_page()
        self.refresh_results_page()
        if self._queue_parallel:
            if not self._queue_pending:
                self._finish_queue_training()
        else:
            self._start_next_serial_worker()

    def on_queue_model_error(self, name, error):
        self._queue_failures.append((name, str(error)))
        self.project["model_queue"] = [item for item in self.project.get("model_queue", []) if item != name]
        if name in self._queue_pending:
            self._queue_pending.remove(name)
        self._set_dirty(True)
        self.refresh_model_page()
        if self._queue_parallel:
            if not self._queue_pending:
                self._finish_queue_training()
        else:
            self._start_next_serial_worker()

    def _finish_queue_training(self):
        self.unified_model_page.set_training(False)
        failed = len(getattr(self, "_queue_failures", []))
        trained = int(getattr(self, "_queue_total", 0)) - failed
        if failed == 0:
            QMessageBox.information(self, "Training Complete", f"Successfully trained {trained} queued model(s).")
            return

        details = "\n".join([f"{name}: {message}" for name, message in self._queue_failures])
        QMessageBox.warning(
            self,
            "Training Completed With Errors",
            f"Trained {trained} model(s), failed {failed}.\n\n{details}",
        )

    def import_external_model(self):
        if self.project is None:
            QMessageBox.warning(self, "No Project", "Create or open a project before importing external models.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import External PKL Model",
            str(self.downloads_dir),
            "Pickle Files (*.pkl)",
        )
        if not path:
            return

        try:
            imported_payload = joblib.load(path)
        except Exception as error:
            self.show_error("Import Error", error)
            return

        # Exports are self-contained PKL packages
        imported, package_metadata, _ = unpack_model_package(imported_payload)
        if imported is None:
            QMessageBox.warning(self, "Import Error", "The PKL did not contain a model estimator.")
            return

        saved_metrics = package_metadata.get("metrics", {}) or {}
        saved_parameters = package_metadata.get("parameters", {}) or {}
        saved_features = package_metadata.get("feature_columns", []) or []
        saved_input_features = package_metadata.get("input_feature_columns", []) or []
        saved_preprocessing = package_metadata.get("preprocessing", {}) or {}
        saved_evaluation = package_metadata.get("evaluation", {}) or {}
        saved_evaluation["metrics"] = saved_evaluation.get("metrics") or saved_metrics
        saved_common = package_metadata.get("common_parameters", {}) or {}
        saved_required = package_metadata.get("required_parameters", {}) or {}
        saved_advanced = package_metadata.get("advanced_parameters", {}) or {}


        category, algorithm = self._infer_external_model_category_algorithm(imported)
        category = str(package_metadata.get("category") or category or "")
        algorithm = str(package_metadata.get("algorithm") or algorithm or "")
        base_name = str(package_metadata.get("display_name") or Path(path).stem)
        existing_names = [item.get("name", "") for item in self.project.get("added_models", [])]
        name = ModelController.unique_name(base_name, existing_names)

        fallback_category = category or "supervised"
        fallback_algorithm = algorithm or "svm"
        entry = ModelController.create_added_model_entry(
            name=name,
            category=fallback_category,
            algorithm=fallback_algorithm,
            label=package_metadata.get("label") or self.project.get("label_column", ""),
            common_parameters=ModelController.default_common_parameters(),
            required_parameters=ModelController.default_required_parameters(fallback_category, fallback_algorithm),
            advanced_parameters=ModelController.default_advanced_parameters(fallback_category, fallback_algorithm),
        )
        entry["trained"] = True
        entry["externally_added"] = True
        feature_columns = []
        if hasattr(imported, "feature_names_in_"):
            feature_columns = [str(col) for col in list(imported.feature_names_in_)]
        if saved_features:
            feature_columns = [str(col) for col in saved_features]
        entry["feature_columns"] = feature_columns
        entry["input_feature_columns"] = [
            str(col) for col in (saved_input_features or feature_columns)
        ]
        entry["preprocessing"] = saved_preprocessing
        entry["metrics"] = saved_metrics
        saved_evaluation["metrics"] = saved_evaluation.get("metrics") or saved_metrics
        entry["evaluation"] = saved_evaluation
        if saved_common:
            entry["common_parameters"] = saved_common
        if saved_required:
            entry["required_parameters"] = saved_required
        if saved_advanced:
            entry["advanced_parameters"] = saved_advanced
        editable = bool(category and algorithm and self.project.get("label_column"))
        editable_features = entry.get("input_feature_columns") or feature_columns
        if editable and editable_features and not self.working_df.empty:
            editable = all(column in self.working_df.columns for column in editable_features)
        entry["editable_external"] = editable

        self.project.setdefault("added_models", []).append(entry)
        self.project.setdefault("models", []).append({
            "display_name": name,
            "category": fallback_category,
            "algorithm": algorithm or "external",
            "label": entry.get("label", ""),
            "model": imported,
            "parameters": saved_parameters,
            "metrics": saved_metrics,
            "feature_columns": feature_columns,
            "input_feature_columns": entry.get("input_feature_columns", []),
            "preprocessing": saved_preprocessing,
            "evaluation": saved_evaluation,
        })

        self._set_dirty(True)
        self.refresh_model_page()
        QMessageBox.information(
            self,
            "External Model Imported",
            "Model added as externally imported."
            + (" It is editable because it matches a supported workflow." if editable else " It is inspect-only for this dataset."),
        )

    def _infer_external_model_category_algorithm(self, model):
        mappings = [
            (SVC, ("supervised", "svm")),
            (KNeighborsClassifier, ("supervised", "knn")),
            (DecisionTreeClassifier, ("supervised", "decision_tree")),
            (RandomForestClassifier, ("supervised", "random_forest")),
            (LogisticRegression, ("supervised", "logistic_regression")),
            (SelfTrainingClassifier, ("semi_supervised", "self_training")),
            (KMeans, ("unsupervised", "kmeans")),
            (DBSCAN, ("unsupervised", "dbscan")),
            (AgglomerativeClustering, ("unsupervised", "hierarchical")),
        ]
        for cls, values in mappings:
            if isinstance(model, cls):
                return values
        return None, None
