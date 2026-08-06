"""Project creation and persistence behavior."""

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox


def _save_project(*args, **kwargs):
    """Preserve the historical main_window.save_project patch point."""
    from src.frontend import main_window

    return main_window.save_project(*args, **kwargs)


class ProjectControllerMixin:
    """Project persistence behavior mixed into the main application window."""

    def create_project(self):
        """Persist the selected dataset configuration as an ICP project."""
        if self.og_df.empty:
            QMessageBox.warning(self, "Missing Dataset", "Open a dataset before creating a project.")
            return False

        project_name = self.project_name.text().strip()
        if not project_name:
            QMessageBox.warning(self, "Missing Project Name", "Enter a project name.")
            return False

        columns = self.selected_columns()
        if not columns:
            QMessageBox.warning(self, "Missing Columns", "Select at least one predictor or response column.")
            return False

        label_col = self.get_selected_label()
        try:
            if self.working_df.empty:
                self.working_df = self.og_df[columns].copy()
            else:
                self.working_df = self.working_df.reindex(columns=columns).copy()

            if self.project is None:
                # Keep project metadata compatible with src.data.process.save_project().
                self.project = {
                    "project_name": project_name,
                    "file_path": str(self.file_path),
                    "dataset": self.dataset,
                    "selected_columns": list(self.working_df.columns),
                    "label_column": label_col,
                    "column_types": {
                        col: str(self.working_df[col].dtype)
                        for col in self.working_df.columns
                    },
                    "preprocessing": [],
                    "visualizations": [],
                    "models": [],
                    "added_models": [],
                    "model_queue": [],
                }
                base_dir = self.current_project_path.parent if self.current_project_path else self.downloads_dir
                target_path = base_dir / f"{project_name}.icp"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self.current_project_path = target_path

                _save_project(
                    self.project,
                    self.og_df,
                    self.working_df,
                    str(self.current_project_path),
                    feature_df=self.feature_df.copy(),
                )
                self._active_project_name = project_name
                self.project["project_name"] = self._active_project_name
                self._set_dirty(False)
            else:
                active_name = str(
                    self._active_project_name
                    or self.project.get("project_name")
                    or (self.current_project_path.stem if self.current_project_path else "project")
                ).strip()
                self._active_project_name = active_name
                self.project["project_name"] = active_name
                self.project["selected_columns"] = list(self.working_df.columns)
                self.project["label_column"] = label_col
                self.project["column_types"] = {
                    col: str(self.working_df[col].dtype)
                    for col in self.working_df.columns
                }
                self.project["file_path"] = str(self.file_path) if self.file_path is not None else self.project.get("file_path")

                if project_name == active_name:
                    # Do not create a duplicate copy with the currently active name.
                    return self.save_current_project()

                base_dir = self.current_project_path.parent if self.current_project_path else self.downloads_dir
                copy_path = base_dir / f"{project_name}.icp"
                copy_path.parent.mkdir(parents=True, exist_ok=True)
                project_copy = dict(self.project)
                project_copy["project_name"] = project_name
                _save_project(
                    project_copy,
                    self.og_df.copy(),
                    self.working_df.copy(),
                    str(copy_path),
                    feature_df=self.feature_df.copy(),
                )

                # Continue on the active project after creating the copy.
                self._suppress_dirty = True
                try:
                    self.project_name.setText(active_name)
                finally:
                    self._suppress_dirty = False
                self._set_dirty(False)
                QMessageBox.information(
                    self,
                    "Project Copy Created",
                    f"Saved copy as {copy_path}. Continuing with {active_name}.",
                )
                return True
        except Exception as error:
            self.show_error("Save Error", error)
            return False

        self.refresh_import_tables()
        QMessageBox.information(self, "Project Created", f"Saved {self.current_project_path}")
        return True

    def save_current_project(self):
        if self.project is None:
            return self.create_project()

        if self.working_df.empty and not self.og_df.empty:
            self.working_df = self.og_df[self.selected_columns()].copy()

        if self.current_project_path is None:
            return self.save_project_as()

        active_name = str(self._active_project_name or self.project.get("project_name") or self.current_project_path.stem or "project").strip()
        self._active_project_name = active_name
        self.project["project_name"] = active_name

        self.project["selected_columns"] = list(self.working_df.columns)
        self.project["column_types"] = {col: str(self.working_df[col].dtype) for col in self.working_df.columns}

        try:
            self.project["file_path"] = str(self.file_path) if self.file_path is not None else self.project.get("file_path")
            if not self._save_named_copy_if_requested(active_name):
                return False
            # feature_df is persisted separately from raw and working data so
            # imported/extracted features survive project reloads.
            _save_project(
                self.project,
                self.og_df.copy(),
                self.working_df.copy(),
                str(self.current_project_path),
                feature_df=self.feature_df.copy(),
            )
            self._set_dirty(False)
            QMessageBox.information(self, "Project Saved", f"Saved {self.current_project_path}")
            return True
        except Exception as error:
            self.show_error("Save Error", error)
            return False

    def _save_named_copy_if_requested(self, active_name):
        requested_name = self.project_name.text().strip()
        if not requested_name or requested_name == active_name:
            return True

        copy_path = self.current_project_path.with_name(f"{requested_name}.icp")
        if copy_path.exists():
            result = QMessageBox.question(
                self,
                "Overwrite Project Copy",
                f"A project named '{copy_path.name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                return False

        project_copy = dict(self.project)
        project_copy["project_name"] = requested_name
        _save_project(
            project_copy,
            self.og_df.copy(),
            self.working_df.copy(),
            str(copy_path),
            feature_df=self.feature_df.copy(),
        )

        # Keep working on the current project after creating the renamed copy.
        self._suppress_dirty = True
        try:
            self.project_name.setText(active_name)
        finally:
            self._suppress_dirty = False

        QMessageBox.information(
            self,
            "Project Copy Saved",
            f"Saved copy as {copy_path.name}. Continuing with {active_name}.",
        )
        return True

    def save_project_as(self):
        if self.project is None:
            QMessageBox.warning(self, "No Project", "Create or open a project before saving.")
            return False

        project_name = self.project_name.text().strip() or self.project.get("project_name") or "project"
        chosen_path = self._choose_project_save_path(default_name=project_name)
        if chosen_path is None:
            return False

        if self.working_df.empty and not self.og_df.empty:
            self.working_df = self.og_df[self.selected_columns()].copy()

        try:
            # Update metadata on the active project before saving a copy.
            self.project["selected_columns"] = list(self.working_df.columns)
            self.project["column_types"] = {col: str(self.working_df[col].dtype) for col in self.working_df.columns}
            self.project["file_path"] = str(self.file_path) if self.file_path is not None else self.project.get("file_path")

            # Save a copy under the chosen path, but keep the active project bound
            # to the original project for future saves.
            project_copy = dict(self.project)
            project_copy["project_name"] = chosen_path.stem
            _save_project(
                project_copy,
                self.og_df.copy(),
                self.working_df.copy(),
                str(chosen_path),
                feature_df=self.feature_df.copy(),
            )
            # Keep the active project path unchanged when saving a copy.
            self._set_dirty(False)
            QMessageBox.information(self, "Project Saved", f"Saved {chosen_path}")
            return True
        except Exception as error:
            self.show_error("Save Error", error)
            return False

    def _choose_project_save_path(self, default_name=None):
        default_name = default_name or "project"
        default_file = self.downloads_dir / f"{default_name}.icp"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(default_file),
            "ICP Project Files (*.icp)",
        )
        if not path:
            return None

        chosen_path = Path(path)
        if chosen_path.suffix.lower() != ".icp":
            chosen_path = chosen_path.with_suffix(".icp")

        # if chosen_path.exists():
        #     result = QMessageBox.question(
        #         self,
        #         "Overwrite Project",
        #         f"A project named '{chosen_path.name}' already exists. Overwrite it?",
        #         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        #         QMessageBox.StandardButton.No,
        #     )
        #     if result == QMessageBox.StandardButton.No:
        #         return None

        chosen_path.parent.mkdir(parents=True, exist_ok=True)
        return chosen_path
