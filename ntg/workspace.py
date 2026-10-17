"""Host-owned task allocation; project selection is never a model tool."""
import uuid
from dataclasses import dataclass
from pathlib import Path

from tools import ToolRegistry
from tools.filesystem import FilesystemTools, ToolError


@dataclass(frozen=True)
class Workspace:
    project_id: str
    path: Path
    tools: ToolRegistry

    @property
    def artifacts(self):
        """Host report storage; never passed to agents or registered model tools."""
        self.tools.validate_workspace()
        return ToolRegistry(self.path, host_artifacts=True)


class WorkspaceManager:
    def __init__(self, base):
        try:
            requested = Path(base).absolute()
            parent = requested.parent.resolve(strict=True)
            base = parent / requested.name
            try:
                base.mkdir(exist_ok=False)
            except FileExistsError:
                pass
            self._base_guard = FilesystemTools(base)
        except (OSError, ToolError) as exc:
            raise ValueError("workspace_unavailable") from exc

    def open(self, project_id):
        """Reopen a specifically selected workspace for validated checkpoint restoration."""
        import re
        if not isinstance(project_id, str) or not re.fullmatch(r"project_[0-9a-f]{32}", project_id):
            raise ValueError("invalid_project_id")
        base = self._base_guard.validate_root()
        path = base / project_id
        guard = FilesystemTools(path)
        if guard.root.parent != base:
            raise ValueError("workspace_unavailable")
        return Workspace(project_id, guard.root, ToolRegistry(guard.root))

    def create(self):
        try:
            for _ in range(3):
                base = self._base_guard.validate_root()
                project_id = "project_" + uuid.uuid4().hex
                candidate = base / project_id
                # lstat also detects dangling links; never follow or reuse a collision.
                try:
                    candidate.lstat()
                except FileNotFoundError:
                    pass
                else:
                    continue
                if candidate.resolve(strict=False).parent != base:
                    raise ValueError("workspace_unavailable")
                try:
                    candidate.mkdir(exist_ok=False)
                except FileExistsError:
                    continue
                self._base_guard.validate_root()
                project = candidate.resolve(strict=True)
                if project.parent != base or FilesystemTools._linked(candidate.lstat()):
                    raise ValueError("workspace_unavailable")
                registry = ToolRegistry(project)
                self._base_guard.validate_root()
                return Workspace(project_id, project, registry)
        except (OSError, ToolError) as exc:
            raise ValueError("workspace_unavailable") from exc
        raise ValueError("workspace_allocation_failed")
