"""Small create-only filesystem operations; never execute generated code."""
import os
import hashlib
import uuid
import re
import stat
from pathlib import Path, PureWindowsPath


class ToolError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class FilesystemTools:
    MAX_BYTES = 65536
    MAX_ENTRIES = 200

    def __init__(self, root):
        try:
            root = Path(root).absolute()
            info = root.lstat()
            if self._linked(info) or not stat.S_ISDIR(info.st_mode):
                raise ToolError("workspace_unavailable")
            self._root = root.resolve(strict=True)
            info = self._root.lstat()
            self._root_identity = (info.st_dev, info.st_ino)
            self.validate_root()
        except (OSError, RuntimeError) as exc:
            raise ToolError("workspace_unavailable") from exc

    @property
    def root(self):
        return self._root

    def validate_root(self):
        """Fail closed if the bound directory was removed, replaced or redirected."""
        try:
            info = self.root.lstat()
            if (self._linked(info) or not stat.S_ISDIR(info.st_mode)
                    or (info.st_dev, info.st_ino) != self._root_identity
                    or self.root.resolve(strict=True) != self.root):
                raise ToolError("workspace_unavailable")
        except (OSError, RuntimeError) as exc:
            raise ToolError("workspace_unavailable") from exc
        return self.root

    @staticmethod
    def _linked(info):
        return stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )

    @staticmethod
    def _allowed_name(name):
        return not (
            name.startswith(".") or name.lower() == "__pycache__"
            or name.lower().endswith(".pyc") or name.endswith((" ", "."))
            or re.search(r'[<>:"|?*\x00-\x1f]', name)
            or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name)
        )

    def _resolve(self, path, *, allow_root=False):
        if not isinstance(path, str) or not path or len(path) > 240:
            raise ToolError("invalid_arguments")
        self.validate_root()
        if path == "." and allow_root:
            return self.root
        windows = PureWindowsPath(path)
        if windows.drive or windows.root or Path(path).is_absolute():
            raise ToolError("path_not_allowed")
        parts = path.replace("\\", "/").split("/")
        if any(not p or not self._allowed_name(p) for p in parts):
            raise ToolError("path_not_allowed")
        candidate = self.root
        for part in parts:
            candidate = candidate / part
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                continue
            if self._linked(info):
                raise ToolError("path_not_allowed")
        candidate = candidate.resolve(strict=False)
        if not candidate.is_relative_to(self.root):
            raise ToolError("path_not_allowed")
        return candidate

    def _relative(self, path):
        return path.relative_to(self.root).as_posix()

    def read_file(self, path):
        target = self._resolve(path)
        info = target.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ToolError("not_a_file")
        if info.st_nlink > 1:
            raise ToolError("path_not_allowed")
        if info.st_size > self.MAX_BYTES:
            raise ToolError("too_large")
        with target.open("rb") as stream:
            content = stream.read(self.MAX_BYTES + 1)
        if len(content) > self.MAX_BYTES:
            raise ToolError("too_large")
        return {"path": self._relative(target), "content": content.decode("utf-8")}

    def write_file(self, path, content):
        target = self._resolve(path)
        if not isinstance(content, str):
            raise ToolError("invalid_arguments")
        encoded = content.encode("utf-8")
        if len(encoded) > self.MAX_BYTES:
            raise ToolError("too_large")
        with target.open("xb") as stream:
            written = stream.write(encoded)
        return {"path": self._relative(target), "bytes_written": written}

    def replace_file(self, path, content, expected_sha256):
        """Host-only atomic update of an existing, hash-checked regular file."""
        target = self._resolve(path)
        encoded = content.encode("utf-8")
        if len(encoded) > self.MAX_BYTES:
            raise ToolError("too_large")
        def unchanged():
            current = self.read_file(path)["content"].encode("utf-8")
            if hashlib.sha256(current).hexdigest() != expected_sha256:
                raise ToolError("source_changed")
        unchanged()
        temporary = target.with_name("ntg_repair_" + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
            self.validate_root()
            unchanged()
            os.replace(temporary, target)
        finally:
            self.validate_root()
            temporary.unlink(missing_ok=True)
        return {"path": self._relative(target), "bytes_written": len(encoded)}

    def list_files(self, path):
        target = self._resolve(path, allow_root=True)
        if not target.is_dir():
            if not target.exists():
                raise FileNotFoundError()
            raise ToolError("not_a_directory")
        entries = []
        with os.scandir(target) as listing:
            for count, entry in enumerate(listing, start=1):
                if count > self.MAX_ENTRIES:
                    raise ToolError("too_many_entries")
                # Windows DirEntry.stat caches st_nlink as zero. lstat supplies
                # the actual link count while retaining reparse-point metadata.
                info = Path(entry.path).lstat()
                if not self._allowed_name(entry.name) or self._linked(info):
                    continue
                if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    kind = "file"
                elif stat.S_ISDIR(info.st_mode):
                    kind = "directory"
                else:
                    continue
                entries.append({"name": entry.name, "type": kind})
        return {"path": self._relative(target), "entries": sorted(entries, key=lambda e: e["name"])}
