"""Host-selected single-file application names."""
import keyword
import re


def validate_entrypoint(name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,59}\.py", name):
        raise ValueError("invalid_entrypoint")
    stem = name[:-3]
    if (keyword.iskeyword(stem) or stem in {"con", "prn", "aux", "nul", "conin", "conout"}
            or re.fullmatch(r"(?:com|lpt)[1-9]", stem)):
        raise ValueError("invalid_entrypoint")
    return name


class ProjectFiles:
    """Tracked source paths, bounded observations and streamed atomic edits."""
    def __init__(self, workspace, paths=()):
        self.workspace = workspace
        self.files = workspace.artifacts._artifacts()
        self.paths = set(paths)
        self.runtime_paths = set()
        self.changed_ranges = {}
        self._hashes = {}
        for path in self.paths:
            self.files._regular(path)

    def manifest(self, refresh=False):
        result = {}
        for path in sorted(self.paths):
            info = self.files._regular(path).stat()
            identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            cached = self._hashes.get(path)
            if refresh or cached is None or cached[0] != identity:
                cached = (identity, self.files.file_sha256(path))
                self._hashes[path] = cached
            result[path] = cached[1]
        return result

    @staticmethod
    def revision(manifest):
        import hashlib, json
        return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def verify_python_inventory(self):
        found = set()
        for path in self.workspace.path.rglob("*.py"):
            relative = path.relative_to(self.workspace.path).as_posix()
            self.files._regular(relative)
            found.add(relative)
        if found != {p for p in self.paths if p.endswith(".py")}:
            raise ValueError("untracked_python_source")

    def listing(self, offset=0, limit=50):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid_range")
        paths = sorted(self.paths)
        return {"files": paths[offset:offset+limit], "next_offset": offset+limit if offset+limit < len(paths) else None}

    def read(self, path, start=1, count=80, chars=12000):
        if path not in self.paths or type(start) is not int or start < 1 or type(count) is not int or not 1 <= count <= 200:
            raise ValueError("invalid_source_range")
        lines, used, more = [], 0, None
        with self.files._regular(path).open("r", encoding="utf-8", newline="") as stream:
            for number, line in enumerate(stream, 1):
                if number < start:
                    continue
                if len(lines) >= count or used + len(line) > chars:
                    more = number
                    break
                lines.append({"line": number, "text": line})
                used += len(line)
        return {"path": path, "sha256": self.files.file_sha256(path), "lines": lines, "next_line": more}

    def search(self, query, limit=30):
        if not isinstance(query, str) or not query or len(query) > 200:
            raise ValueError("invalid_query")
        hits = []
        for path in sorted(self.paths):
            with self.files._regular(path).open("r", encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if query.casefold() in line.casefold():
                        hits.append({"path": path, "line": number, "text": line[:400]})
                        if len(hits) >= limit:
                            return {"matches": hits, "more": True}
        return {"matches": hits, "more": False}

    def edit(self, path, expected, start, end, replacement):
        if not isinstance(path, str) or not path.endswith((".py", ".json", ".toml", ".txt", ".md")):
            raise ValueError("unsupported_project_file")
        target = self.files._resolve(path)
        if not isinstance(replacement, str) or len(replacement.encode()) > 24000:
            raise ValueError("edit_output_too_large")
        if expected is None:
            if path in self.paths or start != 1 or end != 0:
                raise ValueError("invalid_create")
            # Parent creation is host-owned and path-validated before and after mkdir.
            target.parent.mkdir(parents=True, exist_ok=True)
            self.files._resolve(path)
            saved = self.files.write_file(path, replacement)
            self.paths.add(path)
            self.changed_ranges[path] = [(1, max(1, len(replacement.splitlines())))]
            return saved
        if path not in self.paths or type(start) is not int or type(end) is not int or start < 1 or end < start - 1:
            raise ValueError("invalid_edit")
        if end-start+1 > 200:
            raise ValueError("edit_range_too_broad")
        if replacement and not replacement.endswith("\n"):
            replacement += "\n"
        def pieces():
            inserted, last = False, 0
            with self.files._regular(path).open("r", encoding="utf-8", newline="") as stream:
                for number, line in enumerate(stream, 1):
                    last = number
                    if number == start:
                        yield replacement
                        inserted = True
                    if number < start or number > end:
                        yield line
            if start == last + 1 and end == last:
                yield replacement
                inserted = True
            if not inserted or end > last:
                raise ValueError("invalid_edit_range")
        before = self.files.file_sha256(path)
        if before != expected:
            raise ValueError("source_changed")
        saved = self.files._atomic_write(path, pieces(), expected)
        if saved["sha256"] == before:
            raise ValueError("edit_has_no_changes")
        inserted = len(replacement.splitlines())
        delta = inserted - (end-start+1)
        ranges = []
        for first, last in self.changed_ranges.get(path, []):
            if last < start:
                ranges.append((first, last))
            elif first > end:
                ranges.append((max(1, first+delta), max(1, last+delta)))
            else:
                ranges.append((min(first, start), max(start, last+delta, start+inserted-1)))
        ranges.append((start, start+inserted-1) if inserted else (max(1, start-1), max(1, start-1)))
        self.changed_ranges[path] = ranges
        return saved

    def clone_to(self, workspace):
        import shutil
        other = ProjectFiles(workspace)
        other.runtime_paths = set(self.runtime_paths)
        before = self.manifest()
        for path in sorted(self.paths):
            target = other.files._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            other.files._resolve(path)
            shutil.copyfile(self.files._regular(path), target)
            other.paths.add(path)
        if before != self.manifest() or before != other.manifest():
            raise ValueError("source_changed")
        return other
