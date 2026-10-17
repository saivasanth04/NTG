"""Derived Python repository metadata. Never imports or executes indexed source."""
import ast
import hashlib
import io
import sys
import tokenize
from copy import deepcopy
from pathlib import PurePosixPath


def _callee(node):
    if isinstance(node, ast.Name):
        return node.id if len(node.id) <= 200 else None
    if isinstance(node, ast.Attribute):
        base = _callee(node.value)
        name = base + "." + node.attr if base else None
        return name if name and len(name) <= 200 else None
    return None


class _Extract(ast.NodeVisitor):
    def __init__(self, path, sha):
        self.path, self.sha = path, sha
        self.owner = "<module>"
        self.qualified = ""
        self.conditional = 0
        self.symbols, self.imports, self.calls = [], [], []
        self.scopes = {self.owner: {"parent": None, "kind": "module", "bindings": {}}}

    def location(self, node):
        return dict(path=self.path, source_sha256=self.sha, owner_id=self.owner,
                    lineno=node.lineno, end_lineno=getattr(node, "end_lineno", node.lineno),
                    col_offset=node.col_offset, end_col_offset=getattr(node, "end_col_offset", 0))

    def bind(self, name, value):
        self.scopes[self.owner]["bindings"].setdefault(name, []).append(
            ["unknown", "conditional_binding"] if self.conditional else value)

    def definition(self, node, kind):
        qualified = self.qualified + "." + node.name if self.qualified else node.name
        symbol_id = f"{self.path}:{qualified}:{node.lineno}:{node.col_offset}"
        record = dict(self.location(node), id=symbol_id, name=node.name, qualified_name=qualified, kind=kind)
        record["lineno"] = min([node.lineno] + [d.lineno for d in node.decorator_list])
        doc = ast.get_docstring(node, clean=False)
        record["docstring"] = doc.splitlines()[0][:200] if doc else None
        record["docstring_truncated"] = bool(doc and (len(doc) > 200 or "\n" in doc))
        record["parameters"] = []
        if hasattr(node, "args"):
            args = node.args
            for group, values in (("positional_only", args.posonlyargs), ("positional", args.args),
                                  ("keyword_only", args.kwonlyargs), ("vararg", [args.vararg] if args.vararg else []),
                                  ("kwarg", [args.kwarg] if args.kwarg else [])):
                record["parameters"].extend({"name": a.arg, "kind": group,
                    "annotation": (_callee(a.annotation) or type(a.annotation).__name__) if a.annotation else None} for a in values)
            record["return_annotation"] = (_callee(node.returns) or type(node.returns).__name__) if node.returns else None
            record["default_kinds"] = [type(d).__name__ for d in args.defaults + [d for d in args.kw_defaults if d is not None]]
        self.symbols.append(record)
        self.bind(node.name, ["symbol", symbol_id])
        # Defaults, decorators and bases execute in the enclosing lexical scope.
        for expression in node.decorator_list:
            self.visit(expression)
        for expression in getattr(node, "bases", []):
            self.visit(expression)
        for keyword in getattr(node, "keywords", []):
            self.visit(keyword.value)
        if hasattr(node, "args"):
            for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs + [a for a in (node.args.vararg, node.args.kwarg) if a]:
                if argument.annotation:
                    self.visit(argument.annotation)
            if node.returns:
                self.visit(node.returns)
            for expression in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                self.visit(expression)
        previous = self.owner, self.qualified, self.conditional
        self.scopes[symbol_id] = {"parent": self.owner, "kind": kind, "bindings": {}}
        self.owner, self.qualified, self.conditional = symbol_id, qualified, 0
        for parameter in record["parameters"]:
            self.bind(parameter["name"], ["unknown", "parameter_shadow"])
        for statement in node.body:
            self.visit(statement)
        self.owner, self.qualified, self.conditional = previous

    def visit_FunctionDef(self, node):
        self.definition(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self.definition(node, "async_function")

    def visit_ClassDef(self, node):
        self.definition(node, "class")

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            index = len(self.imports)
            self.imports.append(dict(self.location(node), module=alias.name, imported_name=None,
                                     alias=alias.asname, level=0, binding=name, form="import"))
            self.bind(name, ["import", index])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            index = len(self.imports)
            self.imports.append(dict(self.location(node), module=node.module or "", imported_name=alias.name,
                                     alias=alias.asname, level=node.level, binding=name, form="from"))
            self.bind(name, ["import", index])
            if alias.name == "*":
                self.bind("*", ["unknown", "wildcard_import"])

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind(node.id, ["unknown", "name_rebound"])

    def visit_Attribute(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                self.bind(root.id, ["unknown", "attribute_mutation"])
        self.generic_visit(node)

    def visit_Global(self, node):
        for name in node.names:
            self.bind(name, ["unknown", "global_binding"])

    visit_Nonlocal = visit_Global

    def visit_MatchAs(self, node):
        if node.name:
            self.bind(node.name, ["unknown", "pattern_binding"])
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest:
            self.bind(node.rest, ["unknown", "pattern_binding"])
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bind(node.name, ["unknown", "exception_binding"])
        self.generic_visit(node)

    def visit_Call(self, node):
        self.calls.append(dict(self.location(node), callee=_callee(node.func),
                               syntax_kind=type(node.func).__name__))
        self.generic_visit(node)

    def visit_Lambda(self, node):
        # A separate anonymous scope; never resolve its parameters as outer names.
        for expression in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(expression)
        previous = self.owner
        owner = f"{self.path}:<lambda>:{node.lineno}:{node.col_offset}"
        self.scopes[owner] = {"parent": previous, "kind": "function", "bindings": {}}
        self.owner = owner
        for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs + [a for a in (node.args.vararg, node.args.kwarg) if a]:
            self.bind(a.arg, ["unknown", "parameter_shadow"])
        self.visit(node.body)
        self.owner = previous

    def generic_visit(self, node):
        conditional = isinstance(node, (ast.If, ast.Try, ast.TryStar, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Match, ast.IfExp))
        self.conditional += int(conditional)
        super().generic_visit(node)
        self.conditional -= int(conditional)


class RepositoryIndex:
    SCHEMA = 1
    PARSER = "python-ast-3"

    def __init__(self):
        self.files, self.relationships, self.manifest = {}, [], {}
        self.project_revision = None
        self.coverage = {}
        self.stats = {"parsed": 0, "reused": 0}

    def refresh(self, project):
        for attempt in range(2):
            try:
                return self._refresh(project)
            except ValueError as exc:
                if str(exc) != "source_changed" or attempt:
                    raise

    def _refresh(self, project):
        manifest = project.manifest()
        def included(path):
            parts = PurePosixPath(path).parts
            name = parts[-1].casefold()
            return (path not in project.runtime_paths and name != ".env" and not name.startswith(".env.")
                    and name not in ("credentials.json", "secrets.json")
                    and not any(part.casefold() in (".git", ".venv", "venv", "node_modules", "__pycache__") for part in parts))
        inventory = {p: h for p, h in manifest.items() if included(p)}
        files = {}
        self.stats = {"parsed": 0, "reused": 0}
        for path, sha in sorted(inventory.items()):
            old = self.files.get(path)
            if old and old["sha256"] == sha:
                files[path] = old
                self.stats["reused"] += 1
                continue
            target = project.files._regular(path)
            entry = dict(sha256=sha, size_bytes=target.stat().st_size, language="python" if path.endswith(".py") else "other",
                         parse_status="not_python", symbols=[], imports=[], calls=[], scopes={}, diagnostics=[])
            if path.endswith(".py"):
                data = target.read_bytes()
                if hashlib.sha256(data).hexdigest() != sha:
                    raise ValueError("source_changed")
                self.stats["parsed"] += 1
                try:
                    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
                    tree = ast.parse(data.decode(encoding), filename=path)
                    visitor = _Extract(path, sha)
                    visitor.visit(tree)
                    entry.update(parse_status="parsed", symbols=visitor.symbols, imports=visitor.imports,
                                 calls=visitor.calls, scopes=visitor.scopes)
                except (SyntaxError, UnicodeError, LookupError, ValueError, RecursionError) as exc:
                    entry.update(parse_status="error", diagnostics=[{"type": type(exc).__name__,
                        "line": getattr(exc, "lineno", None), "message": str(exc)[:300]}])
            files[path] = entry
        if project.manifest() != manifest:
            raise ValueError("source_changed")
        unchanged = self.manifest == manifest and self.files == files
        self.files, self.manifest = files, manifest
        self.project_revision = project.revision(manifest)
        if not unchanged:
            self._resolve()
        self.coverage = dict(inventory_count=len(files), python_count=sum(p.endswith(".py") for p in files),
            parsed_count=sum(f["parse_status"] == "parsed" for f in files.values()),
            diagnostic_count=sum(len(f["diagnostics"]) for f in files.values()), excluded_count=len(manifest)-len(files))
        return {"project_revision": self.project_revision, "coverage": deepcopy(self.coverage), "stats": dict(self.stats)}

    def _resolve(self):
        modules = {}
        for path in self.files:
            if path.endswith(".py"):
                parts = list(PurePosixPath(path).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                modules.setdefault(".".join(parts), []).append(path)
        symbols = {s["id"]: s for f in self.files.values() for s in f["symbols"]}

        def module_path(name):
            parts = name.split(".")
            for end in range(1, len(parts)):
                parents = modules.get(".".join(parts[:end]), [])
                if len(parents) > 1 or parents and not parents[0].endswith("/__init__.py"):
                    return None
            values = modules.get(name, [])
            return values[0] if len(values) == 1 else None

        def exported(path, name):
            scope = self.files[path]["scopes"].get("<module>", {})
            bindings = scope.get("bindings", {})
            values = bindings.get(name, [])
            return values[0][1] if len(values) == 1 and values[0][0] == "symbol" and "*" not in bindings else None

        relationships = []
        for path, entry in self.files.items():
            import_targets = []
            for imp in entry["imports"]:
                base = imp["module"]
                if imp["level"]:
                    package = list(PurePosixPath(path).parent.parts)
                    if package == ["."]:
                        package = []
                    level = imp["level"]
                    base = ".".join(package[:len(package)-level+1] + ([base] if base else [])) if level <= len(package) else "<invalid-relative>"
                name = imp["imported_name"]
                target, target_id = module_path(base), None
                binding_module = base
                if imp["form"] == "import":
                    binding_module = base if imp["alias"] else base.split(".")[0]
                elif name and name != "*":
                    submodule = module_path(base + "." + name)
                    if submodule:
                        target, binding_module = submodule, base + "." + name
                    elif target:
                        target_id = exported(target, name)
                        binding_module = None
                else:
                    binding_module = None
                import_targets.append((target, target_id, binding_module))
                resolved = target is not None and name != "*" and (imp["form"] == "import" or target_id is not None or binding_module is not None)
                relationships.append(dict(imp, kind="import", target_path=target, target_id=target_id,
                    resolution="resolved" if resolved else "unresolved", reason="local_module" if resolved else "external_ambiguous_or_dynamic"))
            for call in entry["calls"]:
                callee = call["callee"]
                target_id, reason = None, "dynamic_or_oversized_call"
                if callee:
                    parts = callee.split(".")
                    owner = call["owner_id"]
                    values = []
                    while owner is not None:
                        scope = entry["scopes"][owner]
                        bindings = scope["bindings"]
                        if "*" in bindings:
                            values = [["unknown", "wildcard_import"]]
                            break
                        if parts[0] in bindings:
                            values = bindings[parts[0]]
                            break
                        parent = scope["parent"]
                        # Method bodies do not close over class namespaces.
                        while parent and scope["kind"] != "class" and entry["scopes"][parent]["kind"] == "class":
                            parent = entry["scopes"][parent]["parent"]
                        owner = parent
                    reason = "external_unbound_or_shadowed"
                    if len(values) == 1:
                        kind, value = values[0]
                        if kind == "symbol" and len(parts) == 1:
                            target_id = value
                        elif kind == "import":
                            target, imported_id, module = import_targets[value]
                            if len(parts) == 1:
                                target_id = imported_id
                            elif module:
                                target_module = module_path(".".join([module] + parts[1:-1]))
                                if target_module:
                                    target_id = exported(target_module, parts[-1])
                        elif kind == "unknown":
                            reason = value
                target = symbols.get(target_id)
                relationships.append(dict(call, kind="call", target_id=target_id,
                    target_path=target["path"] if target else None,
                    resolution="resolved" if target else "unresolved", reason="unique_lexical_target" if target else reason))
        self.relationships = sorted(relationships, key=lambda r: (r["path"], r["lineno"], r["col_offset"], r["kind"]))

    def _page(self, records, offset, limit):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("invalid_repository_range")
        return dict(project_revision=self.project_revision, records=deepcopy(records[offset:offset+limit]),
                    next_offset=offset+limit if offset+limit < len(records) else None,
                    coverage=deepcopy(self.coverage), diagnostics=[{"path": p, **d} for p, f in self.files.items() for d in f["diagnostics"]][:10],
                    diagnostics_more=sum(len(f["diagnostics"]) for f in self.files.values()) > 10)

    def search(self, query, *, path=None, offset=0, limit=10):
        if not isinstance(query, str) or not query or len(query) > 200:
            raise ValueError("invalid_repository_query")
        if path is not None and path not in self.files:
            raise ValueError("unknown_repository_file")
        rows = []
        for file, entry in self.files.items():
            if path is not None and file != path:
                continue
            rows.append(dict(path=file, source_sha256=entry["sha256"], kind="file", name=file, lineno=0))
            rows.extend(entry["symbols"])
        rows.extend(r for r in self.relationships if path is None or r["path"] == path)
        q = query.casefold()
        matches = []
        for row in rows:
            names = [str(row[k]).casefold() for k in ("name", "qualified_name", "module", "imported_name", "callee", "path") if row.get(k)]
            if any(q in n for n in names):
                rank = 0 if row.get("name", "").casefold() == q else 1 if row.get("qualified_name", "").casefold() == q else 2 if any(n.startswith(q) for n in names) else 3
                matches.append((rank, row))
        matches.sort(key=lambda pair: (pair[0], pair[1]["path"], pair[1]["lineno"], pair[1]["kind"], pair[1].get("id", "")))
        return self._page([r for _, r in matches], offset, limit)

    def dependencies(self, path, *, offset=0, limit=10):
        if path not in self.files:
            raise ValueError("unknown_repository_file")
        rows = []
        for record in self.relationships:
            if record["path"] == path:
                rows.append(dict(record, direction="outgoing"))
            if record["target_path"] == path and record["resolution"] == "resolved":
                rows.append(dict(record, direction="incoming"))
        return self._page(rows, offset, limit)

    def snapshot(self):
        return deepcopy(dict(schema_version=self.SCHEMA, parser_version=self.PARSER,
            python_version=list(sys.version_info[:2]), project_revision=self.project_revision,
            manifest=self.manifest, files=self.files, relationships=self.relationships, coverage=self.coverage))

    @classmethod
    def from_snapshot(cls, snapshot):
        if (snapshot.get("schema_version") != cls.SCHEMA or snapshot.get("parser_version") != cls.PARSER
                or snapshot.get("python_version") != list(sys.version_info[:2])):
            raise ValueError("invalid_repository_cache")
        from .project import ProjectFiles
        if ProjectFiles.revision(snapshot["manifest"]) != snapshot["project_revision"]:
            raise ValueError("invalid_repository_cache")
        index = cls()
        for path, entry in snapshot["files"].items():
            if (PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\\" in path or ":" in path
                    or snapshot["manifest"].get(path) != entry["sha256"]):
                raise ValueError("invalid_repository_cache")
            for key in ("symbols", "imports", "calls", "diagnostics"):
                if not isinstance(entry[key], list):
                    raise ValueError("invalid_repository_cache")
            for row in entry["symbols"] + entry["imports"] + entry["calls"]:
                if row["path"] != path or row["source_sha256"] != entry["sha256"] or type(row["lineno"]) is not int:
                    raise ValueError("invalid_repository_cache")
        index.files = deepcopy(snapshot["files"])
        # Derived edges are always recomputed against the current tracked inventory.
        return index
