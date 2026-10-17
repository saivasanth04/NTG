"""Compile first, execute host-defined checks, and report observed failures."""
import ast
import asyncio
import hashlib
import json
import re
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field

from .project import validate_entrypoint


@dataclass
class ExecutionEvidence:
    project_id: str
    source_sha256: str
    argv: list[str]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    timed_out: bool = False
    output_limited: bool = False
    infrastructure_error: str | None = None

    @property
    def passed(self):
        return (self.exit_code == 0 and not self.timed_out
                and not self.output_limited and not self.infrastructure_error)


def check_passed(check):
    evidence = check["evidence"]
    return (not any(evidence[k] for k in ("timed_out", "output_limited", "infrastructure_error"))
            and evidence["exit_code"] == check["exit_code"]
            and all(text.casefold() in evidence["stdout"].casefold()
                    for text in check["stdout_contains"]))


@dataclass
class TestReport:
    project_id: str
    source_sha256: str
    status: str
    program: ExecutionEvidence | None
    feedback: str
    analysis_error: str | None = None
    compilation: ExecutionEvidence | None = None
    checks: list[dict] = field(default_factory=list)
    failure_type: str | None = None
    failure_signature: str | None = None
    source_context: str = ""

    @property
    def verified(self):
        executions = [self.compilation, self.program]
        return (self.status == "passed" and self.compilation is not None
                and self.compilation.passed and self.program is not None and bool(self.checks)
                and all(e is not None and e.project_id == self.project_id
                        and e.source_sha256 == self.source_sha256 for e in executions)
                and all(c["evidence"]["project_id"] == self.project_id
                        and c["evidence"]["source_sha256"] == self.source_sha256
                        and check_passed(c) for c in self.checks))

    @property
    def score(self):
        if self.compilation is None or not self.compilation.passed:
            return 0
        return 1 + sum(
            int(c["evidence"]["exit_code"] == c["exit_code"])
            + sum(text.casefold() in c["evidence"]["stdout"].casefold()
                  for text in c["stdout_contains"])
            for c in self.checks
        )

    def to_dict(self, *, compact=False):
        data = asdict(self)
        data.update(verified=self.verified, score=self.score)
        if compact:
            data["feedback"] = data["feedback"][:4096]
            executions = [data["program"], data["compilation"]]
            executions += [c["evidence"] for c in data["checks"]]
            for evidence in executions:
                if evidence is not None:
                    for stream in ("stdout", "stderr"):
                        original = evidence[stream]
                        evidence[stream] = original[-2048:]
                        evidence[stream + "_excerpted"] = len(original) > 2048
        return data


class TesterAgent:
    RUN_TIMEOUT = 20
    OUTPUT_LIMIT = 32768

    def __init__(self, llm=None, entrypoint="calculator.py", *, checks=None):
        # Keep llm for API compatibility. Diagnosis deliberately uses no LLM.
        self.llm = llm
        self.entrypoint = validate_entrypoint(entrypoint)
        selected = checks if checks is not None else [
            {"name": "startup", "arguments": [], "exit_code": 0, "stdout_contains": []},
        ]
        if not isinstance(selected, list) or not 1 <= len(selected) <= 8:
            raise ValueError("invalid_checks")
        self.checks = []
        for check in selected:
            if (not isinstance(check, dict)
                    or set(check) != {"name", "arguments", "exit_code", "stdout_contains"}
                    or not isinstance(check["name"], str) or not check["name"].strip()
                    or type(check["exit_code"]) is not int
                    or not isinstance(check["arguments"], list) or len(check["arguments"]) > 32
                    or any(not isinstance(a, str) or "\0" in a for a in check["arguments"])
                    or not isinstance(check["stdout_contains"], list)
                    or len(check["stdout_contains"]) > 8
                    or any(not isinstance(t, str) or not t for t in check["stdout_contains"])):
                raise ValueError("invalid_checks")
            self.checks.append({**check, "arguments": list(check["arguments"]),
                                "stdout_contains": list(check["stdout_contains"])})
        if len({c["name"] for c in self.checks}) != len(self.checks):
            raise ValueError("invalid_checks")

    def describe_checks(self):
        return json.loads(json.dumps(self.checks))

    def _source(self, workspace):
        root = workspace.tools.validate_workspace()
        if root != workspace.path:
            raise ValueError("workspace_mismatch")
        result = workspace.tools.execute("read_file", {"path": self.entrypoint})
        if not result["ok"]:
            raise ValueError(result["error"]["code"])
        source = result["data"]["content"]
        return root, source, hashlib.sha256(source.encode("utf-8")).hexdigest()

    async def _execute(self, root, evidence):
        started = time.monotonic()
        proc = None
        tasks = []
        buffers = [bytearray(), bytearray()]
        overflow = asyncio.Event()

        async def drain(stream, buffer):
            while chunk := await stream.read(4096):
                available = max(0, self.OUTPUT_LIMIT - len(buffer))
                buffer.extend(chunk[:available])
                if len(chunk) > available:
                    overflow.set()

        try:
            proc = await asyncio.create_subprocess_exec(
                *evidence.argv, cwd=root, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            readers = [asyncio.create_task(drain(proc.stdout, buffers[0])),
                       asyncio.create_task(drain(proc.stderr, buffers[1]))]
            exited = asyncio.create_task(proc.wait())
            exceeded = asyncio.create_task(overflow.wait())
            tasks = readers + [exited, exceeded]
            async with asyncio.timeout(self.RUN_TIMEOUT):
                await asyncio.wait([exited, exceeded], return_when=asyncio.FIRST_COMPLETED)
                if not overflow.is_set():
                    await asyncio.gather(*readers)
                evidence.output_limited = overflow.is_set()
                if not evidence.output_limited:
                    evidence.exit_code = await exited
        except TimeoutError:
            evidence.timed_out = True
        except OSError:
            evidence.infrastructure_error = "python_start_failed"
        finally:
            if proc is not None and proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            evidence.stdout = buffers[0].decode("utf-8", "replace")
            evidence.stderr = buffers[1].decode("utf-8", "replace")
            evidence.duration_ms = round((time.monotonic() - started) * 1000, 2)

    async def run(self, workspace, specification):
        try:
            root, source, digest = self._source(workspace)
        except ValueError as exc:
            return TestReport(workspace.project_id, "", "blocked", None, str(exc),
                              failure_type="infrastructure")
        target = str(root / self.entrypoint)
        with tempfile.TemporaryDirectory(prefix="ntg_compile_") as cache:
            compilation = ExecutionEvidence(workspace.project_id, digest, [
                sys.executable, "-B", "-X", f"pycache_prefix={cache}",
                "-m", "py_compile", target,
            ])
            await self._execute(root, compilation)
        report = TestReport(workspace.project_id, digest, "failed", None, "",
                            compilation=compilation)
        if compilation.passed:
            for check in self.checks:
                evidence = ExecutionEvidence(workspace.project_id, digest,
                                             [sys.executable, "-B", target, *check["arguments"]])
                await self._execute(root, evidence)
                if report.program is None:
                    report.program = evidence
                report.checks.append({**check, "evidence": asdict(evidence)})
                if evidence.infrastructure_error:
                    break
        try:
            if self._source(workspace)[2] != digest:
                compilation.infrastructure_error = "source_changed"
        except ValueError:
            compilation.infrastructure_error = "workspace_unavailable"
        failed = [c for c in report.checks if not check_passed(c)]
        evidence = (compilation if not compilation.passed else
                    ExecutionEvidence(**failed[0]["evidence"]) if failed else report.program)
        if evidence.infrastructure_error:
            report.status, report.failure_type = "blocked", "infrastructure"
            report.feedback = evidence.infrastructure_error
        elif evidence.timed_out or evidence.output_limited:
            report.failure_type = "timeout" if evidence.timed_out else "output_limit"
            report.feedback = "Execution reached its " + report.failure_type + " limit."
        elif not compilation.passed:
            report.failure_type = "syntax_error" if re.search(
                r"(?:SyntaxError|IndentationError|TabError):", compilation.stderr) else "compile_error"
            report.feedback = "python -m py_compile failed. No functional check was executed."
        elif failed:
            report.failure_type = "behavior_mismatch" if evidence.exit_code == failed[0]["exit_code"] else "runtime_error"
            if report.failure_type == "runtime_error":
                classes = re.findall(r"(?m)^([A-Za-z_]\w*(?:Error|Exception)):", evidence.stderr)
                if classes:
                    report.failure_type = "runtime:" + classes[-1]
            report.feedback = (f"Check {failed[0]['name']!r}: expected exit {failed[0]['exit_code']} "
                               f"and stdout containing {failed[0]['stdout_contains']!r}; "
                               f"observed exit {evidence.exit_code}.")
            missing = [text for text in failed[0]["stdout_contains"]
                       if text.casefold() not in evidence.stdout.casefold()]
            report.feedback += f" Missing stdout fragments: {missing!r}."
            if (not failed[0]["arguments"] and evidence.exit_code == 0
                    and "usage:" in missing):
                report.feedback += (
                    " No-argument execution already exits successfully, but does not print "
                    "the required help. Do not change an already optional subparser group. "
                    "Inspect the path after parse_args(): print the parser's help and return "
                    "when no command is selected, preserving command dispatch and --help."
                )
                for node in ast.walk(ast.parse(source)):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "parse_args"):
                        report.source_context = self._context(source, node.lineno)
                        break
            if (not failed[0]["arguments"] and failed[0]["exit_code"] == 0 and evidence.exit_code == 2
                    and "arguments are required: command" in evidence.stderr):
                report.failure_type = "argparse_required_subcommand"
                report.feedback += (" Argparse requires a command, but this check requires "
                                    "no-argument help and exit 0. The usage lists existing "
                                    "subcommands; do not infer that subparsers are missing.")
                for node in ast.walk(ast.parse(source)):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "add_subparsers"
                            and any(k.arg == "dest" and isinstance(k.value, ast.Constant)
                                    and k.value.value == "command" for k in node.keywords)
                            and any(k.arg == "required" and isinstance(k.value, ast.Constant)
                                    and k.value.value is True for k in node.keywords)):
                        report.feedback += (f" Source line {node.lineno} sets add_subparsers(required=True). "
                                            "Make this group optional and print help/return 0 "
                                            "when no command is provided; preserve existing subparsers.")
                        report.source_context = self._context(source, node.lineno)
                        break
        else:
            report.status, report.feedback = "passed", "Compile and all configured checks passed."
        if report.status != "passed":
            if not report.source_context:
                match = re.search(r"line (\d+)", evidence.stderr)
                if match:
                    report.source_context = self._context(source, int(match.group(1)))
            report.feedback += "\nActual stderr:\n" + evidence.stderr + "\nActual stdout:\n" + evidence.stdout
            def normalized(text):
                return re.sub(r"line \d+", "line N", text.replace(str(root), "<workspace>"))
            report.failure_signature = hashlib.sha256(json.dumps({
                "type": report.failure_type,
                "compile": None if compilation.passed else normalized(compilation.stderr),
                "checks": [{"name": c["name"], "exit_code": c["evidence"]["exit_code"],
                            "stderr": normalized(c["evidence"]["stderr"]),
                            "stdout": normalized(c["evidence"]["stdout"])} for c in failed],
            }, sort_keys=True).encode()).hexdigest()
        return report

    @staticmethod
    def _context(source, line):
        return "\n".join(f"{i+1}: {text}" for i, text in enumerate(source.splitlines())
                         if max(0, line-4) <= i < line+3)
