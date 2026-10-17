"""Validate model arguments and dispatch only the three bound operations."""
from .filesystem import FilesystemTools, ToolError


ERRORS = {
    "unknown_tool": "Unknown tool",
    "invalid_arguments": "Invalid tool arguments",
    "path_not_allowed": "Path is not allowed",
    "not_found": "Path does not exist",
    "already_exists": "Target already exists",
    "not_a_file": "Path is not a regular file",
    "not_a_directory": "Path is not a directory",
    "too_large": "File exceeds the byte limit",
    "too_many_entries": "Directory exceeds the entry limit",
    "invalid_utf8": "Content is not valid UTF-8",
    "permission_denied": "Permission denied",
    "io_error": "Filesystem operation failed",
}


class ToolRegistry:
    def __init__(self, root):
        filesystem = FilesystemTools(root)
        self.tools = {
            "read_file": filesystem.read_file,
            "write_file": filesystem.write_file,
            "list_files": filesystem.list_files,
        }

    def describe(self):
        return [
            {"name": "read_file", "description": "Read an existing UTF-8 file, at most 65536 bytes",
             "arguments": {"path": "workspace-relative file path (string)"}},
            {"name": "write_file", "description": "Create a new file, at most 65536 UTF-8 bytes; never overwrite; parent must exist",
             "arguments": {"path": "workspace-relative file path (string)", "content": "UTF-8 text (string)"}},
            {"name": "list_files", "description": "List immediate entries, at most 200; use . for workspace root",
             "arguments": {"path": "workspace-relative directory path (string)"}},
        ]

    def execute(self, name, arguments):
        code = None
        if not isinstance(name, str) or name not in self.tools:
            code = "unknown_tool"
        else:
            required = {"path", "content"} if name == "write_file" else {"path"}
            if (not isinstance(arguments, dict) or set(arguments) != required
                    or any(not isinstance(v, str) for v in arguments.values())):
                code = "invalid_arguments"
        if code is None:
            try:
                data = self.tools[name](**arguments)
                return {"ok": True, "tool": name, "data": data, "error": None}
            except ToolError as exc:
                code = exc.code
            except FileExistsError:
                code = "already_exists"
            except FileNotFoundError:
                code = "not_found"
            except IsADirectoryError:
                code = "not_a_file"
            except NotADirectoryError:
                code = "not_a_directory"
            except PermissionError:
                code = "permission_denied"
            except UnicodeError:
                code = "invalid_utf8"
            except OSError:
                code = "io_error"
        return {"ok": False, "tool": name if isinstance(name, str) and name in self.tools else None,
                "data": None, "error": {"code": code, "message": ERRORS[code]}}
