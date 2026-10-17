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
