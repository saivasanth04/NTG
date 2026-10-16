"""Local exact-value leak check. Reports paths only, never secret values."""
from pathlib import Path
from dotenv import dotenv_values

root = Path(__file__).resolve().parents[1]
values = dotenv_values(root / ".env", interpolate=False)
secrets = [v.encode() for k, v in values.items() if v and
           ("API_KEY" in k or "ACCOUNT_ID" in k or k == "NTG_GATEWAY_KEY") and len(v) >= 8]
leaks = []
checked = 0
for path in root.rglob("*"):
    if not path.is_file() or any(part in (".git", ".venv", "__pycache__", ".pytest_cache") for part in path.parts):
        continue
    if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
        continue
    checked += 1
    if any(secret in path.read_bytes() for secret in secrets):
        leaks.append(str(path.relative_to(root)))
print({"files_checked": checked, "leaks": leaks})
raise SystemExit(bool(leaks))
