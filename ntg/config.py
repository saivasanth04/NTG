import os
import re
from pathlib import Path

from dotenv import dotenv_values

from .types import Account

PROVIDERS = {"GEMINI", "GROQ", "OPENROUTER", "CLOUDFLARE", "COHERE"}


def environment(path=".env"):
    # Do not export provider secrets to process-global environment / LiteLLM auto-discovery.
    return {**dotenv_values(path, interpolate=False), **os.environ}


def accounts_from_env(path=".env", values=None):
    values = environment(path) if values is None else values
    accounts = []
    seen = {}
    for name, value in sorted(values.items()):
        match = re.fullmatch(r"([A-Z]+)_API_KEY(?:_(\d+))?", name)
        if not match or match[1] not in PROVIDERS or not value:
            continue
        provider, index = match[1], match[2] or "1"
        prefix = f"{provider}_{index}"
        alias = prefix.lower().replace("_", "-")
        # Unknown ownership means shared provider capacity, not independent quotas.
        group = values.get(f"{prefix}_QUOTA_GROUP") or f"{provider.lower()}-shared"
        # Identical credentials cannot manufacture separate capacity groups.
        group = seen.setdefault((provider, value), group)
        accounts.append(Account(provider.lower(), alias, value,
            [m.strip() for m in (values.get(f"{prefix}_MODELS") or "").split(",") if m.strip()],
            f"{provider.lower()}:{group}", values.get(f"CLOUDFLARE_ACCOUNT_ID_{index}", "") or "",
            values.get(f"{prefix}_FREE_VERIFIED", "").lower() == "true"))
    return accounts


def load_policy(path="policy.json"):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {"models": {}}
