"""Provider credentials and default model pools; access is provider-enforced."""
from dataclasses import dataclass, field
import os
import re
import hashlib
from dotenv import dotenv_values

PREFIX = {"gemini": "gemini", "groq": "groq", "openrouter": "openrouter",
          "cloudflare": "cloudflare", "cohere": "cohere_chat"}

@dataclass
class Account:
    provider: str
    id: str
    key: str = field(repr=False)
    account_id: str = field(default="", repr=False)
    group: str = ""
    plan: str = "unknown"  # Legacy metadata; never an admission gate.
    models: tuple[str, ...] = ()

    @property
    def masked(self):
        # Stable logical ID; never a fragment of the actual key/account ID.
        return self.id.split("-")[0] + "-***" + self.id.split("-")[-1]

def load_accounts(path=".env"):
    values = {**dotenv_values(path), **os.environ}
    accounts = []
    for name in sorted(values):
        match = re.fullmatch(r"(GEMINI|GROQ|OPENROUTER|CLOUDFLARE|COHERE)_API_KEY_(\d+)", name)
        if not match or not values[name]:
            continue
        provider, number = match.group(1).lower(), match.group(2)
        stem = f"{provider.upper()}_{number}"
        identity = f"{provider}-{number}"
        cloudflare_id = values.get(f"CLOUDFLARE_ACCOUNT_ID_{number}", "") if provider == "cloudflare" else ""
        group = ("cf-" + hashlib.sha256(cloudflare_id.encode()).hexdigest()[:16]) if cloudflare_id else identity
        accounts.append(Account(provider, identity, values[name], cloudflare_id,
            group if provider == "cloudflare" else values.get(stem + "_QUOTA_GROUP", group),
            values.get(stem + "_FREE_PLAN", "unknown"),
            tuple(x.strip() for x in values.get(stem + "_MODELS", "").split(",") if x.strip())))
    return accounts

def eligibility(account, model):
    if account.provider == "cloudflare" and not re.fullmatch(r"[a-fA-F0-9]{32}", account.account_id):
        return "missing/invalid Cloudflare account ID"
    if model == "unconfigured":
        return "no default model configured; specify a model"
    return None
