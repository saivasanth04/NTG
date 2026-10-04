"""Lightweight JSON persistence for request timestamps."""

import json
import os
import time
from typing import Dict, List, Optional

from ntg.config import HISTORY_FILE_PATH, NTG_DIR


def ensure_ntg_dir() -> None:
    """Ensures .ntg storage directory exists."""
    if not os.path.exists(NTG_DIR):
        os.makedirs(NTG_DIR, exist_ok=True)


def load_history() -> Dict[str, List[float]]:
    """Loads request timestamps per account from .ntg/request_history.json."""
    if not os.path.exists(HISTORY_FILE_PATH):
        return {}
    try:
        with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {
                    acc: [float(ts) for ts in timestamps if isinstance(ts, (int, float))]
                    for acc, timestamps in data.items()
                    if isinstance(timestamps, list)
                }
    except Exception:
        pass
    return {}


def save_history(history: Dict[str, List[float]]) -> None:
    """Saves request timestamps per account to .ntg/request_history.json."""
    ensure_ntg_dir()
    tmp_file = f"{HISTORY_FILE_PATH}.tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_file, HISTORY_FILE_PATH)
    except Exception:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


def clean_expired_timestamps(
    history: Dict[str, List[float]],
    max_age_seconds: float = 86400.0,
    now: Optional[float] = None,
) -> Dict[str, List[float]]:
    """Removes timestamps older than max_age_seconds (24 hours)."""
    current_time = now if now is not None else time.time()
    cutoff = current_time - max_age_seconds
    cleaned = {}
    for acc, timestamps in history.items():
        valid_ts = [ts for ts in timestamps if ts > cutoff]
        cleaned[acc] = valid_ts
    return cleaned


def record_request_timestamp(
    account_name: str,
    ts: Optional[float] = None,
) -> float:
    """Records a request timestamp for the specified account and persists state."""
    current_ts = ts if ts is not None else time.time()
    history = load_history()
    history = clean_expired_timestamps(history, now=current_ts)
    if account_name not in history:
        history[account_name] = []
    history[account_name].append(current_ts)
    save_history(history)
    return current_ts
