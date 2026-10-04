"""Local quota guardrails.

OpenRouter is authoritative for actual free-tier quota.
NTG local counters exist to reduce accidental overuse.
"""

from __future__ import annotations

from ntg.models import Account


def mark_request(
    account: Account,
) -> None:
    account.mark_attempt()


def is_locally_exhausted(
    account: Account,
) -> bool:

    return (
        account.local_rpd_remaining <= 0
        or account.local_rpm_remaining <= 0
    )


def remaining_local_requests(
    account: Account,
) -> int:

    return account.local_rpd_remaining