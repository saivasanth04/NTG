"""Human-readable NTG diagnostics."""

from __future__ import annotations

from typing import Any

from ntg.models import Account, utc_string


def print_divider(
    title: str | None = None,
    width: int = 72,
) -> None:

    print(
        "\n" + "=" * width
    )

    if title:
        print(title)
        print("=" * width)


def print_account_status(
    accounts: list[Account],
) -> None:

    print_divider(
        "ACCOUNT STATUS"
    )

    for account in accounts:

        account.refresh()

        print(
            f"\n{account.name}"
        )

        print(
            f"  Available          : "
            f"{account.available}"
        )

        print(
            f"  Attempts           : "
            f"{account.attempts}"
        )

        print(
            f"  Successes          : "
            f"{account.successes}"
        )

        print(
            f"  Failures           : "
            f"{account.failures}"
        )

        print(
            f"  429s               : "
            f"{account.rate_limits}"
        )

        print(
            f"  Local RPD          : "
            f"{account.requests_today}/"
            f"{account.rpd_limit}"
        )

        print(
            f"  Local RPD left     : "
            f"{account.local_rpd_remaining}"
        )

        print(
            f"  Local RPM          : "
            f"{account.local_rpm_used}/"
            f"{account.rpm_limit}"
        )

        print(
            f"  Local RPM left     : "
            f"{account.local_rpm_remaining}"
        )

        if account.remaining_cooldown > 0:

            print(
                f"  Block remaining    : "
                f"{account.remaining_cooldown:.1f}s"
            )

        if account.blocked_reason:

            print(
                f"  Block reason       : "
                f"{account.blocked_reason}"
            )


def print_rate_limit_details(
    info: dict[str, Any],
) -> None:

    print_divider(
        "OPENROUTER RATE LIMIT INFORMATION"
    )

    fields = [
        ("category", "Classification"),
        ("status_code", "Status code"),
        ("limit_source", "Limit source"),
        ("provider_name", "Provider"),
        ("limit", "Limit"),
        ("remaining", "Remaining"),
        ("reset", "Reset"),
    ]

    for key, label in fields:
        print(
            f"{label:<18}: "
            f"{info.get(key)}"
        )

    if info.get(
        "reset_timestamp"
    ):
        print(
            f"{'Reset UTC':<18}: "
            f"{utc_string(info['reset_timestamp'])}"
        )

    if info.get(
        "remedy"
    ):
        print(
            f"{'Remedy':<18}: "
            f"{info['remedy']}"
        )


def print_pool_summary(
    accounts: list[Account],
) -> None:

    available = sum(
        account.can_route()
        for account in accounts
    )

    print_divider(
        "POOL SUMMARY"
    )

    print(
        f"Accounts configured : "
        f"{len(accounts)}"
    )

    print(
        f"Accounts available  : "
        f"{available}"
    )

    print(
        f"Configured RPM      : "
        f"{sum(a.rpm_limit for a in accounts)}"
    )

    print(
        f"Configured RPD      : "
        f"{sum(a.rpd_limit for a in accounts)}"
    )