"""Command-line interface and console diagnostic output for NTG."""

from ntg.cli.app import configure_console_encoding, main, run_cli
from ntg.cli.diagnostics import (
    print_account_status,
    print_banner,
    print_divider,
    print_rate_limit_details,
    print_request_execution,
)

__all__ = [
    "main",
    "run_cli",
    "configure_console_encoding",
    "print_divider",
    "print_banner",
    "print_account_status",
    "print_request_execution",
    "print_rate_limit_details",
]
