"""Hierarchical round robin; persists positions, never credentials or quotas."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time


def rotate(items, previous):
    """Start after the last selected identity; cope with changed catalogs."""
    if previous not in items:
        return items
    offset = items.index(previous) + 1
    return items[offset:] + items[:offset]


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + 5
        while True:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Routing position file is busy') from None
                time.sleep(.02)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RoundRobin:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.positions = {}

    def plan(self, routes, provider_order, requested_model=None):
        if self.path is None:
            return self._plan(routes, provider_order, requested_model)
        with file_lock(self.path.with_suffix(self.path.suffix + '.lock')):
            if self.path.exists():
                # Invalid state is reported rather than silently restarting
                # every request on the first route.
                self.positions = json.loads(self.path.read_text(encoding='utf-8'))
            result = self._plan(routes, provider_order, requested_model)
            descriptor, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + '.')
            try:
                with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                    json.dump(self.positions, handle)
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return result

    def _plan(self, routes, provider_order, requested_model):
        providers = [p for p in provider_order if any(a.provider == p for a, m in routes)]
        providers = rotate(providers, self.positions.get('provider'))
        # Explicit model selection takes precedence over automatic balancing.
        exact_provider = next((a.provider for a, m in routes
            if requested_model in (m, f'{a.provider}/{m}')), None) if requested_model is not None else None
        if exact_provider:
            providers.remove(exact_provider)
            providers.insert(0, exact_provider)
        ordered = []
        accounts = self.positions.setdefault('accounts', {})
        models = self.positions.setdefault('models', {})
        for provider in providers:
            ids = list(dict.fromkeys(a.id for a, m in routes if a.provider == provider))
            ids = rotate(ids, accounts.get(provider))
            for identity in ids:
                candidates = [(a, m) for a, m in routes if a.id == identity]
                names = rotate([m for a, m in candidates], models.get(identity))
                exact = [m for m in names if requested_model in (m, f'{provider}/{m}')] if requested_model is not None else []
                names = exact + [m for m in names if m not in exact]
                account = candidates[0][0]
                ordered.extend((account, m) for m in names)
        if ordered:
            account, model = ordered[0]
            self.positions['provider'] = account.provider
            accounts[account.provider] = account.id
            models[account.id] = model
        return ordered
