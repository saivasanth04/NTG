from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ntg.router.config import Account
from ntg.router.scheduler import RoundRobin


def routes():
    result = []
    for provider, count in [('gemini', 2), ('groq', 3), ('cohere', 1)]:
        for number in range(1, count + 1):
            a = Account(provider, f'{provider}-{number}', 'unused')
            for m in ('first', 'second'):
                result.append((a, m))
    return result


class SchedulerTests(unittest.TestCase):
    order = ['gemini', 'groq', 'cohere']

    def test_equal_providers_accounts_and_models_with_unequal_pools(self):
        scheduler = RoundRobin()
        selected = [scheduler.plan(routes(), self.order)[0] for _ in range(36)]
        self.assertEqual(Counter(a.provider for a, m in selected), {'gemini':12, 'groq':12, 'cohere':12})
        for provider in self.order:
            counts = Counter(a.id for a, m in selected if a.provider == provider)
            self.assertEqual(len(set(counts.values())), 1)
        for identity in {a.id for a, m in selected}:
            counts = Counter(m for a, m in selected if a.id == identity)
            self.assertEqual(len(set(counts.values())), 1)

    def test_fallback_order_remains_hierarchical(self):
        scheduler = RoundRobin()
        scheduler.plan(routes(), self.order)
        plan = scheduler.plan(routes(), self.order)
        self.assertEqual([(a.id, m) for a, m in plan[:6]], [('groq-1','first'),('groq-1','second'),
            ('groq-2','first'),('groq-2','second'),('groq-3','first'),('groq-3','second')])

    def test_recreated_scheduler_keeps_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'positions.json'
            selected = [RoundRobin(path).plan(routes(),self.order)[0] for _ in range(3)]
            self.assertEqual([a.provider for a, m in selected],self.order)
            self.assertNotIn('unused',path.read_text())

    def test_concurrent_instances_reserve_distinct_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'positions.json'
            with ThreadPoolExecutor(max_workers=6) as pool:
                selected = list(pool.map(lambda _:RoundRobin(path).plan(routes(),self.order)[0],range(36)))
            self.assertEqual(Counter(a.provider for a,m in selected),{'gemini':12,'groq':12,'cohere':12})
            self.assertEqual(Counter((a.id,m) for a,m in selected)[('cohere-1','first')],6)

    def test_separate_processes_keep_rotation(self):
        module = Path(__file__).resolve().parents[1] / 'ntg/router/scheduler.py'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'positions.json'
            code = '''import importlib.util, sys
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('scheduler',sys.argv[1])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
routes=[(SimpleNamespace(id=p+'-1',provider=p),m) for p in ('gemini','groq') for m in ('first','second')]
a,m=module.RoundRobin(sys.argv[2]).plan(routes,['gemini','groq'])[0]
print(a.id,m)
'''
            outputs = [subprocess.check_output([sys.executable,'-c',code,str(module),str(path)],text=True).strip() for _ in range(4)]
            self.assertEqual(outputs,['gemini-1 first','groq-1 first','gemini-1 second','groq-1 second'])

    def test_explicit_model_and_provider_restrictions(self):
        scheduler=RoundRobin()
        limited=[(a,m) for a,m in routes() if a.provider=='groq']
        selected=[scheduler.plan(limited,self.order,'second')[0] for _ in range(3)]
        self.assertEqual([a.id for a,m in selected],['groq-1','groq-2','groq-3'])
        self.assertTrue(all(m=='second' for a,m in selected))

    def test_removed_routes_do_not_break_rotation(self):
        scheduler=RoundRobin()
        scheduler.plan(routes(),self.order)
        remaining=[(a,m) for a,m in routes() if a.provider!='gemini' and m=='second']
        plan=scheduler.plan(remaining,self.order)
        self.assertTrue(plan)
        self.assertTrue(all(a.provider!='gemini' and m=='second' for a,m in plan))


if __name__ == '__main__':
    unittest.main()
