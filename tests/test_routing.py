import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ntg.router.config import Account
from ntg.router.engine import UnifiedNTGRouter, RoutingError
from ntg.router.state import classify


def account(provider, number=1, models=('first', 'second'), group=None):
    identity = f'{provider}-{number}'
    return Account(provider, identity, 'unused', group=group or identity, models=models)


def error(code=429, text='rate limited', body=None, headers=None):
    result = Exception(text)
    result.status_code = code
    result.body = body
    result.response = SimpleNamespace(headers=headers or {}, status_code=code)
    return result


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_calls_rotate_provider_account_and_model(self):
        calls = []
        async def send(a, m, *args, **kwargs):
            calls.append((a.id, m))
            return SimpleNamespace(model=m)
        router = UnifiedNTGRouter(accounts=[account('gemini'), account('gemini',2), account('groq')], transport=send)
        for _ in range(8):
            await router.acompletion([{'role':'user','content':'Hi'}])
        self.assertEqual(calls, [('gemini-1','first'),('groq-1','first'),
            ('gemini-2','first'),('groq-1','second'),('gemini-1','second'),
            ('groq-1','first'),('gemini-2','second'),('groq-1','second')])

    async def test_no_default_output_cap_and_explicit_cap_preserved(self):
        transport = AsyncMock(return_value=SimpleNamespace(model='first'))
        router = UnifiedNTGRouter(accounts=[account('groq')], transport=transport)
        await router.acompletion([{'role':'user','content':'Hi'}])
        self.assertNotIn('max_tokens', transport.call_args.kwargs)
        self.assertNotIn('max_completion_tokens', transport.call_args.kwargs)
        await router.acompletion([{'role':'user','content':'Hi'}], max_tokens=2048)
        self.assertEqual(transport.call_args.kwargs['max_tokens'], 2048)

    async def test_model_then_account_then_provider(self):
        calls = []
        async def send(a, m, *args, **kwargs):
            calls.append((a.id, m))
            if a.provider != 'groq':
                raise error()
            return SimpleNamespace(model=m)
        router = UnifiedNTGRouter(accounts=[account('groq'), account('gemini', 2), account('gemini')], transport=send)
        result = await router.acompletion([{'role': 'user', 'content': 'Hi'}])
        self.assertEqual(calls, [('gemini-1', 'first'), ('gemini-1', 'second'),
                                 ('gemini-2', 'first'), ('gemini-2', 'second'), ('groq-1', 'first')])
        self.assertEqual(result.attempts, 5)
        self.assertTrue(all(v == 0 for v in router.inflight.values()))

    async def test_shared_project_model_skips_sibling_key_same_model(self):
        calls = []
        async def send(a, m, *args, **kwargs):
            calls.append((a.id, m))
            if a.id == 'gemini-1':
                raise error()
            return SimpleNamespace(model=m)
        router = UnifiedNTGRouter(accounts=[account('gemini', group='project'),
            account('gemini', 2, group='project'), account('gemini', 3)], transport=send)
        await router.acompletion([{'role':'user','content':'Hi'}])
        self.assertEqual(calls, [('gemini-1','first'), ('gemini-1','second'), ('gemini-3','first')])

    async def test_cohere_monthly_cap_skips_all_models(self):
        transport = AsyncMock(side_effect=[error(text='monthly trial quota exceeded'), SimpleNamespace(model='first')])
        router = UnifiedNTGRouter(accounts=[account('cohere'),account('cohere',2)],transport=transport)
        await router.acompletion([{'role':'user','content':'Hi'}])
        self.assertEqual([call.args[0].id for call in transport.call_args_list], ['cohere-1','cohere-2'])

    async def test_cloudflare_task_limit_skips_account(self):
        transport = AsyncMock(side_effect=[error(),SimpleNamespace(model='first')])
        router = UnifiedNTGRouter(accounts=[account('cloudflare'),account('cloudflare',2)],transport=transport)
        # Cloudflare needs its endpoint account ID even in a simulated route.
        for a in router.accounts:
            a.account_id = str(int(a.id[-1])) * 32
        await router.acompletion([{'role':'user','content':'Hi'}])
        self.assertEqual([c.args[0].id for c in transport.call_args_list],['cloudflare-1','cloudflare-2'])

    async def test_openrouter_upstream_block_is_shared_across_keys(self):
        transport=AsyncMock(side_effect=[error(text='temporarily rate-limited upstream'),SimpleNamespace(model='second')])
        router=UnifiedNTGRouter(accounts=[account('openrouter'),account('openrouter',2)],transport=transport)
        await router.acompletion([{'role':'user','content':'Hi'}])
        self.assertIsNotNone(router._block(router.accounts[1],'first'))
        self.assertIsNone(router._block(router.accounts[1],'second'))
        self.assertEqual(transport.call_args.args[1],'second')

    async def test_pinned_model_falls_back_by_default(self):
        transport=AsyncMock(side_effect=[error(),SimpleNamespace(model='first')])
        router=UnifiedNTGRouter(accounts=[account('gemini')],transport=transport)
        await router.acompletion([{'role':'user','content':'Hi'}],model='gemini/second')
        self.assertEqual([c.args[1] for c in transport.call_args_list],['second','first'])

    async def test_explicit_provider_is_respected(self):
        transport=AsyncMock(side_effect=error())
        router=UnifiedNTGRouter(accounts=[account('gemini'),account('groq')],transport=transport)
        with self.assertRaises(RoutingError):
            await router.acompletion([{'role':'user','content':'Hi'}],providers=['groq'])
        self.assertEqual({c.args[0].provider for c in transport.call_args_list},{'groq'})

    async def test_exact_model_and_unlisted_model(self):
        transport=AsyncMock(side_effect=error())
        router=UnifiedNTGRouter(accounts=[account('gemini')],transport=transport)
        with self.assertRaises(RoutingError):
            await router.acompletion([{'role':'user','content':'Hi'}],model='first',allow_model_fallback=False)
        self.assertEqual(transport.await_count,1)
        transport.reset_mock()
        with self.assertRaises(RoutingError):
            await router.acompletion([{'role':'user','content':'Hi'}],model='paid-model',allow_model_fallback=False)
        self.assertEqual(transport.await_count,0)

    async def test_default_budget_reaches_beyond_six_attempts(self):
        models=tuple(f'm{i}' for i in range(8))
        transport=AsyncMock(side_effect=[error() for _ in range(7)]+[SimpleNamespace(model='m7')])
        router=UnifiedNTGRouter(accounts=[account('groq',models=models)],transport=transport)
        result=await router.acompletion([{'role':'user','content':'Hi'}])
        self.assertEqual(result.attempts,8)

    async def test_deadline_releases_reservations(self):
        async def send(*args,**kwargs):
            await asyncio.sleep(1)
        router=UnifiedNTGRouter(accounts=[account('groq')],transport=send)
        with self.assertRaises(RoutingError):
            await router.acompletion([{'role':'user','content':'Hi'}],deadline_seconds=.01)
        self.assertTrue(all(v==0 for v in router.inflight.values()))

    async def test_cooldown_reset_is_respected(self):
        transport=AsyncMock(side_effect=[error(headers={'Retry-After':'60'}),SimpleNamespace(model='second')])
        router=UnifiedNTGRouter(accounts=[account('gemini')],transport=transport)
        await router.acompletion([{'role':'user','content':'Hi'}])
        block=router._block(router.accounts[0],'first')
        self.assertIsNotNone(block)
        with patch('ntg.router.engine.time.time',return_value=block.until+1):
            self.assertIsNone(router._block(router.accounts[0],'first'))

    async def test_stream_failure_does_not_replay(self):
        async def stream():
            yield SimpleNamespace(model='first')
            raise error(code=500)
        transport=AsyncMock(return_value=stream())
        router=UnifiedNTGRouter(accounts=[account('groq')],transport=transport)
        result=await router.acompletion([{'role':'user','content':'Hi'}],stream=True)
        self.assertEqual((await anext(result.response)).model,'first')
        with self.assertRaises(RoutingError):
            await anext(result.response)
        await result.response.aclose()
        self.assertEqual(transport.await_count,1)
        self.assertTrue(all(v==0 for v in router.inflight.values()))


class QuotaTests(unittest.TestCase):
    def test_documented_defaults(self):
        for provider in ('gemini','groq','cohere'):
            self.assertEqual(classify(error(),provider)[2],'group_model')
        for provider in ('cloudflare','openrouter'):
            self.assertEqual(classify(error(),provider)[2],'group')

    def test_structured_shared_quota_overrides_model_default(self):
        body={'error':{'details':[{'violations':[{'quotaId':'RequestsPerProject','quotaDimensions':{'location':'global'}}]}]}}
        self.assertEqual(classify(error(body=body),'gemini')[2],'group')

    def test_structured_model_daily_quota(self):
        body={'error':{'details':[{'violations':[{'quotaId':'RequestsPerDayPerProjectPerModel','quotaDimensions':{'model':'first'}}]},{'retryDelay':'30s'}]}}
        category,retry,scope=classify(error(body=body),'gemini')
        self.assertEqual((category,scope),('long_term_exhaustion','group_model'))
        self.assertIsNone(retry)

    def test_model_mention_does_not_change_shared_scope(self):
        self.assertEqual(classify(error(body={'model':'first'}),'cloudflare')[2],'group')

    def test_groq_rolling_daily_reset_retained(self):
        category,retry,scope=classify(error(text='daily requests exhausted',headers={'x-ratelimit-reset-requests':'2m'}),'groq')
        self.assertEqual((category,scope),('long_term_exhaustion','group_model'))
        self.assertIsNotNone(retry)


if __name__ == '__main__':
    unittest.main()
