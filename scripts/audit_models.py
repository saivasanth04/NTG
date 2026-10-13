"""Read-only model catalogs for configured credentials; never logs secrets."""
import asyncio
import json
import re
import os
from pathlib import Path
from datetime import datetime, timezone
import httpx
from dotenv import dotenv_values


async def main():
    values = {**dotenv_values('.env'), **os.environ}
    semaphore = asyncio.Semaphore(5)
    async with httpx.AsyncClient(timeout=25) as client:
        async def catalog(name, key):
            provider, number = re.fullmatch(r'(GEMINI|GROQ|COHERE|CLOUDFLARE|OPENROUTER)_API_KEY_(\d+)', name).groups()
            provider = provider.lower()
            headers = {'Authorization': f'Bearer {key}'}
            if provider == 'gemini':
                url = 'https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000'
                headers = {'x-goog-api-key': key}
            elif provider == 'groq':
                url = 'https://api.groq.com/openai/v1/models'
            elif provider == 'cohere':
                url = 'https://api.cohere.com/v1/models?page_size=1000'
            elif provider == 'cloudflare':
                account = values.get(f'CLOUDFLARE_ACCOUNT_ID_{number}', '')
                url = f'https://api.cloudflare.com/client/v4/accounts/{account}/ai/models/search?per_page=100'
            else:
                url = 'https://openrouter.ai/api/v1/models'
                headers = {}
            record = {'provider': provider, 'account': f'{provider}-{number}'}
            try:
                async with semaphore:
                    response = await client.get(url, headers=headers)
                record['status'] = response.status_code
                if response.is_success:
                    body = response.json()
                    record['models'] = body.get('models', body.get('data', body.get('result', [])))
                    record['pagination'] = {k: body[k] for k in ('next_page_token', 'result_info') if k in body}
                    if provider == 'cloudflare':
                        # Cloudflare's total_count can include hidden/retired
                        # rows. Follow pages until the visible catalog ends.
                        page = 2
                        names = {m['name'] for m in record['models']}
                        while page <= 20:
                            async with semaphore:
                                reply = await client.get(url + f'&page={page}', headers=headers)
                            reply.raise_for_status()
                            extra = reply.json().get('result', [])
                            fresh = [m for m in extra if m['name'] not in names]
                            if not fresh:
                                break
                            record['models'].extend(fresh)
                            names.update(m['name'] for m in fresh)
                            page += 1
            except Exception as error:
                record['error'] = type(error).__name__
            return record
        records = await asyncio.gather(*(catalog(k, v) for k, v in values.items()
            if re.fullmatch(r'(GEMINI|GROQ|COHERE|CLOUDFLARE|OPENROUTER)_API_KEY_\d+', k) and v))
    pools = json.loads(Path('docs/free-model-pools.json').read_text(encoding='utf-8'))
    for r in records:
        models = r.pop('models', [])
        r['catalog_count'] = len(models)
        r['catalog_ids'] = [m.get('id', m.get('name', '')).removeprefix('models/') for m in models]
        r['selected_models'] = [m for m in pools[r['provider']] if m in r['catalog_ids']]
        print(r['account'], r.get('status', r.get('error')), r['catalog_count'], len(r['selected_models']))
    Path('docs/model-catalog-audit.json').write_text(json.dumps({
        'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Catalog visibility intersected with the documented text-chat free pools. Not an inference or billing verification.',
        'accounts': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(main())
