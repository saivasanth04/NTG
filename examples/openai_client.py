import os
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.environ["NTG_GATEWAY_KEY"],
                max_retries=0, timeout=60)
result = client.chat.completions.create(model="auto", messages=[
    {"role": "user", "content": "Explain this function: def add(a, b): return a + b"}], max_tokens=256)
print(result.choices[0].message.content)
