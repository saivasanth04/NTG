import requests

API_KEY = "mstrl_Dvz1SLYePqYzg8MOOuJ0A7jzOpgZinMf_0yQann"

r = requests.post(
    "https://api.mistral.ai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "mistral-medium-3-5",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "max_tokens": 20
    },
    timeout=30
)

print("Status:", r.status_code)
print("\nHeaders:")
for k, v in r.headers.items():
    if "rate" in k.lower() or "limit" in k.lower() or "retry" in k.lower():
        print(k, ":", v)

print("\nResponse:")
print(r.text)