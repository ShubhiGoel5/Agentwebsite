import json
import os
import sys
import requests

api_key = os.environ.get("OPENROUTER_API_KEY", "")

# Fallback check Windows User registry if set via setx
if not api_key and sys.platform == "win32":
    try:
        import winreg
        reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        val, _ = winreg.QueryValueEx(reg_key, "OPENROUTER_API_KEY")
        if val:
            api_key = val
    except Exception:
        pass

if not api_key:
    print("[ERROR] OPENROUTER_API_KEY environment variable is not set!")
    sys.exit(1)

print("Sending request to OpenRouter using apodex/apodex-1.1-mini:free...")
r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "apodex/apodex-1.1-mini:free",
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
    },
    timeout=60,
)

print(f"Status Code: {r.status_code}")
print("Response Output:")
try:
    print(json.dumps(r.json(), indent=2)[:1500])
except Exception:
    print(r.text[:1500])
