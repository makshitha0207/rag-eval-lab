from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.llm_base_url, max_retries=0, timeout=30)
for name in ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-flash-latest", "gemini-flash-lite-latest"]:
    try:
        r = client.chat.completions.create(model=name, messages=[{"role": "user", "content": "Say hello in 3 words."}])
        print("WORKS:", name, "->", r.choices[0].message.content)
        break
    except Exception as e:
        print("failed:", name, "->", str(e)[:200])
