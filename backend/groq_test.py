from openai import OpenAI
from app.config import settings

print("Key starts with:", settings.openai_api_key[:4], "| length:", len(settings.openai_api_key))
print("Base URL:", settings.llm_base_url)
print("Model:", repr(settings.llm_model))

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.llm_base_url, max_retries=0, timeout=30)
try:
    r = client.chat.completions.create(model=settings.llm_model, messages=[{"role": "user", "content": "Say hello in 3 words."}])
    print("WORKS:", r.choices[0].message.content)
except Exception as e:
    print("ERROR:", str(e)[:400])
