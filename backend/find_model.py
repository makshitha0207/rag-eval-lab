from openai import OpenAI

from app.config import settings

print("Key starts with:", settings.openai_api_key[:4], "| length:", len(settings.openai_api_key))
print("Base URL:", settings.llm_base_url)
print("Model in .env:", repr(settings.llm_model))

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.llm_base_url or None)

names = []
try:
    names = [m.id.replace("models/", "") for m in client.models.list()]
except Exception as e:
    print("Could not list models:", str(e)[:200])

flash = [n for n in names if "flash" in n and "image" not in n and "tts" not in n]
print("Flash models found:", flash)

candidates = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"] + flash
seen = set()
for name in candidates:
    if name in seen:
        continue
    seen.add(name)
    try:
        r = client.chat.completions.create(
            model=name,
            messages=[{"role": "user", "content": "Say hello in 3 words."}],
        )
        print("WORKS:", name)
        print("Reply:", r.choices[0].message.content)
        print("Put this in .env:  LLM_MODEL=" + name)
        break
    except Exception as e:
        print("failed:", name, "->", str(e)[:80])
else:
    print("No model worked.")
