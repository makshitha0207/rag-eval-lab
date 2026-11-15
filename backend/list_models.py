from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.llm_base_url, max_retries=0, timeout=30)
for m in client.models.list():
    print(m.id)
