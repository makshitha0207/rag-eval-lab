from app.config import settings


class LLM:
    """Single interface so you can swap providers without touching the rest."""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> tuple[str, dict]:
        if settings.llm_provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            r = client.messages.create(
                model=settings.llm_model,
                max_tokens=1024,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
            return r.content[0].text, usage

        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url or None,
        )
        r = client.chat.completions.create(
            model=settings.llm_model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = {"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}
        return r.choices[0].message.content, usage