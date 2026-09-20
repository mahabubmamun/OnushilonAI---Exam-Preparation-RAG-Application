"""
LLM call, abstracted behind one `generate()` function so the rest of the
app doesn't care which backend is used.

- LLM_PROVIDER=groq   -> hosted, free tier, fast. Needs GROQ_API_KEY.
                         Sign up free at https://console.groq.com
- LLM_PROVIDER=ollama -> fully local, zero external calls. Needs Ollama
                         installed and `ollama pull <model>` run once.
                         https://ollama.com
Both are $0 — pick groq if you want fast answers without needing a GPU
on your own machine/server; pick ollama if you want zero external calls.
"""
import requests

from app.config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_NEW_TOKENS, TEMPERATURE,
)


def _generate_groq(system_prompt: str, user_prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and put it in your .env file."
        )
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_NEW_TOKENS,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _generate_ollama(system_prompt: str, user_prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def generate(system_prompt: str, user_prompt: str) -> str:
    if LLM_PROVIDER == "groq":
        return _generate_groq(system_prompt, user_prompt)
    elif LLM_PROVIDER == "ollama":
        return _generate_ollama(system_prompt, user_prompt)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} (use 'groq' or 'ollama')")
