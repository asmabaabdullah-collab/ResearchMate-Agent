import os
import json
import re
from openai import OpenAI
#Embedding Model : تحويل النصوص إلى تمثيل رقمي (Vector Representation)
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

#دالة الاتصال بالمودل opennai
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")
    return OpenAI(api_key=api_key)

# دالة استخراج ملف جيسون من رد النموذج 
def extract_json_from_text(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))

    raw = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if raw:
        return json.loads(raw.group(1))

    raise ValueError("Could not parse JSON from model response.")

#دالة ترسل الطلب الى النموذج وترجع لي نص 
def call_llm_text(system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.2):
    client = get_client()
    model = model or DEFAULT_CHAT_MODEL

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

#تطلب من النموذج الرد ومن النص تستخرج رد بصيغة جيسون
def call_llm_json(system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.1):
    text = call_llm_text(system_prompt, user_prompt, model=model, temperature=temperature)
    return extract_json_from_text(text)
