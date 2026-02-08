import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def answer(question, matches):

    ctx = "\n\n".join([
        f"[page {m['metadata']['page']}]\n{m['metadata']['text']}"
        for m in matches
    ])

    prompt = f"""
Answer only from context.
If missing say NOT FOUND.

Context:
{ctx}

Question: {question}
"""

    r = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    return r.choices[0].message.content, r.usage
