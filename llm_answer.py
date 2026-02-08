import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def answer(question, matches):
    """
    Generates a beautifully formatted answer using the retrieved context.
    """

    # 1. Prepare Context with Page References
    # We add a clear separator between chunks
    context_str = "\n---\n".join([
        f"Source (Page {m['metadata']['page']}):\n{m['metadata']['text']}"
        for m in matches
    ])

    # 2. Define System Prompt (The "Brain" & "Style" instructions)
    system_instruction = """
    You are an intelligent and precise Legal Assistant. 
    Your goal is to answer the user's question based ONLY on the provided Context.
    
    ### Formatting Guidelines (Make it look beautiful):
    1. **Structure:** Use clear Markdown headers (###) to separate different parts of the answer.
    2. **Readability:** Use bullet points for lists and **bold text** for key legal terms, Article numbers, or emphasis.
    3. **Tone:** Maintain a professional, neutral, and legal tone.
    4. **Citations:** Briefly mention the source page number if relevant (e.g., "[Page 12]").
    5. **Fallout:** If the answer is not in the context, strictly state: "I could not find the answer in the provided document." Do not hallucinate.
    """

    # 3. Define User Prompt
    user_prompt = f"""
    Context:
    {context_str}

    Question: 
    {question}
    """

    # 4. Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo" if you want cheaper/faster
        temperature=0,   # Keep it factual
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content, response.usage