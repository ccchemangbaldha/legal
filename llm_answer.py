import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the Assistant's instructions (The "Brain")
SYSTEM_INSTRUCTION = """
You are an intelligent and precise Legal Assistant. 
Your goal is to answer the user's question based ONLY on the provided Context.

### Formatting Guidelines (Make it look beautiful):
1. **Structure:** Use clear Markdown headers (###) to separate different parts of the answer.
2. **Readability:** Use bullet points for lists and **bold text** for key legal terms, Article numbers, or emphasis.
3. **Tone:** Maintain a professional, neutral, and legal tone.
4. **Citations:** Briefly mention the source page number if relevant (e.g., "[Page 12]").
5. **Fallout:** If the answer is not in the context, strictly state: "I could not find the answer in the provided document." Do not hallucinate.
"""

def get_or_create_assistant():
    """
    Retrieves an existing Assistant or creates a new one.
    In production, you should create the assistant once and store the ID in .env
    """
    # Check if ID is in env (Best Practice)
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
    if assistant_id:
        return assistant_id

    # Otherwise, create a new one (For this demo)
    assistant = client.beta.assistants.create(
        name="Legal RAG Assistant",
        instructions=SYSTEM_INSTRUCTION,
        model="gpt-4o", # or gpt-4o-mini
    )
    return assistant.id

def answer(question, matches, thread_id=None):
    """
    Uses OpenAI Assistants API (Threads) to maintain persistent memory.
    Returns: (answer_text, usage_dict, thread_id)
    """
    
    # 1. Get Assistant ID
    assistant_id = get_or_create_assistant()

    # 2. Manage Thread (Conversation ID)
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
    
    # 3. Prepare Context + Question Payload
    # We inject the RAG context into the message for this specific turn
    context_str = "\n---\n".join([
        f"Source (Page {m['metadata']['page']}):\n{m['metadata']['text']}"
        for m in matches
    ])

    user_message_content = f"""
    Context:
    {context_str}

    Question: 
    {question}
    """

    # 4. Add Message to the Thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message_content
    )

    # 5. Run the Assistant
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    # 6. Retrieve Response
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        # The latest message is at index 0
        answer_text = messages.data[0].content[0].text.value
        
        # Usage stats (Token counts)
        usage = run.usage # Returns a Usage object
        
        return answer_text, usage, thread_id
    else:
        return f"Error: Run status {run.status}", None, thread_id