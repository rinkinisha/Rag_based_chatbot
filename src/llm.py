import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


def create_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    client = Groq(api_key=api_key)

    return client


def generate_answer(client, question, retrieved_documents, history=None):

    if history is None:
        history=[]

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    conversation=""
    for message in history:
        role=message.get("role" , "")
        content=message.get("content","")

        conversation+= f"{role}:{content}\n"

    prompt = f"""
You are a helpful knowledge-base assistant.

Answer the user's question using ONLY the provided knowledge-base context.

Use the conversation history to understand references
such as "it", "they", "this", or follow-up questions.

If the answer cannot be found in the knowledge-base context,
say:

"I don't know based on the provided knowledge base."

Do not make up information.

Conversation History:
{conversation}

Knowledge Base Context:
{context}

Current Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content