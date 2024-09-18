import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from textwrap import dedent
from typing import List
from fastembed import TextEmbedding
from groq import Groq
from groq.types.chat import ChatCompletionMessage
import streamlit as st
import os

# Set up the Groq API key and client
os.environ["GROQ_API_KEY"] = "gsk_WYYHFWuoruo2c8BEdOiBWGdyb3FYx77sJasdsdee8DG2WknFlDYicPFwTs"
client = Groq()

MODEL = "llama-3.1-70b-versatilwee"
TEMPERATURE = 0
MAX_TOKENS = 4096

# Read text from PDF
def read_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load the PDF
pdf_text = read_pdf("clawset_doc.pdf")

# Prepare the knowledge base from PDF text
KNOWLEDGE_BASE = [
    {"question": "Extracted from PDF", "answer": pdf_text}
]

# Initialize embedding model
embedding_model = TextEmbedding()

# Convert KNOWLEDGE_BASE to a DataFrame
documents_df = pd.DataFrame(KNOWLEDGE_BASE)
documents_df['id'] = documents_df.index

# Create a text column combining question and answer
documents_df['text'] = documents_df['question'] + '\n' + documents_df['answer']

# Embed documents and convert embeddings to lists
document_embeddings = list(embedding_model.embed(documents_df['text'].tolist()))
embedding_dim = document_embeddings[0].shape[0]  # Assuming all embeddings have the same dimension
documents_df['embedding'] = [embedding.tolist() for embedding in document_embeddings]

# Function to retrieve context based on query
def retrieve_context(query: str, k: int = 3) -> str:
    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    
    distances = np.linalg.norm(
        np.array(documents_df['embedding'].tolist()) - np.array(query_embedding),
        axis=1
    )
    
    documents_df['distance'] = distances
    top_k_df = documents_df.nsmallest(k, 'distance')
    
    result = "\n---\n".join([f"{row['question']}\n{row['answer']}" for _, row in top_k_df.iterrows()])
    return result

SYSTEM_PROMPT = """
You're a senior customer support agent for an online e commerce website name clawset.
You're always helpful and answer customer questions only based on the provided
information. If you don't know the answer - just reply with an excuse that you
don't know. Keep your answers brief and to the point. Be kind and respectful.

Use the provided context for your answers. The most relevant information is
at the top. Each piece of information is separated by ---.
"""

def create_user_prompt(query: str) -> str:
    return dedent(
        f"""
Use the following information:

```
{retrieve_context(query)}
```

to answer the question:
{query}
    """
    )



def call_model(query: str, messages: List) -> ChatCompletionMessage:
    messages.append(
        {
            "role": "user",
            "content": create_user_prompt(query),
        },
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    message = response.choices[0].message
    messages.append(message)
    return message

# Streamlit app
def main():
    st.title("Welcome to Customer Support ")
    st.write("Ask your question below:")

    user_query = st.text_input("Your question:")

    if user_query:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        with st.spinner("Generating response..."):
            bot_response = call_model(user_query, messages)

        st.write("**Bot Response:**")
        st.write(bot_response.content)

if __name__ == "__main__":
    main()
