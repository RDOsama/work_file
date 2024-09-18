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
os.environ["GROQ_API_KEY"] = "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"
client = Groq()

MODEL = "llama-3.1-70b-versatile"
TEMPERATURE = 0
MAX_TOKENS = 4096

SYSTEM_PROMPT = """
You're an assistant knowledgeable about female health related questions. Only answer female health-related questions, If user ask something else - just reply with an excuse that you
can answer only female health-related questions. Keep your answers brief and to the point. Be kind and respectful.

"""


def call_model(query: str, messages: List) -> ChatCompletionMessage:
    messages.append(
        {
            "role": "user",
            "content": query,
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
    st.title("Female Health Care")
    # st.write("Ask your question below:")

    user_query = st.text_input("Ask anything:")

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
