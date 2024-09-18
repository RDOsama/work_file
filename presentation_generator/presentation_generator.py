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


SYSTEM_PROMPT = '''You are an AI-Powered Presentation Generator, designed to generate detailed professional PowerPoint presentations with complete explanation of the topics and the topic that user provide must divide in slide by slide sections.
Your primary task is to generate content based on user inputs related to presentation slides. You're always helpful to generate  presenetation only based on the provided topic. If you don't unserstand the topic - just reply with an excuse that you
don't know. Keep your content brief and detailed. Be kind and respectful.'''


# SYSTEM_PROMPT = """
# You are a PowerPoint presentation specialist. You'll get a list of slides, each
# slide containing a title and content. You need to create a PowerPoint presentation
# based on the provided slides.
# But there is a catch: Instead of creating the presentation, provide python code
# that generates the PowerPoint presentation based on the provided slides.
# Use the package python-pptx to create the PowerPoint presentation.
# In your code, use a file called 'template.pptx' as the template for the presentation
# and stick to the template's design.

# If the slides content contains more than one information, make bullet points.

# Your answer should only contain the python code, no explanatory text.

# Slides:

# """


# Streamlit app
def main():
    st.title("Presentation Generator")
    # st.write("Enter Topic:")

    user_query = st.text_input("Enter Topic:", placeholder="Type something in")
    

    if user_query:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        with st.spinner("Generating response..."):
            bot_response = call_model(user_query, messages)

        # st.write("**Bot Response:**")
        st.write(bot_response.content)

if __name__ == "__main__":
    main()



# # Streamlit app
# def main():
#     st.title("Presentation Generator")
#     # st.write("Enter Topic:")

#     user_query = st.text_input("Enter Topic:", placeholder="Type something in")
    

# if __name__ == "__main__":
#     main()
