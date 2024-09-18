from fastapi import FastAPI, Form
from groq import Groq
import os

os.environ["GROQ_API_KEY"]="gsk_WYYHFWuoruo2c8BEdOiBWGdyb3FYx77sJa8DG2WknFlDYiqwcPFwTsqwwe"
app = FastAPI()
@app.post("/ask_anything/")
async def recommendations(query: str = Form(...)):
    
    client = Groq()  # Assuming you have already defined Groq() somewhere
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": query
            },
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    text_content = ''
    for chunk in completion:
        text_content += chunk.choices[0].delta.content or ""
    return text_content

