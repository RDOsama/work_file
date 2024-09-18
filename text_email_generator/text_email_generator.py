from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn
from groq import Groq
import os

os.environ["GROQ_API_KEY"]="gsk_WYYHFWuoruo2c8BEdOiBWGdyb3FYx77sJa8DG2WknFlDYiqwcPFwTsqwwe"
app = FastAPI()

class FormData(BaseModel):
    emp_name: str
    request_type: str
    date_range_start: str
    date_range_end: str
    request_day_type: str
    Subj: str

@app.post("/submit/")
async def submit_form(
    emp_name: str = Form(...),
    request_type: str = Form(...),
    date_range_start: str = Form(...),
    date_range_end: str = Form(...),
    request_day_type: str = Form(...),
    Subj: str = Form(...)
):
    form_data = FormData(
        emp_name=emp_name,
        request_type=request_type,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        request_day_type=request_day_type,
        Subj=Subj
    )
    print('***** request from: ', form_data.emp_name,'\n')
    prom = f'write a {form_data.request_type} for {form_data.request_day_type} email from {form_data.date_range_start} - {form_data.date_range_end} for {form_data.Subj} from {form_data.emp_name} to Dear Sir/Madam with Subj:{form_data.Subj} and best regards by {form_data.emp_name}'    
    client = Groq()  # Assuming you have already defined Groq() somewhere
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": prom
            },
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    email_content = ''
    for chunk in completion:
        email_content += chunk.choices[0].delta.content or ""


    subj_index = email_content.find("Subj")
    if subj_index != -1:
        email_content = email_content[subj_index:]

    print('email ------------------------------', email_content)
    return email_content, prom


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

