import spacy
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_text
from io import BytesIO
import re

app = FastAPI()

nlp = spacy.load("en_core_web_md")

nlp = spacy.load("en_core_web_md")

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def normalize_skill(skill):
    # Remove numeric parts from the skill (e.g., "python3" -> "python")
    return re.sub(r'\d+', '', skill).strip().lower()

def find_skills_in_text(text, skills_list):
    doc = nlp(text.lower())  # Normalize text to lowercase
    skills_found = [skill for skill in skills_list if normalize_skill(skill) in doc.text]
    return skills_found

def extract_contact_info(text):
    # Regular expression for email
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    # Regular expression for Pakistani phone numbers (e.g., +92xxxxxxxxxx, 03xxxxxxxxx, +92-xxxxxxxxxx, +92 xxxxxxxxxx)
    phone_pattern = re.compile(r'\+92-\d{10}|\+92\d{9}|\+03\d{2}-\d{7}|03\d{9}|\+92\s?\d{10}')
    # Extract email and phone number
    email = email_pattern.findall(text)
    phone = phone_pattern.findall(text)

    # Extract name from the first line and first two words of the second line
    lines = text.split('\n')
    name = None
    if lines:
        first_line = lines[0].strip()
        if len(lines) > 1:
            second_line_words = lines[1].strip().split()[:2]
            if "ENGINEER" not in second_line_words:  # Check if the second line contains "ENGINEER"
                name = f"{first_line} {' '.join(second_line_words)}".strip()
            else:
                name = first_line
    
    email = email[0] if email else None
    phone = phone[0] if phone else None

    return name, email, phone

###### extract experience
def split_list_items(input_list):
    output_list = []
    for item in input_list:
        split_items = item.split('\n')
        # print('item', item)
        # print('split_items', split_items)
        output_list.extend(split_items)
    return output_list
def extract_experience_section(resume_text):
    resume_headings = [
        "education",
        "contact information",
        "contact",
        "objective or summary",
        "skills",
        "certifications",
        "projects",
        "volunteer work",
        "awards and honors",
        "languages",
        "summary",
        "skills &",
        "expertise",
        "languagues",
        "interests"
    ]

    resume_text_list = resume_text.split("\n\n")
    resume_text_list = [item.strip() for item in resume_text_list]
    # resume_text_list = [item.split("\n") for item in resume_text_list]
    resume_text_list = split_list_items(resume_text_list)
    resume_text_list = [item.replace('\xa0', ' ') for item in resume_text_list]
    resume_text_list = [item.lower() for item in resume_text_list]
    
    result_string = ' '.join(resume_text_list)

    
    # print('resume_text_list:   ', resume_text_list)
    exp_sen = ""
    if "experience" in resume_text_list:
        exp_ind = resume_text_list.index('experience')
        for section in resume_text_list:
            if section.lower() in resume_headings:
                sec_ind = resume_text_list.index(section)
                # print('******* resume_text_list[sec_ind]', resume_text_list[sec_ind])
                # print('******* resume_text_list[exp_ind]', resume_text_list[exp_ind])
                
                # print('******* exp_ind', exp_ind)
                # print('******* sec_ind', sec_ind)
                
                if exp_ind < sec_ind:
                    only_exp = resume_text_list[exp_ind:sec_ind]
                    # print('####### if only_exp: ', only_exp)
                else:
                    only_exp = resume_text_list[exp_ind:]
                    exists = any(item in only_exp for item in resume_headings)
                    if exists:
                        # print('if exists', exists)
                    
                        # print('####### else only_exp: ', only_exp)
                        # if "experience" in only_exp:
                        exp_ind2 = only_exp.index('experience')
                        for j in only_exp:
                            if j.lower() in resume_headings:
                                # print('sub section: ', j)
                                sec_ind2 = only_exp.index(j)
                                # print('******* 2222resume_text_list[sec_ind]', only_exp[sec_ind2])
                                # print('******* 2222resume_text_list[exp_ind]', only_exp[exp_ind2])
                                
                                # print('******* 2222exp_ind', exp_ind2)
                                # print('******* 2222sec_ind', sec_ind2)
                                if exp_ind2 < sec_ind2:
                                    only_exp2 = only_exp[exp_ind2:sec_ind2]
                                    # print('*************** *####### if only_exp: ', only_exp2)
                                    # exp_sen = ', '.join(only_exp2)
                                    # print('******** only_exp2', only_exp2, '\n\n')
                                    # print('only_exp2', only_exp2)
                                    return only_exp2
                                break
                                    # if only_exp2:
                                    #     exp_sen = ', '.join(only_exp2)
                                    # break
                            
                    else:
                        # exp_sen = ', '.join(only_exp)
                        return only_exp
                    break
    else:
        # exp_sen = '\n'.join(resume_text_list)
        resume_text_list

# Function to convert months to lowercase
def convert_months_to_lowercase(pattern):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for month in months:
        pattern = re.sub(month, month.lower(), pattern)
    return pattern
    
def calculate_months(date_range):
    # Remove unnecessary characters and split the date range
    date_range = date_range[0]
    start_date_str, end_date_str = date_range.split(' - ')

    # Define a dictionary to convert month names to numbers
    month_dict = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 
        'may': 5, 'june': 6, 'july': 7, 'august': 8, 
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    # Parse the start date
    start_month_str, start_year_str = start_date_str.split(' ')
    start_month = month_dict[start_month_str.lower()]
    start_year = int(start_year_str)
    start_date = datetime(start_year, start_month, 1)

    # Parse the end date
    if end_date_str.lower() == 'present':
        end_date = datetime.now()
    else:
        end_month_str, end_year_str = end_date_str.split(' ')
        end_month = month_dict[end_month_str.lower()]
        end_year = int(end_year_str)
        end_date = datetime(end_year, end_month, 1)

    # Calculate the difference in months
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    return total_months



def extract_experience(resume_text):
    resume_text_list = extract_experience_section(resume_text)
    result_string = ' '.join(resume_text_list)
    # print('########### result_string', result_string, '\n\n\n')
    
    
    # Regular expression to match the format "Month Year - Month Year"
    date_pattern1 = r'(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4} - present'
    
    date_pattern2 = r'(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4} - (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}'
    
    date_pattern3 = r'(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}'
    
    # Convert patterns
    date_pattern1 = convert_months_to_lowercase(date_pattern1)
    date_pattern2 = convert_months_to_lowercase(date_pattern2)
    date_pattern3 = convert_months_to_lowercase(date_pattern3)
    
    
    
    combined_pattern = f'({date_pattern1})|({date_pattern2})|({date_pattern3})'
    
    
    matches = re.findall(combined_pattern, result_string)
    
    matches = [item for tup in matches for item in tup if item]
    
    # Print matches
    # print('\n\n********* matches: ', matches, '\n\n')
    
    
    
    months = 0
    print('******** matches', matches)
    for i in matches:
        months = calculate_months([i]) + months
    years, months = divmod(months, 12)
    print('years, months', years, months)
    return years, months
###### extract experience

@app.post("/single_resume")
async def create_upload_file(skills_required: str = Form(...), resume_file: UploadFile = File(...)):
    try:
        resume_contents = await resume_file.read()
        resume_pdf = BytesIO(resume_contents)
        text = extract_text_from_pdf(resume_pdf)

        # Split skills_required by commas first, then split by slashes
        skills_list = [skill.strip() for skill_with_slash in skills_required.split(',') for skill in skill_with_slash.split('/')]
        lower_skill_list = [normalize_skill(s) for s in skills_list]
        skills = set(lower_skill_list)
        skills_extracted = set(find_skills_in_text(text, skills))
        
        # Extract contact information
        name, email, phone = extract_contact_info(text)
        
        # Print the extracted contact information
        print(f"Extracted Name: {name}")
        print(f"Extracted Email: {email}")
        print(f"Extracted Phone: {phone}")

        # Experience
        years, months = extract_experience(text)
        experience = f'{years} years {months} months'
        print(f"Total Experience: {experience}")

        if skills_extracted:
            skill_rate = round((len(skills_extracted) / len(skills)) * 100, 2)
            response_data = {
                "success": True,
                "responseMessage": "Skills Matched",
                "responseCode": "200",
                "data": {
                    "skills_required": skills_required,
                    "confidence": f'{skill_rate}%',
                    "name": name,
                    "email": email,
                    "phone": phone if phone else "no contact info",  # Set phone to "no contact info" if phone is None
                    "experience": experience
                }
            }
        elif email is None and phone is None:
            response_data = {
                "success": False,
                "responseMessage": "No contact info",
                "responseCode": "404",
                "data": {}
            }
        else:
            response_data = {
                "success": True,
                "responseMessage": "Skill Not Matched",
                "responseCode": "200",
                "data": {
                    "skills_required": skills_required,
                    "confidence": "0.0%",
                    "name": name,
                    "email": email,
                    "phone": phone if phone else "no contact info",  # Set phone to "no contact info" if phone is None
                    "experience": experience
                }
            }
    except Exception as e:
        response_data = {
            "success": False,
            "responseMessage": f"Error: {str(e)}",
            "responseCode": "500",
            "data": {}
        }

    return JSONResponse(content=response_data)



@app.post("/bulk_resume")
async def create_upload_files(skills_required: str = Form(...), resume_files: list[UploadFile] = File(...)):
    results = {}
    id_counter = 1  # Initialize the ID counter
    
    for resume_file in resume_files:
        try:
            resume_contents = await resume_file.read()
            resume_pdf = BytesIO(resume_contents)
            text = extract_text_from_pdf(resume_pdf)

            # Split skills_required by commas first, then split by slashes
            skills_list = [skill.strip() for skill_with_slash in skills_required.split(',') for skill in skill_with_slash.split('/')]
            lower_skill_list = [normalize_skill(s) for s in skills_list]
            skills = set(lower_skill_list)
            skills_extracted = set(find_skills_in_text(text, skills))
            
            # Extract contact information
            name, email, phone = extract_contact_info(text)
            
            # Calculate confidence before printing
            if skills_extracted:
                skill_rate = round((len(skills_extracted) / len(skills)) * 100, 2)
            
            result = {
                "name": name,
                "email": email if email else "no contact info",
                "phone": phone if phone else "no contact info",
                "confidence": f'{skill_rate}%' if skills_extracted else "0.0%"
            }
            
            # Add result to dictionary
            results[result["name"]] = [result["name"], result["confidence"], result["email"], result["phone"]]
            id_counter += 1  # Increment the ID counter
            
            # Print the extracted information
            print(f"Name: {name}, Confidence: {result['confidence']}, Email: {email}, Phone: {phone if phone else 'no contact info'}")
        except Exception as e:
            # Handle exceptions
            results[str(id_counter)] = [None, "0.0%", "no contact info", "no contact info"]
            id_counter += 1  # Increment the ID counter
            print(f"Error: {str(e)}")

    return JSONResponse(content=results)
