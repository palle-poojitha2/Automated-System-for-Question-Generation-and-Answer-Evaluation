import csv
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# Load Google Cloud credentials
SERVICE_ACCOUNT_FILE = r'C:\Users\aksha\Llama2-Medical-Chatbot-main\Llama2-DocumentChatbot-main\document-qa-bot-354bf36329ae.json'
SCOPES = ['https://www.googleapis.com/auth/forms.body', 'https://www.googleapis.com/auth/drive']

def create_form_from_csv(csv_file_path):
    # Authenticate with Google API
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('forms', 'v1', credentials=creds)

    # Parse CSV file
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)  # Use DictReader for header handling
        questions = [row['Question'] for row in reader]  # Extract the 'Question' column

    # Step 1: Create the form with just the title
    form = {
        "info": {
            "title": "Generated Form from CSV",
        }
    }
    
    form_response = service.forms().create(body=form).execute()
    form_id = form_response.get('formId')
    print(f"Form created with ID: {form_id}")

    # Step 2: Add questions to the form using batchUpdate
    requests = []
    for question in questions:
        requests.append({
            "createItem": {
                "item": {
                    "title": question,
                    "questionItem": {
                        "question": {
                            "required": False,
                            "textQuestion": {}  # Specify the kind of question (text-based)
                        }
                    }
                },
                "location": {
                    "index": 0  # Add at the start (reverse order insertion)
                }
            }
        })

    batch_update_body = {"requests": requests}
    service.forms().batchUpdate(formId=form_id, body=batch_update_body).execute()

    # Get the responder URL
    form_response = service.forms().get(formId=form_id).execute()
    print(f"Form responder URL: {form_response.get('responderUri')}")

# Specify your CSV file path
csv_path = r'C:\Users\aksha\Llama2-Medical-Chatbot-main\Llama2-DocumentChatbot-main\output\questions.csv'
create_form_from_csv(csv_path)
