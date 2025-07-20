import re
import csv
import os
import argparse



def extract_questions(text):
    """
    Extract questions from a given text using regex.
    Supports both numbered and paragraph formatting.
    """
    question_regex = r'\d+\.\s*(.+?)(?=\n\d+\.|$)'
    questions = re.findall(question_regex, text, re.DOTALL)
    clean_questions = [re.sub(r'\*([^*]+)\*', r'\1', question.strip()) for question in questions]
    return clean_questions

def create_csv(questions, output_path):
    """
    Create a CSV file from a list of questions.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write questions to CSV
        with open(output_path, mode='w', encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['Question'])
            writer.writeheader()
            for question in questions:
                writer.writerow({'Question': question})
        
        print(f"CSV file successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")


# Set up argument parsing
parser = argparse.ArgumentParser(description="Process text input.")
parser.add_argument("--text", type=str, required=True, help="Text input from Streamlit")
args = parser.parse_args()

# Use the text variable
text = args.text


questions = extract_questions(text)

output_path = r'C:\Users\aksha\Llama2-Medical-Chatbot-main\Llama2-DocumentChatbot-main\output\questions.csv'

# Create the CSV file
create_csv(questions, output_path)

print("Questions extracted:", questions)
