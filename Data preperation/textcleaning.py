import os
import re

def remove_whitespace(text):
    # Remove leading and trailing whitespace
    return text.strip()

def ensure_sentence_per_line(text):
    # Ensure each sentence is on a new line
    sentences = re.split(r'(?<=[.!?]) +', text)
    return '\n'.join(sentences)

def standardize_quotes(text):
    # Replace different forms of quotation marks with a standardized one
    text = text.replace('“', "'").replace('”', "'")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace('"', "'").replace('"', "'")
    return text

def convert_to_lowercase(text):
    return text.lower()

def merge_broken_lines(text):
    # Split the text into lines
    lines = text.split('\n')
    merged_lines = []
    buffer = ""
    
    for line in lines:
        if buffer:
            buffer += " " + line.strip()
        else:
            buffer = line.strip()
        
        # Check if the line ends with a sentence-ending punctuation
        if re.search(r'[.!?]$', buffer):
            merged_lines.append(buffer)
            buffer = ""
    
    if buffer:
        merged_lines.append(buffer)
    
    return '\n'.join(merged_lines)

def split_quoted_sentences(text):
    # Use regex to find instances where a sentence ends with a quote, followed by whitespace and a new quote
    text = re.sub(r"(\'.*?\')\s+(\'[^\'].*?\')", r'\1\n\2', text)
    return text

def clean_text(text):
    text = remove_whitespace(text)
    text = ensure_sentence_per_line(text)
    text = convert_to_lowercase(text)
    text = merge_broken_lines(text)
    text = standardize_quotes(text)
    text = split_quoted_sentences(text)
    return text

def process_files(root_dir):
    for foldername in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()

                    cleaned_text = clean_text(text)

                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(cleaned_text)

if __name__ == "__main__":
    root_directory = './Holdout sample'
    process_files(root_directory)
