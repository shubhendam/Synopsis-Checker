import pdfplumber

#load text from .pdf or .txt file
def load_text_from_file(file):
    file_name = file.name.lower()

    #if .txt read lines
    if file_name.endswith(".txt"):
        text = file.read().decode("utf-8")
        return text

    elif file_name.endswith(".pdf"):
        text = extract_text_from_pdf(file)
        return text
    else:
        raise ValueError("Only Input .pdf or .txt files!!")
    
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text