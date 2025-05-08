from transformers import pipeline
import PyPDF2
import docx

# course mapping
COURSE_SUGGESTIONS = {
    "Business Administration": ["Bachelor of Science in Accountancy", "Bachelor of Science in Business Administration"], 
    "Accountancy": ["Bachelor of Science in Accountancy", "Bachelor of Science in Business Administration"],

    "Criminal Justice": ["Bachelor of Science in Criminology"],

    "Public Safety": ["Bachelor of Science in Criminology"],

    "Dentistry":["Doctor of Dental Medicine (DMD)"],

    "Engineering":["Bachelor of Science in Architecture", "Bachelor of Science in Civil Engineering", "Bachelor of Science in Computer Engineering", "Bachelor of Science in Electrical Engineering", "Bachelor of Science in Electronics Engineering", "Bachelor of Science in Mechanical Engineering"],
    "Architecture":["Bachelor of Science in Architecture", "Bachelor of Science in Civil Engineering", "Bachelor of Science in Computer Engineering", "Bachelor of Science in Electrical Engineering", "Bachelor of Science in Electronics Engineering", "Bachelor of Science in Mechanical Engineering"],
    
    "Information Technology":["Bachelor of Science in Computer Science", "Bachelor of Science in Information Technology"],

    "International Hospitality & Tourism Management":["Bachelor of Science in Hospitality Management","Bachelor of Science in Tourism Management"],

    "Law":["Bachelor of Laws (LL.B)"], 

    "Natural Sciences":["Bachelor of Science in Biology","Bachelor of Science in Chemistry","Bachelor of Science in Psychology"], 

    "Nursing":["Bachelor of Science in Nursing"],

    "Teacher Education":["Bachelor of Elementary Education (BEEd)","Bachelor of Secondary Education (BSEd), multiple majors (English, Math, Science, Social Studies)","Bachelor of Arts in Communication Studies","Bachelor of Arts in English Language","Bachelor of Arts in Political Science"],
    "Liberal Arts":["Bachelor of Elementary Education (BEEd)","Bachelor of Secondary Education (BSEd), multiple majors (English, Math, Science, Social Studies)","Bachelor of Arts in Communication Studies","Bachelor of Arts in English Language","Bachelor of Arts in Political Science"]
}

summarizer = pipeline("summarization")

def summarize_essay(text):
    # Basic summarization using Hugging Face model
    result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return result[0]['summary_text']

def extract_course_recommendations(summary):
    summary = summary.lower()
    courses = []
    for keyword, suggestions in COURSE_SUGGESTIONS.items():
        if keyword in summary:
            courses.extend(suggestions)
    return list(set(courses))  # Avoid duplicates

def extract_keywords(summary):
    summary = summary.lower()
    found_topics = []
    for topic in COURSE_SUGGESTIONS:
        if topic in summary:
            found_topics.extend(COURSE_SUGGESTIONS[topic])
    return found_topics



def read_essay(file):
    filename = file.filename.lower()

    if filename.endswith('.txt'):
        return file.read().decode('utf-8')

    elif filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text

    elif filename.endswith('.docx'):
        doc = docx.Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])

    else:
        return "Unsupported file format."