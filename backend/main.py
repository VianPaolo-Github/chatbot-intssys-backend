import os
import json
from flask import Flask, request, make_response
from flask_cors import CORS
from chatbot import ChatbotAssistant
from docx import Document
from PyPDF2 import PdfReader
from transformers import pipeline
from essay_analysis import read_essay, summarize_essay, extract_keywords

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Load chatbot model
assistant = ChatbotAssistant(
    intents_path=os.path.join(os.path.dirname(__file__), 'intents.json'),
)
assistant.parse_intents()
assistant.load_model(
    model_path=os.path.join(os.path.dirname(__file__), 'chatbot_model.pth'),
    dimensions_path=os.path.join(os.path.dirname(__file__), 'dimensions.json'),
)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    message = data.get("message")

    if not message:
        return json_response({"reply": "No message received."}, 400)

    reply = assistant.process_message(message)
    return json_response({"reply": reply})

# Essay summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/upload-essay', methods=['POST'])
def upload_essay():
    if 'file' not in request.files:
        return json_response({'error': 'No file part'}, 400)

    file = request.files['file']
    if file.filename == '':
        return json_response({'error': 'No selected file'}, 400)

    try:
        essay_text = read_essay(file)
        summary = summarize_essay(essay_text)
        suggested_courses = extract_keywords(summary)

        if suggested_courses:
            course_list = ", ".join(suggested_courses[:-1])
            if len(suggested_courses) > 1:
                course_list += f", and {suggested_courses[-1]}"
            else:
                course_list = suggested_courses[0]

            response_text = (
                f"It looks like you're interested in topics such as {course_list}. "
                f"You might want to explore bachelor programs in {course_list}."
            )
        else:
            response_text = (
                "Thanks for your inquiry! I couldn’t detect specific topics, "
                "but feel free to tell me more about your interests, and I’ll suggest a program."
            )

        return json_response({
            "summary": summary,
            "suggested_courses": suggested_courses,
            "response": response_text
        })

    except Exception as e:
        return json_response({'error': f"Failed to process essay: {str(e)}"}, 500)


# 🔧 Custom response function with UTF-8-safe JSON
def json_response(payload, status=200):
    response = make_response(json.dumps(payload, ensure_ascii=False), status)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

if __name__ == "__main__":
    app.run(debug=False, port=5000)
