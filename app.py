from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def on_submit():
    query = request.form['query']
    response = generate_response(query)
    response_message = f"SupportBot Response: {response}"
    return jsonify({'query': query, 'response': response_message})

def generate_response(query):
    # Respond only to feedback/review questions and avoid unrelated AI discussions
    qa_prompt = (
        "You are a customer support bot specializing in addressing customer reviews and feedback. "
        "Your primary role is to acknowledge customer reviews, whether positive or negative, with empathy and professionalism. "
        "For positive reviews, express gratitude and encouragement. For negative reviews, apologize sincerely, "
        "offer a resolution if applicable, and assure the customer their feedback is valued. Avoid discussing technical AI-related topics or "
        "anything outside the scope of customer support. Ensure responses are concise, polite, and focused on resolving the customer's concerns."
    )
    input_text = f"{qa_prompt}\nCustomer review or feedback:\n{query}"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    result = llm.invoke(input_text)
    return result.content

if __name__ == '__main__':
    app.run(debug=True)
