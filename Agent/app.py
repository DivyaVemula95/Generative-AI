from flask import Flask, render_template, request, jsonify
from agricultureservice_assistant import AgricultureServiceAssistant
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()  # Add this at the top, before creating the assistant


# Initialize the assistant
assistant = AgricultureServiceAssistant()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_input = data.get('question', '')
    language = data.get('language', 'en')

    try:
        response = assistant.handle_query(user_input, language)
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)