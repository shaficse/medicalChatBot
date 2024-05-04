from flask import Flask, render_template, request, jsonify
from MedQuard import *

app = Flask(__name__)
# medical_qa = MedicalQA()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']

    # encoded_input = prepare_input(tokenizer, question)
    response =answer_query(question)
    # response = generated_text.split("Answer:")[1].strip()
    # response = medical_qa.ask(question,temperature=0.7, top_p=0.85, repetition_penalty=1.1, do_sample=False)
    return jsonify(answer=response)

if __name__ == '__main__':
    app.run(debug=True)
