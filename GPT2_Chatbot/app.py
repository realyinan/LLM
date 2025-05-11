from flask import Flask, render_template, request, jsonify
from flask_predict import model_predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'answer': '请输入一个问题。'})

    answer = model_predict(question)
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)
