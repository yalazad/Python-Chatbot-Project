from flask import Flask, render_template, jsonify, request
import process_msg


app = Flask(__name__)

app.config['SECRET_KEY'] = 'a-very-secret-key-123456'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())



@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']

        response = process_msg.chatbot_response(the_question)

    return jsonify({"response": response })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)