from flask import Flask, render_template, request
import seq_process_msg


app = Flask(__name__)

app.config['SECRET_KEY'] = 'a-very-secretive-key-123456'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():    
    print("In chatbotResponse")
    if request.method == 'GET':
        the_question = request.args.get('msg')
        print("Question is:", the_question)

        return str(seq_process_msg.chatbot_response(the_question))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)