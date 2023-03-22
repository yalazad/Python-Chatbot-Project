import seq2seq_gru as seq

def chatbot_response(sentence):
    try:
        result, sentence = seq.evaluate(sentence)
        result = result.capitalize()
    except KeyError:
        return "Sorry I don't know the answer to that."

    print('Question: %s' % (sentence))
    print('Predicted answer: {}'.format(result))  
    return result