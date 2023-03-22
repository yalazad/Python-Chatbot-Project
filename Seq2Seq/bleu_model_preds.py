import seq_process_msg as seq

results = []       
with open('bleu_questions.txt', 'r') as f:
    for question in f:
        result = seq.chatbot_response(question)
        print(result)
        results.append(result)

with open('bleu_preds_lstm.txt', 'w+') as f:
    for result in results:
        f.write(result+'\n')
