# Getting BLEU score of a fixed no. of samples
import nltk.translate.bleu_score as bleu
import pandas as pd
import text_preprocessor as pre

bleu_score_list = []

original_answers = []
predicted_answers = []

clean_answers=[pre.remove_tags(pre.clean_text(line)) for line in open('bleu_answers.txt')]

original_answers=[line.strip() for line in clean_answers]
predicted_answers=[line.strip() for line in open('bleu_predicted_gru.txt')] 

df = pd.DataFrame(list(zip(original_answers, predicted_answers)))

bleu_score_list = []

for orig, pred in zip(df[0], df[1]):
    list_orig = orig.split(' ')
    list_pred = pred.split(' ')

    references = [list_orig] # list of references for 1 sentence.
    list_of_references = [references] # list of references for all sentences in corpus.
    list_of_hypotheses = [list_pred] # list of hypotheses that corresponds to list of references.
    bleu_score = bleu.corpus_bleu(list_of_references, list_of_hypotheses)
    print("Bleu score:", bleu_score)
    bleu_score_list.append(bleu_score)

print("Bleu score list:",bleu_score_list)

avg_blue_score = (sum(bleu_score_list) / len(bleu_score_list))
print("\nAverage BLEU score:", avg_blue_score)