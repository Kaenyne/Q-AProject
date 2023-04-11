from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import csv
import logging
from sentence_transformers.cross_encoder import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') # scores come out in a range between -10 to 10
prediction = model.predict([('who is taylor swift', 'drake is a rapper')]) # model.predict([('QUESTION','TEXT')])
#from cosinesimilarity import cosine
import nltk
import openai
from sentence_transformers import SentenceTransformer
model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
from sklearn.metrics.pairwise import cosine_similarity
#similarity = cosine.cosineSimilarity(text1,text2) #outputs a number between 0 & 1
#nltk.download('punkt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
def emb(text, max_length=512):
    # Add the special tokens.
    logging.getLogger("transformers").setLevel(logging.ERROR)
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens and truncate to max_length.
    tokenized_text = tokenizer.tokenize(marked_text)[:max_length-2]

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create segment masks and padding.
    segments_ids = [1] * len(tokenized_text)
    padding_length = max_length - len(tokenized_text)
    indexed_tokens += [0] * padding_length
    segments_ids += [0] * padding_length

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True, )  # Whether the model returns all hidden-states.

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)

    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    token_vecs_cat = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
        token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.numpy()
    return sentence_embedding
texts = []
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\AEHR.txt', 'r') as file:
    text1 = file.read()
    texts.append(text1)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\BYD.txt', 'r') as file:
    text2 = file.read()
    texts.append(text2)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\ALGM.txt', 'r') as file:
    text3 = file.read()
    texts.append(text3)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\SGML.txt', 'r') as file:
    text4 = file.read()
    texts.append(text4)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\TSLA.txt', 'r') as file:
    text5 = file.read()
    texts.append(text5)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\TSLA_BYD_E.txt', 'r') as file:
    text6 = file.read()
    texts.append(text6)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\TSLA_BYD_St.txt', 'r') as file:
    text7 = file.read()
    texts.append(text7)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\TSLA_BYD_Sto.txt', 'r') as file:
    text8 = file.read()
    texts.append(text8)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\EMR.txt', 'r') as file:
    text9 = file.read()
    texts.append(text9)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\AMC.txt', 'r') as file:
    text10 = file.read()
    texts.append(text10)
with open(r'C:\Users\krish\OneDrive\Desktop\TEXT files for ML\NIO.txt', 'r') as file:
    text11 = file.read()
    texts.append(text11)
similarity = []
text_emb = []
for i in range(len(texts)):
    embedding = model1.encode(texts[i])
    text_emb.append(embedding)
for i in range(len(texts)):
    score = cosine_similarity(text_emb[4].reshape(1, -1), text_emb[i].reshape(1, -1))
    similarity.append(score[0][0])

#question = "does Tesla make a model 3?"
question = input('what question do you want to ask?')
scores = []
texts_g = []
nums = []
for i in range(len(texts)):
    score = model.predict([(texts[i],question)])
    scores.append(score)
    if score>0:
        nums.append(i)
        texts_g.append(texts[i])
scores,texts = zip(*sorted(zip(scores,texts)))

high = len(texts)
top_3_articles = [texts[high-1],texts[high-2],texts[high-3]]

sim_to_TSLA = pd.DataFrame(columns = ['Articles'])
sim_to_TSLA['Articles'] = texts
sim_to_TSLA['similarities'] = similarity
sim_to_TSLA['cross_encoding'] = scores

len1 = len(similarity)
print(sim_to_TSLA.head(len1))
print(nums)
print(len(texts_g))

for i in range(len(top_3_articles)):
    print(top_3_articles[i] + "\n____________END OF TEXT__________\n")


