import csv
import random
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from doc2embedding import get_curatedtrec_wikis
from utils.utils import format_query

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_model_path = "checkpoints/tokenizer/step-142/"
tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, add_pooling_layer=False)
test_file = "downloads/data/retriever/qas/curatedtrec-test.csv"
encoding_batch_size = 8

## load QA dataset
queries = []
answers = []
with open(test_file) as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        queries.append(format_query(row[0]))
        answers.append(row[1])

# len(queries) = 694
# len(answers) = 694

query_batches = [queries[idx:idx+encoding_batch_size] for idx in range(0, len(queries), encoding_batch_size)]

# Make FAISS Index
embedding_dimension = 768
index = faiss.IndexFlatIP(embedding_dimension)
data = np.load("downloads/data/wikipedia_split/curatedtrec_embeddings.npy")
index.add(data)

# Load Wikipedia passages
wikis = get_curatedtrec_wikis()

# Embed queries
progress_bar = tqdm(total=len(query_batches), ncols=100, desc='Making embeddings...')
query_encoder = BertModel.from_pretrained("checkpoints/query_encoder/step-142/", add_pooling_layer=False).to(device)
query_embeddings = []
for query in query_batches:
    tokens = tokenizer(query, max_length=256,
                    truncation=True,padding='max_length',return_tensors='pt').to(device)
    query_embedding = query_encoder(**tokens)
    query_embeddings.append(query_embedding.last_hidden_state[:,0,:].detach().cpu().numpy())
    progress_bar.update(1)

query_embeddings = np.concatenate(query_embeddings, axis=0)

# Retrieve top-k documents
top_k = 5
_, I = index.search(query_embeddings, top_k)

# len(I) = 694
# I 
# [[2118  758 1290 1455 1422]
# [1757  197 1250 1422  190]
# ...
# [1760  251 1759  914 2339]]
# 

ids_to_check = random.sample(range(1, len(queries) + 1), 5)

for id in ids_to_check:
    print("Query: ", queries[id])
    print("Answer:", answers[id])
    print("Retrieved Passages:")
    for i, idx in enumerate(I[id]):
        print(i, wikis[idx])