import os
import json
import numpy as np 
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


pretrained_model_path = "checkpoints/doc_encoder/step-142/"
pretrained_tokenizer_path = "checkpoints/tokenizer/step-142/"
output_dir = "downloads/data/wikipedia_split"

doc_encoder = BertModel.from_pretrained(pretrained_model_path, add_pooling_layer=False)
tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer_path, add_pooling_layer=False)
encoding_batch_size = 8

doc_encoder.eval()

def get_curatedtrec_wikis():
    """
    Combine wiki texts from train and dev
    
    Although, these do not directly contain test question answers, 
    the passages can be used to find similarity with the test questions.
    """
    wikis = []

    train_data = json.load(open("downloads/data/retriever/curatedtrec-train.json"))
    dev_data = json.load(open("downloads/data/retriever/curatedtrec-dev.json"))
    for item in train_data + dev_data:
        # Using only the first positive and hard negative context
        # takes about 6 minutes to encode all the passages
        # with a batch size of 8 in CPU. 
        if item['positive_ctxs']:
            p = item['positive_ctxs'][0]
            wikis.append([p['title'], p['text']])
        
        if item['hard_negative_ctxs']:
            p = item['hard_negative_ctxs'][0]
            wikis.append([p['title'], p['text']])

    return wikis  

if __name__ == '__main__':
    # Prepare wiki passages list
    wikis = get_curatedtrec_wikis()   

    # Encode wiki passages
    wiki_batches = [wikis[idx:idx+encoding_batch_size] for idx in range(0, len(wikis), encoding_batch_size)]
    progress_bar = tqdm(total=len(wiki_batches), ncols=100, desc='Making embeddings...')
    doc_embeddings = []
    for data in wiki_batches:
        title = [d[0] for d in data]
        passage = [d[1] for d in data]
        model_input = tokenizer(title, passage, max_length=256, 
                            return_tensors="pt", padding='max_length', truncation=True)
        CLS_POS = 0
        output = doc_encoder(**model_input).last_hidden_state[:,CLS_POS,:].detach().cpu().numpy()
        doc_embeddings.append(output)
        progress_bar.update(1)
        
    doc_embeddings = np.concatenate(doc_embeddings, axis=0)
    print(doc_embeddings.shape)   # (2479, 768)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "curatedtrec_embeddings.npy"), doc_embeddings)