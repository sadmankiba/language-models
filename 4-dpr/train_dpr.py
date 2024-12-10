import json
import random
import types
import functools
import math
import logging

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import (
    format_query,
    read_config_yaml_file,
    get_linear_scheduler,
    set_seed          
)
from transformers import BertTokenizer, BertModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    
    
# 1 Dataset Preparation
class QADataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        self.data = json.load(open(file_path))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples, tokenizer, args, stage):
        """
        collate_fn turns fetched samples into a batch
        
        This function converts a few samples into queries and passages. 
        It tokenizes both queries and passages. A passage contains a title and a document. 
        
        Dataloader creates multiple processes to fetch data and calls this function. 
        Since these processes are created with fork, cuda cannot be used in this function.
        """
        # prepare query input
        queries = [format_query(x['question']) for x in samples]
        query_inputs = tokenizer(queries, max_length=256, padding=True, truncation=True, return_tensors='pt')
        
        # prepare document input
        ## select the first positive document
        ## passage = title + document
        positive_passages = [x['positive_ctxs'][0] for x in samples]
        positive_titles = [x['title'] for x in positive_passages]
        positive_docs = [x['text'] for x in positive_passages]
        
        default_negative_ctx = {'title':'','text':''}
        if stage == 'train':
            ## random choose one negative document
            negative_passages = [random.choice(x['hard_negative_ctxs']) 
                                 if len(x['hard_negative_ctxs']) != 0  
                                 else (
                                     random.choice(x['negative_ctxs'])
                                        if len(x['negative_ctxs']) != 0 
                                        else default_negative_ctx
                                 ) 
                                 for x in samples ]
        elif stage == 'dev':
            negative_passages = [x['hard_negative_ctxs'][:min(args.num_hard_negative_ctx,len(x['hard_negative_ctxs']))]
                                + x['negative_ctxs'][:min(args.num_other_negative_ctx,len(x['negative_ctxs']))] 
                                for x in samples]
            negative_passages = [x for y in negative_passages for x in y]

        negative_titles = [x["title"] for x in negative_passages]
        negative_docs = [x["text"] for x in negative_passages]
        titles = positive_titles + negative_titles
        docs = positive_docs + negative_docs
        
        doc_inputs = tokenizer(titles, docs, max_length=256, padding=True, truncation=True, return_tensors='pt')

        return {
            'query_input_ids':query_inputs.input_ids,
            'query_attention_mask':query_inputs.attention_mask,
            'query_token_type_ids':query_inputs.token_type_ids,

            "doc_input_ids":doc_inputs.input_ids,
            "doc_attention_mask":doc_inputs.attention_mask,
            "doc_token_type_ids":doc_inputs.token_type_ids,
        }
        
# An example call
# len(samples) = 16
# len(queries) = 16
# num words in queries = [9, 5, 3, ..., ]
# query_inputs = {
#    'input_ids': tensor([[ 101,  2054,  2003,  ...,  102,    0,    0], [...], ...), len 24 each
#    'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0], [...], ...),
#   'attention_mask': tensor([[1, 1, 1,  ..., 1, 0, 0], [...], ...),
# }
# len(positive_titles) len = 16
# positive_titles = ['The_Beatles', 'Hamlet', ...]
# len(negative_titles) len = 16
# len(docs) = 32
# docs = ['The Beatles were ...', 'Hamlet is a ...', ...]
# doc_inputs = {
#    'input_ids': tensor([[ 101,  1996,  100,  ...,  102,    0,    0], [...], ...),
#    'token_type_ids': tensor([[0, 0, 0,  ..., 1, 0, 0], [...], ...),
#   'attention_mask': tensor([[1, 1, 1,  ..., 1, 0, 0], [...], ...),
# }


# 2 Query and Doc Encoder

class DualEncoder(nn.Module):
    """Wraps the query and document encoders (BERT models) into a single model."""
    def __init__(self, query_encoder, doc_encoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def forward(
        self,
        query_input_ids, # [bs,seq_len]
        query_attention_mask, # [bs,seq_len]
        query_token_type_ids, # [bs,seq_len],
        doc_input_ids, # [bs*n_doc,seq_len]
        doc_attention_mask, # [bs*n_doc,seq_len]
        doc_token_type_ids, # [bs*n_doc,seq_len]
    ):  
        CLS_POS = 0
        ## [bs,n_dim]
        query_embedding = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask = query_attention_mask,
            token_type_ids = query_token_type_ids,
            ).last_hidden_state[:,CLS_POS,:]
        
        ## [bs * n_doc,n_dim]
        doc_embedding = self.doc_encoder(
            input_ids = doc_input_ids,
            attention_mask = doc_attention_mask,
            token_type_ids = doc_token_type_ids,
            ).last_hidden_state[:,CLS_POS,:]
        
        return query_embedding, doc_embedding 
    
# An Example Call
# query_embedding.shape = (16, 768)
# query_embedding = tensor([[-0.1, 0.2, 0.3, ...], [0.2, -0.3, 0.04, ...], ...])
# doc_embedding.shape = (32, 768)
# doc_embedding = tensor([[0.1, -0.2, 0.3, ...], [-0.2, 0.3, -0.04, ...], ...])

def calculate_dpr_loss(matching_score, labels):
    """
    Calculate loss for predicted scores and true labels.
    
    Args:
        matching_score (tensor): The predicted scores. Size (batch_size, batch_size).
        labels (tensor): The true labels. Size: (batch_size,).
    
    Returns:
        float: The loss value.
    """
    return F.nll_loss(input=F.log_softmax(matching_score, dim=1),target=labels)

def main():
    yaml_config = read_config_yaml_file('config/train_dpr_curatedtrec.yaml')
    args = types.SimpleNamespace(**yaml_config)

    # Prepare encoder
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    query_encoder = BertModel.from_pretrained(args.base_model,add_pooling_layer=False)
    doc_encoder = BertModel.from_pretrained(args.base_model,add_pooling_layer=False)
    dual_encoder = DualEncoder(query_encoder, doc_encoder)
    
    dual_encoder.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual_encoder.to(device)
   
    train_dataset = QADataset(args.train_file)
    train_collate_fn = functools.partial(QADataset.collate_fn, tokenizer=tokenizer, stage='train', args=args,)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, 
                                                   shuffle=True, collate_fn=train_collate_fn, num_workers=4, pin_memory=True)
    
    dev_dataset = QADataset(args.dev_file)
    dev_collate_fn = functools.partial(QADataset.collate_fn,tokenizer=tokenizer,stage='dev',args=args,)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, 
                                                 shuffle=False, collate_fn=dev_collate_fn, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(dual_encoder.parameters(), lr=args.lr, eps=args.adam_eps)
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval, int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = get_linear_scheduler(optimizer, warmup_steps=args.warmup_steps, total_training_steps=MAX_TRAIN_STEPS)
    
    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")  # num_batches_per_epoch / gradient_accumulation_steps * max_train_epochs
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), ncols=100)
    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            query_embedding, doc_embedding  = dual_encoder(**batch)
            single_device_query_num,_ = query_embedding.shape  # [batch_size, n_dim]
            single_device_doc_num,_ = doc_embedding.shape      # [batch_size * 2, n_dim]
            
            # Calculate dot product of query and doc embeddings for loss
            # The optimization will try to make the embedding for query and doc similar 
            # so that we can use nearest neighbor search later to find the most similar 
            # doc for a given query
            matching_score = torch.matmul(query_embedding, doc_embedding.permute(1, 0)) # [batch_size, batch_size]
            labels = torch.cat([torch.arange(single_device_query_num)],dim=0).to(matching_score.device)
            loss = calculate_dpr_loss(matching_score, labels=labels)

            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
            completed_steps += 1
            
            lr_scheduler.step()
            
            logger.info(f"Epoch {epoch}, Step {step}, Training loss {loss:.4f}, Learning rate: {lr_scheduler.get_last_lr()[0]}")    
            
            if completed_steps % EVAL_STEPS == 0:
                dual_encoder.train()
                query_encoder.save_pretrained(f"checkpoints/query_encoder/step-{completed_steps}/")
                doc_encoder.save_pretrained(f"checkpoints/doc_encoder/step-{completed_steps}/")
                tokenizer.save_pretrained(f"checkpoints/tokenizer/step-{completed_steps}/")
            
            # Update loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
if __name__ == "__main__": 
    main()