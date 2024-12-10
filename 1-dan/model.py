import random
import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    if emb_size not in [50, 100, 200, 300]:
        raise ValueError(f'Embedding size {emb_size} is not supported.')
    
    wordvecs = {}
    print(f"Loading embeddings from {emb_file}")
    with open(emb_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            wordvecs[word] = vector
    
    print("Setting up embeddings for the model")
    emb = np.zeros((len(vocab), emb_size))
    for i in range(len(vocab)):
        word = vocab.id2word[i]
        if word in wordvecs:
            emb[i] = wordvecs[word]
        else:
            emb[i] = np.random.uniform(-0.08, 0.08, emb_size)
            
    return emb

class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)
        self.embedding_dropout = nn.Dropout(self.args.emb_drop)
        
        self.hidden = nn.ModuleList([
            nn.Linear(self.args.emb_size if i == 0 else self.args.hid_size, self.args.hid_size) 
            for i in range(self.args.hid_layer)
        ]) 
        self.hidden_dropout = nn.Dropout(self.args.hid_drop)       
        self.linear = nn.Linear(self.args.emb_size, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        init_range = self.args.init_range
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.uniform_(self.linear.weight, -init_range, init_range)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding.weight.data.copy_(
            torch.from_numpy(load_embedding(
                self.vocab, self.args.emb_file, self.args.emb_size)))
        


    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        # word dropout (replace with <unk>)
        x_orig_shape = x.size()
        x = x.view(-1)
        unk_indices = []
        for i in range(len(x)):
            if random.random() < self.args.word_drop:
                unk_indices.append(i)
        x[unk_indices] = self.vocab['<unk>']
        x = x.view(x_orig_shape)
        
        emb = self.embedding(x)  # [batch_size, seq_length, emb_size]
        emb = self.embedding_dropout(emb)
        emb_composed = torch.mean(emb, dim=1)  # [batch_size, emb_size]
        z = emb_composed
        z.to(emb_composed.device)
        for i in range(self.args.hid_layer):
            z = torch.tanh(self.hidden[i](z))
            z = self.hidden_dropout(z)
        scores = self.linear(emb_composed) # [batch_size, ntags]
        return scores
        
# Example values
# x tensor([[    3,    20,    10,  4848,    90,   296]])
# emb tensor([[[-0.1694,  0.0877,  0.0793,  ..., -0.1049, -0.0453, -0.0363], ]]])
# emb_sum tensor([[ 0.2293,  0.2256, -0.1950,  ..., 0.8220,  0.0548, -0.7007 ]])
# z tensor([[-0.1694,  0.0877,  0.0793],  ..., -0.1049, -0.0453, -0.0363],
# scores tensor([[0.1826, 0.3082, 0.2523, 0.2267, 0.1114] ... ]])