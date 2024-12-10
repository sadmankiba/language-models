from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from base_bert import BertPreTrainedModel
from my_transformer import TransformerEncoderLayer
from utils import *


class BertSelfAttention(nn.Module):
  """
  Num attention heads should be a factor of hidden_size
  """
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size # might be less than hidden_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this attention is applied after calculating the attention score following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by splitting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # normalize the scores
    scores = scores / math.sqrt(self.attention_head_size)

    # apply the attention mask
    if attention_mask is not None:
      scores = scores.masked_fill(attention_mask == 0, -1e9)

    # multiply the attention scores to the value and get back V' 
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = self.dropout(attention_weights)
    attn_value = torch.matmul(attention_weights, value)
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    attn_value = attn_value.transpose(1, 2).contiguous() # after transpose, the values may not be contiguous in memory
    attn_value = attn_value.view(attn_value.size(0), attn_value.size(1), self.all_head_size)
    return attn_value

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # size of k,q,v arrays are of [bs, num_attention_heads, seq_len, attention_head_size]
    ks = self.transform(hidden_states, self.key)
    vs = self.transform(hidden_states, self.value)
    qs = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(ks, qs, vs, attention_mask)
    return attn_value

class BertFixedEmbedModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for seq to seq
    self.logits = nn.Linear(config.hidden_size, config.vocab_size)

    
  def forward(self, hidden_states, attention_mask):
    for i, layer_module in enumerate(self.bert_layers):
      hidden_states = layer_module(hidden_states, attention_mask)
    
    logits = self.logits(hidden_states)
    return logits

class BertLayer(nn.Module):
  """
  Pytorch.nn.MultiHeadAttention contains the output projection weight. Here, the 
  output dense layer is separated from the Attention module, and is in BertLayer. 
  """
  def __init__(self, config):
    super(BertLayer, self).__init__()
    # self attention
    self.self_attention = BertSelfAttention(config)
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob) 
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # layer out
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
      
  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    input: the input
    output: the input that requires the Sublayer to transform
    dense_layer, dropout: the Sublayer
    ln_layer: layer norm that takes input + sublayer(output) 
    This function computes ``LayerNorm(input + Sublayer(output))``, where sublayer is a dense_layer followed by dropout.
    """
    # todo
    if dense_layer:
      # print("dense_layer", dense_layer.weight.shape)
      # print("output", output.shape)
      output = dense_layer(output)
    dropout_output = dropout(output)
    return ln_layer(input + dropout_output)

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the output of BertSelfAttention and the input of BertSelfAttention
    3. a feed forward layer
    4. a add-norm that takes the output of feed forward layer and the input of feed forward layer
    
    Args:
    - src (torch.Tensor): (bs, seq_len, d_model)
    - src_mask (torch.Tensor): (seq_len, seq_len)
    
    Returns:
    - src (torch.Tensor): (bs, seq_len, d_model)
    """
    src2 = self.self_attention(hidden_states, attention_mask)
    src = self.add_norm(hidden_states, src2, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    
    src2 = self.dropout(self.interm_af(self.interm_dense(src)))
    src = self.add_norm(src, src2, self.out_dense, self.out_dropout, self.out_layer_norm)
    
    return src

class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) 
          for _ in range(config.num_hidden_layers)])
    

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()
    
    # for seq to seq
    self.logits = nn.Linear(config.hidden_size, config.vocab_size)

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # get word embedding from self.word_embedding
    # todo
    inputs_embeds = self.word_embedding(input_ids)  # [bs, seq_len, hidden_size]

    # get position index and position embedding from self.pos_embedding
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)  # [1, seq_len, hidden_size]

    # get token type ids, since we are not consider token type, just a placeholder
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids) # [bs, seq_len, hidden_size]

    # add three embeddings together
    embeds = inputs_embeds + tk_type_embeds + pos_embeds

    # layer norm and dropout
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)

    return embeds

  def encode(self, hidden_states, attention_mask):
    """
    Args: 
      hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
      attention_mask: [batch_size, seq_len]
    
    Returns:
      hidden_states: the output from the bert layers [batch_size, seq_len, hidden_size]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
    # print("extended_attention_mask shape", extended_attention_mask.shape) # ([1, 1, 1, 4])
    extended_attention_mask = None

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    Args:
      - input_ids: [batch_size, seq_len], seq_len is the max length of the batch
      - attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    
    Returns:
      - last_hidden_state: [batch_size, seq_len, hidden_size]
      - pooler_output: [batch_size, hidden_size]
      - logits: [batch_size, seq_len, vocab_size]
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask) # [batch_size, seq_len, hidden_size]

    # get cls token hidden state
    first_tk = sequence_output[:, 0] # [batch_size, hidden_size]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    # get seq to seq logits
    logits = self.logits(sequence_output)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk, 'logits': logits}
