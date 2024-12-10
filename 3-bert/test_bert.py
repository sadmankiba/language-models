import torch
import torch.nn.functional as F
from bert import BertSelfAttention, BertLayer, BertModel, BertFixedEmbedModel
from config import BertConfig
from test_util import get_sinusoidal_positional_embeddings

def test_self_attention_shape():
    hidden_size = 8
    num_attention_heads = 2
    seq_len = 4
    batch_size = 1
    config = BertConfig(hidden_size=hidden_size, 
                        num_attention_heads=num_attention_heads)
    
    bert_attn = BertSelfAttention(config)
    x = torch.randn(batch_size, seq_len, hidden_size)
    x_next = bert_attn(x, None)
    assert x_next.shape == (batch_size, seq_len, hidden_size)


# Generate simple copy dataset
def generate_copy_data(batch_size, seq_length, embed_dim):
    min_val = 0
    max_val = 10
    gen_t = torch.randint(min_val, max_val, (batch_size, seq_length, embed_dim))
    gen_t = gen_t.float()
    return gen_t

def generate_rotate_embeddings(batch_size, seq_len, embed_dim, vocab_size):
    """
    Rotate sequence by 1 leftwise to create target
    Use random embeddings for input
    
    Returns:
        input_embeds (torch.Tensor): (bs, seq_len, embed_dim)
        target (torch.Tensor): (bs, seq_len)
    """
    min_val = 0
    max_val = vocab_size
    max_seq_len = 20
    tok_embed_table = torch.randn(vocab_size, embed_dim) # (vocab_size, embed_dim)
    pos_embed_table = get_sinusoidal_positional_embeddings(max_seq_len, embed_dim)
    
    input_ids = torch.randint(min_val, max_val, (batch_size, seq_len)).long() # (bs, seq_len)
    tok_embeds = tok_embed_table[input_ids] # (bs, seq_len, embed_dim)
    pos = torch.arange(seq_len)
    pos_embeds = pos_embed_table[pos].unsqueeze(0)
    input_embeds = tok_embeds + pos_embeds
    # rotated sequence
    target = torch.cat((input_ids[:, -1].unsqueeze(1), input_ids[:, :-1]), dim=1)
    return input_embeds, target

def generate_rotate_data(batch_size, seq_len, vocab_size):
    """Rotate sequence by 1 leftwise"""
    min_val = 0
    max_val = vocab_size
    input = torch.randint(min_val, max_val, (batch_size, seq_len)).long()
    output = torch.cat((input[:, -1].unsqueeze(1), input[:, :-1]), dim=1)
    return input, output
    

def test_self_attention_training():
    hidden_size = 1
    num_attention_heads = 1
    seq_len = 4
    batch_size = 2
    config = BertConfig(hidden_size=hidden_size, 
                        num_attention_heads=num_attention_heads)
    bert_attn_model = BertSelfAttention(config)
    for name, param in bert_attn_model.named_parameters():
        print(f"Parameter name: {name}, value: {param}")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(bert_attn_model.parameters(), lr=0.01)
    
    # Training loop
    for step in range(5000):
        data = generate_copy_data(batch_size=batch_size, seq_length=seq_len, embed_dim=hidden_size) # (bs, seq_len, embed_dim)
        output = bert_attn_model(data, None) # (bs, seq_len, embed_dim)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            # print(f"Step {step}, Loss: {loss.item():.4f}")
            # print("data", data, "output", output)
            # print("query.weight", bert_attn_model.query.weight,
            #       "query.bias", bert_attn_model.query.bias,
            #       "key.weight", bert_attn_model.key.weight,
            #       "key.bias", bert_attn_model.key.bias,
            #       "value.weight", bert_attn_model.value.weight,
            #       "value.bias", bert_attn_model.value.bias)
            pass

    data = generate_copy_data(batch_size=batch_size, seq_length=20, embed_dim=hidden_size) 
    output = bert_attn_model(data, None)
    loss = criterion(output, data)
    assert loss.item() < 2
    
    print("Trained parameters")
    for name, param in bert_attn_model.named_parameters():
        print(f"Parameter name: {name}, value: {param}")


def test_bert_layer_shape(): 
    hidden_size = 8
    num_attention_heads = 2
    seq_len = 4
    batch_size = 1
    config = BertConfig(hidden_size=hidden_size, 
                        num_attention_heads=num_attention_heads)
    
    bert_layer = BertLayer(config)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    out = bert_layer(hidden_states, None)
    assert out.shape == (batch_size, seq_len, hidden_size)

def test_bert_layer_training():
    hidden_size = 1
    num_attention_heads = 1
    seq_len = 4
    batch_size = 2
    config = BertConfig(hidden_size=hidden_size,
                        intermediate_size=hidden_size*4, 
                        num_attention_heads=num_attention_heads,
                        hidden_dropout_prob=0.0,
                        attention_probs_dropout_prob=0.0)
    # Comment out dropout and layernorm for testing
    bert_layer_model = BertLayer(config)
    for name, param in bert_layer_model.named_parameters():
        print(f"Parameter name: {name}, value: {param}")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(bert_layer_model.parameters(), lr=0.01)
    
    # Training loop
    for step in range(5000):
        data = generate_copy_data(batch_size=batch_size, seq_length=seq_len, embed_dim=hidden_size) # (bs, seq_len, embed_dim)
        output = bert_layer_model(data, None) # (bs, seq_len, embed_dim)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            print("data", data, "output", output)
            pass
    
    data = generate_copy_data(batch_size=batch_size, seq_length=20, embed_dim=hidden_size) 
    output = bert_layer_model(data, None)
    loss = criterion(output, data)
    assert loss.item() < 2
    
    print("Trained parameters")
    for name, param in bert_layer_model.named_parameters():
        print(f"Parameter name: {name}, value: {param}")  

def test_bert_fixed_model_training():
    """
    Test a single Bert Layer on rotating data.
    """
    hidden_size = 8
    num_attention_heads = 1
    seq_len = 4
    batch_size = 4
    vocab_size = 10
    config = BertConfig(hidden_size=hidden_size,
                        intermediate_size=64, 
                        vocab_size=vocab_size,
                        num_attention_heads=num_attention_heads,
                        num_hidden_layers=1,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1)
    model = BertFixedEmbedModel(config)
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, value: {param}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for step in range(5001):
        input_embs, exp_output = generate_rotate_embeddings(batch_size=batch_size, seq_len=seq_len, \
                embed_dim=hidden_size, vocab_size=vocab_size) 
        attention_mask = torch.ones(1, seq_len)
        logits = model(input_embs, attention_mask) # (bs, seq_len, vocab_size)
        loss = F.nll_loss(F.log_softmax(logits, dim=2).reshape(-1, vocab_size), exp_output.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            # print("logits", logits)
            # print("logits argmax", torch.argmax(logits, dim=2), "exp_output", exp_output)
            pass
    
    # print("Trained parameters")
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, value: {param}")

def test_bert_model_shape():
    hidden_size = 8
    num_attention_heads = 2
    seq_len = 4
    config = BertConfig(hidden_size=hidden_size, 
                        num_attention_heads=num_attention_heads)
    
    bert_model = BertModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    attention_mask = torch.ones(1, seq_len)
    out = bert_model(input_ids, attention_mask)
    assert out["last_hidden_state"].shape == (1, seq_len, hidden_size)

def evaluate_rotate(model, n_steps=100):
    total_out = 0
    corr_out = 0
    
    batch_size = 1
    seq_len = 4
    for i in range(n_steps):
        input_data, exp_output = generate_rotate_data(batch_size=batch_size, seq_len=seq_len, vocab_size=10)
        attention_mask = torch.ones(1, seq_len)
        logits = model(input_data, attention_mask)["logits"]
        out = torch.argmax(logits, dim=-1)
        total_out += seq_len
        corr_out += sum(out.reshape(-1) == exp_output.reshape(-1)).item()
        
        if i % 10 == 0:
            print("Target:", exp_output.reshape(-1), ", Predicted:", out.reshape(-1))
    
    acc = corr_out / total_out
    print("Accuracy:", acc) 
    return acc

def test_bert_model_training():
    """Test Bert model with rotating sequence"""

    hidden_size = 8
    num_attention_heads = 2
    seq_len = 4
    batch_size = 4
    vocab_size = 10
    config = BertConfig(hidden_size=hidden_size,
                        intermediate_size=hidden_size*4, 
                        vocab_size=vocab_size,
                        num_attention_heads=num_attention_heads,
                        num_hidden_layers=1,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1)
    bert_model = BertModel(config)
    
    optimizer = torch.optim.Adam(bert_model.parameters(), lr=0.01)
    
    # Training loop
    for step in range(5001):
        input_data, exp_output = generate_rotate_data(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size) 
        attention_mask = torch.ones(1, seq_len)
        logits = bert_model(input_data, attention_mask)["logits"] # (bs, seq_len, vocab_size)
        loss = F.nll_loss(F.log_softmax(logits, dim=2).reshape(-1, vocab_size), exp_output.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            pass
        if step % 1000 == 0:
            evaluate_rotate(bert_model)
    
    assert evaluate_rotate(bert_model, 400) > 0.88