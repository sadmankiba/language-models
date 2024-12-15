# Beam Search Notes

```py
beam-search(model, input, k, max_token)
  for _ in 1..max_token
    output_seqs = gen_seqs(model, [input], k) 
    input = output_seqs
  return output_seqs

# top-k predicted seqs with 1 more token
gen_seqs(model, input_seqs, k)
  gend_seqs = []
  for seq in input_seqs
    output_seq = model.generate(seq, k) # returns both prob and tokens
    gend_seqs.append(output_seq)
  
  topk_seqs = topk(gend_seqs, k)
  return [t.text for t in topk_seqs]

# top k sequences with max prob
topk(seqs, k)
  probs = []
  for seq in seqs 
    prob = multiply_probs(seq)
    probs.append(prob)
  
  sorted_probs_idx = argsort(probs).reverse()
  return seqs[sorted_probs_idx]
```