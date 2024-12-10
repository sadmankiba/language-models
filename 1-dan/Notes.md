# Notes

## Results
**With training embedding**

- Adding up embeddings and one output layer. 
Epochs: 5.
Test accuracy: 0.4

- Adding up embeddings, 3 hidden layers, no dropout. 
Epochs: 5
Test accuracy: 0.380

- Averaging embeddings, 3 hidden layers, dropout at word, embed, and hidden
Epochs: 5
Word dropout: 0.3
Test accuracy: 0.308

Epochs: 15
Word dropout: 0.3
Test accuracy: 0.353

Epochs: 15
Word dropout: 0.1
Test accuracy: 0.373

Epochs: 15
Word dropout: 0.1
Emb dropout: 0.1
Hidden dropout: 0.1
Test accuracy: 0.392

**With Glove Embedding**

Epochs: 15
Word dropout: 0.1
Emb dropout: 0.1
Hidden dropout: 0.1
Glove embedding
Test accuracy: 0.391


## Testing

```
python3 main.py --batch_size=4 --emb_size=8 --hid_size=3 --hid_layer=1
```