from vocab import Vocab

def test_vocab():
    vocab = Vocab()
    texts = [['the', 'rock', 'is', 'destined', '.'], 
            ['the', 'gorgeously', 'elaborate', 'continuation', '.']]
    
    vocab.build(texts)
    assert vocab['the'] == 0
    assert vocab['.'] == 1
    assert vocab['rock'] == 2
    
    vocab2 = Vocab(pad=True, unk=True)
    vocab2.build(texts)
    assert vocab2['<pad>'] == 0
    assert vocab2['<unk>'] == 1
    assert vocab2['the'] == 2
    assert vocab2['.'] == 3
    assert vocab2['rock'] == 4