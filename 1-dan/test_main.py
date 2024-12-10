from main import read_dataset, data_iter, convert_text_to_ids, pad_sentences
from vocab import Vocab

def test_read_dataset():
    dataset = read_dataset('data/sst-train.txt')
    assert dataset[0][0][0] == 'the'
    assert dataset[0][0][1] == 'rock' 
    assert dataset[0][1] == '3'
    assert dataset[1][0][0] == 'the'
    assert dataset[1][0][1] == 'gorgeously'
    assert dataset[1][1] == '4'
    
def test_data_iter():
    train_text = [
        (['the', 'rock'], '3'),
        (['a', 'person'], '4'),
        ([ 'the', 'stupid'], '1'),
        (['all', 'right'], '2'),
        (['a', 'movie'], '4'),
        (['the', 'gorgeously'], '4')
    ]
    word_vocab = Vocab(pad=True, unk=True)
    word_vocab.build(list(zip(*train_text))[0])
    tag_vocab = Vocab() 
    tag_vocab.build(list(zip(*train_text))[1])
    train_data = convert_text_to_ids(train_text, word_vocab, tag_vocab)
    diter = data_iter(train_data, batch_size=3, shuffle=False)
    
    batch0 = next(diter)
    batch0_sents = batch0[0]
    batch0_tags = batch0[1]
    assert len(batch0_sents) == 3
    assert len(batch0_tags) == 3
    
    batch1 = next(diter)
    batch1_sents = batch1[0]
    batch1_tags = batch1[1]
    assert len(batch1_sents) == 3
    assert len(batch1_tags) == 3
    
    batch2 = next(diter, None)
    assert batch2 is None
    
def test_pad_sentences():
    sents = [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2]
    ]
    sents = pad_sentences(sents, 0)
    assert sents[0] == [1, 2, 3, 0]
    assert sents[1] == [1, 2, 3, 4]
    assert sents[2] == [1, 2, 0, 0]