from dataclasses import dataclass
import torch
from model import DanModel

@dataclass
class Args:
    emb_size: int
    init_range: float
    emb_file: str = None

def test_dan_model():
    args = Args(emb_size=4, init_range=0.08)
    
    vocab = list(range(10))
    tag_size = 3
    model = DanModel(args, vocab, tag_size)
    x = torch.tensor([[1, 2, 6, 0], [3, 4, 5, 8]])
    scores = model(x)
    print(scores)
    assert scores.size() == (2, 3)