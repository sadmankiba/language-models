import torch
import numpy as np
from optimizer import AdamW

seed = 0


def test_optimizer(opt_class=AdamW) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        correct_bias=True,
    )
    for i in range(1000): # 1000
        opt.zero_grad()
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    
    actual = model.weight.detach()

    ref = torch.tensor(np.load("optimizer_test.npy"))
    print("ref", ref)
    print("actual", actual)
    assert torch.allclose(ref, actual)
    print("Optimizer test passed!")

# Output
# ref tensor([[ 0.5548,  0.8667,  0.0729],
#         [-0.4472, -0.2951, -0.2717]])
# SGD actual tensor([[ 0.6253,  0.8874,  0.0657],
#         [-0.4793, -0.3015, -0.1358]])
# SGD with Momentum actual tensor([[ 0.6259,  0.8892,  0.0672],
#         [-0.4808, -0.3028, -0.1358]])
# Adam actual tensor([[ 0.5549,  0.8667,  0.0729],
#         [-0.4473, -0.2951, -0.2717]])
# AdamW actual tensor([[ 0.5548,  0.8667,  0.0729],
#         [-0.4472, -0.2951, -0.2717]])
