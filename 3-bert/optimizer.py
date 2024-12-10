from typing import Callable, Iterable, Tuple

import torch
import numpy as np
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                # Update first and second moments of the gradients
                v_p = beta1 * state.get("v", 0) + (1 - beta1) * grad
                s_p = beta2 * state.get("s", 0) + (1 - beta2) * grad ** 2
                self.state[p]["v"] = v_p
                self.state[p]["s"] = s_p
                
                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                t= self.state[p].get("t", 0) + 1
                v_p_corr = v_p / (1 - beta1 ** t)
                s_p_corr = s_p / (1 - beta2 ** t)
                self.state[p]["t"] = t
                
                # type(p): <class 'torch.nn.parameter.Parameter'>
                # type(alpha): <class 'float'>
                # type(v_p_corr): <class 'torch.Tensor'>
                # type(s_p_corr): <class 'torch.Tensor'>
                # type(eps): <class 'float'>
                
                # Update parameters
                with torch.no_grad():
                    p -= alpha / torch.sqrt(s_p_corr + eps) * v_p_corr

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                with torch.no_grad():
                    p -= alpha * weight_decay * p
                group["params"][i] = p

        return loss
    
    
# Example
# group {'params': [Parameter containing:
# tensor([[-0.0043,  0.3097, -0.4752],
#         [-0.4249, -0.2224,  0.1548]], requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-06, 'weight_decay': 0.0001, 'correct_bias': True}
# p Parameter containing:
# tensor([[-0.0043,  0.3097, -0.4752],
#         [-0.4249, -0.2224,  0.1548]], requires_grad=True)
# grad tensor([[-1.0770, -0.4562, -0.0693],
#         [-0.3609, -0.1529, -0.0232]])
# state {}
