from typing import Callable, Iterable, Tuple

import torch
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                # import pdb; pdb.set_trace()

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                vt_previous = state.get("vt", 0)
                mt_previous = state.get("mt", 0)
                step = state.get("step", 1)
                beta_1, beta_2 = self.defaults["betas"]

                mt = beta_1 * mt_previous + (1 - beta_1) * grad
                vt = beta_2 * vt_previous + (1 - beta_2) * torch.pow(grad, 2)
                state["mt"], state["vt"] = mt, vt

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if self.defaults["correct_bias"]:
                    mt_hat = mt / (1 - beta_1 ** step)
                    vt_hat = vt / (1 - beta_2 ** step)

                # Update parameters
                new_param = p.data - alpha * mt_hat / (vt_hat ** 0.5 + self.defaults["eps"])

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                new_param = new_param - alpha * self.defaults["weight_decay"] * p.data
                p.data = new_param

                if "step" in state: state["step"] += 1
                else: state["step"] = 2

        return loss