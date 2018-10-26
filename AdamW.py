import math
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    """
       Implements Adam algorithm with weight decay fix in PyTorch
       Paper: Fixing Weight Decay Regularization in Adam by Ilya Loshchilov, Frank Hutter
       https://arxiv.org/abs/1711.05101
    """
    def __init__(self, params, lr, b1=0.9, b2=0.999, e=1e-8, l2=0,
                 vector_l2=False, max_grad_norm=-1, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {}".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {}".format(b2))
        if not 0.0 <= e:
            raise ValueError("Invalid epsilon value: {}".format(e))
        defaults = dict(lr=lr, b1=b1, b2=b2, e=e, l2=l2, vector_l2=vector_l2)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['b1'], group['b2']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['e'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Add weight decay at the end (fixed version)
                if (len(p.size()) > 1 or group['vector_l2']) and group['l2'] > 0:
                    p.data.add_(-group['lr'] * group['l2'], p.data)

        return loss
