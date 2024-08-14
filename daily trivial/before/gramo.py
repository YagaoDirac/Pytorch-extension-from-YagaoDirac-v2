not tested yet

import torch

class GradientModificationFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x:torch.Tensor, learning_rate:float, epi=torch.tensor([1e-12]), \
                 suppression_factor = torch.tensor([1e3]))->torch.Tensor:
        ctx.save_for_backward(learning_rate, epi, suppression_factor)
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        learning_rate, epi, suppression_factor = ctx.saved_tensors

        mean = g.mean(dim=0, keepdim=True)
        _centralized = g - mean
        std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is slightly recommended by me.
        std_too_small = std < epi
        std = (std - std * std_too_small) + std_too_small * (epi* suppression_factor)
        _normalized = _centralized / std
        if learning_rate != 1.:
            return _normalized * learning_rate, None, None, None
        else:
            return _normalized, None, None, None
        pass
    pass  # class



class GradientModification(torch.nn.Module):
    def __init__(self, learning_rate:float, epi=torch.tensor([1e-12]), \
                 suppression_factor = torch.tensor([1e3]), \
                  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate =learning_rate
        self.epi=epi
        self.suppression_factor = suppression_factor
        pass
    def forward(self, x:torch.tensor)->torch.tensor:
        #forward(ctx, x:torch.Tensor, learning_rate:float, epi=torch.tensor([1e-12]), \
        #suppression_factor = torch.tensor([1e3]))->torch.Tensor:
        return GradientModificationFunction.apply(x, self.learning_rate, self.epi, \
                                                   self.suppression_factor)
    def set_learning_rate(self, learning_rate:float):
        self.learning_rate = learning_rate
        pass



m = GradientModification(learning_rate=0.01)
x = torch.ones((1,))
y:torch.Tensor = m(x)
y.backward((torch.ones_like(x)))

print(x)




