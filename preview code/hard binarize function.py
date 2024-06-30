from typing import Any, List, Tuple
import torch
import math


class Binarize_01_Forward_only_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x:torch.Tensor = args[0]
        
        dtype = x.dtype
        x = x.gt(0.5)
        x = x.to(dtype)
        return x

    @staticmethod
    def backward(ctx, g):
        return g

    pass  # class

# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = Binarize_01_Forward_only_Function.apply(input)
# print(output, "should be 0., 0., 0., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")

# fds=432



class Binarize_01_Forward_only(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It clamp the forward inter layer data, while doesn't touch the backward propagation.
    """
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not x.requires_grad):
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return Binarize_01_Forward_only_Function.apply(x)

# layer = Binarize_01_Forward_only()
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "should be 0., 0., 0., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")

# fds=432




class Binarize_np_Forward_only_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x:torch.Tensor = args[0]
        
        dtype = x.dtype
        flag = x.gt(0.)
        x = flag.to(dtype)+(flag.logical_not()).to(dtype)*-1.
        return x

    @staticmethod
    def backward(ctx, g):
        return g

    pass  # class

# input = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
# output = Binarize_np_Forward_only_Function.apply(input)
# print(output, "should be -1., -1., -1., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")

# fds=432



class Binarize_np_Forward_only(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It clamp the forward inter layer data, while doesn't touch the backward propagation.
    """
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not x.requires_grad):
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return Binarize_np_Forward_only_Function.apply(x)

# layer = Binarize_np_Forward_only()
# input = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "should be -1., -1., -1., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")

# fds=432








