from typing import Any, List, Tuple
import torch
import math


class Grad_Balancer_2out_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x:torch.Tensor = args[0]
        factor_for_path_1 = args[1]
        factor_for_path_2 = args[2]
        ctx.save_for_backward(factor_for_path_1, factor_for_path_2)
        
        x = torch.stack([x, x], dim=0)
        x = x.requires_grad_()
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        # factor_for_path_1:torch.Tensor
        # factor_for_path_2:torch.Tensor
        factor_for_path_1, factor_for_path_2 = ctx.saved_tensors
        
        return g[0]*factor_for_path_1+g[1]*factor_for_path_2, None, None

    pass  # class

# input = torch.tensor([1., 2., 3.], requires_grad=True)
# factor_for_path_1 = torch.tensor([0.1])
# factor_for_path_2 = torch.tensor([0.01])
# output = Grad_Balancer_2out_Function.apply(input, factor_for_path_1, factor_for_path_2)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# fds=432





class Grad_Balancer_2out(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It duplicates the forward path, 
    and multiplies the gradient from different backward path with a given weight.
    """
    def __init__(self, factor1:float, factor2:float, \
                    device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if factor1<=0.:
            raise Exception("Param:factor1 must > 0.")
        if factor2<=0.:
            raise Exception("Param:factor2 must > 0.")
        
        self.factor_for_path_1 = torch.Tensor([factor1])
        self.factor_for_path_2 = torch.Tensor([factor2])
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not x.requires_grad):
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return Grad_Balancer_2out_Function.apply(x, self.factor_for_path_1, self.factor_for_path_2)
    

# layer = Grad_Balancer_2out(0.1, 0.02)
# input = torch.tensor([1., 2., 3.], requires_grad=True)
# output = layer(input)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# fds=432





class Grad_Balancer_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x:torch.Tensor = args[0]
        factor = args[1]
        x = x.unsqueeze(dim=0)
        result = x
        
        for _ in range(1, len(factor)):
            result = torch.concat([result,x], dim=0)
        
        ctx.save_for_backward(factor)
        
        result = result.requires_grad_()
        return result

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        (factor,) = ctx.saved_tensors#this gives a TUPLE!!!
        g_out = torch.zeros_like(g[0])
        
        for i in range(len(factor)):
            g_out += g[i]*(factor[i].item())
            
        return g_out, None

    pass  # class

# input = torch.tensor([1., 2.], requires_grad=True)
# factor = torch.tensor([0.1, 0.02, 0.003])
# output = Grad_Balancer_Function.apply(input, factor)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# input = torch.tensor([[1., 2.], [3., 4.], ], requires_grad=True)
# factor = torch.tensor([0.1, 0.02, 0.003])
# output = Grad_Balancer_Function.apply(input, factor)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# fds=432




class Grad_Balancer(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It duplicates the forward path, 
    and multiplies the gradient from different backward path with a given weight.
    """
    def __init__(self, weight_tensor_for_grad:torch.Tensor = torch.Tensor([1., 1.]), \
                    device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if len(weight_tensor_for_grad.shape)!=1:
            raise Exception("Param:weight_tensor_for_grad should be a vector.")
        for i in range(len(weight_tensor_for_grad)):
            if weight_tensor_for_grad[i]<=0.:
                raise Exception(f'The [{i}] element in the factor tensor is <=0.. It must be >0..')
            
        self.weight_tensor_for_grad = weight_tensor_for_grad
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not x.requires_grad):
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return Grad_Balancer_Function.apply(x, self.weight_tensor_for_grad)

# factor = torch.tensor([0.1, 0.02, 0.003])
# layer = Grad_Balancer(factor)
# input = torch.tensor([1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# fds=432









