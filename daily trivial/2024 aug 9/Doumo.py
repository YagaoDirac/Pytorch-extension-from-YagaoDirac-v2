
from typing import Any, List, Tuple, Optional, Self
import torch
import math
from Gramo import GradientModification



class XModificationFunction(torch.autograd.Function):
    r'''input param:
    >>> x:torch.Tensor (must be set as require_grad = True)
    >>> scaling_ratio = torch.tensor([1.])
    >>> epi = torch.tensor([1e-5])
    >>> div_me_when_g_too_small = torch.tensor([1e-3])

    retur type: torch.Tensor
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, scaling_ratio:float = torch.tensor([1.]), \
        #               epi=torch.tensor([1e-5]), \
        #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
        #but python grammar punched me.
        x_in:torch.Tensor = args[0]
        scaling_ratio = args[1]
        epi = args[2]
        div_me_when_g_too_small = args[3]
        # the default values:
        # scaling_ratio = torch.tensor([1.])
        # epi = torch.tensor([0.00001])
        # div_me_when_g_too_small = torch.tensor([0.001])
        # the definition of the 3 param are different from the previous version
        
        if len(x_in.shape)!=2:
            raise Exception("XModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")

        length:torch.Tensor = x_in.mul(x_in).sum(dim=1,).sqrt()
        too_small:torch.Tensor = length.le(epi)
        div_me = too_small.logical_not()*length + too_small*div_me_when_g_too_small
        div_me = div_me.unsqueeze(dim=1)
        div_me = div_me.to(x_in.dtype)
        x_out:torch.Tensor = x_in/div_me

        scaling_ratio = scaling_ratio.to(x_in.dtype)
        if 1.!=scaling_ratio.item():
            x_out *= scaling_ratio
            pass
        
        return x_out

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        return g, None, None, None

    pass  # class



# '''dtype adaption.'''
# scaling_ratio = torch.tensor([1.], dtype=torch.float64)
# epi=torch.tensor([1e-5], dtype=torch.float32)
# div_me_when_g_too_small = torch.tensor([1e-3], dtype=torch.float16)
# a = torch.tensor([[0.]], dtype=torch.float16)
# original_dtype = a.dtype
# print(XModificationFunction.apply(a,scaling_ratio,epi,div_me_when_g_too_small))
# print("should be ", original_dtype)
# fds=432

# '''when x_in is too small.'''
# scaling_ratio = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0.0000012]])
# print(XModificationFunction.apply(input,scaling_ratio,epi,div_me_when_g_too_small))
# print("should be ", input/div_me_when_g_too_small)

# '''when x_in is NOT too small.'''
# scaling_ratio = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0.12]])
# print(XModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small))
# print("should be 1.")
# fds=432

# '''The shape is [batches, inside a batch]. Computation is limited inside each batch.'''
# scaling_ratio = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0.12, 0.12]])
# print(XModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small))
# print("should be 0.7, 0.7")

# input = torch.tensor([[0.12], [0.12]])
# print(XModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small))
# print("should be 1., 1.")


# input = torch.tensor([[0.1, 0.173], [0.12, 0.12]])
# print(XModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small))
# print("should be 0.5, 0.86, 0.7, 0.7")
# fds=432

# '''Other 3 input besides the main input x.'''
# input = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# print(XModificationFunction.apply(input, torch.tensor([1.]), torch.tensor([0.001]),torch.tensor([0.1])))
# print("should be 0.001, 0.001, 0.7, 0.7")

# input = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# print(XModificationFunction.apply(input, torch.tensor([2.]), torch.tensor([0.001]),torch.tensor([0.1])))
# print("should be 0.002, 0.002, 1.4, 1.4")
# fds=432




class XModification(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """
    def __init__(self, scaling_ratio:float = 1., \
                       epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio]), requires_grad=False)
        self.scaling_ratio.requires_grad_(False)
        self.epi=torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
        self.epi.requires_grad_(False)
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small]), requires_grad=False)
        self.div_me_when_g_too_small.requires_grad_(False)
        #raise Exception("untested")
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.

        if len(x.shape)!=2:
            raise Exception("XModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return XModificationFunction.apply(x, self.scaling_ratio, self.epi, \
                                                   self.div_me_when_g_too_small)
    def set_scaling_ratio(self, scaling_ratio:float)->None:
        the_device = self.scaling_ratio.device
        the_dtype = self.scaling_ratio.dtype
        self.scaling_ratio.data = torch.tensor([scaling_ratio], device=the_device, dtype=the_dtype)
        self.scaling_ratio.requires_grad_(False)
        pass
    def set_epi(self, epi:float)->None:
        the_device = self.epi.device
        the_dtype = self.epi.dtype
        self.epi.data = torch.tensor([epi], device=the_device, dtype=the_dtype)
        self.epi.requires_grad_(False)
        pass
    def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
        the_device = self.div_me_when_g_too_small.device
        the_dtype = self.div_me_when_g_too_small.dtype
        self.div_me_when_g_too_small.data = torch.tensor([div_me_when_g_too_small], device=the_device, dtype=the_dtype)
        self.div_me_when_g_too_small.requires_grad_(False)
        pass

    def extra_repr(self) -> str:
        return f'scaling_ratio={self.scaling_ratio.item():.4e}, epi={self.epi.item():.4e}, div_me_when_g_too_small={self.div_me_when_g_too_small.item():.4e}'

    pass#end of class
#No tests currently.


class DoubleModification(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.xmo = XModification()
        self.gramo = GradientModification()
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.xmo(x)
        x = self.gramo(x)
        return x
    pass #end of function.



class test_FCNN_with_doumo_stack_test(torch.nn.Module): 
    def __init__(self, in_features: int, out_features: int, \
        mid_width:int, num_layers:int, \
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if num_layers<2:
            raise Exception("emmmm")
        self.in_features = in_features
        self.out_features = out_features
        self.mid_width = mid_width
        self.num_layers = num_layers
        
        self.linears = torch.nn.ParameterList([])        
        self.linears.append(torch.nn.Linear(in_features, mid_width))
        for _ in range(num_layers-2):
            self.linears.append(torch.nn.Linear(mid_width, mid_width))
            pass
        self.linears.append(torch.nn.Linear(mid_width, out_features))
        
        self.gramos = torch.nn.ParameterList([GradientModification() for _ in range(num_layers)])      
        self.relus = torch.nn.ParameterList([torch.nn.ReLU() for _ in range(num_layers-1)])       
        self.xmos = torch.nn.ParameterList([XModification() for _ in range(num_layers-1)])      
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        for i in range(self.num_layers-1):
            self.linears[i](x)
            self.gramos[i](x)
            self.relus[i](x)
            self.xmos[i](x)
            pass
        self.linears[-1](x)
        self.gramos[-1](x)
        return x
    pass #end of function.



继续
