from typing import Any, List, Optional, Self
import torch
import math


class GradientModificationFunction(torch.autograd.Function):
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
        x:torch.Tensor = args[0]
        scaling_ratio = args[1]
        epi = args[2]
        div_me_when_g_too_small = args[3]
        # the default values:
        # scaling_ratio = torch.tensor([1.])
        # epi = torch.tensor([0.00001]) 
        # div_me_when_g_too_small = torch.tensor([0.001]) 
        # the definition of the 3 param are different from the previous version
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")

        ctx.save_for_backward(scaling_ratio, epi, div_me_when_g_too_small)
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        scaling_ratio:torch.Tensor
        scaling_ratio, epi, div_me_when_g_too_small = ctx.saved_tensors

        #the shape should only be rank2 with[batch, something]
        # original_shape = g.shape
        # if len(g.shape) == 1:
        #     g = g.unsqueeze(1)
        # protection against div 0    
        length:torch.Tensor = g.mul(g).sum(dim=1,).sqrt()
        too_small:torch.Tensor = length.le(epi)
        div_me = too_small.logical_not()*length + too_small*div_me_when_g_too_small
        div_me = div_me.unsqueeze(dim=1)
        div_me = div_me.to(g.dtype)
        g_out:torch.Tensor = g/div_me
        
        scaling_ratio = scaling_ratio.to(g.dtype)
        if 1.!=scaling_ratio.item():
            g_out *= scaling_ratio
            pass

        return g_out, None, None, None

    pass  # class


class GradientModification(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """
    scaling_ratio:torch.Tensor
    def __init__(self, scaling_ratio:float = 1., \
                       epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio]))
        self.scaling_ratio.requires_grad_(False)
        self.epi=torch.nn.Parameter(torch.tensor([epi]))
        self.epi.requires_grad_(False)
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small]))
        self.div_me_when_g_too_small.requires_grad_(False)
        #raise Exception("untested")
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if not x.requires_grad:
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction.apply(x, self.scaling_ratio, self.epi, \
                                                   self.div_me_when_g_too_small)
    def set_scaling_ratio(self, scaling_ratio:float)->None:
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio], requires_grad=False))
        self.scaling_ratio.requires_grad_(False)
    def set_epi(self, epi:float)->None:
        self.epi = torch.nn.Parameter(torch.tensor([epi], requires_grad=False))
        self.epi.requires_grad_(False)
    def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small], requires_grad=False))
        self.div_me_when_g_too_small.requires_grad_(False)
        pass



#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111



x = torch.tensor([[5., 6., 7.]])
raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], ])
out_features = raw_weight.shape[0]
index_of_max = raw_weight.max(dim=1).indices
print(index_of_max, "index_of_max")
output = x[:, index_of_max]
print(output, "output")
g_in = torch.tensor([[11., 111.]])
grad_for_raw_weight = torch.zeros_like(raw_weight)
iota = torch.linspace(0, out_features-1, out_features, dtype=torch.int32)
print(iota, "iota")
grad_for_raw_weight[iota, index_of_max] = output#x[:, index_of_max]
print(grad_for_raw_weight, "grad_for_raw_weight")
fds=432








class DigitalMapper_V2(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #还没写参数保护。
        x = args[0]
        raw_weight = args[1]
        
        index_of_max = raw_weight.max
        x = x[:][index_of_max]
        output = x
        ctx.save_for_backward(output, index_of_max)
        return output

    @staticmethod
    def backward(ctx, g_in):
        output:torch.Tensor
        index_of_max:torch.Tensor
        output, index_of_max = ctx.saved_tensors

        return g_out, None

    pass  # class



        














class DigitalMapper_V2_still_optimizable(torch.nn.Module):
    r'''This layer is NOT designed to be used directly.
    Use the wrapper class DigitalMapper.
    '''
    #__constants__ = []
    
    auto_merge_duration:int
    update_count:int
    
    def __init__(self, in_features: int, out_features: int, \
                        scaling_ratio:float = 200., \
                    device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        self.raw_weight.requires_grad_(False)
        self.fake_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.fake_weight.requires_grad_(False)
        
        self.gramo = GradientModification(scaling_ratio=scaling_ratio)
        pass

    def reset_parameters(self) -> None:
        '''copied from torch.nn.Linear'''
        torch.nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        pass
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True    
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def set_scaling_ratio(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
    def __update_fake_weight(self):
        temp = self.raw_weight.data.max(dim=1, keepdim=False)
        
        self.fake_weight.data = torch.zeros_like(self.fake_weight.data)
        self.fake_weight[:][temp[:]] =1.
        
        self.fake_weight = self.fake_weight.requires_grad_()
            
    def get_one_hot_format(self)->torch.nn.Parameter:
        self.__update_fake_weight()
        return self.fake_weight
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not input.requires_grad):
                raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
        if len(input.shape)!=2:
            raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")
        
        #if self.training:
        
        self.__update_fake_weight()
        x = self.fake_weight.matmul(input)
        return x
    
    
    
    
    def before_step(self):
        with torch.no_grad():
            self.raw_weight.grad = self.fake_weight.grad
            self.fake_weight.grad = None
            pass

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
    pass

layer = DigitalMapper_V2_still_optimizable(2,3)
fds=432







# class DigitalMapper_eval_only(torch.nn.Module):
#     r'''This class is not designed to be created by user directly. 
#     Use DigitalMapper.can_convert_into_eval_only_mode 
#     and DigitalMapper.convert_into_eval_only_mode to create this layer.
    
#     And, if I only provide result in this form, it's possible to make puzzles.
#     To solve the puzzle, figure the source of this layer.
#     '''
#     def __init__(self, in_features: int, out_features: int, indexes:torch.Tensor, \
#                     device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         if len(indexes.shape)!=1:
#             raise Exception("Param:indexes must be a rank-1 tensor. This class is not designed to be created by user directly. Use DigitalMapper.can_convert_into_eval_only_mode and DigitalMapper.convert_into_eval_only_mode to create this layer.")
#         self.indexes = torch.nn.Parameter(indexes, requires_grad=False)
#         self.indexes.requires_grad_(False)
#     pass
#     def accepts_non_standard_range(self)->bool:
#         return False
#     def outputs_standard_range(self)->bool:
#         return True    
#     def outputs_non_standard_range(self)->bool:
#         return not self.outputs_standard_range()
#     def forward(self, input:torch.Tensor):
#         x = input[:, self.indexes]
#         #print(x)
#         return x



# class DigitalMapper(torch.nn.Module):
#     r'''This layer is designed to be used between digital layers. 
#     The input should be in STANDARD range so to provide meaningful output
#     in STANDARD range. It works for both 01 and np styles.
    
#     Notice: unlike most layers in this project, this layer is stateful.
#     In other words, it has inner param in neural network path.
    
#     Remember to concat a constant 0. and 1. to the input before sending into this layer.
#     In other words, provide Vdd and Vss as signal to the chip.
#     '''
#     #__constants__ = []
    
#     def __init__(self, in_features: int, out_features: int, \
#                     scaling_ratio_for_gramo:float = 200., \
#                     auto_merge_duration:int = 20, raw_weight_boundary_for_f32:float = 15., \
#                     device=None, dtype=None) -> None:   #, debug_info = ""
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()

#         self.in_features = in_features
#         self.out_features = out_features

#         self.inner_raw = DigitalMapper_Non_standard(in_features, out_features, \
#                             auto_merge_duration, raw_weight_boundary_for_f32, \
#                             scaling_ratio = scaling_ratio_for_gramo, \
#                             device=None, dtype=None)
        
#         #self.Final_Binarize = Binarize_01_Forward_only()
#         self.Final_Binarize = Binarize_01_to_01()#maybe this one is better.
#         #
#         pass
  
  
#     def get_scaling_ratio_for_inner_raw(self)->torch.Tensor:
#         '''simply gets the inner'''
#         return self.inner_raw.gramo.scaling_ratio
#     def set_scaling_ratio_for_inner_raw(self, scaling_ratio:float):
#         '''simply sets the inner'''
#         self.inner_raw.set_scaling_ratio(scaling_ratio)
#         pass
        
#     def accepts_non_standard_range(self)->bool:
#         return False
#     def outputs_standard_range(self)->bool:
#         return True    
#     def outputs_non_standard_range(self)->bool:
#         return not self.outputs_standard_range()
    
#     def print_after_softmax(self):
#         self.inner_raw.print_after_softmax()
    
#     def forward(self, input:torch.Tensor)->torch.Tensor:
#         x = input
#         x = self.inner_raw(x)
#         x = self.Final_Binarize(x)
#         return x

#     def extra_repr(self) -> str:
#         return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'
 
#     def can_convert_into_eval_only_mode(self)->Tuple[torch.Tensor, torch.Tensor]:
#         r'''output:
#         >>> [0] can(True) or cannot(False)
#         >>> [1] The least "max strength" of each output. Only for debug.
#         '''
#         after_softmax = self.inner_raw.raw_weight.softmax(dim=1)
        
#         #print(after_softmax[:5])
#         max_strength_for_each_output = after_softmax.amax(dim=1, keepdim=True)
#         #print(max_strength_for_each_output)
#         least_strength = max_strength_for_each_output.amin()
#         #print(least_strength)
#         result_flag = least_strength>0.7
#         return (result_flag, least_strength)
        
#         #the_least_of_after_softmax = after_softmax.max(dim=1)
        
#         # old code
#         # flag = after_softmax.gt(0.7)#0.7 or at least 0.5.
#         # _check_flag = flag.any(dim=1)
#         # _check_flag_all = _check_flag.all()
        
        
#         return _check_flag_all.item()
        
#     def convert_into_eval_only_mode(self)->DigitalMapper_eval_only:
#         after_softmax = self.inner_raw.raw_weight.softmax(dim=1)
#         flag = after_softmax.gt(0.7)#0.7 or at least 0.5.
#         #print(flag, "flag")
#         _check_flag = flag.any(dim=1)
#         _check_flag_all = _check_flag.all()
#         if not _check_flag_all.item():
#             raise Exception("The mapper can NOT figure out a stable result. Train the model more.")
        
#         argmax = flag.to(torch.int8).argmax(dim=1)
#         #print(argmax, "argmax")
#         result = DigitalMapper_eval_only(self.in_features, self.out_features, argmax)
#         #print(result)
#         return result
#         #raise Exception("Not implemented yet. This feature only helps in deployment cases, but nobody cares about my project. It's never gonne be deployed.")
    
#     pass
