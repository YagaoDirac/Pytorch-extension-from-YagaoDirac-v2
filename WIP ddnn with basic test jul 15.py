'''
The first part is a plain copy from the MIG python file.
Explaination and tests can be found in the original file.

THE NEW CODE STARTS FROM AROUND LINE 150 !!!
THE NEW CODE STARTS FROM AROUND LINE 150 !!!
THE NEW CODE STARTS FROM AROUND LINE 150 !!!
THE NEW CODE STARTS FROM AROUND LINE 150 !!!
THE NEW CODE STARTS FROM AROUND LINE 150 !!!
THE NEW CODE STARTS FROM AROUND LINE 150 !!!
THE NEW CODE STARTS FROM AROUND LINE 150 !!!
THE NEW CODE STARTS FROM AROUND LINE 150 !!!
THE NEW CODE STARTS FROM AROUND LINE 150 !!!
'''
from typing import Any, List, Tuple, Optional, Union
import torch
import math

# __all__ = [
#     'GradientModification',
#     'MirrorLayer',
#     'MirrorWithGramo',
#     'GradientModificationFunction', #Should I expose this?
#     'Linear_gramo', #Should I rename this one? Or somebody help me with the naming?
#     ]

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















































#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////








'''
Here is the end of copying the MIG code.
From here all are DDNN code. DDNN ==  Digital Deep Neural Network.
'''





'''
Documentation!!!

The docs for DDNN, the Digital Deep Neural Network

[[[Range denotation]]]

Since it's impossible to do backward propagation with boolean values,
all the tools in this project use some analogy(real number, or floating point number)
way to represent.
This requires the defination of range.
In this project, 3 ranges are used.
1st is non clamped real number, or the plain floating point number,
any f32, f64, f16, bf16, f8 are all in this sort.
2nd is 01, any number in the 1st format but in the range of 0. to 1..
3rd is np, negetive 1 to positive 1.
But in code, the 1st is processed the same way as the 3rd. 
So, in code, only 01 and np are present.
But the difference between the code for 01 and np is very little. 
I only want the denotation to be in class names, so it's less error prone.

If you send wrong input to any layer, it's undefined behavior.
It's your responsibility to make sure about this.

The relationship between different range of representations:
(float means real number which is -inf to +inf, np means -1 to +1, 01 means 0 to +1)
Binarize layers:
from 01: 01(Binarize_01), np(Binarize_01_to_np)
from np:  01(Binarize_np_to_01), np(Binarize_np)
from real number, it's the same as "np as input"
to binarize into: 01(Binarize_np_to_01), np(Binarize_np)

Gate layers:
from 01: 01 (AND_01, OR_01,XOR_01)
from np: np (AND_np, OR_np,XOR_np)

The following 2 layers only accept input in a very strict range.
Before using them, scale down the range of input to the corresbonding range.
ADC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01  (with AND2_01, OR2_01, ???_01)
(Notice, no np version ADC)

DAC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01
(Notice, no np version DAC)

Analogy signal, real number, should be processed with binarize layer
(Binarize_np_to_01 or Binarize_np) or any ADC layer to become digital, 
then processed with other digital layers.
This is the same as the convention of electronics.



[[[The standard range]]]

In this project, the standard range is defined as: if a range is exact (0., 1.) as a 01 range,
or exact(-1., 1.) as a np range, if we don't care about the error of floating point number calculation,
it's a standard range.

unfinished docs
现在有2个办法解决这个问题了。门层里面只有一个可以得到标准输出范围。

Boundate.scale_back_to_... functions help you scale any tensor back to a standard range.
Why is this important, the reason is that, only binarize layers can handle non-standard range.
All the other layers, gates, adc, dac are sensitive to this detail.
In this project, I provide binarize layers without automatically scale the output back
to standard range, because in most cases, uses need to stack them in customized ways.
I provided an example of  unfinished docs....

Another naming convention. If a layer provides output in non standard range,
its name ends with some weird notification.



[[[More Binarized output and real output range]]]

Since the output and input of Binarize_01 are in the same range, and share the same meaning,
it's possible to stack it a few times to get a more binarized result.
This also applies for Binarize_np.
Then, it's critical to balance the layers count against the big_number param.
Also, the real output range of each layer is not the theorimatic output range of 
sigmoid or tanh, because the input is not -inf to +inf.
This leads to the Boundary_calc function. It calculates the real output range.
This function comes alone with 2 helper functions, 
Boundary_calc_helper_wrapper and Boundary_calc_helper_unwrapper.

To get better precision, scale_back_to_np    [unfinished docs]



[[[Slightly better way to use binarize layers]]]
Basically, the binarize layer do this:
1, offset the input if the input is in a 01 range, because the middle value is 0.5.
2, scale the input up using the factor of big_number.
3, calculate sigmoid or tanh.
4, feed the result into a gramo layer.
Sometimes we want a gramo layer before sigmoid or tanh function to protect the gradient.
But no matter how you stack binarize layers, you still need to manually add 
a gramo layer in the beginning, this is true for gates layers. Because gates layers 
use a very small big_number for the first binarize layer, which also shrink the gradient.



[unfinished docs]
[[[Gates layers]]]
First, the most important thing you really MUST remember to use gates layers properly.
All the gates layers in this project only accepts exactly (0., 1.) or (-1., 1.) range as input.
If you have some input in (0.1, 0.8), it's undefined behavior.
Use the scale_back_to_01 and scale_back_to_np function to scale the input while correct the range.
If the input range is not semetric around the mid value, the scaling function will change
the relationship against the mid value a bit. 
If the input range is (0.1, 0.8), then, logically, 0.45 is the mid value.
If you assume 0.5 as the mid value, ok I didn't prepare a tool for you. 
Since the real range of output in such cases is a bit undefined.
Should I use (0., 1.) as the output range, or (0., 0.875)?
Maybe I would do something in the future.

If the input is in the exact mid, the output is useless.
I simply choose a random way to handle this, it only guarantees no nan or inf is provided.
The XOR and NXOR use multiplication in training mode,
which makes them 2 more sensitive to "out of range input" than others,
if you use out of range range as input.


Second, to protect the gradient, 
you may need to modify the big_number of the first binarize layer in any gates layer.
The principle is that, bigger input dim leads smaller big_number.
If AND2 needs 6., AND3 needs something less than 6., maybe a 4. or 3..
AND, NAND, OR, NOR use similar formula, while XOR, NXOR use another formula,
and the formula affects the choice of big_number.
Example in the documentation string in any gates layer.


If the input of a gate is too many, then, the greatest gradient is a lot times
bigger than the smallest one. If the greatest-grad element needs n epochs to train,
and its gradient is 100x bigger than the smallest, then the least-grad element needs 
100n epochs to train. This may harm the trainability of the least-grad element by a lot.
Basically, the gates layers are designed to use as AND2, AND3, but never AND42 or AND69420.




unfinished docs



'''






#part 1 data gen

def int_into_floats(input:torch.Tensor, bit_count:int)->torch.Tensor:
    if len(input.shape)!=2 or input.shape[1]!=1:
        raise Exception("Param:input must be rank-2. Shape is [batch, 1].")
    
    mask = torch.logspace(0,bit_count-1,bit_count, base=2, dtype=torch.int64)
    mask = mask.to(input.device)
    result = input[:,].bitwise_and(mask)
    result = result.to(torch.bool)
    result = result.to(torch.float32)
    return result

# '''int_into_floats'''  
# input = torch.tensor([[0],[1],[2],[3],[7],])
# print(int_into_floats(input,7))
# fds=432

def floats_into_int(input:torch.Tensor)->torch.Tensor:
    if len(input.shape)!=2:
        raise Exception("Param:input must be rank-2. Shape is [batch, -1].")
    
    bit_count = input.shape[1]
    mask = torch.logspace(0,bit_count-1,bit_count, base=2, dtype=torch.int64)
    mask = mask.unsqueeze(dim=1)
    mask = mask.to(torch.float32)
    input = input.gt(0.5)
    input = input.to(torch.float32)
    result = input.matmul(mask)
    result = result.to(torch.int64)
    return result

# '''floats_into_int'''   
# input = torch.tensor([[0],[1],[2],[3],[7],])
# input = int_into_floats(input,7)
# print(floats_into_int(input))
# fds=432

def data_gen_half_adder_1bit(batch:int, is_cuda:bool=True):#->Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randint(0,2,[batch,1])
    b = torch.randint(0,2,[batch,1])
    if is_cuda:
        a = a.cuda()
        b = b.cuda()
    target = a+b
    a = int_into_floats(a,1)    
    b = int_into_floats(b,1)    
    input = torch.concat([a,b], dim=1)
    input = input.requires_grad_()
    target = int_into_floats(target,2)    

    return (input, target)
  
# '''half_adder_1bit_data_gen'''    
# (input, target) = data_gen_half_adder_1bit(3)
# print(input)
# print(input.shape)
# print(target)
# print(target.shape)
# fds=432

def data_gen_full_adder(bits:int, batch:int, is_cuda:bool=True):#->Tuple[torch.Tensor, torch.Tensor]:
    range = 2**bits
    #print(range)
    a = torch.randint(0,range,[batch,1])
    b = torch.randint(0,range,[batch,1])
    c = torch.randint(0,2,[batch,1])
    if is_cuda:
        a = a.cuda()
        b = b.cuda()
        c = c.cuda()
    target = a+b+c
    a = int_into_floats(a,bits)    
    b = int_into_floats(b,bits)    
    c = int_into_floats(c,1)    
    input = torch.concat([a,b,c], dim=1)
    input = input.requires_grad_()
    target = int_into_floats(target,bits+1)    

    return (input, target)
  
# '''data_gen_full_adder_1bit'''    
# (input, target) = data_gen_full_adder(3,11)
# print(input)
# print(input.shape)
# print(target)
# print(target.shape)
# fds=432







# class Grad_Balancer_2out_Function(torch.autograd.Function):
#     r'''This class is not designed to be used directly.
#     A critical safety check is in the wrapper class.    
#     '''
#     @staticmethod
#     def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
#         x:torch.Tensor = args[0]
#         factor_for_path_1 = args[1]
#         factor_for_path_2 = args[2]
#         ctx.save_for_backward(factor_for_path_1, factor_for_path_2)
        
#         x = torch.stack([x, x], dim=0)
#         x = x.requires_grad_()
#         return x

#     @staticmethod
#     def backward(ctx, g):
#         #super().backward()
#         # factor_for_path_1:torch.Tensor
#         # factor_for_path_2:torch.Tensor
#         factor_for_path_1, factor_for_path_2 = ctx.saved_tensors
        
#         return g[0]*factor_for_path_1+g[1]*factor_for_path_2, None, None

#     pass  # class

# # input = torch.tensor([1., 2., 3.], requires_grad=True)
# # factor_for_path_1 = torch.tensor([0.1])
# # factor_for_path_2 = torch.tensor([0.01])
# # output = Grad_Balancer_2out_Function.apply(input, factor_for_path_1, factor_for_path_2)
# # print(output, "output")
# # g_in = torch.ones_like(output)
# # torch.autograd.backward(output, g_in,inputs= input)
# # print(input.grad, "grad")

# # fds=432





# class Grad_Balancer_2out(torch.nn.Module):
#     r"""This is a wrapper class. It helps you use the inner functional properly.
    
#     It duplicates the forward path, 
#     and multiplies the gradient from different backward path with a given weight.
#     """
#     def __init__(self, factor1:float, factor2:float, \
#                     device=None, dtype=None) -> None:
#         # factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
        
#         if factor1<=0.:
#             raise Exception("Param:factor1 must > 0.")
#         if factor2<=0.:
#             raise Exception("Param:factor2 must > 0.")
        
#         self.factor_for_path_1 = torch.Tensor([factor1])
#         self.factor_for_path_2 = torch.Tensor([factor2])
#         pass
#     def forward(self, x:torch.Tensor)->torch.Tensor:
#         # If you know how pytorch works, you can comment this checking out.
#         if self.training and (not x.requires_grad):
#             raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

#         #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
#         #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
#         return Grad_Balancer_2out_Function.apply(x, self.factor_for_path_1, self.factor_for_path_2)
    

# # layer = Grad_Balancer_2out(0.1, 0.02)
# # input = torch.tensor([1., 2., 3.], requires_grad=True)
# # output = layer(input)
# # print(output, "output")
# # g_in = torch.ones_like(output)
# # torch.autograd.backward(output, g_in,inputs= input)
# # print(input.grad, "grad")

# # fds=432





# class Grad_Balancer_Function(torch.autograd.Function):
#     r'''This class is not designed to be used directly.
#     A critical safety check is in the wrapper class.    
#     '''
#     @staticmethod
#     def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
#         x:torch.Tensor = args[0]
#         factor = args[1]
#         x = x.unsqueeze(dim=0)
#         result = x
        
#         for _ in range(1, len(factor)):
#             result = torch.concat([result,x], dim=0)
        
#         ctx.save_for_backward(factor)
        
#         result = result.requires_grad_()
#         return result

#     @staticmethod
#     def backward(ctx, g):
#         #super().backward()
#         (factor,) = ctx.saved_tensors#this gives a TUPLE!!!
#         g_out = torch.zeros_like(g[0])
        
#         for i in range(len(factor)):
#             g_out += g[i]*(factor[i].item())
            
#         return g_out, None

#     pass  # class

# # input = torch.tensor([1., 2.], requires_grad=True)
# # factor = torch.tensor([0.1, 0.02, 0.003])
# # output = Grad_Balancer_Function.apply(input, factor)
# # print(output, "output")
# # g_in = torch.ones_like(output)
# # torch.autograd.backward(output, g_in,inputs= input)
# # print(input.grad, "grad")

# # input = torch.tensor([[1., 2.], [3., 4.], ], requires_grad=True)
# # factor = torch.tensor([0.1, 0.02, 0.003])
# # output = Grad_Balancer_Function.apply(input, factor)
# # print(output, "output")
# # g_in = torch.ones_like(output)
# # torch.autograd.backward(output, g_in,inputs= input)
# # print(input.grad, "grad")

# # fds=432




# class Grad_Balancer(torch.nn.Module):
#     r"""This is a wrapper class. It helps you use the inner functional properly.
    
#     It duplicates the forward path, 
#     and multiplies the gradient from different backward path with a given weight.
#     """
#     def __init__(self, weight_tensor_for_grad:torch.Tensor = torch.Tensor([1., 1.]), \
#                     device=None, dtype=None) -> None:
#         # factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         if len(weight_tensor_for_grad.shape)!=1:
#             raise Exception("Param:weight_tensor_for_grad should be a vector.")
#         for i in range(len(weight_tensor_for_grad)):
#             if weight_tensor_for_grad[i]<=0.:
#                 raise Exception(f'The [{i}] element in the factor tensor is <=0.. It must be >0..')
            
#         self.weight_tensor_for_grad = weight_tensor_for_grad
#         pass
#     def forward(self, x:torch.Tensor)->torch.Tensor:
#         # If you know how pytorch works, you can comment this checking out.
#         if self.training and (not x.requires_grad):
#             raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

#         #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
#         #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
#         return Grad_Balancer_Function.apply(x, self.weight_tensor_for_grad)

# # factor = torch.tensor([0.1, 0.02, 0.003])
# # layer = Grad_Balancer(factor)
# # input = torch.tensor([1., 2.], requires_grad=True)
# # output = layer(input)
# # print(output, "output")
# # g_in = torch.ones_like(output)
# # torch.autograd.backward(output, g_in,inputs= input)
# # print(input.grad, "grad")

# # fds=432





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







class Binarize_base(torch.nn.Module):
    r"""This layer is not designed to be used by users.
    You should use the 6 following layers instead, they are:
    >>> Binarize_float_to_01(unfinished docs. deleted?)
    >>> Binarize_float_to_np(unfinished docs. deleted?)
    >>> Binarize_01
    >>> Binarize_01_to_np
    >>> Binarize_np
    >>> Binarize_np_to_01
    
    This layer is the base layer of all Binarize layers.
    
    The first 2(Binarize_float_to_01 and Binarize_float_to_np) accepts any range(-inf to +inf) as input,
    then Binarize_01 and Binarize_01_to_np accepts 0. to 1.,
    then Binarize_np and Binarize_np_to_01 accepts -1. to 1..
    But because they all use sigmoid or tanh, they all accepts any real number as input. 
    So you don't have to worry about running into inf or nan.
    
    Binarize_float_to_01, Binarize_01 and Binarize_np_to_01 output 0. to 1.,
    while
    Binarize_float_to_np, Binarize_01_to_np and Binarize_np output -1. to 1..
    But in practice, the real output range depends on all the other params, the input range, the big_number.
    But no matter what happens, the output range is guarunteed to be with in (0., 1.) and (-1., 1.) respectively, and the boundaries are all excluded, because they are direct output of either sigmoid or tanh.
    
    The eval mode is a bit different. The output is directly from step function, in other words, a simple comparison.
    The output can only be the boundary value. For output mode of 01, it's either 0., or 1.. For np it's -1. or 1..
    But if the input equals to the threshold, the behavior is different, it's directly from a preset number.
    To set it, use set_output_when_ambiguous function. The threshold is directly inferred from the input range, 0.5 for 01, 0. for others.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
    """
    __constants__ = ['output_when_ambiguous','big_number',
                     'input_is_01', 'output_is_01',]
    output_when_ambiguous: float
    big_number_list: torch.nn.parameter.Parameter
    input_is_01:bool

    def __init__(self, input_is_01:bool, output_is_01:bool, \
                    big_number_list:torch.nn.Parameter = \
                        torch.nn.Parameter(torch.tensor([5.]), requires_grad = False), \
                    device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #safety:
        # if output_when_ambiguous<-1. or output_when_ambiguous>1. :
        #     raise Exception("Param:output_when_ambiguous is out of range. If you mean to do this, use the set_output_when_ambiguous function after init this layer.")
        if len(big_number_list.shape)!=1:
            raise Exception("Param:big_number must be in rank-1 shape. Or simply use the default value.")
        if big_number_list.shape[0]<1:
            raise Exception("Param:big_number needs at least 1 element. Or simply use the default value.")
        
        self.input_is_01 = input_is_01
        self.output_is_01 = output_is_01
        
        self.big_number_list = big_number_list
        self.big_number_list.requires_grad_(False)
        
        self.output_when_ambiguous = 0.
        if output_is_01:
            self.output_when_ambiguous = 0.5

        #this layer also needs the info for gramo.
        self.gramo = GradientModification(scaling_ratio,epi,div_me_when_g_too_small)
        pass

    def set_output_when_ambiguous(self, output:float):
        if self.output_is_01:
            if output<0. or output>1.:
                raise Exception("The output can only be between 0 and 1.")
            pass
        else:
            if output<-1. or output>1.:
                raise Exception("The output can only be between -1 and 1.")
            pass
        
        self.output_when_ambiguous = output
        pass

    def set_big_number_with_float(self, big_number_list:List[float], \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''I_know_Im_setting_a_value_which_may_be_less_than_1 is 
            designed only for the first binarize layer in gates layers. 
            If inputs doesn't accumulate enough gradient, you should 
            consider using some 0.3 as big_number to protect the 
            trainability of the intput-end of the model.
        '''
        if len(big_number_list)<1:
            raise Exception("Param:big_number needs at least 1 element. Or simply use the default value.")
            pass
        for i, e in enumerate(big_number_list):
            if I_know_Im_setting_a_value_which_may_be_less_than_1:
                # This case is a bit special. I made this dedicated for the gates layers.
                if e<=0.:
                    raise Exception(f"Param:big_number the [{i}] element is not big enough.")
                    pass
                
            else:# The normal case
                if e<1.:
                    raise Exception(f"Param:big_number the [{i}] element is not big enough.Use I_know_Im_setting_a_value_which_may_be_less_than_1 = True if you know what you are doing.")
                    pass
        self.big_number_list:torch.nn.Parameter = torch.nn.Parameter(torch.tensor(big_number_list))
        self.big_number_list.requires_grad_(False)
        #self.update_output_range()
        pass
    
    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass

    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return False
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    def __get_offset(self)->float:
        if self.input_is_01:
            return -0.5
        else:
            return 0.
            pass
        
        #untested?
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

            if self.big_number_list.requires_grad:
                raise Exception("Unreachable path.")
            if len(self.big_number_list.shape)>1:
                raise Exception("The big number list must be rank-1.")

            x = input
            
            # The offset. This only works for 01 as input.
            offset = self.__get_offset()
            if 0. != offset:
                x = x + offset
                pass
            for i in range(self.big_number_list.shape[0]-1):
                big_number = self.big_number_list[i]
                if big_number!=1.:
                    x = x*big_number
                    pass
                # All the intermediate calculation use the np style.
                x = torch.tanh(x)
                pass
            
            big_number = self.big_number_list[-1]
            if big_number!=1.:
                x = x*big_number
                pass
            if self.output_is_01:
                x = torch.sigmoid(x)
            else:
                x = torch.tanh(x)
                pass
            
            result = self.gramo(x)
            return result
        
        else:#eval mode
            '''I know the bit wise operations are much faster. 
            But for now I'll keep it f32 for simplisity.
            '''
            with torch.inference_mode():
                output_dtype = input.dtype
                     
                '''The optimization:
                If output is 01, and output_when_ambiguous is 0., only > is 1.
                If output is 01, and output_when_ambiguous is 1., only >= is 1.
                If output is 01, and output_when_ambiguous is not 0. or 1., it needs 2 parts, the == and >.
                If output is np, and output_when_ambiguous is -1., it needs <= and >.
                If output is np, and output_when_ambiguous is  0., it needs < and >.
                If output is np, and output_when_ambiguous is  1., it needs < and >=.
                If output is np, and output_when_ambiguous is not -1. or 1., it needs all 3 ops.
                '''
                
                ambiguous_at = self.__get_offset()*-1. #opposite of offset.
                
                if self.output_is_01:
                    if 0. == self.output_when_ambiguous:
                        return input.gt(ambiguous_at).to(output_dtype)
                    if 1. == self.output_when_ambiguous:
                        return input.ge(ambiguous_at).to(output_dtype)
                    result = input.eq(ambiguous_at).to(output_dtype)*self.output_when_ambiguous
                    result += input.gt(ambiguous_at).to(output_dtype)
                    return result
                else:# output is np
                    if -1. == self.output_when_ambiguous:
                        result = input.le(ambiguous_at).to(output_dtype)*-1.
                        result += input.gt(ambiguous_at).to(output_dtype)
                        return result
                    if 0. == self.output_when_ambiguous:
                        result = input.lt(ambiguous_at).to(output_dtype)*-1.
                        result += input.gt(ambiguous_at).to(output_dtype)
                        return result
                    if 1. == self.output_when_ambiguous:
                        result = input.lt(ambiguous_at).to(output_dtype)*-1.
                        result += input.ge(ambiguous_at).to(output_dtype)
                        return result
                    result = input.lt(ambiguous_at).to(output_dtype)*-1.
                    result += input.eq(ambiguous_at).to(output_dtype)*self.output_when_ambiguous
                    result += input.gt(ambiguous_at).to(output_dtype)
                    return result
        #End of The function

    
    def extra_repr(self) -> str:
        input_str = "[0. to 1.]"
        if not self.input_is_01:
            input_str = "[-1. to 1.] or (-inf. to +inf.)"
            pass
        output_str = "(0. to 1.)" if self.output_is_01 else "(-1. to 1.)"
        mode_str = "training" if self.training else "evaluating"
        
        result = f'Banarize layer, theorimatic input range:{input_str}, theorimatic output range:{output_str}, in {mode_str} mode'
        return result

    pass


# '''All the big numbers in the list are properly iterated.'''
# model = Binarize_base(True,True)
# model.set_big_number_with_float([2., 3., 4.])
# input = torch.tensor([[0., 0.25, 0.5, 1.]], requires_grad=True)
# pred = model(input)
# fds=432


# bp01 = BoundaryPair.from_float(0., 1.)
# '''01 to 01'''
# layer = Binarize_base(True, True, 1., True)
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), f'should be {layer.get_output_range(bp01).lower.item():.4f}, 0.5, {layer.get_output_range(bp01).upper.item():.4f}')
# layer.set_big_number(9.)
# print(layer(input), "should be 0.01_, 0.5, 0.99_")

# bpnp = BoundaryPair.from_float(-1., 1.)
# '''non 01 to 01'''
# layer = Binarize_base(False, True, 1., True)
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), f'should be {layer.get_output_range(bpnp).lower.item():.4f}, 0.5, {layer.get_output_range(bp01).upper.item():.4f}')
# layer.set_big_number(9.)
# print(layer(input), "should be 0.01_, 0.5, 0.99_")

# ''' real number as input is the same as np. No test for it.''' 

# fds=432


# '''This tests all the output formula of eval mode.'''
# trivial_big_number = 9.
# input = torch.tensor([[-1., 0., 1.]])

# layer = Binarize_base(False, True, trivial_big_number, True,)
# layer.eval()
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be 0., 0., 1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 0., 1., 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0., 0.25, 1.")

# layer = Binarize_base(False, True, trivial_big_number, False,)
# layer.eval()
# layer.set_output_when_ambiguous(-1.)
# print(layer(input), "should be -1., -1., 1.")
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be -1., 0., 1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be -1., 1., 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be -1., 0.25, 1.")


# trivial_big_number = 9.
# bp01 = BoundaryPair.from_float(0., 1.)
# input = torch.tensor([[0., 0.5, 1.]])

# layer = Binarize_base(True, True, trivial_big_number, True,)
# layer.eval()
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be 0., 0., 1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 0., 1., 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0., 0.25, 1.")

# layer = Binarize_base(True, True, trivial_big_number, False,)
# layer.eval()
# layer.set_output_when_ambiguous(-1.)
# print(layer(input), "should be -1., -1., 1.")
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be -1., 0., 1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be -1., 1., 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be -1., 0.25, 1.")

# fds=432






class Binarize_analog_to_01(torch.nn.Module):
    r"""This layer accepts the range of -1 to 1, 
    provides a binarized result in the range of 0 to 1, 
    0 and 1 are both excluded in training mode, 
    but included in eval mode.
    
    When input is right in the mid(0.), the default output is set to 0.5.
    Use set_output_when_ambiguous function to modify it.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
    """
    #__constants__ = []

    def __init__(self, big_number_list:List[float] = [1.5], \
                            device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        self.base = Binarize_base(False, True, 
                                  scaling_ratio = scaling_ratio, epi = epi,
                                  div_me_when_g_too_small = div_me_when_g_too_small)
        self.base.set_big_number_with_float(big_number_list)
        self.Final_Binarize = Binarize_01_Forward_only()
        pass

    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def set_big_number_with_float(self, big_number_list:List[float], \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.base.set_big_number_with_float(big_number_list, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
    # def get_output_range(self, input: BoundaryPair)->BoundaryPair:
    #     '''Gets the output range from inner layer'''
    #     return self.base.get_output_range(input)
    
    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return False
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.base(x)
        x = self.Final_Binarize(x)
        return x

    pass


# '''Tests the new big number list.'''
# model = Binarize_analog_to_01()
# model.set_big_number_with_float([2., 3., 4., 1079.])
# input = torch.tensor([[-1., -0.5, 0., 0.25, 0.5, 1.]], requires_grad=True)
# pred = model(input)
# fds=432

# '''the tests below are mainly about the gradient modification layers.'''
# layer = Binarize_analog_to_01([1.])
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.0012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.25, the same as the dirivitive of sigmoid at 0")

# layer = Binarize_analog_to_01([5.])
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.0012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 1.25, sigmoid(0) is 0.25, big_number is 5.")

# '''An extra gramo layer before the binarize layer to protect the gradient.'''
# gramo = GradientModification()
# layer = Binarize_analog_to_01([1.])
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(gramo(input))
# g_in = torch.tensor([[0.0012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "now it's 1.")

# layer = Binarize_analog_to_01([1.])
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.0000012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, f'should be 0.0000012*1000*0.25({0.0000012*1000*0.25})')

# layer = Binarize_analog_to_01([1.])
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25*0.7 or 0.17_")

# layer = Binarize_analog_to_01([1.])
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.5, 0.05]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be smaller than 0.25, but the first should be near 0.25, the second should be very small. The first one should be 10x of the second one.")

# '''This test is about the batch. Since I specify the shape 
# # for gramo layer as [batch count, length with in each batch].
# # Because I already tested the gramo layer in MIG, I don't plan to redo them here.'''
# layer = Binarize_analog_to_01([1.])
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25*0.7 or 0.17_")
# layer = Binarize_analog_to_01([1.])
# input = torch.tensor([[0.], [0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12], [0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25, since they are in different batches.")

# '''shape doesn't matter.'''
# layer = Binarize_analog_to_01([1.])
# input:torch.Tensor = torch.linspace(-3., 3., 7, dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "4 0s, then 3 1s")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")

# layer = Binarize_analog_to_01([1.])
# input:torch.Tensor = torch.tensor([-5., -1., -0.5, 0., 0.5, 1., 5.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "4 0s, then 3 1s")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")

'''This part tests the output_when_ambiguous. It means, when the input it 0, which doesn't indicate towards any direction. The default output is 0.5. But it's customizable.'''
layer = Binarize_analog_to_01([1.])
input = torch.tensor([[0.]], requires_grad=True)
print(layer(input), "should be 0.5")
layer.eval()
output = layer(input)
print(layer(input), "should also be 0.5")
layer.set_output_when_ambiguous(0.)
print(layer(input), "should be 0.")
layer.set_output_when_ambiguous(1.)
print(layer(input), "should be 1.")
layer.set_output_when_ambiguous(0.25)
print(layer(input), "should be 0.25")
input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
print(layer(input), "should be 0., 0.25, 1.")
继续
继续
继续
继续
继续
继续
继续
继续
继续
继续
继续
继续
'''This part tests the output_when_ambiguous. 
# It means, when the input it 0., which doesn't indicate towards any direction. 
# The default output is 0.5. But it's customizable.'''
# layer = Binarize_analog_to_01([1.])
# input = torch.tensor([[0.]], requires_grad=True)
# print(layer(input), "should be 0.5")
# layer.eval()
# output = layer(input)
# print(layer(input), "should also be 0.5")
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be 0.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0.25")
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be -0., 0.25, 1.")

# '''The output'''
# layer = Binarize_analog_to_01([1.])
# print(layer)

# fds=432




raise Exception("")


class Binarize_01_to_01_non_standard(torch.nn.Module):
    r"""This layer accepts the range of 0 to 1, 
    provides a binarized result in the range of 0 to 1, 
    0 and 1 are both excluded in training mode, 
    but included in eval mode.
    
    When input is right in the mid(0.5), the default output is set to 0.5.
    Use set_output_when_ambiguous function to modify it.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
    """
    #__constants__ = []

    def __init__(self, big_number_list:List[float] = [3.], \
                            device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        self.base = Binarize_base(True, True,  
                                  scaling_ratio = scaling_ratio, epi = epi,
                                  div_me_when_g_too_small = div_me_when_g_too_small)
        self.base.set_big_number_with_float(big_number_list)
        pass


    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.base.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
    # def get_output_range(self, input: BoundaryPair)->BoundaryPair:
    #     '''Gets the output range from inner layer'''
    #     return self.base.get_output_range(input)

    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return False
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    pass



# '''shape doesn't matter.'''
# layer = Binarize_01__non_standard_output(1.)
# input:torch.Tensor = torch.tensor([-2., 0., 0.25, 0.5, 0.75, 1., 3.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "the middle one should be 0.5")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")
# print(layer.get_output_range(BoundaryPair.from_float(-2., 3.)))

# '''This part tests the output_when_ambiguous. 
# It means, when the input it 0.5, which doesn't indicate towards any direction. 
# The default output is 0.5. But it's customizable.'''
# layer = Binarize_01__non_standard_output(1.)
# input = torch.tensor([[0.5]], requires_grad=True)
# print(layer(input), "should be 0.5")
# layer.eval()
# output = layer(input)
# print(layer(input), "should also be 0.5")
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be 0.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0.25")
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), "should be 0., 0.25, 1.")

# fds=432








# example
# example
# example
# example
# example
# example
# example
# example
# example
# '''np to 01 3 times, in this implementation, it's np to 01 to 01 to 01.'''
'''Because the forward path yields the same result as that one before, 
the only difference is grad. The param:big_number affects the grad by a lot.'''
class example_Binarize_analog_to_01_3times(torch.nn.Module):
    r"""This example layer shows how to use the binarize layer.
    """
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        #only the first binarize layer is analog to 01. Others must all be 01 to 01.
        self.Binarize1 = Binarize_analog_to_01_non_standard(1.5)
        self.Binarize2_optional = Binarize_01_to_01_non_standard(4.)
        self.Binarize3_optional = Binarize_01_to_01_non_standard(4.)
        self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        # gramo doesn't affect this.
        
        x = input
        #x = self.optionalGramo(x)#Probably not needed.
        x = self.Binarize1(x)
        x = self.Binarize2_optional(x)
        x = self.Binarize3_optional(x)
        x = self.Final_Binarize(x)
        return x

    pass


# input = torch.tensor([-3., -1., 0., 1., 2.], requires_grad=True)
# input = input.unsqueeze(dim=0)
# layer = example_Binarize_analog_to_01_3times()
# output = layer(input)
# print(output, "output, should be 0 0 0 1 1")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs=input)
# print(input.grad, "grad")

# fds=432









# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.


# '''
# !!! This part is old docs. I don't want to delete it, bacause it tells something. 
# This old test made me add the range system into this project.
# The data shown here was got before that.

# Although this test shows the ability for more layers to binarize the result more, 
# but in fact this behavior is not guaranteed. It heavily depends on the big_number param.
# In this example, the big_number for all the 3 layers are set to 3., 6., 6.. 
# If you set a much smaller big_number, the result will be "blurred".
# A always working trick is that, set breakpoint in the forward function,
# and check out what each layer does to the intermediate result.

# This test happens to show another interesting feature.
# The results are:
# input:         -1.,   -0.5,    0.,     ...
# 1 layer output:0.0474, 0.1824, 0.5000, ...
# 3 layer output:0.0674, 0.0977, 0.5000, ...
# 1 layer grad:  0.0606, 0.2001, 0.3354, ...
# 3 layer grad:  0.0077, 0.0692, 0.7294, ...
# The 3 layer version binarize the -0.5 more(0.0977<0.1824),
# but gets relarively smaller grad(0.0692/0.7294<0.2001/0.3354).
# BUT!!!
# The 3 layer version binarize the -1. LESS!!!(0.0674>0.0474),
# wihle still gets relarively smaller grad(0.0077/0.7294<0.0606/0.3354).

# The grad is affected by gradient modification layer.

# The argument is that, it's not reliable to tell the 
# relationship of grad by only read the relationship of outputs.

# Also, notice, if we only consider the range. If we send a range(0, 1)
# to sigmoid, the result is [0.5, 0.7311). The range is getting smaller.
# With big_number, we can scale the range back to something near to the input.
# But, no matter what happens, is we only multiply the output by a scale number,
# and send it to sigmoid, the result is definitely not gonna fill all the range(0, 1).
# The parts near to the 2 boundaries are lost.
# Even if the result is binarized more, it's not possible to tell by only reading the range.
# But the relationship of grad tells everything.


# Now, the scale_back to standard range feature is added through the project.
# Let's compare the old result vs the new result:
# input:                -1.,   -0.5,    0.,     ...
# OLD 1 layer output:0.0474, 0.1824, 0.5000, ...
# NEW                0.0000, 0.1491, 0.5000, ...

# OLD 3 layer output:0.0674, 0.0977, 0.5000, ...
# NEW                0.0000, 0.0350, 0.5000, ...

# OLD 1 layer grad:  0.0606, 0.2001, 0.3354, ...
# NEW                0.0606, 0.2001, 0.3354, ...

# OLD 3 layer grad:  0.0077, 0.0692, 0.7294, ...
# NEW                0.0077, 0.0692, 0.7294, ...

# The grad is not modified at all, which is probably due to gramo, and probably what I want.
# '''

# old test.
# layer = Binarize_np_to_01__non_standard_output(3.)
# input:torch.Tensor = torch.tensor([-1., -0.5, 0., 0.5, 1.], requires_grad=True,)
# input = input.unsqueeze(0)
# output = layer(input)
# r = layer.get_output_range(BoundaryPair.make_np())
# output = r.scale_back_to_01(output)
# print(output, "processed with 1 layer.")
# g_in = torch.ones_like(output, dtype=torch.float32)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad from 1 layer.")
# print(layer.get_output_range(BoundaryPair.make_np()))

# model = example_Binarize_np_to_01_3times()
# input = torch.tensor([-1., -0.5, 0., 0.5, 1.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# output = model(input)
# print(output, "processed with 3 layer.")
# g_in = torch.ones_like(output, dtype=torch.float32)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad from 3 layer.")


# fds=432





class Binarize_01_to_01(torch.nn.Module):
    r"""This example layer shows how to use the binarize layer.
    """
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        #only the first binarize layer is analog to 01. Others must all be 01 to 01.
        self.Binarize = Binarize_01_to_01_non_standard(10.)
        self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.Binarize.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass
    
    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.Binarize(x)
        x = self.Final_Binarize(x)
        return x

    pass

# input = torch.tensor([[0., 0.1, 0.2, 0.5, 1.]], requires_grad=True)
# #input = input.unsqueeze(dim=0)
# layer = Binarize_01_to_01()
# output = layer(input)
# print(output, "output, should be 0 0 0 0 1")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs=input)
# print(input.grad, "grad")

# fds=432






class DigitalMapper_Non_standard(torch.nn.Module):
    r'''This layer is NOT designed to be used directly.
    Use the wrapper class DigitalMapper.
    '''
    #__constants__ = []
    
    auto_merge_duration:int
    update_count:int
    
    def __init__(self, in_features: int, out_features: int, \
                    auto_merge_duration:int = 20, raw_weight_boundary_for_f32:float = 15., \
                        scaling_ratio:float = 200., \
                    device=None, dtype=None) -> None:   #, debug_info = ""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if raw_weight_boundary_for_f32<5. :
            raise Exception("In my test, it goes to almost 6. in 4000 epochs. If you know what you are doing, comment this checking out.")

        self.in_features = in_features
        self.out_features = out_features
        self.raw_weight_boundary_for_f32 = raw_weight_boundary_for_f32
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        self.gramo = GradientModification(scaling_ratio=scaling_ratio)
        #self.set_scaling_ratio(100.) emmm

        #self.Final_Binarize = Binarize_01_Forward_only()

        #to keep track of the training.
        self.auto_merge_duration:int = auto_merge_duration
        self.update_count:int = 0
        pass

    def reset_parameters(self) -> None:
        '''copied from torch.nn.Linear'''
        torch.nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        pass
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return False    
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def set_scaling_ratio(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
    def print_after_softmax(self):
        print(self.raw_weight.softmax(dim=1))
        pass
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        # For simplicity, no distill here .
        
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not input.requires_grad):
                raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
        if len(input.shape)!=2:
            raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")
        
        if self.training:
            if self.update_count>=self.auto_merge_duration:
                self.update_count = 0
                with torch.no_grad():
                    #this automatic modification may mess sometime.
                    boundary = self.raw_weight_boundary_for_f32
                    if self.raw_weight.dtype == torch.float64:
                        boundary *= 2.
                        pass
                    if self.raw_weight.dtype == torch.float16:
                        boundary *= 0.5
                        pass
                    
                    flag = self.raw_weight.gt(boundary)
                    temp = flag*boundary
                    temp = temp.to(self.raw_weight.dtype)
                    temp = temp + self.raw_weight.data*(flag.logical_not())
                    self.raw_weight.data = temp

                    boundary*=-1.
                    flag = self.raw_weight.lt(boundary)
                    temp = flag*boundary
                    temp = temp.to(self.raw_weight.dtype)
                    temp = temp + self.raw_weight.data*(flag.logical_not())
                    self.raw_weight.data = temp
                    
                    mean = self.raw_weight.mean(dim=1,keepdim=True)
                    self.raw_weight.data = self.raw_weight.data-mean
                    pass
                pass
            else:
                self.update_count+=1
                pass
            pass
    
        x = input.unsqueeze(dim=2)
        
        w = self.gramo(self.raw_weight)
        #w = self.raw_weight
        w_after_softmax = w.softmax(dim=1)
        #w_after_softmax = self.gramo2(w_after_softmax)
        
        # if not self.training:
        #     print(x.shape, "1 x")
        #     print(w_after_softmax.shape, "1 w")
        #     fds=432
        #fds = self.raw_weight.dtype
        x = w_after_softmax.matmul(x)
        
        #print(x.shape)
        x = x.squeeze(dim = 2)
        #print(x.shape)
        return x
              

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
 
    # def convert_into_eval_only_mode(self):
    #     raise Exception("Not implemented yet. This feature only helps in deployment cases, but nobody cares about my project. It's never gonne be deployed.")
    
    pass





class DigitalMapper_eval_only(torch.nn.Module):
    r'''This class is not designed to be created by user directly. 
    Use DigitalMapper.can_convert_into_eval_only_mode 
    and DigitalMapper.convert_into_eval_only_mode to create this layer.
    
    And, if I only provide result in this form, it's possible to make puzzles.
    To solve the puzzle, figure the source of this layer.
    '''
    def __init__(self, in_features: int, out_features: int, indexes:torch.Tensor, \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if len(indexes.shape)!=1:
            raise Exception("Param:indexes must be a rank-1 tensor. This class is not designed to be created by user directly. Use DigitalMapper.can_convert_into_eval_only_mode and DigitalMapper.convert_into_eval_only_mode to create this layer.")
        self.indexes = torch.nn.Parameter(indexes, requires_grad=False)
        self.indexes.requires_grad_(False)
    pass
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True    
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    def forward(self, input:torch.Tensor):
        x = input[:, self.indexes]
        #print(x)
        return x



class DigitalMapper(torch.nn.Module):
    r'''This layer is designed to be used between digital layers. 
    The input should be in STANDARD range so to provide meaningful output
    in STANDARD range. It works for both 01 and np styles.
    
    Notice: unlike most layers in this project, this layer is stateful.
    In other words, it has inner param in neural network path.
    
    Remember to concat a constant 0. and 1. to the input before sending into this layer.
    In other words, provide Vdd and Vss as signal to the chip.
    '''
    #__constants__ = []
    
    def __init__(self, in_features: int, out_features: int, \
                    scaling_ratio_for_gramo:float = 200., \
                    auto_merge_duration:int = 20, raw_weight_boundary_for_f32:float = 15., \
                    device=None, dtype=None) -> None:   #, debug_info = ""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.inner_raw = DigitalMapper_Non_standard(in_features, out_features, \
                            auto_merge_duration, raw_weight_boundary_for_f32, \
                            scaling_ratio = scaling_ratio_for_gramo, \
                            device=None, dtype=None)
        
        #self.Final_Binarize = Binarize_01_Forward_only()
        self.Final_Binarize = Binarize_01_to_01()#maybe this one is better.
        #
        pass
  
  
    def get_scaling_ratio_for_inner_raw(self)->torch.Tensor:
        '''simply gets the inner'''
        return self.inner_raw.gramo.scaling_ratio
    def set_scaling_ratio_for_inner_raw(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.inner_raw.set_scaling_ratio(scaling_ratio)
        pass
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True    
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def print_after_softmax(self):
        self.inner_raw.print_after_softmax()
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        x = input
        x = self.inner_raw(x)
        x = self.Final_Binarize(x)
        return x

    def extra_repr(self) -> str:
        return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'
 
    def can_convert_into_eval_only_mode(self)->Tuple[torch.Tensor, torch.Tensor]:
        r'''output:
        >>> [0] can(True) or cannot(False)
        >>> [1] The least "max strength" of each output. Only for debug.
        '''
        after_softmax = self.inner_raw.raw_weight.softmax(dim=1)
        
        #print(after_softmax[:5])
        max_strength_for_each_output = after_softmax.amax(dim=1, keepdim=True)
        #print(max_strength_for_each_output)
        least_strength = max_strength_for_each_output.amin()
        #print(least_strength)
        result_flag = least_strength>0.7
        return (result_flag, least_strength)
        
        #the_least_of_after_softmax = after_softmax.max(dim=1)
        
        # old code
        # flag = after_softmax.gt(0.7)#0.7 or at least 0.5.
        # _check_flag = flag.any(dim=1)
        # _check_flag_all = _check_flag.all()
        
        
        return _check_flag_all.item()
        
    def convert_into_eval_only_mode(self)->DigitalMapper_eval_only:
        after_softmax = self.inner_raw.raw_weight.softmax(dim=1)
        flag = after_softmax.gt(0.7)#0.7 or at least 0.5.
        #print(flag, "flag")
        _check_flag = flag.any(dim=1)
        _check_flag_all = _check_flag.all()
        if not _check_flag_all.item():
            raise Exception("The mapper can NOT figure out a stable result. Train the model more.")
        
        argmax = flag.to(torch.int8).argmax(dim=1)
        #print(argmax, "argmax")
        result = DigitalMapper_eval_only(self.in_features, self.out_features, argmax)
        #print(result)
        return result
        #raise Exception("Not implemented yet. This feature only helps in deployment cases, but nobody cares about my project. It's never gonne be deployed.")
    
    pass




# '''eval_only_mode, so we can have some puzzle.'''
# layer = DigitalMapper(2,3)
# layer.inner_raw.raw_weight.data=torch.Tensor([[2., -2.],[ 0.1, 0.],[ -2., 2.]])
# print(layer.can_convert_into_eval_only_mode())

# input = torch.Tensor([[1., 0.], [0., 1.], [0.123, 0.321],  ])
# input.requires_grad_()
# layer = DigitalMapper(2,1)
# layer.inner_raw.raw_weight.data=torch.Tensor([[2., -2.]])
# print(layer(input), "layer(input)")
# eval_only = layer.convert_into_eval_only_mode()
# print(eval_only)
# print(eval_only(input), "eval_only(input)")

# fds=432


# '''basic test'''
# layer = DigitalMapper(2,3)
# # print(layer.raw_weight.data)
# # print(layer.raw_weight.data.shape)
# layer.raw_weight.data=torch.Tensor([[2., -2.],[ 0.1, 0.],[ -2., 2.]])
# input = torch.Tensor([[1., 0.]])
# input = input.requires_grad_()
# #print(input.softmax(dim=1))#dim = 1!!!
# print(layer(input), "should be 1 1 0")
# print(layer.raw_weight.softmax(dim=1))
# fds=432

# layer = DigitalMapper(2,3)
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# input = input.requires_grad_()
# #print(input.softmax(dim=1))#dim = 1!!!
# print(layer(input), "either 0 or 1")
# print(layer.raw_weight.softmax(dim=1), "they sum up to 1 for each line?")

# layer = DigitalMapper(2,3)
# layer.eval()
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# print(layer(input), "either 0 or 1")
# print(layer.raw_weight.softmax(dim=1), "they sum up to 1 for each line?")


# '''some real training'''
# model = DigitalMapper_Non_standard(2,1)
# model.set_scaling_ratio(1000.)
# loss_function = torch.nn.MSELoss()
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# #input = torch.Tensor([[1., 0.],[0., 1.]])
# input = input.requires_grad_()
# target = torch.Tensor([[1.],[1.],[0.],[0.],])
# #target = torch.Tensor([[1.],[0.],])
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
# # for p in model.parameters():
# #     print(p)

# iter_per_print = 111#1111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
    
#     model.train()
#     pred = model(input)
    
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(pred.shape, "pred")
#     #     print(target.shape, "target")
    
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight.grad, "grad")
        
#     #model.raw_weight.grad = model.raw_weight.grad*-1.
#     #optimizer.param_groups[0]["lr"] = 0.01
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, model.raw_weight.grad, "before update")
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, "raw weight itself")
#     #     print(model.raw_weight.grad, "grad")
#     #     fds=432
#     optimizer.step()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, "after update")

#     model.eval()
#     if epoch%iter_per_print == iter_per_print-1:
#         # print(loss, "loss")
#         # if True:
#         #     print(model.raw_weight.softmax(dim=1), "eval after softmax")
#         #     print(model.raw_weight, "eval before softmax")
#         #     print("--------------")
#         pass

# model.eval()
# print(model(input), "should be", target)
# print(model.raw_weight.softmax(dim=1), "raw_weight after softmax")

# fds=432


# '''The DigitalMapper version. You should use this one only.'''
# in_feature = 2
# model = DigitalMapper(in_feature,1)
# model.set_scaling_ratio(100000.)
# loss_function = torch.nn.MSELoss()
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# input = input.requires_grad_()
# target = torch.Tensor([[1.],[1.],[0.],[0.],])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

# iter_per_print = 1111
# print_count = 3

# print(model.inner_raw.raw_weight.softmax(dim=1), "raw_weight after softmax")
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     # print(pred.shape, "pred.shape")
#     # print(target.shape, "target.shape")
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if epoch%iter_per_print == iter_per_print-1:
#         print(model.inner_raw.raw_weight.softmax(dim=1), "raw_weight after softmax")

# model.eval()
# #input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# print(model(input), "should be", target)
# print(model.inner_raw.raw_weight.softmax(dim=1), "raw_weight after softmax")

# fds=432







class AND_01(torch.nn.Module):
    r""" 
    unfinished docs
    
    AND gate. 
    Only accepts STANDARD 01 range.
    Output is customizable. But if it's not standard, is it useful ?
    
    Input shape: [batch, gate_count*input_count]
    
    Output shape: [batch, gate_count*input_count*(1 or 2)](depends on what you need)
    
    Usually, reshape the output into [batch, all the output], 
    concat it with output of other gates, and a 0 and an 1.
    
    
    unfinished docs.
    first big number和input数量的关系，必须写上。
    
    
    
    Takes any number of inputs. Range is (0., 1.).
    1. is True.
    
    Notice: training mode uses arithmetic ops and binarize layers,
    while eval mode uses simple comparison.
    
    example:
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],], requires_grad=True)
    >>> bound_of_a = Boundary_calc_helper_wrapper(0.1, 0.9)
    >>> a = scale_back_to_01(a, bound_of_a)
    >>> b = torch.tensor([[0.2],[0.2],[0.8],], requires_grad=True)
    >>> bound_of_b = Boundary_calc_helper_wrapper(0.2, 0.8)
    >>> b = scale_back_to_01(b, bound_of_b)
    >>> '''(optional) c = torch.tensor([[0.3],[0.3],[0.7],], requires_grad=True)'''
    >>> '''bound_of_c = Boundary_calc_helper_wrapper(0.3, 0.7)'''
    >>> '''c = scale_back_to_01(c, bound_of_c)'''
    >>> input = torch.concat((a,b), 1)# dim = 1 !!!
    >>> ''' or input = torch.concat((a,b,c), 1) '''
    >>> layer = AND_01()
    >>> result = layer(input)
    
    The evaluating mode is a bit different. It's a simply comparison.
    It relies on the threshold. If the input doesn't use 0.5 as threshold, offset it.
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],]) # requires_grad doesn't matter in this case.
    >>> b = torch.tensor([[0.2],[0.2],[0.8],])
    >>> c = torch.tensor([[0.3],[0.3],[0.7],])
    >>> input = torch.concat((a,b,c), 1) 
    >>> layer = AND_01()
    >>> layer.eval()
    >>> result = layer(input)
    
    These 2 tests show how to protect the gradient.
    Basically the big_number of the first binarize layer is the key.
    By making it smaller, you make the gradient of inputs closer over each elements.
    For AND gate, if it gets a lot False as input, the corresponding gradient is very small.

    >>> a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
    >>> b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
    >>> input = torch.concat((a,b,b), 1) 
    >>> layer = AND_01()
    >>> layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
    >>> result = layer(input)
    >>> print(result)
    >>> g_in = torch.ones_like(result)
    >>> torch.autograd.backward(result, g_in,inputs= input)
    >>> print(input.grad)
    >>> #more~
    >>> input = torch.concat((a,b,b,b,b,b,b,b,b,b,b,b,b), 1) 
    >>> layer = AND_01()
    >>> layer.Binarize1.set_big_number(0.3, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
    >>> result = layer(input)
    >>> print(result)
    >>> g_in = torch.ones_like(result)
    >>> torch.autograd.backward(result, g_in,inputs= input)
    >>> print(input.grad)
    
    example for calculate the output range:
    (remember to modify the function if you modified the forward function)
    
    >>> a = torch.tensor([[0.],[1.],], requires_grad=True)
    >>> input = torch.concat((a,a), 1) 
    >>> layer = AND_01()
    >>> result = layer(input)
    >>> print(result)
    >>> print(layer.get_output_range(2), "the output range. Should be the same as the result.")

    If the performance doesn't meet your requirement, 
    modify the binarize layers. More explaination in the source code.
    """
    
                 
                 
                 #first_big_number:float = 3., 
    def __init__(self, input_per_gate:int = 2, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        
        if input_per_gate<2:
            raise Exception("Param:input_per_gate should >=2.")
        self.input_per_gate = input_per_gate
        
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01_to_01()
        #self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
       
        # No matter what happens, this layer should be designed to output standard binarized result.
        # Even if the sigmoid is avoidable, you still need a final binarize layer.
        # But, Binarize_01_to_01 does this for you, so you don't need an individual final_binarize layer.
        # self.Final_Binarize = Binarize_01_Forward_only()
        pass
   
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            x = x.sum(dim=2, keepdim=False)#dim=2
            #back to rank-2
            
            offset = float(self.input_per_gate)*-1.+1.
            x = x + offset
            
            # binarize 
            x = self.Binarize1(x)

            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = 1.-x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.all(dim=2, keepdim=False)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x.to(input.dtype)
                else:
                    opposite = x.logical_not()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1).to(input.dtype)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite.to(input.dtype)
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'AND/NAND layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass

    

     

# '''Not important. The shape of eval path'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,a,b,a,b,a,b), dim=1) 
# print(input.shape)
# layer = AND_01(input_per_gate=2)
# layer.eval()
# result = layer(input)
# print(result.shape)
# fds=432
    
    

# '''a basic test'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# layer = AND_01()
# layer.Binarize1.set_big_number(3.)#So you can adjust the range of grad.
# result = layer(input)
# print(result, "should be 0 0 1")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# fds=432


# '''This old test shows a trick. You can use slightly non standard input to help
# you track the behavior of the tested layer.'''
# '''These 2 tests show how to protect the gradient.
# Basically the big_number of the first binarize layer is the key.
# By making it smaller, you make the gradient of inputs closer over each elements.
# For AND gate, if it gets a lot false as input, the corresponding gradient is very small.
# '''
# a = torch.tensor([[0.001],[0.992],[0.993],], requires_grad=True)
# b = torch.tensor([[0.004],[0.005],[0.996],], requires_grad=True)
# input = torch.concat((a,b,b,b), dim=1) 
# layer = AND_01(input_per_gate=4, first_big_number=1.1)
# # layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
# result = layer(input)
# print(result, "should be 0., 0., 1.")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad, should be 3x of a's")
# # more~

# old test but tells something.
# '''If the input of a gate is too many, then, the greatest gradient is a lot times
# bigger than the smallest one. If the greatest-grad element needs n epochs to train,
# and its gradient is 100x bigger than the smallest, then the least-grad element needs 
# 100n epochs to train. This may harm the trainability of the least-grad element by a lot.
# Basically, the gates layers are designed to use as AND2, AND3, but never AND42 or AND69420.'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,b,b,b,b,b,b,b,b,b), dim=1) 
# layer = AND_01(input_per_gate=11, first_big_number=0.35)
# result = layer(input)
# print(result)
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad, should be 10x of a's")

# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = AND_01(input_per_gate=3)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 0., 1.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = AND_01()#first big number 5?
# print(layer(input), "should be 0., 0., 1.")
# layer.eval()
# print(layer(input), "should be 0., 0., 1.")
# layer = AND_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0., 0., 1., 1., 1., 0.")
# layer.eval()
# print(layer(input), "should be 0., 0., 1., 1., 1., 0.")
# layer = AND_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 1., 1., 0.")
# layer.eval()
# print(layer(input), "should be 1., 1., 0.")

# '''__str__'''
# layer = AND_01()
# print(layer)

# fds=432




class OR_01(torch.nn.Module):
    r""" 
    OR gate. Takes any number of inputs. Range is (0., 1.).
    1. is True.
    
    Notice: training mode uses arithmetic ops and binarize layers,
    while eval mode uses simple comparison.
    
    example:
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],], requires_grad=True)
    >>> bound_of_a = Boundary_calc_helper_wrapper(0.1, 0.9)
    >>> a = scale_back_to_01(a, bound_of_a)
    >>> b = torch.tensor([[0.2],[0.2],[0.8],], requires_grad=True)
    >>> bound_of_b = Boundary_calc_helper_wrapper(0.2, 0.8)
    >>> b = scale_back_to_01(b, bound_of_b)
    >>> '''(optional) c = torch.tensor([[0.3],[0.3],[0.7],], requires_grad=True)'''
    >>> '''bound_of_c = Boundary_calc_helper_wrapper(0.3, 0.7)'''
    >>> '''c = scale_back_to_01(c, bound_of_c)'''
    >>> input = torch.concat((a,b), 1)# dim = 1 !!!
    >>> ''' or input = torch.concat((a,b,c), 1) '''
    >>> layer = OR_01()
    >>> result = layer(input)
    
    The evaluating mode is a bit different. It's a simply comparison.
    It relies on the threshold. If the input doesn't use 0.5 as threshold, offset it.
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],]) # requires_grad doesn't matter in this case.
    >>> b = torch.tensor([[0.2],[0.2],[0.8],])
    >>> c = torch.tensor([[0.3],[0.3],[0.7],])
    >>> input = torch.concat((a,b,c), 1) 
    >>> layer = OR_01()
    >>> layer.eval()
    >>> result = layer(input)
    
    These 2 tests show how to protect the gradient.
    Basically the big_number of the first binarize layer is the key.
    By making it smaller, you make the gradient of inputs closer over each elements.
    For OR gate, if it gets a lot True as input, the corresponding gradient is very small.

    >>> a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
    >>> b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
    >>> input = torch.concat((a,b,b), 1) 
    >>> layer = AND_01()
    >>> layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
    >>> result = layer(input)
    >>> print(result)
    >>> g_in = torch.ones_like(result)
    >>> torch.autograd.backward(result, g_in,inputs= input)
    >>> print(input.grad)
    >>> #more~
    >>> input = torch.concat((a,b,b,b,b,b,b,b,b,b,b,b,b), 1) 
    >>> layer = AND_01()
    >>> layer.Binarize1.set_big_number(0.3, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
    >>> result = layer(input)
    >>> print(result)
    >>> g_in = torch.ones_like(result)
    >>> torch.autograd.backward(result, g_in,inputs= input)
    >>> print(input.grad)

    example for calculate the output range:
    (remember to modify the function if you modified the forward function)
    
    >>> a = torch.tensor([[0.],[1.],], requires_grad=True)
    >>> input = torch.concat((a,a), 1) 
    >>> layer = OR_01()
    >>> result = layer(input)
    >>> print(result)
    >>> print(layer.get_output_range(2), "the output range. Should be the same as the result.")

    If the performance doesn't meet your requirement, 
    modify the binarize layers. More explaination in the source code.
    """
    

    def __init__(self, input_per_gate:int = 2, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        
        if input_per_gate<2:
            raise Exception("Param:input_per_gate should >=2.")
        self.input_per_gate = input_per_gate
        
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01_to_01()
        #self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
       
        # No matter what happens, this layer should be designed to output standard binarized result.
        # Even if the sigmoid is avoidable, you still need a final binarize layer.
        # But, Binarize_01_to_01 does this for you, so you don't need an individual final_binarize layer.
        # self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            x = x.sum(dim=2, keepdim=False)#dim=2
            #back to rank-2
            
            # no offset needed for OR gate.
            
            x = self.Binarize1(x)
            
            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = 1.-x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.any(dim=2, keepdim=False)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x.to(input.dtype)
                else:
                    opposite = x.logical_not()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1).to(input.dtype)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite.to(input.dtype)
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'OR/NOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass


# '''Not important. The shape of eval path'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,a,b,a,b,a,b), dim=1) 
# print(input.shape)
# layer = OR_01(input_per_gate=2)
# layer.eval()
# result = layer(input)
# print(result.shape)
# fds=432

# '''a basic test'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# layer = OR_01()
# layer.Binarize1.set_big_number(3.)#So you can adjust the range of grad.
# result = layer(input)
# print(result, "should be 0 1 1")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# fds=432



# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = OR_01(input_per_gate=3)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 1., 1.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = OR_01()#first big number 5?
# print(layer(input), "should be 0., 1., 1.")
# layer.eval()
# print(layer(input), "should be 0., 1., 1.")
# layer = OR_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0., 1., 1., 1., 0., 0.")
# layer.eval()
# print(layer(input), "should be 0., 1., 1., 1., 0., 0.")
# layer = OR_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 1., 0., 0.")
# layer.eval()
# print(layer(input), "should be 1., 0., 0.")

# '''__str__'''
# layer = OR_01()
# print(layer)

# fds=432





class XOR_01(torch.nn.Module):
    r""" 
    OR gate. Takes any number of inputs. Range is (0., 1.).
    1. is True.
    
    Notice: training mode uses arithmetic ops and binarize layers,
    while eval mode uses simple comparison.
    
    example:
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],], requires_grad=True)
    >>> bound_of_a = Boundary_calc_helper_wrapper(0.1, 0.9)
    >>> a = scale_back_to_01(a, bound_of_a)
    >>> b = torch.tensor([[0.2],[0.2],[0.8],], requires_grad=True)
    >>> bound_of_b = Boundary_calc_helper_wrapper(0.2, 0.8)
    >>> b = scale_back_to_01(b, bound_of_b)
    >>> '''(optional) c = torch.tensor([[0.3],[0.3],[0.7],], requires_grad=True)'''
    >>> '''bound_of_c = Boundary_calc_helper_wrapper(0.3, 0.7)'''
    >>> '''c = scale_back_to_01(c, bound_of_c)'''
    >>> input = torch.concat((a,b), 1)# dim = 1 !!!
    >>> ''' or input = torch.concat((a,b,c), 1) '''
    >>> layer = OR_01()
    >>> result = layer(input)
    
    The evaluating mode is a bit different. It's a simply comparison.
    It relies on the threshold. If the input doesn't use 0.5 as threshold, offset it.
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],]) # requires_grad doesn't matter in this case.
    >>> b = torch.tensor([[0.2],[0.2],[0.8],])
    >>> c = torch.tensor([[0.3],[0.3],[0.7],])
    >>> input = torch.concat((a,b,c), 1) 
    >>> layer = OR_01()
    >>> layer.eval()
    >>> result = layer(input)
    
    These 2 tests show how to protect the gradient.
    Basically the big_number of the first binarize layer is the key.
    By making it smaller, you make the gradient of inputs closer over each elements.
    For OR gate, if it gets a lot True as input, the corresponding gradient is very small.

    >>> a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
    >>> b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
    >>> input = torch.concat((a,b,b), 1) 
    >>> layer = AND_01()
    >>> layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
    >>> result = layer(input)
    >>> print(result)
    >>> g_in = torch.ones_like(result)
    >>> torch.autograd.backward(result, g_in,inputs= input)
    >>> print(input.grad)
    >>> #more~
    >>> input = torch.concat((a,b,b,b,b,b,b,b,b,b,b,b,b), 1) 
    >>> layer = AND_01()
    >>> layer.Binarize1.set_big_number(0.3, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
    >>> result = layer(input)
    >>> print(result)
    >>> g_in = torch.ones_like(result)
    >>> torch.autograd.backward(result, g_in,inputs= input)
    >>> print(input.grad)

    example for calculate the output range:
    (remember to modify the function if you modified the forward function)
    
    >>> a = torch.tensor([[0.],[1.],], requires_grad=True)
    >>> input = torch.concat((a,a), 1) 
    >>> layer = OR_01()
    >>> result = layer(input)
    >>> print(result)
    >>> print(layer.get_output_range(2), "the output range. Should be the same as the result.")

    If the performance doesn't meet your requirement, 
    modify the binarize layers. More explaination in the source code.
    """
    
    def __init__(self, input_per_gate:int = 2, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        
        if input_per_gate<2:
            raise Exception("Param:input_per_gate should >=2.")
        self.input_per_gate = input_per_gate
        
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_analog_to_01()
        #self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
       
        # No matter what happens, this layer should be designed to output standard binarized result.
        # Even if the sigmoid is avoidable, you still need a final binarize layer.
        # But, Binarize_01_to_01 does this for you, so you don't need an individual final_binarize layer.
        # self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
     
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])

            x = x-0.5
            
            # this redundant *2 makes sure the grad is scaled to the same level as AND and OR path.
            # If you know what you are doing, you can comment this line out.
            x = x*(-2. )

            #back to rank-2
            x = x.prod(dim=2)
            # now, x is NXOR with threshold of 0.
            #x = -x+0.5 # some optimization needed in the future 
                        
            x = self.Binarize1(x)
            
            # Now x is actually the XNOR
            # The output part is different from AND and OR styles.
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = 1.-x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([opposite, x], dim=1)
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
          
          
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.to(torch.int8)
                #overflow doesn't affect the result. 
                x = x.sum(dim=2, keepdim=False)
                x = x%2
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x.to(input.dtype)
                else:
                    opposite = x.logical_not()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1).to(input.dtype)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite.to(input.dtype)
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'OR/NOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass

# '''odd ones results in 1.'''
# input = torch.tensor([[1., 1.]], requires_grad=True)
# layer = XOR_01(input_per_gate=2)
# print(layer(input), "should be 0")
# input = torch.tensor([[1., 1., 1.]], requires_grad=True)
# layer = XOR_01(input_per_gate=3)
# print(layer(input), "should be 1")
# input = torch.tensor([[1., 1., 1., 1.]], requires_grad=True)
# layer = XOR_01(input_per_gate=4)
# print(layer(input), "should be 0")
# fds=432


# '''Not important. The shape of eval path'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,a,b,a,b,a,b), dim=1) 
# print(input.shape)
# layer = XOR_01(input_per_gate=2)
# layer.eval()
# result = layer(input)
# print(result.shape)
# fds=432

# '''a basic test'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# layer = XOR_01()
# layer.Binarize1.set_big_number(3.)#So you can adjust the range of grad.
# result = layer(input)
# print(result, "should be 0 1 0")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# fds=432


# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = XOR_01(input_per_gate=3)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 1., 1.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = XOR_01()#first big number 5?
# print(layer(input), "should be 0., 1., 0.")
# layer.eval()
# print(layer(input), "should be 0., 1., 0.")
# layer = XOR_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0., 1., 0., 1., 0., 1.")
# layer.eval()
# print(layer(input), "should be 0., 1., 0., 1., 0., 1.")
# layer = XOR_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 1., 0., 1.")
# layer.eval()
# print(layer(input), "should be 1., 0., 1.")

# '''__str__'''
# layer = XOR_01()
# print(layer)

# fds=432



# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!






class DigitalSignalProcessingUnit_layer(torch.nn.Module):
    r'''DSPU, Digital Signal Processing Unit.
    
    It's a mapper followed by a compound gate layer.
    
    
    unfinished docs
    It accepts standard binarized range as input, and 
    also outputs standard binarized range.
    
    
    
    This example shows how to handle pure digital signal.
    Only standard binary signals are allowed.
    
    All the gates are 2-input, both-output.
    
    Basic idea:
    >>> x = digital_mapper(input)# the in_mapper
    >>> AND_head = and_gate(input)
    >>> OR_head = or_gate(input)
    >>> XOR_head = xor_gate(input)
    >>> output_unmapped = concat(AND_head, OR_head, XOR_head, 0s, 1s)
    >>> output = digital_mapper(output_unmapped)# the out_mapper
    '''
    #__constants__ = ['',]

    def __init__(self, in_features: int, \
        and_gates: int, or_gates: int, xor_gates: int, \
            scaling_ratio_for_gramo_in_mapper:float = 200., \
                    device=None, dtype=None) -> None:       #, debug_info = ""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_mapper_in_features = in_features
        self.in_mapper_out_features = (and_gates+or_gates+xor_gates)*2
        self.and_gates = and_gates
        self.or_gates = or_gates
        self.xor_gates = xor_gates
        
        # Creates the layers.
        self.in_mapper = DigitalMapper(self.in_mapper_in_features, self.in_mapper_out_features,
                                       scaling_ratio_for_gramo=scaling_ratio_for_gramo_in_mapper)#,
                                       #debug_info = debug_info)
        
        output_mode = 1#1_is_both
        self.and_gate = AND_01(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
        self.or_gate = OR_01(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
        self.xor_gate = XOR_01(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
       
        #end of function        
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    
    def get_in_features(self)->int:
        return self.in_mapper_in_features
    def get_out_features(self)->int:
        return self.in_mapper_out_features+2
    def get_mapper_raw_weight(self)->torch.nn.Parameter:
        return self.in_mapper.inner_raw.raw_weight
    
    def get_mapper_scaling_ratio_for_inner_raw(self)->torch.Tensor:
        '''simply gets from the inner layer.'''
        return self.in_mapper.get_scaling_ratio_for_inner_raw()
    def set_scaling_ratio_for_inner_raw(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        self.in_mapper.set_scaling_ratio_for_inner_raw(scaling_ratio)
    
    def get_gates_big_number(self)->Tuple[float,float,float]:
        big_number_of_and_gate = self.and_gate.Binarize1.Binarize.base.big_number_list
        big_number_of_or_gate = self.or_gate.Binarize1.Binarize.base.big_number_list
        big_number_of_xor_gate = self.xor_gate.Binarize1.Binarize.base.big_number_list
        return (big_number_of_and_gate,big_number_of_or_gate,big_number_of_xor_gate)
    def set_gates_big_number(self, big_number_of_and_gate:float,
                    big_number_of_or_gate:float, big_number_of_xor_gate:float):
        '''use any number <=0. to tell the code not to modify the corresponding one.'''
        if big_number_of_and_gate>0.:
            self.and_gate.Binarize1.Binarize.base.big_number_list = big_number_of_and_gate
        if big_number_of_or_gate>0.:
            self.or_gate.Binarize1.Binarize.base.big_number_list = big_number_of_or_gate
        if big_number_of_xor_gate>0.:
            self.xor_gate.Binarize1.Binarize.base.big_number_list = big_number_of_xor_gate
        #end of function.
     
    def print_after_softmax(self):
        self.in_mapper.print_after_softmax()
     
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''This example only shows the training path.'''
        x = input
        x = self.in_mapper(x)
            
        pos_from = 0
        pos_to = self.and_gates*2
        and_head:torch.Tensor = self.and_gate(x[:, pos_from:pos_to])
        pos_from = pos_to
        pos_to += self.or_gates*2
        or_head = self.or_gate(x[:, pos_from:pos_to])              
        pos_from = pos_to
        pos_to += self.xor_gates*2
        xor_head = self.xor_gate(x[:, pos_from:pos_to])
        zeros = torch.zeros([input.shape[0],1])
        zeros = zeros.to(and_head.device).to(and_head.dtype)
        ones = torch.ones([input.shape[0],1])
        ones = ones.to(and_head.device).to(and_head.dtype)
        x = torch.concat([and_head, or_head, xor_head, zeros, ones],dim=1)
        
        
        #补一个测试。。
        
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
#        x 需要 grad吗，应该需要吧。
        
        return x
    #end of function
    
    def extra_repr(self) -> str:
        result = f'In features:{self.in_mapper.in_features}, out features:{self.get_out_features()}, AND2:{self.and_gates}, OR2:{self.or_gates}, XOR2:{self.xor_gates}'
        return result
    
    def get_info(self, directly_print:bool=False) -> str:
        result = f'In features:{self.in_mapper.in_features}, out features:{self.get_out_features()}, AND2:{self.and_gates}, OR2:{self.or_gates}, XOR2:{self.xor_gates}'
        if directly_print:
            print(result)
        return result
    
    pass

# input = torch.tensor([[1.]])
# input = input.requires_grad_()
# layer = DigitalSignalProcessingUnit_layer(1,1,1,1)
# output = layer(input)
# print(output, "should be 1., 0., 1., 0., 0., 1., 0., 1.")
# fds=432

# layer = DigitalSignalProcessingUnit(11,3,4,5)
# layer.get_info(True)
# fds=432



class DSPU(torch.nn.Module):
    r'''n DSPU layers in a row. Interface is the same as the single layer version.
    
    It contains an extra mapper layer in the end. 
    
    It's a ((mapper+gates)*n)+mapper structure.
    
    More info see DigitalSignalProcessingUnit.
    '''
    #__constants__ = ['',]
    def __init__(self, in_features: int, out_features:int, \
                and_gates: int, or_gates: int, xor_gates: int, num_layers:int, \
                    scaling_ratio_for_gramo_in_mapper:float = 200., \
                    scaling_ratio_for_gramo_in_mapper_for_first_layer:Optional[float] = None, \
                    scaling_ratio_for_gramo_in_mapper_for_out_mapper:Optional[float] = None, \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        if scaling_ratio_for_gramo_in_mapper_for_first_layer is None:
            scaling_ratio_for_gramo_in_mapper_for_first_layer = scaling_ratio_for_gramo_in_mapper
            
        if scaling_ratio_for_gramo_in_mapper_for_out_mapper is None:
            scaling_ratio_for_gramo_in_mapper_for_out_mapper = scaling_ratio_for_gramo_in_mapper
        
        self.first_layer = DigitalSignalProcessingUnit_layer(
            in_features, and_gates, or_gates, xor_gates, 
            scaling_ratio_for_gramo_in_mapper=scaling_ratio_for_gramo_in_mapper_for_first_layer)#, debug_info = "first layer")
        
        mid_width = self.first_layer.get_out_features()
        self.second_to_last_layers = torch.nn.modules.container.ModuleList( \
                [DigitalSignalProcessingUnit_layer(mid_width, and_gates, or_gates, xor_gates, 
                        scaling_ratio_for_gramo_in_mapper=scaling_ratio_for_gramo_in_mapper)   
                    for i in range(num_layers-1)])
        self.num_layers = num_layers
           
        self.out_mapper = DigitalMapper(mid_width, out_features,
                                       scaling_ratio_for_gramo=scaling_ratio_for_gramo_in_mapper_for_out_mapper)# , debug_info="out mapper")
        #end of function        
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    def get_mapper_scaling_ratio_for_inner_raw_from_all(self)->List[torch.Tensor]:
        result:List[torch.Tensor] = []
        result.append(self.first_layer.get_mapper_scaling_ratio_for_inner_raw())
        for layer in self.second_to_last_layers:
            result.append(layer.get_mapper_scaling_ratio_for_inner_raw())
            pass
        result.append(self.out_mapper.get_scaling_ratio_for_inner_raw())
        return result
    def print_mapper_scaling_ratio_for_inner_raw_from_all(self):
        result = self.get_mapper_scaling_ratio_for_inner_raw_from_all()
        print("Scaling ratio of the inner raw weight:")
        print("First layer:", result[0].item())
        list_for_middle_layers:List[float] = []
        for i_ in range(len(self.second_to_last_layers)):
            the_scaling_ratio_here = result[i_+1]
            list_for_middle_layers.append(the_scaling_ratio_here.item())     
            pass
        if len(list_for_middle_layers)>0:
            print("Mid layers: ", list_for_middle_layers)
            pass
        print("Last layer:", result[-1].item())
        return 
    
    # 3 setters for the inner gramo
    def set_scaling_ratio_for_inner_raw_for_first_layer(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        self.first_layer.set_scaling_ratio_for_inner_raw(scaling_ratio)
        pass
    def set_scaling_ratio_for_inner_raw_for_middle_layers(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        for layer in self.second_to_last_layers:
            layer.set_scaling_ratio_for_inner_raw(scaling_ratio)
            pass
        pass
    def set_scaling_ratio_for_inner_raw_for_out_mapper(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        self.out_mapper.set_scaling_ratio_for_inner_raw(scaling_ratio)
        pass
    
    
    def get_gates_big_number_from_all(self)->List[Tuple[float,float,float]]:
        result:List[Tuple[float,float,float]] = []
        result.append(self.first_layer.get_gates_big_number())
        for layer in self.second_to_last_layers:
            result.append(layer.get_gates_big_number())
            pass
        return result
    def print_gates_big_number_from_all(self):
        result = self.get_gates_big_number_from_all()
        str_result = str(result[0])
        for i in range(1, len(result)):
            this_one = result[i]
            str_result +="\n"
            str_result +=str(this_one)
            pass
        print(str_result)
        return 
    
    def set_gates_big_number_for_all(self, big_number_of_and_gate:float,
                    big_number_of_or_gate:float, big_number_of_xor_gate:float):
        '''use any number <=0. not to modify the corresponding one.'''
        self.first_layer.set_gates_big_number(big_number_of_and_gate = big_number_of_and_gate,
                                            big_number_of_or_gate = big_number_of_or_gate, 
                                            big_number_of_xor_gate = big_number_of_xor_gate)
        for layer in self.second_to_last_layers:
            layer.set_gates_big_number(big_number_of_and_gate = big_number_of_and_gate,
                                    big_number_of_or_gate = big_number_of_or_gate, 
                                    big_number_of_xor_gate = big_number_of_xor_gate)
            pass
        pass
        #end of function
        
    def Are_all_mapper_over_threshold(self, can_print_debug_info:bool = False)->bool:
        least_that_number = 1.
        raise Exception("")
        #self.first_layer.in_mapper.
        
        
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''This example only shows the training path.'''
        x = input
        fds = self.first_layer.in_mapper.inner_raw.raw_weight.dtype
        x = self.first_layer(x)
        fdsa = self.first_layer.in_mapper.inner_raw.raw_weight.dtype

        for layer in self.second_to_last_layers:
            x = layer(x)
            pass
        x = self.out_mapper(x)
        return x
    #end of function
    
    def get_info(self, directly_print:bool=False) -> str:
        result = f'{self.num_layers} DSPUs in a row. In features:{self.in_features}, out features:{ self.out_features}, AND2:{self.first_layer.and_gates}, OR2:{self.first_layer.or_gates}, XOR2:{self.first_layer.xor_gates}'
        if directly_print:
            print(result)
        return result
    
    pass
# layer = DSPU(11,7,2,3,4,5)
# layer.get_info(True)
# fds=432

def DSPU_adder_test(in_a:int, in_b:int, in_c:int, model:DSPU):
    device = next(model.parameters()).device
    string = f'{in_a}+{in_b}+{in_c} is evaluated as: '
    a = torch.tensor([[in_a]], device=device)
    b = torch.tensor([[in_b]], device=device)
    c = torch.tensor([[in_c]], device=device)
    target = a+b+c
    a = int_into_floats(a,1)    
    b = int_into_floats(b,1)    
    c = int_into_floats(c,1)    
    input = torch.concat([a,b,c], dim=1)
    input = input.requires_grad_()
    target = int_into_floats(target,1+1) 
    print(string, model(input), "should be", target)


# '''a basic test'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# target = torch.tensor([[0.], [1.], [1.]])
# model = DSPU(input.shape[1],target.shape[1],1,1,1,1)
# model.get_info(directly_print=True)

# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
# # for p in model.parameters():
# #     print(p)

# iter_per_print = 111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     if False:
#       print(pred.shape)
#       print(target.shape)
#       fds=423
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(pred, "pred")
#     #     print(target, "target")
    
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight.grad, "grad")
        
#     #optimizer.param_groups[0]["lr"] = 0.01
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, model.raw_weight.grad, "before update")
#     optimizer.step()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, "after update")

# model.eval()
# pred = model(input)
# print(pred, "should be ", target)

# fds=432



# '''does it fit in half adder'''
# batch = 10000
# (input, target) = data_gen_half_adder_1bit(batch, is_cuda=False)
# model = DSPU(input.shape[1],target.shape[1],1,1,1,1)
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

# iter_per_print = 111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     if False:
#         print(pred.shape)
#         print(target.shape)
#         fds=432
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# with torch.inference_mode():
#     model.eval()
#     pred = model(input)
#     #print(pred, "should be ", target)

#     temp = pred.eq(target)
#     temp = temp.sum()
#     temp = temp.to(torch.float32)
#     acc = temp/float(batch*target.shape[1])
#     acc = acc.item()
#     print(acc, "<- the accuracy")

# fds=432



# '''4 layers should be able to fit in AND3'''
# batch = 10000
# input = torch.randint(0,2,[batch,3])
# target = input.all(dim=1,keepdim=True)
# input = input.to(torch.float32)
# input = input.requires_grad_()
# target = target.to(torch.float32)
# # print(input)
# # print(output)
# model = DSPU(input.shape[1],target.shape[1],4,4,4,4)
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

# model.cuda()
# input = input.cuda()
# target = target.cuda()
# iter_per_print = 111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     if epoch%iter_per_print == iter_per_print -1:
#         print(epoch)
#     model.train()
#     pred = model(input)
#     if False:
#         print(pred.shape)
#         print(target.shape)
#         fds=432
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# with torch.inference_mode():
#     model.eval()
#     pred = model(input)

#     temp = pred.eq(target)
#     temp = temp.sum()
#     temp = temp.to(torch.float32)
#     acc = temp/float(batch*target.shape[1])
#     acc = acc.item()
#     print(acc, "<- the accuracy")
#     if False:
#         # print(model(int_into_floats(torch.tensor([[0]]),3)))
#         # print(model(int_into_floats(torch.tensor([[1]]),3)))
#         # print(model(int_into_floats(torch.tensor([[2]]),3)))
#         # print(model(int_into_floats(torch.tensor([[3]]),3)))
#         # print(model(int_into_floats(torch.tensor([[4]]),3)))
#         # print(model(int_into_floats(torch.tensor([[5]]),3)))
#         # print(model(int_into_floats(torch.tensor([[6]]),3)))
#         # print(model(int_into_floats(torch.tensor([[7]]),3)))
#         pass
# fds=432




'''does it fit in FULL adder. Probably YES!!!'''

'''

to do:
f16

f32, bit=1,layer=4,gates=4,batch=50,lr=0.00001 >>> 500,5k5, 23k, 1., but not fully.
f32, bit=1,layer=4,gates=6,batch=50,lr=0.00001 >>> 1k~4k, 1., but not fully.
f32, bit=1,layer=4,gates=8,batch=50,lr=0.00001 >>> 18k, 1., but not fully.
f32, bit=1,layer=4,gates=8,batch=50,lr=0.00005 >>> 500, 1k2, 8k8 1., but not fully.
redo:
f16, bit=1,layer=4,gates=4,batch=50,lr=0.00001 >>> 6k(w/o xor), 9k(with xor)
f16, bit=1,layer=4,gates=8,batch=50,lr=0.00001 >>> 0.9????????? lr is too small.
f16, bit=1,layer=4,gates=8,batch=50,lr=0.00005 >>> 500, 2k
f16, bit=2,layer=4,gates=8,batch=50,lr=0.00005 >>> 15k?????batch too small.
f16, bit=2,layer=4,gates=8,batch=200,lr=0.00005 >>> 18k (is this possible ???? or is the test too wrong????)
f16, bit=3,layer=4,gates=8,batch=500,lr=0.00005 >>> 21k
f16, bit=3,layer=4,gates=8,batch=500,lr=0.0005 >>> 2k, 3k5, 4k
f16, bit=3,layer=4,gates=8,batch=500,lr=0.001 >>> 1k, 1k5, 2k
f16, bit=3,layer=4,gates=8,batch=500,lr=0.005 >>> 0.5, pure random~ lr too big.
f16, bit=4,layer=4,gates=8,batch=3000,lr=0.00005 >>> 0.5, pure random~
f16, bit=4,layer=5,gates=8,batch=3000,lr=0.0005 >>> 0.7???????????
f16, bit=4,layer=6,gates=8,batch=3000,lr=0.0005 >>>0.7???????????
f16, bit=4,layer=6,gates=12,batch=3000,lr=0.0005 >>>0.8
f16, bit=4,layer=6,gates=12,batch=3000,lr=0.0001 >>>0.5?????? debug this.
f16, bit=4,layer=7,gates=12,batch=3000,lr=0.0005 >>>0.5??????
f16, bit=4,layer=7,gates=12,batch=3000,lr=0.002 >>>
????lr
'''


def test_config_dispatcher(bit:int, layer:int, gates:int, batch:int, lr:float):
    return(bit, layer, gates, batch, lr)
(bit, layer, gates, batch, lr) = test_config_dispatcher(bit=1,layer=4,gates=8,batch=50,lr=0.00005)
is_f16 = True
iter_per_print = 500#1111
print_count = 20000
# bit = 1
# layer = 4
# gates = 6
# batch = 5     # doesn't affect the accumulated grad.
# lr = 0.00001      #0.00001 with scaling ratio of 30/200/200
#noise_base = 1.
(input, target) = data_gen_full_adder(bit,batch, is_cuda=True)
model = DSPU(input.shape[1],target.shape[1],gates,gates,gates,layer, 
             scaling_ratio_for_gramo_in_mapper=200.,
             scaling_ratio_for_gramo_in_mapper_for_first_layer=30.,
             scaling_ratio_for_gramo_in_mapper_for_out_mapper=200.,)
model.cuda()
if is_f16:
    input = input.to(torch.float16)
    target = target.to(torch.float16)
    model.half()
    pass

# model.print_mapper_scaling_ratio_for_inner_raw_from_all()
# model.print_gates_big_number_from_all()
# fds=432


# print(model.get_mapper_scaling_ratio_for_inner_raw_from_all())
# print(model.get_gates_big_number_from_all())
#model.first_layer.set_scaling_ratio()

#model = model.to(torch.float16)
#print(next(model.parameters()).dtype)
#fds=432

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


for epoch in range(iter_per_print*print_count):
    if epoch%iter_per_print == iter_per_print -1:
        print(epoch+1, "<- epoch+1    ", end="")
        with torch.inference_mode():
            model.eval()
            pred = model(input)
            #print(pred, "should be ", target)

            temp = pred.eq(target)
            # print(temp.sum()/float(temp.shape[0]), "correct per batch?")
            # fds=432
            temp = temp.sum()
            temp = temp.to(torch.float32)
            acc = temp/float(batch*target.shape[1])
            acc = acc.item()
            print(acc, "<- accuracy")
            if 1. == acc or False:#or False.
                (finished, least_one) = model.first_layer.in_mapper.can_convert_into_eval_only_mode()
                if finished:
                    print("Training finished. Acc:100%. All mappers are trained enough.")
                    break
                else:
                    print("Acc:100%. But some mappers are not trained enough. Least strength is ", 
                          least_one.item(), " . It's less than 0.7.")
                    print("but now it's not possible to go for 0.7, so let's break the loop here.")
                    break
                    pass
                pass
            pass
        pass
    model.train()
    pred = model(input)
    if False:
        print(pred.shape)
        print(target.shape)
        fds=432
    loss = loss_function(pred, target)
    optimizer.zero_grad()
    loss.backward()
    if False and "some_noise":
        with torch.no_grad():
            for param in model.parameters():
                if not param.grad is None:
                    some_noise = torch.randn_like(param.grad)
                    some_noise = torch.pow(noise_base, some_noise)
                    #print(some_noise)
                    param.grad = param.grad*some_noise
                    pass
                pass
            pass
        pass
    if False and "compares the weight and grad":
        if epoch%iter_per_print == iter_per_print -1:
            layer = model.first_layer
            print(layer.get_mapper_raw_weight(), layer.debug_info, " raw weight")
            print(layer.get_mapper_raw_weight().grad, layer.debug_info, " grad")
            for layer in model.second_to_last_layers:
                print(layer.get_mapper_raw_weight(), layer.debug_info, " raw weight")
                print(layer.get_mapper_raw_weight().grad, layer.debug_info, " grad")
                pass
            pass
        pass
    if False and True and "weight before/after step":
        if epoch%iter_per_print == iter_per_print -1:
            layer = model.first_layer
            print(layer.get_mapper_raw_weight()[:2][:16], "1111111 first_layer raw weight")
            optimizer.step()
            print(layer.get_mapper_raw_weight()[:2][:16], "1111111 first_layer raw weight")
            layer = model.second_to_last_layers[0]
            print(layer.get_mapper_raw_weight()[0][:16], "2222222222222222 2nd_layer raw weight")
            optimizer.step()
            print(layer.get_mapper_raw_weight()[0][:16], "2222222222222222 2nd_layer raw weight")
            layer = model.second_to_last_layers[1]
            print(layer.get_mapper_raw_weight()[0][:16], "3rd_layer raw weight")
            optimizer.step()
            print(layer.get_mapper_raw_weight()[0][:16], "3rd_layer raw weight")
            layer = model.out_mapper
            print(layer.inner_raw.raw_weight[0][:16], "oooooooooooo out_mapper raw weight")
            optimizer.step()
            print(layer.inner_raw.raw_weight[0][:16], "oooooooooooo out_mapper raw weight")
            fds=432
            pass  
        pass  
    optimizer.step()


with torch.inference_mode():
    model.eval()
    pred = model(input)
    #print(pred, "should be ", target)

    temp = pred.eq(target)
    temp = temp.sum()
    temp = temp.to(torch.float32)
    acc = temp/float(batch*target.shape[1])
    acc = acc.item()
    print(acc, "<- the accuracy")

print(model.first_layer.get_mapper_raw_weight(), "layer 1 raw weight")
model.first_layer.print_after_softmax()
print(model.second_to_last_layers[0].get_mapper_raw_weight())
print(model.second_to_last_layers[1].get_mapper_raw_weight())
print(model.out_mapper.inner_raw.raw_weight, "out mapper raw weight")

fds=432










































'''
to do list
g(raw_weight)/raw_weight确认一下。
output mode
big number?
'''
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
# class ADC(torch.nn.Module):
#     r""" """
#     def __init__(self, first_big_number:float, \
#                  output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
#                 device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
        
#         if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
#             raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
#         self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
            
#         # The intermediate result will be binarized with following layers.
#         self.Binarize1 = Binarize_01__non_standard_output(1.)
#         self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
#         '''The only param is the big_number. 
#         Bigger big_number leads to more binarization and slightly bigger result range.
#         More layers leads to more binarization.
#         '''
#         # you may also needs some more Binarize layers.
#         #self.Binarize2 = Binarize_01(5.)
#         #self.Binarize3 = Binarize_01(5.)
#         pass

    
        
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         if self.training:
#             # If you know how pytorch works, you can comment this checking out.
#             if not input.requires_grad:
#                 raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
#             if len(input.shape)!=2:
#                 raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, all the inputs]")
            
#             x = input
#             # no offset is needed for OR
#             x = x-0.5
#             x = x.prod(dim=1, keepdim=True)
#             x = -x+0.5
            
#             x = self.Binarize1(x)
            
#             # you may also needs some more Binarize layers.
#             # x = self.Binarize2(x)
#             # x = self.Binarize3(x)

#             if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                 return x
#             if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                 opposite = 1.-x
#                 return torch.concat([x,opposite])
#             if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                 opposite = 1.-x
#                 return opposite
#             raise Exception("unreachable code.")

            
#         else:#eval mode
#             with torch.inference_mode():
#                 x = input.gt(0.5)
#                 x = x.to(torch.int8)
#                 #overflow doesn't affect the result. 
#                 x = x.sum(dim=1, keepdim=True)
#                 x = x%2
                
#                 if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                     x = x.to(input.dtype)
#                     return x
#                 if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                     opposite = x.logical_not()
#                     return torch.concat([x,opposite]).to(input.dtype)
#                 if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                     opposite = x.logical_not()
#                     return opposite.to(input.dtype)
#                 raise Exception("unreachable code.")
#         #end of function
        
#     pass






















'''
Progression:
Binarize layers: 6.
Basically done. Tests are dirty.

IM HERE!!!

Gate layers:
from float, you can go to: np, 01 (AND2_all_range, OR2_all_range, ???_all_range)
(But this layer doesn't stack, or doesn't work after get stacked.)
|||||||||||||||||||||||||||||||||||||
from 01: 01 (AND2_01, OR2_01, ???_01)
from np: np (AND2_np, OR2_np, ???_np)


可能需要一个计算最终二值化的极限值的辅助函数。


The following 2 layers only accept input in a very strict range.
Before using them, scale down the range of input to the corresbonding range.
ADC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01  (with AND2_01, OR2_01, ???_01)

DAC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01

'''








'''to do list:
unfinished docs
__all__
set big number 可能要允许设置小于1的数。。不然门层的可训练性可能会出问题。

数字层间的选线器

门层的第一个二值化层的big number可能要测试。
二值化层的big number和输出的关系。
'''









