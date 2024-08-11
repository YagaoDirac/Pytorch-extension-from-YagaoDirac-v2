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

from typing import Any, List, Tuple, Optional, Self
import sys
import math
import torch

#my customized.
from util import make_grad_noisy, __line__str, bitwise_acc, debug_Rank_1_parameter_to_List_float
from util import int_into_floats, data_gen_full_adder
from Gramo import GradientModification




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






class Binarize_Forward_only_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        input:torch.Tensor = args[0]
        output_is_01:torch.Tensor = args[1]#this is a bool.

        x = input

        threshold = output_is_01*0.5

        dtype = x.dtype
        x = x.gt(threshold)
        x = x.to(dtype)

        the_True = torch.tensor([True], device=output_is_01.device)

        if output_is_01.ne(the_True):
            x = x*2. - 1.
            pass

        input_needs_grad = torch.tensor([input.requires_grad])
        ctx.save_for_backward(input_needs_grad)
        return x

    @staticmethod
    def backward(ctx, g):
        input_needs_grad:torch.Tensor
        input_needs_grad,  = ctx.saved_tensors
        if input_needs_grad:
            return g, None
        else:
            return None, None

    pass  # class



# '''Does this gate layer protect the grad?'''
# input = torch.tensor([[-1., -1.],[-1., 1.],[1., 1.]], requires_grad=True)
# pred = Binarize_Forward_only_Function.apply(input, torch.tensor([False]))
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[input])
# print(input.grad)
# fds=432

# '''some basic test'''
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output_is_01 = torch.tensor([True])
# output = Binarize_Forward_only_Function.apply(input, output_is_01)
# print(output, "should be 0., 0., 0., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")
# fds=432
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output_is_01 = torch.tensor([False])
# output = Binarize_Forward_only_Function.apply(input, output_is_01)
# print(output, "should be -1 -1 +1 +1 +1")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")
# fds=432



class Binarize_Forward_only(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.

    It clamp the forward inter layer data, while doesn't touch the backward propagation.
    """
    def __init__(self, output_is_01:bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.output_is_01 = torch.nn.Parameter(torch.tensor([output_is_01]), requires_grad=False)
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return Binarize_Forward_only_Function.apply(x, self.output_is_01)


# layer = Binarize_Forward_only(True)
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "should be 0., 0., 0., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")
# fds=432
# layer = Binarize_Forward_only(False)
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "should be -1 -1, 1., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")
# fds=432






class Binarize(torch.nn.Module):
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
    
    __constants__ = ['big_number_list',
                     'input_is_01', 'output_is_01',]
    #output_when_ambiguous: float
    big_number_list: torch.nn.parameter.Parameter
    input_is_01:bool
    output_is_01:bool
    gramos:torch.nn.modules.container.ModuleList

    def __init__(self, input_is_01:bool, output_is_01:bool, layer_count:int, \
                    needs_gramo:bool, \
                    big_number_list:torch.nn.Parameter = \
                        torch.nn.Parameter(torch.tensor([5.]), requires_grad = False), \
                    device=None, dtype=None, \
                    scaling_ratio_list:List[float] = [1.], epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #safety:
        # if output_when_ambiguous<-1. or output_when_ambiguous>1. :
        #     raise Exception("Param:output_when_ambiguous is out of range. If you mean to do this, use the set_output_when_ambiguous function after init this layer.")

        if layer_count<1:
            raise Exception("Param:layer_count must be at least 1.")
        if len(big_number_list.shape)!=1:
            raise Exception("Param:big_number_list must be in rank-1 shape. Or simply use the default value.")
        if big_number_list.shape[0]<1:
            raise Exception("Param:big_number needs at least 1 element. Or simply use the default value.")
        if len(scaling_ratio_list)!=1:
            raise Exception("Param:scaling_ratio_list must be in rank-1 shape. Or simply use the default value.")
        if scaling_ratio_list[0]<1:
            raise Exception("Param:scaling_ratio_list needs at least 1 element. Or simply use the default value.")
        # if big_number_list.shape[0]!=layer_count or len(scaling_ratio_list)!=layer_count:
        #     raise Exception("Param:scaling_ratio_list and big_number_list must have the same shape, and equal to the layer_count")

        self.input_is_01 = input_is_01
        self.output_is_01 = output_is_01
        self.layer_count = layer_count

        self.big_number_list = big_number_list
        self.big_number_list.requires_grad_(False)

        # self.output_when_ambiguous = 0.
        # if output_is_01:
        #     self.output_when_ambiguous = 0.5

        # this layer also needs the info for gramo.
        self.gramos = torch.nn.ModuleList()
        if needs_gramo:
            self.gramos = torch.nn.ModuleList(
                [GradientModification(scaling_ratio_list[i],epi,div_me_when_g_too_small)
                for i in range(big_number_list.shape[0])]
            )

        self.auto_print_x_before_Binarize = False

        self.Final_Binarize = Binarize_Forward_only(output_is_01)
        # the sequence between gramo and Binarize_Forward_only doesn't matter at all.
        # gramo only affect the backward, while Binarize_Forward_only only affect the forward.
        pass

    # @staticmethod
    # def create_analog_to_01(big_number_list:List[float] = [1.5], \
    #                         device=None, dtype=None, \
    #                 scaling_ratio_list:List[float] = [1.], epi=1e-5, \
    #                    div_me_when_g_too_small = 1e-3, ):
    #     result = Binarize(False, #  input is 01
    #                   True,  # output is 01
    #                   layer_count = 1,
    #                     scaling_ratio_list = scaling_ratio_list, epi = epi,
    #                     div_me_when_g_too_small = div_me_when_g_too_small)
    #     result.set_big_number_with_float(big_number_list)
    #     return result
    @staticmethod
    def create_analog_to_np(needs_gramo: bool, big_number_list:List[float] = [1.5], \
                            device=None, dtype=None, \
                    scaling_ratio_list:List[float] = [1.], epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ):
        result = Binarize(False, #  input is 01
                      False,  # output is 01
                      layer_count = 1,
                      needs_gramo = needs_gramo,
                        scaling_ratio_list = scaling_ratio_list, epi = epi,
                        div_me_when_g_too_small = div_me_when_g_too_small)
        result.set_big_number_with_float(big_number_list)
        return result
    # @staticmethod
    # def create_01_to_01(big_number_list:List[float] = [1.5], \
    #                         device=None, dtype=None, \
    #                 scaling_ratio_list:List[float] = [1.], epi=1e-5, \
    #                    div_me_when_g_too_small = 1e-3, ):
    #     result = Binarize(True, #  input is 01
    #                   True,  # output is 01
    #                   layer_count = 1,
    #                     scaling_ratio_list = scaling_ratio_list, epi = epi,
    #                     div_me_when_g_too_small = div_me_when_g_too_small)
    #     result.set_big_number_with_float(big_number_list)
    #     return result
    # @staticmethod
    # def create_01_to_np(big_number_list:List[float] = [1.5], \
    #                         device=None, dtype=None, \
    #                 scaling_ratio_list:List[float] = [1.], epi=1e-5, \
    #                    div_me_when_g_too_small = 1e-3, ):
    #     result = Binarize(True, #  input is 01
    #                   False,  # output is 01
    #                   layer_count = 1,
    #                     scaling_ratio_list = scaling_ratio_list, epi = epi,
    #                     div_me_when_g_too_small = div_me_when_g_too_small)
    #     result.set_big_number_with_float(big_number_list)
    #     return result


    # def set_output_when_ambiguous(self, output:float):
    #     if self.output_is_01:
    #         if output<0. or output>1.:
    #             raise Exception("The output can only be between 0 and 1.")
    #         pass
    #     else:
    #         if output<-1. or output>1.:
    #             raise Exception("The output can only be between -1 and 1.")
    #         pass

    #     self.output_when_ambiguous = output
    #     pass

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
        if len(big_number_list)!=self.layer_count:
            raise Exception("I didn't test. Maybe you can modify self.layer_count first.")
            #raise Exception("Temparorily, this new list has to be the same shape as the old one. If you need to change the layer count, make a new Binarize layer.")
            pass

        for i, e in enumerate(big_number_list):
            if I_know_Im_setting_a_value_which_may_be_less_than_1:
                # This case is a bit special. I made this dedicated for the gates layers.
                if e<=0.:
                    raise Exception(f"Param:big_number the [{i}] element({e}) is not big enough.")
                    pass

            else:# The normal case
                if e<1.:
                    raise Exception(f"Param:big_number the [{i}] element({e}) is not big enough.Use I_know_Im_setting_a_value_which_may_be_less_than_1 = True if you know what you are doing.")
                    pass
        self.big_number_list:torch.nn.Parameter = torch.nn.Parameter(torch.tensor(big_number_list), requires_grad=False)
        self.big_number_list.requires_grad_(False)
        #self.update_output_range()
        pass

    def set_scaling_ratio(self, scaling_ratio_list:List[float])->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''

        if len(scaling_ratio_list)!=self.layer_count:
            raise Exception("I didn't test. Maybe you can modify self.layer_count first.")
            #raise Exception("Temparorily, this new list has to be the same shape as the old one. If you need to change the layer count, make a new Binarize layer.")

        for i in range(len(scaling_ratio_list)):
            if scaling_ratio_list[i]<=0.:
                raise Exception("Must be > 0.")
            pass

        #some old variable.
        layer:GradientModification
        if len(self.gramos)>0:
            layer = self.gramos[0]
        else:
            layer = GradientModification()
        epi = layer.epi.item()
        div_me_when_g_too_small = layer.div_me_when_g_too_small.item()

        # clear()
        for i in range(len(self.gramos)):
            self.gramos.pop(-1)

        for scaling_ratio in scaling_ratio_list:
            self.gramos.append(GradientModification(scaling_ratio,
                                                    epi,div_me_when_g_too_small))
        pass

    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return False
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    def __get_offset_according_to_input_mode_for_training_pass(self)->float:
        if self.input_is_01:
            return -0.5
        else:
            return 0.

    def __get_offset_according_to_input_mode_for_eval_pass(self)->float:
        offset = 0.
        if self.input_is_01:
            offset = -0.5
            pass
        if self.output_is_01:
            offset += 0.5
            pass
        return offset

        #untested?
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

            if self.big_number_list.requires_grad:
                raise Exception("Unreachable path.")
            if len(self.big_number_list.shape)>1:
                raise Exception("The big number list must be rank-1.")
            big_number_len = self.big_number_list.shape[0]
            if len(self.gramos)>0:
                gramos_len = len(self.gramos)
                if big_number_len!=self.layer_count or gramos_len!=self.layer_count:
                    raise Exception("They must equal.")

            x = input

            # The offset. This only works for 01 as input.
            offset = self.__get_offset_according_to_input_mode_for_training_pass()
            if 0. != offset:
                x = x + offset
                pass
            the_gramo_layer: GradientModification
            for i in range(self.layer_count-1):
                if len(self.gramos)>0:
                    the_gramo_layer = self.gramos[i]
                    x = the_gramo_layer(x)
                    pass
                big_number = self.big_number_list[i]
                if big_number!=1.:
                    x = x*big_number
                    pass
                # All the intermediate calculation use the np style.
                x = torch.tanh(x)
                pass


            if len(self.gramos)>0:
                the_gramo_layer = self.gramos[-1]
                x = the_gramo_layer(x)
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

            if self.auto_print_x_before_Binarize:
                print(x)
            x = self.Final_Binarize(x)
            return x

        else:#eval mode
            '''I know the bit wise operations are much faster.
            But for now I'll keep it f32 for simplisity.
            '''

            x = input

            # The offset. This only works for 01 as input.
            offset = self.__get_offset_according_to_input_mode_for_eval_pass()
            if 0. != offset:
                x = x + offset
                pass
            x = self.Final_Binarize(x)
            return x
        # end of function.


    def extra_repr(self) -> str:
        input_str = "[0. to 1.]"
        if not self.input_is_01:
            input_str = "[-1. to 1.] or (-inf. to +inf.)"
            pass
        output_str = "(0. to 1.)" if self.output_is_01 else "(-1. to 1.)"
        mode_str = "training" if self.training else "evaluating"

        result = f'Banarize layer, theorimatic input range:{input_str}, theorimatic output range:{output_str}, in {mode_str} mode'
        return result

    def debug_info(self, verbose:int = 0)->str:
        result = "Input is "
        input_str = "01" if self.input_is_01 else "np"
        result += input_str
        result += ", output is "
        output_str = "01" if self.output_is_01 else "np"
        result += output_str
        result += ". "
        if verbose>=1:
            result += "Big numbers are: "
            for i in range(len(self.big_number_list)):
                result += str(self.big_number_list[i].item())
                if i<len(self.big_number_list)-1:
                    result += ", "
                else:
                    result += ". "
                    pass
                pass
            pass
        l:GradientModification
        if verbose>=2:
            result += "Scaling ratios are: "
            for i, l in enumerate(self.gramos):
                result += "{:.6f}".format(l.scaling_ratio.item())
                if i<len(self.gramos)-1:
                    result += ", "
                else:
                    result += ". "
                    pass
                pass
            pass
        if verbose>=3:
            result += "Epi are: "
            for i, l in enumerate(self.gramos):
                result += "{:.6f}".format(l.epi.item())
                if i<len(self.gramos)-1:
                    result += ", "
                else:
                    result += ". "
                    pass
                pass
            result += "div_me_when_g_too_small are: "
            for i, l in enumerate(self.gramos):
                result += "{:.6f}".format(l.div_me_when_g_too_small.item())
                if i<len(self.gramos)-1:
                    result += ", "
                else:
                    result += ". "
                    pass
                pass
            pass
        return result

    pass


# '''When doesn't need gramo'''
# #input = torch.tensor([[-2.,-1.,-0.,1.,]], requires_grad=True)
# input = torch.tensor([[0.,]], requires_grad=True)
# layer = Binarize.create_analog_to_np(False, big_number_list=[1.23])
# pred = layer(input)
# g_in = torch.ones_like(pred)*100
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "input.grad WITHOUT gramo", __line__str())
# input = torch.tensor([[0.,]], requires_grad=True)
# layer = Binarize.create_analog_to_np(True, big_number_list=[1.23])
# pred = layer(input)
# g_in = torch.ones_like(pred)*100
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "input.grad WITH gramo", __line__str())
# fds=432


# '''concat passes the grad.'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# g_in_1 = torch.ones_like(input)
# torch.autograd.backward(input, g_in_1, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432

# '''Does this gate layer protect the grad?'''
# input = torch.tensor([[-1., -1.],[-1., 1.],[1., 1.]], requires_grad=True)
# layer = Binarize.create_analog_to_np()
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[input])
# print(input.grad)
# fds=432


# ''' All 4 configs.'''
# model = Binarize.create_01_to_01()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be 0 0 0 0 1")
# model = Binarize.create_01_to_01()
# model.eval()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be 0 0 0 0 1")
# model = Binarize.create_01_to_np()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be -1 -1 -1 -1 1")
# model = Binarize.create_01_to_np()
# model.eval()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be -1 -1 -1 -1 1")

# model = Binarize.create_analog_to_01()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be 0 0 1 1 1")
# model = Binarize.create_analog_to_01()
# model.eval()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be 0 0 1 1 1")
# model = Binarize.create_analog_to_np()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be -1 -1 1 1 1")
# model = Binarize.create_analog_to_np()
# model.eval()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be -1 -1 1 1 1")
# fds=432


# '''All the big numbers in the list are properly iterated.
# No print. Test with breakpoint.'''
# model = Binarize(True,True,3)
# model.set_big_number_with_float([2., 3., 4.])
# model.set_scaling_ratio([5., 6., 7.])
# input = torch.tensor([[0., 0.25, 0.5, 1.]], requires_grad=True)
# pred = model(input)
# fds=432


# '''some extra test to provide a intuition of how to control and protect the gradient.'''
# model = Binarize.create_analog_to_np()
# input = torch.tensor([[0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]], requires_grad=True)
# #input = torch.tensor([[0., 0., 0., 0.]], requires_grad=True)
# g_in = torch.ones_like(input)

# #model.layer_count = 1
# model.set_big_number_with_float([1.])
# model.set_scaling_ratio([1.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.35(sqrt(1/8)), but decreasing.")

# model.set_scaling_ratio([2.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.7, but decreasing.")

# model.set_big_number_with_float([2.])
# model.set_scaling_ratio([1.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.35(sqrt(1/8)), but decreasing MUCH MORE.")

# model.layer_count = 2
# model.set_big_number_with_float([1., 1.])
# model.set_scaling_ratio([1., 1.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.35(sqrt(1/8)), but decreasing MUCH MORE.")

# model.layer_count = 3
# model.set_big_number_with_float([1., 1., 1.])
# model.set_scaling_ratio([1., 1., 1.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.35(sqrt(1/8)), but decreasing MUCH MORE.")
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
# '''Because the forward path yields the same result as that one before, 
# the only difference is grad. The param:big_number affects the grad by a lot.'''
# class example_Binarize_analog_to_01_3times(torch.nn.Module):
#     r"""This example layer shows how to use the binarize layer.
#     """
#     def __init__(self, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         self.Binarize = Binarize.create_analog_to_01()
#         self.Binarize.set_big_number_with_float([0.75, 1., 2.], I_know_Im_setting_a_value_which_may_be_less_than_1=True)
#         pass
#     # 3 optional function.
#     # def accepts_non_standard_range(self)->bool:
#     #     return True
#     # def outputs_standard_range(self)->bool:
#     #     return True
#     # def outputs_non_standard_range(self)->bool:
#     #     return not self.outputs_standard_range()
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         x = input
#         x = self.Binarize(x)
#         return x

#     pass


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




class DigitalMapper_V1_4(torch.nn.Module):
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
                    auto_print_difference:bool = False, \
                    scaling_ratio_for_learning_gramo:Optional[float] = None, \
                    #protect_param_every____training:int = 5, \
                    #raw_weight_boundary_for_f32:float = 15., \
                        training_ghost_weight_probability = 0., \
                            eval_mode_0_is_raw__1_is_sharp = 0, \
                        #shaper_factor = 1.01035, \
                    device=None, dtype=None) -> None:   #, debug_info = ""
                        #shaper_factor = 1.0035, \

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if in_features<2:
            raise Exception("emmmm")

        self.in_features = in_features
        self.log_of_in_features = torch.nn.Parameter(torch.log(torch.tensor([in_features])), requires_grad=False)
        self.out_features = out_features
        self.sqrt_of_out_features = torch.nn.Parameter(torch.sqrt(torch.tensor([out_features])), requires_grad=False)
        self.out_iota = torch.nn.Parameter(torch.linspace(0,out_features-1, out_features, dtype=torch.int32), requires_grad=False)
          
        # self.raw_weight_boundary_for_f32 = torch.nn.Parameter(torch.tensor([raw_weight_boundary_for_f32]), requires_grad=False)
        # self.raw_weight_boundary_for_f32.requires_grad_(False)
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.__reset_parameters__the_plain_rand01_style()
        if training_ghost_weight_probability<0. or training_ghost_weight_probability>1.:
            raise Exception("Hey, it's a probability.")
        self.training_ghost_weight_probability = torch.nn.Parameter(torch.tensor([training_ghost_weight_probability]), requires_grad=False)
        self.use_non_trainable_training_pass_once = False
        #self.backup_raw_weight = torch.nn.Parameter(self.raw_weight.detach().clone(), requires_grad=False)
        #self.raw_weight_max = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.raw_weight_min = torch.nn.Parameter(torch.tensor([-30.-self.log_of_in_features]), requires_grad=False)

        # self.with_ghost__sharped_mode = False
        # self.with_ghost = False

        #self.ghost_weight_length = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        #temp_max_length = temp_the_log_of_in_features+50.# but 20 should also be enough.
        #self.ghost_weight_length_max = torch.nn.Parameter(temp_max_length, requires_grad=False)
        #self.ghost_weight_length_step = torch.nn.Parameter(temp_the_log_of_in_features*0.005, requires_grad=False)#or 0.02?
        #self.ghost_weight_length.data = self.ghost_weight_length.to(self.raw_weight.dtype)#?does this do anything?

        if scaling_ratio_for_learning_gramo is None:
            #if lr is 0.001, then, *10 to *200 are the same. Too small(<*5) leads slowing down. Too big(>*200) doesn't train, SUDDENLY!!!
            scaling_ratio_for_learning_gramo_tensor = self.log_of_in_features*self.sqrt_of_out_features*10.
            scaling_ratio_for_learning_gramo = scaling_ratio_for_learning_gramo_tensor.item()
            pass
        self.gramo_for_raw_weight = GradientModification(scaling_ratio=scaling_ratio_for_learning_gramo)
        self.set_auto_print_difference_between_epochs(auto_print_difference)

        self.out_binarize_does_NOT_need_gramo = Binarize.create_analog_to_np(needs_gramo=False)
        self.out_gramo = GradientModification()

        #to keep track of the training.
        # self.protect_param_every____training = protect_param_every____training
        # self.protect_param__training_count = 0

        #self.last_acc = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)

        #threshold to be able to convert into eval only mode.
        self.can_convert_to_eval_only_mode__the_threshold = torch.nn.Parameter(torch.tensor([0.51], **factory_kwargs), requires_grad=False)
        self.can_convert_to_eval_only_mode__the_threshold.requires_grad_(False)
        self.debug_least_strength_last_time = 0.
        self.eval_mode_0_is_raw__1_is_sharp = eval_mode_0_is_raw__1_is_sharp
        pass

    def __reset_parameters__the_plain_rand01_style(self) -> None:
        '''copied from torch.nn.Linear'''
        self.raw_weight.data = torch.rand_like(self.raw_weight)*-1.# they should be <0.
        pass
    
    # def __reset_parameters(self) -> None:
    #     '''copied from torch.nn.Linear'''
    #     torch.nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
    #     pass

    def accepts_non_standard_range(self)->bool:
        return False#emmm unfinished.
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    
    def reset_scaling_ratio_for_raw_weight(self):
        '''simply sets the inner'''
        self.gramo_for_raw_weight.set_scaling_ratio((self.log_of_in_features*self.sqrt_of_out_features).item()*10.)
        pass
    def scale_the_scaling_ratio_for_raw_weight(self, by:float):
        '''simply sets the inner'''
        self.gramo_for_raw_weight.set_scaling_ratio((self.gramo_for_raw_weight.scaling_ratio*by).item())
        pass
    def set_scaling_ratio_for_raw_weight(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.gramo_for_raw_weight.set_scaling_ratio(scaling_ratio)
        pass
    def set_can_convert_to_eval_only_mode__the_threshold(self, the_threshold:float):
        if the_threshold<=0.5:
            raise Exception("Param:the_threshold must > 0.5")
        if the_threshold>0.9:
            raise Exception("Trust me.")
        self.can_convert_to_eval_only_mode__the_threshold = torch.nn.Parameter(torch.tensor([the_threshold], requires_grad=False))
        self.can_convert_to_eval_only_mode__the_threshold.requires_grad_(False)
        pass

    # def set_shaper_factor(self, shaper_factor:float, I_know_what_Im_doing = False):
    #     if shaper_factor<=1.:
    #         raise Exception("Param:shaper_factor must > 1.")
    #     if not I_know_what_Im_doing and shaper_factor>1.2:
    #         raise Exception("I believe this is wrong. But if you know what you are doint, comment this checking out.")

    #     self.shaper_factor.data = torch.tensor([shaper_factor], requires_grad=False)
    #     self.shaper_factor.requires_grad_(False)
    #     pass

    def set_auto_print_difference_between_epochs(self, set_to:bool = True):
        with torch.no_grad():
            if not set_to:
                self.raw_weight_before = torch.nn.Parameter(torch.empty([0,], requires_grad=False))
                self.raw_weight_before.requires_grad_(False)
                # use self.raw_weight_before.nelement() == 0 to test it.
                pass
            if set_to:
                if self.raw_weight is None:
                    raise Exception("This needs self.raw_weight first. Report this bug to the author, thanks.")
                if self.raw_weight.nelement() == 0:
                    raise Exception("Unreachable code. self.raw_weight contains 0 element. It's so wrong.")
                #if not hasattr(self, "raw_weight_before") or self.raw_weight_before is None:
                if self.raw_weight_before is None:
                    self.raw_weight_before = torch.nn.Parameter(torch.empty_like(self.raw_weight), requires_grad=False)
                    self.raw_weight_before.requires_grad_(False)
                    pass
                self.raw_weight_before.data = self.raw_weight.detach().clone()
                pass
            pass


    def set_training_ghost_weight_probability(self, set_to:float):
        '''0. is probably the best.'''
        self.training_ghost_weight_probability.data = torch.tensor([set_to])
        self.training_ghost_weight_probability.to(self.raw_weight.device).to(self.raw_weight.dtype)
        self.training_ghost_weight_probability.requires_grad_(False)
        pass
        

    # old code
    # def try_increasing_ghost_weight_length(self, factor = 1.):
    #     if 0. == self.ghost_weight_length:
    #         the_log = torch.log(torch.tensor([self.in_features], dtype=torch.float32)).item()
    #         the_log10 = the_log/torch.log(torch.tensor([10.], dtype=torch.float32))
    #         log10_minus_1 = the_log10-1.5
    #         log10_minus_1 = log10_minus_1.to(self.ghost_weight_length_step.device)
    #         if log10_minus_1>self.ghost_weight_length_step:
    #             self.ghost_weight_length.data = log10_minus_1*factor
    #             self.ghost_weight_length.data = self.ghost_weight_length.to(self.raw_weight.dtype)
    #             return
    #     self.ghost_weight_length.data = self.ghost_weight_length + self.ghost_weight_length_step*factor
    #     if self.ghost_weight_length > self.ghost_weight_length_max:
    #         self.ghost_weight_length.data = self.ghost_weight_length_max
    #         pass
    #     self.ghost_weight_length.data = self.ghost_weight_length.to(self.raw_weight.dtype)
    #     pass
    # def try_decreasing_ghost_weight_length(self, factor = 1.):
    #     self.ghost_weight_length.data = self.ghost_weight_length - self.ghost_weight_length_step*factor
    #     if self.ghost_weight_length < 0.:
    #         self.ghost_weight_length.data = self.ghost_weight_length*0.
    #         pass
    #     self.ghost_weight_length.data = self.ghost_weight_length.to(self.raw_weight.dtype)
    #     pass

    @staticmethod
    def bitwise_acc(a:torch.Tensor, b:torch.Tensor, print_out_when_exact_one = True, \
                    print_out:bool = False)->float:
        with torch.no_grad():
            temp = a.eq(b)
            if temp.all() and print_out_when_exact_one:
                print(1., "(NO ROUNDING!!!)   <- the accuracy    inside bitwise_acc function __line 859 ")
                return 1.
            temp = temp.sum().to(torch.float32)
            acc = temp/float(a.shape[0]*a.shape[1])
            acc_float = acc.item()
            if print_out:
                print("{:.4f}".format(acc_float), "<- the accuracy")
                pass
            return acc_float


    def get_softmax_format(self, with_ghost_weight :Optional[bool] = None):
        raise Exception("unfinished.")
        r'''When the input param:with_ghost_weight is None, it uses self.with_ghost.'''
        if with_ghost_weight is None:
            with_ghost_weight = self.with_ghost
            pass

        # previous_with_ghost = self.with_ghost
        # previous_with_ghost_
        #raise Exception("untested.")
        with torch.no_grad():
            if with_ghost_weight:
                w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
                the_max_index = self.raw_weight.max(dim=1,keepdim=False).indices
                ghost_weight = torch.zeros_like(self.raw_weight)
                ghost_weight[self.out_iota, the_max_index] = self.ghost_weight_length
                w_with_ghost = w_after_gramo + ghost_weight
                w_after_softmax = w_with_ghost.softmax(dim=1)
                return w_after_softmax
            else:
                result = self.raw_weight.softmax(dim=1)
                return result

        #print(self.raw_weight.mul(self.update_progress_factor()).softmax(dim=1))
        #end of function.

    # duplicated.
    # def get_index_format(self)->torch.Tensor:
    #     index_of_max_o = self.raw_weight.max(dim=1).indices
    #     return index_of_max_o
    def get_one_hot_format(self)->torch.Tensor:
        with torch.no_grad():
            #raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
            out_features_s = self.raw_weight.shape[0]
            out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
            #print(out_features_iota, "out_features_iota")
            index_of_max_o = self.raw_weight.max(dim=1).indices
            #print(index_of_max_o, "index_of_max_o")

            one_hot_o_i = torch.zeros_like(self.raw_weight)
            one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
            return one_hot_o_i

    def debug_get_zero_grad_ratio(self, directly_print_out:float = False)->float:
        with torch.no_grad():
            result = 0.
            if not self.raw_weight.grad is None:
                flags = self.raw_weight.grad.eq(0.)
                total_amount = flags.sum().item()
                result = float(total_amount)/self.raw_weight.nelement()
            if directly_print_out:
                print("get_zero_grad_ratio:", result)
            return result


    def debug_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01, \
                                print_out = False)->float:
        #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
        if self.raw_weight.grad is None:
            if print_out:
                print(0., "inside debug_micro_grad_ratio function __line 1082")
                pass
            return 0.

        the_device=self.raw_weight.device
        epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
        raw_weight_abs = self.raw_weight.abs()
        flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

        epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
        raw_weight_grad_abs = self.raw_weight.grad.abs()
        flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

        ten = torch.tensor([10.], device=the_device)
        log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
        corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
        flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

        flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
        result = (flag_useful_g.sum().to(torch.float32)/self.raw_weight.nelement()).item()
        if print_out:
            print(result, "inside debug_micro_grad_ratio function __line 1082")
            pass
        return result

    def debug_print_param_overlap_ratio(self):
        with torch.no_grad():
            the_max_index = self.get_index_format()
            the_dtype = torch.int32
            if self.out_features<=1:
                print("Too few output, The overlapping ratio doesn't mean anything. __line__903")
            else:
                total_overlap_count = 0
                total_possible_count = self.in_features*(self.in_features-1)//2
                for i in range(self.in_features-1):
                    host_index = torch.tensor([i], dtype=the_dtype)
                    guest_index = torch.linspace(i+1, self.in_features-1,
                                            self.in_features-i-1, dtype=the_dtype)
                    flag_overlapped = the_max_index[guest_index].eq(the_max_index[host_index])
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("overlap_ratio:",
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.SIG_gate_count>0:
            pass
    #end of function.

    # def set_with_ghost(self, set_to:bool):
    #     self.with_ghost = set_to
    #     if (not set_to) and self.with_ghost__sharped_mode:
    #         the_info = '''The self.with_ghost_test_sharped_mode priotizes over self.with_ghost
    #         to control the sharpen behavior. Now self.with_ghost_test_sharped_mode is True,
    #         setting self.with_ghost to False doesn't change the behavior.'''
    #         print(the_info)
    #         pass
    #     pass
    # def set_with_ghost__sharp_mode(self, set_to:bool):
    #     '''Do NOT train the layer in sharp mode.'''
    #     self.with_ghost__sharped_mode = set_to
    #     pass

    def forward(self, input:torch.Tensor)->torch.Tensor:
        if len(input.shape)!=2:
            raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")

        # raw weight changes???
        #use self.raw_weight_before.nelement() == 0 to test it.
        if self.raw_weight_before.nelement() != 0:
            ne_flag = self.raw_weight_before.data.ne(self.raw_weight)
            nan_inf_flag = self.raw_weight_before.data.isnan().logical_and(self.raw_weight_before.data.isinf())
            report_these_flag = ne_flag.logical_and(nan_inf_flag.logical_not())
            if report_these_flag.any()>0:
                to_report_from = self.raw_weight_before[report_these_flag]
                to_report_from = to_report_from[:16]
                to_report_to = self.raw_weight[report_these_flag]
                to_report_to = to_report_to[:16]
                line_number_info = "    Line number: "+str(sys._getframe(1).f_lineno)
                print("Raw weight changed, from:\n", to_report_from, ">>>to>>>\n",
                        to_report_to, line_number_info)
            else:
                print("Raw weight was not changed in the last stepping")
                pass
            self.raw_weight_before.data = self.raw_weight.detach().clone()
            pass



        if self.training:
            with torch.no_grad():
                self.anti_nan_for_raw_weight()
                self.protect_raw_weight()
                
                if self.training_ghost_weight_probability!=0.:
                        
                    #debug_local_var_raw_weight = self.raw_weight.data
                        
                    the_max_o_1 = self.raw_weight.max(dim=1,keepdim=True)
                    the_max_index_o = the_max_o_1.indices.squeeze(dim=1)
                    #the_max_value_o_1 = the_max_o_1.values
                    
                    exp_of_raw_weight_o_i = torch.exp(self.raw_weight)#-the_max_value_o_1) not needed. After protection, it's <=0.
                    sum_of_exp_o = exp_of_raw_weight_o_i.sum(dim=1, keepdim=False)
                    sum_of_exp_o_1 = sum_of_exp_o.unsqueeze(dim=1)
                    the_softmax_o_i = exp_of_raw_weight_o_i/sum_of_exp_o_1

                    if torch.isnan(the_softmax_o_i).any():
                        fds = 432
                        pass
                    if torch.isinf(the_softmax_o_i).any():
                        fds = 432
                        pass

                    #ref_the_softmax = self.raw_weight.softmax(dim=1)

                    top_cooked_weight_o = the_softmax_o_i[self.out_iota,the_max_index_o]
                    flag_weight_too_soft_o:torch.Tensor = top_cooked_weight_o.lt(0.5)
                    flag_some_rand_o = torch.rand(flag_weight_too_soft_o.shape,device=flag_weight_too_soft_o.device).lt(self.training_ghost_weight_probability)
                    if self.use_non_trainable_training_pass_once:
                        self.use_non_trainable_training_pass_once = False
                        flag_some_rand_o = torch.ones(flag_some_rand_o.shape).to(torch.bool)
                        pass
                    flag_sharpen_these_o = flag_weight_too_soft_o.logical_and(flag_some_rand_o)

                    #exp_of_the_amax = torch.exp(the_max.values)

                    #top_of_exp_o = exp_of_raw_weight_o_i[self.out_iota,the_max_index_o]#always 1. opt able.
                    #top_of_exp_o_1 = top_of_exp_o.unsqueeze(dim=1)
                    sum_of_exp_with_out_top_one_o = sum_of_exp_o - 1.#top_of_exp_o always 1...
                    new_top_of_exp_o = sum_of_exp_with_out_top_one_o*1.084#1.0833333
                    #temp = flag_sharpen_these*(The_A_prime)+(flag_sharpen_these.logical_not())*top_of_exp
                    new__RAW__top_raw_weight_o = new_top_of_exp_o.log()
                    #temp = temp.log()
                    
                    if torch.isnan(new__RAW__top_raw_weight_o).any():
                        fds = 432
                        pass
                    if torch.isinf(new__RAW__top_raw_weight_o).any():
                        fds = 432
                        pass
                    
                    new__RAW__top_raw_weight_o.nan_to_num_(0.)#inf becomes super big but normal number.
                    
                    new_top_raw_weight_o = flag_sharpen_these_o*new__RAW__top_raw_weight_o
                    new_top_raw_weight_o = new_top_raw_weight_o.maximum(torch.zeros([1,], dtype= new_top_raw_weight_o.dtype, device=new_top_raw_weight_o.device ))
                    
                    The_raw_ghost_length_o = new_top_raw_weight_o-self.raw_weight[self.out_iota,the_max_index_o]

                    #The_ghost_length_o = flag_sharpen_these_o*(The_raw_ghost_length_o)
                    #The_ghost_length_o = The_ghost_length_o.nan_to_num(-4242.)
                    
                    ghost_weight_o_i = torch.zeros_like(self.raw_weight)
                    ghost_weight_o_i[self.out_iota,the_max_index_o] = The_raw_ghost_length_o
                    #ghost_weight_o_i.nan_to_num_(0.)
                    #self.raw_weight.data[self.out_iota,the_max_index] = temp
                    pass
                pass



            x = input
            x = x.unsqueeze(dim=2)

            # old code.
            #w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
            #final_raw_weight = self.get_final_raw_weight(self.training)

            w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
            if self.training_ghost_weight_probability!=0.:
                w_with_ghost = w_after_gramo+ghost_weight_o_i
                w_after_softmax = w_with_ghost.softmax(dim=1)
            else:
                w_after_softmax = w_after_gramo.softmax(dim=1)
                pass
            
            x = w_after_softmax.matmul(x)

            x = x.squeeze(dim = 2)

            if torch.isnan(x).any():
                fds = 432
                pass


            x = self.out_binarize_does_NOT_need_gramo(x)
            x = self.out_gramo(x)#here is the only gramo.
            return x
        else:#eval mode.
            if 0 == self.eval_mode_0_is_raw__1_is_sharp:
                #raw mode
                x = input
                x = x.unsqueeze(dim=2)
                w_after_softmax = self.raw_weight.softmax(dim=1)
                x = w_after_softmax.matmul(x)
                x = x.squeeze(dim = 2)
                x = self.out_binarize_does_NOT_need_gramo(x)
                return x
            if 1 == self.eval_mode_0_is_raw__1_is_sharp:
                #sharp mode
                the_max_index = self.get_max_index()
                x = input[:, the_max_index]
                return x
            raise Exception("unreachable code.")
    #end of function.

    def anti_nan_for_raw_weight(self):
        with torch.no_grad():
            flag_nan_and_neg_inf = self.raw_weight.isnan().logical_or(self.raw_weight.isneginf())
            flag_pos_inf = self.raw_weight.isposinf()
            if flag_nan_and_neg_inf.any():
                print(flag_nan_and_neg_inf.sum().item(), "  <- elements of raw_weight became nan or neg inf.  Probably the scaling_ratio of gramo is too big.  __line 1113")
                pass
            self.raw_weight.nan_to_num_(0.)
            '''self.raw_weight_min == -30.-self.log_of_in_features'''
            self.raw_weight.data = flag_nan_and_neg_inf*(self.raw_weight_min+torch.rand_like(self.raw_weight)*self.log_of_in_features)+flag_nan_and_neg_inf.logical_not()*self.raw_weight
            self.raw_weight.data = flag_pos_inf*torch.rand_like(self.raw_weight)*self.log_of_in_features*-1.+flag_pos_inf.logical_not()*self.raw_weight


            # flag_nan_after = torch.isnan(self.raw_weight)
            # if flag_nan.any():
            #     print(flag_nan_after.sum().item())
            #     pass
            # pass
        pass


    # def get_final_raw_weight(self, training:bool)->torch.Tensor:
    #     the_length = self.get_ghost_weight_length()
    #     if training:
    #         w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
    #         the_result = DigitalMapper_V1_2.apply_ghost_weight(w_after_gramo,
    #                     the_length,self.out_iota, self.get_max_index(),
    #                     training = training)
    #         return the_result
    #     else:
    #         with torch.no_grad():
    #             the_result = DigitalMapper_V1_2.apply_ghost_weight(self.raw_weight,
    #                     the_length,self.out_iota, self.get_max_index(),
    #                     training = training)
    #             return the_result
    #     #end of function.

    # def set_ghost_weight_length_with_float(self, set_to:float):
    #     self.ghost_weight_length.data = torch.tensor([set_to]).to(self.raw_weight.dtype)
    #     self.ghost_weight_length.requires_grad_(False)
    #     pass

    # def get_ghost_weight_length(self)->torch.nn.parameter.Parameter:
    #     if self.with_ghost__sharped_mode:
    #         return self.ghost_weight_length_max
    #     elif self.with_ghost:
    #         return self.ghost_weight_length
    #     else:
    #         return torch.nn.Parameter(torch.zeros([1,]), requires_grad=False)
    #     #end of function.

    def get_max_index(self)->torch.Tensor:
        with torch.no_grad():
            the_max_index = self.raw_weight.max(dim=1,keepdim=False).indices
            return the_max_index

    # @staticmethod
    # def apply_ghost_weight(the_tensor : torch.Tensor, \
    #         ghost_weight_length:torch.nn.parameter.Parameter, \
    #         out_iota: torch.tensor, the_max_index: torch.tensor, training = False)->torch.Tensor:
    #     if training:
    #         if ghost_weight_length != 0.:
    #             result = the_tensor.clone()
    #             result[out_iota, the_max_index] = result[out_iota, the_max_index]+ghost_weight_length
    #             pass
    #         else:
    #             result = the_tensor
    #             pass
    #         return result
    #     else:#eval mode.
    #         with torch.no_grad():
    #             if ghost_weight_length != 0.:
    #                 result = the_tensor.clone()
    #                 result[out_iota, the_max_index] = result[out_iota, the_max_index]+ghost_weight_length
    #                 pass
    #             else:
    #                 result = the_tensor
    #                 pass
    #             return result
    #     #end of function.

    # @staticmethod
    # def weight_after_softmax(input:torch.Tensor, training = False)->torch.Tensor:
    #     改
    #     if training:
    #         after_softmax = input.softmax(dim=1)
    #         return after_softmax
    #     else:
    #         with torch.no_grad():
    #             after_softmax = input.softmax(dim=1)
    #             return after_softmax


    # def before_test_sharped_mode(self):
    #     self.test_sharped_mode = True
    #     pass
    # def after_test_sharped_mode(self):
    #     self.test_sharped_mode = False
    #     pass

    def extra_repr(self) -> str:
        return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'

    def debug__get_strong_grad_ratio(self, log10_diff = 3., epi_for_w = 0.01, epi_for_g = 0.00001, \
                                print_out = False)->float:
        with torch.no_grad():
            #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
            if self.raw_weight.grad is None:
                if print_out:
                    print(0., "inside debug_micro_grad_ratio function __line 1082")
                    pass
                return 0.

            the_device=self.raw_weight.device
            epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
            flag_w_big_enough = self.raw_weight.gt(epi_for_w_tensor)

            epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
            flag_g_big_enough = self.raw_weight.grad.gt(epi_for_g_tensor)

            ten = torch.tensor([10.], device=the_device)
            log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
            corresponding_g = self.raw_weight.grad*torch.pow(ten, log10_diff_tensor)
            flag_w_lt_corresponding_g = self.raw_weight.lt(corresponding_g)

            flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
            result_tensor = flag_useful_g.sum().to(torch.float32)/self.raw_weight.nelement()
            result = result_tensor.item()
            if print_out:
                print(result, "inside debug_micro_grad_ratio function __line 1082")
                pass
            return result

    def can__old_func__convert_into_eval_only_mode(self, print_least_strength_now = False)->Tuple[torch.Tensor, torch.Tensor]:
        r'''This function tests if the raw_weight itself can provide a binarized behavior.
        No ghost weight or sharpenning algorithm is applied.

        output:
        >>> [0] can(True) or cannot(False)
        >>> [1] The least "max strength" of each output. Only for debug.
        '''
        with torch.no_grad():
            w_after_softmax = self.raw_weight.softmax(dim=1)
            max_of_each_output = w_after_softmax.amax(dim=1)
            least_strength = max_of_each_output.amin()
            least_strength = least_strength.to(self.raw_weight.device)
            #print(least_strength)
            result_flag = least_strength>self.can_convert_to_eval_only_mode__the_threshold
            result_flag = result_flag.to(self.raw_weight.device)

            if print_least_strength_now:
                least_strength_now = least_strength.item()
                print(least_strength_now,"least_strength_now  __line 1098")

                # old code.
                # if least_strength_now<0.52 and least_strength_now>0.2:
                #     if least_strength_now == self.debug_least_strength_last_time:
                #         print(least_strength_now,"least_strength_now")
                #         pass
                #     pass
                # self.debug_least_strength_last_time = least_strength_now
                pass

            return (result_flag, least_strength)

    # def convert_into_eval_only_mode(self)->DigitalMapper_eval_only:
    #     after_softmax = self.inner_raw.raw_weight.softmax(dim=1)
    #     flag = after_softmax.gt(0.7)#0.7 or at least 0.5.
    #     #print(flag, "flag")
    #     _check_flag = flag.any(dim=1)
    #     _check_flag_all = _check_flag.all()
    #     if not _check_flag_all.item():
    #         raise Exception("The mapper can NOT figure out a stable result. Train the model more.")

    #     argmax = flag.to(torch.int8).argmax(dim=1)
    #     #print(argmax, "argmax")
    #     result = DigitalMapper_eval_only(self.in_features, self.out_features, argmax)
    #     #print(result)
    #     return result
    #     #raise Exception("Not implemented yet. This feature only helps in deployment cases, but nobody cares about my project. It's never gonne be deployed.")

    # def before_step(self):
    #     self.backup_raw_weight.data = self.raw_weight.detach().clone()
    #     self.backup_raw_weight.data.requires_grad_(False)
    #     pass
    def after_step(self, print_out_level = 0):
        '''the algorithm:
        the top raw weight element is a, the exp(a) is A
        the exp(...) or all other sums up to B
        so:
        A == k*(A+B), where k is the softmax result of the top element.
        (I'm not going to figure k out.)

        Then, I need a new A' to make the k back to 0.52, it's like:
        A' == 0.52*(A'+B)
        B is known,
        0.48A' == 0.52B
        A' == 1.08B
        then, log(A') is what I need.

        The code is like, i manually calc ed the softmax, so I can access the intermidiate result.
        The max index of the raw weight is also of the cooked weight.
        By cooking, I mean the softmax operation.
        '''
        with torch.no_grad():
            self.anti_nan_for_raw_weight()
            self.protect_raw_weight()
            
            #debug_local_var_raw_weight = self.raw_weight.data
            
            the_max_o_1 = self.raw_weight.max(dim=1,keepdim=True)
            the_max_index_o = the_max_o_1.indices.squeeze(dim=1)
            #the_max_value_o_1 = the_max_o_1.values
            
            exp_of_raw_weight_o_i = torch.exp(self.raw_weight)#-the_max_value_o_1) no need for this, after protection, it's <=0.
            sum_of_exp_o_1 = exp_of_raw_weight_o_i.sum(dim=1, keepdim=True)
            #sum_of_exp_o_1 = sum_of_exp_o.unsqueeze(dim=1)
            the_softmax_o_i = exp_of_raw_weight_o_i/(sum_of_exp_o_1+0.00001)
            
            top_cooked_weight_o = the_softmax_o_i[self.out_iota,the_max_index_o]
            flag_weight_too_hard_o:torch.Tensor = top_cooked_weight_o.gt(0.6)


            #top_of_exp_o_1 = exp_of_raw_weight_o_i[self.out_iota,the_max_index_o] #in safe softmax, it's always 1.
            #top_of_exp_o_1 = top_of_exp_o_1.unsqueeze(dim=1)
            sum_of_exp_o = sum_of_exp_o_1.squeeze(dim = 1)
            sum_of_exp_with_out_top_one_o = sum_of_exp_o - 1.#top_of_exp_o_1 it's always 1.
            new_top_of_exp_o = sum_of_exp_with_out_top_one_o*1.084#0.52/0.48 is around 1.084
            
            temp_new_top_o = new_top_of_exp_o.log()
            flag_pos_inf_after_log = temp_new_top_o.isposinf()
            temp_new_top_o.nan_to_num_(0.)
            flag_useful = flag_weight_too_hard_o.logical_and(flag_pos_inf_after_log.logical_not())
            new_top_o = flag_useful*temp_new_top_o
            self.raw_weight.data[self.out_iota,the_max_index_o] = new_top_o
            
            #debug_new_softmax =  self.raw_weight.data.softmax(dim=1)
            pass
            # if print_out_level>0:
            #     raise Exception("unfinished.")

            # old code.
            #then step2.
            # param protection!!!
            # if self.protect_param__training_count<self.protect_param_every____training:
            #     self.protect_param__training_count+=1
            # else:
            #     self.protect_param__training_count = 1
            #     if print_out_level>1:
            #         raise Exception("unfinished.")
            #     self.protect_raw_weight()
            #     pass
            # pass
        pass
    #end of function

    def protect_raw_weight(self, anti_nan = True):
        '''Moves everything between 0 and ~-30
        
        Call this after anti_nan()'''
        with torch.no_grad():
            #step 0, anti nan.
            if anti_nan:
                if torch.isnan(self.raw_weight).any():
                    fds=432
                    pass
                self.anti_nan_for_raw_weight()
                # flag_is_nan = torch.isnan(self.raw_weight)
                # self.raw_weight[flag_is_nan] = torch.rand_like(self.raw_weight[flag_is_nan])*self.raw_weight_min.abs()*0.1 + self.raw_weight_min
                if torch.isnan(self.raw_weight).any():
                    fds=432
                    pass
                pass

            #step 1, move to prevent too big element.
            #a = self.raw_weight.max(dim=1,keepdim=True).values
            the_device = self.raw_weight.device
            move_left = self.raw_weight.max(dim=1,keepdim=True).values#-self.raw_weight_max this is 0.
            #move_left = move_left.maximum(torch.tensor([0.], device=the_device))# don't do this.
            move_left = move_left.to(self.raw_weight.device).to(self.raw_weight.dtype)
            self.raw_weight.data = self.raw_weight-move_left

            #step 2, cap to prevent too small element.
            flag_too_small = self.raw_weight.lt(self.raw_weight_min)
            temp = flag_too_small*self.raw_weight_min + flag_too_small.logical_not() *self.raw_weight.data
            temp = temp.to(self.raw_weight.device).to(self.raw_weight.dtype)
            self.raw_weight.data = temp

            return
    # end of function.

    pass
fast_traval____end_of_digital_mapper_layer_class = 432


# # '''the protection in forward pass!!!!!'''
# layer = DigitalMapper_V1_4(3,2, training_ghost_weight_probability= 1.)
# layer.raw_weight.data = torch.tensor([[-11, -11.,0.],[-0.1,0.,-0.1,],])
# input = torch.tensor([[1.,2,3.]])
# layer(input)
# # layer.anti_nan_for_raw_weight()
# # print(layer.raw_weight)
# fds=432



# '''anti_nan_for_raw_weight'''
# layer = DigitalMapper_V1_4(6,1)
# layer.raw_weight.data = torch.tensor([[torch.nan, torch.inf, torch.inf*-1, 0.,1, -1.]])
# layer.anti_nan_for_raw_weight()
# print(layer.raw_weight)
# fds=432


# '''scaling ratio test'''
# layer = DigitalMapper_V1_4(2,1)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer = DigitalMapper_V1_4(20,1)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer = DigitalMapper_V1_4(200,1)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer = DigitalMapper_V1_4(2,10)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer = DigitalMapper_V1_4(2,100)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer = DigitalMapper_V1_4(20,10)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer = DigitalMapper_V1_4(200,100)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer = DigitalMapper_V1_4(2,1)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer.scale_the_scaling_ratio_for_raw_weight(2.)
# print(layer.gramo_for_raw_weight.scaling_ratio)
# layer.reset_scaling_ratio_for_raw_weight()
# print(layer.gramo_for_raw_weight.scaling_ratio)
# fds=432


# '''can convert to eval only mode'''
# layer = DigitalMapper_V1_4(3,2)
# layer.raw_weight.data = torch.tensor([[0.5, 0.,0.,], [1., 0.,0.,],])
# print(layer.can_convert_into_eval_only_mode())
# layer.raw_weight.data = torch.tensor([[0.,1.5, 0.,], [0.,1., 0.,],])
# print(layer.can_convert_into_eval_only_mode())
# fds=432


# '''the core protection of v1_3. What would it be in v1_4?'''
# layer = DigitalMapper_V1_4(4,2)#, protect_param_every____training=0)
# layer.raw_weight.data = torch.tensor([[0., 0.,1.,], [0.,2., 0.,],])
# print(layer.raw_weight.data)
# layer.after_step()
# print(layer.raw_weight.data)
# print(layer.raw_weight.data.softmax(dim=1))
# print(layer.can__old_func__convert_into_eval_only_mode(), "should be around 0.52")
# fds=432


# '''micro grad test'''
# layer = DigitalMapper_V1_3(4,1, every?)
# layer.raw_weight.data = torch.tensor([[0.1,0.001,0.1     ,1.1,],])
# layer.raw_weight.grad = torch.tensor([[0.1,0.1,  0.000001,0.001,],])
# layer.debug__get_strong_grad_ratio(print_out=True)
# print("should be 0.25")
# fds=432


# '''param protection'''
# layer = DigitalMapper_V1_4(4,1)
# layer.raw_weight.data = torch.tensor([[100.,0.,-300, torch.nan]])
# layer.protect_raw_weight()
# print(layer.raw_weight)
# fds=432


# '''repeat this one to get different pred.'''
# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_4(3,1)
# #layer.use_non_trainable_training_pass_once = True
# layer.raw_weight.data = torch.tensor([[0.1,0,0]])
# #print(layer.raw_weight.shape)
# pred = layer(input)
# #print(pred.shape)
# print(pred, "pred")
# fds=432


# '''basic 2 pass test.'''
# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_4(3,1)
# layer.use_non_trainable_training_pass_once = True
# layer.raw_weight.data = torch.tensor([[0.1,0,0]])
# #print(layer.raw_weight.shape)
# pred = layer(input)
# #print(pred.shape)
# print(pred, "pred")
# g_in = torch.ones_like(pred)*10#0000 dosen't change any grad.
# torch.autograd.backward(pred, g_in, inputs=[input,layer.raw_weight])
# print(input.grad, "input.grad")
# print(layer.raw_weight.grad, "layer.raw_weight.grad")

# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_4(3,1)
# layer.use_non_trainable_training_pass_once = True
# layer.raw_weight.data = torch.tensor([[0.2,0,0]])
# pred = layer(input)
# #print(pred.shape)
# print(pred, "pred")
# g_in = torch.ones_like(pred)*10#0000 dosen't change any grad.
# torch.autograd.backward(pred, g_in, inputs=[input,layer.raw_weight])
# print(input.grad, "input.grad")
# print(layer.raw_weight.grad, "layer.raw_weight.grad")

# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_4(3,1)
# layer.use_non_trainable_training_pass_once = True
# layer.raw_weight.data = torch.tensor([[10.,0,0]])
# pred = layer(input)
# #print(pred.shape)
# print(pred, "pred")
# g_in = torch.ones_like(pred)*10#0000 dosen't change any grad.
# torch.autograd.backward(pred, g_in, inputs=[input,layer.raw_weight])
# print(input.grad, "input.grad")
# print(layer.raw_weight.grad, "layer.raw_weight.grad")


# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_4(3,1)
# layer.use_non_trainable_training_pass_once = True
# layer.raw_weight.data = torch.tensor([[10.,0,0]])
# layer.after_step()#the only extra thing comparing with the previous one.
# print(layer.raw_weight)
# pred = layer(input)
# #print(pred.shape)
# print(pred, "pred")
# g_in = torch.ones_like(pred)*10#0000 dosen't change any grad.
# torch.autograd.backward(pred, g_in, inputs=[input,layer.raw_weight])
# print(input.grad, "input.grad")
# print(layer.raw_weight.grad, "layer.raw_weight.grad")

# print("summarize. The first 2 got the same grad for input. The third is to sharp, it's really untrainable.")

# print("Then, eval mode.")
# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_4(3,1)
# layer.eval()
# layer.eval_mode_0_is_raw__1_is_sharp = 0
# layer.raw_weight.data = torch.tensor([[0.1,0,0]])
# pred = layer(input)
# print(pred, "pred")

# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_4(3,1)
# layer.eval()
# layer.eval_mode_0_is_raw__1_is_sharp = 1
# layer.raw_weight.data = torch.tensor([[0.1,0,0]])
# pred = layer(input)
# print(pred, "pred")
# fds=432


# '''can_convert_into_eval_only_mode function shows if the difference between
# param is already big enough, and if it's safe to convert into pure binary mode.'''
# layer = DigitalMapper_V1_4(2,3)
# layer.set_can_convert_to_eval_only_mode__the_threshold(0.7)
# layer.raw_weight.data = torch.tensor([[0.,0.8],[0.,5.],[0.,5.],])
# print(layer.can_convert_into_eval_only_mode(), "can_convert_into_eval_only_mode()")
# print(torch.softmax(torch.tensor([[0.,0.8]]),dim=1)[0,1])
# layer = DigitalMapper_V1_4(2,3)
# layer.set_can_convert_to_eval_only_mode__the_threshold(0.7)
# layer.raw_weight.data = torch.tensor([[0.,0.85],[0.,5.],[0.,5.],])
# print(layer.can_convert_into_eval_only_mode(), "can_convert_into_eval_only_mode()")
# print(torch.softmax(torch.tensor([[0.,0.85]]),dim=1)[0,1])
# fds=432



fast_traval____single_layer_training_test = 432
# '''some real training'''
# batch = 50_000
# n_in = 2566
# n_out = 1231
# (input, target) = data_gen_for_digital_mapper_directly_test(batch,n_in,n_out)
# input.requires_grad_()
# #input = torch.Tensor([[1., 1.],[1., -1.],[-1., 1.],[-1., -1.],])
# #target = torch.Tensor([[1.],[1.],[-1.],[-1.],])
# # print(input)
# # print(target)

# model = DigitalMapper_V1_4(input.shape[1],target.shape[1])#,protect_param_every____training=20)
# #model.scale_the_scaling_ratio_for_raw_weight(0.5)#0.1(slower),0.2(slow)//0.5,1,2,5ok//7is bad.
# #model.set_training_ghost_weight_probability(0.)#0(1,1,1)/0.01(3,2,2)/0.05(3,3,3)/0.5(10,11)/0.95(100)
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# if False:
#     for name, p in zip(model._parameters, model.parameters()):
#         print(name, p)

# model.half().cuda()
# input = input.to(torch.float16).cuda()
# target = target.to(torch.float16).cuda()

# iter_per_print = 1#1111
# print_count = 333
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#         print(pred.shape, "pred.shape")
#         print(target.shape, "target.shape")
#         fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred[:5], "pred")
#             print(target[:5], "target")
#             pass
#         pass
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if False and "make_grad_noisy":
#         make_grad_noisy(model, 1.5)
#         pass
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.raw_weight.grad[:2,:7], "grad")
#             pass
#         pass
#     if False and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             layer = model
#             print(layer.raw_weight[:2,:7], "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight[:2,:7], "first_layer.in_mapper   after update")
#             layer.after_step()############################
#             print(layer.raw_weight[:2,:7], "first_layer.in_mapper   after THE PROTECTION")
#             pass
#         pass
#     if False and "print zero grad ratio":
#         if epoch%iter_per_print == iter_per_print-1:
#             result = model.debug_get_zero_grad_ratio()
#             print("print zero grad ratio: ", result)
#             pass
#         pass
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.after_step()############################
#     if False and "print param overlap":
#         every = 1
#         if epoch%every == every-1:
#             model.print_param_overlap_ratio()
#             pass
#         pass
#     if epoch%iter_per_print == iter_per_print-1:
#         with torch.inference_mode():
#             #this raw_mode_acc is not important. Only for a ref.
#             model.eval_mode_0_is_raw__1_is_sharp = 0
#             raw_mode_pred = model(input)
#             raw_mode_acc = DigitalMapper_V1_4.bitwise_acc(pred, target)

#             model.eval()
#             model.eval_mode_0_is_raw__1_is_sharp = 1
#             sharp_mode_pred = model(input)
#             #print(pred, "pred", __line__str())
#             #print(target, "target")
#             sharp_mode_acc = DigitalMapper_V1_4.bitwise_acc(sharp_mode_pred, target)

#             print(epoch+1, "    ep/   raw/sharp mode acc    ",f"{raw_mode_acc:.3f}"," / ", f"{sharp_mode_acc:.3f}")
#             if 1. == sharp_mode_acc:#FINISHED
#                 print(sharp_mode_pred[:2,:7], "pred", __line__str())
#                 print(target[:2,:7], "target")
#                 print(model.can__old_func__convert_into_eval_only_mode(), "THIS ONE IS NOT IMPORTANT.model.can_convert_into_eval_only_mode")
#                 print(epoch+1, "Training finished    __line  1256")
#                 print(epoch+1, "Training finished    __line  1256")
#                 print(epoch+1, "Training finished    __line  1256")
#                 break
#             pass
#         pass

#     pass# the training loop.
# fds=432


#继续，重新跑一遍。
# 想清楚结束条件。
# 想清楚怎么设置那个类似DO的。



class dry_stack_test_for_digital_mapper_v1_4(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_width:int, num_layers:int, \
                    training_ghost_weight_probability = 0.65, \
                        scaling_ratio_scaled_by = 1., \
                    # auto_print_difference:bool = False, \
                    # scaling_ratio_for_learning_gramo:float = 100., \
                    #protect_param_every____training:int = 1, \
                    # raw_weight_boundary_for_f32:float = 15., \
                    #     shaper_factor = 1.0035, \
                    device=None, dtype=None) -> None:   #, debug_info = ""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if num_layers<2:
            raise Exception("emmmmmmmmmmmm....")
        if mid_width<out_features*1.1:
            raise Exception("emmmmmmmmmmmm....")
        if mid_width<out_features:
            raise Exception("Prepare for your 0.50x acc.")
        
        
        
        '''according to test. training_ghost_weight_probability from 0.5 to 0.8 all works. 
        But for only 1 layer, 0 works much better. It's true hyperparam.'''

        self.first_layer = DigitalMapper_V1_4(in_features,mid_width,training_ghost_weight_probability=training_ghost_weight_probability)
        self.mid_layers = torch.nn.ModuleList([
            DigitalMapper_V1_4(mid_width,mid_width,training_ghost_weight_probability=training_ghost_weight_probability)
            for _ in range(num_layers-2)
        ])
        self.last_layer = DigitalMapper_V1_4(mid_width,out_features,training_ghost_weight_probability=training_ghost_weight_probability)

        self.all_layers:List[DigitalMapper_V1_4] = []
        self.all_layers.append(self.first_layer)
        layer:DigitalMapper_V1_4
        for layer in self.mid_layers:
            self.all_layers.append(layer)
            pass
        self.all_layers.append(self.last_layer)

        for layer in self.all_layers:
            layer.scale_the_scaling_ratio_for_raw_weight(scaling_ratio_scaled_by)
            pass

        #self.sharpen_ratio = 0.
        pass
    #end of function
    def forward(self, input:torch.Tensor)->torch.Tensor:
        x = input
        x = self.first_layer(x)
        for layer in self.mid_layers:
            x = layer(x)
            pass
        x = self.last_layer(x)
        return x

    def after_step(self):
        layer:DigitalMapper_V1_4
        for layer in self.all_layers:
            layer.after_step()
            pass
        pass

    # @staticmethod
    # def get_rand_scalar()->float:
    #     res = torch.rand([1,]).item()
    #     return res
    # @staticmethod
    # def get_rand_bool(p:float)->bool:
    #     r = torch.rand([1,]).item()
    #     return r<p

    # def set_acc(self, acc:float):
    #     something = torch.tensor([0.],device=self.first_layer.raw_weight.device)
    #     if acc>0.5:
    #         temp = (acc-0.5)*2.
    #         something = torch.tensor([temp],device=self.first_layer.raw_weight.device)
    #         pass

    #     r = torch.rand([1,],device=self.first_layer.raw_weight.device)
    #     layer:DigitalMapper_V1_4

    #     drag_factor = torch.tensor([0.995],device=self.first_layer.raw_weight.device)
    #     if r<something:
    #         self.sharpen_ratio = drag_factor*self.sharpen_ratio+(1.-drag_factor)*something
    #         self.sharpen_ratio = self.sharpen_ratio+0.01
    #         if self.sharpen_ratio>0.4:#0.75:
    #             self.sharpen_ratio = torch.tensor([0.4],device=self.first_layer.raw_weight.device)
    #             pass
    #         pass
    #     else:
    #         self.sharpen_ratio = drag_factor*self.sharpen_ratio+(1.-drag_factor)*something
    #         self.sharpen_ratio = self.sharpen_ratio*0.95-0.01
    #         if self.sharpen_ratio<0.0:#0.
    #             self.sharpen_ratio= torch.tensor([0.0],device=self.first_layer.raw_weight.device)
    #             pass
    #         pass


    #     something2 = torch.tensor([0.],device=self.first_layer.raw_weight.device)
    #     if acc>0.5:
    #         temp = (acc-0.5)*2.
    #         temp = temp*temp
    #         something2 = torch.tensor([temp],device=self.first_layer.raw_weight.device)
    #         pass
    #     for layer in self.all_layers:
    #         rr = torch.rand([1,],device=self.first_layer.raw_weight.device)
    #         if rr<something2:
    #             layer.try_increasing_ghost_weight_length()
    #         else:
    #             layer.try_decreasing_ghost_weight_length(5.)
    #             pass
    #         pass

    #     self.re_rand_with_ghost()
    #     pass
    # #end of function.

    # def re_rand_with_ghost(self):
    #     for layer in self.all_layers:
    #         rand_bool = dry_stack_test_for_digital_mapper_v1_4.get_rand_bool(self.sharpen_ratio)
    #         layer.set_with_ghost(rand_bool)
    #         pass


        # self.first_layer.set_with_ghost(dry_stack_test_for_digital_mapper_v1_2.get_rand_bool(self.sharpen_ratio))
        # for layer in self.mid_layers:
        #     layer.set_with_ghost(dry_stack_test_for_digital_mapper_v1_2.get_rand_bool(self.sharpen_ratio))
        #     pass
        # self.last_layer.set_with_ghost(dry_stack_test_for_digital_mapper_v1_2.get_rand_bool(self.sharpen_ratio))

        # old code.
        # layer:DigitalMapper_V1_1
        # some_threshold = 0.52
        # layer = self.first_layer
        # if acc!=1. or layer.can_convert_into_eval_only_mode()[1]<some_threshold:
        # #if not layer.can_convert_into_eval_only_mode()[0]:
        #     layer.set_acc(acc)
        #     pass
        # for layer in self.mid_layers:
        #     if acc!=1. or layer.can_convert_into_eval_only_mode()[1]<some_threshold:
        #     #if not layer.can_convert_into_eval_only_mode()[0]:
        #         layer.set_acc(acc)
        #         pass
        #     pass
        # layer = self.last_layer
        # if acc!=1. or layer.can_convert_into_eval_only_mode()[1]<some_threshold:
        # #if not layer.can_convert_into_eval_only_mode()[0]:
        #     layer.set_acc(acc)
        #     pass
        # pass
    
    
    def reset_scaling_ratio_for_raw_weight(self):
        '''simply sets the inner'''
        layer:DigitalMapper_V1_4
        for layer in self.all_layers:
            layer.reset_scaling_ratio_for_raw_weight()
            pass
        pass
    def scale_the_scaling_ratio_for_raw_weight(self, by:float):
        '''simply sets the inner'''
        layer:DigitalMapper_V1_4
        for layer in self.all_layers:
            layer.scale_the_scaling_ratio_for_raw_weight(by)
            pass
        pass

    def print_zero_grad_ratio(self):
        result = self.first_layer.debug_get_zero_grad_ratio()
        print("First layer: zero grad ratio: ", result)
        layer:DigitalMapper_V1_4
        for i, layer in enumerate(self.mid_layers):
            result = layer.debug_get_zero_grad_ratio()
            print(f"{i+2}th layer: zero grad ratio: ", result)
            pass
        result = self.last_layer.debug_get_zero_grad_ratio()
        print("Last layer: zero grad ratio: ", result)
        pass

    def print_strong_grad_ratio(self, log10_diff = 0., epi_for_w = 0.01, epi_for_g = 0.01,):
        result = self.first_layer.debug_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g)
        print("First layer: strong grad ratio: ", "{:.3f}".format(result))
        layer:DigitalMapper_V1_4
        for i, layer in enumerate(self.mid_layers):
            result = layer.debug_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g)
            print(f"{i+2}th layer: strong grad ratio: ", "{:.3f}".format(result))
            pass
        result = self.last_layer.debug_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g)
        print("Last layer: strong grad ratio: ", "{:.3f}".format(result))
        pass

    def debug__old_func__can_convert_into_eval_only_mode(self, epoch:int, print_result = False)->Tuple[torch.Tensor, torch.Tensor]:
        temp_list:List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.all_layers:
            temp_list.append(layer.can__old_func__convert_into_eval_only_mode())
            pass

        # temp_list.append(self.first_layer.can_convert_into_eval_only_mode())
        # layer:DigitalMapper_V1_2
        # for layer in self.mid_layers:
        #     temp_list.append(layer.can_convert_into_eval_only_mode())
        #     pass
        # temp_list.append(self.last_layer.can_convert_into_eval_only_mode())

        if print_result:
            for obj in temp_list:
                print(f"{obj[1].item():.3f}", end=",,,")
                pass
            print("    ", epoch, "    from dry stack test can_convert_into_eval_only_mode function.")
            pass

        the_flag = torch.tensor([True], device=self.first_layer.raw_weight.device)
        the_number = torch.tensor([1.], device=self.first_layer.raw_weight.device)
        for temp in temp_list:
            the_flag = the_flag.logical_and(temp[0])
            the_number = the_number.minimum(temp[1])
            pass
        return (the_flag, the_number)
    def print_scaling_ratio_for_raw_weight(self):
        for i, layer in enumerate(self.all_layers):
            temp = layer.gramo_for_raw_weight.scaling_ratio
            print("Scaling ratio of the ",i,"th layer: ", "{:.3e}".format(temp.item()), "      __line 1873")
            pass
        pass
        
        
    # def before_step(self):
    #     self.first_layer.before_step()
    #     for layer in self.mid_layers:
    #         layer.before_step()
    #         pass
    #     self.last_layer.before_step()
    #     pass

    # def before_test_sharped_mode(self):
    #     for layer in self.all_layers:
    #         layer.set_with_ghost__sharp_mode(True)
    #         pass

        # self.first_layer.set_with_ghost(True)
        # for layer in self.mid_layers:
        #     layer.set_with_ghost(True)
        #     pass
        # self.last_layer.set_with_ghost(True)
        pass



    # def after_test_sharped_mode(self):
    #     for layer in self.all_layers:
    #         layer.set_with_ghost__sharp_mode(False)
    #         pass
    #     #self.re_rand_with_ghost()
    #     pass

    # def print_ghost_length(self):
    #     print("ghost_length:    ", end="")
    #     temp_list:List[float] = []
    #     for layer in self.all_layers:
    #         temp_list.append(layer.ghost_weight_length)
    #         pass

        # temp_list.append(self.first_layer.ghost_weight_length)
        # for layer in self.mid_layers:
        #     temp_list.append(layer.ghost_weight_length)
        #     pass
        # temp_list.append(self.last_layer.ghost_weight_length)
        # for temp in temp_list:
        #     print(f"{temp.item():.3f}", end=", ")
        #     pass
        # print("   __line 1566")
        # pass

    def set_eval_mode(self, eval_mode_0_is_raw__1_is_sharp):
        for layer in self.all_layers:
            layer.eval_mode_0_is_raw__1_is_sharp = eval_mode_0_is_raw__1_is_sharp
            pass
        pass

    def print__old_func__can_convert_into_eval_only_mode(self):
        self.first_layer.can__old_func__convert_into_eval_only_mode(True)
        for layer in self.all_layers:
            layer.can__old_func__convert_into_eval_only_mode(True)
            pass
        self.last_layer.can__old_func__convert_into_eval_only_mode(True)
        pass

    pass

fast_traval____dry_stack_test = 432
# batch = 50_000
# n_in = 20
# n_out = 10
# mid_width = 40
# num_layers = 20
# ghost_weight_p = 0.6##
# scaling_ratio_scaled_by = 0.5#1.
# iter_per_print = 50#1111
# print_count = 333333
# # start_scaling_mul = 3.
# # start_scaling_epoch = 100

# (input, target) = data_gen_for_digital_mapper_directly_test(batch,n_in,n_out)
# input.requires_grad_()
# #input = torch.Tensor([[1., 1.],[1., -1.],[-1., 1.],[-1., -1.],])
# #target = torch.Tensor([[1.],[1.],[-1.],[-1.],])
# # print(input, "input")
# # print(target, "target")

# model = dry_stack_test_for_digital_mapper_v1_4(input.shape[1],target.shape[1],mid_width,num_layers=num_layers, training_ghost_weight_probability=ghost_weight_p, scaling_ratio_scaled_by = scaling_ratio_scaled_by)
# #model.scale_the_scaling_ratio_for_raw_weight(3.)
# model.print_scaling_ratio_for_raw_weight()
# #model.first_layer.set_scaling_ratio_for_raw_weight(model.first_layer.gramo_for_raw_weight.scaling_ratio*0.5)
# #model.last_layer.set_scaling_ratio_for_raw_weight(model.last_layer.gramo_for_raw_weight.scaling_ratio*0.5)
# #model.mid_layers[0].set_scaling_ratio_for_raw_weight(model.first_layer.gramo_for_raw_weight.scaling_ratio*100.)

# #loss_function = torch.nn.MSELoss()#
# loss_function = torch.nn.L1Loss()#
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# if True and "print parameters":
#     if True:# and "only the training params":
#         for name, p in zip(model._parameters, model.parameters()):
#             if p.requires_grad:
#                 print(name, p)
#                 pass
#             pass
#         pass
#     else:# prints all the params.
#         for name, p in zip(model._parameters, model.parameters()):
#             print(name, p)
#             pass
#         pass

# if True and "f16 & GPU":
#     model.half().cuda()
#     input = input.to(torch.float16).cuda()
#     target = target.to(torch.float16).cuda()
# else:
#     model.cuda()
#     input = input.cuda()
#     target = target.cuda()

# flag = False
# previous_sharp_acc = 0.
# previous_raw_acc = 0.

# for epoch in range(iter_per_print*print_count):
#     if 100 == epoch:
#         model.reset_scaling_ratio_for_raw_weight()
#         pass
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#         print(pred.shape, "pred.shape")
#         print(target.shape, "target.shape")
#         fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred[:2,:5], "pred")
#             print(target[:2,:5], "target")
#             pass
#         pass
#     loss:torch.Tensor = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #if epoch>19:
#     #print(model.first_layer.raw_weight_boundary_for_f32.item())
#     #print(model.first_layer.raw_weight_boundary_for_f32.requires_grad)
#     if False and "make_grad_noisy":
#         make_grad_noisy(model, 1.05)
#         pass
#     #print(model.first_layer.raw_weight_boundary_for_f32.item())
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.first_layer.raw_weight.grad[:3,:7], "grad")
#             print(model.last_layer.raw_weight.grad[:3,:7], "grad")
#             pass
#         pass
#     if False and "print the weight":
#         every = 1
#         if epoch%every == every-1:
#         #if epoch%iter_per_print == iter_per_print-1:
#             layer = model.first_layer
#             # print(layer.raw_weight[:2,:7], "first_layer.in_mapper   before update")
#             # optimizer.step()
#             # print(layer.raw_weight[:2,:7], "first_layer.in_mapper   after update")
#             # layer.after_step()
#             # print(layer.raw_weight[:2,:7], "first_layer.in_mapper   after after step")

#             if torch.isnan(layer.raw_weight).any():
#                 fds=432
#                 pass

#             layer = model.last_layer
#             # print(layer.raw_weight[:2,:7], "last_layer.in_mapper   before update")
#             # optimizer.step()
#             # print(layer.raw_weight[:2,:7], "last_layer.in_mapper   after update")
#             # layer.after_step()
#             # print(layer.raw_weight[:2,:7], "last_layer.in_mapper   after after step")
#             if torch.isnan(layer.raw_weight).any():
#                 fds=432
#                 pass

#             pass
#         pass
#     if False and "print strong grad ratio":#############################
#         if epoch%iter_per_print == iter_per_print-1:
#             model.print_strong_grad_ratio()
#             # 看这里
#             # here is the reason this v1_2 failed.
#             # It still push the softmax too hard, which prevents backward pass.
#             pass
#         pass
#     #optimizer.param_groups[0]["lr"] = 0.01
#     #print(model.first_layer.raw_weight.requires_grad,"    __line 1720")
# #    model.before_step()
#     optimizer.step()
#     model.after_step()
#     if False and "print param overlap":
#         # every = 1
#         # if epoch%every == every-1:
#         if epoch%iter_per_print == iter_per_print-1:

#             model.print_param_overlap_ratio()
#             pass
#         pass
    

#     with torch.inference_mode():

#         #every = 10
#         #if epoch%every == every-1:
#         model.eval()
#         model.set_eval_mode(1)
#         sharp_mode_pred = model(input)
#         #print(pred, "pred", __line__str())
#         #print(target, "target")
#         sharp_mode_acc = DigitalMapper_V1_4.bitwise_acc(sharp_mode_pred, target)
#         # if epoch>500 :
#         #     every = 10
#         #     if epoch%every == every-1:
#         #         if sharp_mode_acc == previous_sharp_acc:
#         #             flag = True
#         #             pass
#         #         previous_sharp_acc = sharp_mode_acc
#         #         previous_raw_acc = raw_mode_acc
#         #         pass
#         #     pass

#         #every = 100
#         if epoch%iter_per_print == iter_per_print-1:
#             model.eval()
#             model.set_eval_mode(0)
#             raw_mode_pred = model(input)
#             raw_mode_acc = DigitalMapper_V1_4.bitwise_acc(raw_mode_pred, target)
#             print(epoch+1, "    ep/   raw/sharp mode acc    ",f"{raw_mode_acc:.3f}"," / ", f"{sharp_mode_acc:.3f}")
#             pass
#         if 1. == sharp_mode_acc:#FINISHED
#             print(sharp_mode_pred[:2,:7], "pred", __line__str())
#             print(target[:2,:7], "target")
#             print(model.print__old_func__can_convert_into_eval_only_mode(), "THIS ONE IS NOT IMPORTANT.model.can_convert_into_eval_only_mode")
#             print(epoch+1, "Training finished    __line  1256")
#             print(epoch+1, "Training finished    __line  1256")
#             print(epoch+1, "Training finished    __line  1256")
#             break
#         pass
#     pass# the training loop.
# fds=432



class single_input_gate_np(torch.nn.Module):
    r""" 
    unfinished docs
    """
                 
                 
    def __init__(self, output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        pass
   
    # def accepts_non_standard_range(self)->bool:
    #     return False
    # def outputs_standard_range(self)->bool:
    #     return True
    # def outputs_non_standard_range(self)->bool:
    #     return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            
            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x
                else:
                    opposite = x.neg()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'Single input gate(SAME/NOT) layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass

# '''basic test'''
# input = torch.tensor([[-2., -1, 0, 1,2]], requires_grad=True)
# layer = single_input_gate_np(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# output = layer(input)
# print(output, "output", __line__str())
# g_in = torch.ones_like(output)
# g_in[0,1] = 0.
# torch.autograd.backward(output, g_in, inputs=input)
# print(input.grad, "input.grad")

# input = torch.tensor([[-2., -1, 0, 1,2]])
# layer = single_input_gate_np(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# layer.eval()
# print(layer(input), "layer(input)")
# fds=432



class AND_np(torch.nn.Module):
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
        # self.input_per_gate = torch.nn.Parameter(torch.tensor([input_per_gate]), requires_grad=False)
        # self.input_per_gate.requires_grad_(False)
        
        # The intermediate result will be binarized with following layers.
        #self.Binarize = Binarize.create_01_to_01()
        self.Binarize_doesnot_need_gramo = Binarize.create_analog_to_np(False)
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
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            x = x.sum(dim=2, keepdim=False)#dim=2
            #back to rank-2
            
            offset = torch.tensor([self.input_per_gate], dtype=x.dtype, device=x.device).neg()+1.
            x = x + offset
            
            # binarize 
            x = self.Binarize_doesnot_need_gramo(x)
            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.all(dim=2, keepdim=False)
                x = x.to(torch.int8)
                x = x*2-1
                x = x.to(input.dtype)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x
                else:
                    opposite = x.neg()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite
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

# '''Does this gate layer protect the grad?'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# print(input)
# print(input.shape)
# layer = AND_np(input_per_gate=2)
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432


# '''Not important. The shape of eval path'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b,a,b,a,b,a,b), dim=1) 
# print(input.shape)
# layer = AND_01(input_per_gate=2)
# layer.eval()
# result = layer(input)
# print(result.shape)
# fds=432


# '''a basic test'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
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




class OR_np(torch.nn.Module):
    r""" unfinished docs"""
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
        # self.input_per_gate = torch.nn.Parameter(torch.tensor([input_per_gate]), requires_grad=False)
        # self.input_per_gate.requires_grad_(False)
        
        # The intermediate result will be binarized with following layers.
        #self.Binarize = Binarize.create_01_to_01()
        self.Binarize_doesnot_need_gramo = Binarize.create_analog_to_np(False)
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
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            x = x.sum(dim=2, keepdim=False)#dim=2
            #back to rank-2
            
            #offset = float(self.input_per_gate)-1.
            offset = torch.tensor([self.input_per_gate], dtype=x.dtype, device=x.device)-1.
            x = x + offset
            
            # binarize 
            x = self.Binarize_doesnot_need_gramo(x)
            
            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.any(dim=2, keepdim=False)
                x = x.to(torch.int8)
                x = x*2-1
                x = x.to(input.dtype)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x
                else:
                    opposite = x.neg()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite
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


# '''Does this gate layer protect the grad?'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# print(input)
# print(input.shape)
# layer = OR_np(input_per_gate=2)
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432


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





class XOR_np(torch.nn.Module):
    r""" unfinished docs"""
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
        # self.input_per_gate = torch.nn.Parameter(torch.tensor([input_per_gate]), requires_grad=False)
        # self.input_per_gate.requires_grad_(False)
        
        # The intermediate result will be binarized with following layers.
        #self.Binarize = Binarize.create_analog_to_01()
        #self.Binarize_doesnot_need_gramo = Binarize.create_analog_to_np(False)
        
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
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            #no offset for XOR_np
            x = x.prod(dim=2)
            #back to rank-2

            raise Exception("the following line is removed. untested.")
            #x = self.Binarize_doesnot_need_gramo(x)#hey, you don't need it.
            
            # Now x is actually the XNOR
            # The output part is different from AND and OR styles.
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([opposite, x], dim=1)
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
          
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.to(torch.int8)
                #overflow doesn't affect the result. 
                x = x.sum(dim=2, keepdim=False)
                x = x%2
                x = x*2-1
                x = x.to(input.dtype)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x.to(input.dtype)
                else:
                    opposite = x.neg()
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

# '''Does this gate layer protect the grad?'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# print(input)
# print(input.shape)
# layer = XOR_np(input_per_gate=2)
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432


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
    
    # protect_param_every____training:int
    # training_count:int

    def __init__(self, in_features: int, \
        SIG_single_input_gates: int, and_gates: int, or_gates: int, xor_gates: int, \
            is_last_layer:bool, \
                    training_ghost_weight_probability = 0.65, \
                        scaling_ratio_scaled_by = 1., \
            check_dimentions:bool = True, \
                    #protect_param_every____training:int = 20, \
            #scaling_ratio_for_gramo_in_mapper:float = 200., \
                    device=None, dtype=None) -> None:       #, debug_info = ""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if check_dimentions:
            raise Exception("to do")
        
        
        self.in_mapper_in_features = in_features
        self.in_mapper_out_features = SIG_single_input_gates+(and_gates+or_gates+xor_gates)*2
        self.SIG_count = SIG_single_input_gates
        self.and_gate_count = and_gates
        self.or_gate_count = or_gates
        self.xor_gate_count = xor_gates
        self.update_gate_input_index()
        self.is_last_layer = is_last_layer
        
        # Creates the layers.
        self.in_mapper = DigitalMapper_V1_4(self.in_mapper_in_features, self.in_mapper_out_features, training_ghost_weight_probability = training_ghost_weight_probability)#,
                                       #debug_info = debug_info)
        self.in_mapper.scale_the_scaling_ratio_for_raw_weight(scaling_ratio_scaled_by)
        
        output_mode = 1#1_is_both
        self.single_input_gate = single_input_gate_np(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
        self.and_gate = AND_np(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
        self.or_gate = OR_np(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
        self.xor_gate = XOR_np(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
       
        # mapper already has a gramo.
        #self.out_gramo = GradientModification()# I guess this should be default param.
       
        # For the param protection.
        #self.protect_param_every____training = protect_param_every____training 
        #self.training_count = 1 
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
        if self.is_last_layer:
            return self.SIG_count+self.in_mapper_out_features+2#with -1 and 1.
        else:
            return self.SIG_count+self.in_mapper_out_features
        #old version. return self.in_mapper_out_features+2
    def get_mapper_raw_weight(self)->torch.nn.Parameter:
        return self.in_mapper.raw_weight
    
    def get_mapper_scaling_ratio_for_inner_raw(self)->torch.nn.parameter.Parameter:
        '''simply gets from the inner layer.'''
        return self.in_mapper.get_scaling_ratio()
    def set_scaling_ratio_for_inner_raw(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        self.in_mapper.set_scaling_ratio(scaling_ratio)
    def reset_scaling_ratio_for_inner_raw(self):
        '''simply sets the inner layer.'''
        self.in_mapper.reset_scaling_ratio_for_raw_weight()
    def scale_the_scaling_ratio_for_inner_raw(self, by:float):
        '''simply sets the inner layer.'''
        self.in_mapper.scale_the_scaling_ratio_for_raw_weight(by)
    
    def get_gates_big_number(self)->Tuple[torch.nn.parameter.Parameter,
                            torch.nn.parameter.Parameter,torch.nn.parameter.Parameter]:
        big_number_of_and_gate = self.and_gate.Binarize_doesnot_need_gramo.big_number_list
        big_number_of_or_gate = self.or_gate.Binarize_doesnot_need_gramo.big_number_list
        big_number_of_xor_gate = self.xor_gate.Binarize_doesnot_need_gramo.big_number_list
        return (big_number_of_and_gate,big_number_of_or_gate,big_number_of_xor_gate)
    def set_gates_big_number(self, big_number_for_and_gate:List[float],
                    big_number_for_or_gate:List[float], big_number_for_xor_gate:List[float]):
        '''use any number <=0. to tell the code not to modify the corresponding one.'''
        self.and_gate.Binarize_doesnot_need_gramo.set_big_number_with_float(big_number_for_and_gate)
        self.or_gate.Binarize_doesnot_need_gramo.set_big_number_with_float(big_number_for_or_gate)
        self.xor_gate.Binarize_doesnot_need_gramo.set_big_number_with_float(big_number_for_xor_gate)
        #end of function.
             
    def set_auto_print_difference_between_epochs(self, set_to:bool = True):
        '''Simply sets the inner.'''
        self.in_mapper.set_auto_print_difference_between_epochs(set_to)        
        pass

    def get_zero_grad_ratio(self, directly_print_out:float = False)->float:
        return self.in_mapper.get_zero_grad_ratio(directly_print_out)
    
    
    def debug_strong_grad_ratio(self, log10_diff = 0., epi_for_w = 0.01, epi_for_g = 0.01,) -> float:
        return self.in_mapper.debug_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g)
        
    def after_step(self):
        self.in_mapper.after_step()
        pass
    
    def debug__old_func__can_convert_into_eval_only_mode(self)->Tuple[torch.Tensor, torch.Tensor]:
        return self.in_mapper.can__old_func__convert_into_eval_only_mode()
    
    def set_eval_mode(self, eval_mode_0_is_raw__1_is_sharp):
        self.in_mapper.eval_mode_0_is_raw__1_is_sharp = eval_mode_0_is_raw__1_is_sharp
        pass
    
    
    
    def update_gate_input_index(self):
        pos_to = 0
        
        pos_from = pos_to
        pos_to += self.SIG_count#no *2
        self.SIG_from = pos_from
        self.SIG_to = pos_to
            
        pos_from = pos_to
        pos_to += self.and_gate_count*2
        self.and_gate_from = pos_from
        self.and_gate_to = pos_to
        
        pos_from = pos_to
        pos_to += self.or_gate_count*2
        self.or_gate_from = pos_from
        self.or_gate_to = pos_to
        
        pos_from = pos_to
        pos_to += self.xor_gate_count*2
        self.xor_gate_from = pos_from
        self.xor_gate_to = pos_to
        pass
        #end of function.
    def print_gate_input_index(self):
        result_str = f'Single input gate:{self.SIG_from}>>>{self.SIG_to}, '
        result_str += f'and gate:{self.and_gate_from}>>>{self.and_gate_to}, '
        result_str += f'or gate:{self.or_gate_from}>>>{self.or_gate_to}, '
        result_str += f'xor gate:{self.xor_gate_from}>>>{self.xor_gate_to}.'
        print(result_str)        
        pass
    
    def print_param_overlap_ratio(self):
        raise Exception("untested. maybe unfinished.")
        '''Important: this function doesn't call the in_mapper version directly.
        The reason is that, the meaning of overlapping are different in 2 cases.
        Only the SIG part is the same as the in_mapper.'''
        with torch.no_grad():
            the_max_index = self.in_mapper.raw_weight.data.max(dim=1).indices
            the_dtype = torch.int32
            if self.SIG_count>1:
                total_overlap_count = 0
                total_possible_count = self.SIG_count*(self.SIG_count-1)//2
                for i in range(self.SIG_count-1):
                    host_index = torch.tensor([self.SIG_from+(i)], dtype=the_dtype)
                    guest_index = torch.linspace(self.SIG_from+(i+1)*2,
                                            self.SIG_from+(self.SIG_count-1)*2,
                                            self.SIG_count-i-1, dtype=the_dtype)
                    flag_overlapped = the_max_index[guest_index].eq(the_max_index[host_index])
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("SIG_overlap_ratio:", 
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.SIG_gate_count>0:
            
            if self.and_gate_count>1:
                total_overlap_count = 0
                total_possible_count = self.and_gate_count*(self.and_gate_count-1)//2
                for i in range(self.and_gate_count-1):
                    host_index = torch.tensor([self.and_gate_from+(i)*2], dtype=the_dtype)
                    guest_index = torch.linspace(self.and_gate_from+(i+1)*2,
                                            self.and_gate_from+(self.and_gate_count-1)*2,
                                            self.and_gate_count-i-1, dtype=the_dtype)
                    flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
                    flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
                    flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("AND_overlap_ratio:", 
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.and_gate_count>0:
                
            if self.or_gate_count>1:
                total_overlap_count = 0
                total_possible_count = self.or_gate_count*(self.or_gate_count-1)//2
                for i in range(self.or_gate_count-1):
                    host_index = torch.tensor([self.or_gate_from+(i)*2], dtype=the_dtype)
                    guest_index = torch.linspace(self.or_gate_from+(i+1)*2,
                                            self.or_gate_from+(self.or_gate_count-1)*2,
                                            self.or_gate_count-i-1, dtype=the_dtype)
                    flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
                    flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
                    flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("OR_overlap_ratio:", 
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.or_gate_count>0:
            
            if self.xor_gate_count>1:
                total_overlap_count = 0
                total_possible_count = self.xor_gate_count*(self.xor_gate_count-1)//2
                for i in range(self.xor_gate_count-1):
                    host_index = torch.tensor([self.xor_gate_from+(i)*2], dtype=the_dtype)
                    guest_index = torch.linspace(self.xor_gate_from+(i+1)*2,
                                            self.xor_gate_from+(self.xor_gate_count-1)*2,
                                            self.xor_gate_count-i-1, dtype=the_dtype)
                    flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
                    flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
                    flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("XOR_overlap_ratio:", 
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.xor_gate_count>0:
            pass
        pass
    #end of function
        
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''This example only shows the training path.'''
        
        # some extra safety. It's safe to comment this part out.
        if len(self.and_gate.Binarize_doesnot_need_gramo.gramos)>0:
            raise Exception("self.and_gate has gramo !!!")
        if len(self.or_gate.Binarize_doesnot_need_gramo.gramos)>0:
            raise Exception("self.or_gate has gramo !!!")
        if len(self.xor_gate.Binarize_doesnot_need_gramo.gramos)>0:
            raise Exception("self.xor_gate has gramo !!!")
        
        #param protection
        # if self.training_count<self.protect_param_every____training:
        #     self.training_count +=1
        # else:
        #     self.training_count = 1
            # with torch.no_grad():
            #     if self.print_overlapping_param:
            #         #print("if self.print_overlapping_param   __line__   3174")
            #         #print(self.in_mapper.raw_weight.data) 
            #         #SIG = single input gate.
            #         the_max_index = self.in_mapper.raw_weight.data.max(dim=1).indices
            #         #print(the_max_index)
            #         #single input gate, or the NOT gate.
            #         the_dtype = torch.int32
                    
            #         if self.SIG_count>1:
            #             total_overlap_count = 0
            #             total_possible_count = self.SIG_count*(self.SIG_count-1)//2
            #             for i in range(self.SIG_count-1):
            #                 host_index = torch.tensor([self.SIG_from+(i)], dtype=the_dtype)
            #                 guest_index = torch.linspace(self.SIG_from+(i+1)*2,
            #                                         self.SIG_from+(self.SIG_count-1)*2,
            #                                         self.SIG_count-i-1, dtype=the_dtype)
            #                 flag_overlapped = the_max_index[guest_index].eq(the_max_index[host_index])
            #                 #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
            #                 total_overlap_count += int(flag_overlapped.sum().item())
            #                 pass
            #             overlap_ratio = float(total_overlap_count)/total_possible_count
            #             print("SIG_overlap_ratio:", 
            #                     f'{overlap_ratio:.4f}',", ", total_overlap_count,
            #                     "/", total_possible_count)
            #             pass#if self.SIG_gate_count>0:
                    
            #         if self.and_gate_count>1:
            #             total_overlap_count = 0
            #             total_possible_count = self.and_gate_count*(self.and_gate_count-1)//2
            #             for i in range(self.and_gate_count-1):
            #                 host_index = torch.tensor([self.and_gate_from+(i)*2], dtype=the_dtype)
            #                 guest_index = torch.linspace(self.and_gate_from+(i+1)*2,
            #                                         self.and_gate_from+(self.and_gate_count-1)*2,
            #                                         self.and_gate_count-i-1, dtype=the_dtype)
            #                 flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
            #                 flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
            #                 flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
            #                 #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
            #                 total_overlap_count += int(flag_overlapped.sum().item())
            #                 pass
            #             overlap_ratio = float(total_overlap_count)/total_possible_count
            #             print("AND_overlap_ratio:", 
            #                     f'{overlap_ratio:.4f}',", ", total_overlap_count,
            #                     "/", total_possible_count)
            #             pass#if self.and_gate_count>0:
                        
            #         if self.or_gate_count>1:
            #             total_overlap_count = 0
            #             total_possible_count = self.or_gate_count*(self.or_gate_count-1)//2
            #             for i in range(self.or_gate_count-1):
            #                 host_index = torch.tensor([self.or_gate_from+(i)*2], dtype=the_dtype)
            #                 guest_index = torch.linspace(self.or_gate_from+(i+1)*2,
            #                                         self.or_gate_from+(self.or_gate_count-1)*2,
            #                                         self.or_gate_count-i-1, dtype=the_dtype)
            #                 flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
            #                 flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
            #                 flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
            #                 #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
            #                 total_overlap_count += int(flag_overlapped.sum().item())
            #                 pass
            #             overlap_ratio = float(total_overlap_count)/total_possible_count
            #             print("OR_overlap_ratio:", 
            #                     f'{overlap_ratio:.4f}',", ", total_overlap_count,
            #                     "/", total_possible_count)
            #             pass#if self.or_gate_count>0:
                    
            #         if self.xor_gate_count>1:
            #             total_overlap_count = 0
            #             total_possible_count = self.xor_gate_count*(self.xor_gate_count-1)//2
            #             for i in range(self.xor_gate_count-1):
            #                 host_index = torch.tensor([self.xor_gate_from+(i)*2], dtype=the_dtype)
            #                 guest_index = torch.linspace(self.xor_gate_from+(i+1)*2,
            #                                         self.xor_gate_from+(self.xor_gate_count-1)*2,
            #                                         self.xor_gate_count-i-1, dtype=the_dtype)
            #                 flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
            #                 flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
            #                 flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
            #                 #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
            #                 total_overlap_count += int(flag_overlapped.sum().item())
            #                 pass
            #             overlap_ratio = float(total_overlap_count)/total_possible_count
            #             print("XOR_overlap_ratio:", 
            #                     f'{overlap_ratio:.4f}',", ", total_overlap_count,
            #                     "/", total_possible_count)
            #             pass#if self.xor_gate_count>0:
                    
            #             pass
                    
            pass#end of param protection.
        
        x = input
        x = self.in_mapper(x)
        not_head:torch.Tensor = self.single_input_gate(
            x[:, self.SIG_from:self.SIG_to])
        #print(not_head, "not_head")
            
        and_head:torch.Tensor = self.and_gate(x[:, self.and_gate_from:self.and_gate_to])
        #print(and_head, "and_head")
        
        or_head = self.or_gate(x[:, self.or_gate_from:self.or_gate_to])  
        #print(or_head, "or_head")
        
        xor_head = self.xor_gate(x[:, self.xor_gate_from:self.xor_gate_to])
        #print(xor_head, "xor_head")
        
        if not self.is_last_layer:
            x = torch.concat([not_head, and_head, or_head, xor_head],dim=1)
            return x
        else:
            neg_ones = torch.empty([input.shape[0],1])
            neg_ones = neg_ones.fill_(-1.)
            neg_ones = neg_ones.to(and_head.device).to(and_head.dtype)
            ones = torch.ones([input.shape[0],1])
            ones = ones.to(and_head.device).to(and_head.dtype)
            x = torch.concat([not_head, and_head, or_head, xor_head, neg_ones, ones],dim=1)
            return x
    #end of function
    
    def extra_repr(self) -> str:
        result = f'In features:{self.in_mapper.in_features}, out features:{self.get_out_features()}, AND2:{self.and_gate_count}, OR2:{self.or_gate_count}, XOR2:{self.xor_gate_count}'
        return result
    
    def get_info(self, directly_print:bool=False) -> str:
        result = f'In features:{self.in_mapper.in_features}, out features:{self.get_out_features()}, AND2:{self.and_gate_count}, OR2:{self.or_gate_count}, XOR2:{self.xor_gate_count}'
        if directly_print:
            print(result)
        return result
    
    pass
fast_traval____end_of_DSPU_layer = 432
'''This is not a test. This is actually a reminder. The input and output sequence of gates 
layers are not very intuitive. If the gates are denoted as a,b,c, inputs are a1,a2,b1,b2, 
outputs are ao,-ao,bo,-bo, then the real sequence is:
input: a1,a2,b1,b2,c1,c2,d1,d2
output: a,b,c,d,-a,-b,-c,-d,-1,1
The last 2 are from neg_ones and ones. They are a bit similar to the bias in FCNN(wx+b).
'''
input = torch.tensor([[1., 2., 3.]], requires_grad=True)
layer = DigitalSignalProcessingUnit_layer(in_features=3,SIG_single_input_gates=2,and_gates=0,
            or_gates=0,xor_gates=0, is_last_layer=True, check_dimentions=False, 
            training_ghost_weight_probability=0.5)
print(layer.in_mapper.raw_weight.data.shape)
layer.in_mapper.raw_weight.data = torch.tensor([[ 1.,  0.,  0.],[ 0.,  1.,  0.],])
layer.after_step()
print(layer.in_mapper.raw_weight.data.shape)
layer.print_gate_input_index()
print(layer(input), "should be 1,2,-1,-2,-1,1, the last -1,1 are neg_ones and ones")

input = torch.tensor([[-1., 1.]], requires_grad=True)
layer = DigitalSignalProcessingUnit_layer(in_features=3,SIG_single_input_gates=2,and_gates=0,
            or_gates=0,xor_gates=0, is_last_layer=True, check_dimentions=False, 
            training_ghost_weight_probability=0.5)
layer.in_mapper.raw_weight.data = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],])
layer.after_step()
print(layer.in_mapper.raw_weight.data.shape)
layer.print_gate_input_index()
print(layer(input), "should be -1-1-1 1 1 1,-1,1, the last -1,1 are neg_ones and ones")

fds=432



# '''finished???? It can only tell how many overlap. 
# parameter overlapping and protection test.'''
# input = torch.tensor([[1., 2.]], requires_grad=True)
# layer = DigitalSignalProcessingUnit_layer(in_features=2,
#                 SIG_single_input_gates=0,and_gates=3,or_gates=0,xor_gates=0, 
#                 check_dimentions=False, protect_param_every____training=0)
# # 6 SIGs or 3 of any other gate.
# #print(layer.in_mapper.raw_weight.data.shape)
# layer.in_mapper.raw_weight.data = torch.tensor([
#     [ 0.,  1.],[ 0.,  1.],[ 0.,  1.],[ 0.,  1.],[ 0.,  1.],[ 0.,  1.],])
# layer.print_param_overlap_ratio()

# input = torch.tensor([[-1.,-1.,-1.,]])#this doesn't matter.
# layer = DigitalSignalProcessingUnit_layer(in_features=2,
#                         SIG_single_input_gates=5,and_gates=3,or_gates=4,xor_gates=5,
#                         check_dimentions=False, protect_param_every____training=0)
# #print(layer.in_mapper.raw_weight.data.shape)
# layer.in_mapper.raw_weight.data = torch.tensor([
#         [5,2,3],[1,5,3],[1,2,3],[1,2,3],[1,2,3],
#         [1,2,3],[1,2,3],[1,2,3],[1,6,3],[6,2,3],[1,2,3],
#         [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,6,3],[1,2,3],[1,6,3],[1,2,3],
#         [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,5,3],[1,2,3],[1,6,3],
#                                                 ],dtype=torch.float32)
# layer.print_param_overlap_ratio()

# fds=432






# '''This test shows the gramo in digital_mapper is the only gramo in main grad pass.
# '''
# input = torch.tensor([[1.]], requires_grad=True)
# layer = DigitalSignalProcessingUnit_layer(in_features=1,single_input_gates=1,and_gates=1,
#                                             or_gates=1,xor_gates=1)
# #print(len(layer.and_gate.Binarize.gramos), __line__str())
# output = layer(input)
# print(output, "should be 1., -1.,  1., -1.,  1., -1., -1.,  1.")
# if False:
#     g_in = torch.zeros_like(output)
#     g_in[0,0] = 1.
#     pass
# if True:
#     g_in = torch.ones_like(output)*111111
#     g_in[0,0] = 1.
#     pass
# torch.autograd.backward(output, g_in, inputs=input)
# print(input.grad, "input.grad", __line__str())
# fds=432

# '''basic test. eval mode.'''
# input = torch.tensor([[1.]])
# layer = DigitalSignalProcessingUnit_layer(1,1,1,1,1)
# layer.eval()
# output = layer(input)
# print(output, "should be 1., -1.,  1., -1.,  1., -1., -1.,  1.")
# fds=432

# layer = DigitalSignalProcessingUnit_layer(2,3,5,7,11)
# layer.get_info(True)
# print("out feature should be ", 3*2+5*2+7*2+11*2)
# fds=432



class DSPU(torch.nn.Module):
    r'''n DSPU layers in a row. Interface is the same as the single layer version.
    
    It contains an extra mapper layer in the end. 
    
    It's a ((mapper+gates)*n)+mapper structure.
    
    More info see DigitalSignalProcessingUnit.
    '''
    #__constants__ = ['',]
    def __init__(self, in_features: int, out_features:int, \
                single_input_gates: int, and_gates: int, or_gates: int, \
                    xor_gates: int, num_layers:int, \
                    scaling_ratio_for_gramo_in_mapper:float = 200., \
                    scaling_ratio_for_gramo_in_mapper_for_first_layer:Optional[float] = None, \
                    scaling_ratio_for_gramo_in_mapper_for_out_mapper:Optional[float] = None, \
                    protect_param_every____training:int = 20, \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        if scaling_ratio_for_gramo_in_mapper_for_first_layer is None:
            scaling_ratio_for_gramo_in_mapper_for_first_layer = scaling_ratio_for_gramo_in_mapper
            
        if scaling_ratio_for_gramo_in_mapper_for_out_mapper is None:
            scaling_ratio_for_gramo_in_mapper_for_out_mapper = scaling_ratio_for_gramo_in_mapper
        
        
        #debug_print_raw_param_in_training_pass
        self.first_layer = DigitalSignalProcessingUnit_layer(
            in_features, single_input_gates, and_gates, or_gates, xor_gates, 
            scaling_ratio_for_gramo_in_mapper=scaling_ratio_for_gramo_in_mapper_for_first_layer, 
            check_dimentions=False)#, debug_info = "first layer")
        
        mid_width = self.first_layer.get_out_features()
        self.mid_layers = torch.nn.modules.container.ModuleList( \
                [DigitalSignalProcessingUnit_layer(mid_width, single_input_gates, 
                                                   and_gates, or_gates, xor_gates, 
                        scaling_ratio_for_gramo_in_mapper=scaling_ratio_for_gramo_in_mapper,
                        check_dimentions=False)   
                    for i in range(num_layers-1)])
        self.last_layer = 
        self.num_layers = num_layers
           
        self.out_mapper = DigitalMapper_V1_4(mid_width+2, out_features,
                                       )# , debug_info="out mapper")
        

        self.protect_param_every____training = protect_param_every____training 
        self.training_count = 1
        self.set_protect_param_every____training(protect_param_every____training)
        #end of function        
           
    def set_protect_param_every____training(self, set_to= 20):
        self.protect_param_every____training = set_to 
        self.first_layer.protect_param_every____training = set_to
        for layer in self.second_to_last_layers:
            layer.protect_param_every____training = set_to
            pass
        pass 
        
    # def set_auto_print_overlapping_param(self, set_to:bool = True):
    #     self.first_layer.print_overlapping_param = set_to
    #     for layer in self.second_to_last_layers:
    #         layer.print_overlapping_param = set_to
    #         pass
    #     pass
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    def get_mapper_scaling_ratio_for_inner_raw_from_all(self)->List[torch.nn.parameter.Parameter]:
        result:List[torch.nn.parameter.Parameter] = []
        result.append(self.first_layer.get_mapper_scaling_ratio_for_inner_raw())
        layer: DigitalSignalProcessingUnit_layer
        for layer in self.second_to_last_layers:
            result.append(layer.get_mapper_scaling_ratio_for_inner_raw())
            pass
        result.append(self.out_mapper.get_scaling_ratio())
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
    
    
    def get_gates_big_number_from_all(self)->List[Tuple[torch.nn.parameter.Parameter,
                        torch.nn.parameter.Parameter, torch.nn.parameter.Parameter]]:
        result:List[Tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter,
                         torch.nn.parameter.Parameter]] = []
        result.append(self.first_layer.get_gates_big_number())
        for layer in self.second_to_last_layers:
            result.append(layer.get_gates_big_number())
            pass
        return result
    def print_gates_big_number_from_all(self):
        L_T_P:List[Tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter,
                         torch.nn.parameter.Parameter]]
        L_T_P = self.get_gates_big_number_from_all()
        
        print("big_number_print:------------------")
        for i, layer_info in enumerate(L_T_P):
            str_to_print = f"layer_{i}: and gate:"
            str_to_print += str(debug_Rank_1_parameter_to_List_float(layer_info[0]))
            str_to_print = f", or gate:"
            str_to_print += str(debug_Rank_1_parameter_to_List_float(layer_info[1]))
            str_to_print = f", xor gate:"
            str_to_print += str(debug_Rank_1_parameter_to_List_float(layer_info[2]))
            pass
        print(str_to_print)
        return 
    
    def set_auto_print_difference_between_epochs_for_second_to_last_DSPU_layers(self, set_to:bool = True):
        for layer in self.second_to_last_layers:
            layer.set_auto_print_difference_between_epochs(set_to)
            pass    
        pass  
    
    def set_auto_print_difference_between_epochs(self, set_to:bool = True, for_whom:int = 1):
        '''The param:for_whom:
        >>> 0 all
        >>> 1 important(first, second, out_mapper)
        >>> 2 first
        >>> 3 second
        >>> 4 second to last(DSPU_layer)
        >>> 5 out_mapper
        '''
        DSPU_layer:DigitalSignalProcessingUnit_layer
        if 0 == for_whom:#all
            self.first_layer.set_auto_print_difference_between_epochs(set_to)
            self.set_auto_print_difference_between_epochs_for_second_to_last_DSPU_layers(set_to)
            self.out_mapper.set_auto_print_difference_between_epochs(set_to)
        if 1 == for_whom:#important(first, second, out_mapper)
            self.first_layer.set_auto_print_difference_between_epochs(set_to)
            if len(self.second_to_last_layers)>0:
                DSPU_layer = self.second_to_last_layers[0]
                DSPU_layer.set_auto_print_difference_between_epochs(set_to)
            self.out_mapper.set_auto_print_difference_between_epochs(set_to)
        if 2 == for_whom:#first
            self.first_layer.set_auto_print_difference_between_epochs(set_to)
        if 3 == for_whom:#second
            if len(self.second_to_last_layers)>0:
                DSPU_layer = self.second_to_last_layers[0]
                DSPU_layer.set_auto_print_difference_between_epochs(set_to)
        if 4 == for_whom:#second to last(DSPU_layer)
            self.set_auto_print_difference_between_epochs_for_second_to_last_DSPU_layers(set_to)
        if 5 == for_whom:#out_mapper
            self.out_mapper.set_auto_print_difference_between_epochs(set_to)
        pass
    
    def reset_auto_print_difference_between_epochs(self):
        self.set_auto_print_difference_between_epochs(False, 0)
        pass
    
    # def set_gates_big_number_for_all(self, big_number_of_and_gate:float,
    #                 big_number_of_or_gate:float, big_number_of_xor_gate:float):
    #     '''use any number <=0. not to modify the corresponding one.'''
    #     self.first_layer.set_gates_big_number(big_number_for_and_gate = big_number_of_and_gate,
    #                                         big_number_for_or_gate = big_number_of_or_gate, 
    #                                         big_number_for_xor_gate = big_number_of_xor_gate)
    #     for layer in self.second_to_last_layers:
    #         layer.set_gates_big_number(big_number_of_and_gate = big_number_of_and_gate,
    #                                 big_number_of_or_gate = big_number_of_or_gate, 
    #                                 big_number_of_xor_gate = big_number_of_xor_gate)
    #         pass
    #     pass
        #end of function
    
    
    
    def get_zero_grad_ratio(self, directly_print_out:float = False)->List[float]:
        result:List[float] = []
        result.append(self.first_layer.get_zero_grad_ratio(directly_print_out))
        for layer in self.second_to_last_layers:
            result.append(layer.get_zero_grad_ratio(directly_print_out))
            pass
        result.append(self.out_mapper.get_zero_grad_ratio(directly_print_out))
        return result
    
    def print_param_overlap_ratio(self):
        self.first_layer.print_param_overlap_ratio()
        for layer in self.second_to_last_layers:
            layer.print_param_overlap_ratio()
            pass
        pass
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''This example only shows the training path.'''
        x = input
        #fds = self.first_layer.in_mapper.inner_raw.raw_weight.dtype
        x = self.first_layer(x)
        #print(x, "xxxxxxxxxxxxxxxxxxxxxx")

        for layer in self.second_to_last_layers:
            x = layer(x)
            pass
        x = self.out_mapper(x)
        return x
    #end of function
    
    def get_info(self, directly_print:bool=False) -> str:
        
        result_str = f'{self.num_layers} DSPUs in a row. In features:'
        result_str += str(self.in_features)
        result_str += ", out features:"
        result_str += str(self.out_features)
        result_str += ", AND2:"
        result_str += str(self.first_layer.and_gate_count)
        result_str += ", OR2:"
        result_str += str(self.first_layer.or_gate_count)
        result_str += ", XOR2:"
        result_str += str(self.first_layer.xor_gate_count)
        if directly_print:
            print(result_str)
        return result_str
    
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
    a = int_into_floats(a,1, False)    
    b = int_into_floats(b,1, False)    
    c = int_into_floats(c,1, False)    
    input = torch.concat([a,b,c], dim=1)
    input = input.requires_grad_()
    target = int_into_floats(target,1+1, False)    
    print(string, model(input), "should be", target)





# input = torch.ones([1,10], requires_grad=True)
# layer = single_input_gate_np(1)
# output = layer(input)
# g_in = torch.zeros_like(output)
# g_in[0,:5] = 1.
# torch.autograd.backward(output, g_in, inputs=input)
# print(g_in)
# print(input.grad)
# fds=432

# '''and, or, xor takes any grad from its output and duplicate it to both input'''
# input = torch.ones([1,10], requires_grad=True)
# layer = AND_np(2,1)
# output = layer(input)
# g_in = torch.zeros_like(output)
# g_in[0,:3] = 1.
# torch.autograd.backward(output, g_in, inputs=input)
# print(g_in)
# print(input.grad)
# fds=432

#DigitalMapper_eval_only_v2






# '''a basic test.
# This test actually shows how to print out the difference before/after step.

# The result of this test is a bit crazy. Half of the times, it finishes in 1 epoch.
# '''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# target = torch.tensor([[-1.], [1.], [1.]])
# model = DSPU(input.shape[1],target.shape[1],2,1,1,1,num_layers=1, scaling_ratio_for_gramo_in_mapper=200.)
# model.get_info(directly_print=True)
# model.set_auto_print_overlapping_param(True)
# model.set_protect_param_every____training(0)
# #model.set_auto_print_difference_between_epochs()

# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# iter_per_print = 1#111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#       print(pred.shape)
#       print(target.shape)
#       fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred, "pred")
#             print(target, "target")
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "make_grad_noisy":
#         if epoch%iter_per_print == iter_per_print-1:
#             make_grad_noisy(model, 2.)
#             pass
#         pass
    
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.first_layer.in_mapper.raw_weight.grad, "grad")
#             #print(model.second_to_last_layers[0])
#             print(model.out_mapper.raw_weight.grad, "grad")
#             pass
#         pass
#     if False and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             #layer = model.out_mapper
#             layer = model.first_layer.in_mapper
#             print(layer.raw_weight, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "first_layer.in_mapper   after update")
            
#             layer = model.out_mapper
#             print(layer.raw_weight, "out_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "out_mapper   after update")
            
#             pass    
#         pass    
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     if True and "print acc":
#         if epoch%iter_per_print == iter_per_print-1:
#             model.eval()
#             pred = model(input)
#             #print(pred, "pred", __line__str())
#             #print(target, "target")
#             acc = bitwise_acc(pred, target)
#             #print(epoch+1, "    ep/acc    ", acc)
#             if 1. == acc:
#                 print(epoch+1, "    ep/acc    ", acc)
#                 break
#             pass
#         pass

# fds=432



# '''does it fit in half adder.

# All the hyperparameters are somewhat tuned by me.'''
# batch = 100#10000
# (input, target) = data_gen_half_adder_1bit(batch, is_output_01=False,is_cuda=False)
# model = DSPU(input.shape[1],target.shape[1],2,1,1,1,num_layers=1, #1,2,3 are ok
#              scaling_ratio_for_gramo_in_mapper=1000.,
#              scaling_ratio_for_gramo_in_mapper_for_out_mapper=2000.)
# model.get_info(directly_print=True)
# #model.set_auto_print_difference_between_epochs()
# #model.set_auto_print_difference_between_epochs(True,5)
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# iter_per_print = 1#111
# print_count = 1555
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#         print(pred.shape, "pred.shape")
#         print(target.shape, "target.shape")
#         fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred, "pred")
#             print(target, "target")
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "make_grad_noisy":
#         make_grad_noisy(model, 2.)
#         pass
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.first_layer.in_mapper.raw_weight.grad, "grad")
#             #print(model.second_to_last_layers[0])
#             print(model.out_mapper.raw_weight.grad, "grad")
#             pass
#         pass
#     if False and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             #layer = model.out_mapper
#             layer = model.first_layer.in_mapper
#             print(layer.raw_weight, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "first_layer.in_mapper   after update")
            
#             layer = model.out_mapper
#             print(layer.raw_weight, "out_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "out_mapper   after update")
            
#             pass    
#         pass    
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     if True and "print acc":
#         if epoch%iter_per_print == iter_per_print-1:
#             model.eval()
#             pred = model(input)
#             #print(pred, "pred", __line__str())
#             #print(target, "target")
#             acc = bitwise_acc(pred, target)
#             print(epoch+1, "    ep/acc    ", acc)
#             if 1. == acc:
#                 print(epoch+1, "    ep/acc    ", acc)
#                 # print(pred[:5], "pred", __line__str())
#                 # print(target[:5], "target")
#                 break
#             pass
#         pass

# fds=432



# '''4 layers should be able to fit in AND3'''
# batch = 5#10000
# input = torch.randint(0,2,[batch,3])
# target = input.all(dim=1,keepdim=True)
# input = input*2-1
# input = input.to(torch.float32)
# target = target*2-1
# target = target.to(torch.float32)
# # print(input)
# # print(target)

# model = DSPU(input.shape[1],target.shape[1],8,4,4,4,num_layers=4, 
#              scaling_ratio_for_gramo_in_mapper=1000.,
#              scaling_ratio_for_gramo_in_mapper_for_out_mapper=2000.)
# model.get_info(directly_print=True)
# #model.set_auto_print_difference_between_epochs()
# #model.set_auto_print_difference_between_epochs(True,5)
# # model.set_auto_print_overlapping_param(True)
# # model.set_protect_param_every____training(0)
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# model.cuda()
# input = input.cuda()
# target = target.cuda()
# iter_per_print = 1#111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#         print(pred.shape, "pred.shape")
#         print(target.shape, "target.shape")
#         fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred, "pred")
#             print(target, "target")
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "make_grad_noisy":
#         make_grad_noisy(model, 1.5)
#         pass
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.first_layer.in_mapper.raw_weight.grad, "grad")
#             #print(model.second_to_last_layers[0])
#             print(model.out_mapper.raw_weight.grad, "grad")
#             pass
#         pass
#     if False and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             #layer = model.out_mapper
#             layer = model.first_layer.in_mapper
#             print(layer.raw_weight, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "first_layer.in_mapper   after update")
            
#             layer = model.out_mapper
#             print(layer.raw_weight, "out_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "out_mapper   after update")
            
#             pass    
#         pass    
#     if True and "print zero grad ratio":
#         result = model.get_zero_grad_ratio()
#         print("print zero grad ratio: ", result)
#         pass
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     if True and "print acc":
#         if epoch%iter_per_print == iter_per_print-1:
#             model.eval()
#             pred = model(input)
#             #print(pred, "pred", __line__str())
#             #print(target, "target")
#             acc = bitwise_acc(pred, target)
#             print(epoch+1, "    ep/acc    ", acc)
#             if 1. == acc:
#                 print(epoch+1, "    ep/acc    ", acc)
#                 # print(pred[:5], "pred", __line__str())
#                 # print(target[:5], "target")
#                 break
#             pass
#         pass

# fds=432




# 门层里面不应该gramo。应该在dspu层里面统一gramo，统一二值化。
# 宽度解决很多问题？或者说在补谁的问题？
# 反正mapper层还有一个做法。



#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。

def test_config_dispatcher(bit:int, layer:int, gates:int, batch:int, lr:float, noise_base:float):
    return(bit, layer, gates, batch, lr, noise_base)

'''does it fit in FULL adder. Probably YES!!!'''

'''
old
f16, bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.5>>> 1k2,2k ,0k2,0k3,1k5,1k3
f16, bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.2>>> 2k8,0k5,0k3,0k4,0k3,1k5
f16, bit=2,layer=6,gates=8,batch=100,lr=0.001,noise_base=1.2>>> random.
f16, bit=2,layer=6,gates=8,batch=100,lr=0.0001,noise_base=1.2>>> random.
f16, bit=2,layer=6,gates=8,batch=100,lr=0.01,noise_base=1.2>>> random.
f16, bit=2,layer=6,gates=8,batch=100,lr=0.001,noise_base=2.>>> random.???


to do:
UNIT:::: K epochs.
f16, bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.5  not*3>>> 0k3,6k7,0k2,0k1,5k
f16, bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.2  not*3>>> 0k4,0k9,1k, 0k3,1k7
f16, bit=2,layer=6,gates=8,batch=100,lr=0.001,noise_base=1.2  not*3>>> random
f16, bit=2,layer=6,gates=16,batch=100,lr=0.001,noise_base=1.2  not*3>>> random??

'''

# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个


(bit, layer, gates, batch, lr, noise_base) = test_config_dispatcher(
    bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.2)
is_f16 = True
iter_per_print = 111#1111
print_count = 5555

(input, target) = data_gen_full_adder(bit,batch, is_output_01=False, is_cuda=True)
# print(input[:5])
# print(target[:5])

model = DSPU(input.shape[1],target.shape[1],gates*2,gates,gates,gates,layer, 
             scaling_ratio_for_gramo_in_mapper=1000.,
             scaling_ratio_for_gramo_in_mapper_for_first_layer=330.,
             scaling_ratio_for_gramo_in_mapper_for_out_mapper=1000.,)#1000,330,1000
model.get_info(directly_print=True)
#model.set_auto_print_difference_between_epochs(True,4)
#model.set_auto_print_difference_between_epochs(True,5)

#model.set_protect_param_every____training(0)

model.cuda()
if is_f16:
    input = input.to(torch.float16)
    target = target.to(torch.float16)
    model.half()
    pass

# model_DSPU.print_mapper_scaling_ratio_for_inner_raw_from_all()
# model_DSPU.print_gates_big_number_from_all()
# fds=432


# print(model_DSPU.get_mapper_scaling_ratio_for_inner_raw_from_all())
# print(model_DSPU.get_gates_big_number_from_all())
#model_DSPU.first_layer.set_scaling_ratio()


#model.set_auto_print_difference_between_epochs(True,5)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model.cuda()
input = input.cuda()
target = target.cuda()
#iter_per_print = 1#111
#print_count = 1555
for epoch in range(iter_per_print*print_count):
    model.train()
    pred = model(input)
    #print(pred, "pred", __line__str())
    if False and "shape":
        print(pred.shape, "pred.shape")
        print(target.shape, "target.shape")
        fds=423
    if False and "print pred":
        if epoch%iter_per_print == iter_per_print-1:
            print(pred[:5], "pred")
            print(target[:5], "target")
            pass
        pass
    loss = loss_function(pred, target)
    optimizer.zero_grad()
    loss.backward()
    if True and "make_grad_noisy":
        make_grad_noisy(model, noise_base)
        pass
    if False and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            print(model.first_layer.in_mapper.raw_weight.grad, "first_layer   grad")
            print(model.second_to_last_layers[0].in_mapper.raw_weight.grad, "second_to_last_layers[0]   grad")
            print(model.out_mapper.raw_weight.grad, "out_mapper   grad")
            pass
        pass
    if False and "print the weight":
        if epoch%iter_per_print == iter_per_print-1:
            #layer = model.out_mapper
            layer = model.first_layer.in_mapper
            print(layer.raw_weight, "first_layer.in_mapper   before update")
            optimizer.step()
            print(layer.raw_weight, "first_layer.in_mapper   after update")
            
            layer = model.model.second_to_last_layers[0]
            print(layer.raw_weight, "second_to_last_layers[0]   before update")
            optimizer.step()
            print(layer.raw_weight, "second_to_last_layers[0]   after update")
            
            layer = model.out_mapper
            print(layer.raw_weight, "out_mapper   before update")
            optimizer.step()
            print(layer.raw_weight, "out_mapper   after update")
            
            pass    
        pass    
    if True and "print zero grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            result = model.get_zero_grad_ratio()
            print("print zero grad ratio: ", result)
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    optimizer.step()
    if True and "print param overlap":
        every = 100
        if epoch%every == every-1:
            model.print_param_overlap_ratio()
            pass
        pass
    if True and "print acc":
        if epoch%iter_per_print == iter_per_print-1:
            model.eval()
            pred = model(input)
            #print(pred, "pred", __line__str())
            #print(target, "target")
            acc = bitwise_acc(pred, target)
            print(epoch+1, "    ep/acc    ", acc)
            if 1. == acc:
                print(epoch+1, "    ep/acc    ", acc)
                print(pred[:5], "pred", __line__str())
                print(target[:5], "target")
                break
            pass
        pass

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









