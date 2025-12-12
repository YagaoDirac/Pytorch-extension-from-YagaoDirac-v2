from typing import List, Tuple, Optional, TypeGuard
import torch
import math





def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
if "test" and True:
    assert __DEBUG_ME__()
    pass

import sys
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######
if "test" and False:
    a = _line_()
    b = _line_()
    c = _line_()
    pass

def _float_equal(a:float, b:float, epi:float = 0.0001)->bool:
    assert epi>0.
    return abs(a-b)<epi
if "test" and __DEBUG_ME__() and True:
    assert _float_equal(1., 1.)
    assert _float_equal(1., 1.0000001)
    assert _float_equal(1., 1.01) == False
    assert _float_equal(1., 1.01, 0.1) 
    pass
def _tensor_equal(  a:torch.Tensor|list[float]|list[list[float]], \
                    b:torch.Tensor|list[float]|list[list[float]], \
                        epi:float = 0.0001)->bool:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        pass
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
        pass
    
    assert a.shape == b.shape
    with torch.inference_mode():
        diff = a-b
        abs_of_diff = diff.abs()
        less_than = abs_of_diff.lt(epi)
        after_all = less_than.all()
        assert after_all.dtype == torch.bool
        the_item = after_all.item()
        assert isinstance(the_item, bool)
        return the_item
    pass#end of function
if "test" and __DEBUG_ME__() and True:
    assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.]))
    assert _tensor_equal(torch.tensor([1.,2.]), [1.,2.])
    #assert _tensor_equal(torch.tensor([1.]), torch.tensor([[1.]]))
    assert _tensor_equal(torch.tensor([[1.]]), torch.tensor([[1.]]))
    assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.000001]))
    assert _tensor_equal(torch.tensor([1.]), torch.tensor([0.99999]))
    assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.001])) == False
    pass

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from timeit_yagaodirac import timeit
# from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
pass


def vector_length_norm(input:torch.Tensor, epi = 0.000001)->torch.Tensor:
    r'''The shape must be [batch, dim]'''
    if len(input.shape)!=2:
        raise Exception("The shape must be [batch, dim]")
    with torch.no_grad():
        
        length_of_input_b_1 = input.mul(input).sum(dim=1,keepdim=True).sqrt()
        epi_tensor = torch.tensor([epi], device=length_of_input_b_1.device, dtype=length_of_input_b_1.dtype)
        length_of_input_safe_b = length_of_input_b_1.maximum(epi_tensor)
        result = input/length_of_input_safe_b#.unsqueeze(dim=1)
        return result
    #end of function.
if '''some basic test.''' and __DEBUG_ME__() and True:
    input = torch.tensor([[0.,0.],[0.,1.],[1.,1.]])
    output = vector_length_norm(input)
    assert _tensor_equal(output, [[0.,0.],[0.,1.],[0.7,0.7]], 0.05)
    _vector_len = output.mul(output).sum(dim=1)
    assert _tensor_equal(_vector_len[0], torch.zeros_like(_vector_len[0]), 0.05)
    assert _tensor_equal(_vector_len[1:], torch.ones_like(_vector_len[1:]), 0.05)
    pass



# def protect_rotation_matrix(input:torch.Tensor, epi = 0.000001):#->torch.Tensor:
#     if len(input.shape)!=2:
#         raise Exception("send matrix here.")
#     dim = input.shape[0]
#     if dim!=input.shape[1]:
#         raise Exception("It must be square.")
    
#     with torch.no_grad():
#         # two_triagles = (input-input.T)*0.5
#         # diagonal = input.mul(torch.eye(dim))
#         # output_raw = two_triagles+diagonal
        
#         length_of_output_raw_b = input.mul(input).sum(dim=1,keepdim=False).sqrt()
#         epi_tensor = torch.tensor([epi], device=length_of_output_raw_b.device, dtype=length_of_output_raw_b.dtype)
#         length_of_output_raw_safe_b = length_of_output_raw_b.maximum(epi_tensor)
#         sqrt_of_length_b = length_of_output_raw_safe_b.sqrt()
#         #result = input/length_of_input_safe_b#.unsqueeze(dim=1)
#         output = input/sqrt_of_length_b.unsqueeze(dim=1)/sqrt_of_length_b.unsqueeze(dim=0)
        
#         raise Exception("test not passed..")
#         fds=432
    
#     #output = vector_length_norm(output_raw)#shape is intentional.
    
#     return output
# raw_from_randn = torch.tensor([[0.5,2],[3.,4]])#randn([2,2])
# rotation_matrix = protect_rotation_matrix(raw_from_randn)
# print(rotation_matrix[0].mul(rotation_matrix[0]).sum())
# print(rotation_matrix[1].mul(rotation_matrix[1]).sum())
# print(rotation_matrix.T[0].mul(rotation_matrix.T[0]).sum())
# print(rotation_matrix.T[1].mul(rotation_matrix.T[1]).sum())
# unit_length_vec = vector_length_norm(torch.randn([1,2])).unsqueeze(dim=2)
# print(unit_length_vec.mul(unit_length_vec).sum(), "unit_length_vec")
# after_rotation = rotation_matrix.matmul(unit_length_vec).squeeze(dim=2)
# print(after_rotation.mul(after_rotation).sum())
# length_after_rotation = after_rotation.mul(after_rotation).sum(dim=1)

# fds=432
    
    
        



# def float_to_spherical(input:torch.Tensor, mix = False)->torch.Tensor:
#     '''Basically, the mix flag only helps with debug. It may be slower a bit.'''
#     if len(input.shape)!=2:
#         raise Exception("The shape must be [batch, dim]")
#     if input.amax()>1. or input.amin()<0.:
#         raise Exception("Value must be inside [0., 1.] (both included.)")
#     input_in_rad =  input*torch.pi/2.
#     the_cos = input_in_rad.cos()
#     the_sin = input_in_rad.sin()
#     if not mix:
#         result = torch.concat([the_cos, the_sin], dim=1)
#         return result
#     the_cos = the_cos.unsqueeze(dim=2)
#     the_sin = the_sin.unsqueeze(dim=2)
#     result = torch.concat([the_cos, the_sin], dim=2)
#     result = result.view([input.shape[0], -1])
#     return result
# '''some basic test.'''
# input = torch.tensor([[0., 0.33333, 0.5], [0.6, 0.7, 0.8]])
# print(float_to_spherical(input))
# print(float_to_spherical(input, True))
# fds=432
        

# def spherical_to_float(input:torch.Tensor, mix = False, rigorous = False)->torch.Tensor:
#     if len(input.shape)!=2:
#         raise Exception("The shape must be [batch, dim]")
#     if input.shape[1]%2 == 1:
#         raise Exception("The dim must be 2x something. They are pairs of cos and sin.")
#     if rigorous and (input.amax()>1. or input.amin()<0.):
#         raise Exception("Value must be inside [0., 1.] (both included.). Or set the param:rigorous to False.")
#     if not mix:
#         reshaped_input = input.view([input.shape[0], 2, -1])
#         the_cos = reshaped_input[:,0,:]
#         the_sin = reshaped_input[:,1,:]
#         result_in_rad = torch.atan2(the_sin, the_cos)
#         result = result_in_rad*2./torch.pi
#         return result
#     # mixed.
#     reshaped_input = input.view([input.shape[0], -1, 2])
#     the_cos = reshaped_input[:,:,0]
#     the_sin = reshaped_input[:,:,1]
#     result_in_rad = torch.atan2(the_sin, the_cos)
#     result = result_in_rad*2./torch.pi
#     return result
# '''some basic test.'''
# temp = torch.tensor([[0., 0.33333, 0.5], [0.6, 0.7, 0.8]])
# input = float_to_spherical(temp)
# print(spherical_to_float(input))
# input = float_to_spherical(temp, mix=True)
# print(spherical_to_float(input, mix=True))
# fds=432




# 写法是v1的写法。而且应该是多输出的。
# 需要额外写一个function.set_materialize什么什么函数的实例。
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
# if '''some basic test.''' and __DEBUG_ME__() and True:
#     input = torch.tensor([1., 2., 3.], requires_grad=True)
#     factor_for_path_1 = torch.tensor([0.1])
#     factor_for_path_2 = torch.tensor([0.01])
#     output = Grad_Balancer_2out_Function.apply(input, factor_for_path_1, factor_for_path_2)
#     print(output, "output")
#     g_in = torch.ones_like(output)
#     torch.autograd.backward(output, g_in,inputs= input)
#     print(input.grad, "grad")
#     pass




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

#     pass # class
# if '''some basic test.''' and __DEBUG_ME__() and True:
#     layer = Grad_Balancer_2out(0.1, 0.02)
#     input = torch.tensor([1., 2., 3.], requires_grad=True)
#     output = layer(input)
#     print(output, "output")
#     g_in = torch.ones_like(output)
#     torch.autograd.backward(output, g_in,inputs= input)
#     print(input.grad, "grad")
#     pass



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
# if '''some basic test.''' and __DEBUG_ME__() and True:
#     input = torch.tensor([1., 2.], requires_grad=True)
#     factor = torch.tensor([0.1, 0.02, 0.003])
#     output = Grad_Balancer_Function.apply(input, factor)
#     print(output, "output")
#     g_in = torch.ones_like(output)
#     torch.autograd.backward(output, g_in,inputs= input)
#     print(input.grad, "grad")

#     input = torch.tensor([[1., 2.], [3., 4.], ], requires_grad=True)
#     factor = torch.tensor([0.1, 0.02, 0.003])
#     output = Grad_Balancer_Function.apply(input, factor)
#     print(output, "output")
#     g_in = torch.ones_like(output)
#     torch.autograd.backward(output, g_in,inputs= input)
#     print(input.grad, "grad")
#     pass




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
# if '''some basic test.''' and __DEBUG_ME__() and True:
#     factor = torch.tensor([0.1, 0.02, 0.003])
#     layer = Grad_Balancer(factor)
#     input = torch.tensor([1., 2.], requires_grad=True)
#     output = layer(input)
#     print(output, "output")
#     g_in = torch.ones_like(output)
#     torch.autograd.backward(output, g_in,inputs= input)
#     print(input.grad, "grad")
#     pass






# 应该是过时了，以前gramo没有维度自适应的，现在有了，这个就不用了。
# def init_weight_vec_len_maintaining(in_features:int, out_features:int)->Tuple[torch.Tensor, float]:
#     '''output list:
#     >>> weight:torch.Tensor
#     >>> recommended scaling ratio for gramo after this weight:float
    
#     The reason:
    
#     This init only provides weight (no bias). If the input x has a vector-length of 1., 
#     after matmul ,it's still very close to 1. Unless the dim is very small.
#     >>> in_features = 300
#     >>> out_features = 400
#     >>> for _ in range(5):
#     >>>     input_temp = torch.rand([1,in_features, 1])
#     >>>     length_of_input_temp = input_temp.mul(input_temp).sum().sqrt()
#     >>>     input = input_temp/length_of_input_temp
#     >>>     debug_checks_the_length = input.mul(input).sum()
#     >>>     the_factor = math.sqrt(3.)/math.sqrt(out_features)#*in_features)
#     >>>     w = (torch.rand([out_features, in_features])*2.-1.)*the_factor
#     >>>     output = w.matmul(input)
#     >>>     print(output.mul(output).sum())
#     I want the vector length of output always near to 1.
#     '''
#     sqrt_3 = math.sqrt(3.)
#     the_factor = sqrt_3/math.sqrt(out_features)#*in_features)
    
#     #the_factor = 3./math.sqrt(out_features)#*in_features)
#     result = (torch.rand([out_features, in_features])*2.-1.)*the_factor
#     return result, sqrt_3




def debug_zero_grad_ratio(parameter:torch.nn.parameter.Parameter, \
    print_out:float = False)->float:
    if parameter.grad is None:
        if print_out:
            print(f"{0.}, inside debug_zero_grad_ratio function __line {_line_()}")
            pass
        return 0.
    with torch.no_grad():
        result = 0.
        if not parameter.grad is None:
            flags = parameter.grad.eq(0.)
            total_amount = flags.sum().item()
            result = float(total_amount)/parameter.nelement()
        if print_out:
            print("get_zero_grad_ratio:", result)
        return result
    
def debug_strong_grad_ratio(parameter:torch.nn.parameter.Parameter, log10_diff = 0., \
            epi_for_w = 0.01, epi_for_g = 0.01, print_out = False)->float:
    r'''the log10_diff should be approximately calculated like, 
    >>> log10(planned_epoch * learning_rate)
    I my test, I usually plan <3k epoch, and use 0.001 as lr, 
    so the default value for log10_diff  is 0.'''
    #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
    if parameter.grad is None:
        if print_out:
            print(0., "inside debug_strong_grad_ratio function __line 1082")
            pass
        return 0.

    the_device=parameter.device
    epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
    raw_weight_abs = parameter.abs()
    flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

    epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
    raw_weight_grad_abs = parameter.grad.abs()
    flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

    ten = torch.tensor([10.], device=the_device)
    log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
    corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
    flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

    flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
    result = (flag_useful_g.sum().to(torch.float32)/parameter.nelement()).item()
    if print_out:
        print(result, "inside debug_micro_grad_ratio function __line 1082")
        pass
    return result







"a performance test. in pytorch, int tensor comparison is the same speed as tensor tensor comparison."
if "perf test." and __DEBUG_ME__() and False:
    '''result. in pytorch, int tensor comparison is the same speed as tensor tensor comparison.'''
    def func_a_t():
        input = torch.rand(size=(100,100), device='cuda')
        a_i:int = input.nelement()
        a_t:torch.Tensor = torch.tensor(a_i, device='cuda')
        the_sum_t:torch.Tensor = input.gt(2.).sum()
        for _ in range(20):
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            pass
        pass
    time_of_tensor_ver, _log = timeit(func_a_t,time_at_most=2., _debug__provides_log = True)

    def func_a_i():
        input = torch.rand(size=(100,100), device='cuda')
        a_i:int = input.nelement()
        a_t:torch.Tensor = torch.tensor(a_i, device='cuda')
        the_sum_t:torch.Tensor = input.gt(2.).sum()
        for _ in range(20):
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            pass
        pass
    time_of_int_ver,_ = timeit(func_a_i,time_at_most=2.)

    def func_empty():
        input = torch.rand(size=(100,100), device='cuda')
        a_i:int = input.nelement()
        a_t:torch.Tensor = torch.tensor(a_i, device='cuda')
        the_sum_t:torch.Tensor = input.gt(2.).sum()
        for _ in range(20):
            pass
        pass
    time_of_empty,_ = timeit(func_empty,time_at_most=2.)
    print("int ver:  ", time_of_int_ver-time_of_empty)
    print("tensor ver:  ", time_of_tensor_ver-time_of_empty)
    pass



    

def get_mask_of_top_element__rough(input:torch.Tensor, top_ratio = 0.9, error_of_percent__at_least = 0.01, \
                            bottom = False, careful_level:int = 3, epsilon:float|torch.Tensor|None = None, \
                                _debug_needs_log = False)->tuple[torch.Tensor, list[str]]:
    ''' 
    重新整理一下思路
    这个函数有2个退出模式。
    1是，准确的找到了所需要的比例。比例的目标区间本身是越来越宽的。
    2是，类似二分查找的那种上下限，如果距离足够近，也就退出了。
    核心思想就是，
    目标，和error，算出允许的比例的上下限  top_ratio  error   at_least/most_this_amount 
    二分查找的那个标准是最大值和最小值的平均值，然后根据需要的方向来缩小。   max, min,(threshold), epi
    
    
    return shape is the same as input. dtype of return is torch.bool.
    
    if shape is too small, this may not work.
    
    shape is [*B*atch, *I*nput]
    '''
    assert top_ratio>0.
    assert top_ratio<1.
    assert error_of_percent__at_least>=0.
    assert careful_level>0
    assert careful_level<64, "or modify the data type. search for repeating__b = torch.zeros_like(_the_max_to_calc_threshold__b, dtype=torch.int8)"
    if epsilon:
        if isinstance(epsilon, float):
            epsilon = torch.tensor(epsilon, device=input.device, dtype=input.dtype)
            pass
        else:
            assert epsilon is torch.Tensor
            assert epsilon
            epsilon = epsilon.to(input.dtype)        
        pass
    if _debug_needs_log:
        _log:list[str] = [f"epsilon:{epsilon.item()}"]
        pass
    else:
        _log = None  
        pass
    
    #dtype uint
    #best dtype for count the amount.
    _total_nelement = input[0].nelement()
    if _total_nelement<=(1<<8):
        uint_dtype = torch.uint8
        pass
    elif _total_nelement<=(1<<16):
        uint_dtype = torch.uint16
        pass
    elif _total_nelement<=(1<<32):
        uint_dtype = torch.uint32
        pass
    else:
        uint_dtype = torch.uint64
        pass
    if _debug_needs_log:
        _log.append(f"uint type:{str(uint_dtype)}")
        pass
    # device = input.device
    # param_factory = {"device":device, "dtype":dtype}
    #dtype int
    if _total_nelement<=(1<<7):
        int_dtype = torch.int8
        pass
    elif _total_nelement<=(1<<15):
        int_dtype = torch.int16
        pass
    elif _total_nelement<=(1<<31):
        int_dtype = torch.int32
        pass
    else:
        int_dtype = torch.int64
        pass
    if _debug_needs_log:
        _log.append(f"int type:{str(int_dtype)}")
        pass
    
    with torch.no_grad():
        #into torch.
        careful_level__s:torch.Tensor = torch.tensor(careful_level, device=input.device)
        del careful_level
        if bottom:
            top_ratio = 1.- top_ratio
            pass
        top_ratio__s = torch.tensor(top_ratio, dtype=torch.float64, device=input.device)
        del top_ratio
        
        if _debug_needs_log:
            _log.append(f"top ratio:{top_ratio}, is bottom:{bottom}")
            pass
        
        
        #init error_of_percent 
        better_error_of_percent = 0.501/input.nelement()
        if better_error_of_percent<error_of_percent__at_least:
            better_error_of_percent = error_of_percent__at_least
            pass
        del error_of_percent__at_least
        error_of_percent__b = torch.empty(size=[input.shape[0]], device=input.device)
        error_of_percent__b.fill_(better_error_of_percent)
        if _debug_needs_log:
            _log.append(f"error_of_percent__b init to:{error_of_percent__b}")
            pass
        
        #ratio+-error, this segment appears twice in this function.
        at_least_this_amount__b = ((input.nelement()-1)*(top_ratio__s - error_of_percent__b)).to(int_dtype)
        at_most_this_amount__b =  ((input.nelement()-1)*(top_ratio__s + error_of_percent__b)).to(int_dtype)
        if _debug_needs_log:
            _log.append(f"at_least_this_amount__b init to:{at_least_this_amount__b}")
            _log.append(f"at_most_this_amount__b init to:{at_most_this_amount__b}")
            pass
        
        #safety, or maybe a early return.
        _flag_all_true_early_return__b = at_least_this_amount__b.ge(input.nelement())
        if _flag_all_true_early_return__b.all():
            _temp_tensor = torch.ones_like(input, dtype=torch.bool, device=input.device)
            return _temp_tensor, _log
        _flag_all_true_early_return__b = at_most_this_amount__b.le(0)
        if _flag_all_true_early_return__b.all():
            _temp_tensor = torch.zeros_like(input, dtype=torch.bool, device=input.device)
            return _temp_tensor, _log
        
        #maybe optimizable. reverse+reverse = nothing.
        if_finished__b = (_flag_all_true_early_return__b).logical_or(_flag_all_true_early_return__b)
        if _debug_needs_log:
            _log.append(f"if_finished__b init to:{if_finished__b}")
            pass
        
        # init before loop
        _the_max_to_calc_threshold__b:torch.Tensor = input.max(dim=1).values
        _the_min_to_calc_threshold__b:torch.Tensor = input.min(dim=1).values
        if input.dtype != torch.float64 and input.dtype != torch.float32:
            _the_max_to_calc_threshold__b.to(torch.float16)
            _the_min_to_calc_threshold__b.to(torch.float16)
            pass
        if _debug_needs_log:
            _log.append(f"_the_max_to_calc_threshold__b init to:{_the_max_to_calc_threshold__b}")
            _log.append(f"_the_min_to_calc_threshold__b init to:{_the_min_to_calc_threshold__b}")
            pass
        
        repeating__b = torch.zeros_like(_the_max_to_calc_threshold__b, dtype=torch.int8)
        old_unqualified_RESULT__b_i = torch.zeros_like(input,dtype=torch.bool)
        # now is this one: if_finished__b
        # it was init_ed_the_flag_result
        if _debug_needs_log:
            _log.append(f"repeating__b init to:{repeating__b}")
            _log.append(f"old_unqualified_RESULT__b_i init to:{old_unqualified_RESULT__b_i}")
            pass
        
        _input_gt_guess__count__b = torch.zeros_like(if_finished__b, dtype=uint_dtype)
        if _debug_needs_log:
            _log.append(f"_input_gt_guess__count__b init to:{_input_gt_guess__count__b}")
            pass
        if _debug_needs_log:
            loop_count = 0
            pass
        
        while True:
            if _debug_needs_log:
                _log.append(f"----  loop {loop_count}  ----")
                pass
            1w 好吧，无语了，写log了。
            #similar to binary search
            _guess_threshold = torch.zeros_like(if_finished__b,dtype=_the_max_to_calc_threshold__b.dtype)
            _guess_threshold[~if_finished__b] = (_the_max_to_calc_threshold__b[~if_finished__b]+_the_min_to_calc_threshold__b[~if_finished__b])/2.#maybe optimizable.

            #the real comparison
            RESULT_input_gt_guess__b_i___mask_if_finish = torch.zeros_like(input, dtype=torch.bool)
            RESULT_input_gt_guess__b_i___mask_if_finish[~if_finished__b] = input[~if_finished__b].gt(_guess_threshold[~if_finished__b])
            #if guessed too big, then, less true
            _input_gt_guess__count__b[~if_finished__b] = RESULT_input_gt_guess__b_i___mask_if_finish[~if_finished__b].to(uint_dtype).sum(dim=1)
            #_guess_count = flag_result.to(int_dtype).sum(dim=1)
            
            # #flag_gt
            # _if__guess_not_too_big___b = torch.zeros_like(if_finished__b)
            # _if__guess_not_too_big___b[~if_finished__b] = _guess_count__b[~if_finished__b].le(at_most_this_amount__b[~if_finished__b])
            # # ^^^ true is good. ^^^
            # _the_min_to_calc_threshold__b[~_if__guess_not_too_big___b] = _guess_threshold[~_if__guess_not_too_big___b]
            
            #flag_gt
            _if__guess_too_big___b = torch.zeros_like(if_finished__b)
            _if__guess_too_big___b[~if_finished__b] = _input_gt_guess__count__b[~if_finished__b].lt(at_least_this_amount__b[~if_finished__b])
            # ^^^ true is bad. ^^^
            _the_max_to_calc_threshold__b[_if__guess_too_big___b] = _guess_threshold[_if__guess_too_big___b]
            
            
            # #flag_lt
            # _if__guess_not_too_small___b = torch.zeros_like(if_finished__b)
            # _if__guess_not_too_small___b[~if_finished__b] = _guess_count__b[~if_finished__b].ge(at_least_this_amount__b[~if_finished__b])
            # # ^^^ true is good. ^^^
            # _the_max_to_calc_threshold__b[~_if__guess_not_too_small___b] = _guess_threshold[~_if__guess_not_too_small___b]
            
            #flag_lt
            _if__guess_too_small___b = torch.zeros_like(if_finished__b)
            _if__guess_too_small___b[~if_finished__b] = _input_gt_guess__count__b[~if_finished__b].gt(at_most_this_amount__b[~if_finished__b])
            # ^^^ true is bad. ^^^
            _the_min_to_calc_threshold__b[_if__guess_too_small___b] = _guess_threshold[_if__guess_too_small___b]
            
            
            _flag__not_too_loose__and__not_too_tight = (~_if__guess_too_big___b) and (~_if__guess_too_small___b)
            # ^^^ true is good. ^^^                       ^^^ true is bad. ^^^          ^^^ true is bad. ^^^  
            if_finished__b.logical_or_(_flag__not_too_loose__and__not_too_tight)
            
            if epsilon is not None:
                _flag_less_than_epsilon = (_the_max_to_calc_threshold__b-_the_min_to_calc_threshold__b).lt(epsilon)
                if_finished__b.logical_or_(_flag_less_than_epsilon)
                pass#if epsilon
            
            # this is the only [return] timing.
            if if_finished__b.all():
                if bottom:
                    RESULT_input_gt_guess__b_i___mask_if_finish.logical_not_()
                    pass
                return RESULT_input_gt_guess__b_i___mask_if_finish
                pass #if if_finished__b.all():
            
            
            #if the new result[b,i] unchanged?
            _if__unchanged__b = torch.zeros_like(if_finished__b, dtype=torch.bool)
            _if__unchanged__b[~if_finished__b] = old_unqualified_RESULT__b_i[~if_finished__b].eq( \
                RESULT_input_gt_guess__b_i___mask_if_finish[~if_finished__b]).all(dim=1)
            # ^^^ true is bad. ^^^
            repeating__b[_if__unchanged__b].add_(1)
            
            #if 
            _if__repeated_enough__b = repeating__b.ge(careful_level__s)
            repeating__b[_if__repeated_enough__b] = 0
            #update the finishing flags.
            error_of_percent__b[_if__repeated_enough__b].mul_(2.)#this 2. is not tested.
            #maybe wrong??? is it updated?
            
            #ratio+-error, this segment appears twice in this function.
            #[1]+[] is []. So this is safe.
            at_least_this_amount__b[_if__repeated_enough__b] = ((input.nelement()-1)*(top_ratio__s - \
                error_of_percent__b[_if__repeated_enough__b])).to(int_dtype)
            at_most_this_amount__b[_if__repeated_enough__b] =  ((input.nelement()-1)*(top_ratio__s + \
                error_of_percent__b[_if__repeated_enough__b])).to(int_dtype)
            # no detect for return here. reason:
            # even if this range-like can mean a range covering all the range, bc I believe it unlikely to happen.
            # I decide to delay the return to the next round.
            
            #tail
            old_unqualified_RESULT__b_i = RESULT_input_gt_guess__b_i___mask_if_finish
            if _debug_needs_log:
                loop_count += 1
                pass
            pass#while true
        
        pass#  no_grad
    pass# end of function
    
if "test" and __DEBUG_ME__() and True:
    # torch.topk is not what I need.
    # fdsfds = torch.topk(torch.tensor([1,2,3,4,5]),3, sorted=False)
    # fdsfds2 = torch.topk(torch.tensor([1,2,3,4,5]),3, sorted=True)
    
    n = 5
    top_ratio = 0.1
    error_of_percent = 0.1
    _floor_offset = 0.
    lower_bound = (n-1)*(top_ratio - error_of_percent)+_floor_offset
    upper_bound = (n-1)*(top_ratio + error_of_percent)+_floor_offset
    #assert int(lower_bound) == int(upper_bound)
    #it's possible they equal. 
    
    a1 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.1)
    a2 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.2)
    a3 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.3)
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.1).eq(torch.tensor([False,False,False,False,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.2).eq(torch.tensor([False,False,False,True,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3,4,1]]),top_ratio=0.1).eq(torch.tensor([True,False,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3,4,1]]),top_ratio=0.2).eq(torch.tensor([True,False,False,True,False])).all()
    
    
    
    n = 5
    top_ratio = 0.1
    error_of_percent = 0.1
    _floor_offset = 0.
    lower_bound = (n-1)*(top_ratio - error_of_percent)+_floor_offset
    upper_bound = (n-1)*(top_ratio + error_of_percent)+_floor_offset
    #1w继续。
    
    b1 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.1,bottom=True)
    b2 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.2,bottom=True)
    b3 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.3,bottom=True)
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.1,bottom=True).eq( torch.tensor([True,False,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.2,bottom=True).eq( torch.tensor([True,True,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3,4,1]]),top_ratio=0.1,bottom=True).eq( torch.tensor([False,False,False,False,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3,4,1]]),top_ratio=0.1,bottom=True).eq( torch.tensor([False,True,False,False,True])).all()
    
    # assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.  ).eq(torch.tensor([False,False,False])).all()
    # assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.33).eq(torch.tensor([False,False,True ])).all()
    # assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.66).eq(torch.tensor([False,True ,True ])).all()
    # assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=1.  ).eq(torch.tensor([True ,True ,True ])).all()

    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0. , bottom=True).eq(torch.tensor([False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.4, bottom=True).eq(torch.tensor([True ,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=1. , bottom=True).eq(torch.tensor([True ,True ,True ])).all()

    for _shift_by in [6,7,8,9,15,16,17]:
        _1_left_shift_with_True = torch.empty(size=(1<<_shift_by ,), dtype=torch.bool)
        _1_left_shift_with_True.fill_(value=True)
        for dtype in [torch.float16, torch.float32, torch.float64]:
            _the_sum = get_mask_of_top_element__rough(torch.rand(size=(1<<_shift_by ,),dtype=dtype),0.99).sum()
            assert _the_sum>(1<<_shift_by)*0.97 and _the_sum<(1<<_shift_by)
        pass

    #when the ratio doesn't exist.
    #if this test returns, it's good.
    get_mask_of_top_element__rough(torch.tensor([[1.,1,1,1,1]]),top_ratio=0.5 , bottom=True)
    

    #gpu
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]],device='cuda'),top_ratio=0.33).eq(torch.tensor([False,False,True ],device='cuda')).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]],device='cuda'),top_ratio=0.33).device.type == 'cuda'
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]],device='cuda'),top_ratio=0.).eq(torch.tensor([False,False,False ],device='cuda')).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]],device='cuda'),top_ratio=0.).device.type == 'cuda'
    
    
    the_linspace = torch.linspace(1.,100.,99)
    the_linspace = the_linspace.reshape([1,-1])
    _temp_int = get_mask_of_top_element__rough(the_linspace,top_ratio=0.2, error_of_percent__at_least=0.01).sum().item()
    assert _temp_int>18 and _temp_int<22
    _temp_int = get_mask_of_top_element__rough(the_linspace,top_ratio=0.2, error_of_percent__at_least=0.1).sum().item()
    assert _temp_int>8 and _temp_int<32
    
    
    # careful_level
    the_tensor = torch.linspace(1.,100.,99)
    the_tensor = the_tensor.reshape([1,-1])
    the_tensor[-1] = 99999
    #step into the function and see how it works.
    get_mask_of_top_element__rough(the_tensor, top_ratio=0.5, error_of_percent__at_least=0.01, careful_level = 1)
    #1w 写一下会发生什么。
    #1w 整个这个里面的都还没测试。。。。
    
    # epsilon
    the_tensor = torch.rand([1,10])
    get_mask_of_top_element__rough(the_tensor, top_ratio=0.5, error_of_percent__at_least=0.01, epsilon=0.001)
    get_mask_of_top_element__rough(the_tensor, top_ratio=0.5, error_of_percent__at_least=0.01, epsilon=0.1)
    get_mask_of_top_element__rough(the_tensor, top_ratio=0.5, error_of_percent__at_least=0.01, epsilon=0.51)
    
    # batch
    the_tensor = torch.tensor( [[1.,2,3,4,5],
                                [5.,2,3,4,1]])
    the_result = torch.tensor([[False,False,False,False,True],
                                [True,False,False,False,False]])
    assert get_mask_of_top_element__rough(the_tensor, top_ratio=0.2).eq(the_result).all()
    
    
    
    
    
    
    pass



def 应该是不用了get_top_percent如果没改就不要了_上面已经搞定了(input:torch.Tensor, top_ratio = 0.5, error_of_percent = 0.01, \
                            bottom = False)->torch.Tensor:
    ''' 
    return shape is the same as input. dtype of return is torch.bool.
    
    if shape is too small, this may not work.
    '''
    assert input.shape.__len__()==2
    nelement_per_batch__s = input.shape[1]
    with torch.no_grad():
        #safety first
        _at_least_this_amount__cpu_int = int(nelement_per_batch__s*(top_ratio - error_of_percent)+0.4999999999999)
        at_least_this_amount__s = torch.tensor(_at_least_this_amount__cpu_int, device=input.device)
        _at_most_this_amount__cpu_int =  int(nelement_per_batch__s*(top_ratio + error_of_percent)+0.4999999999999)
        at_most_this_amount__s =  torch.tensor(_at_most_this_amount__cpu_int, device=input.device)
        # if at_least_this_amount == at_most_this_amount: xxxxxxxxxxxxxx
        #     at_most_this_amount = at_least_this_amount  +1
        #     pass
        if _at_least_this_amount__cpu_int >= nelement_per_batch__s:
            _temp_tensor = torch.ones_like(input, dtype=torch.bool, device=input.device)
            return _temp_tensor
        if _at_most_this_amount__cpu_int <= 0.:
            _temp_tensor = torch.zeros_like(input, dtype=torch.bool, device=input.device)
            return _temp_tensor
        assert error_of_percent>=0.
        
        #real job.
        #best dtype for count the amount.
        if nelement_per_batch__s<=(1<<8):
            dtype = torch.uint8
            pass
        elif nelement_per_batch__s<=(1<<16):
            dtype = torch.uint16
            pass
        elif nelement_per_batch__s<=(1<<32):
            dtype = torch.uint32
            pass
        else:
            dtype = torch.uint64
            pass
        # device = input.device
        # param_factory = {"device":device, "dtype":dtype}
        
        #init before loop
        _the_max_threshold__b:torch.Tensor = input.max(dim=1).values.to(torch.float64)
        _the_min_threshold__b:torch.Tensor = input.min(dim=1).values.to(torch.float64)
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        #1w 加一个flag
        while True:
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            #1w 加一个强制退出条件。
            _guess_threshold = (_the_max_threshold__b+_the_min_threshold__b)/2.
            if bottom:
                flag_result = input.lt(_guess_threshold)
                _guess_count = flag_result.to(dtype).sum()
                if _guess_count>at_most_this_amount__s:
                    _the_max_threshold__b = _guess_threshold
                    pass
                elif _guess_count<at_least_this_amount__s:
                    _the_min_threshold__b = _guess_threshold
                    pass
                else:
                    return flag_result
                pass#if bottom:
            else:#top
                flag_result = input.gt(_guess_threshold)
                _guess_count = flag_result.to(dtype).sum()
                if _guess_count>at_most_this_amount__s:
                    _the_min_threshold__b = _guess_threshold
                    pass
                elif _guess_count<at_least_this_amount__s:
                    _the_max_threshold__b = _guess_threshold
                    pass
                else:
                    return flag_result
                pass#top
                
            pass#while
        pass#  no_grad
    pass# end of function
    

















def debug_avg_log10___no_batch_dim(input:torch.Tensor, top_ratio = 0.9)->float:
    '''I dont really remember this function too much. 
    
    I believe it provides a avg of log10. 
    
    A lot lines in here is safety.
    If the data has a lot elements very close to 0., this function may not be very helpful.
    '''
    the_log = input.abs().log10()
    #safety
    flag_inf = the_log.isinf()
    count_of_nan_and_inf = the_log.isnan().sum()+flag_inf.sum()
    no_nan_log = the_log.nan_to_num(0.)
    safe_log = flag_inf.logical_not()*no_nan_log
    #safe_log is safe.
    flag_non_trivial_log = get_mask_of_top_element__rough(safe_log, top_ratio)
    
    assert False
    
    numerator = safe_log.sum()
    nelement = input.nelement()
    result = numerator/(nelement-count_of_nan_and_inf)
    return result.item()

if "test" and __DEBUG_ME__() and True:
    input = torch.tensor([torch.nan, torch.inf, torch.inf*-1])
    assert math.isnan(debug_avg_log10___no_batch_dim(input))
    input = torch.tensor([torch.nan, torch.inf, torch.inf*-1, 0., -100])
    assert _float_equal(debug_avg_log10___no_batch_dim(input), 2.)
    input = torch.tensor([1e10,-1e20])
    assert _float_equal(debug_avg_log10___no_batch_dim(input), 15.)
    
    assert False, "jixu batch还没有。"
    input = torch.tensor([1,1,1,1,1,11e10,-1e20], top_percent = True)
    assert _float_equal(debug_avg_log10___no_batch_dim(input), 15.)
    
    
    pass






# backup code.
    # def debug_get_zero_grad_ratio(self, directly_print_out:float = False)->float:
    #     with torch.no_grad():
    #         result = 0.
    #         if not self.raw_weight_o_i.grad is None:
    #             flags = self.raw_weight_o_i.grad.eq(0.)
    #             total_amount = flags.sum().item()
    #             result = float(total_amount)/self.raw_weight_o_i.nelement()
    #         if directly_print_out:
    #             print("get_zero_grad_ratio:", result)
    #         return result


    # def debug_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01, \
    #                             print_out = False)->float:
    #     #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
    #     if self.raw_weight_o_i.grad is None:
    #         if print_out:
    #             print(0., "inside debug_micro_grad_ratio function __line 1082")
    #             pass
    #         return 0.

    #     the_device=self.raw_weight_o_i.device
    #     epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
    #     raw_weight_abs = self.raw_weight_o_i.abs()
    #     flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

    #     epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
    #     raw_weight_grad_abs = self.raw_weight_o_i.grad.abs()
    #     flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

    #     ten = torch.tensor([10.], device=the_device)
    #     log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
    #     corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
    #     flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

    #     flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
    #     result = (flag_useful_g.sum().to(torch.float32)/self.raw_weight_o_i.nelement()).item()
    #     if print_out:
    #         print(result, "inside debug_micro_grad_ratio function __line 1082")
    #         pass
    #     return result


    # def debug_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01, \
    #                             print_out = False)->float:
    #     #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
    #     if self.raw_weight_o_i.grad is None:
    #         if print_out:
    #             print(0., "inside debug_micro_grad_ratio function __line 1082")
    #             pass
    #         return 0.

    #     the_device=self.raw_weight_o_i.device
    #     epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
    #     raw_weight_abs = self.raw_weight_o_i.abs()
    #     flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

    #     epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
    #     raw_weight_grad_abs = self.raw_weight_o_i.grad.abs()
    #     flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

    #     ten = torch.tensor([10.], device=the_device)
    #     log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
    #     corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
    #     flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

    #     flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
    #     result = (flag_useful_g.sum().to(torch.float32)/self.raw_weight_o_i.nelement()).item()
    #     if print_out:
    #         print(result, "inside debug_micro_grad_ratio function __line 1082")
    #         pass
    #     return result















def make_grad_noisy(model:torch.nn.Module, noise_base:float = 1.5):
    for p in model.parameters():
        if p.requires_grad and (not p.grad is None):
            temp = torch.randn_like(p.grad)
            noise_factor = torch.pow(noise_base, temp)
            with torch.no_grad():
                #p.grad = p.grad.detach().clone().mul(noise_factor)
                p.grad = p.grad.detach().mul(noise_factor)
                pass
            pass
        pass
    pass

# p = torch.nn.Parameter(torch.tensor([42.]))
# p.grad = torch.tensor([1.])
# p.grad = p.grad.detach().clone().mul(torch.tensor([1.23]))
# print(p.grad)
# fds=432


import sys
# def __line__int():
#     return sys._getframe(1).f_lineno
def __line__str():
    return "    Line number: "+str(sys._getframe(1).f_lineno)
#print('This is line', __line__())



def debug_Rank_1_parameter_to_List_float(input:torch.nn.parameter.Parameter)->List[float]:
    result : List[float] = []
    for i in range(input.shape[0]):
        result.append(input[i].item())
        pass
    return result
# p = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
# l = debug_Rank_1_parameter_to_List_float(p)
# print(p)
# print(l)
# fds=432






#part 1 data gen

def int_into_floats(input:torch.Tensor, bit_count:int, is_output_01:bool)->torch.Tensor:
    if len(input.shape)!=2 or input.shape[1]!=1:
        raise Exception("Param:input must be rank-2. Shape is [batch, 1].")
    
    mask = torch.logspace(0,bit_count-1,bit_count, base=2, dtype=torch.int64)
    mask = mask.to(input.device)
    result = input[:,].bitwise_and(mask)
    result = result.to(torch.bool)
    result = result.to(torch.float32)
    if not is_output_01:
        result = result*2.-1.
    return result
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
    input = torch.tensor([[0],[1],[2],[3],[7],])
    print(int_into_floats(input,7,True))
    print(int_into_floats(input,7,False))
    pass



def int_into_floats_with_str(input:torch.Tensor, bit_count:int, is_output_01:bool)->torch.Tensor:
    if len(input.shape)!=2 or input.shape[1]!=1:
        raise Exception("Param:input must be rank-2. Shape is [batch, 1].")
    
    mask = torch.logspace(0,bit_count-1,bit_count, base=2, dtype=torch.int64)
    mask = mask.to(input.device)
    result = input[:,].bitwise_and(mask)
    result = result.to(torch.bool)
    result = result.to(torch.float32)
    if not is_output_01:
        result = result*2.-1.
        pass
    result *= mask/mask[-1]
    return result
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
    input = torch.tensor([[0],[1],[2],[3],[7],])
    print(int_into_floats_with_str(input,4,True))
    print(int_into_floats_with_str(input,4,False))
    fds=432



def floats_into_int(input:torch.Tensor)->torch.Tensor:
    if len(input.shape)!=2:
        raise Exception("Param:input must be rank-2. Shape is [batch, -1].")
    
    bit_count = input.shape[1]
    mask = torch.logspace(0,bit_count-1,bit_count, base=2, dtype=torch.int64)
    mask = mask.unsqueeze(dim=1)
    mask = mask.to(torch.float32)
    #input = input.gt(0.5)
    input = input.gt(0.)
    input = input.to(torch.float32)
    result = input.matmul(mask)
    result = result.to(torch.int64)
    return result
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
    input = torch.tensor([[0],[1],[2],[3],[7],])
    input = int_into_floats(input,7, True)
    print(floats_into_int(input).T)
    input = torch.tensor([[0],[1],[2],[3],[7],])
    input = int_into_floats_with_str(input,7, True)
    print(floats_into_int(input).T)
    input = torch.tensor([[0],[1],[2],[3],[7],])
    input = int_into_floats(input,7, False)
    print(floats_into_int(input).T)
    input = torch.tensor([[0],[1],[2],[3],[7],])
    input = int_into_floats_with_str(input,7, False)
    print(floats_into_int(input).T)
    fds=432


def data_gen_for_directly_stacking_test(batch:int, n_in:int, n_out:int, dtype = torch.float32, is_input_01 = False,\
        no_duplicated = True)->Tuple[torch.Tensor, torch.Tensor]:
    input = torch.randint(0,2,[batch, n_in],dtype = dtype)
    if not is_input_01:
        input = input*2-1
        pass
    answer_index = torch.randint(0,n_in,[n_out])
    if n_in<n_out and no_duplicated:
        raise Exception("more out from less in, it's always duplicating.")
    if no_duplicated:
        while answer_index.shape[0]!= answer_index.unique().shape[0]:
            answer_index = torch.randint(0,n_in,[n_out])
            pass
        pass
    target = input[:, answer_index]
    return input, target
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
    a,b = data_gen_for_directly_stacking_test(5,3,2)
    print(a)
    print(b)
    a,b = data_gen_for_directly_stacking_test(5,3,2, no_duplicated=True)
    fds=423



def data_gen_for_directly_stacking_test_same_dim_no_duplicated(\
        batch:int, dim:int, dtype = torch.float32, is_input_01 = False)->Tuple[torch.Tensor, torch.Tensor]:
    input = torch.randint(0,2,[batch, dim],dtype = dtype)
    if not is_input_01:
        input = input*2-1
        pass
    answer_index:torch.Tensor = torch.linspace(0,dim-1,dim, dtype=torch.int64)
    for _ in range(dim+torch.randint(0,dim,[1]).item()):
        rand_i = torch.randint(0,dim,[1])
        rand_ii = torch.randint(0,dim,[1])
        temp = answer_index[rand_i]
        answer_index[rand_i] = answer_index[rand_ii]
        answer_index[rand_ii] = temp
        pass
    target = input[:, answer_index]
    return input, target
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
    # a,b = data_gen_for_directly_stacking_test_same_dim_no_duplicated(5,3)
    # print(a)
    # print(b)
    # a,b = data_gen_for_directly_stacking_test(5,3,2, no_duplicated=True)
    # fds=423



def data_gen_half_adder_1bit(batch:int, is_output_01:bool, is_cuda:bool=True):#->Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randint(0,2,[batch,1])
    b = torch.randint(0,2,[batch,1])
    if is_cuda:
        a = a.cuda()
        b = b.cuda()
    target = a+b
    a = int_into_floats(a,1, is_output_01)    
    b = int_into_floats(b,1, is_output_01)        
    input = torch.concat([a,b], dim=1)
    #input = input.requires_grad_()
    target = int_into_floats(target,2, is_output_01)    

    return (input, target)
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
# '''half_adder_1bit_data_gen'''    
# (input, target) = data_gen_half_adder_1bit(3, True)
# print(input)
# print(input.shape)
# print(target)
# print(target.shape)
# (input, target) = data_gen_half_adder_1bit(3, False)
# print(input)
# print(target)
# fds=432

def data_gen_full_adder(bits:int, batch:int, is_output_01:bool, is_cuda:bool=True):#->Tuple[torch.Tensor, torch.Tensor]:
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
    a = int_into_floats(a,bits, is_output_01)    
    b = int_into_floats(b,bits, is_output_01)      
    c = int_into_floats(c,1, is_output_01)    
    input = torch.concat([a,b,c], dim=1)
    #input = input.requires_grad_()
    target = int_into_floats(target,bits+1, is_output_01)    

    return (input, target)
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
# '''data_gen_full_adder_1bit'''    
# (input, target) = data_gen_full_adder(3,2, True)
# print(input)
# print(input.shape)
# print(target)
# print(target.shape)
# (input, target) = data_gen_full_adder(3,2, False)
# print(input)
# print(target)
# fds=432










# old version.
# def bitwise_acc(a:torch.Tensor, b:torch.Tensor, print_out:bool = False)->float:
#     temp = a.eq(b)
#     temp = temp.sum().to(torch.float32)
#     acc = temp/float(a.shape[0]*a.shape[1])
#     acc_float = acc.item()
#     if print_out:
#         print("{:.4f}".format(acc_float), "<- the accuracy")
#         pass
#     return acc_float
#     pass

def data_gen_from_random_teacher(teacher:torch.nn.Module, input:torch.Tensor)->torch.Tensor:
    output = teacher(input).detach().clone()
    return output


class Debug__LinearTeacher(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, num_layers = 2, mid_width =Optional[int], \
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.layers = torch.nn.ParameterList()
        if 1 == num_layers:
            self.layers.append(torch.nn.Linear(in_features, out_features,bias))
        else:
            self.layers.append(torch.nn.Linear(in_features, mid_width,bias))
            for _ in range(num_layers-2):
                self.layers.append(torch.nn.Linear(mid_width, mid_width,bias))
                pass
            self.layers.append(torch.nn.Linear(mid_width, out_features, bias))
            pass
        pass 
    #end of function
    def forward(self, input_b_i:torch.Tensor) -> torch.Tensor:
        x = input_b_i
        layer:torch.nn.Linear
        for layer in self.layers:
            x = layer(x)
            pass
        return x
    #end of function
    pass
        



def bitwise_acc(a:torch.Tensor, b:torch.Tensor, output_is_01 = False, print_out_when_exact_one = True, \
                print_out:bool = False)->Tuple[float, bool]:
    with torch.no_grad():
        if output_is_01:
            temp = a.gt(0.5) == b.gt(0.5)
        else:
            temp = a.gt(0.) == b.gt(0.)
            pass
        if temp.all():
            if print_out_when_exact_one:
                print(1., "(NO ROUNDING!!!)   <- the accuracy    inside bitwise_acc function __line 859 ")
                pass
            return (1., True)
        temp2 = temp.sum().to(torch.float32)
        acc = temp2/float(a.shape[0]*a.shape[1])
        acc_float = acc.item()
        if print_out:
            print("{:.4f}".format(acc_float), "<- the accuracy")
            pass
        return (acc_float, False)
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
# a = torch.tensor([[1,1,],[1,1,],[1,1,],])
# b = torch.tensor([[1,1,],[1,1,],[1,1,],])
# print(bitwise_acc(a,b, print_out=True))
# b = torch.tensor([[1,1,],[1,1,],[1,-1,],])
# print(bitwise_acc(a,b, print_out=True))
# b = torch.tensor([[-1,-1,],[-1,-1,],[-1,-1,],])
# print(bitwise_acc(a,b, print_out=True))
# fds=432




def bitwise_acc_with_str(a:torch.Tensor, b:torch.Tensor, print_out_when_exact_one = True, \
                print_out:bool = False)->Tuple[float, bool]:
    with torch.no_grad():
        if (a.gt(0.) == b.gt(0.)).all():
            print(1., "(NO ROUNDING!!!)   <- the accuracy    inside bitwise_acc function __line 784 ")
            return (1., True)
        a_b = a*b
        total_weight = a_b.abs().sum()#(dim=0,keepdim=True)
        sum_of_all = a_b.sum()#(dim=0,keepdim=True)
        ratio = ((sum_of_all/total_weight+1.)/2.).item()
        if print_out:
            print("{:.4f}".format(ratio), "<- the accuracy")
            pass
        return (ratio, False)
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
# a = torch.tensor([[1,1,],[1,0.5,],[1,0.1,],])
# b = torch.tensor([[1,1,],[1,1,],[1,1,],])
# bitwise_acc_with_str(a,b, print_out=True)
# b = torch.tensor([[1,1,],[1,1,],[1,-1,],])
# bitwise_acc_with_str(a,b, print_out=True)
# b = torch.tensor([[-1,-1,],[-1,-1,],[-1,-1,],])
# bitwise_acc_with_str(a,b, print_out=True)

# a = torch.tensor([[1.,0.0000000001,]])
# b = torch.tensor([[1,-1,]])
# print(bitwise_acc_with_str(a,b, print_out=True))
# fds=432







# def debug_Rank_1_parameter_to_List_float(input:torch.nn.parameter.Parameter)->List[float]:
#     result : List[float] = []
#     for i in range(input.shape[0]):
#         result.append(input[i].item())
#         pass
#     return result
# # p = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
# # l = debug_Rank_1_parameter_to_List_float(p)
# # print(p)
# # print(l)
# # fds=432





class Print_Timing:
    r'''
    >>> pt = Print_Timing(max_gap = 100, start_with = 0, first = 3, density:float = 4.)
    >>> for i in range(501):
    >>>     if pt.check(i):
    >>>         print(i, end = ", ")
    >>>         pass
    >>>     pass
    The result is 0, 1, 2, 5, 10, 19, 34, 62, 100, 200, 300, 400, 500, 
    '''
    def __init__(self, max_gap = 100, start_with = 0, first = 1, density:float = 1.):
        self.start_with = start_with
        self.first = first
        self.max_gap = max_gap
        
        self.return_true_when:List[float] = []
        the_exp = 0
        if first-start_with-1>0:
            the_exp = math.log10(first-start_with-1)
            pass
        end_log = math.log10(max_gap)
        invert_of_density = 1/float(density)
        while the_exp<end_log:
            self.return_true_when.append(int(math.pow(10, the_exp)))
            the_exp += invert_of_density
            pass
        pass
    #end of function
    
    def check(self, epoch:int)->bool:
        if epoch>=self.max_gap and epoch%self.max_gap==0:
            return True
        
        calibrated_epoch = epoch-self.start_with+1
        if calibrated_epoch<=self.first:
            return True
        if calibrated_epoch in self.return_true_when:
            return True
        return False
    #end of function
            
    pass# end of class
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
# pt = Print_Timing()
# for i in range(501):
#     if pt.check(i):
#         print(i, end = ", ")
#         pass
#     pass



def print_as_np_1(print_me:torch.Tensor):
    flag_pos = print_me.gt(0.).to(torch.float32)
    flag_neg = print_me.lt(0.).to(torch.float32)
    combined = flag_pos-flag_neg
    print(combined)
    pass
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
# a = torch.tensor([-3.,-1,-0.1,0,0.1,1,3])
# print_as_np_1(a)
# fds=432
    
    
    
    
    
def softmax_dim_1_from_yagaodirac(the_tensor:torch.Tensor, epi:Optional[torch.Tensor]=None)->torch.Tensor:
    if the_tensor.shape.__len__()!=2:
        raise Exception("According to my convention, the shape should be [batch, dim].")
    top_raw_element_of_each_row_b_d = the_tensor.amax(dim=1, keepdim=True)
    offset_input_b_d = the_tensor-top_raw_element_of_each_row_b_d
    the_exp_b_d = offset_input_b_d.exp()
    #only positive values.
    sum_of_each_row_b_1 = the_exp_b_d.sum(dim=1, keepdim=True)
    if epi is None:
        if torch.float16 == the_tensor.dtype:
            epi = torch.tensor(1e-3,dtype=torch.float16,device=the_tensor.device)
            pass
        elif torch.float32 == the_tensor.dtype:
            epi = torch.tensor(1e-6,dtype=torch.float32,device=the_tensor.device)
            pass
        else:
            raise Exception("dtype is weird. No implemented for fp64 now.")
    sum_of_each_row__safe__b_1 = sum_of_each_row_b_1.maximum(epi)
    result = the_exp_b_d/sum_of_each_row__safe__b_1
    return result
if "test" and __DEBUG_ME__() and True:
    input = torch.tensor([[0.,1]],dtype=torch.float16)
    print(softmax_dim_1_from_yagaodirac(input))
    print(input.to(torch.float32).softmax(dim=1))
    pass
if "test" and __DEBUG_ME__() and True:
    dummy = torch.tensor([[0,1]],dtype=torch.int64)
    import random
    input = torch.randn((random.randint(2,5),random.randint(2,5)),dtype=torch.float16)
    print(softmax_dim_1_from_yagaodirac(input))
    print(input.to(torch.float32).softmax(dim=1))
    pass

