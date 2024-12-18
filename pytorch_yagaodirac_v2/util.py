from typing import Any, List, Tuple, Optional, Self
import torch
import math



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
# '''some basic test.'''
# input = torch.tensor([[0.,0.],[0.,1.],[1.,1.]])
# output = vector_length_norm(input)
# print(output)
# print(output.mul(output).sum(dim=1))
# fds=432



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

    pass # class
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







def init_weight_vec_len_maintaining(in_features:int, out_features:int)->Tuple[torch.Tensor, float]:
    '''output list:
    >>> weight:torch.Tensor
    >>> recommended scaling ratio for gramo after this weight:float
    
    The reason:
    
    This init only provides weight (no bias). If the input x has a vector-length of 1., 
    after matmul ,it's still very close to 1. Unless the dim is very small.
    >>> in_features = 300
    >>> out_features = 400
    >>> for _ in range(5):
    >>>     input_temp = torch.rand([1,in_features, 1])
    >>>     length_of_input_temp = input_temp.mul(input_temp).sum().sqrt()
    >>>     input = input_temp/length_of_input_temp
    >>>     debug_checks_the_length = input.mul(input).sum()
    >>>     the_factor = math.sqrt(3.)/math.sqrt(out_features)#*in_features)
    >>>     w = (torch.rand([out_features, in_features])*2.-1.)*the_factor
    >>>     output = w.matmul(input)
    >>>     print(output.mul(output).sum())
    I want the vector length of output always near to 1.
    '''
    sqrt_3 = math.sqrt(3.)
    the_factor = sqrt_3/math.sqrt(out_features)#*in_features)
    
    #the_factor = 3./math.sqrt(out_features)#*in_features)
    result = (torch.rand([out_features, in_features])*2.-1.)*the_factor
    return result, sqrt_3




def debug_zero_grad_ratio(parameter:torch.nn.parameter.Parameter, \
    print_out:float = False)->float:
    if parameter.grad is None:
        if print_out:
            print(0., "inside debug_zero_grad_ratio function __line 338")
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



def debug_avg_log(input:torch.Tensor)->float:
    the_log = input.abs().log10()
    flag_inf = the_log.isinf()
    count_of_nan_and_inf = the_log.isnan().sum()+flag_inf.sum()
    no_nan_log = the_log.nan_to_num(0.)
    safe_log = flag_inf.logical_not()*no_nan_log
    numerator = safe_log.sum()
    nelement = input.nelement()
    result = numerator/(nelement-count_of_nan_and_inf)
    return result.item()
# input = torch.tensor([torch.nan, torch.inf, torch.inf*-1, 0., -100])
# print(debug_avg_log(input))
# input = torch.tensor([10000000000000000000.,-100000000000000000000000000.])
# print(debug_avg_log(input))
# fds=432






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

# '''int_into_floats'''  
# input = torch.tensor([[0],[1],[2],[3],[7],])
# print(int_into_floats(input,7,True))
# print(int_into_floats(input,7,False))
# fds=432



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

# '''int_into_floats'''  
# input = torch.tensor([[0],[1],[2],[3],[7],])
# print(int_into_floats_with_str(input,4,True))
# print(int_into_floats_with_str(input,4,False))
# fds=432



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

# '''floats_into_int'''   
# input = torch.tensor([[0],[1],[2],[3],[7],])
# input = int_into_floats(input,7, True)
# print(floats_into_int(input).T)
# input = torch.tensor([[0],[1],[2],[3],[7],])
# input = int_into_floats_with_str(input,7, True)
# print(floats_into_int(input).T)
# input = torch.tensor([[0],[1],[2],[3],[7],])
# input = int_into_floats(input,7, False)
# print(floats_into_int(input).T)
# input = torch.tensor([[0],[1],[2],[3],[7],])
# input = int_into_floats_with_str(input,7, False)
# print(floats_into_int(input).T)
# fds=432


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

# a,b = data_gen_for_directly_stacking_test(5,3,2)
# print(a)
# print(b)
# a,b = data_gen_for_directly_stacking_test(5,3,2, no_duplicated=True)
# fds=423



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
if 'basic test' and False:
    input = torch.tensor([[0.,1]],dtype=torch.float16)
    print(softmax_dim_1_from_yagaodirac(input))
    print(input.to(torch.float32).softmax(dim=1))
    pass
if 'basic test' and False:
    dummy = torch.tensor([[0,1]],dtype=torch.int64)
    import random
    input = torch.randn((random.randint(2,5),random.randint(2,5)),dtype=torch.float16)
    print(softmax_dim_1_from_yagaodirac(input))
    print(input.to(torch.float32).softmax(dim=1))
    pass

