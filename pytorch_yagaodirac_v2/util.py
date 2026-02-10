from typing import List, Tuple, Optional
import torch
import math, random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit





def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
# if "test" and True:
#     assert __DEBUG_ME__()
#     pass

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

def _float_equal(a:float|torch.Tensor, b:float, epsilon:float = 0.0001)->bool:
    if isinstance(a, torch.Tensor):
        a = a.item()
        pass
    if isinstance(b, torch.Tensor):
        b = b.item()
        pass
    assert epsilon>0.
    return abs(a-b)<epsilon
if "test" and __DEBUG_ME__() and False:
    assert _float_equal(1., 1.)
    assert _float_equal(1., 1.0000001)
    assert _float_equal(1., 1.01) == False
    assert _float_equal(1., 1.01, 0.1) 
    def ____test_____float_equal():
        a = torch.tensor(1.)
        assert _float_equal(a, 1.) 
        assert isinstance(a, torch.Tensor)
        return 
    ____test_____float_equal()
    pass
def _tensor_equal(  a:torch.Tensor|list[float]|list[list[float]], \
                    b:torch.Tensor|list[float]|list[list[float]], \
                        epsilon:float = 0.0001)->bool:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        pass
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
        pass
    #check the shape.
    if a.shape == torch.Size([]):
        assert b.shape == torch.Size([]) or b.shape == torch.Size([1])
        pass
    elif b.shape == torch.Size([]):#a is not Size([])
        assert a.shape == torch.Size([1])
        pass
    else:#no Size([]), a normal check.
        assert a.shape == b.shape
        pass
    
    
    with torch.inference_mode():
        diff = a-b
        abs_of_diff = diff.abs()
        less_than = abs_of_diff.lt(epsilon)
        after_all = less_than.all()
        assert after_all.dtype == torch.bool
        the_item = after_all.item()
        assert isinstance(the_item, bool)
        return the_item
    pass#end of function
if "test" and __DEBUG_ME__() and False:
    assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.]))
    assert _tensor_equal(torch.tensor([1.,2.]), [1.,2.])
    #assert _tensor_equal(torch.tensor([1.]), torch.tensor([[1.]]))
    assert _tensor_equal(torch.tensor([[1.]]), torch.tensor([[1.]]))
    assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.000001]))
    assert _tensor_equal(torch.tensor([1.]), torch.tensor([0.99999]))
    assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.001])) == False
    
    #shape
    assert _tensor_equal(torch.tensor([0.]), torch.tensor([0.]))
    assert _tensor_equal(torch.tensor([0.]), torch.tensor(0.))
    assert _tensor_equal(torch.tensor(0.), torch.tensor([0.]))
    assert _tensor_equal(torch.tensor(0.), torch.tensor(0.))
    pass

def str_the_list(the_list:list, precision = 3)->str:
    result = "["
    command_str = "{:."+str(precision)+"f}"
    all_the_sub_strings = [command_str.format(the_number) for the_number in the_list]
    for ii in range(all_the_sub_strings.__len__()):
        sub_str = all_the_sub_strings[ii]
        if sub_str[0]!='-':
            all_the_sub_strings[ii] = " "+all_the_sub_strings[ii]
            pass
        pass
            
    mid_str = ", ".join(all_the_sub_strings)
    result = f"[{mid_str}]"
    return result
if "test":
    def ____test____str_the_list():
        the_str = str_the_list([1.23467,-2.23467], 3)
        assert the_str == "[ 1.235, -2.235]"
        
        return
    ____test____str_the_list()
    pass

def have_same_elements(tensor_1:torch.Tensor, tensor_2:torch.Tensor)->bool:
    assert tensor_1.shape == tensor_2.shape

    tensor_1_elements = tensor_1.reshape([-1]).sort().values
    tensor_2_elements = tensor_2.reshape([-1]).sort().values
    return tensor_1_elements.eq(tensor_2_elements).all()
if "test" and __DEBUG_ME__() and False:
    def ____test____have_same_elements():
        t1 = torch.tensor([1,2,3])
        t2 = torch.tensor([3,2,1])
        assert have_same_elements(t1,t2)
        
        t1 = torch.tensor([[1,2,3]])
        t2 = torch.tensor([[3,2,1]])
        assert have_same_elements(t1,t2)
        
        t1 = torch.tensor([[1,2,1111]])
        t2 = torch.tensor([[3,2,1]])
        assert have_same_elements(t1,t2) == False
        return
    ____test____have_same_elements()
    pass

def is_square_matrix(matrix:torch.Tensor)->bool:
    if matrix.shape.__len__() != 2:
        return False
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return True
if "test" and __DEBUG_ME__() and False:
    assert is_square_matrix(torch.randn(size=(2,2)))
    assert is_square_matrix(torch.randn(size=(3,3)))
    assert is_square_matrix(torch.randn(size=(2,3))) == False
    assert is_square_matrix(torch.randn(size=(2,))) == False
    assert is_square_matrix(torch.randn(size=(2,3,3))) == False
    pass

from typing import TypeAlias,Literal
DeviceLikeType: TypeAlias = str|torch.device|int
def iota(how_many:int, dtype_is_int64=False,\
            device: DeviceLikeType|None = None)->torch.Tensor:
    dtype = torch.int64
    if not dtype_is_int64 and how_many<(1<<31):
        dtype = torch.int32
        pass
    return torch.linspace(start=0,end=how_many-1,steps=how_many ,dtype=dtype, device=device)

if "torch linspace dtype test" and __DEBUG_ME__() and False:
    for device in ["cpu", "cuda"]:
        for dtype in [torch.int8,torch.int16,torch.int32,torch.int64,torch.int,torch.uint8,torch.long]:
                                            #but uint16,uint32,uint64 are not allowed
            _temp = torch.linspace(start=0,end=7,steps=8 ,dtype=dtype, device=device)
            pass
        pass
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int8)
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int16)
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int32)
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int64)
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint8)
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint16) not working
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint32) not working
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint64) not working
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int)#int32
    assert _temp_tensor.dtype == torch.int32
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.long)#int64
    assert _temp_tensor.dtype == torch.int64
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int8,device='cuda')
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int16,device='cuda')
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int32,device='cuda')
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int64,device='cuda')
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint8,device='cuda')
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint16,device='cuda') not working
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint32,device='cuda') not working
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint64,device='cuda') not working
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int,device='cuda')#int32
    assert _temp_tensor.dtype == torch.int32
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.long,device='cuda')#int64
    assert _temp_tensor.dtype == torch.int64
    pass    
if "test" and __DEBUG_ME__() and False:
    _1_leftshift_31_minus_1 = (1<<31)-1
    _temp = iota(_1_leftshift_31_minus_1)
    assert _temp.__len__() == _1_leftshift_31_minus_1
    assert _temp[-1] == _1_leftshift_31_minus_1-1
    assert _temp.dtype == torch.int32
    
    _1_leftshift_31 = 1<<31
    _temp = iota(_1_leftshift_31)
    assert _temp.__len__() == _1_leftshift_31
    assert _temp[-1] == _1_leftshift_31-1
    assert _temp.dtype == torch.int64
    
    _temp = iota(3, dtype_is_int64=True)
    assert _tensor_equal(_temp, [0.,1,2])
    assert _temp.dtype == torch.int64
    pass
if "can it be index?" and __DEBUG_ME__() and False:
    "pytorch only allows int32 or int64 as index."
    _data = torch.linspace(0,99,100, dtype=torch.float32).reshape([10,10])
    
    iota_of_data = iota(4)
    assert iota_of_data.dtype == torch.int32
    _part_of_data = _data[iota_of_data, iota_of_data]
    assert _tensor_equal(_part_of_data, [0.,11,22,33])
    
    iota_of_data = iota(4, dtype_is_int64=True)
    assert iota_of_data.dtype == torch.int64
    _part_of_data = _data[iota_of_data, iota_of_data]
    assert _tensor_equal(_part_of_data, [0.,11,22,33])
    
    if "the following don't work. They raise." and False and False and False:
        _data = torch.linspace(0,99,100, dtype=torch.float32).reshape([10,10])
        iota_of_data = iota(4)
        _part_of_data = _data[iota_of_data.to(torch.int8), iota_of_data]
        _part_of_data = _data[iota_of_data.to(torch.int16), iota_of_data]
        _part_of_data = _data[iota_of_data.to(torch.uint8), iota_of_data]
        pass
    pass





def vector_length_norm(input:torch.Tensor, epi = 0.000001, dtype_inner = torch.float64)->torch.Tensor:
    r'''The shape must be [batch, dim]'''
    if len(input.shape)!=2:
        raise Exception("The shape must be [batch, dim]")
    with torch.no_grad():
        
        # if not transform:
        #     length_of_input_b_1 = input.mul(input).sum(dim=1,keepdim=True).sqrt()
        #     pass
        # else:
        #     length_of_input_b_1 = input.mul(input).sum(dim=0,keepdim=True).sqrt()
        #     pass
        length_of_input_b_1 = input.mul(input).sum(dim=1,keepdim=True,dtype=dtype_inner).sqrt()
        
        epi_tensor = torch.tensor([epi], device=length_of_input_b_1.device, dtype=dtype_inner)
        length_of_input_safe__b = length_of_input_b_1.maximum(epi_tensor)
        length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b.expand([-1,input.shape[1]])
        # if not transform:
        #     result = input/length_of_input_safe_b.expand([-1,input.shape[1]])
        #     pass
        # else:
        #     result = input/length_of_input_safe_b.expand([input.shape[0],-1])
        #     pass
        length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b_1EXPANDdim.to(dtype=input.dtype)
        result = input.div(length_of_input_safe__b_1EXPANDdim)
        return result
    #end of function.
if '''some basic test.''' and __DEBUG_ME__() and False:
    input = torch.tensor([[0.,0.],[0.,1.],[1.,1.]])
    output = vector_length_norm(input)
    assert _tensor_equal(output, [[0.,0.],[0.,1.],[0.7,0.7]], 0.05)
    assert output.dtype == torch.float32
    _vector_len = output.mul(output).sum(dim=1)
    assert _tensor_equal(_vector_len[0], torch.zeros_like(_vector_len[0]), 0.05)
    assert _tensor_equal(_vector_len[1:], torch.ones_like(_vector_len[1:]), 0.05)
    
    #transform
    # input = torch.tensor([[1.,1],[0.1,0.1]])
    # output = vector_length_norm(input, transform=True)
    # assert _tensor_equal(output,   [[0.9950, 0.9950],
    #                                 [0.0995, 0.0995]], epsilon=0.001)
    input = torch.tensor([[1.,1],[0.1,0.1]])
    output = vector_length_norm(input.T).T
    assert _tensor_equal(output,   [[0.9950, 0.9950],
                                    [0.0995, 0.0995]], epsilon=0.001)
    
    pass

def get_vector_length(input:torch.Tensor, result_dtype = torch.float64)->torch.Tensor:
    _temp = input*input
    #if input.shape.__len__() == 2:
    _temp = _temp.sum(dim=-1, dtype=result_dtype)
    _temp.sqrt_()
    return _temp
if "test get_vector_length" and __DEBUG_ME__() and False:
    def ____test____get_vector_length():
        input = torch.tensor([1.,1])
        output = get_vector_length(input)
        assert output.shape == torch.Size([])
        assert output.dtype == torch.float64
        assert _tensor_equal(output, [1.4142])
        
        input = torch.tensor([[1.,1],[1,2]])
        output = get_vector_length(input)
        assert output.shape == torch.Size([2])
        assert _tensor_equal(output, [1.4142,2.2361])
        
        input = torch.tensor([[[1.,1],[1,2]],[[2,1],[2,2]],[[3,1],[3,2]]])
        output = get_vector_length(input)
        assert output.shape == torch.Size([3,2])
        assert _tensor_equal(output, [[1.4142,2.2361],[2.2361, 2.8284],[3.1623,3.6056]])
        
        "dtype"
        input = torch.tensor([1.,1])
        output = get_vector_length(input, result_dtype=torch.float16)
        assert output.dtype == torch.float16
        
        return 
    ____test____get_vector_length()
    pass
    
    



if "old rotation neural net related" and False:


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
    pass

if "grad balancer,          not in use at the moment" and False:

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
    pass

if "probably the old gramo???" and False:

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
    pass




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









# Here I provide 2 versions.
# the old version is on top. It's better but not a pure gpu implementation.
# Then I made a new torch-specified version. It's faster.
# if you write code on other tool, I recommend shift the old version, bc the also is better.
# but if you write on pytorch, use the new version.
# the behavior and results are slightly different.

# I did tests in log10 measurement.py file. Check it out if you are learning this tool.

def get_mask_of_top_element__rough(input__b_i:torch.Tensor, top_ratio = 0.9, error_of_ratio__at_least = 0.01, \
                            bottom = False, careful_level:int|None = 3, epsilon__for_binary_search:float|torch.Tensor|None = None, \
                    _needs_log__before_loop = False, _needs_log__basic_of_loop = False, \
                    _needs_log__binary_search_in_loop = False, \
                    _needs_log__error_ratio_in_loop = False, \
                                    )->tuple[torch.Tensor, dict[str, list[str]]|None]:
    ''' 
    return _temp_tensor, _log
    return _temp_tensor, _log
    return _temp_tensor, _log
    return _temp_tensor, _log
    
    重新整理一下思路
    这个函数有2个退出模式。
    1是，准确的找到了所需要的比例。比例的目标区间本身是越来越宽的。
    2是，类似二分查找的那种上下限，如果距离足够近，也就退出了。
    核心思想就是，
    目标，和error，算出允许的比例的上下限  top_ratio  error   at_least/most_this_amount 
    二分查找的那个标准是最大值和最小值的平均值，然后根据需要的方向来缩小。   max, min,(threshold), epi
    
    maintainance note:
    this function is init+loop, the loop is binary_search+if_return+if_repeating
    or, it's
    def ...():
        init
        while true:
            binary_search
            if_return?(the only return)
            if_repeating
            pass#while true
        pass#end of function
    In init, there's a bit early return. I didn't remove it, but it should not be triggered.
    The binary search is not an exact binary search, but it feels similar. The max or min is assigned with mid 
        according to the condition, to modify the threshold.
    Then test if the return condition is met.
    The if_repeating is something detecting if the loop is too repeating. If so, it broadens the error_of_ratio,
        and makes the return condition looser.
    The function guaruntees to return nomatter what you provide, unless there's inf,-inf or nan. 
    I didn't test this extreme case. So please make sure the input is legit numbers.
    
    Shape:
    
    Input is [batch, something], return shape is the same as input. dtype of return is torch.bool.
    
    Dimention name:
    
    shape is [*B*atch, *I*nput]
    
    if shape is too small, this may not work. Valid code uses 3 or 5 elements/batch, but this function is design 
        to process >10 elements/batch input.
    '''
    assert input__b_i.shape.__len__() == 2
    assert top_ratio>0.
    assert top_ratio<1.
    assert error_of_ratio__at_least>=0.
    if careful_level is None:
        careful_level = 3
    assert careful_level>0
    assert careful_level<64, "or modify the data type. search for repeating__b = torch.zeros_like(_the_max_to_calc_threshold__b, dtype=torch.int8)"
    
    _cpu = input__b_i.device.type != "cuda"
    if _cpu and _needs_log__before_loop:
        _log:dict[str, list[str]]|None = {}
        pass
    else:
        _log = None  
        pass
    
    if epsilon__for_binary_search:
        if isinstance(epsilon__for_binary_search, float):
            epsilon__for_binary_search__s = torch.tensor(epsilon__for_binary_search, device=input__b_i.device, dtype=input__b_i.dtype)
            pass
        else:
            # if the assertions don't pass, modify it. I didn't test it very carefully. 
            # simply reference the line above.
            assert isinstance(epsilon__for_binary_search, torch.Tensor)
            assert epsilon__for_binary_search.dtype == input__b_i.dtype
            epsilon__for_binary_search__s = epsilon__for_binary_search.clone().to(input__b_i.device)
            if epsilon__for_binary_search.shape == torch.Size([]):
                epsilon__for_binary_search__s = epsilon__for_binary_search__s.reshape([1,])
                pass
            assert epsilon__for_binary_search__s.shape == torch.Size([1])
        pass
    if (_log is not None) and _needs_log__before_loop:
        _log["before_loop"] = [f"epsilon__for_binary_search:{epsilon__for_binary_search}"]
        pass
    
    
    #dtype uint
    #best dtype for count the amount.
    _element_per_batch__s = input__b_i.shape[1]
    # device = input.device
    # param_factory = {"device":device, "dtype":dtype}
    #dtype int
    if _element_per_batch__s<=(1<<6):
        int_dtype = torch.int8
        pass
    elif _element_per_batch__s<=(1<<14):
        int_dtype = torch.int16
        pass
    elif _element_per_batch__s<=(1<<30):
        int_dtype = torch.int32
        pass
    else:
        int_dtype = torch.int64
        pass
    if _log and _needs_log__before_loop:
        _log["before_loop"].append(f"int type:{str(int_dtype)}")
        pass
    
    with torch.no_grad():
        #into torch.
        careful_level__s:torch.Tensor = torch.tensor(careful_level, device=input__b_i.device)
        del careful_level
        if bottom:
            top_ratio = 1.- top_ratio
            pass
        top_ratio__s = torch.tensor(top_ratio, dtype=torch.float64, device=input__b_i.device)
        del top_ratio
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"top ratio:{top_ratio__s}, is bottom:{bottom}")
            pass
        
        #init error_of_ratio 
        better_error_of_ratio = 0.501/_element_per_batch__s
        if better_error_of_ratio<error_of_ratio__at_least:
            better_error_of_ratio = error_of_ratio__at_least
            pass
        del error_of_ratio__at_least
        error_of_ratio__b = torch.empty(size=[input__b_i.shape[0]], device=input__b_i.device)#.reshape([-1,1])
        error_of_ratio__b.fill_(better_error_of_ratio)
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"error_of_ratio__b init to:{error_of_ratio__b}")
            pass
        
        #ratio+-error, this segment appears twice in this function.
        at_least_this_amount__b = ((_element_per_batch__s-2)*(top_ratio__s - error_of_ratio__b)+1.4999).to(int_dtype)#.reshape([-1,1])
        at_most_this_amount__b =  ((_element_per_batch__s-2)*(top_ratio__s + error_of_ratio__b)+1.5001).to(int_dtype)#.reshape([-1,1])
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"at_least_this_amount__b init to:{at_least_this_amount__b}")
            _log["before_loop"].append(f"at_most_this_amount__b init to:{at_most_this_amount__b}")
            pass
        
        #safety, or maybe a early return.
        _flag_all_true_early_return__b = at_least_this_amount__b.ge(_element_per_batch__s)
        if _flag_all_true_early_return__b.all():
            _temp_tensor = torch.ones_like(input__b_i, dtype=torch.bool, device=input__b_i.device)
            if _log and _needs_log__before_loop:
                _log["before_loop"].append(f"_flag_all_true_early_return__b:{_flag_all_true_early_return__b}, all true, [return]")
                pass
            return _temp_tensor, _log
        _flag_all_true_early_return__b = at_most_this_amount__b.le(0)
        if _flag_all_true_early_return__b.all():
            _temp_tensor = torch.zeros_like(input__b_i, dtype=torch.bool, device=input__b_i.device)
            if _log and _needs_log__before_loop:
                _log["before_loop"].append(f"_flag_all_true_early_return__b:{_flag_all_true_early_return__b}, all true, [return]")
                pass
            return _temp_tensor, _log
        
        #maybe optimizable. reverse+reverse = nothing.
        if_finished__b = (_flag_all_true_early_return__b).logical_or(_flag_all_true_early_return__b)
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"if_finished__b init to:{if_finished__b}")
            pass
        
        # init before loop
        _the_max_to_calc_threshold__b:torch.Tensor = input__b_i.max(dim=1).values#.reshape([-1,1])#111111111111111111
        _the_min_to_calc_threshold__b:torch.Tensor = input__b_i.min(dim=1).values#.reshape([-1,1])
        if input__b_i.dtype != torch.float64 and input__b_i.dtype != torch.float32:
            _the_max_to_calc_threshold__b.to(torch.float16)
            _the_min_to_calc_threshold__b.to(torch.float16)
            pass
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"_the_max_to_calc_threshold__b init to:{_the_max_to_calc_threshold__b}")
            _log["before_loop"].append(f"_the_min_to_calc_threshold__b init to:{_the_min_to_calc_threshold__b}")
            pass
        
        #all the zero init.
        _guess_threshold__b = torch.zeros_like(if_finished__b,dtype=_the_max_to_calc_threshold__b.dtype)#.reshape([-1,1])#11111111111111
        _if__guess_too_big___b = torch.zeros_like(if_finished__b)
        _if__guess_too_small___b = torch.zeros_like(if_finished__b)
        _input_gt_guess__count__b = torch.zeros_like(if_finished__b, dtype=int_dtype)
        
        RESULT__if__input_gt_guess__b_i = torch.zeros_like(input__b_i, dtype=torch.bool)
        old_unqualified_RESULT__if__input_gt_guess__b_i = torch.zeros_like(input__b_i,dtype=torch.bool)
        
        _if__unchanged__b = torch.zeros_like(if_finished__b, dtype=torch.bool)
        repeating__b = torch.zeros_like(_the_max_to_calc_threshold__b, dtype=torch.int8)#.squeeze_()#11111111111111
        
        # now is this one: if_finished__b
        # it was init_ed_the_flag_result
        if _log and _needs_log__before_loop:
            _log["before_loop"].append("vvvv   below are all the init to zero   vvvv")
            
            _log["before_loop"].append(f"_guess_threshold__b init to:{_guess_threshold__b}")
            _log["before_loop"].append(f"_if__guess_too_big___b init to:{_if__guess_too_big___b}")
            _log["before_loop"].append(f"_if__guess_too_small___b init to:{_if__guess_too_small___b}")
            _log["before_loop"].append(f"_input_gt_guess__count__b init to:{_input_gt_guess__count__b}")
            
            _log["before_loop"].append(f"RESULT__if__input_gt_guess__b_i init to:{RESULT__if__input_gt_guess__b_i}")
            _log["before_loop"].append(f"old_unqualified_RESULT__b_i init to:{old_unqualified_RESULT__if__input_gt_guess__b_i}")
            
            _log["before_loop"].append(f"_if__unchanged__b init to:{_if__unchanged__b}")
            _log["before_loop"].append(f"repeating__b init to:{repeating__b}")
            pass
        
        #before while
        _needs_log__loop_count = _cpu and (_needs_log__basic_of_loop or _needs_log__binary_search_in_loop or \
                                    _needs_log__error_ratio_in_loop)
        if _needs_log__loop_count:
            if _log is None:
                _log = {}
                pass
        if _cpu and (_log is not None) and _needs_log__loop_count:
            loop_count = 0
            _log["in_loop"] = []
            pass
        while True:
            if _log:
                if _needs_log__loop_count:
                    _log["in_loop"].append(f"----  loop {loop_count}  ----")
                    pass
                if _needs_log__basic_of_loop:
                    _log["in_loop"].append(f"if_finished__b:{if_finished__b}")
                    pass
                pass
            #similar to binary search
            _guess_threshold__b[~if_finished__b] = (_the_max_to_calc_threshold__b[~if_finished__b]+_the_min_to_calc_threshold__b[~if_finished__b])/2.#maybe optimizable.
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"_guess_threshold:{_guess_threshold__b}")
                pass
            #the real comparison
            RESULT__if__input_gt_guess__b_i[~if_finished__b] = input__b_i[~if_finished__b].gt \
                                                (_guess_threshold__b[~if_finished__b].reshape([-1,1]).expand([-1,_element_per_batch__s]))
            #if guessed too big, then, less true
            _input_gt_guess__count__b[~if_finished__b] = RESULT__if__input_gt_guess__b_i[~if_finished__b].sum(dim=1, dtype=int_dtype)
            #_guess_count = flag_result.to(int_dtype).sum(dim=1)
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"RESULT_input_gt_guess__b_i___mask_if_finish:{RESULT__if__input_gt_guess__b_i}")
                _log["in_loop"].append(f"_input_gt_guess__count__b:{_input_gt_guess__count__b}")
                pass
            
            
            # #flag_gt old code
            # _if__guess_not_too_big___b = torch.zeros_like(if_finished__b)
            # _if__guess_not_too_big___b[~if_finished__b] = _guess_count__b[~if_finished__b].le(at_most_this_amount__b[~if_finished__b])
            # # ^^^ true is good. ^^^
            # _the_min_to_calc_threshold__b[~_if__guess_not_too_big___b] = _guess_threshold[~_if__guess_not_too_big___b]
            
            #flag_gt
            _if__guess_too_big___b[~if_finished__b] = _input_gt_guess__count__b[~if_finished__b].lt(at_least_this_amount__b[~if_finished__b])
            # ^^^ true is bad. ^^^
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"_if__guess_too_big___b(true is bad):{_if__guess_too_big___b}")
                _log["in_loop"].append(f"_the_max_to_calc_threshold__b, from:{_the_max_to_calc_threshold__b}")
                pass
            _the_max_to_calc_threshold__b[_if__guess_too_big___b] = _guess_threshold__b[_if__guess_too_big___b]
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{_the_max_to_calc_threshold__b}")
                pass
            
            
            # #flag_lt old code
            # _if__guess_not_too_small___b = torch.zeros_like(if_finished__b)
            # _if__guess_not_too_small___b[~if_finished__b] = _guess_count__b[~if_finished__b].ge(at_least_this_amount__b[~if_finished__b])
            # # ^^^ true is good. ^^^
            # _the_max_to_calc_threshold__b[~_if__guess_not_too_small___b] = _guess_threshold[~_if__guess_not_too_small___b]
            
            #flag_lt
            _if__guess_too_small___b[~if_finished__b] = _input_gt_guess__count__b[~if_finished__b].gt(at_most_this_amount__b[~if_finished__b])
            # ^^^ true is bad. ^^^
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"_if__guess_too_small___b(true is bad):{_if__guess_too_small___b}")
                _log["in_loop"].append(f"_the_min_to_calc_threshold__b:{_the_min_to_calc_threshold__b}")
                pass
            _the_min_to_calc_threshold__b[_if__guess_too_small___b] = _guess_threshold__b[_if__guess_too_small___b]
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{_the_min_to_calc_threshold__b}")
                pass
            
            _flag__not_too_loose__and__not_too_tight___b_1 = (~_if__guess_too_big___b).logical_and(~_if__guess_too_small___b)
            #           ^^^ true is good. ^^^                   ^^^ true is bad. ^^^                  ^^^ true is bad. ^^^  
            if _log: 
                if _needs_log__binary_search_in_loop:
                    _log["in_loop"].append(f"_flag__not_too_loose__and__not_too_tight___b_1(true is good):{_flag__not_too_loose__and__not_too_tight___b_1}")
                    pass
                if _needs_log__basic_of_loop:
                    _log["in_loop"].append(f"if_finished__b, from:{if_finished__b}")
                    pass
                pass
            if_finished__b.logical_or_(_flag__not_too_loose__and__not_too_tight___b_1)
            if _log and _needs_log__basic_of_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{if_finished__b}")
                pass
            
            if epsilon__for_binary_search is not None:
                _flag_less_than_epsilon = (_the_max_to_calc_threshold__b-_the_min_to_calc_threshold__b).lt(epsilon__for_binary_search__s)
                if _log and _needs_log__binary_search_in_loop:
                    _log["in_loop"].append(f"epsilon__for_binary_search__s:{epsilon__for_binary_search__s}")
                    _log["in_loop"].append(f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{_flag_less_than_epsilon \
                                                    }, and it makes if_finished__b from:{if_finished__b}")
                    pass
                if_finished__b.logical_or_(_flag_less_than_epsilon)
                if _log and _needs_log__binary_search_in_loop:
                    _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{if_finished__b}")
                    pass
                pass#if epsilon
            
            # this is the only [return] timing.
            if if_finished__b.all():
                if _log:
                    _log["in_loop"].append(f"[return]")
                    pass
                if bottom:
                    if _log:
                        _log["in_loop"].append(f"{_log["in_loop"].pop()}[bc it's bottom=true, returns the reversed result]")
                        pass
                    RESULT__if__input_gt_guess__b_i.logical_not_()
                    pass
                return RESULT__if__input_gt_guess__b_i, _log
                pass #if if_finished__b.all():
            
            
            #if the new result[b,i] unchanged?
            _if__unchanged__b[~if_finished__b] = old_unqualified_RESULT__if__input_gt_guess__b_i[~if_finished__b].eq( \
                                                                RESULT__if__input_gt_guess__b_i[~if_finished__b]).all(dim=1)
            # ^^^ true is bad. ^^^
            
            if _log and _needs_log__error_ratio_in_loop:
                _log["in_loop"].append("-- the second return condition --")
                _log["in_loop"].append(f"_if__unchanged__b:{_if__unchanged__b}")
                _log["in_loop"].append(f"repeating__b, from:{repeating__b}")
                pass
            repeating__b[_if__unchanged__b] = repeating__b[_if__unchanged__b].add(1)
            if _log and _needs_log__error_ratio_in_loop:
                assert _log
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{repeating__b}")
                pass
            
            
            #if 
            _if__repeated_enough__b = repeating__b.ge(careful_level__s)
            if _log and _needs_log__error_ratio_in_loop:
                _log["in_loop"].append(f"_if__repeated_enough__b:{_if__repeated_enough__b}")
                _log["in_loop"].append(f"repeating__b, from:{repeating__b}")
                pass
            repeating__b[_if__repeated_enough__b] = 0
            if _log and _needs_log__error_ratio_in_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{repeating__b}")
                _log["in_loop"].append(f"error_of_ratio__b, from:{error_of_ratio__b}")
                pass
            #update the finishing flags.
            error_of_ratio__b[_if__repeated_enough__b] = error_of_ratio__b[_if__repeated_enough__b].mul(2.)#this 2. is not tested.
            #maybe wrong??? is it updated?
            if _log and _needs_log__error_ratio_in_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{error_of_ratio__b}")
                pass
            
            
            
            if _log:
                if _needs_log__error_ratio_in_loop:
                    _log["in_loop"].append(f"_if__repeated_enough__b:{_if__repeated_enough__b}")
                    pass
                if _needs_log__basic_of_loop:
                    _log["in_loop"].append(f"at_least_this_amount__b, from:{at_least_this_amount__b}")
                    _log["in_loop"].append(f"at_most_this_amount__b, from:{at_most_this_amount__b}")
                    pass
                pass
            #ratio+-error, this segment appears twice in this function.
            #[1]+[] is []. So this is safe.
            
            at_least_this_amount__b[_if__repeated_enough__b] = ((_element_per_batch__s-2)*(top_ratio__s - \
                error_of_ratio__b[_if__repeated_enough__b])+1.4999).to(int_dtype)
            at_most_this_amount__b[_if__repeated_enough__b] =  ((_element_per_batch__s-2)*(top_ratio__s + \
                error_of_ratio__b[_if__repeated_enough__b])+1.5001).to(int_dtype)
            # no detect for return here. reason:
            # even if this range-like can mean a range covering all the range, bc I believe it unlikely to happen.
            # I decide to delay the return to the next round.
            if _log and _needs_log__basic_of_loop: 
                _temp_str_at_most_this_amount__b_from = _log["in_loop"].pop()
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{at_least_this_amount__b}")
                _log["in_loop"].append(f"{_temp_str_at_most_this_amount__b_from}, to:{at_most_this_amount__b}")
                pass
            
            #tail
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"RESULT__b_i, from:{old_unqualified_RESULT__if__input_gt_guess__b_i}, to:{RESULT__if__input_gt_guess__b_i}")
                pass
            old_unqualified_RESULT__if__input_gt_guess__b_i = RESULT__if__input_gt_guess__b_i
            if _log and _needs_log__loop_count:
                _log["in_loop"].append(f"loop {loop_count} ends.")
                loop_count += 1
                pass
            pass#while true
        
        pass#  no_grad
    pass# end of function
if "performance test    slow" and __DEBUG_ME__() and False:
    "result"
    "my version is basically only about the amount of data. torch version is better unless it's a lot data and on cpu. "
    "so, although I wrote my version, but it's still faster to move the data to gpu and do torch version."
    "ok, this is cool."
    # cpu: my:0.001839, torch:0.000041   [  10,  100]   gpu: my:0.013041, torch:0.000045   torch move to gpu: my:0.000063   [  10,  100]
    # cpu: my:0.001658, torch:0.000178   [  10,  330]   gpu: my:0.015410, torch:0.000037   torch move to gpu: my:0.000064   [  10,  330]
    # cpu: my:0.001906, torch:0.000672   [  10, 1000]   gpu: my:0.010878, torch:0.000035   torch move to gpu: my:0.000064   [  10, 1000]
    # cpu: my:0.002109, torch:0.001553   [  10, 3300]   gpu: my:0.011062, torch:0.000042   torch move to gpu: my:0.000094   [  10, 3300]
    # cpu: my:0.002413, torch:0.003691   [  10,10000]   gpu: my:0.011196, torch:0.000230   torch move to gpu: my:0.000253   [  10,10000]
    # cpu: my:0.002128, torch:0.000131   [  33,  100]   gpu: my:0.013590, torch:0.000046   torch move to gpu: my:0.000073   [  33,  100]
    # cpu: my:0.002270, torch:0.000586   [  33,  330]   gpu: my:0.014319, torch:0.000048   torch move to gpu: my:0.000064   [  33,  330]
    # cpu: my:0.002400, torch:0.001306   [  33, 1000]   gpu: my:0.010810, torch:0.000043   torch move to gpu: my:0.000094   [  33, 1000]
    # cpu: my:0.002779, torch:0.003278   [  33, 3300]   gpu: my:0.010969, torch:0.000091   torch move to gpu: my:0.000163   [  33, 3300]
    # cpu: my:0.003225, torch:0.004388   [  33,10000]   gpu: my:0.013599, torch:0.000470   torch move to gpu: my:0.000599   [  33,10000]
    # cpu: my:0.002397, torch:0.000412   [ 100,  100]   gpu: my:0.013237, torch:0.000046   torch move to gpu: my:0.000070   [ 100,  100]
    # cpu: my:0.002473, torch:0.001105   [ 100,  330]   gpu: my:0.014954, torch:0.000047   torch move to gpu: my:0.000100   [ 100,  330]
    # cpu: my:0.002923, torch:0.002156   [ 100, 1000]   gpu: my:0.015217, torch:0.000062   torch move to gpu: my:0.000132   [ 100, 1000]
    # cpu: my:0.003317, torch:0.003772   [ 100, 3300]   gpu: my:0.011056, torch:0.000224   torch move to gpu: my:0.000372   [ 100, 3300]
    # cpu: my:0.006159, torch:0.013583   [ 100,10000]   gpu: my:0.011261, torch:0.001141   torch move to gpu: my:0.001530   [ 100,10000]
    # cpu: my:0.003091, torch:0.000771   [ 330,  100]   gpu: my:0.019449, torch:0.000058   torch move to gpu: my:0.000093   [ 330,  100]
    # cpu: my:0.003503, torch:0.001913   [ 330,  330]   gpu: my:0.014683, torch:0.000103   torch move to gpu: my:0.000179   [ 330,  330]
    # cpu: my:0.003942, torch:0.003066   [ 330, 1000]   gpu: my:0.013820, torch:0.000172   torch move to gpu: my:0.000320   [ 330, 1000]
    # cpu: my:0.006749, torch:0.012893   [ 330, 3300]   gpu: my:0.011935, torch:0.000608   torch move to gpu: my:0.001016   [ 330, 3300]
    # cpu: my:0.018486, torch:0.041036   [ 330,10000]   gpu: my:0.013873, torch:0.004045   torch move to gpu: my:0.005211   [ 330,10000]
    # cpu: my:0.003851, torch:0.001376   [1000,  100]   gpu: my:0.019149, torch:0.000077   torch move to gpu: my:0.000159   [1000,  100]
    # cpu: my:0.004946, torch:0.002477   [1000,  330]   gpu: my:0.014508, torch:0.000281   torch move to gpu: my:0.000426   [1000,  330]
    # cpu: my:0.007018, torch:0.008683   [1000, 1000]   gpu: my:0.015103, torch:0.000425   torch move to gpu: my:0.000792   [1000, 1000]
    # cpu: my:0.019464, torch:0.036045   [1000, 3300]   gpu: my:0.017305, torch:0.001718   torch move to gpu: my:0.002898   [1000, 3300]
    # cpu: my:0.060130, torch:0.116552   [1000,10000]   gpu: my:0.021209, torch:0.010704   torch move to gpu: my:0.015695   [1000,10000]
    
    def ____test____is_my_version_faster():
        from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
        for batch in [10,33,100,330,1000]:
            for dim in [100,330,1000,3300,10000]:
                size = torch.Size([batch, dim])
                if batch*dim<=1_000_000:
                    time_at_most = 1.
                    pass
                else:
                    time_at_most = 4.
                    pass
                
                # vec = torch.rand(size=size)
                # def my_version():
                #     a = get_mask_of_top_element__rough(vec)
                #     pass
                # my_version_time = timeit(my_version, time_at_most=time_at_most)
                
                # vec = torch.rand(size=size)
                # def torch_version():
                #     b = vec.sort().values[:,:9000]
                #     pass
                # torch_version_time = timeit(torch_version, time_at_most=time_at_most)
                
                # vec_gpu = torch.rand(size=size,device='cuda')
                # def my_gpu_version():
                #     a = get_mask_of_top_element__rough(vec_gpu)
                #     pass
                # my_gpu_version_time = timeit(my_gpu_version, time_at_most=time_at_most)
                
                # vec_gpu = torch.rand(size=size,device='cuda')
                # def torch_gpu_version():
                #     b = vec_gpu.sort().values[:,:9000]
                #     pass
                # torch_gpu_version_time = timeit(torch_gpu_version, time_at_most=time_at_most)
                
                vec = torch.rand(size=size)
                def torch_move_to_gpu_version():
                    b = vec.to('cuda').sort().values[:,:9000]
                    pass
                torch_move_to_gpu_version_time = timeit(torch_move_to_gpu_version, time_at_most=time_at_most)
                
                #print(f"cpu: my:{my_version_time[0]:.6f}, torch:{torch_version_time[0]:.6f}   [{batch:4},{dim:5
                #   }]   gpu: my:{my_gpu_version_time[0]:.6f}, torch:{torch_gpu_version_time[0]:.6f}")
                print(f"cpu: my:{torch_move_to_gpu_version_time[0]:.6f}   [{batch:4},{dim:5}]")
                pass
            pass
        return
    ____test____is_my_version_faster()
    pass
if "test" and __DEBUG_ME__() and False:
    
    if "some real case,      batch>1      " and False:
        _input = torch.ones(size=[6,11])
        _input[0] = _input[0] + torch.randn(size=[1,11])*0.001
        _input[1] = _input[1]*-1 + torch.randn(size=[1,11])*0.001
        _input[2,0] = 0.
        _input[3,0] = 1e-10
        _input[4,0] = 1e-21
        _input[4,1] = 1e-10
        _input[5,0] = 1e-10
        _input[5,1] = 1e10
        
        log_of_input = _input.abs().log10()
        no_nan_log = log_of_input.nan_to_num(-999999.,posinf=-999999.,neginf=-999999.) 
        
        _result_tuple = get_mask_of_top_element__rough(no_nan_log, \
                            _needs_log__before_loop = True, _needs_log__basic_of_loop = True, \
                        _needs_log__binary_search_in_loop = True, _needs_log__error_ratio_in_loop = True)
        _log = _result_tuple[1]
        #a = _log["before_loop"]
        #b = _log["in_loop"]
        pass
    
    
    # torch.topk is not what I need.
    # a = torch.topk(torch.tensor([1,2,3,4,5]),3, sorted=False)
    # b = torch.topk(torch.tensor([1,2,3,4,5]),3, sorted=True)
    
    if "to test the formula for bounds" and False:
        n = 5
        top_ratio_list:list[float] = []
        for ii in range(1,10):
            top_ratio_list.append(ii*0.1)
            pass
        error_of_ratio = 0.1
        _floor_offset = 1.
        
        for top_ratio in top_ratio_list:
            lower_bound = (n-2)*(top_ratio - error_of_ratio)+_floor_offset
            upper_bound = (n-2)*(top_ratio + error_of_ratio)+_floor_offset
            # print(f"{top_ratio:.2f}, {lower_bound:.2f}/{upper_bound:.2f}")
            # from 1. to 4.
            # but in the real case, the offset is around 1.5, bc it's truncated into integer later.
            pass
        
        n = 5
        top_ratio_list = []
        for ii in range(1,30):
            top_ratio_list.append(ii*0.01)
            pass
        error_of_ratio = 0.1
        _floor_offset = 1.5
        
        for top_ratio in top_ratio_list:
            lower_bound = (n-2)*(top_ratio - error_of_ratio)+_floor_offset
            upper_bound = (n-2)*(top_ratio + error_of_ratio)+_floor_offset
            #print(f"{top_ratio:.2f}, {lower_bound:.2f}/{upper_bound:.2f}")
            pass
        
        for top_ratio in [0.06,0.07, 0.39,0.4, 0.73,0.74]:
            lower_bound = (n-2)*(top_ratio - error_of_ratio)+_floor_offset
            upper_bound = (n-2)*(top_ratio + error_of_ratio)+_floor_offset
            #print(f"{top_ratio:.2f}, {lower_bound:.2f}/{upper_bound:.2f}")
            pass
        
        #but this one looks symmetry.
        n = 10
        top_ratio_list = []
        for ii in range(1,100):
            top_ratio_list.append(ii*0.01)
            pass
        error_of_ratio = 0.1
        _floor_offset = 1.5
        
        for top_ratio in top_ratio_list:
            lower_bound = (n-2)*(top_ratio - error_of_ratio)+_floor_offset
            upper_bound = (n-2)*(top_ratio + error_of_ratio)+_floor_offset
            #print(f"{top_ratio:.2f}, {lower_bound:.2f}/{upper_bound:.2f}")
        pass
    
    
    
    
    #a1 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.01, _debug_needs_log = True)
    #a2 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.33, _debug_needs_log = True)
    #a3 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.67, _debug_needs_log = True)
    #a4 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.99, _debug_needs_log = True)
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.01)[0].eq(torch.tensor([False,False,False,False,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.33)[0].eq(torch.tensor([False,False,False,True,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.67)[0].eq(torch.tensor([False,False,True,True,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.99)[0].eq(torch.tensor([False,True,True,True,True])).all()
    
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3,4,1]]),top_ratio=0.01)[0].eq(torch.tensor([True,False,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3,4,1]]),top_ratio=0.33)[0].eq(torch.tensor([True,False,False,True,False])).all()
    

    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.01,bottom=True)[0].eq(torch.tensor([True,False,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.33,bottom=True)[0].eq(torch.tensor([True,True,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.67,bottom=True)[0].eq(torch.tensor([True,True,True,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.99,bottom=True)[0].eq(torch.tensor([True,True,True,True,False])).all()
    
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.01)[0].eq(torch.tensor([False,False,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.99)[0].eq(torch.tensor([False,True,True])).all()
    
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3]]),top_ratio=0.01)[0].eq(torch.tensor([True,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3]]),top_ratio=0.99)[0].eq(torch.tensor([True,False,True])).all()
    
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.01,bottom=True)[0].eq(torch.tensor([True,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.99,bottom=True)[0].eq(torch.tensor([True,True,False])).all()
    
        
    _shift_by_list = [          6,          7,         14,         15,]
    _int_dtype_list = [ torch.int8, torch.int16, torch.int16,torch.int32]
    for ii in range(_shift_by_list.__len__()):
        _shift_by = _shift_by_list[ii]
        _1_left_shift_by_shift_by_ = 1<<_shift_by
        for dtype in [torch.float16, torch.float32, torch.float64]:
            _input = torch.rand(size=[1,_1_left_shift_by_shift_by_],dtype=dtype)
            _result_tuple__tensor__list = get_mask_of_top_element__rough(_input,top_ratio=0.9, _needs_log__before_loop = True, \
                                                                                            _needs_log__basic_of_loop = True)
            _the_sum = _result_tuple__tensor__list[0].sum().item()
            assert _the_sum>(_1_left_shift_by_shift_by_)*0.8
            assert _the_sum<(_1_left_shift_by_shift_by_)
            
            _log = _result_tuple__tensor__list[1]
            assert _log
            assert _log["before_loop"][1]
            assert _log["before_loop"][1] == f"int type:{_int_dtype_list[ii]}"
            assert _log["in_loop"].__len__() >1
            
        pass

    # epsilon
    # epsilon helps when all elements are equal or nearly equal.
    # When the ratio doesn't exist, it's also helpful. At least the function returns.
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,1,1,1,1]]),top_ratio=0.5, bottom=True, epsilon__for_binary_search=0.001, \
                    _needs_log__before_loop = True,             _needs_log__basic_of_loop = True, \
                    _needs_log__binary_search_in_loop = True,   _needs_log__error_ratio_in_loop = True, )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                        }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"
    
    _input = torch.zeros(size=[1,30])
    _input[0,0] = -1.
    _input[0,-1] = 1.
    _result_tuple__tensor__list = get_mask_of_top_element__rough(_input,top_ratio=0.5, epsilon__for_binary_search=0.01, \
                    _needs_log__binary_search_in_loop = True,   _needs_log__error_ratio_in_loop = True, )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                        }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"
    assert isinstance(_log["in_loop"], list)
    _temp_log_len_for_epsilon_0_01 = _log["in_loop"].__len__()
    
    _result_tuple__tensor__list = get_mask_of_top_element__rough(_input,top_ratio=0.5, epsilon__for_binary_search=0.1, \
                    _needs_log__binary_search_in_loop = True,   _needs_log__error_ratio_in_loop = True, )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                        }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"
    assert isinstance(_log["in_loop"], list)
    _temp_log_len_for_epsilon_0_1 = _log["in_loop"].__len__()
    
    _result_tuple__tensor__list = get_mask_of_top_element__rough(_input,top_ratio=0.5, epsilon__for_binary_search=1., \
                    _needs_log__binary_search_in_loop = True,   _needs_log__error_ratio_in_loop = True, )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                        }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"
    assert isinstance(_log["in_loop"], list)
    _temp_log_len_for_epsilon_1 = _log["in_loop"].__len__()
    
    assert _temp_log_len_for_epsilon_0_01>_temp_log_len_for_epsilon_0_1
    assert _temp_log_len_for_epsilon_0_1>_temp_log_len_for_epsilon_1
    
    
        
        
    #epsilon__for_binary_search as tensor
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,1,1,1,1]]),top_ratio=0.5, \
                                                                    epsilon__for_binary_search=torch.tensor(0.001), \
                                                                _needs_log__binary_search_in_loop = True,  )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                            }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"



    #gpu
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,2,3]], device='cuda'),top_ratio=0.25)
    assert _result_tuple__tensor__list[0].device.type == "cuda"
    assert _result_tuple__tensor__list[0].eq(torch.tensor([False,False,True ],device='cuda')).all()
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,2,3]],device='cuda'),top_ratio=0.75)
    assert _result_tuple__tensor__list[0].device.type == "cuda"
    assert _result_tuple__tensor__list[0].eq(torch.tensor([False,True,True ],device='cuda')).all()
    #gpu has no log.
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,2,3]],device='cuda'), _needs_log__before_loop = True)
    assert _result_tuple__tensor__list[1] is None
    
        
    #error_of_ratio__at_least
    the_linspace = torch.linspace(1.,100.,99).reshape([1,-1])
    _result_tuple__tensor__list = get_mask_of_top_element__rough(the_linspace,top_ratio=0.2, error_of_ratio__at_least=0.01, \
                                                                    _needs_log__basic_of_loop = True)
    _temp_int = _result_tuple__tensor__list[0].sum().item()
    assert _temp_int>20-2 and _temp_int<20+2
    _log = _result_tuple__tensor__list[1]
    assert _log
    _log_len_for__error_ratio_0_01 = _log["in_loop"].__len__()
    
    _result_tuple__tensor__list = get_mask_of_top_element__rough(the_linspace,top_ratio=0.2, error_of_ratio__at_least=0.1, \
                                                                    _needs_log__basic_of_loop = True)
    _temp_int = _result_tuple__tensor__list[0].sum().item()
    assert _temp_int>20-11 and _temp_int<20+11
    _log = _result_tuple__tensor__list[1]
    assert _log
    _log_len_for__error_ratio_0_1 = _log["in_loop"].__len__()
    
    assert _log_len_for__error_ratio_0_01>_log_len_for__error_ratio_0_1
    

    
    # careful_level
    # when the binary search doesn't work stably, this careful_level controls the second way of exit.
    # when the loop repeats too much without any progress, the real error_of_ratio(or maybe something else) is modified to 
    # eventually break the loop. The result maybe rougher, but at least, you get some result.
    _input = torch.zeros(size=[1,100])
    _input[0,0] = 99999
    #step into the function and see how it works.
    _result_tuple__tensor__list = get_mask_of_top_element__rough(_input, top_ratio=0.5, error_of_ratio__at_least=0.0000001, \
                                        careful_level = 1, 
                                            _needs_log__basic_of_loop = True, _needs_log__binary_search_in_loop = True, \
                                            _needs_log__error_ratio_in_loop = True)
    _log = _result_tuple__tensor__list[1]
    assert _log
    _from_list = [0.005,0.005,0.01, 0.02,0.04,0.08,0.16]
    _to_list =   [0.005, 0.01,0.02, 0.04,0.08,0.16,0.32]
    _epsi_list = [0.001,0.001,0.001,0.01,0.01,0.01,0.01]
    
    for ii in range((180-16)//22):
        log_index = ii*22 +16
        _log_item = _log["in_loop"][log_index]
        _pos_0 = _log_item.find("error_of_ratio__b, from:tensor([", 0)
        assert _pos_0 == 0
        _pos_1 = _log_item.find("]), to:", 32)
        _pos_2 = _log_item.find("])", 40)
        #aaaaaa = _log_item[32:_pos_1]
        _number_from = float(_log_item[32:_pos_1])
        _float_equal(_number_from, _from_list[ii], epsilon=_epsi_list[ii])
        #bbbbbb = _log_item[_pos_1+15:_pos_2]
        _number_to = float(_log_item[_pos_1+15:_pos_2])
        _float_equal(_number_to, _to_list[ii], epsilon=_epsi_list[ii])
        pass
        
    
    
    # batch>1
    the_tensor = torch.tensor( [[1.,2,3,4,5],
                                [5.,2,3,4,1]])
    the_result = torch.tensor([[False,False,False,False,True],
                                [True,False,False,False,False]])
    _result_tuple__tensor__list = get_mask_of_top_element__rough(the_tensor, top_ratio=0.01)
    assert _result_tuple__tensor__list[0].eq(the_result).all()
    the_tensor = torch.tensor( [[1.,2,3,4,5],
                                [5.,2,3,4,1]], device='cuda')
    the_result = torch.tensor([[False,False,False,False,True],
                                [True,False,False,False,False]], device='cuda')
    _result_tuple__tensor__list = get_mask_of_top_element__rough(the_tensor, top_ratio=0.01)
    assert _result_tuple__tensor__list[0].eq(the_result).all()
    pass
if "old version of get_mask_of_top_element__rough function" and False:
    def 应该是不用了get_top_ratio如果没改就不要了_上面已经搞定了(input:torch.Tensor, top_ratio = 0.5, error_of_ratio = 0.01, \
                                bottom = False)->torch.Tensor:
        ''' 
        return shape is the same as input. dtype of return is torch.bool.
        
        if shape is too small, this may not work.
        '''
        assert input.shape.__len__()==2
        nelement_per_batch__s = input.shape[1]
        with torch.no_grad():
            #safety first
            _at_least_this_amount__cpu_int = int(nelement_per_batch__s*(top_ratio - error_of_ratio)+0.4999999999999)
            at_least_this_amount__s = torch.tensor(_at_least_this_amount__cpu_int, device=input.device)
            _at_most_this_amount__cpu_int =  int(nelement_per_batch__s*(top_ratio + error_of_ratio)+0.4999999999999)
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
            assert error_of_ratio>=0.
            
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
    pass# old code.        

def _raw_log10_avg_safe__with_batch(input:torch.Tensor, top_ratio = 0.9, careful_level:int|None=None)->torch.Tensor:
    '''check out the log10 measurement.py to see how this measurement function helps.
    
    This is the old version. Result is slightly different from the new version.
    
    If you are not using torch, or a version much newer than 2.9, maybe this version is better?
    This version output the measurement for each batch. If the entire thing is in 1 batch, 
    use the new version(log10_avg_safe), it's always faster.
    
    I found the new way to accelerate this function is always gpu+torch.sort
    So, if a data is on cpu, so move it to gpu and sort it and get the amount, avg, return.
    Yeah, the old my version retired. No need for a mask anymore.
    
    
    old docs below. 
    
    Calcs the average of log10 of abs of input. 
    
    The least log intermediate results are ignored. Because if a number is very close to 0, 
    the log10 of it is very negative, and any noise on such elements will introduce a bit 
    noise into the final result. So they are ignored.

    Inside this function, it calls get_mask_of_top_element__rough to help filter the
    bad intermediate results. That function also helps in a lot other cases.
    
    This function is mainly designed to help extract info from tensors, and to help
    measure some aspects in neural network training dynamics.
    
    If you don't like the shape convention and you know what you are doing, fell free
    to modify this function and the inner get_mask_of_top_element__rough function.
    '''
    #assert input.shape.__len__() == 2, "my convention, shape is [batch, anything]"
    assert input.shape.__len__() <=2
    
    if input.shape.__len__() == 1:
        ori_shape_is_1d = True
        input = input.reshape([1,-1])
        pass
    else:
        ori_shape_is_1d = False
        pass
    
    with torch.no_grad():
        log_of_input = input.abs().log10()
        #safety
        no_nan_log = log_of_input.nan_to_num(-999999.,posinf=-999999.,neginf=-999999.) 
        #safe_log is safe.
        useful_flag:torch.Tensor = get_mask_of_top_element__rough(no_nan_log, top_ratio = top_ratio,\
                                                                    careful_level=careful_level)[0]
        _masked_tensor = torch.masked.masked_tensor(no_nan_log,useful_flag)
        _masked_mean = _masked_tensor.mean(dim=1)
        assert hasattr(_masked_mean, "_masked_data")
        assert hasattr(_masked_mean, "_masked_mask")
        _masked_mean_data:torch.Tensor = _masked_mean._masked_data
        _masked_mean_data[_masked_mean._masked_mask.logical_not()] = torch.nan
        
        if ori_shape_is_1d:
            input = input.reshape([-1])
            pass
        
        return _masked_mean_data
    pass#end of function
if "top ratio" and __DEBUG_ME__() and False:
    "result. diff from 0.99 to 0.999 is 0.010, "
    "                  0.9  to 0.99  is 0.099, "
    "                  0.6  to 0.9   is 0.184,   this holds across all the scale_factor from 1e-3 to 1e3."
    "this result is only for the randn. It's a bit different for rand."
    #prin(-3.268+3.258,-3.258+3.159,-3.159+2.975,)
    #prin(-0.268+0.258,-0.258+0.159,-0.159- 0.025,)
    #prin(2.732-2.742,2.742-2.841,2.841-3.025,)
    
    
    #randn
    # scale_factor= 0.001, top_ratio=0.999, avg=-3.268, std=0.0078
    # scale_factor= 0.001, top_ratio=0.990, avg=-3.258, std=0.0115
    # scale_factor= 0.001, top_ratio=0.980, avg=-3.243, std=0.0086
    # scale_factor= 0.001, top_ratio=0.970, avg=-3.229, std=0.0081
    # scale_factor= 0.001, top_ratio=0.960, avg=-3.216, std=0.0076
    # scale_factor= 0.001, top_ratio=0.900, avg=-3.159, std=0.0094
    # scale_factor= 0.001, top_ratio=0.600, avg=-2.975, std=0.0062
    # scale_factor=   1.0, top_ratio=0.999, avg=-0.268, std=0.0079
    # scale_factor=   1.0, top_ratio=0.990, avg=-0.258, std=0.0113
    # scale_factor=   1.0, top_ratio=0.980, avg=-0.243, std=0.0086
    # scale_factor=   1.0, top_ratio=0.970, avg=-0.229, std=0.0082
    # scale_factor=   1.0, top_ratio=0.960, avg=-0.216, std=0.0076
    # scale_factor=   1.0, top_ratio=0.900, avg=-0.159, std=0.0098
    # scale_factor=   1.0, top_ratio=0.600, avg= 0.025, std=0.0062
    # scale_factor=1000.0, top_ratio=0.999, avg= 2.732, std=0.0079
    # scale_factor=1000.0, top_ratio=0.990, avg= 2.742, std=0.0113
    # scale_factor=1000.0, top_ratio=0.980, avg= 2.757, std=0.0086
    # scale_factor=1000.0, top_ratio=0.970, avg= 2.771, std=0.0082
    # scale_factor=1000.0, top_ratio=0.960, avg= 2.784, std=0.0077
    # scale_factor=1000.0, top_ratio=0.900, avg= 2.841, std=0.0098
    # scale_factor=1000.0, top_ratio=0.600, avg= 3.025, std=0.0062
    
    #rand
    # scale_factor= 0.001, top_ratio=0.999, avg=-3.423, std=0.0074
    # scale_factor= 0.001, top_ratio=0.990, avg=-3.417, std=0.0083
    # scale_factor= 0.001, top_ratio=0.980, avg=-3.402, std=0.0087
    # scale_factor= 0.001, top_ratio=0.970, avg=-3.388, std=0.0075
    # scale_factor= 0.001, top_ratio=0.960, avg=-3.377, std=0.0068
    # scale_factor= 0.001, top_ratio=0.900, avg=-3.323, std=0.0080
    # scale_factor= 0.001, top_ratio=0.600, avg=-3.169, std=0.0048
    # scale_factor=   1.0, top_ratio=0.999, avg=-0.423, std=0.0074
    # scale_factor=   1.0, top_ratio=0.990, avg=-0.417, std=0.0083
    # scale_factor=   1.0, top_ratio=0.980, avg=-0.402, std=0.0087
    # scale_factor=   1.0, top_ratio=0.970, avg=-0.388, std=0.0074
    # scale_factor=   1.0, top_ratio=0.960, avg=-0.377, std=0.0070
    # scale_factor=   1.0, top_ratio=0.900, avg=-0.323, std=0.0079
    # scale_factor=   1.0, top_ratio=0.600, avg=-0.169, std=0.0051
    # scale_factor=1000.0, top_ratio=0.999, avg= 2.577, std=0.0074
    # scale_factor=1000.0, top_ratio=0.990, avg= 2.583, std=0.0083
    # scale_factor=1000.0, top_ratio=0.980, avg= 2.598, std=0.0087
    # scale_factor=1000.0, top_ratio=0.970, avg= 2.612, std=0.0075
    # scale_factor=1000.0, top_ratio=0.960, avg= 2.623, std=0.0069
    # scale_factor=1000.0, top_ratio=0.900, avg= 2.677, std=0.0081
    # scale_factor=1000.0, top_ratio=0.600, avg= 2.831, std=0.0045
    
    def ____test____top_ratio_scan_____raw_log10_avg_safe__with_batch():
        for scale_factor in [1e-3,1.,1e3]:
            #for top_ratio in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 ,0.1]:
            #for top_ratio in [0.999, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.8, 0.7, 0.6, 0.5]:
            for top_ratio in [0.999, 0.99, 0.98, 0.97, 0.96, 0.9, 0.6]:
                test_time = 100
                _raw_result_of__mean = torch.empty(size=[test_time])
                _raw_result_of__std = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    some_randn = torch.randn(size=[100, 10000], device='cuda')*scale_factor
                    _temp_result = _raw_log10_avg_safe__with_batch(some_randn, top_ratio=top_ratio)
                    
                    _the_mean = _temp_result.mean().cpu().item()
                    #assert _float_equal(_the_mean, -0.16, 0.02)
                    _raw_result_of__mean[test_count] = _the_mean
                    
                    _the_std = _temp_result.std().cpu().item()
                    #assert _the_std<0.02
                    _raw_result_of__std[test_count] = _the_std
                    pass
                    
                print(f"scale_factor={scale_factor:6}, top_ratio={top_ratio:.3f}, avg={_raw_result_of__mean.mean():.3f}, std={_raw_result_of__std.mean():.4f}")
                pass
            pass
        return 
    ____test____top_ratio_scan_____raw_log10_avg_safe__with_batch()
    pass
if "basic behavior test         come back later" and __DEBUG_ME__() and False:
    def ____test____basic_behavior_of_____raw_log10_avg_safe__with_batch():
        _input = torch.ones(size=[1,20])
        _input = _input + torch.randn_like(_input)*0.001
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([0.]),epsilon=0.01)
        _input = torch.ones(size=[1,20])*-1.
        _input = _input + torch.randn_like(_input)*0.001
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([0.]),epsilon=0.01)
        

        _input = torch.ones(size=[1,11])
        _input[0,0] = 0.
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([0.]),epsilon=0.01)
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-10
        _result = _raw_log10_avg_safe__with_batch(_input)
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([0.]),epsilon=0.01)
        
        # 1w 继续。
        # 今天要做的。
        # 做成2个完全不同的版本，
        # 新版本不能输出batch。
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-21
        _input[0,1] = 1e-10
        _result = _raw_log10_avg_safe__with_batch(_input)
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([-1.]),epsilon=0.01)
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-10
        _input[0,1] = 1e10
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([1.]),epsilon=0.01)

        
        for _ in range(11):
            _rand_number = random.random()
            _input = torch.ones(size=[1,11])
            _input[0,0] = 1e-10
            _input[0,1] = 10**_rand_number
            assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([_rand_number/10.]),epsilon=0.01)
            pass
        
        
        
        #batch>1
        _input = torch.ones(size=[6,11])
        _input[0] = _input[0] + torch.randn(size=[1,11])*0.001
        _input[1] = _input[1]*-1 + torch.randn(size=[1,11])*0.001
        _input[2,0] = 0.
        _input[3,0] = 1e-10
        _input[4,0] = 1e-21
        _input[4,1] = 1e-10
        _input[5,0] = 1e-10
        _input[5,1] = 1e10
        _answer = torch.tensor([0., 0., 0., 0., -1., 1.])
        a = _raw_log10_avg_safe__with_batch(_input)
        _result = _raw_log10_avg_safe__with_batch(_input)
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), _answer,epsilon=0.01)
        
        #about the stability
        _input = torch.randn(size=[1,10000])
        _ref_answer = _raw_log10_avg_safe__with_batch(_input)
        _input = torch.randn(size=[1,1000])
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), _ref_answer,epsilon=0.03)
        _input = torch.randn(size=[1,100000])
        a = _raw_log10_avg_safe__with_batch(_input)
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), _ref_answer,epsilon=0.018)
        
        _dim_list =     [1e2,   1e3,    1e4,    1e5,    1e6,  ]#  1e7] # the last one is too slow.
        _mean_list =    [-0.16, -0.16,  -0.16,  -0.16,  -0.16,]#  -0.2]
        _mean_epsilon = [0.005, 0.005,  0.005,  0.005,  0.002,]#  0.005]
        _std_max =      [0.05,  0.02,   0.015,   0.015,  0.04,]#   0.07]
        for ii in range(_dim_list.__len__()):
            _dim = int(_dim_list[ii])
            _input = torch.randn(size=[100,_dim], device='cuda')
            _output = _raw_log10_avg_safe__with_batch(_input)
            mean_of_output = _output.mean().cpu()
            _tensor_equal(mean_of_output.reshape([1]), [_mean_list[ii]], _mean_epsilon[ii])    
            std_of_output = _output.std().cpu()
            assert std_of_output.lt(_std_max[ii])
            #prin(f"{_dim}  {mean_of_output}  {std_of_output}")
            pass
        
        return 
    
    ____test____basic_behavior_of_____raw_log10_avg_safe__with_batch()
    pass

def log10_avg_safe__with_batch(input:torch.Tensor, careful_level:int|None=None)->torch.Tensor:
    '''based on the test in top ratio, this function adjust the result automatically. 
    It call the inner raw function twice.'''
    
    "result. diff from 0.99 to 0.999 is 0.010, "
    "                  0.9  to 0.99  is 0.099, "
    "                  0.6  to 0.9   is 0.184,   this holds across all the scale_factor from 1e-3 to 1e3."
    "this result is only for the randn. It's a bit different for rand."
    
    _the_0_6_result = _raw_log10_avg_safe__with_batch(input=input,top_ratio=0.6,careful_level=careful_level)
    _the_0_9_result = _raw_log10_avg_safe__with_batch(input=input,top_ratio=0.9,careful_level=careful_level)
    _diff = _the_0_9_result-_the_0_6_result
    _diff = _diff*(0.110/0.184)
    result = _the_0_9_result+_diff
    return result
if "test" and __DEBUG_ME__() and False:
    # scale_factor= 0.001, avg=-3.268, std=0.0145
    # scale_factor=   1.0, avg=-0.268, std=0.0147
    # scale_factor=1000.0, avg= 2.732, std=0.0143
    def ____test____log10_avg_safe__with_batch():
        for scale_factor in [1e-3,1.,1e3]:
            test_time = 100
            _raw_result_of__mean = torch.empty(size=[test_time])
            _raw_result_of__std = torch.empty(size=[test_time])
            for test_count in range(test_time):
                some_randn = torch.randn(size=[100, 10000], device='cuda')*scale_factor
                _temp_result = log10_avg_safe__with_batch(some_randn)
                
                _the_mean = _temp_result.mean().cpu().item()
                #assert _float_equal(_the_mean, -0.16, 0.02)
                _raw_result_of__mean[test_count] = _the_mean
                
                _the_std = _temp_result.std().cpu().item()
                #assert _the_std<0.02
                _raw_result_of__std[test_count] = _the_std
                pass
                
            print(f"scale_factor={scale_factor:6}, avg={_raw_result_of__mean.mean():.3f}, std={_raw_result_of__std.mean():.4f}")
            pass
        return
    ____test____log10_avg_safe__with_batch()
    pass

# ^^^ old   ///   vvv new

def avg_of_top(input:torch.Tensor, top_ratio = 0.9, greater_true_smaller_false = True)->torch.Tensor:
    assert input.shape.__len__() == 1
    n_elements_needed = int(input.nelement()*top_ratio+0.5)
    if n_elements_needed < 1:
        n_elements_needed = 1
        pass
    
    _sorted = input.sort(descending=greater_true_smaller_false).values
    _before_mean = _sorted[:n_elements_needed]
    the_mean = _before_mean.mean()#last dim
    return the_mean
def avg_of_bottom(input:torch.Tensor, bottom_ratio = 0.9)->torch.Tensor:
    return avg_of_top(input, bottom_ratio, False)
if "test" and __DEBUG_ME__() and False:
    def ____test____avg_of_one_side():
        "avg_of_top"
        _temp_random = torch.rand(size=[90])+1.
        _ref = _temp_random.mean()
        _temp_list = (_temp_random).tolist()
        _temp_list.extend((torch.rand(size=[10])).tolist())
        random.shuffle(_temp_list)
        random.shuffle(_temp_list)
        random.shuffle(_temp_list)
        input = torch.tensor(_temp_list)
        result = avg_of_top(input)
        assert _tensor_equal(result, _ref)
        
        "avg_of_bottom"
        _temp_random = torch.rand(size=[90])
        _ref = _temp_random.mean()
        _temp_list = (_temp_random).tolist()
        _temp_list.extend((torch.rand(size=[10])+1.).tolist())
        random.shuffle(_temp_list)
        random.shuffle(_temp_list)
        random.shuffle(_temp_list)
        input = torch.tensor(_temp_list)
        result = avg_of_bottom(input)
        assert _tensor_equal(result, _ref)
        result = avg_of_top(input, greater_true_smaller_false = False)
        assert _tensor_equal(result, _ref)
        
        return 
    
    for _ in range(1):
        ____test____avg_of_one_side()
        pass
    pass

def _raw_log10_avg_safe(input:torch.Tensor, top_ratio = 0.9, recommended_gpu_device:torch.device = 'cuda')->torch.Tensor:
    '''check out the log10 measurement.py to see how this measurement function helps.
    
    This is the new version. 
    
    I found the new way to accelerate this function is always gpu+torch.sort
    So, if a data is on cpu, so move it to gpu and sort it and get the amount, avg, return.
    Yeah, the old my version retired. No need for a mask anymore.
    
    this function treat the entire tensor as a vector. It only returns 1 number as result.
    
    
    old docs below. 
    
    Calcs the average of log10 of abs of input. 
    
    The least log intermediate results are ignored. Because if a number is very close to 0, 
    the log10 of it is very negative, and any noise on such elements will introduce a bit 
    noise into the final result. So they are ignored.

    Inside this function, it calls get_mask_of_top_element__rough to help filter the
    bad intermediate results. That function also helps in a lot other cases.
    
    This function is mainly designed to help extract info from tensors, and to help
    measure some aspects in neural network training dynamics.
    
    If you don't like the shape convention and you know what you are doing, fell free
    to modify this function and the inner get_mask_of_top_element__rough function.
    '''
    #assert input.shape.__len__() <= 2, "my convention, shape is [batch, anything]"
    #ori_shape = input.shape
    
    if isinstance(recommended_gpu_device, str):
        assert recommended_gpu_device != 'cpu', "I tested, it's slower, or run any benchmark first and decide."
        pass
    else:
        assert recommended_gpu_device.type != 'cpu', "I tested, it's slower, or run any benchmark first and decide."
        pass
    
    ori_cpu = (input.device.type == 'cpu')
    input = input.to(device=recommended_gpu_device)
    with torch.no_grad():
        log_of_input = input.abs().log10()
        
        #safety
        #log only returns -torch.inf, torch.nan.    only +inf when input +inf. They all too wrong, let's remove them all.
        no_nan_log = log_of_input.nan_to_num( -999.,posinf=-999.,neginf=-999.) 
        _flag_wrong = no_nan_log.lt(-998.)
        if _flag_wrong.any():
            no_nan_log = no_nan_log[_flag_wrong.logical_not()]
            pass
        #safe_log is safe.
        #assert False
        
        the_mean = avg_of_top(no_nan_log.reshape([-1]), top_ratio=top_ratio)
        #old, now in a new function
        # n_elements_needed = int(input.nelement()*top_ratio+0.5)
        # if n_elements_needed < 1:
        #     n_elements_needed = 1
        #     pass
        # 
        # no_nan_log = no_nan_log.sort(descending=True).values[:n_elements_needed]
        # the_mean = no_nan_log.mean()#last dim
        
        #now data is on gpu.
        if ori_cpu:
            return the_mean.to('cpu')
        return the_mean
    pass#end of function
"Bc random.py imports this file. So this test is done here, with a function in random.py copy pasted here."
if "device adaption" and __DEBUG_ME__() and False:
    def ____test____log10_avg_safe____device_adaption():
        a = torch.tensor([1.])
        b = _raw_log10_avg_safe(a)
        assert a.device.type == 'cpu'
        assert a.shape == torch.Size([1])
        assert b.device.type == 'cpu'
        
        a = torch.tensor([[1.]], device='cuda')
        b = _raw_log10_avg_safe(a)
        assert a.device.type == 'cuda'
        assert a.shape == torch.Size([1,1])
        assert b.device.type == 'cuda'
        
        return 
    ____test____log10_avg_safe____device_adaption()
    pass
if "basic behavior test" and __DEBUG_ME__() and False:
    def ____test____basic_behavior_of____log10_avg_safe():
        if False:
            _input = torch.ones(size=[1,20])
            _input = _input + torch.randn_like(_input)*0.001
            assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([0.]),epsilon=0.01)
            _input = torch.ones(size=[1,20])*-1.
            _input = _input + torch.randn_like(_input)*0.001
            assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([0.]),epsilon=0.01)
            

            _input = torch.ones(size=[1,11])
            _input[0,0] = 0.
            assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([0.]),epsilon=0.01)
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-10
        _result = _raw_log10_avg_safe(_input)
        assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([0.]),epsilon=0.01)
        
        # 1w 继续。
        # 今天要做的。
        # 做成2个完全不同的版本，
        # 新版本不能输出batch。
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-21
        _input[0,1] = 1e-10
        _result = _raw_log10_avg_safe(_input)
        assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([-1.]),epsilon=0.01)
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-10
        _input[0,1] = 1e10
        assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([1.]),epsilon=0.01)

        
        for _ in range(11):
            _rand_number = random.random()
            _input = torch.ones(size=[1,11])
            _input[0,0] = 1e-10
            _input[0,1] = 10**_rand_number
            assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([_rand_number/10.]),epsilon=0.01)
            pass
        
        
        # only the with batch version works with this.
        # #batch>1
        # _input = torch.ones(size=[6,11])
        # _input[0] = _input[0] + torch.randn(size=[1,11])*0.001
        # _input[1] = _input[1]*-1 + torch.randn(size=[1,11])*0.001
        # _input[2,0] = 0.
        # _input[3,0] = 1e-10
        # _input[4,0] = 1e-21
        # _input[4,1] = 1e-10
        # _input[5,0] = 1e-10
        # _input[5,1] = 1e10
        # _answer = torch.tensor([0., 0., 0., 0., -1., 1.])
        # a = log10_avg_safe(_input)
        # _result = log10_avg_safe(_input)
        # assert _tensor_equal(log10_avg_safe(_input), _answer,epsilon=0.01)
        #
        #about the stability
        # _input = torch.randn(size=[1,10000])
        # _ref_answer = log10_avg_safe(_input)
        # _input = torch.randn(size=[1,1000])
        # assert _tensor_equal(log10_avg_safe(_input), _ref_answer,epsilon=0.03)
        # _input = torch.randn(size=[1,100000])
        # a = log10_avg_safe(_input)
        # assert _tensor_equal(log10_avg_safe(_input), _ref_answer,epsilon=0.018)
        
        # _dim_list =     [1e2,   1e3,    1e4,    1e5,    1e6,  ]#  1e7] # the last one is too slow.
        # _mean_list =    [-0.16, -0.16,  -0.16,  -0.16,  -0.16,]#  -0.2]
        # _mean_epsilon = [0.005, 0.005,  0.005,  0.005,  0.002,]#  0.005]
        # _std_max =      [0.05,  0.02,   0.015,   0.015,  0.04,]#   0.07]
        # for ii in range(_dim_list.__len__()):
        #     _dim = int(_dim_list[ii])
        #     _input = torch.randn(size=[100,_dim], device='cuda')
        #     _output = log10_avg_safe(_input)
        #     mean_of_output = _output.mean().cpu()
        #     _tensor_equal(mean_of_output.reshape([1]), [_mean_list[ii]], _mean_epsilon[ii])    
        #     std_of_output = _output.std().cpu()
        #     assert std_of_output.lt(_std_max[ii])
        #     #prin(f"{_dim}  {mean_of_output}  {std_of_output}")
        #     pass
        
        return 
    
    ____test____basic_behavior_of____log10_avg_safe()
    pass
if "top ratio scan" and __DEBUG_ME__() and False:
    "result. diff from 0.99 to 0.999 is 0.018, "
    "                  0.9  to 0.99  is 0.097, "
    "                  0.6  to 0.9   is 0.183,   this holds across all the scale_factor from 1e-3 to 1e3."
    "this result is only for the randn. It's a bit different for rand."
    # prin(-3.273+3.255,-3.255+3.158,-3.158+2.975,)
    # prin(-0.273+0.255,-0.255+0.158,-0.158- 0.025,)
    # prin(2.727-2.745,2.745-2.842,2.842-3.025,)
    
    #randn
    # scale_factor= 0.001, top_ratio=0.999, avg=-3.273, std=0.000487
    # scale_factor= 0.001, top_ratio=0.990, avg=-3.255, std=0.000469
    # scale_factor= 0.001, top_ratio=0.980, avg=-3.240, std=0.000452
    # scale_factor= 0.001, top_ratio=0.970, avg=-3.227, std=0.000436
    # scale_factor= 0.001, top_ratio=0.960, avg=-3.215, std=0.000449
    # scale_factor= 0.001, top_ratio=0.900, avg=-3.158, std=0.000408
    # scale_factor= 0.001, top_ratio=0.600, avg=-2.975, std=0.000353
    # scale_factor=   1.0, top_ratio=0.999, avg=-0.273, std=0.000501
    # scale_factor=   1.0, top_ratio=0.990, avg=-0.255, std=0.000459
    # scale_factor=   1.0, top_ratio=0.980, avg=-0.240, std=0.000445
    # scale_factor=   1.0, top_ratio=0.970, avg=-0.227, std=0.000451
    # scale_factor=   1.0, top_ratio=0.960, avg=-0.215, std=0.000448
    # scale_factor=   1.0, top_ratio=0.900, avg=-0.158, std=0.000412
    # scale_factor=   1.0, top_ratio=0.600, avg= 0.025, std=0.000333
    # scale_factor=1000.0, top_ratio=0.999, avg= 2.727, std=0.000495
    # scale_factor=1000.0, top_ratio=0.990, avg= 2.745, std=0.000478
    # scale_factor=1000.0, top_ratio=0.980, avg= 2.760, std=0.000467
    # scale_factor=1000.0, top_ratio=0.970, avg= 2.773, std=0.000445
    # scale_factor=1000.0, top_ratio=0.960, avg= 2.785, std=0.000458
    # scale_factor=1000.0, top_ratio=0.900, avg= 2.842, std=0.000413
    # scale_factor=1000.0, top_ratio=0.600, avg= 3.025, std=0.000347
    
    def ____test____top_ratio_scan_____raw_log10_avg_safe():
        for scale_factor in [1e-3,1.,1e3]:
            #for top_ratio in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 ,0.1]:
            #for top_ratio in [0.999, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.8, 0.7, 0.6, 0.5]:
            for top_ratio in [0.999, 0.99, 0.98, 0.97, 0.96, 0.9, 0.6]:
                test_time = 1000
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    some_randn = torch.randn(size=[100, 10000], device='cuda')*scale_factor
                    _temp_result = _raw_log10_avg_safe(some_randn, top_ratio=top_ratio)
                    _raw_result[test_count] = _temp_result
                    pass
                    
                print(f"scale_factor={scale_factor:6}, top_ratio={top_ratio:.3f}, avg={_raw_result.mean():.3f
                        }, std={_raw_result.std():.6f}")
                pass
            pass
        return 
    ____test____top_ratio_scan_____raw_log10_avg_safe()
    pass

def log10_avg_safe(input:torch.Tensor)->torch.Tensor:
    '''based on the test in top ratio, this function adjust the result automatically. 
    It call the inner raw function twice.'''
    
    "result. diff from 0.99 to 0.999 is 0.018, "
    "                  0.9  to 0.99  is 0.097, "
    "                  0.6  to 0.9   is 0.183,   this holds across all the scale_factor from 1e-3 to 1e3."
    "this result is only for the randn. It's a bit different for rand."
    
    _the_0_6_result = _raw_log10_avg_safe(input=input,top_ratio=0.6)
    _the_0_9_result = _raw_log10_avg_safe(input=input,top_ratio=0.9)
    _diff = _the_0_9_result-_the_0_6_result
    _diff = _diff*(0.117/0.183)
    result = _the_0_9_result+_diff
    return result
if "test" and __DEBUG_ME__() and False:
    # scale_factor= 0.001, avg=-3.268, std=0.0145
    # scale_factor=   1.0, avg=-0.268, std=0.0147
    # scale_factor=1000.0, avg= 2.732, std=0.0143
    def ____test____log10_avg_safe():
        for scale_factor in [1e-3,1.,1e3]:
            test_time = 1000
            _raw_result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                some_randn = torch.randn(size=[100, 10000], device='cuda')*scale_factor
                _temp_result = log10_avg_safe(some_randn)
                _raw_result[test_count] = _temp_result
                pass
                
            print(f"scale_factor={scale_factor:6}, avg={_raw_result.mean():.3f
                    }, std={_raw_result.std():.6f}")
            pass
        return
    ____test____log10_avg_safe()
    pass











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


if "oops. Use tolist() instead. Tensor has it." and False:
    def debug_Rank_1_parameter_to_List_float(input:torch.nn.parameter.Parameter)->List[float]:
        result : List[float] = []
        for i in range(input.shape[0]):
            result.append(input[i].item())
            pass
        return result
    pass






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
    for _ in range(dim+int(torch.randint(0,dim,[1]).item())):
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
        the_exp = 0.
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

