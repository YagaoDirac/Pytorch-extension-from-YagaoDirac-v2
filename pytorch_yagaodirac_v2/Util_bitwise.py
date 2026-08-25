import time
from typing import List, Tuple, Optional, Literal
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
    
    