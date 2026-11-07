from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _bool_equal___0_as_false, all_False, \
        _tensor_equal, _tensor_shape_check, iota, is_list_int, _float_equal
#         iota, str_the_list


import random
import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######




def str_1d_bool_tensor_as_if_its_binary(input:torch.Tensor, pad_to_length = 0)->str:
    assert input.shape.__len__() == 1
    assert isinstance(input, torch.Tensor)
    bool_py_list = input.tolist()

    char_py_list = []
    for the_bool in bool_py_list:
        if the_bool:
            char_py_list.append("1")
            pass
        else:
            char_py_list.append("0")
            pass
        pass#for the_bool
    raw_str = "".join(char_py_list)

    needs_zeros = pad_to_length - raw_str.__len__()
    if needs_zeros>0:
        return "0"*needs_zeros+raw_str
    else:
        return raw_str
    pass#end of function

def bool_tensor_into_int_list(input:torch.Tensor, dtype = torch.int64)->list[int]:#torch.Tensor this is v2, but it doesn't handle bit int:
    '''notice, int("") doesn't work. So is input is empty, this function doesn't work.'''
    assert input.shape.__len__() == 2
    result_list:list[int] = []
    for the_slice in input:
        temp_str = str_1d_bool_tensor_as_if_its_binary(the_slice)
        result_list.append(int(temp_str, base=2))# notice, int("") doesn't work. So is input is empty, this function doesn't work.
        pass
    return result_list

'debug and print'
def print_2d_bool_tensor(the_tensor:torch.Tensor, true_as = "1", false_as = "0", print_directly = True)->list[str]:
    assert true_as  != "0"
    assert false_as != "1"
    _temp___some_special_chunk___in_list_int = bool_tensor_into_int_list(the_tensor)
    length = the_tensor.shape[1]
    all_the_str:list[str] = []
    for _temp_element in _temp___some_special_chunk___in_list_int:
        _temp__str = bin(_temp_element)[2:].rjust(length, "0")
        _temp__str = _temp__str.replace("1", true_as)
        _temp__str = _temp__str.replace("0", false_as)
        all_the_str.append(_temp__str)
        pass
    if print_directly:
        print_this = "\n".join(all_the_str)
        print(print_this)
        pass
    return all_the_str#end of function











if "pytorch     detach feature test" and True:
    def ____test____detach_feature_test():
        if "self - self trick" and False:
            for _iiii in range(432):
                I_want_this_result = torch.randn(size=[1])
                input = torch.randn(size=[1], requires_grad=True)
                output = I_want_this_result + (input + -1.*input.detach())
                assert output == I_want_this_result

                grad = torch.randn(size=[1])
                output.backward(gradient=grad, inputs=[input])
                assert input.grad == grad
                pass#for _iiii
            pass#/test

        if "advanced self - self trick" and True:
            for _iiii in range(432):
                scale_grad_with_this_factor = torch.randn(size=[1])

                I_want_this_result = torch.randn(size=[1])
                input = torch.randn(size=[1], requires_grad=True)
                output = I_want_this_result + ( scale_grad_with_this_factor*input + \
                                            -1.*scale_grad_with_this_factor*input.detach())
                assert _tensor_equal(output, I_want_this_result)
                assert output == I_want_this_result

                grad = torch.randn(size=[1])
                output.backward(gradient=grad, inputs=[input])
                assert _tensor_equal(input.grad, grad * scale_grad_with_this_factor)
                assert input.grad == grad * scale_grad_with_this_factor
                pass#for _iiii


            for _iiii in range(432):
                result_is_multiple_of_input = 5
                grad_is_multiple_of_input = 3

                input = torch.randn(size=[1], requires_grad=True)
                output =                                grad_is_multiple_of_input   *input + \
                        (result_is_multiple_of_input -  grad_is_multiple_of_input)  *input.detach()
                assert _tensor_equal(output, input*result_is_multiple_of_input)
                # assert output == input*result_is_multiple_of_input  #  probably the rounding up error.

                grad = torch.randn(size=[1])
                output.backward(gradient=grad, inputs=[input])
                assert _tensor_equal(input.grad, grad * grad_is_multiple_of_input)
                assert input.grad == grad * grad_is_multiple_of_input
                pass#for _iiii
            pass#/test

        if "detach the denominator trick" and False:
            for _iiii in range(555):  

                #  init and forward
                while True:
                    aaa = torch.randn(size=[1], requires_grad=True)
                    bbb = torch.randn(size=[1], requires_grad=True)
                    the_sum = aaa+bbb                ##############       no detach
                    if the_sum.abs()>0.1:  #  safety
                        break
                    pass#while True
                output = aaa/the_sum
                #  backward
                grad = torch.randn(size=[1], requires_grad=True)
                output.backward(gradient=grad, inputs=[aaa, bbb])
                assert _tensor_equal([aaa.grad], grad*bbb/(the_sum*the_sum))
                assert _tensor_equal([bbb.grad], -grad*aaa/(the_sum*the_sum))
                pass#for _iiii

            for _iiii in range(555): 
                #  init and forward
                while True:
                    aaa = torch.randn(size=[1], requires_grad=True)
                    bbb = torch.randn(size=[1], requires_grad=True)
                    the_sum = (aaa+bbb).detach()      ##############       detach
                    if the_sum.abs()>0.1:  #  safety
                        break
                    pass#while True
                output = aaa/the_sum
                #  backward
                grad = torch.randn(size=[1])
                output.backward(gradient=grad, inputs=[aaa, bbb])
                #  assert
                assert _tensor_equal([aaa.grad], grad/the_sum)
                assert bbb.grad is None
                pass#for _iiii

            pass#/test

        if "sigmoid in pytorch             no detach, ofc." and False:
            for _iiii in range(555):
                #1 aaa = torch.randn(size=[1], requires_grad=True)
                aaa = torch.randn(size=[1], requires_grad=True)
                pytorch_sigmoid = aaa.sigmoid()

                grad = torch.randn(size=[1])
                pytorch_sigmoid.backward(gradient=grad, inputs=[aaa])
                print(aaa.grad)
                assert _tensor_equal([aaa.grad], grad*(1+(-aaa).exp()).pow(-2)*(-aaa).exp())
                assert _tensor_equal([aaa.grad], grad*pytorch_sigmoid*(1.-pytorch_sigmoid))
                pass#for _iiii

            for _iiii in range(555):
                #1 aaa = torch.randn(size=[1], requires_grad=True)
                aaa = torch.randn(size=[1], requires_grad=True)
                my_sigmoid = 1./(1+(-aaa).exp())          ##########          no detach

                grad = torch.randn(size=[1])
                my_sigmoid.backward(gradient=grad, inputs=[aaa])
                print(aaa.grad)
                assert _tensor_equal([aaa.grad], grad*(1+(-aaa).exp()).pow(-2)*(-aaa).exp())
                assert _tensor_equal([aaa.grad], grad*my_sigmoid*(1.-my_sigmoid))
                pass#for _iiii

        return 
    ____test____detach_feature_test()
    pass


if "adaptive methods" and True:
    def ____adaptive_methods():
        '''conclusion:  in sub the max, only the max_element gets different grad.
        Other NON_max elements get exactly the same grad. The math formula is very clear.
        no_detach ver grad sum to 0.
        detach ver equals to the g_in.'''
        if "sub the max" and False:
            for dim in [2,3,4,6,11,31]:
                for _iii in range(66):
                    input = torch.rand(size=[dim])
                    grad___l = torch.rand(size=[dim])

                    #  no_detach
                    #  no_detach
                    no_detach_input___d = input.detach().clone()
                    no_detach_input___d.requires_grad_()
                    max_index = no_detach_input___d.argmax()
                    _temp___max_value = no_detach_input___d[max_index]   ###########     no detach    ###########
                    no_detach_result___d = no_detach_input___d - _temp___max_value
                    assert _tensor_equal(no_detach_result___d[max_index], [0.])

                    no_detach_result___d.backward(gradient=grad___l, inputs=[no_detach_input___d])
                    assert _tensor_equal(no_detach_input___d.grad.sum(), [0.])

                    #  grad formula validation
                        #  one_hot mask
                    one_hot_of_max_index = torch.zeros(size=[dim], dtype=torch.bool)
                    one_hot_of_max_index[max_index] = True
                    reverse_of_one_hot = one_hot_of_max_index.logical_not()
                    iota_of_dim = iota(dim)
                        #  /one_hot mask

                    partial_derivative__from_max_element_to_all___l = torch.zeros(size=[dim])
                        #  from_max_element_to_all == from_max_element_to_max + from_max_element_to_NON_max
                    partial_derivative__from_NON_max_element_to_max_element___l = torch.ones(size=[dim]) * -1.
                    partial_derivative__from_NON_max_element_to_max_element___l[max_index] = 0.

                    partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l = torch.ones(size=[dim])
                    partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l[max_index] = 0.

                    partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l = torch.zeros(size=[dim, dim])
                    partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[max_index, :] = 0.
                    partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[:, max_index] = 0.
                    partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[iota_of_dim, iota_of_dim] = 0.
                        #  into a buffer
                    buffer___d_d = partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l.detach().clone()
                    assert buffer___d_d.eq(0.).all()
                    buffer___d_d[max_index, :] += partial_derivative__from_max_element_to_all___l
                    buffer___d_d[:, max_index] += partial_derivative__from_NON_max_element_to_max_element___l
                    buffer___d_d[iota_of_dim, iota_of_dim] += partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l
                        #  adaption to the g_in
                        #  adaption to the g_in
                    _temp___grad___l_EXPANDl = grad___l.reshape(shape=[dim, 1]).expand([dim, dim])
                    assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                    manual_grad_for___no_detach = (buffer___d_d*_temp___grad___l_EXPANDl).sum(dim=0) 
                    del _temp___grad___l_EXPANDl
                    #del buffer___d_d
                    assert _tensor_equal(manual_grad_for___no_detach, no_detach_input___d.grad)




                    #  with_detach
                    #  with_detach
                    with_detach_input___d = input.detach().clone()
                    with_detach_input___d.requires_grad_()
                    max_index = with_detach_input___d.argmax()
                    _temp___max_value = with_detach_input___d[max_index].detach()   ###########     detach    ###########
                    with_detach_result___d = with_detach_input___d - _temp___max_value
                    assert _tensor_equal(with_detach_result___d[max_index], [0.])

                    with_detach_result___d.backward(gradient=grad___l, inputs=[with_detach_input___d])
                    assert _tensor_equal(with_detach_input___d.grad, grad___l)

                    #  grad formula validation
                        #  one_hot mask
                    one_hot_of_max_index = torch.zeros(size=[dim], dtype=torch.bool)
                    one_hot_of_max_index[max_index] = True
                    reverse_of_one_hot = one_hot_of_max_index.logical_not()
                    iota_of_dim = iota(dim)
                        #  /one_hot mask

                    partial_derivative___l = torch.ones(size=[dim])
                    buffer___d_d = torch.zeros(size=[dim, dim])
                    buffer___d_d[iota_of_dim, iota_of_dim] += partial_derivative___l
                        #  adaption to the g_in
                        #  adaption to the g_in
                    _temp___grad___l_EXPANDl = grad___l.reshape(shape=[dim, 1]).expand([dim, dim])
                    assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                    manual_grad_for___with_detach = (buffer___d_d*_temp___grad___l_EXPANDl).sum(dim=0) 
                    del _temp___grad___l_EXPANDl
                    #del buffer___d_d
                    assert _tensor_equal(manual_grad_for___with_detach, with_detach_input___d.grad)

                    #  over all test
                    #  over all test
                    assert _tensor_equal(no_detach_result___d, with_detach_result___d)
                    assert _tensor_equal(no_detach_input___d.grad.sum(), [0.])  # actually duplicated.
                    assert _tensor_equal(no_detach_input___d.grad[reverse_of_one_hot], with_detach_input___d.grad[reverse_of_one_hot])


                    # prin(no_detach_input___d.grad)
                    # prin(with_detach_input___d.grad)
                    # prin(input[max_index])
                    # prin(max_index)

                    pass#for _iii
                pass#for dim

            pass#/test
        
        
        '''conclusion:  in sub the avg, 2 grad are different. The differece between them is a constant.
        In other words, in grad_ver_1 - grad_ver_2, all elements are the same.'''
        if "sub the avg" and False:
            for dim in [3,2,4,6,11,31]:
                for _iii in range(66):
                    input = torch.rand(size=[dim])
                    grad___l = torch.rand(size=[dim])

                    #  no_detach
                    #  no_detach
                    no_detach_input___d = input.detach().clone()
                    no_detach_input___d.requires_grad_()
                    _temp___avg_value = no_detach_input___d.mean()   ###########     no detach    ###########
                    no_detach_result___d = no_detach_input___d - _temp___avg_value
                    assert _tensor_equal(no_detach_result___d.sum(), [0.])

                    no_detach_result___d.backward(gradient=grad___l, inputs=[no_detach_input___d])
                    assert _tensor_equal(no_detach_input___d.grad.sum(), [0.]) 

                    #  grad formula validation
                    iota_of_dim = iota(dim)

                    partial_derivative__same_element___l      = torch.ones(size=[dim])*      (float(dim-1)/float(dim))
                    partial_derivative__different_element___l_l = torch.ones(size=[dim, dim])* (    -1.     /float(dim))
                    partial_derivative__different_element___l_l[iota_of_dim, iota_of_dim] = 0.
                        #  into a buffer
                    buffer___d_d = partial_derivative__different_element___l_l.detach().clone()
                    buffer___d_d[iota_of_dim, iota_of_dim] += partial_derivative__same_element___l
                        #  adaption to the g_in
                        #  adaption to the g_in
                    _temp___grad___l_EXPANDl = grad___l.reshape(shape=[dim, 1]).expand([dim, dim])
                    assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                    manual_grad_for___no_detach = (buffer___d_d*_temp___grad___l_EXPANDl).sum(dim=0) 
                    del _temp___grad___l_EXPANDl
                    #del buffer___d_d
                    assert _tensor_equal(manual_grad_for___no_detach, no_detach_input___d.grad)




                    #  with_detach
                    #  with_detach
                    with_detach_input___d = input.detach().clone()
                    with_detach_input___d.requires_grad_()
                    _temp___avg_value = with_detach_input___d.mean().detach()   ###########     detach    ###########
                    with_detach_result___d = with_detach_input___d - _temp___avg_value
                    assert _tensor_equal(with_detach_result___d.sum(), [0.])

                    with_detach_result___d.backward(gradient=grad___l, inputs=[with_detach_input___d])
                    assert _tensor_equal(with_detach_input___d.grad, grad___l)

                    iota_of_dim = iota(dim)

                    partial_derivative___l = torch.ones(size=[dim])
                    buffer___d_d = torch.zeros(size=[dim, dim])
                    buffer___d_d[iota_of_dim, iota_of_dim] += partial_derivative___l
                        #  adaption to the g_in
                        #  adaption to the g_in
                    _temp___grad___l_EXPANDl = grad___l.reshape(shape=[dim, 1]).expand([dim, dim])
                    assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                    manual_grad_for___with_detach = (buffer___d_d*_temp___grad___l_EXPANDl).sum(dim=0) 
                    del _temp___grad___l_EXPANDl
                    #del buffer___d_d
                    assert _tensor_equal(manual_grad_for___with_detach, with_detach_input___d.grad)
                    #  over all test
                    #  over all test
                    assert _tensor_equal(with_detach_result___d, no_detach_result___d)
                    assert _tensor_equal(no_detach_input___d.grad.sum(), [0.])  #  duplicated
                        # the difference is a constant number.
                    _temp__grad_diff___d = with_detach_input___d.grad - no_detach_input___d.grad
                    _temp__first_element_of_grad_diff___s = _temp__grad_diff___d[0]
                    _temp__first_element_of_grad_diff___EXPANDd = _temp__first_element_of_grad_diff___s.reshape(shape=[1]).expand(size=[dim])
                    assert _tensor_equal(_temp__grad_diff___d, _temp__first_element_of_grad_diff___EXPANDd)

                    # prin(no_detach_input___d.grad)
                    # prin(with_detach_input___d.grad)
                    # prin(with_detach_input___d.grad - no_detach_input___d.grad)

                    pass#for _iii
                pass#for dim
            pass#/test
            
            
        '''conclusion:  in abs, the adaptive behavior is automatically a detach ver, there's no no_detach ver.'''


        '''conclusion:  in div the max abs, the adaptive behaviors are abs and max. The abs only has detach behavior. 
        The only difference is the max behavior. Similar to the sub the max test. 
        Only the special element gets different grad. And a lil bit extra consistance with the sign.'''
        if "div the max abs" and False:
            for dim in [5,2,4,6,11,31]:
                for _iii in range(66):
                    input = torch.randn(size=[dim])  #  randn, not rand.
                    #input = torch.tensor([10., 1, 1, 1, 2])  #  randn, not rand.
                    grad___l = torch.rand(size=[dim])#  positive only
                    #grad___l = torch.tensor([113, 1, 1, 1, 1])



                    #  no_detach
                    #  no_detach
                    no_detach_input___d = input.detach().clone()
                    no_detach_input___d.requires_grad_()
                    _temp___abs = no_detach_input___d.abs()
                    max_index_of_abs = _temp___abs.argmax()
                    sign_of_the_special_element = no_detach_input___d[max_index_of_abs].ge(0.).to(torch.int32)*2 -1# this is natually detach ver.
                    from pytorch_yagaodirac_v2.Util import _either_1_or_neg1
                    assert _either_1_or_neg1(sign_of_the_special_element)
                    divide_this = no_detach_input___d[max_index_of_abs]*sign_of_the_special_element   #######    no detach   ########
                    assert divide_this.ge(0.)
                    no_detach_result___d = no_detach_input___d/divide_this
                    assert no_detach_result___d.abs().le(1.).all()

                        #  one_hot mask
                    one_hot_of_max_index_of_abs = torch.zeros(size=[dim], dtype=torch.bool)
                    one_hot_of_max_index_of_abs[max_index_of_abs] = True
                    reverse_of_one_hot = one_hot_of_max_index_of_abs.logical_not()
                    iota_of_dim = iota(dim)
                        #  /one_hot mask

                    no_detach_result___d.backward(gradient=grad___l, inputs=[no_detach_input___d])
                        #  a bit assert
                    sign_of___other_element_grad = no_detach_input___d.grad [reverse_of_one_hot].gt(0.).to(torch.int32)*2-1
                    sign_of___other_g_in         = grad___l                 [reverse_of_one_hot].gt(0.).to(torch.int32)*2-1
                    assert sign_of___other_element_grad.eq(sign_of___other_g_in).all()

                    # this is unstable. I probably messed with the formula. no plan.
                    # sign_of___special_element_grad = no_detach_input___d.grad[one_hot_of_max_index_of_abs].gt(0.).to(torch.int32)*2-1
                    # _temp___special_element_of___input_mul_g_in = (grad___l*input)[one_hot_of_max_index_of_abs]
                    # sign_of___special_element_of___input_mul_g_in = _temp___special_element_of___input_mul_g_in.gt(0.).to(torch.int32)*2-1

                    # aaa = sign_of___special_element_grad*sign_of___special_element_of___input_mul_g_in*sign_of_the_special_element 
                    # assert sign_of___special_element_grad*sign_of___special_element_of___input_mul_g_in*sign_of_the_special_element == -1


                    #  grad formula validation
                    #  grad formula validation
                    
                    partial_derivative__SPECIAL_element_to_all___d = torch.zeros(size=[dim])
                            # partial_derivative__SPECIAL_element_to_all___l == partial_derivative__SPECIAL_element_to SPECIAL + to OTHER

                    '''partial_derivative__other_element_to_SPECIAL_element___d'''
                    _temp___original_of_special_element = no_detach_input___d[max_index_of_abs]
                    partial_derivative__other_element_to_SPECIAL_element___d = \
                            sign_of_the_special_element * no_detach_input___d * -1. /(_temp___original_of_special_element*_temp___original_of_special_element)
                    partial_derivative__other_element_to_SPECIAL_element___d[max_index_of_abs] = 0.

                    partial_derivative__other_element_to_other_element___same_element___d = torch.ones(size=[dim]) /divide_this
                    partial_derivative__other_element_to_other_element___same_element___d[max_index_of_abs] = 0.

                    partial_derivative__other_element_to_other_element___different_element___d_d = torch.zeros(size=[dim, dim])
                    partial_derivative__other_element_to_other_element___different_element___d_d[max_index_of_abs, :] = 0.
                    partial_derivative__other_element_to_other_element___different_element___d_d[:, max_index_of_abs] = 0.
                    partial_derivative__other_element_to_other_element___different_element___d_d[iota_of_dim, iota_of_dim] = 0.
                        #  into a buffer
                    buffer___d_d = partial_derivative__other_element_to_other_element___different_element___d_d.detach().clone()
                    buffer___d_d[max_index_of_abs, :] += partial_derivative__SPECIAL_element_to_all___d
                    buffer___d_d[:, max_index_of_abs] += partial_derivative__other_element_to_SPECIAL_element___d
                    buffer___d_d[iota_of_dim, iota_of_dim] += partial_derivative__other_element_to_other_element___same_element___d

                    _temp_buffer___d_d = partial_derivative__other_element_to_other_element___different_element___d_d.detach().clone()
                    _temp_buffer___d_d[max_index_of_abs, :] = torch.nan
                    _temp_buffer___d_d[:, max_index_of_abs] = torch.nan
                    _temp_buffer___d_d[iota_of_dim, iota_of_dim] = torch.nan
                    assert buffer___d_d.eq(_temp_buffer___d_d).to(torch.int32).sum() == dim*dim -dim*3 +2

                    _temp_buffer___d_d = torch.empty(size=[dim, dim], dtype=torch.float32)
                    _temp_buffer___d_d.fill_(torch.nan)
                    _temp_buffer___d_d[max_index_of_abs, :] = partial_derivative__SPECIAL_element_to_all___d
                    assert buffer___d_d.eq(_temp_buffer___d_d).to(torch.int32).sum() == dim

                    _temp_buffer___d_d = torch.empty(size=[dim, dim], dtype=torch.float32)
                    _temp_buffer___d_d.fill_(torch.nan)
                    _temp_buffer___d_d[:, max_index_of_abs] = partial_derivative__other_element_to_SPECIAL_element___d
                    _temp_buffer___d_d[max_index_of_abs, max_index_of_abs] = torch.nan
                    assert buffer___d_d.eq(_temp_buffer___d_d).to(torch.int32).sum() == dim -1

                    _temp_buffer___d_d = torch.empty(size=[dim, dim], dtype=torch.float32)
                    _temp_buffer___d_d.fill_(torch.nan)
                    _temp_buffer___d_d[iota_of_dim, iota_of_dim] = partial_derivative__other_element_to_other_element___same_element___d
                    _temp_buffer___d_d[max_index_of_abs, max_index_of_abs] = torch.nan
                    assert buffer___d_d.eq(_temp_buffer___d_d).to(torch.int32).sum() == dim -1
                        #  adaption to the g_in
                        #  adaption to the g_in
                    _temp___grad___l_EXPANDl = grad___l.reshape(shape=[dim, 1]).expand([dim, dim])
                    assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                    manual_grad_for___no_detach = (buffer___d_d*_temp___grad___l_EXPANDl).sum(dim=0) 
                    del _temp___grad___l_EXPANDl
                    #del buffer___d_d
                    assert _tensor_equal(manual_grad_for___no_detach, no_detach_input___d.grad)




                    #  with_detach
                    #  with_detach
                    with_detach_input___d = input.detach().clone()
                    with_detach_input___d.requires_grad_()
                    _temp___abs = with_detach_input___d.abs()
                    max_index_of_abs = _temp___abs.argmax()
                    sign_of_the_special_element = with_detach_input___d[max_index_of_abs].ge(0.).to(torch.int32)*2 -1# this is natually detach ver.
                    assert _either_1_or_neg1(sign_of_the_special_element)
                    divide_this = (with_detach_input___d[max_index_of_abs]*sign_of_the_special_element).detach()####### detach #######
                    assert divide_this.ge(0.)
                    with_detach_result___d = with_detach_input___d/divide_this
                    assert with_detach_result___d.abs().le(1.).all()

                    with_detach_result___d.backward(gradient=grad___l, inputs=[with_detach_input___d])
                    #  grad formula validation
                    #  grad formula validation
                    manual_grad_for___with_detach = grad___l/divide_this
                    assert _tensor_equal(manual_grad_for___with_detach, with_detach_input___d.grad)




                    #  over all test
                    #  over all test
                    assert _tensor_equal(with_detach_result___d, no_detach_result___d)
                    assert _tensor_equal(no_detach_input___d.grad[reverse_of_one_hot], with_detach_input___d.grad[reverse_of_one_hot])

                    # prin(no_detach_input___d.grad)
                    # prin(with_detach_input___d.grad)
                    # prin(with_detach_input___d.grad - no_detach_input___d.grad)

                    pass#for _iii
                pass#for dim
            pass#/test


        '''This is also the vector length into 1.'''
        if "div the rms" and False:
            for dim in [3,2,4,6,11,31]:
                iota_of_dim = iota(dim)
                for _iii in range(66):
                    input = torch.randn(size=[dim])  
                    grad___l = torch.randn(size=[dim])



                    #  no_detach
                    #  no_detach
                    no_detach_input___d = input.detach().clone()
                    no_detach_input___d.requires_grad_()
                    the_rms = (no_detach_input___d*no_detach_input___d).sum().sqrt()  #######   no detach   #######
                    no_detach_result___d = no_detach_input___d/the_rms
                        #  a bit assert
                    _temp___result_rms = (no_detach_result___d*no_detach_result___d).sum().sqrt()
                    assert _tensor_equal(_temp___result_rms, [1.])
                    del _temp___result_rms

                    no_detach_result___d.backward(gradient=grad___l, inputs=[no_detach_input___d])

                    #  grad formula validation
                    #  grad formula validation
                    
                    partial_derivative__same_element___d = (the_rms - no_detach_input___d*no_detach_input___d/the_rms)/(the_rms*the_rms)

                    '''partial_derivative__different_element___d'''
                    no_detach_input___d_EXPANDd = no_detach_input___d.reshape(shape=[dim, 1]).expand(size=[dim, dim])
                    no_detach_input___EXPANDd_d = no_detach_input___d.reshape(shape=[1, dim]).expand(size=[dim, dim])
                    partial_derivative__different_element___d = -1.*no_detach_input___d_EXPANDd*no_detach_input___EXPANDd_d/(the_rms*the_rms*the_rms)
                    partial_derivative__different_element___d[iota_of_dim, iota_of_dim] = 0.

                    buffer___d_d = partial_derivative__different_element___d.detach().clone()
                    buffer___d_d[iota_of_dim, iota_of_dim] = partial_derivative__same_element___d
                    buffer___d_d.eq(partial_derivative__same_element___d).to(torch.int32).sum() == dim
                    buffer___d_d.eq(partial_derivative__different_element___d).to(torch.int32).sum() == dim*dim-dim
                    
                    assert buffer___d_d.eq(buffer___d_d.T).all()
                        #  adaption to the g_in
                        #  adaption to the g_in
                    _temp___grad___l_EXPANDl = grad___l.reshape(shape=[dim, 1]).expand([dim, dim])
                    assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                    manual_grad_for___no_detach = (buffer___d_d*_temp___grad___l_EXPANDl).sum(dim=0) 
                    del _temp___grad___l_EXPANDl
                    #del buffer___d_d
                    assert _tensor_equal(manual_grad_for___no_detach, no_detach_input___d.grad)


                    #  with_detach
                    #  with_detach
                    with_detach_input___d = input.detach().clone()
                    with_detach_input___d.requires_grad_()
                    the_rms = (with_detach_input___d*with_detach_input___d).sum().sqrt().detach()  #######     detach   #######
                    with_detach_result___d = with_detach_input___d/the_rms
                        #  a bit assert
                    _temp___result_rms = (with_detach_result___d*with_detach_result___d).sum().sqrt()
                    assert _tensor_equal(_temp___result_rms, [1.])
                    del _temp___result_rms

                    with_detach_result___d.backward(gradient=grad___l, inputs=[with_detach_input___d])

                    #  grad formula validation
                    #  grad formula validation
                    manual_grad_for___with_detach = grad___l/the_rms
                    assert _tensor_equal(manual_grad_for___with_detach, with_detach_input___d.grad)


                    #  over all test
                    #  over all test
                    assert _tensor_equal(with_detach_result___d, no_detach_result___d)

                    # prin(no_detach_input___d.grad)
                    # prin(with_detach_input___d.grad)
                    # prin(with_detach_input___d.grad - no_detach_input___d.grad)

                    pass#for _iii
                pass#for dim
            pass#/test


        '''Anyway. There are 2 possible steps which can be detached. 
        So in total, there are 4 possible versions of this into normal distribution.
        No plan for this one. 
        But even without any test, it's still possible to get some intuition.
        In the previous test, sub the mean, or div the rms, the detach does something on both.
        So, all the 4 versions probably don't share anything.
        
        conclusion:       ????????????         '''
        if "into normal distribution" and False:
            assert False, "unfinished, no plan."
            for dim in [3,2,4,6,11,31]:
                iota_of_dim = iota(dim)
                for _iii in range(66):
                    input = torch.randn(size=[dim])  
                    grad___l = torch.randn(size=[dim])



                    #  no_detach
                    #  no_detach
                    no_detach_input___d = input.detach().clone()
                    no_detach_input___d.requires_grad_()
                    the_mean = no_detach_input___d.mean()
                    after_sub_the_mean___d = no_detach_input___d - the_mean
                    the_rms = (after_sub_the_mean___d*after_sub_the_mean___d).sum().sqrt()  #######   no detach   #######
                    no_detach_result___d = no_detach_input___d/the_rms
                        #  a bit assert
                    assert _tensor_equal(no_detach_result___d.sum(), [0.])
                    _temp___result_rms = (no_detach_result___d*no_detach_result___d).sum().sqrt()
                    assert _tensor_equal(_temp___result_rms, [1.])
                    del _temp___result_rms

                    no_detach_result___d.backward(gradient=grad___l, inputs=[no_detach_input___d])

                    # #  grad formula validation
                    # #  grad formula validation
                    
                    # partial_derivative__same_element___d = (the_rms - no_detach_input___d*no_detach_input___d/the_rms)/(the_rms*the_rms)

                    # '''partial_derivative__different_element___d'''
                    # no_detach_input___d_EXPANDd = no_detach_input___d.reshape(shape=[dim, 1]).expand(size=[dim, dim])
                    # no_detach_input___EXPANDd_d = no_detach_input___d.reshape(shape=[1, dim]).expand(size=[dim, dim])
                    # partial_derivative__different_element___d = -1.*no_detach_input___d_EXPANDd*no_detach_input___EXPANDd_d/(the_rms*the_rms*the_rms)
                    # partial_derivative__different_element___d[iota_of_dim, iota_of_dim] = 0.

                    # buffer___d_d = partial_derivative__different_element___d.detach().clone()
                    # buffer___d_d[iota_of_dim, iota_of_dim] = partial_derivative__same_element___d
                    # buffer___d_d.eq(partial_derivative__same_element___d).to(torch.int32).sum() == dim
                    # buffer___d_d.eq(partial_derivative__different_element___d).to(torch.int32).sum() == dim*dim-dim
                    
                    # assert buffer___d_d.eq(buffer___d_d.T).all()
                    #     #  adaption to the g_in
                    #     #  adaption to the g_in
                    # _temp___grad___l_EXPANDl = grad___l.reshape(shape=[dim, 1]).expand([dim, dim])
                    # assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                    # manual_grad_for___no_detach = (buffer___d_d*_temp___grad___l_EXPANDl).sum(dim=0) 
                    # del _temp___grad___l_EXPANDl
                    # #del buffer___d_d
                    # assert _tensor_equal(manual_grad_for___no_detach, no_detach_input___d.grad)


                    #  with_detach
                    #  with_detach
                    with_detach_input___d = input.detach().clone()
                    with_detach_input___d.requires_grad_()
                    the_rms = (with_detach_input___d*with_detach_input___d).sum().sqrt().detach()  #######     detach   #######
                    with_detach_result___d = with_detach_input___d/the_rms
                        #  a bit assert
                    _temp___result_rms = (with_detach_result___d*with_detach_result___d).sum().sqrt()
                    assert _tensor_equal(_temp___result_rms, [1.])
                    del _temp___result_rms

                    with_detach_result___d.backward(gradient=grad___l, inputs=[with_detach_input___d])

                    #  grad formula validation
                    #  grad formula validation
                    manual_grad_for___with_detach = grad___l/the_rms
                    assert _tensor_equal(manual_grad_for___with_detach, with_detach_input___d.grad)


                    #  over all test
                    #  over all test
                    assert _tensor_equal(with_detach_result___d, no_detach_result___d)

                    # prin(no_detach_input___d.grad)
                    # prin(with_detach_input___d.grad)
                    # prin(with_detach_input___d.grad - no_detach_input___d.grad)

                    pass#for _iii
                pass#for dim
            pass#/test


        return
    ____adaptive_methods()
    pass






'''conclusion:  pytorch's softmax is safe softmax. 
Otherwise if anything >89 in the input, the entire thing breaks.'''
if "a lil bit fp32 test" and False:
    def ___fp32____softmax_test():
        assert torch.tensor([88.5]).exp().isinf() == False
        assert torch.tensor([89  ]).exp().isinf() == True
        
        aaaaaa = torch.tensor([100., 0, 0]).softmax(dim=0)
        bbbbbb = torch.tensor([100., 0, 0]).softmax(dim=0).isinf()
        assert all_False(torch.tensor([100., 0, 0]).softmax(dim=0).isinf())
        return 
    ___fp32____softmax_test()
    pass


'''conclusion:  2 adaptive steps can detach without break the forward formula.
In total, 4 different backward pass possible.
2 are mathmatically same. In total 3 different backward pass formula.
The other 2 are similar, except for the max element gets different grad.
Do we let the 2 adaptive steps also pass grad through the "adjust" branch of backward?'''

if "all the softmax" and True:
    def all_possible_softmax_versions():
        '''Notice. For simplicity, I didn't adapt some formula to cases where vec_length is 1 or 2.'''
        for vec_length in [3,4,5,10,33]:
            for _iiii in range(33):

                iota_of_length = iota(vec_length)
                input___l = torch.randn(size=[vec_length])
                grad___l  = torch.randn(size=[vec_length])




                #  pytorch forward
                #  pytorch forward
                pytorch_input___l = input___l.detach().clone()
                pytorch_input___l = pytorch_input___l.requires_grad_()
                pytorch_softmax___l = pytorch_input___l.softmax(dim=0)
                assert _tensor_equal(pytorch_softmax___l.sum(), [1.])
                pytorch_softmax___l.backward(gradient=grad___l, inputs=[pytorch_input___l])
                assert pytorch_input___l.grad is not None
                assert _tensor_equal(pytorch_input___l.grad.sum(), [0.])    #    grad.sum is 0.  not every one does this.




                #  no_detach 
                #  no_detach 
                no_detach_input___l = input___l.detach().clone()
                no_detach_input___l = no_detach_input___l.requires_grad_()
                _temp___the_max___s = no_detach_input___l.max()#  ############     no detach       !!!!!!!!!!!!!
                _temp___input_shifted___l = no_detach_input___l - _temp___the_max___s
                safe_exp_vector___l = _temp___input_shifted___l.exp()
                the_sum___s = safe_exp_vector___l.sum()#   ############     no detach       !!!!!!!!!!!!!
                no_detach_softmax___l = safe_exp_vector___l/the_sum___s
                assert _tensor_equal(no_detach_softmax___l.sum(), [1.])

                no_detach_softmax___l.backward(gradient=grad___l, inputs=[no_detach_input___l])
                assert no_detach_input___l.grad is not None
                assert _tensor_equal(no_detach_input___l.grad.sum(), [0.])    
                    #  backward formula validation
                    #  backward formula validation
                '''  Notice, although the formula of softmax can be simplified a bit, but I still do the full version.
                Reason is, if I simplify it, it's the same as the minus_detach version. 
                Btw, I'm showing off my ability of coding. '''
                _temp___max_index__in_tensor = no_detach_input___l.argmax()
                max_index = _temp___max_index__in_tensor.item()
                #del _temp___max_index__in_tensor
                assert type(max_index) == int
                assert _tensor_equal(_temp___input_shifted___l[max_index], [0.])
                assert _tensor_equal(safe_exp_vector___l[max_index], [1.])

                one_hot_of_max_index = torch.zeros(size=[vec_length], dtype=torch.bool)
                one_hot_of_max_index[max_index] = True
                reverse_one_hot = one_hot_of_max_index.logical_not()

                partial_derivative__from_max_element_to_max_element___s = safe_exp_vector___l[reverse_one_hot].sum()/(the_sum___s*the_sum___s) 
                assert partial_derivative__from_max_element_to_max_element___s.nelement() == 1

                partial_derivative__from_max_element_to_NON_max_element___l = -1.*safe_exp_vector___l/(the_sum___s*the_sum___s) 
                partial_derivative__from_max_element_to_NON_max_element___l[max_index] = 0.
                assert partial_derivative__from_max_element_to_NON_max_element___l.nelement() == vec_length

                partial_derivative__from_NON_max_element_to_max_element___l = \
                        (-1.*safe_exp_vector___l*the_sum___s + safe_exp_vector___l*(the_sum___s-1.))/(the_sum___s*the_sum___s) 
                        #or (-1.* safe_exp_vector___l)/(the_sum___s*the_sum___s) 
                partial_derivative__from_NON_max_element_to_max_element___l[max_index] = 0.
                assert partial_derivative__from_NON_max_element_to_max_element___l.nelement() == vec_length

                assert _tensor_equal(   partial_derivative__from_max_element_to_NON_max_element___l, \
                                        partial_derivative__from_NON_max_element_to_max_element___l)#  they share the same formula.

                partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l = \
                        (safe_exp_vector___l*the_sum___s - safe_exp_vector___l*safe_exp_vector___l)/(the_sum___s*the_sum___s) 
                partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l[max_index] = 0.
                assert partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l.nelement() == vec_length

                '''partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l'''
                _temp___safe_exp_vector___l_EXPANDl = safe_exp_vector___l.reshape(shape=[vec_length, 1]).expand(size=[vec_length, vec_length])
                _temp___safe_exp_vector___EXPANDl_l = safe_exp_vector___l.reshape(shape=[1, vec_length]).expand(size=[vec_length, vec_length])
                partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l = \
                        -1.* _temp___safe_exp_vector___l_EXPANDl*_temp___safe_exp_vector___EXPANDl_l/(the_sum___s*the_sum___s) 
                        #  remove un wanted elements
                partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[iota_of_length, iota_of_length] = 0.#diagonal
                partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[max_index, :] = 0.
                partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[:, max_index] = 0.
                        #  /remove un wanted elements
                assert _tensor_shape_check(partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l, vec_length,vec_length)
                assert (partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l - \
                        partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l.T).abs().lt(0.0001).all()# symmetric. 

                buffer___l_l = partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l.detach().clone()#clone. So allows checks later.
                buffer___l_l[max_index, max_index] += partial_derivative__from_max_element_to_max_element___s
                buffer___l_l[max_index, :] += partial_derivative__from_max_element_to_NON_max_element___l
                buffer___l_l[:, max_index] += partial_derivative__from_NON_max_element_to_max_element___l
                buffer___l_l[iota_of_length, iota_of_length] += partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l

                assert (buffer___l_l-buffer___l_l.T).abs().lt(0.0001).all()# buffer is symmetric. 
                                # The only reason is max_to_NON_max and NON_max_to_max share the same formula.

                #  check a lil bit
                #  check a lil bit
                #  check a lil bit
                    #  NON_to_NON_diff is 2 triangles
                equal_bool_tensor__NON_to_NON_diff = buffer___l_l.eq(partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l)
                assert all_False(equal_bool_tensor__NON_to_NON_diff[iota_of_length, iota_of_length])
                assert all_False(equal_bool_tensor__NON_to_NON_diff[max_index, :])
                assert all_False(equal_bool_tensor__NON_to_NON_diff[:, max_index])
                #prin_2d_bool_tensor(equal_bool_tensor__NON_to_NON_diff, false_as=".")
                equal_amount_of__NON_to_NON_diff = equal_bool_tensor__NON_to_NON_diff.to(torch.int32).sum()
                assert equal_amount_of__NON_to_NON_diff.nelement() == 1
                assert equal_amount_of__NON_to_NON_diff == vec_length*vec_length - 3*vec_length + 2


                    #  NON_to_NON_same is diagonal without 1 element
                _temp_buffer = torch.empty_like(buffer___l_l)
                _temp_buffer.fill_(torch.nan)
                _temp_buffer[iota_of_length, iota_of_length] = partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l
                equal_bool_tensor__NON_to_NON_same = buffer___l_l.eq(_temp_buffer)
                assert equal_bool_tensor__NON_to_NON_same.eq(equal_bool_tensor__NON_to_NON_same.T).all()#symmetric
                assert all_False(equal_bool_tensor__NON_to_NON_same[0:max_index+1, max_index:   ])
                assert all_False(equal_bool_tensor__NON_to_NON_same[max_index:,    0:max_index+1])
                assert all_False(equal_bool_tensor__NON_to_NON_same[max_index, max_index])#duplicated.
                assert equal_bool_tensor__NON_to_NON_same[iota_of_length, iota_of_length][reverse_one_hot].all()
                #prin_2d_bool_tensor(equal_bool_tensor__NON_to_NON_same, false_as=",")
                equal_amount_of__NON_to_NON_same = equal_bool_tensor__NON_to_NON_same.to(torch.int32).sum()
                assert equal_amount_of__NON_to_NON_same.nelement() == 1
                assert equal_amount_of__NON_to_NON_same == vec_length - 1


                    #  max_to_NON is row without 1 element
                _temp_buffer = torch.empty_like(buffer___l_l)
                _temp_buffer.fill_(torch.nan)
                _temp_buffer[max_index, :] = partial_derivative__from_max_element_to_NON_max_element___l
                equal_bool_tensor__max_to_NON = buffer___l_l.eq(_temp_buffer)
                #prin_2d_bool_tensor(equal_bool_tensor__max_to_NON, false_as=" ")
                assert all_False(equal_bool_tensor__max_to_NON[:max_index   , :])
                assert all_False(equal_bool_tensor__max_to_NON[ max_index+1:, :])
                assert all_False(equal_bool_tensor__max_to_NON[max_index, max_index])
                assert _temp_buffer[max_index, :][reverse_one_hot].all() 
                equal_amount_of__max_to_NON = equal_bool_tensor__max_to_NON.to(torch.int32).sum()
                assert equal_amount_of__max_to_NON.nelement() == 1
                assert equal_amount_of__max_to_NON == vec_length - 1


                    #  NON_to_max is colomn without 1 element
                _temp_buffer = torch.empty_like(buffer___l_l)
                _temp_buffer.fill_(torch.nan)
                _temp_buffer[:, max_index] = partial_derivative__from_NON_max_element_to_max_element___l
                equal_bool_tensor__NON_to_max = buffer___l_l.eq(_temp_buffer)
                #prin_2d_bool_tensor(equal_bool_tensor__NON_to_max, false_as=".")
                assert all_False(equal_bool_tensor__NON_to_max[:, :max_index   ])
                assert all_False(equal_bool_tensor__NON_to_max[:,  max_index+1:])
                assert all_False(equal_bool_tensor__NON_to_max[max_index, max_index])
                assert _temp_buffer[max_index, :][reverse_one_hot].all() 
                equal_amount_of__NON_to_max = equal_bool_tensor__NON_to_max.to(torch.int32).sum()
                assert equal_amount_of__NON_to_max.nelement() == 1
                assert equal_amount_of__NON_to_max == vec_length - 1
                    #  max_to_max is only 1 element
                assert buffer___l_l[max_index, max_index] == partial_derivative__from_max_element_to_max_element___s
                assert buffer___l_l[max_index, max_index].nelement() == 1#duplicated, just in case.
                assert partial_derivative__from_max_element_to_max_element___s.nelement() == 1#duplicated, just in case.
                #  adaption to the g_in
                #  adaption to the g_in
                _temp___grad___l_EXPANDl = grad___l.reshape(shape=[vec_length, 1]).expand([vec_length, vec_length])
                assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                manual_grad_for___no_detach = (buffer___l_l*_temp___grad___l_EXPANDl).sum(dim=0) 
                del _temp___grad___l_EXPANDl
                del buffer___l_l
                assert _tensor_equal(manual_grad_for___no_detach, no_detach_input___l.grad)




                #  minus_detach 
                #  minus_detach 
                minus_detach_input___l = input___l.detach().clone()
                minus_detach_input___l = minus_detach_input___l.requires_grad_()
                _temp___the_max___s = minus_detach_input___l.max().detach() ############      detach       !!!!!!!!!!!!!
                _temp___input_shifted___l = minus_detach_input___l - _temp___the_max___s
                _temp___the_exp_vector___l = _temp___input_shifted___l.exp()
                the_sum___s = _temp___the_exp_vector___l.sum()#                     ############    no detach       !!!!!!!!!!!!!
                minus_detach_softmax___l = _temp___the_exp_vector___l/the_sum___s
                assert _tensor_equal(minus_detach_softmax___l.sum(), [1.])

                minus_detach_softmax___l.backward(gradient=grad___l, inputs=[minus_detach_input___l])
                assert minus_detach_input___l.grad is not None
                assert _tensor_equal(minus_detach_input___l.grad.sum(), [0.])    
                    #  backward formula validation
                    #  backward formula validation
                buffer__to_manually_calc_grad = torch.zeros(size=[vec_length, vec_length], dtype=torch.float32)
                buffer__to_manually_calc_grad[iota_of_length, iota_of_length] += _temp___the_exp_vector___l[iota_of_length]*the_sum___s/(the_sum___s*the_sum___s)#real formula part 1

                _temp___the_exp_vector___l_EXPANDl = _temp___the_exp_vector___l.reshape(shape=[vec_length, 1]).expand(size=[vec_length, vec_length])
                _temp___the_exp_vector___EXPANDl_l = _temp___the_exp_vector___l.reshape(shape=[1, vec_length]).expand(size=[vec_length, vec_length])
                buffer__to_manually_calc_grad += -_temp___the_exp_vector___l_EXPANDl*_temp___the_exp_vector___EXPANDl_l/(the_sum___s*the_sum___s)#real formula part 2
                assert _tensor_equal(buffer__to_manually_calc_grad, buffer__to_manually_calc_grad.T)
                    #  adaption to the g_in
                _temp___grad___l_EXPANDl = grad___l.reshape(shape=[vec_length, 1]).expand([vec_length, vec_length])
                assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                manual_grad_for___minus_detach = (buffer__to_manually_calc_grad*_temp___grad___l_EXPANDl).sum(dim=0) 
                del _temp___grad___l_EXPANDl
                del buffer__to_manually_calc_grad
                assert _tensor_equal(manual_grad_for___minus_detach, minus_detach_input___l.grad)




                #  denominator_detach 
                #  denominator_detach 
                denominator_detach_input___l = input___l.detach().clone()
                denominator_detach_input___l = denominator_detach_input___l.requires_grad_()
                _temp___the_max___s = denominator_detach_input___l.max()#   ############    no detach       !!!!!!!!!!!!!
                _temp___input_shifted___l = denominator_detach_input___l - _temp___the_max___s
                the_exp_vector___l = _temp___input_shifted___l.exp()
                the_sum___s = the_exp_vector___l.sum().detach()     ############      detach       !!!!!!!!!!!!!
                denominator_detach_softmax___l = the_exp_vector___l/the_sum___s
                assert _tensor_equal(denominator_detach_softmax___l.sum(), [1.])

                denominator_detach_softmax___l.backward(gradient=grad___l, inputs=[denominator_detach_input___l])
                assert denominator_detach_input___l.grad is not None
                assert _tensor_equal(denominator_detach_input___l.grad.sum(), [0.])   

                    #  backward formula validation
                    #  backward formula validation
                        #  duplicated
                _temp___max_index__in_tensor = no_detach_input___l.argmax()
                max_index = _temp___max_index__in_tensor.item()
                #del _temp___max_index__in_tensor
                assert type(max_index) == int
                assert _tensor_equal(_temp___input_shifted___l[max_index], [0.])
                assert _tensor_equal(safe_exp_vector___l[max_index], [1.])

                one_hot_of_max_index = torch.zeros(size=[vec_length], dtype=torch.bool)
                one_hot_of_max_index[max_index] = True
                reverse_one_hot = one_hot_of_max_index.logical_not()
                        #  /duplicated

                partial_derivative__from_max_element_to_max_element___s = torch.tensor(0.)
                assert partial_derivative__from_max_element_to_max_element___s.nelement() == 1

                partial_derivative__from_max_element_to_NON_max_element___l = torch.zeros(size=[vec_length])
                partial_derivative__from_max_element_to_NON_max_element___l[max_index] = 0. #  this actually works as a mask
                assert partial_derivative__from_max_element_to_NON_max_element___l.nelement() == vec_length

                partial_derivative__from_NON_max_element_to_max_element___l = -1.*the_exp_vector___l/the_sum___s 
                partial_derivative__from_NON_max_element_to_max_element___l[max_index] = 0. #  this actually works as a mask
                assert partial_derivative__from_NON_max_element_to_max_element___l.nelement() == vec_length

                partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l = the_exp_vector___l/the_sum___s 
                partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l[max_index] = 0. #  this actually works as a mask
                assert partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l.nelement() == vec_length

                '''partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l'''
                #_temp___safe_exp_vector___l_EXPANDl = safe_exp_vector___l.reshape(shape=[vec_length, 1]).expand(size=[vec_length, vec_length])
                #_temp___safe_exp_vector___EXPANDl_l = safe_exp_vector___l.reshape(shape=[1, vec_length]).expand(size=[vec_length, vec_length])
                partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l = torch.zeros(size=[vec_length, vec_length])
                        #  remove un wanted elements
                partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[iota_of_length, iota_of_length] = 0.#diagonal
                partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[max_index, :] = 0.
                partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[:, max_index] = 0.
                        #  /remove un wanted elements
                assert _tensor_shape_check(partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l, vec_length, vec_length)

                assert (partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l - \
                        partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l.T).abs().lt(0.0001).all()# symmetric.

                buffer___l_l = partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l.detach().clone()# all 0. clone. So allows checks later.
                buffer___l_l[max_index, max_index] += partial_derivative__from_max_element_to_max_element___s# all 0.
                buffer___l_l[max_index, :] += partial_derivative__from_max_element_to_NON_max_element___l# all 0.
                buffer___l_l[:, max_index] += partial_derivative__from_NON_max_element_to_max_element___l# something
                buffer___l_l[iota_of_length, iota_of_length] += partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l# something

                #  check a lil bit
                #  check a lil bit
                simpler_buffer___l_l = torch.zeros(size=[vec_length, vec_length])#  no +=, only =. this is simpler,
                simpler_buffer___l_l[:, max_index] = partial_derivative__from_NON_max_element_to_max_element___l# something
                simpler_buffer___l_l[iota_of_length, iota_of_length] = partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l# something
                assert simpler_buffer___l_l.eq(buffer___l_l).all()
                del simpler_buffer___l_l

                simpler_buffer___l_l = torch.empty(size=[vec_length, vec_length], dtype=torch.float32)#  init with nan
                simpler_buffer___l_l.fill_(torch.nan)
                simpler_buffer___l_l[:, max_index] = partial_derivative__from_NON_max_element_to_max_element___l# something
                simpler_buffer___l_l[iota_of_length, iota_of_length] = partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l# something
                assert simpler_buffer___l_l[max_index, max_index] == 0.
                assert simpler_buffer___l_l[:, max_index].eq(partial_derivative__from_NON_max_element_to_max_element___l).all()
                assert simpler_buffer___l_l[iota_of_length, iota_of_length].eq(partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l).all()
                old_amount_of_nan = simpler_buffer___l_l.isnan().to(torch.int32).sum().item()
                amount_of_nan = int(old_amount_of_nan)
                assert amount_of_nan == old_amount_of_nan
                assert amount_of_nan == vec_length*vec_length - vec_length * 2  + 1
                del simpler_buffer___l_l
                    #  adaption to the g_in
                    #  adaption to the g_in
                _temp___grad___l_EXPANDl = grad___l.reshape(shape=[vec_length, 1]).expand([vec_length, vec_length])
                assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                manual_grad_for___denominator_detach = (buffer___l_l*_temp___grad___l_EXPANDl).sum(dim=0) 
                del _temp___grad___l_EXPANDl
                del buffer___l_l
                assert _tensor_equal(manual_grad_for___denominator_detach, denominator_detach_input___l.grad)




                #  double_detach 
                #  double_detach 
                double_detach_input___l = input___l.detach().clone()
                double_detach_input___l = double_detach_input___l.requires_grad_()
                the_max___s = double_detach_input___l.max().detach()  ############      detach       !!!!!!!!!!!!!
                _temp___input_shifted___l = double_detach_input___l - the_max___s
                the_exp_vector___l = _temp___input_shifted___l.exp()
                the_sum___s = the_exp_vector___l.sum().detach()       ############      detach       !!!!!!!!!!!!!
                double_detach_softmax___l = the_exp_vector___l/the_sum___s
                assert _tensor_equal(double_detach_softmax___l.sum(), [1.])

                double_detach_softmax___l.backward(gradient=grad___l, inputs=[double_detach_input___l])
                assert double_detach_input___l.grad is not None
                assert _tensor_equal(double_detach_input___l.grad.sum(), [0.]) == False  #  xxxxxx
                    #  backward formula validation
                    #  backward formula validation
                partial_derivative___same_element___l = the_exp_vector___l/the_sum___s
                partial_derivative___different_element___l_l = torch.zeros(size=[vec_length, vec_length])# all 0s.

                buffer___l_l = partial_derivative___different_element___l_l.detach().clone()# all 0. clone. So allows checks later.
                buffer___l_l[iota_of_length, iota_of_length] += partial_derivative___same_element___l# diagonal
                simpler_buffer___l_l = torch.zeros(size=[vec_length, vec_length])
                simpler_buffer___l_l[iota_of_length, iota_of_length] = partial_derivative___same_element___l#no +=, only =
                assert simpler_buffer___l_l.eq(buffer___l_l).all()
                #  no check here, too simple.

                    #  adaption to the g_in
                    #  adaption to the g_in
                _temp___grad___l_EXPANDl = grad___l.reshape(shape=[vec_length, 1]).expand([vec_length, vec_length])
                assert _temp___grad___l_EXPANDl[0,0] == _temp___grad___l_EXPANDl[0,1]
                manual_grad_for___double_detach = (buffer___l_l*_temp___grad___l_EXPANDl).sum(dim=0) 
                del _temp___grad___l_EXPANDl
                del buffer___l_l
                assert _tensor_equal(manual_grad_for___double_detach, double_detach_input___l.grad)




                #  ^^^^^   formula validation for all the versions
                #  vvvvv   relation through all of them


                #  forward pass are exactly the same.
                assert _tensor_equal(pytorch_softmax___l, no_detach_softmax___l)
                assert _tensor_equal(pytorch_softmax___l, minus_detach_softmax___l)
                assert _tensor_equal(pytorch_softmax___l, denominator_detach_softmax___l)
                assert _tensor_equal(pytorch_softmax___l, double_detach_softmax___l)

                assert _tensor_equal(no_detach_softmax___l, pytorch_softmax___l)
                assert _tensor_equal(no_detach_softmax___l, minus_detach_softmax___l)
                assert _tensor_equal(no_detach_softmax___l, denominator_detach_softmax___l)
                assert _tensor_equal(no_detach_softmax___l, double_detach_softmax___l)

                assert _tensor_equal(minus_detach_softmax___l, pytorch_softmax___l)
                assert _tensor_equal(minus_detach_softmax___l, no_detach_softmax___l)
                assert _tensor_equal(minus_detach_softmax___l, denominator_detach_softmax___l)
                assert _tensor_equal(minus_detach_softmax___l, double_detach_softmax___l)

                assert _tensor_equal(denominator_detach_softmax___l, pytorch_softmax___l)
                assert _tensor_equal(denominator_detach_softmax___l, no_detach_softmax___l)
                assert _tensor_equal(denominator_detach_softmax___l, minus_detach_softmax___l)
                assert _tensor_equal(denominator_detach_softmax___l, double_detach_softmax___l)

                assert _tensor_equal(double_detach_softmax___l, pytorch_softmax___l)
                assert _tensor_equal(double_detach_softmax___l, no_detach_softmax___l)
                assert _tensor_equal(double_detach_softmax___l, minus_detach_softmax___l)
                assert _tensor_equal(double_detach_softmax___l, denominator_detach_softmax___l)
                #  /forward pass are exactly the same.
                #  the forward result sum to 1.   ofc, bc this is softmax.   this part is duclicated.
                assert _tensor_equal(pytorch_softmax___l.sum(), [1.])
                assert _tensor_equal(no_detach_softmax___l.sum(), [1.])
                assert _tensor_equal(minus_detach_softmax___l.sum(), [1.])
                assert _tensor_equal(denominator_detach_softmax___l.sum(), [1.])
                assert _tensor_equal(double_detach_softmax___l.sum(), [1.])
                #  /the forward result sum to 1.   ofc, bc this is softmax.   this part is duclicated.



                '''conclusion:
                pytorch_ver softmax is either no_detach, or minus_detach, and they yield the exactly same result in both pass.
                But, denominator_detach and double_detach don't yield the same result. 
                In total, we have 3 mathmatically different versions of softmax.  
                Addition, although the first 2 styles looks very different in code, but their formula are mathmatically same.'''
                assert _tensor_equal(pytorch_input___l.grad, no_detach_input___l.grad)
                assert _tensor_equal(pytorch_input___l.grad,                           minus_detach_input___l.grad)
                assert _tensor_equal(                        no_detach_input___l.grad, minus_detach_input___l.grad)
                assert _tensor_equal(pytorch_input___l.grad, denominator_detach_input___l.grad)     == False
                assert _tensor_equal(pytorch_input___l.grad,                                    double_detach_input___l.grad) == False
                assert _tensor_equal(                        denominator_detach_input___l.grad, double_detach_input___l.grad) == False


                assert _tensor_equal(pytorch_input___l           .grad.sum(), [0.])   
                assert _tensor_equal(no_detach_input___l         .grad.sum(), [0.]) 
                assert _tensor_equal(minus_detach_input___l      .grad.sum(), [0.])
                assert _tensor_equal(denominator_detach_input___l.grad.sum(), [0.])
                assert _tensor_equal(double_detach_input___l     .grad.sum(), [0.]) == False  #  false  

                #要追加一下3个的公式。已经有一个了。

                # prin(pytorch_input___l           .grad)
                # prin(no_detach_input___l         .grad)
                # prin(minus_detach_input___l      .grad)
                # prin(denominator_detach_input___l.grad)
                # prin(double_detach_input___l     .grad)
                # prin(max_index)

                pass# for _iiii
            pass# for vec_length
        return
    all_possible_softmax_versions()
    pass










