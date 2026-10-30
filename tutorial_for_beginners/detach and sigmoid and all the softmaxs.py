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










