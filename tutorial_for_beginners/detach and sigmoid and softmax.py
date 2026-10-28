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




















vec_length = 3
iota_of_length = iota(vec_length)
input___l = torch.randn(size=[vec_length])
grad___l  = torch.randn(size=[vec_length])

# input___l = torch.tensor([1., 0., 0.])
# grad___l  = torch.tensor([1., 0., 0.])
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
_temp___max_index__in_tensor = no_detach_input___l.argmax()
max_index = _temp___max_index__in_tensor.item()
del _temp___max_index__in_tensor
assert type(max_index) == int
assert _tensor_equal(no_detach_input___l[max_index].values(), [0.])#需要吗？

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

partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l = \
        (safe_exp_vector___l*the_sum___s - safe_exp_vector___l*safe_exp_vector___l)/(the_sum___s*the_sum___s) 
partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l[max_index] = 0.
assert partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l.nelement() == vec_length

'''partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l'''
_temp___safe_exp_vector___l_EXPANDl = safe_exp_vector___l.reshape(shape=[vec_length, 1]).expand(size=[vec_length, vec_length])
_temp___safe_exp_vector___EXPANDl_l = safe_exp_vector___l.reshape(shape=[1, vec_length]).expand(size=[vec_length, vec_length])
partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l = \
        -1.* _temp___safe_exp_vector___l_EXPANDl*_temp___safe_exp_vector___EXPANDl_l/(the_sum___s*the_sum___s) 
partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l[iota_of_length, iota_of_length] = 0.
assert _tensor_shape_check(partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l, vec_length,vec_length)

buffer___l_l = partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l.detach().clone()#clone. So allows checks later.
buffer___l_l[max_index, max_index] += partial_derivative__from_max_element_to_max_element___s
buffer___l_l[max_index, :] += partial_derivative__from_max_element_to_NON_max_element___l
buffer___l_l[:, max_index] += partial_derivative__from_NON_max_element_to_max_element___l
buffer___l_l[iota_of_length, iota_of_length] += partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l


#  check a lil bit
#  check a lil bit
#  check a lil bit
_equal_bool_tensor__NON_to_NON_diff = buffer___l_l.eq(partial_derivative__from_NON_max_element_to_NON_max_element___different_element___l_l)
#print_2d_bool_tensor(_equal_bool_tensor__NON_to_NON_diff)
_equal_amount_of__NON_to_NON_diff = _equal_bool_tensor__NON_to_NON_diff.to(torch.int32).sum()
assert _equal_amount_of__NON_to_NON_diff.nelement() == 1
assert _equal_amount_of__NON_to_NON_diff == vec_length*vec_length - 3*vec_length + 2


_temp_buffer = torch.empty_like(buffer___l_l)
_temp_buffer.fill_(torch.nan)
_temp_buffer[iota_of_length, iota_of_length] = partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l
_equal_bool_tensor__NON_to_NON_same = buffer___l_l.eq(_temp_buffer)
#print_2d_bool_tensor(_equal_bool_tensor__NON_to_NON_same)
_equal_amount_of__NON_to_NON_same = _equal_bool_tensor__NON_to_NON_same.to(torch.int32).sum()
assert _equal_amount_of__NON_to_NON_same.nelement() == 1
assert _equal_amount_of__NON_to_NON_same == vec_length - 1


_temp_buffer = torch.empty_like(buffer___l_l)
_temp_buffer.fill_(torch.nan)
_temp_buffer[max_index, :] = partial_derivative__from_max_element_to_NON_max_element___l
_equal_bool_tensor__max_to_NON = buffer___l_l.eq(_temp_buffer)
#print_2d_bool_tensor(_equal_bool_tensor__max_to_NON)
_equal_amount_of__max_to_NON = _equal_bool_tensor__max_to_NON.to(torch.int32).sum()
assert _equal_amount_of__max_to_NON.nelement() == 1
assert _equal_amount_of__max_to_NON == vec_length - 1


_temp_buffer = torch.empty_like(buffer___l_l)
_temp_buffer.fill_(torch.nan)
_temp_buffer[:, max_index] = partial_derivative__from_NON_max_element_to_max_element___l
_equal_bool_tensor__NON_to_max = buffer___l_l.eq(_temp_buffer)
#print_2d_bool_tensor(_equal_bool_tensor__NON_to_max)
_equal_amount_of__NON_to_max = _equal_bool_tensor__NON_to_max.to(torch.int32).sum()
assert _equal_amount_of__NON_to_max.nelement() == 1
assert _equal_amount_of__NON_to_max == vec_length - 1


assert _temp_buffer[max_index, max_index] == partial_derivative__from_NON_max_element_to_NON_max_element___same_element___l

1w来跑一下。














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
_temp___the_exp_vector___l = _temp___input_shifted___l.exp()
the_sum___s = _temp___the_exp_vector___l.sum().detach()      ############      detach       !!!!!!!!!!!!!
denominator_detach_softmax___l = _temp___the_exp_vector___l/the_sum___s
assert _tensor_equal(denominator_detach_softmax___l.sum(), [1.])

denominator_detach_softmax___l.backward(gradient=grad___l, inputs=[denominator_detach_input___l])
assert denominator_detach_input___l.grad is not None
assert _tensor_equal(denominator_detach_input___l.grad.sum(), [0.])   

#  double_detach 
#  double_detach 
double_detach_input___l = input___l.detach().clone()
double_detach_input___l = double_detach_input___l.requires_grad_()
_temp___the_max___s = double_detach_input___l.max().detach()  ############      detach       !!!!!!!!!!!!!
_temp___input_shifted___l = double_detach_input___l - _temp___the_max___s
_temp___the_exp_vector___l = _temp___input_shifted___l.exp()
the_sum___s = _temp___the_exp_vector___l.sum().detach()       ############      detach       !!!!!!!!!!!!!
double_detach_softmax___l = _temp___the_exp_vector___l/the_sum___s
assert _tensor_equal(double_detach_softmax___l.sum(), [1.])

double_detach_softmax___l.backward(gradient=grad___l, inputs=[double_detach_input___l])
assert double_detach_input___l.grad is not None
assert _tensor_equal(double_detach_input___l.grad.sum(), [0.]) == False  #  xxxxxx










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


# print(pytorch_input___l.grad)
# print(no_detach_input___l.grad)
# print(minus_detach_input___l.grad)
# print(denominator_detach_input___l.grad)
# print(double_detach_input___l.grad)

'''conclusion:
pytorch_ver softmax is either no_detach, or minus_detach, and they yield the exactly same result in both pass.
But, denominator_detach and double_detach don't yield the same result. 
In total, we have 3 mathmatically different versions of softmax.  '''
assert _tensor_equal(pytorch_input___l.grad, no_detach_input___l.grad)
assert _tensor_equal(pytorch_input___l.grad,                           minus_detach_input___l.grad)
assert _tensor_equal(                        no_detach_input___l.grad, minus_detach_input___l.grad)
assert _tensor_equal(pytorch_input___l.grad, denominator_detach_input___l.grad)     == False
assert _tensor_equal(pytorch_input___l.grad,                                    double_detach_input___l.grad) == False
assert _tensor_equal(                        denominator_detach_input___l.grad, double_detach_input___l.grad) == False


assert _tensor_equal(pytorch_input___l.grad.sum(), [0.])   
assert _tensor_equal(no_detach_input___l.grad.sum(), [0.]) 
assert _tensor_equal(minus_detach_input___l.grad.sum(), [0.])
assert _tensor_equal(denominator_detach_input___l.grad.sum(), [0.])
assert _tensor_equal(double_detach_input___l.grad.sum(), [0.]) == False    

#要追加一下3个的公式。已经有一个了。
fds = 432



























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
                pass
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


        if "softmax in pytorch             ????" and False:
            ''' with the pytorch ver of softmax, 
            sum of output of softmax is 1.
            sum of the grad from softmax is 0.'''
            for _iiii in range(11):
                aaa = torch.torch.randn(size=[5], requires_grad=True)
                pytorch_softmax = aaa.softmax(dim=0)
                assert _tensor_equal(pytorch_softmax.sum(), [1.])

                grad = torch.randn(size=[5])
                pytorch_softmax.backward(gradient=grad, inputs=[aaa])
                assert _tensor_equal(aaa.grad.sum(), [0.])

                pass

            '''the conclusion is very clear, pytorch ver softmax doesn't do the denominator detach.'''
            for _iiii in range(5555):
                vec_length = 4
                iota_of_length = iota(vec_length)

                #  my forward
                my_input___l = torch.randn(size=[vec_length], requires_grad=True)
                _temp___the_max___s = my_input___l.max()
                _temp___input_shifted___l = my_input___l - _temp___the_max___s
                _temp___the_exp_vector___l = _temp___input_shifted___l.exp()
                the_sum = _temp___the_exp_vector___l.sum()
                my_softmax = _temp___the_exp_vector___l/the_sum
                assert _tensor_equal(my_softmax.sum(), [1.])

                #  pytorch forward
                pytorch_input = my_input___l.detach().clone()
                pytorch_input.requires_grad_()
                pytorch_softmax = pytorch_input.softmax(dim=0)

                assert _tensor_equal(my_softmax, pytorch_softmax)

                #  backward
                grad___l = torch.randn(size=[vec_length])

                buffer = torch.empty(size=[vec_length, vec_length], dtype=torch.float32)
                buffer.fill_(0.)
                buffer[iota_of_length, iota_of_length] += _temp___the_exp_vector___l[iota_of_length]*the_sum/(the_sum*the_sum)

                _temp___the_exp_vector___l_EXPANDl = _temp___the_exp_vector___l.reshape(shape=[vec_length, 1]).expand(size=[vec_length, vec_length])
                _temp___the_exp_vector___EXPANDl_l = _temp___the_exp_vector___l.reshape(shape=[1, vec_length]).expand(size=[vec_length, vec_length])
                buffer += -_temp___the_exp_vector___l_EXPANDl*_temp___the_exp_vector___EXPANDl_l/(the_sum*the_sum)
                #print(buffer)
                assert _tensor_equal(buffer, buffer.T)
                grad___l_EXPANDl = grad___l.reshape(shape=[grad___l.shape[0], 1]).expand(size=[grad___l.shape[0], grad___l.shape[0]])
                assert grad___l_EXPANDl[0,0] == grad___l_EXPANDl[0,1]
                my_manual_grad = (buffer*grad___l_EXPANDl).sum(dim=0) 
                del grad___l_EXPANDl
                assert _tensor_equal(my_manual_grad.sum(), [0.])


                my_softmax.backward(gradient=grad___l, inputs=[my_input___l])
                assert _tensor_equal(my_manual_grad, my_input___l.grad)
                assert _tensor_equal(my_input___l.grad.sum(), [0.])


                pytorch_softmax.backward(gradient=grad___l, inputs=[pytorch_input])
                assert _tensor_equal(my_manual_grad, pytorch_input.grad)
                assert _tensor_equal(pytorch_input.grad.sum(), [0.])

                pass#for _iiii


            '''what happens if I do the denominator detach'''
            pass


        if True:

            for _iiii in range(1):
                vec_length = 3







                #  my forward
                #my_input___l = torch.randn(size=[vec_length], requires_grad=True)
                my_input___l = torch.tensor([1., 0., 0.], requires_grad=True)
                _temp___the_max___s = my_input___l.max()
                _temp___input_shifted___l = my_input___l - _temp___the_max___s
                _temp___the_exp_vector___l = _temp___input_shifted___l.exp()
                the_sum = _temp___the_exp_vector___l.sum().detach()       ############      detach       !!!!!!!!!!!!!
                my_softmax = _temp___the_exp_vector___l/the_sum
                assert _tensor_equal(my_softmax.sum(), [1.])

                #  pytorch forward
                pytorch_input = my_input___l.detach().clone()
                pytorch_input.requires_grad_()
                pytorch_softmax = pytorch_input.softmax(dim=0)
                assert _tensor_equal(my_softmax, pytorch_softmax)        

                #  backward
                grad___l = torch.randn(size=[vec_length])

                my_softmax.backward(gradient=grad___l, inputs=[my_input___l])
                assert _tensor_equal(my_input___l.grad.sum(), [0.])     #   ???????????????

                pytorch_softmax.backward(gradient=grad___l, inputs=[pytorch_input])
                assert _tensor_equal(pytorch_input.grad.sum(), [0.])

                print(my_input___l.grad)
                print(pytorch_input.grad)
                print(my_input___l.grad-pytorch_input.grad)
                
                pass












        return 
    ____test____detach_feature_test()
    pass














