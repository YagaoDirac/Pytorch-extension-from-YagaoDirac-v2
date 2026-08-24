from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, _either_1_or_neg1, _tensor_shape_check, \
        iota, str_the_list
from pytorch_yagaodirac_v2.Random import rand_sign

import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######












'''图方便而已'''
def DNN___to_posneg1(input:torch.Tensor, gt_0__true___ge_0__false = True, dtype = torch.float32)->torch.Tensor:
    if dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
        assert False, "Bad param: dtype must have sign. Use any float(fp16 32 64, bf16) or int(i8 16 32 64) instead."  

    if gt_0__true___ge_0__false:
        _temp_1 = input.gt(0.)
        pass
    else:
        _temp_1 = input.ge(0.)
        pass
    _temp_2 = _temp_1.to(dtype)

    result = _temp_2*2 -1
    #assert _either_1_or_neg1(result)  debug code.
    return result
if "test" and False:
    def ____test____DNN___to_posneg1():
        for _ in range(33):
            a = torch.randn(size=[10000])
            b = DNN___to_posneg1(a)
            assert _either_1_or_neg1(b)
            pass

        for _ in range(33):
            a = torch.rand(size=[10000])
            b = DNN___to_posneg1(a, gt_0__true___ge_0__false=False)
            assert _tensor_equal(b, torch.ones_like(b))
            pass

        for _ in range(33):
            a = torch.rand(size=[10000]) * -1
            b = DNN___to_posneg1(a, gt_0__true___ge_0__false=True)
            assert _tensor_equal(b, torch.ones_like(b)*-1.)
            pass

        if "dtype adaption" and True:
            a = torch.rand(size=[3])
            b = DNN___to_posneg1(a, dtype=torch.int32)
            assert b.dtype == torch.int32

            a = torch.rand(size=[3])
            b = DNN___to_posneg1(a, dtype=torch.float32)
            assert b.dtype == torch.float32

        return 
    ____test____DNN___to_posneg1()
    pass


'''图方便而已'''
def _test___DNN_forward___full_safety(input___b_i:torch.Tensor, raw_weight___o_i:torch.Tensor, 
                input_is_already_posneg1 = False, safety_check = True)->tuple[torch.Tensor, torch.Tensor]:
    '''return output_posneg1___b_o, index_of_max_of_raw_weight___o
    '''
    #<  safety
    if input_is_already_posneg1:
        input_posneg1___b_i = input___b_i
        pass
    else:
        input_posneg1___b_i = DNN___to_posneg1(input___b_i)
        pass
    #<  real payload
    index_of_max_of_raw_weight___o = raw_weight___o_i.max(dim=1).indices
    output_posneg1___b_o = input_posneg1___b_i[:, index_of_max_of_raw_weight___o]

    if safety_check:
        assert _either_1_or_neg1(input_posneg1___b_i)
        assert _either_1_or_neg1(output_posneg1___b_o)
        pass
    return output_posneg1___b_o, index_of_max_of_raw_weight___o
if "test" and False:
    def ____test_____test___DNN_forward___full_safety():
        if "emmmm.  This should be enough.":
            batch = 5
            in_dim = 7
            out_dim = 11
            for input__mul_me in [1., -1.]:
                for _ in range(6):
                    input_posneg1___b_i = torch.ones(size=[batch, in_dim])*input__mul_me
                    raw_weight___o_i = torch.rand(size=[out_dim, in_dim])*-1.
                    output_posneg1___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i, 
                                                                                raw_weight___o_i=raw_weight___o_i)
                    assert output_posneg1___b_o.eq(input__mul_me).all()
                    assert _tensor_shape_check(output_posneg1___b_o, batch, out_dim)
                    pass#for _
                pass#for input__mul_me
            pass#/ test

        return 
    ____test_____test___DNN_forward___full_safety()
    pass


'''图方便而已'''
def _test___binary_accuracy___full_safety(target___b_o:torch.Tensor, output_posneg1___b_o:torch.Tensor, 
                    mean_per:Literal["per_batch", "per_output", "for_all"], 
                    target_is_already_posneg1 = False, safety_check = True)->tuple[torch.Tensor, str]:
    '''return accuracy___o, recommended_result_value_name
    

    >>> accuracy___?, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, output_posneg1___b_o=output_posneg1___b_o, mean_per= ??? )
    >>> assert return_value_name == "accuracy___?"
    >>> assert _tensor_shape_check(accuracy___?, target___b_o.shape[?])
    '''
    if target_is_already_posneg1:
        target_posneg1___b_o = target___b_o
        pass
    else:
        target_posneg1___b_o = DNN___to_posneg1(target___b_o)
        
        # old code    target_posneg1___b_o = target___b_o.gt(0.)
        # target_posneg1___b_o = target_posneg1___b_o.to(torch.int32)
        # target_posneg1___b_o = target_posneg1___b_o*2 -1
        pass

    element_mul_of_target_and_output___b_o = target_posneg1___b_o * output_posneg1___b_o
    element_mul_of_target_and_output___b_o = element_mul_of_target_and_output___b_o.to(torch.float32)

    match mean_per:
        case "per_batch": 
            accuracy = element_mul_of_target_and_output___b_o.mean(dim=1)
            return_value_name = "accuracy___b"
            assert accuracy.shape.__len__() == 1
            assert accuracy.shape[0] == target___b_o.shape[0]
            pass
        case "per_output": 
            accuracy = element_mul_of_target_and_output___b_o.mean(dim=0)
            return_value_name = "accuracy___o"
            assert accuracy.shape.__len__() == 1
            assert accuracy.shape[0] == target___b_o.shape[1]
            pass
        case "for_all":
            accuracy = element_mul_of_target_and_output___b_o.mean()
            return_value_name = "accuracy___s"
            assert accuracy.shape.__len__() == 0
            pass
        case _:
            assert False, "Bad param: mean_per."
        # end of   match mean_per
    accuracy = (accuracy +1.)*0.5

    if safety_check:
        assert _either_1_or_neg1(target_posneg1___b_o)
        assert _either_1_or_neg1(output_posneg1___b_o)
        target_posneg1___b_o.shape.__len__() == 2
        output_posneg1___b_o.shape.__len__() == 2
        assert accuracy.ge(0.).all()
        assert accuracy.le(1.).all()
        pass
    return accuracy, return_value_name
if "test" and False:
    def ____test_____test___binary_accuracy___full_safety():

        if "per_batch" and True:
            target___b_o = torch.tensor([[1., 1, -1]])
            output_posneg1___b_o = torch.tensor([[1., 1, -1]])
            accuracy___b, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                                            output_posneg1___b_o=output_posneg1___b_o, mean_per="per_batch")
            assert return_value_name == "accuracy___b"
            assert _tensor_shape_check(accuracy___b, target___b_o.shape[0])
            assert _tensor_equal(accuracy___b, [1.])

            
            accuracy___b, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                        output_posneg1___b_o=output_posneg1___b_o, mean_per="per_batch", target_is_already_posneg1=True)
            assert return_value_name == "accuracy___b"
            assert _tensor_shape_check(accuracy___b, target___b_o.shape[0])
            assert _tensor_equal(accuracy___b, [1.])


            target___b_o = torch.tensor([[0.1, 0.1, -0.5]])
            output_posneg1___b_o = torch.tensor([[1., 1, -1]])
            accuracy___b, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                                            output_posneg1___b_o=output_posneg1___b_o, mean_per="per_batch")
            assert return_value_name == "accuracy___b"
            assert _tensor_shape_check(accuracy___b, target___b_o.shape[0])
            assert _tensor_equal(accuracy___b, [1.])


            target___b_o = torch.tensor([[0.1, 0.1, 0.5]])
            output_posneg1___b_o = torch.tensor([[1., 1, -1]])
            accuracy___b, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                                            output_posneg1___b_o=output_posneg1___b_o, mean_per="per_batch")
            assert return_value_name == "accuracy___b"
            assert _tensor_shape_check(accuracy___b, target___b_o.shape[0])
            assert _tensor_equal(accuracy___b, [0.6666666])


            target___b_o = torch.tensor([           [0.1, 0.1, -0.5],
                                                    [0.1, 0.1, 0.5]])
            output_posneg1___b_o = torch.tensor([   [1.,  1,   -1],
                                                    [1.,  1,   -1]])
            accuracy___b, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                                            output_posneg1___b_o=output_posneg1___b_o, mean_per="per_batch")
            assert return_value_name == "accuracy___b"
            assert _tensor_shape_check(accuracy___b, target___b_o.shape[0])
            assert _tensor_equal(accuracy___b, [1, 0.6666666])
            del accuracy___b
            pass#/ test

        if "per_output" and True:
            target___b_o = torch.tensor([[1.], [1], [-1]])
            output_posneg1___b_o = torch.tensor([[1.], [1], [-1]])
            accuracy___o, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                                            output_posneg1___b_o=output_posneg1___b_o, mean_per="per_output")
            assert return_value_name == "accuracy___o"
            assert _tensor_shape_check(accuracy___o, target___b_o.shape[1])
            assert _tensor_equal(accuracy___o, [1.])


            target___b_o = torch.tensor([[1.], [1], [-1]])
            output_posneg1___b_o = torch.tensor([[1.], [1], [-1]])
            accuracy___o, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                        output_posneg1___b_o=output_posneg1___b_o, mean_per="per_output", target_is_already_posneg1=True)
            assert return_value_name == "accuracy___o"
            assert _tensor_shape_check(accuracy___o, target___b_o.shape[1])
            assert _tensor_equal(accuracy___o, [1.])


            target___b_o = torch.tensor([[0.1], [0.1], [-0.5]])
            output_posneg1___b_o = torch.tensor([[1.], [1], [-1]])
            accuracy___o, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                        output_posneg1___b_o=output_posneg1___b_o, mean_per="per_output")
            assert return_value_name == "accuracy___o"
            assert _tensor_shape_check(accuracy___o, target___b_o.shape[1])
            assert _tensor_equal(accuracy___o, [1.])


            target___b_o = torch.tensor([           [0.1, 0.1, -0.5],
                                                    [0.1, 0.1, 0.5]])
            output_posneg1___b_o = torch.tensor([   [1.,  1,   -1],
                                                    [1.,  1,   -1]])
            accuracy___o, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                                            output_posneg1___b_o=output_posneg1___b_o, mean_per="per_output")
            assert return_value_name == "accuracy___o"
            assert _tensor_shape_check(accuracy___o, target___b_o.shape[1])
            assert _tensor_equal(accuracy___o, [1, 1, 0.5])
            del accuracy___o
            pass#/ test


        if "for_all" and True:
            target___b_o = torch.tensor([           [0.1, 0.1, -0.5],
                                                    [0.1, 0.1, 0.5]])
            output_posneg1___b_o = torch.tensor([   [1.,  1,   -1],
                                                    [1.,  1,   -1]])
            accuracy___s, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                                            output_posneg1___b_o=output_posneg1___b_o, mean_per="for_all")
            assert return_value_name == "accuracy___s"
            assert _tensor_shape_check(accuracy___s, 1)
            assert _tensor_equal(accuracy___s, [0.83333333])
            pass#/ test

        return 
    ____test_____test___binary_accuracy___full_safety()
    pass


'''图方便而已'''
def 好像没做对_test___optimizer_algo___full_safety(ori__raw_weight___o_i:torch.Tensor, grad_like_for___raw_weight___o_i:torch.Tensor, 
            learning_rate: torch.Tensor|float, safety_check = True)->torch.Tensor:
    '''return new__raw_weight___o_i
    
    The formula is like, shift all grad to negative. Scale it to around -1 to 0, 
    but the formula only makes the avg to -0.5. So some extreme elements may still be < -1.
    Then, new raw_weight is tanh(raw_weight + grad_like * lr).

    It's +, not -. Different from error propagation.

    The grad_like name is from the convention of deep learning. 
    But in this case, it's actually the ideal raw_weight in this epoch. 
    '''

    protected__grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___o_i - grad_like_for___raw_weight___o_i.max()
    if safety_check:
        assert protected__grad_like_for___raw_weight___o_i.le(0).all()
        pass
    protected__grad_like_for___raw_weight___o_i = protected__grad_like_for___raw_weight___o_i.to(torch.float32)
    _temp__mean_of__protected__grad_like_for___raw_weight___s = protected__grad_like_for___raw_weight___o_i.mean().abs()
    protected__grad_like_for___raw_weight___o_i /= _temp__mean_of__protected__grad_like_for___raw_weight___s
    protected__grad_like_for___raw_weight___o_i *= 0.5
    if safety_check:
        assert protected__grad_like_for___raw_weight___o_i.le(0.).all()
        assert _tensor_equal(protected__grad_like_for___raw_weight___o_i.mean(), [-0.5])
        pass
    new__raw_weight___o_i = torch.tanh(ori__raw_weight___o_i + protected__grad_like_for___raw_weight___o_i* learning_rate) #没乘任何系数  可能要改？？？？？？？？
    if safety_check:
        assert ori__raw_weight___o_i.le(0.).all()
        assert new__raw_weight___o_i.le(0.).all()
        pass
    return new__raw_weight___o_i
if "test" and False:
    def ____test_____test___optimizer_algo___full_safety():
        if "does it work???" and True: 
            import random

            learning_rate = 1.
    
            out_dim = 7
            in_dim = 11
            ori__raw_weight___o_i = torch.rand(size=[out_dim, in_dim])*-1.
            grad_like_for___raw_weight___o_i = torch.randn(size=[out_dim, in_dim])* (random.random()+0.1)*3. + random.random()*5.

            new__raw_weight___o_i = _test___optimizer_algo___full_safety(ori__raw_weight___o_i = ori__raw_weight___o_i,
                    grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___o_i, learning_rate = learning_rate)
            pass

        return
    ____test_____test___optimizer_algo___full_safety()
    pass


'''data gen'''
'''data gen'''
'''data gen'''
def partly_reasonable_label_from_input(input___b_i, out_dim:int, random_ratio:float|torch.Tensor, 
                            input_is_already_posneg1 = False, safety_check = True)->torch.Tensor:
    '''return target___b_o
    
    When random_ratio is 1., the output is purely random. In this case, the accuracy should be 0.5.
    If the random_ratio is 0., the output is purely reasonable. For a trained model, the accuracy can go up to 1..
    
    Anyway, the range for accuracy should always be [0.5, 1]. But it can be a bit less than 0.5.'''
    if random_ratio == 1.:
        target___b_o = rand_sign(size=[input___b_i.shape[0], out_dim], dtype=torch.float32)
        if safety_check:
            assert _either_1_or_neg1(target___b_o)
            pass
        return target___b_o

    #<  safety    
    assert random_ratio>=0.
    assert random_ratio<1.

    if input_is_already_posneg1:
        input_posneg1___b_i = input___b_i
        pass
    else:
        input_posneg1___b_i = DNN___to_posneg1(input___b_i)
        pass
    #<  real payload
    _temp___index_of_max_of_raw_weight___o = torch.randint(low=0, high=input___b_i.shape[1], size=[out_dim])
    target___b_o = input_posneg1___b_i[:, _temp___index_of_max_of_raw_weight___o]
    del _temp___index_of_max_of_raw_weight___o

    if random_ratio == 0.:
        if safety_check:
            assert _either_1_or_neg1(target___b_o)
            pass
        return target___b_o
    
    flag_to_random = torch.rand_like(target___b_o)
    flag_to_random = flag_to_random.lt(random_ratio)

    target___b_o =  flag_to_random              * rand_sign(size=[input___b_i.shape[0], out_dim], \
                                                            dtype=torch.float32) + \
                    flag_to_random.logical_not()* target___b_o
    if safety_check:
        assert _either_1_or_neg1(target___b_o)
        pass
    return target___b_o
'''behavior test is in the test for _algo_test__backward_function'''
if "basic test" and False:
    def ____test____partly_reasonable_label_from_input():
        if "no random" and True:
            batch = 5
            out_dim = 7
            in_dim = 11
            input___b_i = torch.ones(size=(batch, in_dim))

            target___b_o = partly_reasonable_label_from_input(input___b_i = input___b_i, out_dim = out_dim, 
                    random_ratio = 0., input_is_already_posneg1 = True)
            assert _tensor_equal(target___b_o, torch.ones_like(target___b_o))
            pass


        if "random_ratio scan" and True:
            if "result" and False:
                # random_ratio [ 0.0,       0.2,       0.4,       0.7,       1.0]
                # the_max      [ 100000,     80592,     60870,     31182,     1336]
                # the_min      [ 100000,     79350,     59192,     29062,    -1106]
                # the_avg      [ 100000.00,  79996.37,  59992.75,  30008.44,  4.12]
                pass
            number_of_tests = 1000

            the_max = [] # dont modify this
            the_min = [] # dont modify this
            the_avg = [] # dont modify this

            random_ratio_list = [ 0., 0.2, 0.4, 0.7, 1.]
            for ii_random_ratio_list in range(random_ratio_list.__len__()):
                random_ratio = random_ratio_list[ii_random_ratio_list]


                _raw_result__of_sum_of_target = torch.empty(size=[number_of_tests])
                #_when_start = time.perf_counter()
                
                for ii__test in range(number_of_tests):
                    batch = 1000
                    out_dim = 100
                    in_dim = 300

                    input___b_i = torch.ones(size=(batch, in_dim))
            
                    target___b_o = partly_reasonable_label_from_input(input___b_i = input___b_i, out_dim = out_dim, 
                            random_ratio = random_ratio, input_is_already_posneg1 = True)
                    _sum_of_target___s = target___b_o.sum()
                    assert _tensor_shape_check(_sum_of_target___s)
                    _raw_result__of_sum_of_target[ii__test] = _sum_of_target___s
                    pass#for ii__test
                #_when_end = time.perf_counter()
                #print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                
                the_max.append(_raw_result__of_sum_of_target.max().item())
                the_min.append(_raw_result__of_sum_of_target.min().item())
                the_avg.append(_raw_result__of_sum_of_target.mean().item())
                pass#for ii_random_ratio_list
        
            print(f"random_ratio {str_the_list(random_ratio_list, 1, separator=",      ")}")
            print(f"the_max      {str_the_list(the_max, precision=0, separator=",    ")}")
            print(f"the_min      {str_the_list(the_min, precision=0, separator=",    ")}")
            print(f"the_avg      {str_the_list(the_avg, precision=2, )}")
            pass#/ test

        return
    ____test____partly_reasonable_label_from_input()
    pass



'''general GPU container'''






'''申请内存的函数单独拿出来，方便以后调整。'''
def _only_for_Index_container_to_use____calc_bigger_capacity__for_in(
        extra_len:int, 
        len_now:int, recommended_min = 16)->int:
    '''return new_in_dim'''
    total_len_needed = extra_len+len_now
    ONE_M = 1<<20
    if total_len_needed<ONE_M:
        assert recommended_min>0
        result = total_len_needed*2+recommended_min
        return result
    ONE_G = 1<<30
    if total_len_needed<ONE_G:
        return int(total_len_needed*1.25)
    return int(total_len_needed*1.1)
    #end of function
if " test" and __DEBUG_ME__() and False:
    "感觉不用很严格？"
    def ____test_____only_for_Index_container_to_use____calc_bigger_capacity__for_in():
        if "result must be greater than input combined" and True:
            extra_len = 0
            len_now   = 0

            new_len = _only_for_Index_container_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now)
            #<  assert
            assert new_len >= extra_len + len_now
            assert new_len < 50


            extra_len = 10
            len_now   = 10

            new_len = _only_for_Index_container_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now)
            #<  assert
            assert new_len >= extra_len + len_now
            assert new_len < 100


            extra_len = 100
            len_now   = 100

            new_len = _only_for_Index_container_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now)
            #<  assert
            assert new_len >= extra_len + len_now
            assert new_len < 500


            extra_len = 1000
            len_now   = 1000

            new_len = _only_for_Index_container_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now)
            #<  assert
            assert new_len >= extra_len + len_now
            assert new_len < 5000

            extra_len = 10000
            len_now   = 10000

            new_len = _only_for_Index_container_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now)
            #<  assert
            assert new_len >= extra_len + len_now
            assert new_len < 50000

            extra_len = 100000
            len_now   = 100000

            new_len = _only_for_Index_container_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now)
            #<  assert
            assert new_len >= extra_len + len_now
            assert new_len < 500000

        return
    ____test_____only_for_Index_container_to_use____calc_bigger_capacity__for_in()
    pass




class Index_container(torch.nn.Module):
    '''The only difference from DNN_input_container_2026 is, this class doesn't have batch, and dtype is always int(not uint).'''
    _data___CAPlen:torch.nn.parameter.Parameter
    _len:int
    init_to_neg1:bool

    #customized memory related function
    _calc_bigger_capacity:function
    def __init__(self, dtype:torch.dtype|None = None, device:torch.device|str|None = "cpu", 
                init_capacity = 16, init_to_neg1 = False):
        
        super().__init__()
        if dtype is None:
            dtype = torch.int32
            pass
        self._data___CAPlen = torch.nn.Parameter(torch.empty(size=[init_capacity], 
                    dtype=dtype, device=device, requires_grad=False), requires_grad=False)
        assert self._data___CAPlen.requires_grad == False
        assert self._data___CAPlen.dtype in [torch.int, torch.int32, torch.int64, torch.int16]
        self._len = 0
        self.init_to_neg1 = init_to_neg1
        if init_to_neg1:
            self._data___CAPlen.fill_(-1)
            pass

        self._calc_bigger_capacity = _only_for_Index_container_to_use____calc_bigger_capacity__for_in
        return
    def _capacity(self)->int:
        '''get'''
        return self._data___CAPlen.shape[0]
    def __len__(self)->int:
        '''get'''
        return self._len
    def squeeze(self):
        self._data___CAPlen.data = self.get_useful()
        return

    def append(self, new_element:torch.Tensor|int)->None:
        if isinstance(new_element, int):
            new_element = torch.tensor(new_element, dtype=self._data___CAPlen.dtype, device=self._data___CAPlen.device)
            pass
        if isinstance(new_element, torch.Tensor):
            new_element = new_element.reshape([1])
            pass
        self.extend(new_element)
        return
    
    def extend(self, other:torch.Tensor)->None:
        assert other.shape.__len__() == 1
        with torch.no_grad():
                
            _temp__how_many_to_add = other.shape[0]
            _len_after = self._len + _temp__how_many_to_add
            if _len_after > self._capacity():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity(extra_len = _temp__how_many_to_add, len_now = self._len)

                _temp___new_container = torch.empty(size=[_temp___new_capacity], 
                        dtype=self._data___CAPlen.dtype, device=self._data___CAPlen.device)
                if self.init_to_neg1:
                    _temp___new_container.fill_(-1)
                    pass
                _temp___new_container[0:self._len] = self.get_useful()
                self._data___CAPlen.data = _temp___new_container
                pass

            self._data___CAPlen[self._len:self._len + _temp__how_many_to_add] = other
            self._len = _len_after
            return
        pass#end of function

    def get_useful(self)->torch.Tensor:
        result = self._data___CAPlen[:self._len]
        return result
    
    # def __repr__(self):
    #     return f"{self.get_useful().__repr__()}, size:{self._size}, _only_for_output_container_to_use____DNN_container_2026"
    # def __str__(self):
    #     return f"{self.get_useful().__str__()}, size:{self._size}, _only_for_output_container_to_use____DNN_container_2026"

    pass
if "how to add element." and __DEBUG_ME__() and False:
    def ____test____index_container():
        if "extend function" and True:
            #<  the container
            the_container = Index_container(init_capacity=6, init_to_neg1=True)
            assert the_container.__len__() == 0
            assert the_container._capacity() == 6
            assert the_container.init_to_neg1 == True
            the_container.extend(torch.tensor([ 11,  22,  33]))
            assert the_container.__len__() == 3
            assert the_container._capacity() == 6
            the_container.extend(torch.tensor([ 77,  88]))
            assert the_container.__len__() == 5
            assert the_container._capacity() == 6
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 11,  22,  33,  77,  88]))
            assert _tensor_equal(the_container._data___CAPlen, torch.tensor([ 11,  22,  33,  77,  88, -1]))

            the_container.extend(torch.tensor([ 111,  222]))
            assert the_container.__len__() == 7
            assert the_container._capacity() >= 7
            assert the_container._capacity() <= 50
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 11,  22,  33,  77,  88,  111,  222]))
            assert _tensor_equal(the_container._data___CAPlen[:10], torch.tensor([ 11,  22,  33,  77,  88,  111,  222, -1, -1, -1]))#may not stable.
            pass#/ test

        if "append " and True:
            #<  the container
            the_container = Index_container(init_capacity=2, init_to_neg1=True)
            assert the_container.__len__() == 0
            assert the_container._capacity() == 2
            assert the_container.init_to_neg1 == True

            the_container.append(torch.tensor(11))
            assert the_container.__len__() == 1
            assert the_container._capacity() == 2
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 11]))
            assert _tensor_equal(the_container._data___CAPlen, torch.tensor([ 11, -1]))

            the_container.append(22)
            assert the_container.__len__() == 2
            assert the_container._capacity() == 2
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 11, 22]))
            assert _tensor_equal(the_container._data___CAPlen, torch.tensor([ 11, 22]))

            
            the_container.append(torch.tensor([33]))
            assert the_container.__len__() == 3
            #assert the_container._capacity() == 4#may unstable, it depends on the algo.
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 11, 22, 33]))
            assert _tensor_equal(the_container._data___CAPlen[:4], torch.tensor([ 11, 22, 33, -1]))
            pass#/ test

        if "squeeze function" and True:
            the_container = Index_container(init_capacity=6, init_to_neg1=True)
            assert the_container.__len__() == 0
            assert the_container._capacity() == 6
            assert the_container.init_to_neg1 == True
            the_container.squeeze()
            assert the_container.__len__() == 0
            assert the_container._capacity() == 0
            the_container.append(11)
            assert the_container.__len__() == 1
            assert the_container._capacity() >=1
            assert the_container._capacity() <=50
            the_container.squeeze()
            assert the_container.__len__() == 1
            assert the_container._capacity() == 1
            pass#/ test

        if "device adaption" and True:
            the_container = Index_container()
            assert the_container._data___CAPlen.device.type == "cpu"
            the_container.cuda()
            assert the_container._data___CAPlen.device.type == "cuda"
            the_container = Index_container(device="cuda")
            assert the_container._data___CAPlen.device.type == "cuda"
            the_container.cpu()
            assert the_container._data___CAPlen.device.type == "cpu"
            pass

        return
    ____test____index_container()
    pass





