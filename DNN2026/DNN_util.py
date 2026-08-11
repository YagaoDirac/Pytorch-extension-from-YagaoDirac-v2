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
        
            print(f"random_ratio {str_the_list(random_ratio_list, 1, segment=",      ")}")
            print(f"the_max      {str_the_list(the_max, precision=0, segment=",    ")}")
            print(f"the_min      {str_the_list(the_min, precision=0, segment=",    ")}")
            print(f"the_avg      {str_the_list(the_avg, precision=2, )}")
            pass#/ test

        return
    ____test____partly_reasonable_label_from_input()
    pass

