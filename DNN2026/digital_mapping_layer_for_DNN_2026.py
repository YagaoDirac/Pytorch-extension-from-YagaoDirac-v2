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





# 正在   缓冲区的行为，更新当中的更新动力学？？？可能单独写一个gramo？或者单独的优化器？
# 一个整体的容器
#trace back 需要容器的支持。 从整体的class里面得到新的输入数据。
# 重新做干堆测试。






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
def _test___optimizer_algo___full_safety(ori__raw_weight___o_i:torch.Tensor, grad_like_for___raw_weight___o_i:torch.Tensor, 
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
        if "does it run???" and True: 
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



if "backward algo     shape validation.     Before I rephrase it into a function." and False:
    def ____backward_algo_test____():

        if "fixed shape test" and False:
            batch = 3
            in_dim = 5
            out_dim = 2
            SOME_HYPER_PARAM___s = 1.

            #<  all the data.
            target___b_o = torch.tensor([   [1., 1,],
                                            [1., 1,],
                                            [1., 1,],])
            assert _tensor_shape_check(target___b_o, batch, out_dim)

            input_posneg1___b_i = torch.tensor([[1., 1,1,1,1,],
                                        [1., 1,1,1,1,],
                                        [1., 1,1,1,1,],])
            assert _tensor_shape_check(input_posneg1___b_i, batch, in_dim)
            assert _either_1_or_neg1(input_posneg1___b_i)

            raw_weight___o_i = torch.tensor([   [0., -0.5, -1, -1, -1,],
                                                [0., -0.5, -1, -1, -1,],])
            assert _tensor_shape_check(raw_weight___o_i, out_dim, in_dim)
            assert raw_weight___o_i.ge(-1.).all()
            assert raw_weight___o_i.le( 0.).all()

            #recomputation
            index_of_max_of_raw_weight___o = raw_weight___o_i.max(dim=1).indices
            assert _tensor_shape_check(index_of_max_of_raw_weight___o, out_dim)
            output_posneg1___b_o = input_posneg1___b_i[:, index_of_max_of_raw_weight___o]
            # output_posneg1___b_o = torch.tensor([   [1., 1,],
            #                                         [1., 1,],
            #                                         [1., 1,],])
            assert _either_1_or_neg1(output_posneg1___b_o)
            assert _tensor_shape_check(output_posneg1___b_o, batch, out_dim)

            #<  real payload
            #<  init results to None
            grad_like_for___input___b_i     :torch.Tensor|None = None
            grad_like_for___raw_weight___o_i:torch.Tensor|None = None

            if "raw_weight___o_i.requires_grad" or "input___b_i.requires_grad":
                target___b_o_EXPANDi = target___b_o.reshape(shape=[target___b_o.shape[0], target___b_o.shape[1], 1]). \
                        expand(size=[-1, -1, input_posneg1___b_i.shape[1]])
                assert _tensor_shape_check(target___b_o_EXPANDi, batch, out_dim, in_dim)
                pass


            if "raw_weight___o_i.requires_grad":
                input_posneg1___b_oEXPAND_i = input_posneg1___b_i.reshape(shape=[input_posneg1___b_i.shape[0], 1, input_posneg1___b_i.shape[1]]). \
                        expand(size=[-1, target___b_o.shape[1], -1])
                assert _tensor_shape_check(input_posneg1___b_oEXPAND_i, batch, out_dim, in_dim)

                grad_like_for___raw_weight___before_sum___b_o_i = input_posneg1___b_oEXPAND_i*target___b_o_EXPANDi
                assert grad_like_for___raw_weight___before_sum___b_o_i.shape == torch.Size([batch, out_dim, in_dim])
                grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___before_sum___b_o_i.sum(dim=0)
                assert grad_like_for___raw_weight___o_i.shape == torch.Size([out_dim, in_dim])

                #控制变量的范围是优化器的事情.
                pass


            if "input___b_i.requires_grad":
                #<  accuracy
                target_posneg1___b_o = target___b_o.gt(0.)
                target_posneg1___b_o = target_posneg1___b_o.to(torch.int32)
                target_posneg1___b_o = target_posneg1___b_o*2 -1
                assert _either_1_or_neg1(target_posneg1___b_o)
                assert target_posneg1___b_o.shape == torch.Size([batch, out_dim])
                element_mul_of_target_and_output___b_o = target_posneg1___b_o * output_posneg1___b_o
                element_mul_of_target_and_output___b_o = element_mul_of_target_and_output___b_o.to(torch.float32)
                accuracy___o = element_mul_of_target_and_output___b_o.mean(dim=0)
                accuracy___o = (accuracy___o +1.)*0.5
                assert accuracy___o.shape == torch.Size([out_dim])
                assert accuracy___o.ge(0.).all()
                assert accuracy___o.le(1.).all()
                #assert False, "the sharpness-controlled softmax also is not tested."
                sharpen_factor__from_accuracy___o:torch.Tensor = accuracy___o*SOME_HYPER_PARAM___s
                assert sharpen_factor__from_accuracy___o.shape == torch.Size([out_dim])
                assert sharpen_factor__from_accuracy___o.ge(0.).all()
                sharpen_factor__from_accuracy___o_EXPANDi = sharpen_factor__from_accuracy___o. \
                        reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])
                assert sharpen_factor__from_accuracy___o_EXPANDi.shape == torch.Size([out_dim, in_dim])

                sharpened_raw_weight___o_i = raw_weight___o_i*sharpen_factor__from_accuracy___o_EXPANDi

                soft_part_of_the_one_hot___o_i = sharpened_raw_weight___o_i.softmax(dim=1)
                _assert_only___sum_of_so_how_should_I_name_it___o = soft_part_of_the_one_hot___o_i.sum(dim=1)
                assert _assert_only___sum_of_so_how_should_I_name_it___o.shape == torch.Size([out_dim])
                assert _tensor_equal(_assert_only___sum_of_so_how_should_I_name_it___o, 
                                        torch.ones_like(_assert_only___sum_of_so_how_should_I_name_it___o))

                if "some pytorch feature test" and True:
                    a = torch.zeros(size=[2,7])
                    a[[0,1], [6,3]] = 1.
                    assert _tensor_equal(a, [   [0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 1, 0, 0, 0]])
                    
                    a = torch.zeros(size=[2,7])
                    a[:, [6,3]] = 1.
                    assert _tensor_equal(a, [   [0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 1, 0, 0, 0]]) == False
                    pass
                iota_of_out = iota(out_dim)
                hard_part_of_the_one_hot___o_i = torch.zeros_like(soft_part_of_the_one_hot___o_i)
                hard_part_of_the_one_hot___o_i[iota_of_out, index_of_max_of_raw_weight___o] = 1.
                

                accuracy___o_EXPANDi = accuracy___o.reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])
                assert _tensor_shape_check(accuracy___o_EXPANDi, out_dim, in_dim)
                the_one_hot___o_i =       accuracy___o_EXPANDi  * hard_part_of_the_one_hot___o_i + \
                                    (1. - accuracy___o_EXPANDi) * soft_part_of_the_one_hot___o_i
                
                
                the_one_hot___EXPANDb_o_i = the_one_hot___o_i. \
                        reshape(shape=[1, the_one_hot___o_i.shape[0], the_one_hot___o_i.shape[1]]). \
                        expand(size=[target___b_o_EXPANDi.shape[0], -1, -1])
                assert the_one_hot___EXPANDb_o_i.shape == torch.Size([batch, out_dim, in_dim])
                
                
                grad_like_for___input___before_sum___b_o_i = the_one_hot___EXPANDb_o_i*target___b_o_EXPANDi
                assert grad_like_for___input___before_sum___b_o_i.shape == torch.Size([batch, out_dim, in_dim])
                grad_like_for___input___b_i = grad_like_for___input___before_sum___b_o_i.sum(dim=1)
                assert grad_like_for___input___b_i.shape == torch.Size([batch, in_dim])

                #更精细的控制是gramo的事情。
                pass
                #return ???????????
            pass#/ test


        if "一个模板，不用于测试。测试在后面" and False:
            batch = 3
            in_dim = 5
            out_dim = 2
            SOME_HYPER_PARAM___s = 1.

            #<  all the data.
            target___b_o = torch.tensor([   [1., 1,],
                                            [1., 1,],
                                            [1., 1,],])
            assert _tensor_shape_check(target___b_o, batch, out_dim)

            input_posneg1___b_i = torch.tensor([[1., 1, 1, 1, 1,],
                                                [1., 1, 1, 1, 1,],
                                                [1., 1, 1, 1, 1,],])
            assert _tensor_shape_check(input_posneg1___b_i, batch, in_dim)
            assert _either_1_or_neg1(input_posneg1___b_i)

            raw_weight___o_i = torch.tensor([   [0., -0.5, -1, -1, -1,],
                                                [0., -0.5, -1, -1, -1,],])
            assert _tensor_shape_check(raw_weight___o_i, out_dim, in_dim)
            assert raw_weight___o_i.ge(-1.).all()
            assert raw_weight___o_i.le( 0.).all()

            #recomputation
            index_of_max_of_raw_weight___o = raw_weight___o_i.max(dim=1).indices
            output_posneg1___b_o = input_posneg1___b_i[:, index_of_max_of_raw_weight___o]
            assert _either_1_or_neg1(output_posneg1___b_o)

            #<  real payload
            #<  init results to None
            grad_like_for___input___b_i     :torch.Tensor|None = None
            grad_like_for___raw_weight___o_i:torch.Tensor|None = None

            if "raw_weight___o_i.requires_grad" or "input___b_i.requires_grad":
                target___b_o_EXPANDi = target___b_o.reshape(shape=[target___b_o.shape[0], target___b_o.shape[1], 1]). \
                        expand(size=[-1, -1, input_posneg1___b_i.shape[1]])
                pass


            if "raw_weight___o_i.requires_grad":
                input_posneg1___b_oEXPAND_i = input_posneg1___b_i.reshape(shape=[input_posneg1___b_i.shape[0], 1, input_posneg1___b_i.shape[1]]). \
                        expand(size=[-1, target___b_o.shape[1], -1])

                grad_like_for___raw_weight___before_sum___b_o_i = input_posneg1___b_oEXPAND_i*target___b_o_EXPANDi
                grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___before_sum___b_o_i.sum(dim=0)

                #控制变量的范围是优化器的事情.
                pass


            if "input___b_i.requires_grad":
                #<  accuracy
                target_posneg1___b_o = target___b_o.gt(0.)
                target_posneg1___b_o = target_posneg1___b_o.to(torch.int32)
                target_posneg1___b_o = target_posneg1___b_o*2 -1
                assert _either_1_or_neg1(target_posneg1___b_o)

                element_mul_of_target_and_output___b_o = target_posneg1___b_o * output_posneg1___b_o
                element_mul_of_target_and_output___b_o = element_mul_of_target_and_output___b_o.to(torch.float32)

                accuracy___o = element_mul_of_target_and_output___b_o.mean(dim=0)
                accuracy___o = (accuracy___o +1.)*0.5
                assert accuracy___o.ge(0.).all()
                assert accuracy___o.le(1.).all()

                #assert False, "the sharpness-controlled softmax also is not tested."
                sharpen_factor__from_accuracy___o:torch.Tensor = accuracy___o*SOME_HYPER_PARAM___s
                assert sharpen_factor__from_accuracy___o.ge(0.).all()
                sharpen_factor__from_accuracy___o_EXPANDi = sharpen_factor__from_accuracy___o. \
                        reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])

                sharpened_raw_weight___o_i = raw_weight___o_i*sharpen_factor__from_accuracy___o_EXPANDi

                soft_part_of_the_one_hot___o_i = sharpened_raw_weight___o_i.softmax(dim=1)
                # _assert_only___sum_of_so_how_should_I_name_it___o = soft_part_of_the_one_hot___o_i.sum(dim=1)
                # assert _assert_only___sum_of_so_how_should_I_name_it___o.shape == torch.Size([out_dim])
                # assert _tensor_equal(_assert_only___sum_of_so_how_should_I_name_it___o, 
                #                         torch.ones_like(_assert_only___sum_of_so_how_should_I_name_it___o))

                iota_of_out = iota(out_dim)
                hard_part_of_the_one_hot___o_i = torch.zeros_like(soft_part_of_the_one_hot___o_i)
                hard_part_of_the_one_hot___o_i[iota_of_out, index_of_max_of_raw_weight___o] = 1.
                #linear interpolation.
                accuracy___o_EXPANDi = accuracy___o.reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])
                the_one_hot_like___o_i =       accuracy___o_EXPANDi  * hard_part_of_the_one_hot___o_i + \
                                    (1. - accuracy___o_EXPANDi) * soft_part_of_the_one_hot___o_i
                
                #the backward mapping relationship
                the_one_hot___EXPANDb_o_i = the_one_hot_like___o_i. \
                        reshape(shape=[1, the_one_hot_like___o_i.shape[0], the_one_hot_like___o_i.shape[1]]). \
                        expand(size=[target___b_o_EXPANDi.shape[0], -1, -1])
                
                
                grad_like_for___input___before_sum___b_o_i = the_one_hot___EXPANDb_o_i*target___b_o_EXPANDi
                grad_like_for___input___b_i = grad_like_for___input___before_sum___b_o_i.sum(dim=1)

                #更精细的控制是gramo的事情。
                pass
                #return ???????????
            pass#/ test









        return
    ____backward_algo_test____()
    pass

def _algo_test__backward_function(input_posneg1___b_i:torch.Tensor, 
            target___b_o:torch.Tensor, raw_weight___o_i:torch.Tensor, 
            SOME_HYPER_PARAM___s = 1. )->tuple[torch.Tensor|None, torch.Tensor|None]:
    '''
    !!!!!!!!!!!!!!!!!!!! assert False, "the sharpness-controlled softmax also is not tested."
    
    return grad_like_for___input___b_i, grad_like_for___raw_weight___o_i

    This is algo test for the backward function.
    '''
    #<  safety
    assert _either_1_or_neg1(input_posneg1___b_i)
    assert raw_weight___o_i.ge(-1.).all()
    assert raw_weight___o_i.le( 0.).all()
    #<  shape
    assert input_posneg1___b_i.shape.__len__() == 2
    batch = input_posneg1___b_i.shape[0]  
    in_dim = input_posneg1___b_i.shape[1]  

    assert target___b_o.shape.__len__() == 2
    assert target___b_o.shape[0] == batch, "not sure if this one is wrong or the previous one?"
    out_dim = target___b_o.shape[1]

    assert _tensor_shape_check(raw_weight___o_i, out_dim, in_dim), "not sure if this one is wrong or the previous one?"


    #<  init results to None
    grad_like_for___input___b_i     :torch.Tensor|None = None
    grad_like_for___raw_weight___o_i:torch.Tensor|None = None

    #<  real payload
    if "raw_weight___o_i.requires_grad" or "input___b_i.requires_grad":
        target___b_o_EXPANDi = target___b_o.reshape(shape=[target___b_o.shape[0], target___b_o.shape[1], 1]). \
                expand(size=[-1, -1, input_posneg1___b_i.shape[1]])
        pass# if "raw_weight___o_i.requires_grad" or "input___b_i.requires_grad":


    if "raw_weight___o_i.requires_grad":
        input_posneg1___b_oEXPAND_i = input_posneg1___b_i.reshape(shape=[input_posneg1___b_i.shape[0], 1, input_posneg1___b_i.shape[1]]). \
                expand(size=[-1, target___b_o.shape[1], -1])

        grad_like_for___raw_weight___before_sum___b_o_i = input_posneg1___b_oEXPAND_i*target___b_o_EXPANDi
        grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___before_sum___b_o_i.sum(dim=0)

        #控制变量的范围是优化器的事情.
        pass# if "raw_weight___o_i.requires_grad":


    if "input___b_i.requires_grad":
        #recomputation
        output_posneg1___b_o, index_of_max_of_raw_weight___o = _test___DNN_forward___full_safety( \
                input___b_i = input_posneg1___b_i, raw_weight___o_i = raw_weight___o_i, input_is_already_posneg1=True)
        # index_of_max_of_raw_weight___o = raw_weight___o_i.max(dim=1).indices
        # output_posneg1___b_o = input_posneg1___b_i[:, index_of_max_of_raw_weight___o]
        # assert _either_1_or_neg1(output_posneg1___b_o)
        #<  accuracy
        #target_posneg1___b_o = DNN___to_posneg1(target___b_o) ??????????????????????????????????
        # old code    target_posneg1___b_o = target___b_o.gt(0.)
        # target_posneg1___b_o = target_posneg1___b_o.to(torch.int32)
        # target_posneg1___b_o = target_posneg1___b_o*2 -1
        # assert _either_1_or_neg1(target_posneg1___b_o)


        accuracy___o, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
                                                output_posneg1___b_o=output_posneg1___b_o, mean_per= "per_output" )
        assert return_value_name == "accuracy___o"
        assert _tensor_shape_check(accuracy___o, out_dim)
        # old code    element_mul_of_target_and_output___b_o = target_posneg1___b_o * output_posneg1___b_o
        # element_mul_of_target_and_output___b_o = element_mul_of_target_and_output___b_o.to(torch.float32)

        # accuracy___o = element_mul_of_target_and_output___b_o.mean(dim=0)
        # accuracy___o = (accuracy___o +1.)*0.5
        # assert accuracy___o.ge(0.).all()
        # assert accuracy___o.le(1.).all()



        # assert False, "the sharpness-controlled softmax also is not tested."
        #<  soft part of the backward mapping.
        assert SOME_HYPER_PARAM___s > 0.
        sharpen_factor__from_accuracy___o:torch.Tensor = accuracy___o*SOME_HYPER_PARAM___s
        assert _tensor_shape_check(sharpen_factor__from_accuracy___o, out_dim)
        assert sharpen_factor__from_accuracy___o.ge(0.).all()
        sharpen_factor__from_accuracy___o_EXPANDi = sharpen_factor__from_accuracy___o. \
                reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])

        sharpened_raw_weight___o_i = raw_weight___o_i*sharpen_factor__from_accuracy___o_EXPANDi
        soft_part_of_the_one_hot___o_i = sharpened_raw_weight___o_i.softmax(dim=1)
        #   a bit assertion
        _assert_only___sum_of_soft_part_of_the_one_hot___o = soft_part_of_the_one_hot___o_i.sum(dim=1)
        assert _tensor_shape_check(_assert_only___sum_of_soft_part_of_the_one_hot___o, out_dim)
        assert _tensor_equal(_assert_only___sum_of_soft_part_of_the_one_hot___o, 
                                torch.ones_like(_assert_only___sum_of_soft_part_of_the_one_hot___o))

        #<  hard part of the backward mapping.
        iota_of_out = iota(out_dim)
        hard_part_of_the_one_hot___o_i = torch.zeros_like(soft_part_of_the_one_hot___o_i)
        hard_part_of_the_one_hot___o_i[iota_of_out, index_of_max_of_raw_weight___o] = 1.
        #   a bit assertion
        _assert_only___sum_of_hard_part_of_the_one_hot___o = soft_part_of_the_one_hot___o_i.sum(dim=1)
        assert _tensor_shape_check(_assert_only___sum_of_hard_part_of_the_one_hot___o, out_dim)
        assert _tensor_equal(_assert_only___sum_of_hard_part_of_the_one_hot___o, 
                                torch.ones_like(_assert_only___sum_of_hard_part_of_the_one_hot___o))

        assert _tensor_equal(_assert_only___sum_of_hard_part_of_the_one_hot___o, 
                                torch.ones_like(_assert_only___sum_of_hard_part_of_the_one_hot___o))
        
        #linear interpolation.
        accuracy___o_EXPANDi = accuracy___o.reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])
        assert _tensor_shape_check(accuracy___o_EXPANDi, out_dim, in_dim)
        #   accurate output makes the corresponding target go through a  sharp  mapping. 
        # inaccurate output makes the corresponding target go through a blurred mapping. 
        the_one_hot_like___o_i =       accuracy___o_EXPANDi  * hard_part_of_the_one_hot___o_i + \
                            (1. - accuracy___o_EXPANDi) * soft_part_of_the_one_hot___o_i
        
        #the backward mapping relationship
        the_one_hot___EXPANDb_o_i = the_one_hot_like___o_i. \
                reshape(shape=[1, the_one_hot_like___o_i.shape[0], the_one_hot_like___o_i.shape[1]]). \
                expand(size=[target___b_o_EXPANDi.shape[0], -1, -1])
        assert _tensor_shape_check(the_one_hot___EXPANDb_o_i, batch, out_dim, in_dim)
        
        grad_like_for___input___before_sum___b_o_i = the_one_hot___EXPANDb_o_i*target___b_o_EXPANDi
        grad_like_for___input___b_i = grad_like_for___input___before_sum___b_o_i.sum(dim=1)
        assert isinstance(grad_like_for___input___b_i, torch.Tensor)
        assert _tensor_shape_check(grad_like_for___input___b_i, batch, in_dim)

        #更精细的层间控制是gramo的事情。（如果你不熟悉gramo，不一定是某一个特定的gramo，不一定是绿色五角星那个）
        pass

    return grad_like_for___input___b_i, grad_like_for___raw_weight___o_i
if "test" and False:
    def ____test_____algo_test__backward_function________grad_like_for___raw_weight___o_i()-> None:
        if "VISUAL     grad_like_for___raw_weight___o_i  distribution and how to protect it." and True:
            target_style:Literal["reasonable", "random"] = "random"

            batch = 1000
            in_dim = 500
            out_dim = 100
            for _ in range(5):
                input_posneg1___b_i = rand_sign(size=[batch, in_dim], dtype=torch.int32)
                assert _either_1_or_neg1(input_posneg1___b_i)

                match target_style:
                    case "reasonable":
                        _index_of_max_of_raw_weight___o = torch.randint(low=0, high=in_dim, size=[out_dim])
                        target___b_o = input_posneg1___b_i[:, _index_of_max_of_raw_weight___o]
                        assert _either_1_or_neg1(target___b_o)
                        pass
                    case "random":
                        target___b_o = torch.rand(size=[batch, out_dim])*2. -1. #  -1 to 1
                        pass
                    case _:
                        assert False, "unreachable"
                        pass
                    #end of match.

                ori__raw_weight___o_i = torch.rand(size=[out_dim, in_dim])*-1.

                _, grad_like_for___raw_weight___o_i = _algo_test__backward_function( \
                    input_posneg1___b_i=input_posneg1___b_i, target___b_o=target___b_o,raw_weight___o_i=ori__raw_weight___o_i)
                assert isinstance(grad_like_for___raw_weight___o_i, torch.Tensor)
                assert _tensor_shape_check(grad_like_for___raw_weight___o_i, out_dim, in_dim)
                assert grad_like_for___raw_weight___o_i.ge(-batch).all()
                assert grad_like_for___raw_weight___o_i.le( batch).all()

                from matplotlib import pyplot as plt
                plt.hist(grad_like_for___raw_weight___o_i.reshape([-1]), bins=100)
                plt.title(f"original    grad_like_for___raw_weight     out {out_dim}   in {in_dim}")
                plt.show()
                plt.clf()
                plt.cla()

                protected__grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___o_i - grad_like_for___raw_weight___o_i.max()
                assert protected__grad_like_for___raw_weight___o_i.le(0).all()
                protected__grad_like_for___raw_weight___o_i = protected__grad_like_for___raw_weight___o_i.to(torch.float32)
                _temp__mean_of__protected__grad_like_for___raw_weight___s = protected__grad_like_for___raw_weight___o_i.mean().abs()
                protected__grad_like_for___raw_weight___o_i = \
                        protected__grad_like_for___raw_weight___o_i/_temp__mean_of__protected__grad_like_for___raw_weight___s * 0.5
                assert protected__grad_like_for___raw_weight___o_i.le(0.).all()
                plt.hist(protected__grad_like_for___raw_weight___o_i.reshape([-1]), bins=100)
                plt.title(f"protected    grad_like_for___raw_weight     out {out_dim}   in {in_dim}")
                plt.show()
                plt.clf()
                plt.cla()
                pass#for _ 
            pass#/ test

        if "prototype      this is only a backup version.       dont use it." and False:
            assert False, "this is only a backup version."
            learning_rate = 0.5
            batch = 1000
            in_dim = 500
            out_dim = 100
            for _ in range(11):

                input_posneg1___b_i = rand_sign(size=[batch, in_dim], dtype=torch.int32)
                assert _either_1_or_neg1(input_posneg1___b_i)

                _index_of_max_of_raw_weight___o = torch.randint(low=0, high=in_dim, size=[out_dim])
                target___b_o = input_posneg1___b_i[:, _index_of_max_of_raw_weight___o]
                assert _either_1_or_neg1(target___b_o)
                # this is a pure random target target___b_o = torch.rand(size=[batch, out_dim])*2. -1. #  -1 to 1

                ori__raw_weight___o_i = torch.rand(size=[out_dim, in_dim])*-1.

                _, grad_like_for___raw_weight___o_i = _algo_test__backward_function( \
                    input_posneg1___b_i=input_posneg1___b_i, target___b_o=target___b_o,raw_weight___o_i=ori__raw_weight___o_i)
                assert isinstance(grad_like_for___raw_weight___o_i, torch.Tensor)
                assert _tensor_shape_check(grad_like_for___raw_weight___o_i, out_dim, in_dim)

                #<  target into pos neg 1 form.      This is for both ori and new.
                target_posneg1___b_o = target___b_o.gt(0.)
                target_posneg1___b_o = target_posneg1___b_o.to(torch.int32)*2 -1
                assert _either_1_or_neg1(target_posneg1___b_o)


                #<  calc    ori  output
                ori__output_posneg1___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i, 
                                                raw_weight___o_i=ori__raw_weight___o_i, input_is_already_posneg1=True)

                # old code    ori__index_of_max_of_raw_weight___o = ori__raw_weight___o_i.max(dim=1).indices
                # ori__output_posneg1___b_o = input_posneg1___b_i[:, ori__index_of_max_of_raw_weight___o]
                # assert _either_1_or_neg1(ori__output_posneg1___b_o)

                #<  ori   accuracy
                ori__accuracy___o, recommended_result_value_name = \
                        _test___binary_accuracy___full_safety(target___b_o=target_posneg1___b_o, 
                                output_posneg1___b_o=ori__output_posneg1___b_o, mean_per="per_output", target_is_already_posneg1=True)
                assert recommended_result_value_name == "accuracy___o"
                # old code     ori__element_mul_of_target_and_output___b_o = target_posneg1___b_o * ori__output_posneg1___b_o
                # ori__element_mul_of_target_and_output___b_o = ori__element_mul_of_target_and_output___b_o.to(torch.float32)

                # ori__accuracy___o = ori__element_mul_of_target_and_output___b_o.mean(dim=0)
                # ori__accuracy___o = ori__accuracy___o *0.5 + 0.5
                # assert ori__accuracy___o.ge(0.).all()
                # assert ori__accuracy___o.le(1.).all()


                #<  new raw_weight
                #new__raw_weight___o_i = ori__raw_weight___o_i+grad_like_for___raw_weight___o_i.to(torch.float32)/float(batch) * 0.3 #1w 需要一个自适应。 #没乘任何系数  可能要改？？？？？？？？
                new__raw_weight___o_i = _test___optimizer_algo___full_safety(ori__raw_weight___o_i = ori__raw_weight___o_i,
                        grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___o_i, learning_rate = learning_rate)



                # old code     protected__grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___o_i - grad_like_for___raw_weight___o_i.max()
                # assert protected__grad_like_for___raw_weight___o_i.le(0).all()
                # protected__grad_like_for___raw_weight___o_i = protected__grad_like_for___raw_weight___o_i.to(torch.float32)
                # _temp__mean_of__protected__grad_like_for___raw_weight___s = protected__grad_like_for___raw_weight___o_i.mean().abs()
                # protected__grad_like_for___raw_weight___o_i = \
                #         protected__grad_like_for___raw_weight___o_i/_temp__mean_of__protected__grad_like_for___raw_weight___s * 0.5
                # assert protected__grad_like_for___raw_weight___o_i.le(0.).all()
                # new__raw_weight___o_i = torch.tanh(ori__raw_weight___o_i + protected__grad_like_for___raw_weight___o_i* learning_rate) #没乘任何系数  可能要改？？？？？？？？
                # assert new__raw_weight___o_i.le(0.).all()


                #<  calc    new  output
                new__output_posneg1___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i, 
                                        raw_weight___o_i=new__raw_weight___o_i, input_is_already_posneg1=True)
                #  old code     new__index_of_max_of_raw_weight___o = new__raw_weight___o_i.max(dim=1).indices
                # new__output_posneg1___b_o = input_posneg1___b_i[:, new__index_of_max_of_raw_weight___o]
                # assert _either_1_or_neg1(new__output_posneg1___b_o)
                #<  new   accuracy
                new__accuracy___o, recommended_result_value_name = \
                        _test___binary_accuracy___full_safety(target___b_o=target_posneg1___b_o, 
                                output_posneg1___b_o=new__output_posneg1___b_o, mean_per="per_output", target_is_already_posneg1=True)
                assert recommended_result_value_name == "accuracy___o"

                # old code     new__element_mul_of_target_and_output___b_o = target_posneg1___b_o * new__output_posneg1___b_o
                # new__element_mul_of_target_and_output___b_o = new__element_mul_of_target_and_output___b_o.to(torch.float32)

                # new__accuracy___o = new__element_mul_of_target_and_output___b_o.mean(dim=0)
                # new__accuracy___o = new__accuracy___o *0.5 + 0.5
                # assert new__accuracy___o.ge(0.).all()
                # assert new__accuracy___o.le(1.).all()

                print(ori__accuracy___o.mean().item(), new__accuracy___o.mean().item())
                pass#for _
            pass#/ test


        if "prototype.    scan" and True:
            if "result" and False:
                # random rate 0.0
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc                = [ 0.500,  0.502,  0.507,  0.512,  0.532,  0.591,  0.753,  1.000]
                # acc gain           = [ 0.000,  0.001,  0.006,  0.011,  0.031,  0.089,  0.251,  0.499]
                # random rate 0.1
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc                = [ 0.501,  0.501,  0.504,  0.509,  0.530,  0.568,  0.724,  0.950]
                # acc gain           = [ 0.000,  0.001,  0.003,  0.009,  0.029,  0.067,  0.222,  0.449]
                # random rate 0.2
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc                = [ 0.502,  0.501,  0.504,  0.512,  0.526,  0.571,  0.699,  0.900]
                # acc gain           = [ 0.000,  0.001,  0.003,  0.010,  0.026,  0.070,  0.198,  0.400]
                # random rate 0.3
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc                = [ 0.501,  0.502,  0.504,  0.510,  0.527,  0.563,  0.666,  0.850]
                # acc gain           = [ 0.001,  0.002,  0.003,  0.009,  0.027,  0.062,  0.165,  0.349]
                # random rate 0.5
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc                = [ 0.500,  0.501,  0.505,  0.509,  0.524,  0.551,  0.619,  0.750]
                # acc gain           = [ 0.000,  0.001,  0.004,  0.008,  0.023,  0.051,  0.119,  0.249]
                # random rate 0.7
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc                = [ 0.501,  0.502,  0.504,  0.510,  0.522,  0.537,  0.572,  0.643]
                # acc gain           = [ 0.000,  0.001,  0.004,  0.009,  0.021,  0.037,  0.071,  0.143]
                pass

            #------------------#------------------#------------------
            number_of_tests = 20
            random_ratio_list = [0., 0.1, 0.2, 0.3, 0.5, 0.7]
            for ii_random_ratio in range(random_ratio_list.__len__()):
                random_ratio = random_ratio_list[ii_random_ratio]
                #print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------
                result_acc     :list = []#don't modify this.
                result_acc_gain:list = []#don't modify this.
                learning_rate_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.]################################################
                #_when_start = time.perf_counter()
                
                for learning_rate in learning_rate_list:
                    _raw_result__accuracy = torch.empty(size=[number_of_tests])
                    _raw_result__accuracy_gain = torch.empty(size=[number_of_tests])
                    for ii__test in range(number_of_tests):

                        batch = 1000
                        in_dim = 500
                        out_dim = 100
                        #<  dataset
                        input_posneg1___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                        assert _either_1_or_neg1(input_posneg1___b_i)

                        target_posneg1___b_o = partly_reasonable_label_from_input(input___b_i=input_posneg1___b_i, out_dim = out_dim,
                                    random_ratio=random_ratio, input_is_already_posneg1 = True)
                        assert _either_1_or_neg1(target_posneg1___b_o)
                        #<  model param
                        ori__raw_weight___o_i = torch.rand(size=[out_dim, in_dim])*-1.
                        #<  calc
                        _, grad_like_for___raw_weight___o_i = _algo_test__backward_function( \
                            input_posneg1___b_i=input_posneg1___b_i, target___b_o=target_posneg1___b_o,raw_weight___o_i=ori__raw_weight___o_i)
                        assert isinstance(grad_like_for___raw_weight___o_i, torch.Tensor)
                        assert _tensor_shape_check(grad_like_for___raw_weight___o_i, out_dim, in_dim)

                        #<  ori   accuracy
                        ori__output_posneg1___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i, 
                                                        raw_weight___o_i=ori__raw_weight___o_i, input_is_already_posneg1=True)
                        ori__accuracy___s, recommended_result_value_name = \
                                _test___binary_accuracy___full_safety(target___b_o=target_posneg1___b_o, 
                                        output_posneg1___b_o=ori__output_posneg1___b_o, mean_per="for_all", target_is_already_posneg1=True)
                        assert recommended_result_value_name == "accuracy___s"

                        #<  step
                        new__raw_weight___o_i = _test___optimizer_algo___full_safety(ori__raw_weight___o_i = ori__raw_weight___o_i,
                                grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___o_i, learning_rate = learning_rate)

                        #<  new   accuracy
                        new__output_posneg1___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i, 
                                                raw_weight___o_i=new__raw_weight___o_i, input_is_already_posneg1=True)
                        new__accuracy___s, recommended_result_value_name = \
                                _test___binary_accuracy___full_safety(target___b_o=target_posneg1___b_o, 
                                        output_posneg1___b_o=new__output_posneg1___b_o, mean_per="for_all", target_is_already_posneg1=True)
                        assert recommended_result_value_name == "accuracy___s"

                        #assert new__accuracy___s>ori__accuracy___s

                        _raw_result__accuracy[ii__test] = new__accuracy___s
                        _raw_result__accuracy_gain[ii__test] = new__accuracy___s - ori__accuracy___s

                        pass#for ii__test
                                            
                    result_acc     .append(_raw_result__accuracy.     mean().item())
                    result_acc_gain.append(_raw_result__accuracy_gain.mean().item())
                    
                    pass#for scanned_param
                #_when_end = time.perf_counter()
                #print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                

                print(f"random rate {random_ratio}")
                print(f"learning_rate_list = {str_the_list(learning_rate_list, 3)}")#########################
                print(f"acc              = {str_the_list(result_acc, 3)}")#########################
                print(f"acc gain         = {str_the_list(result_acc_gain, 3)}")#########################
                ################################
                pass#for ii_outter_param_set
            pass#/ test

        return 
    ____test_____algo_test__backward_function________grad_like_for___raw_weight___o_i()
    pass

















from typing import Any

class autograd_function_class_for__DigitalMapper_layer__2026(torch.autograd.Function):
    r'''
    forward input list:
    >>> input_posneg1___b_i(if the input is not posneg1, then the output is not guarunteed to be posneg1)
    >>> raw_weight___o_i (make sure this is output of get_useful())
    >>> SOME_HYPER_PARAM___s(>0.)
    
    backward input list:
    >>> target___b_o (shape must be [batch, out_features]. can be any value.)
    '''
    @staticmethod
    def forward(input_posneg1___b_i:torch.Tensor, raw_weight___o_i:torch.Tensor, 
                SOME_HYPER_PARAM___s:torch.Tensor)->torch.Tensor:
        # input___b_i:torch.Tensor = args[0]# shape must be [batch, in_features]
        # raw_weight___o_i:torch.Tensor = args[1]# shape must be [out_features, in_features]

        #<  safety
        #the only safety is if the input is already posneg1( positive or negative 1). 
        # I plan to do it outside in the module class.
        

        #<  real payload
        #copied from the function _test___DNN_forward___full_safety section "real payload"
        index_of_max_of_raw_weight___o = raw_weight___o_i.max(dim=1).indices
        output_posneg1___b_o = input_posneg1___b_i[:, index_of_max_of_raw_weight___o]

        return output_posneg1___b_o

    @staticmethod
    def setup_context(ctx, inputs, output)->None:
        input_posneg1___b_i:torch.Tensor = inputs[0]
        raw_weight___o_i:torch.Tensor = inputs[1]
        SOME_HYPER_PARAM___s:torch.Tensor = inputs[2]
        #output___b_o:torch.Tensor = output
        ctx.save_for_backward(input_posneg1___b_i, raw_weight___o_i, SOME_HYPER_PARAM___s)

    @staticmethod
    def backward(ctx:Any,  *grad_outputs:Any) -> Any:    #tuple[torch.Tensor|None, torch.Tensor|None]: 
        #copied from function _algo_test__backward_function
        #some safety checks are removed.
        #shape of g_in must be [batch, out_features]
        input_posneg1___b_i:torch.Tensor
        raw_weight___o_i:torch.Tensor
        target___b_o:torch.Tensor = grad_outputs[0]

        #debug code         ##################################################################################################################################
        assert isinstance(target___b_o, torch.Tensor)

        (input_posneg1___b_i, raw_weight___o_i, SOME_HYPER_PARAM___s) = ctx.saved_tensors
        #debug code         ##################################################################################################################################


        #<  init results to None
        grad_like_for___input___b_i     :torch.Tensor|None = None
        grad_like_for___raw_weight___o_i:torch.Tensor|None = None

        #<  real payload
        if "raw_weight___o_i.requires_grad" or "input___b_i.requires_grad":
            target___b_o_EXPANDi = target___b_o.reshape(shape=[target___b_o.shape[0], target___b_o.shape[1], 1]). \
                    expand(size=[-1, -1, input_posneg1___b_i.shape[1]])
            pass# if "raw_weight___o_i.requires_grad" or "input___b_i.requires_grad":


        if "raw_weight___o_i.requires_grad":
            input_posneg1___b_oEXPAND_i = input_posneg1___b_i.reshape(shape=[input_posneg1___b_i.shape[0], 1, input_posneg1___b_i.shape[1]]). \
                    expand(size=[-1, target___b_o.shape[1], -1])

            grad_like_for___raw_weight___before_sum___b_o_i = input_posneg1___b_oEXPAND_i*target___b_o_EXPANDi
            del input_posneg1___b_oEXPAND_i
            grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___before_sum___b_o_i.sum(dim=0)
            del grad_like_for___raw_weight___before_sum___b_o_i

            #控制变量的范围是优化器的事情.
            pass# if "raw_weight___o_i.requires_grad":


        if "input___b_i.requires_grad":
            
            #<  recomputation
            # output_posneg1___b_o, index_of_max_of_raw_weight___o = _test___DNN_forward___full_safety( \
            #         input_posneg1___b_i = input_posneg1___b_i, raw_weight___o_i = raw_weight___o_i, input_is_already_posneg1=True)
            #manually unroll.
            index_of_max_of_raw_weight___o = raw_weight___o_i.max(dim=1).indices
            output_posneg1___b_o = input_posneg1___b_i[:, index_of_max_of_raw_weight___o]
            #del index_of_max_of_raw_weight___o later


            #<  accuracy
            # accuracy___o, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
            #                                         output_posneg1___b_o=output_posneg1___b_o, mean_per= "per_output" )
            # assert return_value_name == "accuracy___o"
            # assert _tensor_shape_check(accuracy___o, out_dim)

            target_posneg1___b_o = target___b_o.gt(0.).to(torch.float32)*2. -1.  # DNN___to_posneg1
            assert _either_1_or_neg1(target_posneg1___b_o)######################################################################################

            # _test___binary_accuracy___full_safety
            element_mul_of_target_and_output___b_o = target_posneg1___b_o * output_posneg1___b_o
            element_mul_of_target_and_output___b_o = element_mul_of_target_and_output___b_o.to(torch.float32)
            accuracy___o = element_mul_of_target_and_output___b_o.mean(dim=0)
            del target_posneg1___b_o, output_posneg1___b_o, element_mul_of_target_and_output___b_o
            accuracy___o = (accuracy___o +1.)*0.5

            # assert False, "the sharpness-controlled softmax also is not tested."
            #<  soft part of the backward mapping.
            assert SOME_HYPER_PARAM___s > 0.#############################################################移出去
            sharpen_factor__from_accuracy___o:torch.Tensor = accuracy___o*SOME_HYPER_PARAM___s
            assert sharpen_factor__from_accuracy___o.dtype == accuracy___o.dtype##################################################
            assert sharpen_factor__from_accuracy___o.ge(0.).all()##########################################################
            sharpen_factor__from_accuracy___o_EXPANDi = sharpen_factor__from_accuracy___o. \
                    reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])
            sharpened_raw_weight___o_i = raw_weight___o_i*sharpen_factor__from_accuracy___o_EXPANDi
            soft_part_of_the_one_hot___o_i = sharpened_raw_weight___o_i.softmax(dim=1)
            del sharpen_factor__from_accuracy___o, sharpen_factor__from_accuracy___o_EXPANDi
            del sharpened_raw_weight___o_i
            #   a bit assertion       ###############################################################################
            _assert_only___sum_of_soft_part_of_the_one_hot___o = soft_part_of_the_one_hot___o_i.sum(dim=1)
            assert _tensor_equal(_assert_only___sum_of_soft_part_of_the_one_hot___o, 
                                    torch.ones_like(_assert_only___sum_of_soft_part_of_the_one_hot___o))

            #<  hard part of the backward mapping.
            out_dim = target___b_o.shape[1]
            iota_of_out = torch.linspace(start=0,end=out_dim-1,steps=out_dim,
                                            dtype=torch.int32, device=target___b_o.device)
            hard_part_of_the_one_hot___o_i = torch.zeros_like(soft_part_of_the_one_hot___o_i, device=target___b_o.device)
            hard_part_of_the_one_hot___o_i[iota_of_out, index_of_max_of_raw_weight___o] = 1.
            del index_of_max_of_raw_weight___o, iota_of_out
            #   a bit assertion      ###############################################################################
            _assert_only___sum_of_hard_part_of_the_one_hot___o = soft_part_of_the_one_hot___o_i.sum(dim=1)
            assert _tensor_shape_check(_assert_only___sum_of_hard_part_of_the_one_hot___o, out_dim)
            assert _tensor_equal(_assert_only___sum_of_hard_part_of_the_one_hot___o, 
                                    torch.ones_like(_assert_only___sum_of_hard_part_of_the_one_hot___o))

            assert _tensor_equal(_assert_only___sum_of_hard_part_of_the_one_hot___o, 
                                    torch.ones_like(_assert_only___sum_of_hard_part_of_the_one_hot___o))
            
            #linear interpolation.
            accuracy___o_EXPANDi = accuracy___o.reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])
            #   accurate output makes the corresponding target go through a  sharp  mapping. 
            # inaccurate output makes the corresponding target go through a blurred mapping. 
            the_one_hot_like___o_i =       accuracy___o_EXPANDi  * hard_part_of_the_one_hot___o_i + \
                                (1. - accuracy___o_EXPANDi) * soft_part_of_the_one_hot___o_i
            del accuracy___o, accuracy___o_EXPANDi
            del hard_part_of_the_one_hot___o_i, soft_part_of_the_one_hot___o_i
            
            #the backward mapping relationship
            the_one_hot___EXPANDb_o_i = the_one_hot_like___o_i. \
                    reshape(shape=[1, the_one_hot_like___o_i.shape[0], the_one_hot_like___o_i.shape[1]]). \
                    expand(size=[target___b_o_EXPANDi.shape[0], -1, -1])
            
            grad_like_for___input___before_sum___b_o_i = the_one_hot___EXPANDb_o_i*target___b_o_EXPANDi
            del the_one_hot_like___o_i, the_one_hot___EXPANDb_o_i
            grad_like_for___input___b_i = grad_like_for___input___before_sum___b_o_i.sum(dim=1)
            del grad_like_for___input___before_sum___b_o_i
            assert isinstance(grad_like_for___input___b_i, torch.Tensor)

            #更精细的层间控制是gramo的事情。（如果你不熟悉gramo，不一定是某一个特定的gramo，不一定是绿色五角星那个）
            pass

        return grad_like_for___input___b_i, grad_like_for___raw_weight___o_i, None # 手动维护类型吧。实在写不来了。。
    
    
    pass  # class
if "equivalence of this class version and the prototype function version" and False:
    def ____test____equivalence_of_this_class_and_______():
        if "forward" and True:

            batch = 100
            in_dim = 320
            out_dim = 77
            for _ in range(33):
                input_posneg1___b_i = rand_sign(size=[batch, in_dim])
                raw_weight___o_i = torch.rand(size=[out_dim, in_dim])*-1.
                function_output, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i, 
                                                                        raw_weight___o_i=raw_weight___o_i)
                class_output = autograd_function_class_for__DigitalMapper_layer__2026.apply(input_posneg1___b_i, 
                                                                    raw_weight___o_i, torch.tensor(1.))

                assert function_output.eq(class_output).all()
                assert _tensor_shape_check(function_output, batch, out_dim)
                assert _either_1_or_neg1(function_output)
                pass#for _
            pass#/ test


        if "backward" and True:

            batch = 100
            in_dim = 320
            out_dim = 77
            some_hyper_param = torch.tensor(1.)
            for _ in range(33):
                #<  dataset and forward
                input_posneg1___b_i = rand_sign(size=[batch, in_dim])
                input_posneg1___b_i.requires_grad_()
                raw_weight___o_i = torch.rand(size=[out_dim, in_dim], requires_grad=True)*-1.
                target___b_o = torch.randn(size=[batch, out_dim])
                function_output, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i.detach().clone(), 
                                                                        raw_weight___o_i=raw_weight___o_i.detach().clone())
                class_output:torch.Tensor = autograd_function_class_for__DigitalMapper_layer__2026.apply(input_posneg1___b_i, 
                                                                    raw_weight___o_i, some_hyper_param)
                assert function_output.eq(class_output).all()#redundant a bit.
                #<  backward
                class_output.backward(gradient=target___b_o, inputs=[input_posneg1___b_i, raw_weight___o_i])

                function_backward___grad_like_for___input___b_i, function_backward___grad_like_for___raw_weight___o_i = \
                        _algo_test__backward_function(input_posneg1___b_i = input_posneg1___b_i.detach().clone(), 
                                target___b_o = target___b_o.detach().clone(), raw_weight___o_i = raw_weight___o_i.detach().clone(), 
                                SOME_HYPER_PARAM___s = some_hyper_param.detach().clone())

                assert function_backward___grad_like_for___input___b_i.eq(input_posneg1___b_i.grad).all()
                assert _tensor_shape_check(function_backward___grad_like_for___input___b_i, batch, in_dim)
                assert function_backward___grad_like_for___raw_weight___o_i.eq(raw_weight___o_i.grad).all()
                assert _tensor_shape_check(function_backward___grad_like_for___raw_weight___o_i, out_dim, in_dim)

                pass#for _
            pass#/ test

        return
    ____test____equivalence_of_this_class_and_______()
    pass
if "dtype adaption" and False:
    def ____test____dtype_adaption_of____the_backward_function_in__autograd_subclass()->None:
        '''the output of forward function should be the same as the input. 
        It must keep the dtype across layers.'''

        float_dtype_list = [torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16]
        
        batch = 2
        in_dim = 3
        out_dim = 7
        for some_hype_param_dtype in float_dtype_list:
            for input_dtype in float_dtype_list:
                some_hyper_param = torch.tensor(1., dtype=some_hype_param_dtype)
                #<  dataset and forward
                input_posneg1___b_i = rand_sign(size=[batch, in_dim], dtype=input_dtype)
                assert _either_1_or_neg1(input_posneg1___b_i)
                input_posneg1___b_i.requires_grad_()

                raw_weight___o_i = torch.rand(size=[out_dim, in_dim], requires_grad=True)*-1.
                target___b_o = torch.randn(size=[batch, out_dim])

                class_output:torch.Tensor = autograd_function_class_for__DigitalMapper_layer__2026.apply(input_posneg1___b_i, 
                                                                    raw_weight___o_i, some_hyper_param)
                assert class_output.dtype == input_dtype

                #<  backward
                '''the dtype of the result of backward is aligned to the forward dtype by pytorch.
                So, no need to check it.'''
                class_output.backward(gradient=target___b_o, inputs=[input_posneg1___b_i, raw_weight___o_i])

                assert isinstance(input_posneg1___b_i.grad, torch.Tensor)
                assert input_posneg1___b_i.grad.dtype == input_posneg1___b_i.dtype
                assert isinstance(raw_weight___o_i.grad, torch.Tensor)
                assert raw_weight___o_i.grad.dtype == raw_weight___o_i.dtype
                pass#for some_hype_param_dtype
            pass#for input_dtype
        pass#/ test

        return
    ____test____dtype_adaption_of____the_backward_function_in__autograd_subclass()
    pass












'''2个申请内存的函数单独拿出来，方便以后调整。'''
def _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in(
        extra_in_dim:int, 
        in_dim_now:int, out_dim_now:int, )->int:
    '''return new_in_dim'''
    total_in_dim_needed = extra_in_dim+in_dim_now
    min_new_nelement = total_in_dim_needed*out_dim_now
    ONE_M = 1<<20
    if min_new_nelement<(ONE_M):
        return total_in_dim_needed*2
    ONE_G = 1<<30
    if min_new_nelement<(ONE_G):
        return int(total_in_dim_needed*1.25)
    return int(total_in_dim_needed*1.1)
    #end of function
if " test" and __DEBUG_ME__() and False:
    "感觉不用很严格？"
    def ____test_____only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in():
        if "result must be greater than input combined" and True:
            
            extra_in_dim = 10
            in_dim_now   = 10
            out_dim_now  = 10

            new_in_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 50


            extra_in_dim = 100
            in_dim_now   = 100
            out_dim_now  = 100

            new_in_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 500

            
            extra_in_dim = 1000
            in_dim_now   = 1000
            out_dim_now  = 1000

            new_in_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 3000


            extra_in_dim = 10000
            in_dim_now   = 10000
            out_dim_now  = 10000

            new_in_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 30000

            
            extra_in_dim = 100000
            in_dim_now   = 100000
            out_dim_now  = 100000

            new_in_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 300000

        return
    ____test_____only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in()
    pass

def _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out(
        extra_out_dim:int, 
        in_dim_now:int, out_dim_now:int, )->int:
    total_out_dim_needed = extra_out_dim+out_dim_now
    min_new_nelement = in_dim_now*total_out_dim_needed
    ONE_M = 1<<20
    if min_new_nelement<(ONE_M):
        return total_out_dim_needed*2
    ONE_G = 1<<30
    if min_new_nelement<(ONE_G):
        return int(total_out_dim_needed*1.25)
    return int(total_out_dim_needed*1.1)
    #end of function
if " test" and __DEBUG_ME__() and False:
    "感觉不用很严格？"
    def ____test______only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out():
        if "result must be greater than input combined" and True:
            
            extra_out_dim = 10
            in_dim_now    = 10
            out_dim_now   = 10

            new_out_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 50


            extra_out_dim = 100
            in_dim_now    = 100
            out_dim_now   = 100

            new_out_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 500


            extra_out_dim = 1000
            in_dim_now    = 1000
            out_dim_now   = 1000

            new_out_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 3000


            extra_out_dim = 10000
            in_dim_now    = 10000
            out_dim_now   = 10000

            new_out_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 30000


            extra_out_dim = 100000
            in_dim_now    = 100000
            out_dim_now   = 100000

            new_out_dim = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 300000


        return
    ____test______only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out()
    pass

'''随机初始化的的函数，单独拿出来，方便以后调整'''
def _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style( \
        out_features:int, in_features:int, device = None, dtype = None) -> torch.Tensor:
    result = torch.rand(size=[out_features, in_features], device=device, dtype=dtype)*-1.
    return result
if " test" and __DEBUG_ME__() and False:
    def ____test_____only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style():
        import random
        if "basic behavior" and True:
            for _ in range(33):
                out_features = random.randint(3,100)
                in_features = random.randint(5,87)
                some_random_tensor = _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style(
                    out_features = out_features, in_features=in_features)
                assert some_random_tensor.le(0.).all()
                pass#for _  
            pass#/ test

        if "dtype adaption" and True:
            
            some_random_tensor = _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style(
                    out_features = 3, in_features=2, dtype=torch.bfloat16)
            assert some_random_tensor.dtype == torch.bfloat16
            
            some_random_tensor = _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style(
                    out_features = 3, in_features=2, dtype=torch.float64)
            assert some_random_tensor.dtype == torch.float64
            pass
        
        if "device adaption" and True:
            
            some_random_tensor = _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style(
                    out_features = 3, in_features=2, device='cuda')
            assert some_random_tensor.device.type == 'cuda'
            pass

        return 
    ____test_____only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style()
    pass






class DigitalMapper_layer__2026(torch.nn.Module):
    in_dim         :int
    out_dim        :int
    _init_to_nan   :bool
    _raw_weight___oCAP_iCAP     :torch.nn.parameter.Parameter
    some_hyper_param            :torch.nn.parameter.Parameter
    _always_check_input_is_posneg1__in_forward :bool

    #customizable functions.
    _random_init_algo               :function
    _calc_bigger_capacity__for_in   :function
    _calc_bigger_capacity__for_out  :function
    # _calc_bigger_capacity

    def __init__(self, in_features: int, out_features: int, some_hyper_param = 1., 
                init_capacity__for_in = 16, init_capacity__for_out = 16, init_to_nan = True, \
                _dtype_for_raw_weight = torch.float32, 
                _always_check_input_is_posneg1__in_forward = True, 
                    device=None,
                    #dtype=None
                    ) -> None:  
        
        #this dtype is only for a inner memory in training. It must be float point number.
        #<  pytorch format
        #factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #<  safety
        assert _dtype_for_raw_weight in [torch.float, torch.float32, torch.float64, torch.float16, torch.bfloat16]
        if init_capacity__for_in < in_features:
            init_capacity__for_in = in_features
            pass
        if init_capacity__for_out < out_features:
            init_capacity__for_out = out_features
            pass

        self.in_dim = in_features
        self.out_dim = out_features
        self._init_to_nan = init_to_nan
        self._always_check_input_is_posneg1__in_forward = _always_check_input_is_posneg1__in_forward
        self._raw_weight___oCAP_iCAP = torch.nn.Parameter(torch.empty(
                size=[init_capacity__for_out, init_capacity__for_in], dtype=_dtype_for_raw_weight, device=device, 
                        requires_grad = False),#, **factory_kwargs), 
                        requires_grad = False)
        assert self._raw_weight___oCAP_iCAP.dtype in [torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16]
        if self._init_to_nan:
            self._raw_weight___oCAP_iCAP.fill_(torch.nan)
            pass

        if isinstance(some_hyper_param, float):
            self.some_hyper_param = torch.nn.Parameter(torch.tensor(some_hyper_param, dtype=torch.float64, device=device, 
                        requires_grad = False),#, **factory_kwargs), 
                        requires_grad = False)
            pass
        elif isinstance(some_hyper_param, torch.Tensor):
            self.some_hyper_param = torch.nn.Parameter(some_hyper_param.detach().clone(), requires_grad = False)
            pass
        self.some_hyper_param.data = self.some_hyper_param.to(self._raw_weight___oCAP_iCAP.device)
        #if this is a higher precision, the final result may get effected. It doesn't help. So let's keep it simple.
        self.some_hyper_param.data = self.some_hyper_param.to(self._raw_weight___oCAP_iCAP.dtype)
        assert self.some_hyper_param.data.requires_grad == False

        #<  modulized functions.
        self._random_init_algo = _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style
        self._raw_weight___oCAP_iCAP[:self.out_dim, :self.in_dim] = \
                self._random_init_algo(out_features, in_features, 
                        device=device, dtype=self._raw_weight___oCAP_iCAP.dtype)
        self._calc_bigger_capacity__for_in = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in
        self._calc_bigger_capacity__for_out = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out
        pass
    '''plain shape related.'''
    def capacity_of_in_dim(self)->int:
        '''get'''
        return self._raw_weight___oCAP_iCAP.shape[1] 
    def capacity_of_out_dim(self)->int:
        '''get'''
        return self._raw_weight___oCAP_iCAP.shape[0] 

    
    if "idk if it's still useful" and False:

        def get_one_hot_format(self)->torch.Tensor:
            iota_of_out_dim___o = iota(self.out_dim)
            index_of_max___o = self.raw_weight.max(dim=1).indices

            one_hot___o_i = torch.zeros_like(self._raw_weight___)
            one_hot___o_i[iota_of_out_dim___o, index_of_max___o] = 1.
            assert False, "untested."
            return one_hot___o_i

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
            return
        pass

    '''forward function for neural net like.'''
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''input.shape must be [ batch, _ ]. Use the input container.'''
        #<  rename with the shape
        #<  safety
        input___b_i = input
        if self._always_check_input_is_posneg1__in_forward:
            assert _either_1_or_neg1(input___b_i)
            pass

        #<  real payload
        output___b_o = autograd_function_class_for__DigitalMapper_layer__2026.apply(input___b_i, 
                                        self.get_useful_part_of_raw_weight(), self.some_hyper_param)
                                        # self._raw_weight___oCAP_iCAP, self.some_hyper_param)

        return output___b_o
        #end of function.
    
    '''index tool'''
    def get_max_index(self)->torch.Tensor:
        '''return index_of_max_of_useful_part___o
        '''
        _temp_useful_part = self.get_useful_part_of_raw_weight()
        index_of_max_of_useful_part___o = _temp_useful_part.max(dim=1).indices#copied from _test___DNN_forward___full_safety
        return index_of_max_of_useful_part___o

    def backward_index_quiry(self, output_slot_list:torch.Tensor)->torch.Tensor:
        index_of_max_of_useful_part___o = self.get_max_index()
        result = index_of_max_of_useful_part___o[output_slot_list]
        assert False, "untested      都必须比in dim小"
        return result


    ''' get useful part         squeeze'''
    def get_useful_part_of_raw_weight___and_squeeze(self, squeeze_in = False, squeeze_out = False)->torch.Tensor:
        self._squeeze(squeeze_in = squeeze_in, squeeze_out = squeeze_out)
        result = self._raw_weight___oCAP_iCAP[:self.out_dim,:self.in_dim]
        return result
    
    def get_useful_part_of_raw_weight(self)->torch.Tensor:
        result = self._raw_weight___oCAP_iCAP[:self.out_dim,:self.in_dim]
        return result
    
    def _squeeze(self, squeeze_in = False, squeeze_out = False):
        '''This function is designed for inner use inside this class. 

        If you need to control the timing and you know what you are doing, feel free to do anything.'''
        #<  safety
        assert squeeze_in or squeeze_out, "No real payload is asked. Why do you call this function? Or if you know what you are doing, comment this line out."

        #<  real payload
        # calc new capacity.
        _temp_new_out_capacity = self.capacity_of_out_dim()
        if squeeze_out:
            _temp_new_out_capacity = self.out_dim
            pass
        _temp_new_in_capacity = self.capacity_of_in_dim()
        if squeeze_in:
            _temp_new_in_capacity = self.in_dim
            pass

        _temp_new_memory = torch.empty(size=[_temp_new_out_capacity,_temp_new_in_capacity], 
                    dtype=self._raw_weight___oCAP_iCAP.dtype, device=self._raw_weight___oCAP_iCAP.device)
        if self._init_to_nan:
            _temp_new_memory.fill_(torch.nan)
            pass
        _temp_new_memory[self.out_dim, self.in_dim] = self._raw_weight___oCAP_iCAP.data[self.out_dim, self.in_dim]
        self._raw_weight___oCAP_iCAP.data = _temp_new_memory
        assert False, "untested"
        return

    '''add input slot'''
    def add_input_slot__to_the_tail(self, how_many = 0, new_raw_weight_part:torch.Tensor = torch.empty(size=[0]))->None:
        '''The param combination is either (0, some tensor), or (some number, empty tensor). 
        
        If new_raw_weight_part is not empty, its shape must be [out_dim, extra_in_dim]'''
        #<  wash the param.
        if how_many == 0:
            if new_raw_weight_part.nelement() == 0:
                assert False, "Bad param combination. Either how_many > 0, or new_raw_weight_part is provided."
                pass# if new_raw_weight_part.nelement() == 0:
            assert new_raw_weight_part.shape.__len__() == 2
            how_many = new_raw_weight_part.shape[1]
            pass# if how_many == 0:
        else:# how_many != 0:
            #assert how_many>0#duplicated. 
            assert new_raw_weight_part.nelement() == 0, "Bad param combination. Both are provided. Remove one of them."

            new_raw_weight_part = self._random_init_algo(self.out_dim, how_many, device=self._raw_weight___oCAP_iCAP.device, dtype=self._raw_weight___oCAP_iCAP.dtype)
            pass# else of if how_many == 0:
        assert new_raw_weight_part.shape[0] == self.out_dim

        
        #<  real payload

        with torch.no_grad():
                
            _size_after = self.in_dim + how_many
            if _size_after > self.capacity_of_in_dim():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity__for_in(
                        extra_in_dim = how_many, in_dim_now = self.in_dim, out_dim_now = self.out_dim)

                _temp___new_container = torch.empty(size=[self._raw_weight___oCAP_iCAP.shape[0], _temp___new_capacity],
                        dtype=self._raw_weight___oCAP_iCAP.dtype, device=self._raw_weight___oCAP_iCAP.device)
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:self.out_dim, :self.in_dim] = self.get_useful_part_of_raw_weight()
                self._raw_weight___oCAP_iCAP.data = _temp___new_container
                pass

            self._raw_weight___oCAP_iCAP.data[:self.out_dim, self.in_dim:self.in_dim + how_many] = new_raw_weight_part
            self.in_dim = _size_after
            return
            #end of function


    '''output slot'''
    def add_output_slot__to_the_tail(self, how_many = 0, new_raw_weight_part:torch.Tensor = torch.empty(size=[0]))->None:
        '''The param combination is either (0, some tensor), or (some number, empty tensor). 
        
        If new_raw_weight_part is not empty, its shape must be [out_dim, extra_in_dim]'''
        #<  wash the param.
        if how_many == 0:
            if new_raw_weight_part.nelement() == 0:
                assert False, "Bad param combination. Either how_many > 0, or new_raw_weight_part is provided."
                pass# if new_raw_weight_part.nelement() == 0:
            assert new_raw_weight_part.shape.__len__() == 2
            how_many = new_raw_weight_part.shape[0]
            pass# if how_many == 0:
        else:# how_many != 0:
            #assert how_many>0#duplicated. 
            assert new_raw_weight_part.nelement() == 0, "Bad param combination. Both are provided. Remove one of them."

            new_raw_weight_part = self._random_init_algo(how_many, self.in_dim, device=self._raw_weight___oCAP_iCAP.device, dtype=self._raw_weight___oCAP_iCAP.dtype)
            pass# else of if how_many == 0:
        assert new_raw_weight_part.shape[1] == self.in_dim
        
        #<  real payload

        with torch.no_grad():
                
            _size_after = self.out_dim + how_many
            if _size_after > self.capacity_of_out_dim():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity__for_out(
                        extra_out_dim = how_many, in_dim_now = self.in_dim, out_dim_now = self.out_dim)

                _temp___new_container = torch.empty(size=[_temp___new_capacity, self._raw_weight___oCAP_iCAP.shape[1]],
                        dtype=self._raw_weight___oCAP_iCAP.dtype, device=self._raw_weight___oCAP_iCAP.device)
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:self.out_dim, :self.in_dim] = self.get_useful_part_of_raw_weight()
                self._raw_weight___oCAP_iCAP.data = _temp___new_container
                pass

            self._raw_weight___oCAP_iCAP.data[self.out_dim:self.out_dim + how_many, :self.in_dim] = new_raw_weight_part
            self.out_dim = _size_after
            return
        pass# a dead pass to denote the end of function

    def keep_output_slot(self, keep_which:torch.Tensor, squeeze_the_input_dim = False)->None:
        '''This function also squeeze the memory to minimum.'''
        assert keep_which.shape.__len__() == 1
        assert keep_which.dtype == torch.bool
        #<  real payload
        _temp__useful_part = self.get_useful_part_of_raw_weight()
        _temp__useful_part = _temp__useful_part[keep_which,:]
        if squeeze_the_input_dim:
            self._raw_weight___oCAP_iCAP.data = _temp__useful_part
            pass
        else:# not to squeeze the input dim.
            _temp___keep_which___in_int = keep_which.to(torch.int32)
            how_many_to_keep = int(_temp___keep_which___in_int.sum().to(torch.int32).item())
            self._raw_weight___oCAP_iCAP.data = torch.empty(size=[how_many_to_keep, self.capacity_of_in_dim()])
            if self._init_to_nan:
                self._raw_weight___oCAP_iCAP.data.fill_(torch.nan)
                pass
            self._raw_weight___oCAP_iCAP.data[:, :self.in_dim] = _temp__useful_part
            pass
        self.out_dim = self._raw_weight___oCAP_iCAP.shape[0]
        return 
    def remove_output_slot(self, remove_which:torch.Tensor, squeeze_the_input_dim = False)->None:
            self.keep_output_slot(remove_which.logical_not(), 
                    squeeze_the_input_dim = squeeze_the_input_dim)
            return

    '''stringify'''
    def extra_repr(self) -> str:
        return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'

    # def __repr__(self):
    #     return f"{self.get_useful().__repr__()}, size:{self._size}, DNN input container 2026"
    # def __str__(self):
    #     return f"{self.get_useful().__str__()}, size:{self._size}, DNN input container 2026"

    pass# end of class.

if "forward in module class      basic behavior test" and False:
    def ____test____forward_in_module_class():
        if "allow non posneg1 input?" and True:
            the_layer = DigitalMapper_layer__2026(in_features=3, out_features=2, _always_check_input_is_posneg1__in_forward= True)
            output = the_layer(torch.tensor([[1., 1, 1], [1, -1, 1]]))
            #output = the_layer(torch.tensor([[1.1, 1, 1], [1, -1, 1]]))   this must NOT work.

            the_layer = DigitalMapper_layer__2026(in_features=3, out_features=2, _always_check_input_is_posneg1__in_forward= False)
            output = the_layer(torch.tensor([[1., 1, 1], [1, -1, 1]]))
            output = the_layer(torch.tensor([[1.1, 1, 1], [1, -1, 1]]))   
            pass#/ test

        if "equivalence with the algo test function?":

            for batch in[2, 13, 37]:
                for out_dim in[3, 14, 53]:
                    for in_dim in[5, 17, 71]:

                        for _ in range(16):
                            #<  dataset
                            input___b_i = rand_sign(size=[batch, in_dim])
                            assert input___b_i.shape == torch.Size([batch, in_dim])

                            #<  the layer
                            the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)

                            #<  calc
                            layer_output = the_layer(input___b_i)

                            function_output, _ = _test___DNN_forward___full_safety(input___b_i=input___b_i, 
                                    raw_weight___o_i=the_layer.get_useful_part_of_raw_weight().detach().clone())

                            #<  assert
                            assert layer_output.eq(function_output).all()
                            pass#for _

                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test

        return
    ____test____forward_in_module_class()
    pass
if "get_max_index in module class      basic behavior test" and False:
    def ____test____get_max_index_in_module_class():
        import random
        if "allow non posneg1 input?" and True:
            for _ in range(33):
                in_dim = random.randint(2,100)
                the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features= random.randint(2,100))
                the_max_index = the_layer.get_max_index()
                assert the_max_index.lt(in_dim).all()
                pass#for _
            pass#/ test
        return
    ____test____get_max_index_in_module_class()
    pass
    
#改形状，3种改法。反向查询索引。    

if "add input slot     algo test      and class equivalence" and __DEBUG_ME__() and False:
    def ____add_input____():

        if "add input.     full assert      no class     no shape scan" and True:

            batch = 2
            out_dim = 3
            in_dim___ori = 5
            in_dim___new = 7

            #<  dataset
            input___b_i = torch.tensor([[11.,  12,  13,  14,  15],
                                        [21.,  22,  23,  24,  25],])
            assert input___b_i.shape == torch.Size([batch, in_dim___ori])

            extra_input___b_ii = torch.tensor([
                    [511.,  512,  513,  514,  515,  516,  517],
                    [521.,  522,  523,  524,  525,  526,  527],])
            assert extra_input___b_ii.shape == torch.Size([batch, in_dim___new])
            
            #<  model param
            ori___training_buffer___o_i = torch.tensor([  
                                                    [0.1, 0.2, 0.3, 0.4, 0.5],
                                                    [0.1, 0.2, 0.3, 1.4, 0.5],
                                                    [0.1, 1.2, 0.3, 0.4, 0.5],
                                                    ])### 542 or 431
            
            #<  original    forward path
            ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input___b_i, 
                                    raw_weight___o_i = ori___training_buffer___o_i, 
                                    input_is_already_posneg1 = True, safety_check=False)# in order to fool the function. debug purpose.
            # old code      _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
            # ori___output___b_o = input___b_i[:, _temp_one_hot___o]
            # del _temp_one_hot___o
            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
            assert _tensor_equal(ori___output___b_o, torch.tensor([ [15,  14,  12,],  
                                                                    [25,  24,  22,],]))

            #<  the new shape
            in_dim_in_total = in_dim___ori + in_dim___new
            new___training_buffer___o_ii = torch.empty(size=[out_dim, in_dim_in_total])
            new___training_buffer___o_ii[:, :in_dim___ori] = ori___training_buffer___o_i[:, :in_dim___ori]
            new___training_buffer___o_ii[:, in_dim___ori:in_dim_in_total] = torch.tensor([  
                                                                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                                                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                                                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                                                ])# nothing.
            assert _tensor_equal(new___training_buffer___o_ii, torch.tensor([  
                            [0.1, 0.2, 0.3, 0.4, 0.5,           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                            [0.1, 0.2, 0.3, 1.4, 0.5,           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                            [0.1, 1.2, 0.3, 0.4, 0.5,           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                                                ]))

            assert new___training_buffer___o_ii.shape == torch.Size([out_dim, in_dim_in_total])

            #<  new        input
            new___input___b_i = torch.empty(size=[batch, in_dim___ori+in_dim___new])
            new___input___b_i[:, :in_dim___ori] = input___b_i
            new___input___b_i[:, in_dim___ori:] = extra_input___b_ii

            #<  new         forward path
            new___output___b_o, _temp___new____one_hot___o = _test___DNN_forward___full_safety(input___b_i=new___input___b_i, 
                                                                            raw_weight___o_i = new___training_buffer___o_ii, 
                                    input_is_already_posneg1 = True, safety_check=False)# in order to fool the function. debug purpose.

            # old code     _temp___new____one_hot___o = new___training_buffer___o_ii.argmax(dim=1)
            # new___output___b_o = new___input___b_i[:, _temp___new____one_hot___o]
            flag___same_output_as_ori___o = _temp___new____one_hot___o.lt(in_dim___ori)
            del _temp___new____one_hot___o
            #<  assert 
            assert _tensor_equal(new___output___b_o, torch.tensor([ [517,  14,  12,],  
                                                                    [527,  24,  22,],]))

            assert _tensor_equal(new___output___b_o[:, [1,2]], ori___output___b_o[:, [1,2]])

            assert _tensor_equal(new___output___b_o[:, [False, True, True]], ori___output___b_o[:, [False, True, True]])
            assert _tensor_equal(new___output___b_o[:, flag___same_output_as_ori___o], ori___output___b_o[:, flag___same_output_as_ori___o])

            pass#/ test 

        if "add input.     no assert        no class     with shape scan" and True:
            for batch in[2, 13, 37]:
                for out_dim in[3, 14, 53]:
                    for in_dim___ori in[5, 17, 71]:
                        for in_dim___new in[7, 21, 92]:
                            for _ in range(22):

                                #<  dataset
                                input___b_i = torch.rand(size=[batch, in_dim___ori])
                                extra_input___b_ii = torch.rand(size=[batch, in_dim___new])
                                
                                #<  model param
                                ori___training_buffer___o_i =torch.rand(size=[out_dim, in_dim___ori])
                                
                                #<  original    forward path

                                ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input___b_i, 
                                    raw_weight___o_i = ori___training_buffer___o_i, 
                                    input_is_already_posneg1 = True, safety_check=False)# in order to fool the function. debug purpose.
                                # old code      _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
                                # ori___output___b_o = input___b_i[:, _temp_one_hot___o]
                                # del _temp_one_hot___o
                                assert ori___output___b_o.shape == torch.Size([batch, out_dim])

                                #<  the new shape
                                in_dim_in_total = in_dim___ori + in_dim___new
                                new___training_buffer___o_ii = torch.empty(size=[out_dim, in_dim_in_total])
                                new___training_buffer___o_ii[:, :in_dim___ori] = ori___training_buffer___o_i[:, :in_dim___ori]
                                new___training_buffer___o_ii[:, in_dim___ori:in_dim_in_total] = torch.rand(size=[out_dim, in_dim___new])
                                assert new___training_buffer___o_ii.shape == torch.Size([out_dim, in_dim_in_total])
                                #<  new        input
                                new___input___b_i = torch.empty(size=[batch, in_dim___ori+in_dim___new])
                                new___input___b_i[:, :in_dim___ori] = input___b_i
                                new___input___b_i[:, in_dim___ori:] = extra_input___b_ii

                                #<  new         forward path

                                new___output___b_o, _temp___new____one_hot___o = _test___DNN_forward___full_safety(input___b_i=new___input___b_i, 
                                                                            raw_weight___o_i = new___training_buffer___o_ii, 
                                    input_is_already_posneg1 = True, safety_check=False)# in order to fool the function. debug purpose.
                                # old code     _temp___new____one_hot___o = new___training_buffer___o_ii.argmax(dim=1)
                                # new___output___b_o = new___input___b_i[:, _temp___new____one_hot___o]
                                flag___same_output_as_ori___o = _temp___new____one_hot___o.lt(in_dim___ori)
                                del _temp___new____one_hot___o
                                #<  assert 
                                assert _tensor_equal(new___output___b_o[:, flag___same_output_as_ori___o], ori___output___b_o[:, flag___same_output_as_ori___o])

                                pass#for _
                            pass#for batch
                        pass#for out_dim
                    pass#for in_dim___ori
                pass#for in_dim___new

            pass#/ test 
        #assert False, '''继续'''
        if "class equivalence" and True:
            for batch in[2, 13, 37]:
                for out_dim in[3, 14, 53]:
                    for in_dim___ori in[5, 17, 71]:
                        for in_dim___new in[7, 21, 92]:
                            for is_posneg1 in [True, False]:
                                for _ in range(6):

                                    #<  dataset
                                    if is_posneg1:
                                        input___b_i = rand_sign(size=[batch, in_dim___ori], dtype=torch.float32)
                                        extra_input___b_ii = rand_sign(size=[batch, in_dim___new], dtype=torch.float32)
                                        pass
                                    else:#debug purpose.
                                        input___b_i = torch.randn(size=[batch, in_dim___ori])#debug purpose.
                                        extra_input___b_ii = torch.randn(size=[batch, in_dim___new])#debug purpose.
                                        pass
                                    
                                    #<  model param
                                    if is_posneg1:
                                        the_layer = DigitalMapper_layer__2026(in_features=in_dim___ori, out_features=out_dim)
                                        pass
                                    else:#debug purpose.
                                        the_layer = DigitalMapper_layer__2026(in_features=in_dim___ori, out_features=out_dim, 
                                                            _always_check_input_is_posneg1__in_forward = False)#debug purpose.
                                        pass

                                    ori___training_buffer___o_i = the_layer.get_useful_part_of_raw_weight().detach().clone()
                                    
                                    #<  original    forward path
                                    layer_ori___output___b_o = the_layer(input___b_i)


                                    ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input___b_i, 
                                        raw_weight___o_i = ori___training_buffer___o_i, 
                                        input_is_already_posneg1 = True, safety_check=False)# in order to fool the function. debug purpose.
                                    # old code      _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
                                    # ori___output___b_o = input___b_i[:, _temp_one_hot___o]
                                    # del _temp_one_hot___o
                                    assert ori___output___b_o.shape == torch.Size([batch, out_dim])

                                    assert layer_ori___output___b_o.eq(ori___output___b_o).all()

                                    #<  the new shape
                                    the_layer.add_input_slot__to_the_tail(how_many=in_dim___new)

                                    in_dim_in_total = in_dim___ori + in_dim___new
                                    new___training_buffer___o_ii = torch.empty(size=[out_dim, in_dim_in_total])
                                    new___training_buffer___o_ii[:, :in_dim___ori] = ori___training_buffer___o_i[:, :in_dim___ori]
                                    #new___training_buffer___o_ii[:, in_dim___ori:in_dim_in_total] =  torch.rand(size=[out_dim, in_dim___new])
                                    new___training_buffer___o_ii[:, in_dim___ori:in_dim_in_total] =  the_layer._raw_weight___oCAP_iCAP[:out_dim, in_dim___ori:in_dim_in_total]#########
                                    assert new___training_buffer___o_ii.shape == torch.Size([out_dim, in_dim_in_total])

                                    _temp___useful_part = the_layer.get_useful_part_of_raw_weight()
                                    assert _temp___useful_part.eq(new___training_buffer___o_ii).all()
                                    del _temp___useful_part

                                    #<  new input      the new in_dim is different.
                                    new___input___b_i = torch.empty(size=[batch, in_dim___ori+in_dim___new])
                                    new___input___b_i[:, :in_dim___ori] = input___b_i
                                    new___input___b_i[:, in_dim___ori:] = extra_input___b_ii

                                    #<  new         forward path
                                    layer_new___output___b_o = the_layer(new___input___b_i)
                                    _temp___layer_new____one_hot___o = the_layer.get_useful_part_of_raw_weight().argmax(dim=1)#debug purpose
                                    layer___flag___same_output_as_ori___o = _temp___layer_new____one_hot___o.lt(in_dim___ori)#debug purpose
                                    del _temp___layer_new____one_hot___o



                                    new___output___b_o, _temp___new____one_hot___o = _test___DNN_forward___full_safety(input___b_i=new___input___b_i, 
                                                                                raw_weight___o_i = new___training_buffer___o_ii, 
                                        input_is_already_posneg1 = True, safety_check=False)# in order to fool the function. debug purpose.
                                    # old code     _temp___new____one_hot___o = new___training_buffer___o_ii.argmax(dim=1)
                                    # new___output___b_o = new___input___b_i[:, _temp___new____one_hot___o]
                                    flag___same_output_as_ori___o = _temp___new____one_hot___o.lt(in_dim___ori)#debug purpose
                                    del _temp___new____one_hot___o

                                    assert layer_new___output___b_o.eq(new___output___b_o).all()
                                    assert layer___flag___same_output_as_ori___o.eq(flag___same_output_as_ori___o).all()#debug purpose

                                    #<  assert 
                                    _temp___iota_of_out = iota(out_dim)
                                    flag_in_index___same_output_as_ori___FAKEo = _temp___iota_of_out[flag___same_output_as_ori___o]
                                    del _temp___iota_of_out
                                    assert _tensor_equal(   new___output___b_o      [:, flag_in_index___same_output_as_ori___FAKEo], 
                                                            ori___output___b_o      [:, flag_in_index___same_output_as_ori___FAKEo])
                                    assert _tensor_equal(   layer_new___output___b_o[:, flag_in_index___same_output_as_ori___FAKEo], 
                                                            layer_new___output___b_o[:, flag_in_index___same_output_as_ori___FAKEo])

                                    pass#for _
                                pass#for is_posneg1
                            pass#for batch
                        pass#for out_dim
                    pass#for in_dim___ori
                pass#for in_dim___new

            pass#/ test 

        return 
    ____add_input____()
    pass
if "add input slot with specified new raw_weight" and __DEBUG_ME__() and False:
    def ____add_input_with_specified_new_raw_weight____():
        for in_dim in [3,6,11]:
            for out_dim in [2,8,15]:
                for _ in range(6):
                    the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                    the_layer.add_input_slot__to_the_tail(new_raw_weight_part = torch.ones(size=[out_dim, 1]))
                    the_max_index___o = the_layer.get_max_index()
                    assert the_max_index___o.eq(torch.ones(size=[out_dim])*in_dim).all()

                    pass#for _
                pass#for out_dim
            pass#for in_dim

        return
    ____add_input_with_specified_new_raw_weight____()
    pass
if "add output slot     algo test      and class equivalence" and __DEBUG_ME__() and False:
    def ____add_output____():

        if "add output.     full assert      no class     no shape scan" and True:

            batch = 2
            in_dim = 3
            out_dim___ori = 5
            out_dim___new = 7

            #<  dataset
            input___b_i = torch.tensor([[11.,  12,  13],
                                        [21.,  22,  23],])
            assert input___b_i.shape == torch.Size([batch, in_dim])

            #<  model param
            ori___training_buffer___o_i = torch.tensor([  
                                                    [0.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3],
                                                    [1.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3],
                                                    [1.1, 0.2, 0.3],
                                                    ])### 32121 or 21010
            
            #<  original    forward path
            ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                    raw_weight___o_i = ori___training_buffer___o_i, 
                    input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
            assert ori___output___b_o.shape == torch.Size([batch, out_dim___ori])
            assert _tensor_equal(ori___output___b_o, torch.tensor([ [13, 12, 11, 12, 11],  
                                                                    [23, 22, 21, 22, 21],]))

            #<  the new shape
            out_dim_in_total = out_dim___ori + out_dim___new
            new___training_buffer___oo_i = torch.empty(size=[out_dim_in_total, in_dim])
            new___training_buffer___oo_i[:out_dim___ori, :] = ori___training_buffer___o_i[:out_dim___ori, :in_dim]

            new___training_buffer___oo_i[out_dim___ori:out_dim_in_total, :in_dim] = torch.tensor([  
                                                                [0.1, 0.2, 0.3],
                                                                [0.1, 1.2, 0.3],
                                                                [1.1, 0.2, 0.3],
                                                                [0.1, 1.2, 0.3],
                                                                [1.1, 0.2, 0.3],
                                                                [0.1, 0.2, 0.3],
                                                                [1.1, 0.2, 0.3],
                                                                ])# 3212131 or 2101020
            assert _tensor_equal(new___training_buffer___oo_i, torch.tensor([  
                                                                [0.1, 0.2, 0.3],
                                                                [0.1, 1.2, 0.3],
                                                                [1.1, 0.2, 0.3],
                                                                [0.1, 1.2, 0.3],
                                                                [1.1, 0.2, 0.3],
                                                                [0.1, 0.2, 0.3],
                                                                [0.1, 1.2, 0.3],
                                                                [1.1, 0.2, 0.3],
                                                                [0.1, 1.2, 0.3],
                                                                [1.1, 0.2, 0.3],
                                                                [0.1, 0.2, 0.3],
                                                                [1.1, 0.2, 0.3],
                                                                ]))
            assert new___training_buffer___oo_i.shape == torch.Size([out_dim_in_total, in_dim])
            #<  new         forward path
            new___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                                raw_weight___o_i= new___training_buffer___oo_i, 
                    input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
            #<  assert 

            assert _tensor_equal(new___output___b_o, torch.tensor([ [13, 12, 11, 12, 11, 13, 12, 11, 12, 11, 13, 11],  
                                                                    [23, 22, 21, 22, 21, 23, 22, 21, 22, 21, 23, 21],]))
            assert _tensor_equal(new___output___b_o[:, :out_dim___ori], ori___output___b_o)

            pass#/ test 


        if "add input.     no assert        no class     with shape scan" and True:
            for batch in[2, 13, 37]:
                for in_dim in[3, 14, 53]:
                    for out_dim___ori in[5, 17, 71]:
                        for out_dim___new in[7, 21, 92]:
                            for _ in range(22):

                                #<  dataset
                                input___b_i = torch.rand(size=[batch, in_dim])
                                assert input___b_i.shape == torch.Size([batch, in_dim])

                                #<  model param
                                ori___training_buffer___o_i = torch.rand(size=[out_dim___ori, in_dim])*-1.# this *-1. doesn't matter in this test.
                                
                                #<  original    forward path
                                ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                                        raw_weight___o_i = ori___training_buffer___o_i, 
                                        input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
                                assert ori___output___b_o.shape == torch.Size([batch, out_dim___ori])

                                #<  the new shape
                                out_dim_in_total = out_dim___ori + out_dim___new
                                new___training_buffer___oo_i = torch.empty(size=[out_dim_in_total, in_dim])
                                new___training_buffer___oo_i[:out_dim___ori, :] = ori___training_buffer___o_i[:out_dim___ori, :in_dim]

                                new___training_buffer___oo_i[out_dim___ori:out_dim_in_total, :in_dim] = \
                                        torch.rand(size=[out_dim___new, in_dim])*-1.# this *-1. doesn't matter in this test.
                                assert new___training_buffer___oo_i.shape == torch.Size([out_dim_in_total, in_dim])
                                #<  new         forward path
                                new___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                                                    raw_weight___o_i= new___training_buffer___oo_i, 
                                        input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
                                #<  assert 
                                assert _tensor_equal(new___output___b_o[:, :out_dim___ori], ori___output___b_o)

                                pass#for _
                            pass#for batch
                        pass#for out_dim
                    pass#for in_dim___ori
                pass#for in_dim___new

            pass#/ test 


        if "class equivalence" and True:
            for batch in[2, 13, 37]:
                for in_dim in[3, 14, 53]:
                    for out_dim___ori in[5, 17, 71]:
                        for out_dim___new in[7, 21, 92]:
                            for is_posneg1 in [True, False]:
                                for _ in range(6):

                                    #<  dataset
                                    if is_posneg1:
                                        input___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                                        pass
                                    else:#debug purpose.
                                        input___b_i = torch.rand(size=[batch, in_dim])#debug purpose.
                                        pass
                                    assert input___b_i.shape == torch.Size([batch, in_dim])

                                    #<  model param
                                    if is_posneg1:
                                        the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim___ori)
                                        pass
                                    else:#debug purpose.
                                        the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim___ori, 
                                                            _always_check_input_is_posneg1__in_forward = False)#debug purpose.
                                        pass
                                    
                                    ori___training_buffer___o_i = the_layer.get_useful_part_of_raw_weight().detach().clone()
                                    assert _tensor_shape_check(ori___training_buffer___o_i, out_dim___ori, in_dim)

                                    #<  original    forward path
                                    layer_ori___output___b_o = the_layer(input___b_i)

                                    ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                                            raw_weight___o_i = ori___training_buffer___o_i, 
                                            input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
                                    assert ori___output___b_o.shape == torch.Size([batch, out_dim___ori])

                                    assert layer_ori___output___b_o.eq(ori___output___b_o).all()

                                    #<  the new shape
                                    the_layer.add_output_slot__to_the_tail(how_many=out_dim___new)

                                    out_dim_in_total = out_dim___ori + out_dim___new
                                    new___training_buffer___oo_i = torch.empty(size=[out_dim_in_total, in_dim])
                                    new___training_buffer___oo_i[:out_dim___ori, :] = ori___training_buffer___o_i[:out_dim___ori, :in_dim]
                                    new___training_buffer___oo_i[out_dim___ori:out_dim_in_total, :in_dim] = \
                                            the_layer._raw_weight___oCAP_iCAP[out_dim___ori:out_dim_in_total, :in_dim]
                                    
                                    assert new___training_buffer___oo_i.shape == torch.Size([out_dim_in_total, in_dim])

                                    assert the_layer.get_useful_part_of_raw_weight().eq(new___training_buffer___oo_i).all()

                                    #<  new         forward path
                                    layer_new___output___b_o = the_layer(input___b_i)

                                    new___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                                                        raw_weight___o_i= new___training_buffer___oo_i, 
                                            input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose

                                    assert layer_new___output___b_o.eq(new___output___b_o).all()
                                    #<  assert 
                                    assert _tensor_equal(layer_new___output___b_o[:, :out_dim___ori], layer_ori___output___b_o)
                                    assert _tensor_equal(new___output___b_o[:, :out_dim___ori], ori___output___b_o)
                                    assert the_layer.get_useful_part_of_raw_weight().eq(new___training_buffer___oo_i).all()

                                    pass#for _
                                pass#for is_posneg1
                            pass#for batch
                        pass#for out_dim
                    pass#for in_dim___ori
                pass#for in_dim___new

            pass#/ test 

        return 
    ____add_output____()
    pass
if "add output slot with specified new raw_weight" and __DEBUG_ME__() and False:
    def ____add_output_with_specified_new_raw_weight____():
        for in_dim in [3,6,11]:
            for out_dim in [2,8,15]:
                for _ in range(6):
                    the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                    the_layer.add_output_slot__to_the_tail(new_raw_weight_part = torch.ones(size=[1, in_dim]))
                    the_layer._raw_weight___oCAP_iCAP[out_dim, in_dim-1] = 2.123
                    the_max_index___o = the_layer.get_max_index()
                    assert the_max_index___o[out_dim-1+1] == in_dim-1

                    the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                    the_layer.add_output_slot__to_the_tail(new_raw_weight_part = torch.ones(size=[1, in_dim]))
                    the_layer._raw_weight___oCAP_iCAP[out_dim, 2] = 5.123
                    the_max_index___o = the_layer.get_max_index()
                    assert the_max_index___o[out_dim-1+1] == 2

                    pass#for _
                pass#for out_dim
            pass#for in_dim

        return
    ____add_output_with_specified_new_raw_weight____()
    pass




















if "delete output slot" and __DEBUG_ME__() and True:
    def ____delete_output____():
        if "delete output.      without class" and False:

            batch = 2
            out_dim = 5
            in_dim = 3
            #<  the answer
            keep_these_output = torch.tensor([1,1,1,0,1], dtype=torch.bool)
            new_out_dim = int(keep_these_output.sum().item())
            #assert isinstance(new_out_dim, int)

            #<  dataset
            input___b_i = torch.tensor([[11.,  12,  13],
                                        [21.,  22,  23],])
            #<  model param
            ori___training_buffer___o_i = torch.tensor([  
                                                    [0.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3],
                                                    [1.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3],
                                                    [1.1, 0.2, 0.3],])#32121
            #<  original    forward path
            ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                    raw_weight___o_i = ori___training_buffer___o_i, 
                    input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
            assert _tensor_equal(ori___output___b_o, torch.tensor([   [13, 12, 11, 12, 11],  
                                                                        [23, 22, 21, 22, 21],]))

            #<  the new shape
            new___training_buffer___o_i = ori___training_buffer___o_i[keep_these_output, :]
            assert new___training_buffer___o_i.shape == torch.Size([new_out_dim, in_dim])
            #<  new         forward path
            new___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                    raw_weight___o_i= new___training_buffer___o_i, 
                    input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
            #<  assert 
            assert _tensor_equal(new___output___b_o, ori___output___b_o[:, keep_these_output])

            pass#/ test

        if "delete output,      without class         scan it" and False:
            for batch in [2,5,10]:
                for out_dim in [3,7,11]:
                    for in_dim in [6,9,13]:
                        for _ in range(5):
                            #<  the answer
                            keep_these_output = torch.rand(size=[out_dim])
                            keep_these_output = keep_these_output.gt(0.5)

                            new_out_dim = int(keep_these_output.sum().item())

                            #<  dataset
                            input___b_i = torch.rand(size=[batch, in_dim])
                            #<  model param
                            ori___training_buffer___o_i = torch.rand(size=[out_dim, in_dim])*-1.#the *-1. is debug purpose
                            #<  original    forward path
                            ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                                    raw_weight___o_i = ori___training_buffer___o_i, 
                                    input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
                            assert ori___output___b_o.shape == torch.Size([batch, out_dim])

                            #<  the new shape
                            new___training_buffer___o_i = ori___training_buffer___o_i[keep_these_output,:]
                            assert new___training_buffer___o_i.shape == torch.Size([new_out_dim, in_dim])

                            #<  new         forward path
                            new___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                                    raw_weight___o_i= new___training_buffer___o_i, 
                                    input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose

                            #<  assert 
                            assert _tensor_equal(new___output___b_o, ori___output___b_o[:, keep_these_output])
                            pass#for _
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test
        
        if "delete output,      with class         scan it" and True:
            for batch in [2,5,10]:
                for out_dim in [3,7,11]:
                    for in_dim in [6,9,13]:
                        for is_posneg1 in [True, False]:
                            for _ in range(5):
                                #<  the answer
                                keep_these_output = torch.rand(size=[out_dim])
                                keep_these_output = keep_these_output.gt(0.5)

                                new_out_dim = int(keep_these_output.sum().item())

                                #<  dataset

                                if is_posneg1:
                                    input___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                                    pass
                                else:#debug purpose.
                                    input___b_i = torch.rand(size=[batch, in_dim])#debug purpose.
                                    pass
                                assert input___b_i.shape == torch.Size([batch, in_dim])

                                #<  model param
                                if is_posneg1:
                                    the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                                    pass
                                else:#debug purpose.
                                    the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim, 
                                                _always_check_input_is_posneg1__in_forward = False)#debug purpose.
                                    pass

                                ori___training_buffer___o_i = the_layer.get_useful_part_of_raw_weight().detach().clone()
                                assert _tensor_shape_check(ori___training_buffer___o_i, out_dim___ori, in_dim)

                                #<  original    forward path
                                layer_ori___output___b_o = the_layer(input___b_i)

                                ori___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i.detach().clone(), 
                                                                        raw_weight___o_i = ori___training_buffer___o_i.detach().clone(), 
                                        input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose
                                assert ori___output___b_o.shape == torch.Size([batch, out_dim])

                                assert layer_ori___output___b_o.eq(ori___output___b_o).all()
                                #<  the new shape
                                the_layer.keep_output_slot(keep_these_output)
                                assert _tensor_shape_check(the_layer.get_useful_part_of_raw_weight(), new_out_dim, in_dim)

                                new___training_buffer___o_i = ori___training_buffer___o_i[keep_these_output, :]
                                assert _tensor_shape_check(new___training_buffer___o_i, new_out_dim, in_dim)

                                #<  new         forward path
                                layer_new___output___b_o = the_layer(input___b_i)


                                new___output___b_o, _ = _test___DNN_forward___full_safety(input___b_i = input___b_i, 
                                                                                    raw_weight___o_i= new___training_buffer___o_i, 
                                        input_is_already_posneg1 = True, safety_check=False)#in order to fool the function. debug purpose

                                assert layer_new___output___b_o.eq(new___output___b_o).all()
                                #<  assert 
                                1w
                                1w
                                1w
                                assert _tensor_equal(layer_new___output___b_o[:, :out_dim___ori], layer_ori___output___b_o)
                                assert _tensor_equal(new___output___b_o[:, :out_dim___ori], ori___output___b_o)
                                assert the_layer.get_useful_part_of_raw_weight().eq(new___training_buffer___oo_i).all()


                                new___output___b_o = the_layer.forward(input___b_i)
                                #<  assert 
                                assert _tensor_equal(new___output___b_o, ori___output___b_o[:, keep_these_output])
                                pass#for _
                            pass#for is_posneg1
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test




        return 
    ____delete_output____()
    pass











if "basic reshape.     data member test" and __DEBUG_ME__() and True:
    def ____test____basic_reshape____():

        if "add_input_slot__to_the_tail" and True:
            in_dim = 5
            out_dim = 33
            the_layer = DigitalMapper_layer__2026(5, 33)
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
            assert flag__is_nan.all()

            x = 7
            the_layer.add_input_slot__to_the_tail(how_many=x)
            new__in_dim = in_dim + x
            assert the_layer.in_dim == new__in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_in_dim() >= new__in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
            assert flag__is_nan.all()


            for in_dim in [5,17,33]:
                for out_dim in [7,21,37]:
                    for x in [12,27,57]:
                        for _ in range(33):
                            the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                
                            the_layer.add_input_slot__to_the_tail(how_many=x)
                            new__in_dim = in_dim + x
                            assert the_layer.in_dim == new__in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_in_dim() >= new__in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                            pass#for _
                        pass#for x
                    pass#for out_dim
                pass#for in_dim

            for in_dim in [5,17,33]:
                for out_dim in [7,21,37]:
                    for x in [12,27,57]:
                        for _ in range(33):
                            the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                
                            the_layer.add_input_slot__to_the_tail(new_raw_weight_part=torch.rand(size=[out_dim, x])-10.)
                            new__in_dim = in_dim + x
                            assert the_layer.in_dim == new__in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_in_dim() >= new__in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                            flag__new_added = the_layer._raw_weight___oCAP_iCAP[:out_dim, the_layer.in_dim:new__in_dim].lt(-5)
                            assert flag__new_added.all()

                            flag__ori = the_layer._raw_weight___oCAP_iCAP[:out_dim, :in_dim].gt(-2)
                            assert flag__ori.all()
                            pass#for _
                        pass#for x
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "add_output_slot__to_the_tail" and True:
            in_dim = 5
            out_dim = 7
            the_layer = DigitalMapper_layer__2026(5, 7)
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
            assert flag__is_nan.all()

            x = 12
            the_layer.add_output_slot__to_the_tail(how_many=x)
            new__out_dim = out_dim + x
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == new__out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            assert the_layer.capacity_of_out_dim() >= new__out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
            assert flag__is_nan.all()

            for in_dim in [5,17,33]:
                for out_dim in [7,21,37]:
                    for x in [12,27,57]:
                        for _ in range(33):
                            the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                
                            the_layer.add_output_slot__to_the_tail(how_many=x)
                            new__out_dim = out_dim + x
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == new__out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            assert the_layer.capacity_of_out_dim() >= new__out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                            
                            pass#for _
                        pass#for x
                    pass#for out_dim
                pass#for in_dim

            for in_dim in [5,17,33]:
                for out_dim in [7,21,37]:
                    for x in [12,27,57]:
                        for _ in range(33):
                            the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                
                            the_layer.add_output_slot__to_the_tail(new_raw_weight_part=torch.rand(size=[x, in_dim])-10.)
                            new__out_dim = out_dim + x
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == new__out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            assert the_layer.capacity_of_out_dim() >= new__out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                            
                            flag__new_added = the_layer._raw_weight___oCAP_iCAP[out_dim:out_dim+x, :in_dim].lt(-5)
                            assert flag__new_added.all()
                            flag__ori = the_layer._raw_weight___oCAP_iCAP[:out_dim, :in_dim].gt(-2)
                            assert flag__ori.all()
                            pass#for _
                        pass#for x
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "keep_output_slot      no squeeze on input" and False:
            '''to valid the result, this test calculates the max_index.'''
            '''
            before the function call, the data looks like
            aaaaaaaaaaaaaaa........
            aaaaaaaaaaaaaaa........
            aaaaaaaaaaaaaaa........
            aaaaaaaaaaaaaaa........
            .......................
            .......................
            (where a is a number, . is nan)
            after the function call, the data looks like
            aaaaaaaaaaaaaaa........
            aaaaaaaaaaaaaaa........
            
            In_dim capacity is untouched.
            '''
            in_dim = 4#5
            out_dim = 3#7
            the_layer = DigitalMapper_layer__2026(in_dim, out_dim, 
                    init_capacity__for_out = 9, init_capacity__for_in = 6)
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
            assert flag__is_nan.all()

            #<  manually 
            max_index = the_layer.get_max_index()
            keep_which = torch.tensor([1, 0, 1])#1,1,0,0,1,1,0])
            new__out_dim = keep_which.sum()
            keep_which = keep_which.to(torch.bool)
            #prin(the_layer._raw_weight___oCAP_iCAP.tolist())
            the_layer.keep_output_slot(keep_which, squeeze_the_input_dim=False)#calc
            #prin(the_layer._raw_weight___oCAP_iCAP.tolist())
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == new__out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() == new__out_dim#no useless output dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
            assert flag__is_nan.all()
            assert the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :].nelement() == 0
            # flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
            # assert flag__is_nan.all()

            manual__max_index = max_index[keep_which]
            new__max_index = the_layer.get_max_index()
            assert _tensor_equal(manual__max_index, new__max_index)

            #  re random useless numbers. If anything relies on this part, the assertion will probably fail.
            # the_layer._raw_weight___oCAP_iCAP.data[:, :] = \          how to fail the assertion.
            #         torch.randn_like(the_layer._raw_weight___oCAP_iCAP.data[:, :])*123.    how to fail the assertion.
            the_layer._raw_weight___oCAP_iCAP.data[:, in_dim:] = \
                    torch.randn_like(the_layer._raw_weight___oCAP_iCAP.data[:, in_dim:])*123.
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP.data)
            assert not flag__is_nan.any()
            new__max_index_2 = the_layer.get_max_index()
            assert _tensor_equal(manual__max_index, new__max_index_2)


            for in_dim in [5,17,53]:
                for out_dim in [7,21,67]:
                    for _ in range(15):

                        the_layer = DigitalMapper_layer__2026(in_dim, out_dim)
                        assert the_layer.in_dim == in_dim
                        assert the_layer.out_dim == out_dim
                        assert the_layer.capacity_of_in_dim() >= in_dim
                        assert the_layer.capacity_of_out_dim() >= out_dim
                        flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                        assert not flag__is_nan.any()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                        assert flag__is_nan.all()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                        assert flag__is_nan.all()
            
                        #<  manually 
                        max_index = the_layer.get_max_index()
                        keep_which = torch.randn(size=[out_dim]).gt(0.)
                        new__out_dim = keep_which.sum()
                        keep_which = keep_which.to(torch.bool)
                        the_layer.keep_output_slot(keep_which, squeeze_the_input_dim=False)#calc

                        assert the_layer.in_dim == in_dim
                        assert the_layer.out_dim == new__out_dim
                        assert the_layer.capacity_of_in_dim() >= in_dim
                        assert the_layer.capacity_of_out_dim() == new__out_dim#no useless output dim
                        flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                        assert not flag__is_nan.any()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                        assert flag__is_nan.all()
                        assert the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :].nelement() == 0
            
                        manual__max_index = max_index[keep_which]
                        new__max_index = the_layer.get_max_index()
                        assert _tensor_equal(manual__max_index, new__max_index)

                        #  re random useless numbers. If anything relies on this part, the assertion will probably fail.
                        the_layer._raw_weight___oCAP_iCAP.data[:, in_dim:] = \
                                torch.randn_like(the_layer._raw_weight___oCAP_iCAP.data[:, in_dim:])*123.
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP.data)
                        assert not flag__is_nan.any()
                        new__max_index_2 = the_layer.get_max_index()
                        assert _tensor_equal(manual__max_index, new__max_index_2)
            
                        pass#for _
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "keep_output_slot      with squeeze on input" and False:
            '''to valid the result, this test calculates the max_index.'''
            '''
            before the function call, the data looks like
            aaaaaaaaaaaaaaa........
            aaaaaaaaaaaaaaa........
            aaaaaaaaaaaaaaa........
            aaaaaaaaaaaaaaa........
            .......................
            .......................
            (where a is a number, . is nan)
            after the function call, the data looks like
            aaaaaaaaaaaaaaa
            aaaaaaaaaaaaaaa
            '''
            in_dim = 4#5
            out_dim = 3#7
            the_layer = DigitalMapper_layer__2026(in_dim, out_dim, 
                    init_capacity__for_out = 9, init_capacity__for_in = 6)
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
            assert flag__is_nan.all()

            #<  manually 
            max_index = the_layer.get_max_index()
            keep_which = torch.tensor([1, 0, 1])#1,1,0,0,1,1,0])
            new__out_dim = keep_which.sum()
            keep_which = keep_which.to(torch.bool)
            #prin(the_layer._raw_weight___oCAP_iCAP.tolist())
            the_layer.keep_output_slot(keep_which, squeeze_the_input_dim=True)#calc
            #prin(the_layer._raw_weight___oCAP_iCAP.tolist())
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == new__out_dim
            assert the_layer.capacity_of_in_dim() == in_dim
            assert the_layer.capacity_of_out_dim() == new__out_dim#no useless output dim
            assert the_layer._raw_weight___oCAP_iCAP.shape == the_layer.get_useful_part_of_raw_weight().shape

            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP)
            assert not flag__is_nan.any()

            manual__max_index = max_index[keep_which]
            new__max_index = the_layer.get_max_index()
            assert _tensor_equal(manual__max_index, new__max_index)

            for in_dim in [5,17,53]:
                for out_dim in [7,21,67]:
                    for _ in range(15):

                        the_layer = DigitalMapper_layer__2026(in_dim, out_dim)
                        assert the_layer.in_dim == in_dim
                        assert the_layer.out_dim == out_dim
                        assert the_layer.capacity_of_in_dim() >= in_dim
                        assert the_layer.capacity_of_out_dim() >= out_dim
                        flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                        assert not flag__is_nan.any()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[:, the_layer.in_dim:])
                        assert flag__is_nan.all()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[the_layer.out_dim:, :])
                        assert flag__is_nan.all()
            
                        #<  manually 
                        max_index = the_layer.get_max_index()
                        keep_which = torch.randn(size=[out_dim]).gt(0.)
                        new__out_dim = keep_which.sum()
                        keep_which = keep_which.to(torch.bool)
                        the_layer.keep_output_slot(keep_which, squeeze_the_input_dim=True)#calc

                        assert the_layer.in_dim == in_dim
                        assert the_layer.out_dim == new__out_dim
                        assert the_layer.capacity_of_in_dim() == in_dim
                        assert the_layer.capacity_of_out_dim() == new__out_dim#no useless output dim
                        assert the_layer._raw_weight___oCAP_iCAP.shape == the_layer.get_useful_part_of_raw_weight().shape
            
                        flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                        assert not flag__is_nan.any()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP)
                        assert not flag__is_nan.any()
            
                        manual__max_index = max_index[keep_which]
                        new__max_index = the_layer.get_max_index()
                        assert _tensor_equal(manual__max_index, new__max_index)
                        pass#for _
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        

        return 
    ____test____basic_reshape____()
    pass











assert False, "专用的优化器还没写"


