from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, _either_1_or_neg1, _tensor_shape_check, \
        iota, str_the_list
from pytorch_yagaodirac_v2.Random import rand_sign
from DNN2026.DNN_util import DNN___to_posneg1, _test___DNN_forward___full_safety, _test___binary_accuracy___full_safety, \
        partly_reasonable_label_from_input
import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######










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
if "test" and __DEBUG_ME__() and False:
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
                        assert _either_1_or_neg1(target_posneg1___b_o)#debug purpose
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
'''auto grad function class'''
'''auto grad function class'''
'''auto grad function class'''
class autograd_function_class_for__DigitalMapping_layer__2026(torch.autograd.Function):
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
        assert raw_weight___o_i.shape[1] == input_posneg1___b_i.shape[1]
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
if "equivalence of this class version and the prototype function version" and __DEBUG_ME__() and False:
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
                class_output = autograd_function_class_for__DigitalMapping_layer__2026.apply(input_posneg1___b_i, 
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
                class_output:torch.Tensor = autograd_function_class_for__DigitalMapping_layer__2026.apply(input_posneg1___b_i, 
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
if "dtype adaption" and __DEBUG_ME__() and False:
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

                class_output:torch.Tensor = autograd_function_class_for__DigitalMapping_layer__2026.apply(input_posneg1___b_i, 
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














'''pytorch feature test          I found the bug. I add some assert. I believe it's solved now. So this section is not important any more'''
if "trivial test" and __DEBUG_ME__() and True:
    def ____pytorch_feature_test():
        if "buffer_0" and True:
            buffer_0 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_0_clipped = buffer_0[:15, :20]
            input = rand_sign(size=[2,20], dtype=torch.float32)
            assert _either_1_or_neg1(input)
            #<  forward
            x:torch.Tensor
            x = autograd_function_class_for__DigitalMapping_layer__2026.apply(input, buffer_0_clipped, torch.tensor(1.))
            x.backward(gradient=rand_sign(size=[2, 15]), inputs=[buffer_0])# worked

            assert buffer_0_clipped.grad is None

            assert buffer_0.grad is not None
            assert isinstance(buffer_0.grad, torch.Tensor)
            pass#/ test

        if "buffer_0_clipped" and True:
            buffer_0 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_0_clipped = buffer_0[:15, :20]
            input = rand_sign(size=[2,20], dtype=torch.float32)
            assert _either_1_or_neg1(input)
            #<  forward
            x:torch.Tensor
            x = autograd_function_class_for__DigitalMapping_layer__2026.apply(input, buffer_0_clipped, torch.tensor(1.))
            x.backward(gradient=rand_sign(size=[2, 15]), inputs=[buffer_0_clipped])# worked

            assert buffer_0_clipped.grad is not None
            assert isinstance(buffer_0_clipped.grad, torch.Tensor)

            assert buffer_0.grad is None
            pass#/ test

        if "buffer_0, buffer_1" and True:
            buffer_0 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_0_clipped = buffer_0[:15, :20]
            buffer_1 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_1_clipped = buffer_1[:10, :15]
            input = rand_sign(size=[2,20], dtype=torch.float32)
            assert _either_1_or_neg1(input)
            #<  forward
            x     :torch.Tensor
            output:torch.Tensor
            x      = autograd_function_class_for__DigitalMapping_layer__2026.apply(input, buffer_0_clipped, torch.tensor(1.))
            output = autograd_function_class_for__DigitalMapping_layer__2026.apply(x,     buffer_1_clipped, torch.tensor(1.))
            #<  backward
            output.backward(gradient=rand_sign(size=[2, 10]), inputs=[buffer_0, buffer_1])

            assert buffer_0_clipped.grad is None

            assert buffer_0.grad is not None
            assert isinstance(buffer_0.grad, torch.Tensor)

            assert buffer_1_clipped.grad is None

            assert buffer_1.grad is not None
            assert isinstance(buffer_1.grad, torch.Tensor)
            pass#/ test

        if "buffer_0_clipped, buffer_1_clipped" and True:
            buffer_0 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_0_clipped = buffer_0[:15, :20]
            buffer_1 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_1_clipped = buffer_1[:10, :15]
            input = rand_sign(size=[2,20], dtype=torch.float32)
            assert _either_1_or_neg1(input)
            #<  forward
            x     :torch.Tensor
            output:torch.Tensor
            x      = autograd_function_class_for__DigitalMapping_layer__2026.apply(input, buffer_0_clipped, torch.tensor(1.))
            output = autograd_function_class_for__DigitalMapping_layer__2026.apply(x,     buffer_1_clipped, torch.tensor(1.))
            #<  backward
            output.backward(gradient=rand_sign(size=[2, 10]), inputs=[buffer_0_clipped, buffer_1_clipped])

            assert buffer_0_clipped.grad is not None
            assert isinstance(buffer_0_clipped.grad, torch.Tensor)

            assert buffer_0.grad is None

            assert buffer_1_clipped.grad is not None
            assert isinstance(buffer_1_clipped.grad, torch.Tensor)

            assert buffer_1.grad is None
            pass#/ test
        
        return
    ____pytorch_feature_test()
    pass













'''2个申请内存的函数单独拿出来，方便以后调整。'''
def _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in(
        extra_in_dim:int, 
        in_dim_now:int, out_dim_now:int, recommended_min = 16 )->int:
    '''return new_in_dim'''
    total_in_dim_needed = extra_in_dim+in_dim_now
    min_new_nelement = total_in_dim_needed*out_dim_now

    ONE_M = 1<<20
    if min_new_nelement<ONE_M:
        assert recommended_min>0
        result = total_in_dim_needed*2+recommended_min
        return result
    
    ONE_G = 1<<30
    if min_new_nelement<ONE_G:
        return int(total_in_dim_needed*1.25)
    return int(total_in_dim_needed*1.1)
    #end of function
if " test" and __DEBUG_ME__() and False:
    "感觉不用很严格？"
    def ____test_____only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in():
        if "result must be greater than input combined" and True:
            extra_in_dim = 0
            in_dim_now   = 0
            out_dim_now  = 10

            new_in_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 50


            extra_in_dim = 10
            in_dim_now   = 10
            out_dim_now  = 10

            new_in_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 100


            extra_in_dim = 100
            in_dim_now   = 100
            out_dim_now  = 100

            new_in_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 500

            
            extra_in_dim = 1000
            in_dim_now   = 1000
            out_dim_now  = 1000

            new_in_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 3000


            extra_in_dim = 10000
            in_dim_now   = 10000
            out_dim_now  = 10000

            new_in_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 30000

            
            extra_in_dim = 100000
            in_dim_now   = 100000
            out_dim_now  = 100000

            new_in_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in(
                    extra_in_dim = extra_in_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_in_dim >= extra_in_dim + in_dim_now
            assert new_in_dim < 300000

        return
    ____test_____only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in()
    pass

def _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out(
        extra_out_dim:int, 
        in_dim_now:int, out_dim_now:int, recommended_min = 16)->int:
    total_out_dim_needed = extra_out_dim+out_dim_now
    min_new_nelement = in_dim_now*total_out_dim_needed

    ONE_M = 1<<20
    if min_new_nelement<ONE_M:
        assert recommended_min>0
        result = total_out_dim_needed*2+recommended_min
        return result
    
    ONE_G = 1<<30
    if min_new_nelement<ONE_G:
        return int(total_out_dim_needed*1.25)
    return int(total_out_dim_needed*1.1)
    #end of function
if " test" and __DEBUG_ME__() and False:
    "感觉不用很严格？"
    def ____test______only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out():
        if "result must be greater than input combined" and True:

            extra_out_dim = 0
            in_dim_now    = 10
            out_dim_now   = 0

            new_out_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 50


            extra_out_dim = 10
            in_dim_now    = 10
            out_dim_now   = 10

            new_out_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 100


            extra_out_dim = 100
            in_dim_now    = 100
            out_dim_now   = 100

            new_out_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 500


            extra_out_dim = 1000
            in_dim_now    = 1000
            out_dim_now   = 1000

            new_out_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 3000


            extra_out_dim = 10000
            in_dim_now    = 10000
            out_dim_now   = 10000

            new_out_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 30000


            extra_out_dim = 100000
            in_dim_now    = 100000
            out_dim_now   = 100000

            new_out_dim = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out(
                    extra_out_dim = extra_out_dim, in_dim_now = in_dim_now, out_dim_now = out_dim_now)
            #<  assert
            assert new_out_dim >= extra_out_dim + out_dim_now
            assert new_out_dim < 300000


        return
    ____test______only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out()
    pass

'''随机初始化的的函数，单独拿出来，方便以后调整'''
def _only_for_DigitalMapping_layer__2026_to_use__reset_parameters__the_plain_rand01_style( \
        out_features:int, in_features:int, device = None, dtype = None) -> torch.Tensor:
    result = torch.rand(size=[out_features, in_features], device=device, dtype=dtype)*-1.
    return result
if " test" and __DEBUG_ME__() and False:
    def ____test_____only_for_DigitalMapping_layer__2026_to_use__reset_parameters__the_plain_rand01_style():
        import random
        if "basic behavior" and True:
            for _ in range(33):
                out_features = random.randint(3,100)
                in_features = random.randint(5,87)
                some_random_tensor = _only_for_DigitalMapping_layer__2026_to_use__reset_parameters__the_plain_rand01_style(
                    out_features = out_features, in_features=in_features)
                assert some_random_tensor.le(0.).all()
                pass#for _  
            pass#/ test

        if "dtype adaption" and True:
            
            some_random_tensor = _only_for_DigitalMapping_layer__2026_to_use__reset_parameters__the_plain_rand01_style(
                    out_features = 3, in_features=2, dtype=torch.bfloat16)
            assert some_random_tensor.dtype == torch.bfloat16
            
            some_random_tensor = _only_for_DigitalMapping_layer__2026_to_use__reset_parameters__the_plain_rand01_style(
                    out_features = 3, in_features=2, dtype=torch.float64)
            assert some_random_tensor.dtype == torch.float64
            pass
        
        if "device adaption" and True:
            
            some_random_tensor = _only_for_DigitalMapping_layer__2026_to_use__reset_parameters__the_plain_rand01_style(
                    out_features = 3, in_features=2, device='cuda')
            assert some_random_tensor.device.type == 'cuda'
            pass

        return 
    ____test_____only_for_DigitalMapping_layer__2026_to_use__reset_parameters__the_plain_rand01_style()
    pass





'''DigitalMapping_layer__2026                the layer'''
'''DigitalMapping_layer__2026                the layer'''
from collections.abc import Iterator
class DigitalMapping_layer__2026(torch.nn.Module):
    in_dim         :int
    out_dim        :int
    _init_to_nan   :bool
    _raw_weight___oCAP_iCAP     :torch.nn.ParameterList
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

        _temp__param = torch.nn.Parameter(torch.empty(size=[init_capacity__for_out, init_capacity__for_in], 
                dtype=_dtype_for_raw_weight, device=device))

        self._raw_weight___oCAP_iCAP = torch.nn.ParameterList([_temp__param])#no [0]
        assert self._raw_weight___oCAP_iCAP[0].dtype in [torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16]
        if self._init_to_nan:
            with torch.no_grad():
                self._raw_weight___oCAP_iCAP[0].fill_(torch.nan)
                pass
            pass# if self._init_to_nan:

        if isinstance(some_hyper_param, float) or isinstance(some_hyper_param, int):
            self.some_hyper_param = torch.nn.Parameter(torch.tensor(some_hyper_param, dtype=torch.float64, device=device, 
                        requires_grad = False),#, **factory_kwargs), 
                        requires_grad = False)
            pass
        elif isinstance(some_hyper_param, torch.Tensor):
            self.some_hyper_param = torch.nn.Parameter(some_hyper_param.detach().clone(), requires_grad = False)
            pass
        else:
            assert False, "unreachable"
        self.some_hyper_param.data = self.some_hyper_param.to(self._raw_weight___oCAP_iCAP[0].device)
        #if this is a higher precision, the final result may get effected. It doesn't help. So let's keep it simple.
        self.some_hyper_param.data = self.some_hyper_param.to(self._raw_weight___oCAP_iCAP[0].dtype)
        assert self.some_hyper_param.data.requires_grad == False
        assert self.some_hyper_param.shape.__len__() == 0#not important

        #<  modulized functions.
        self._random_init_algo = _only_for_DigitalMapping_layer__2026_to_use__reset_parameters__the_plain_rand01_style
        with torch.no_grad():
            self._raw_weight___oCAP_iCAP[0][:self.out_dim, :self.in_dim] = \
                    self._random_init_algo(out_features, in_features, 
                            device=device, dtype=self._raw_weight___oCAP_iCAP[0].dtype)
            pass
        self._calc_bigger_capacity__for_in = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_in
        self._calc_bigger_capacity__for_out = _only_for_DigitalMapping_layer__2026_to_use____calc_bigger_capacity__for_out
        pass

    '''parameters function.         This only gives out the raw_weight___o_i.'''
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        r"""This only gives out the raw_weight___o_i.

        Copied from pytorch code."""
        for param in self._raw_weight___oCAP_iCAP:
            yield param


    '''plain shape related.'''
    def capacity_of_in_dim(self)->int:
        '''get'''
        return self._raw_weight___oCAP_iCAP[0].shape[1] 
    def capacity_of_out_dim(self)->int:
        '''get'''
        return self._raw_weight___oCAP_iCAP[0].shape[0] 

    
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
        output___b_o = autograd_function_class_for__DigitalMapping_layer__2026.apply(input___b_i, 
                                        self.get_useful_part_of_raw_weight(), self.some_hyper_param)

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
    def get_useful_part_of_raw_weight(self)->torch.Tensor:
        result = self._raw_weight___oCAP_iCAP[0][:self.out_dim,:self.in_dim]
        return result
    def _get_useful_part_of_raw_weight_grad(self)->torch.Tensor|None:
        if self._raw_weight___oCAP_iCAP[0].grad is None:
            return None
        result = self._raw_weight___oCAP_iCAP[0].grad[:self.out_dim,:self.in_dim]
        return result
    def set_useful_part_of_raw_weight(self, input:torch.Tensor, no_grad = True)->None:
        assert input.shape == torch.Size([self.out_dim, self.in_dim])
        if no_grad:
            with torch.no_grad():
                self._raw_weight___oCAP_iCAP[0][:self.out_dim,:self.in_dim] = input
                return
            pass
        else:#with grad
            self._raw_weight___oCAP_iCAP[0][:self.out_dim,:self.in_dim] = input
            return
        #end of function.    
    def get_useful_part_of_raw_weight___and_squeeze(self, squeeze_in = False, squeeze_out = False)->torch.Tensor:
        self._squeeze(squeeze_in = squeeze_in, squeeze_out = squeeze_out)
        #result = self._raw_weight___oCAP_iCAP[0][:self.out_dim,:self.in_dim]
        return self.get_useful_part_of_raw_weight()
    
    def _squeeze(self, squeeze_in = False, squeeze_out = False):
        '''This function is designed for inner use inside this class. 

        If you need to control the timing and you know what you are doing, feel free to do anything.'''
        #<  safety
        assert squeeze_in or squeeze_out, "No real payload is asked. Why do you call this function? Or if you want to make no-op in this case, comment this line out, the code is ready for you."
        # # the no-op style.
        # if not(squeeze_in or squeeze_out):
        #     assert False, "untested"
        #     return

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
                    dtype=self._raw_weight___oCAP_iCAP[0].dtype, device=self._raw_weight___oCAP_iCAP[0].device)
        if self._init_to_nan:
            _temp_new_memory.fill_(torch.nan)
            pass
        _temp_new_memory[:self.out_dim, :self.in_dim] = self._raw_weight___oCAP_iCAP[0].data[:self.out_dim, :self.in_dim]
        with torch.no_grad():
            self._raw_weight___oCAP_iCAP[0] = torch.nn.Parameter(_temp_new_memory)
            #torch.nn.Parameter.requires_grad
            assert self._raw_weight___oCAP_iCAP[0].requires_grad == True
            pass
        return

    '''add input slot'''
    def add_input_slot__to_the_tail(self, how_many = 0, new_raw_weight_part:torch.Tensor = torch.empty(size=[0]))->None:
        '''The param combination is either (0, some tensor), or (some number, empty tensor). 
        
        If (0, empty tensor) is provided, this function is no-op.

        If new_raw_weight_part is not empty, its shape must be [out_dim, extra_in_dim]'''
        #<  wash the param.
        if how_many == 0:
            if new_raw_weight_part.nelement() == 0:
                return
                # old code    assert False, "Bad param combination. Either how_many > 0, or new_raw_weight_part is provided."
                pass# if new_raw_weight_part.nelement() == 0:
            assert new_raw_weight_part.shape.__len__() == 2
            how_many = new_raw_weight_part.shape[1]
            pass# if how_many == 0:
        else:# how_many != 0:
            #assert how_many>0#duplicated. 
            assert new_raw_weight_part.nelement() == 0, "Bad param combination. Both are provided. Remove one of them."

            new_raw_weight_part = self._random_init_algo(self.out_dim, how_many, device=self._raw_weight___oCAP_iCAP[0].device, dtype=self._raw_weight___oCAP_iCAP[0].dtype)
            pass# else of if how_many == 0:
        assert new_raw_weight_part.shape[0] == self.out_dim

        
        #<  real payload

        with torch.no_grad():
                
            _size_after = self.in_dim + how_many
            if _size_after > self.capacity_of_in_dim():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity__for_in(
                        extra_in_dim = how_many, in_dim_now = self.in_dim, out_dim_now = self.out_dim)

                _temp___new_container = torch.empty(size=[self._raw_weight___oCAP_iCAP[0].shape[0], _temp___new_capacity],
                        dtype=self._raw_weight___oCAP_iCAP[0].dtype, device=self._raw_weight___oCAP_iCAP[0].device)
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:self.out_dim, :self.in_dim] = self.get_useful_part_of_raw_weight()

                self._raw_weight___oCAP_iCAP[0] = torch.nn.Parameter(_temp___new_container)
                #torch.nn.Parameter.requires_grad
                assert self._raw_weight___oCAP_iCAP[0].requires_grad == True
                pass

            self._raw_weight___oCAP_iCAP[0].data[:self.out_dim, self.in_dim:self.in_dim + how_many] = new_raw_weight_part
            self.in_dim = _size_after
            return
            #end of function


    '''output slot'''
    def add_output_slot__to_the_tail(self, how_many = 0, new_raw_weight_part:torch.Tensor = torch.empty(size=[0]))->None:
        '''The param combination is either (0, some tensor), or (some number, empty tensor). 

        If (0, empty tensor) is provided, this function is no-op.
        
        If new_raw_weight_part is not empty, its shape must be [out_dim, extra_in_dim]'''
        #<  wash the param.
        if how_many == 0:
            if new_raw_weight_part.nelement() == 0:
                return 
                #old code    assert False, "Bad param combination. Either how_many > 0, or new_raw_weight_part is provided."
                pass# if new_raw_weight_part.nelement() == 0:
            assert new_raw_weight_part.shape.__len__() == 2
            how_many = new_raw_weight_part.shape[0]
            pass# if how_many == 0:
        else:# how_many != 0:
            #assert how_many>0#duplicated. 
            assert new_raw_weight_part.nelement() == 0, "Bad param combination. Both are provided. Remove one of them."

            new_raw_weight_part = self._random_init_algo(how_many, self.in_dim, device=self._raw_weight___oCAP_iCAP[0].device, dtype=self._raw_weight___oCAP_iCAP[0].dtype)
            pass# else of if how_many == 0:
        assert new_raw_weight_part.shape[1] == self.in_dim
        
        #<  real payload

        with torch.no_grad():
                
            _size_after = self.out_dim + how_many
            if _size_after > self.capacity_of_out_dim():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity__for_out(
                        extra_out_dim = how_many, in_dim_now = self.in_dim, out_dim_now = self.out_dim)

                _temp___new_container = torch.empty(size=[_temp___new_capacity, self._raw_weight___oCAP_iCAP[0].shape[1]],
                        dtype=self._raw_weight___oCAP_iCAP[0].dtype, device=self._raw_weight___oCAP_iCAP[0].device)
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:self.out_dim, :self.in_dim] = self.get_useful_part_of_raw_weight()

                self._raw_weight___oCAP_iCAP[0] = torch.nn.Parameter(_temp___new_container)
                #torch.nn.Parameter.requires_grad
                assert self._raw_weight___oCAP_iCAP[0].requires_grad == True
                pass

            self._raw_weight___oCAP_iCAP[0].data[self.out_dim:self.out_dim + how_many, :self.in_dim] = new_raw_weight_part
            self.out_dim = _size_after
            return
        pass# a dead pass to denote the end of function

    def keep_output_slot(self, keep_which:torch.Tensor, squeeze_the_input_dim = True)->None:
        '''This function also squeeze the memory to minimum.'''
        assert keep_which.shape.__len__() == 1
        assert keep_which.dtype == torch.bool
        #<  real payload
        with torch.no_grad():

            _temp__useful_part = self.get_useful_part_of_raw_weight()
            _temp__useful_part = _temp__useful_part[keep_which,:]
            if squeeze_the_input_dim:
                self._raw_weight___oCAP_iCAP[0] = torch.nn.Parameter(_temp__useful_part)
                #torch.nn.Parameter.requires_grad
                assert self._raw_weight___oCAP_iCAP[0].requires_grad == True

                self.out_dim = self._raw_weight___oCAP_iCAP[0].shape[0]
                return 
            else:# not to squeeze the input dim.
                assert False, "unreachable, untested, also code is old" 
                # for non-last layer, if the later layer needs more input, then this layer needs more output dimention.
                # but for last layer, in no case it needs any extra output dimention.
                _temp___keep_which___in_int = keep_which.to(torch.int32)
                how_many_to_keep = int(_temp___keep_which___in_int.sum().to(torch.int32).item())

                self._raw_weight___oCAP_iCAP[0] = torch.nn.parameter(torch.empty(size=[how_many_to_keep, self.capacity_of_in_dim()]))
                #torch.nn.Parameter.requires_grad
                assert self._raw_weight___oCAP_iCAP[0].requires_grad == True

                if self._init_to_nan:
                    self._raw_weight___oCAP_iCAP[0].data.fill_(torch.nan)
                    pass
                self._raw_weight___oCAP_iCAP[0].data[:, :self.in_dim] = _temp__useful_part
                pass
            self.out_dim = how_many_to_keep
            return 
        #end of function
    def remove_output_slot(self, remove_which:torch.Tensor, squeeze_the_input_dim = True)->None:
            self.keep_output_slot(remove_which.logical_not(), 
                    squeeze_the_input_dim = squeeze_the_input_dim)
            return

    '''stringify'''
    def extra_repr(self) -> str:
        if self._always_check_input_is_posneg1__in_forward:
            return f'Output is pos/neg 1. In_features={self.in_features}, out_features={self.out_features}'
        return f'In_features={self.in_features}, out_features={self.out_features}'
        

    # def __repr__(self):
    #     return f"{self.get_useful().__repr__()}, size:{self._size}, DNN input container 2026"
    # def __str__(self):
    #     return f"{self.get_useful().__str__()}, size:{self._size}, DNN input container 2026"

    pass# end of class.












# all the forward related                 forward
if "forward in module class      basic behavior test" and __DEBUG_ME__() and True:
    def ____test____forward_in_module_class():
        if "allow non posneg1 input?" and True:
            the_layer = DigitalMapping_layer__2026(in_features=3, out_features=2, _always_check_input_is_posneg1__in_forward= True)
            output = the_layer(torch.tensor([[1., 1, 1], [1, -1, 1]]))
            #output = the_layer(torch.tensor([[1.1, 1, 1], [1, -1, 1]]))   this must NOT work.

            the_layer = DigitalMapping_layer__2026(in_features=3, out_features=2, _always_check_input_is_posneg1__in_forward= False)
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
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)

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
if "get_max_index in module class      basic behavior test" and __DEBUG_ME__() and True:
    def ____test____get_max_index_in_module_class():
        import random
        if "allow non posneg1 input?" and True:
            for _ in range(33):
                in_dim = random.randint(2,100)
                the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features= random.randint(2,100))
                the_max_index = the_layer.get_max_index()
                assert the_max_index.lt(in_dim).all()
                pass#for _
            pass#/ test
        return
    ____test____get_max_index_in_module_class()
    pass

# all the backward related            backward
if "backward equivalence" and __DEBUG_ME__() and True:
    def ____test____backward_in_module_class()->None:
        if "allow non posneg1 input?" and True:
            the_layer = DigitalMapping_layer__2026(in_features=3, out_features=2, _always_check_input_is_posneg1__in_forward= True)
            input = torch.tensor([[1., 1, 1], [1, -1, 1]], requires_grad=True)
            output:torch.Tensor = the_layer(input)
            output.backward(gradient=torch.tensor([[1.1, 1], [1, -1]]), inputs=[input])

            the_layer = DigitalMapping_layer__2026(in_features=3, out_features=2, _always_check_input_is_posneg1__in_forward= True)
            #output = the_layer(torch.tensor([[1.1, 1, 1], [1, -1, 1]]))   this must NOT work.

            the_layer = DigitalMapping_layer__2026(in_features=3, out_features=2, _always_check_input_is_posneg1__in_forward= False)
            input = torch.tensor([[1., 1, 1], [1, -1, 1]], requires_grad=True)
            output = the_layer(input)
            output.backward(gradient=torch.tensor([[1.1, 1], [1, -1]]), inputs=[input])
            
            the_layer = DigitalMapping_layer__2026(in_features=3, out_features=2, _always_check_input_is_posneg1__in_forward= False)
            input = torch.tensor([[1.1, 1, 1], [1, -1, 1]], requires_grad=True)
            output = the_layer(input)
            output.backward(gradient=torch.tensor([[1.1, 1], [1, -1]]), inputs=[input])
            pass#/ test

        if "equivalence with the algo test function?":

            for batch in[2, 13, 37]:
                for out_dim in[3, 14, 53]:
                    for in_dim in[5, 17, 71]:

                        for _ in range(16):
                            #<  dataset
                            layer___input___b_i = rand_sign(size=[batch, in_dim])
                            layer___input___b_i.requires_grad_()
                            function___input___b_i = layer___input___b_i.detach().clone()

                            target___b_o = torch.randn(size=[batch, out_dim])

                            #<  the layer
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)

                            raw_weight___o_i = the_layer.get_useful_part_of_raw_weight().detach().clone()

                            #<  forward pass
                            layer_output:torch.Tensor = the_layer(layer___input___b_i)

                            function_output, _ = _test___DNN_forward___full_safety(input___b_i=function___input___b_i, 
                                    raw_weight___o_i=raw_weight___o_i)

                            #<  backward pass
                            layer_output.backward(gradient=target___b_o.detach().clone(), 
                                                        inputs=[layer___input___b_i, the_layer._raw_weight___oCAP_iCAP[0]])


                            function___grad_like_for___input___b_i, function___grad_like_for___raw_weight___o_i = \
                                        _algo_test__backward_function( input_posneg1___b_i = function___input___b_i, 
                                        target___b_o=target___b_o.detach().clone(), raw_weight___o_i = raw_weight___o_i, 
                                                    SOME_HYPER_PARAM___s = torch.tensor(1.))

                            #<  assert
                            assert isinstance(layer___input___b_i.grad, torch.Tensor)
                            assert isinstance(function___grad_like_for___input___b_i, torch.Tensor)
                            assert layer___input___b_i.grad.eq(function___grad_like_for___input___b_i).all()
                            assert isinstance(the_layer._raw_weight___oCAP_iCAP[0].grad, torch.Tensor)
                            assert isinstance(function___grad_like_for___raw_weight___o_i, torch.Tensor)
                            assert _tensor_shape_check(raw_weight___o_i, the_layer.out_dim, the_layer.in_dim)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad[:the_layer.out_dim, :the_layer.in_dim].eq( \
                                                function___grad_like_for___raw_weight___o_i).all()

                            pass#for _
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test

        return
    ____test____backward_in_module_class()
    pass

#<  all the shape related                 shape
if "add input slot     algo test      and class equivalence" and __DEBUG_ME__() and True:
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
                                        the_layer = DigitalMapping_layer__2026(in_features=in_dim___ori, out_features=out_dim)
                                        pass
                                    else:#debug purpose.
                                        the_layer = DigitalMapping_layer__2026(in_features=in_dim___ori, out_features=out_dim, 
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
                                    new___training_buffer___o_ii[:, in_dim___ori:in_dim_in_total] =  the_layer._raw_weight___oCAP_iCAP[0][:out_dim, in_dim___ori:in_dim_in_total]#########
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
if "add input slot with specified new raw_weight" and __DEBUG_ME__() and True:
    def ____add_input_with_specified_new_raw_weight____():
        for in_dim in [3,6,11]:
            for out_dim in [2,8,15]:
                for _ in range(6):
                    the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                    the_layer.add_input_slot__to_the_tail(new_raw_weight_part = torch.ones(size=[out_dim, 1]))
                    the_max_index___o = the_layer.get_max_index()
                    assert the_max_index___o.eq(torch.ones(size=[out_dim])*in_dim).all()

                    pass#for _
                pass#for out_dim
            pass#for in_dim

        return
    ____add_input_with_specified_new_raw_weight____()
    pass
if "add output slot     algo test      and class equivalence" and __DEBUG_ME__() and True:
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
                                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim___ori)
                                        pass
                                    else:#debug purpose.
                                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim___ori, 
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
                                            the_layer._raw_weight___oCAP_iCAP[0][out_dim___ori:out_dim_in_total, :in_dim]
                                    
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
if "add output slot with specified new raw_weight" and __DEBUG_ME__() and True:
    def ____add_output_with_specified_new_raw_weight____():
        for in_dim in [3,6,11]:
            for out_dim in [2,8,15]:
                for _ in range(6):
                    the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                    the_layer.add_output_slot__to_the_tail(new_raw_weight_part = torch.ones(size=[1, in_dim]))
                    with torch.no_grad():
                        the_layer._raw_weight___oCAP_iCAP[0][out_dim, in_dim-1] = 2.123
                        pass
                    the_max_index___o = the_layer.get_max_index()
                    assert the_max_index___o[out_dim-1+1] == in_dim-1

                    the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                    the_layer.add_output_slot__to_the_tail(new_raw_weight_part = torch.ones(size=[1, in_dim]))
                    with torch.no_grad():
                        the_layer._raw_weight___oCAP_iCAP[0][out_dim, 2] = 5.123
                        pass
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
        if "delete output.      without class" and True:

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

        if "delete output,      without class         scan it" and True:
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
        
        if "delete output,       class equivalence" and True:
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
                                    the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                                    pass
                                else:#debug purpose.
                                    the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim, 
                                                _always_check_input_is_posneg1__in_forward = False)#debug purpose.
                                    pass

                                ori___training_buffer___o_i = the_layer.get_useful_part_of_raw_weight().detach().clone()
                                assert _tensor_shape_check(ori___training_buffer___o_i, out_dim, in_dim)

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
                                assert new___output___b_o.eq(ori___output___b_o[:, keep_these_output]).all()
                                assert layer_new___output___b_o.eq(layer_ori___output___b_o[:, keep_these_output]).all()
                                #assert _tensor_equal(layer_new___output___b_o[:, :out_dim___ori], layer_ori___output___b_o)
                                #assert _tensor_equal(new___output___b_o[:, :out_dim___ori], ori___output___b_o)
                                assert the_layer.get_useful_part_of_raw_weight().eq(new___training_buffer___o_i).all()

                                pass#for _
                            pass#for is_posneg1
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test


        if "keep function and the remove function, equivalence" and True:
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
                                    the_layer_keep = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                                    the_layer_remove = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                                    pass
                                else:#debug purpose.
                                    the_layer_keep = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim, 
                                                _always_check_input_is_posneg1__in_forward = False)#debug purpose.
                                    the_layer_remove = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim,
                                                _always_check_input_is_posneg1__in_forward = False)#debug purpose.
                                    pass

                                the_layer_keep          ._raw_weight___oCAP_iCAP[0].data[:out_dim, :in_dim] = \
                                        the_layer_remove._raw_weight___oCAP_iCAP[0].data[:out_dim, :in_dim].detach().clone()

                                #<  original    forward path
                                keep_ver_ori___output___b_o   = the_layer_keep  (input___b_i)
                                remove_ver_ori___output___b_o = the_layer_remove(input___b_i)

                                assert _tensor_shape_check(keep_ver_ori___output___b_o,   batch, out_dim)
                                assert _tensor_shape_check(remove_ver_ori___output___b_o, batch, out_dim)

                                assert keep_ver_ori___output___b_o.eq(remove_ver_ori___output___b_o).all()
                                #<  the new shape
                                the_layer_keep.keep_output_slot(keep_these_output)
                                the_layer_remove.remove_output_slot(remove_which = keep_these_output.logical_not(), squeeze_the_input_dim=True)

                                assert _tensor_shape_check(the_layer_keep.get_useful_part_of_raw_weight(), new_out_dim, in_dim)
                                assert _tensor_shape_check(the_layer_remove.get_useful_part_of_raw_weight(), new_out_dim, in_dim)

                                assert the_layer_keep.get_useful_part_of_raw_weight().eq(the_layer_remove.get_useful_part_of_raw_weight()).all()

                                #<  new         forward path
                                keep_ver___new___output___b_o = the_layer_keep(input___b_i)
                                remove_ver___new___output___b_o = the_layer_remove(input___b_i)

                                assert keep_ver___new___output___b_o.eq(remove_ver___new___output___b_o).all()

                                pass#for _
                            pass#for is_posneg1
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test

        return 
    ____delete_output____()
    pass

if "basic reshape.     data member for the shape info, and padding with nan, test" and __DEBUG_ME__() and True:
    def ____test____basic_reshape____():

        if "add_input_slot__to_the_tail" and True:
            in_dim = 5
            out_dim = 33
            the_layer = DigitalMapping_layer__2026(5, 33)
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
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
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
            assert flag__is_nan.all()


            for in_dim in [5,17,33]:
                for out_dim in [7,21,37]:
                    for x in [12,27,57]:
                        for _ in range(33):
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
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
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                            pass#for _
                        pass#for x
                    pass#for out_dim
                pass#for in_dim

            for in_dim in [5,17,33]:
                for out_dim in [7,21,37]:
                    for x in [12,27,57]:
                        for _ in range(33):
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
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
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                            flag__new_added = the_layer._raw_weight___oCAP_iCAP[0][:out_dim, the_layer.in_dim:new__in_dim].lt(-5)
                            assert flag__new_added.all()

                            flag__ori = the_layer._raw_weight___oCAP_iCAP[0][:out_dim, :in_dim].gt(-2)
                            assert flag__ori.all()
                            pass#for _
                        pass#for x
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "add_output_slot__to_the_tail" and True:
            in_dim = 5
            out_dim = 7
            the_layer = DigitalMapping_layer__2026(5, 7)
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
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
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
            assert flag__is_nan.all()

            for in_dim in [5,17,33]:
                for out_dim in [7,21,37]:
                    for x in [12,27,57]:
                        for _ in range(33):
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
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
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                            
                            pass#for _
                        pass#for x
                    pass#for out_dim
                pass#for in_dim

            for in_dim in [5,17,33]:
                for out_dim in [7,21,37]:
                    for x in [12,27,57]:
                        for _ in range(33):
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                            assert the_layer.in_dim == in_dim
                            assert the_layer.out_dim == out_dim
                            assert the_layer.capacity_of_in_dim() >= in_dim
                            assert the_layer.capacity_of_out_dim() >= out_dim
                            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                            assert not flag__is_nan.any()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
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
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                            assert flag__is_nan.all()
                            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
                            assert flag__is_nan.all()
                            
                            flag__new_added = the_layer._raw_weight___oCAP_iCAP[0][out_dim:out_dim+x, :in_dim].lt(-5)
                            assert flag__new_added.all()
                            flag__ori = the_layer._raw_weight___oCAP_iCAP[0][:out_dim, :in_dim].gt(-2)
                            assert flag__ori.all()
                            pass#for _
                        pass#for x
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        '''deprecated test.
        According to the design, if a layer needs to remove some of the output slot, it's the last layer, 
        then it doesn't needs to keep any capacity for input slots.'''
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
            the_layer = DigitalMapping_layer__2026(in_dim, out_dim, 
                    init_capacity__for_out = 9, init_capacity__for_in = 6)
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
            assert flag__is_nan.all()

            #<  manually 
            max_index = the_layer.get_max_index()
            keep_which = torch.tensor([1, 0, 1])#1,1,0,0,1,1,0])
            new__out_dim = keep_which.sum()
            keep_which = keep_which.to(torch.bool)
            #prin(the_layer._raw_weight___oCAP_iCAP[0].tolist())
            the_layer.keep_output_slot(keep_which, squeeze_the_input_dim=False)#calc
            #prin(the_layer._raw_weight___oCAP_iCAP[0].tolist())
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == new__out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() == new__out_dim#no useless output dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
            assert flag__is_nan.all()
            assert the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :].nelement() == 0
            # flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
            # assert flag__is_nan.all()

            manual__max_index = max_index[keep_which]
            new__max_index = the_layer.get_max_index()
            assert _tensor_equal(manual__max_index, new__max_index)

            #  re random useless numbers. If anything relies on this part, the assertion will probably fail.
            # the_layer._raw_weight___oCAP_iCAP[0].data[:, :] = \          how to fail the assertion.
            #         torch.randn_like(the_layer._raw_weight___oCAP_iCAP[0].data[:, :])*123.    how to fail the assertion.
            the_layer._raw_weight___oCAP_iCAP[0].data[:, in_dim:] = \
                    torch.randn_like(the_layer._raw_weight___oCAP_iCAP[0].data[:, in_dim:])*123.
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0].data)
            assert not flag__is_nan.any()
            new__max_index_2 = the_layer.get_max_index()
            assert _tensor_equal(manual__max_index, new__max_index_2)


            for in_dim in [5,17,53]:
                for out_dim in [7,21,67]:
                    for _ in range(15):

                        the_layer = DigitalMapping_layer__2026(in_dim, out_dim)
                        assert the_layer.in_dim == in_dim
                        assert the_layer.out_dim == out_dim
                        assert the_layer.capacity_of_in_dim() >= in_dim
                        assert the_layer.capacity_of_out_dim() >= out_dim
                        flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                        assert not flag__is_nan.any()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                        assert flag__is_nan.all()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
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
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                        assert flag__is_nan.all()
                        assert the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :].nelement() == 0
            
                        manual__max_index = max_index[keep_which]
                        new__max_index = the_layer.get_max_index()
                        assert _tensor_equal(manual__max_index, new__max_index)

                        #  re random useless numbers. If anything relies on this part, the assertion will probably fail.
                        the_layer._raw_weight___oCAP_iCAP[0].data[:, in_dim:] = \
                                torch.randn_like(the_layer._raw_weight___oCAP_iCAP[0].data[:, in_dim:])*123.
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0].data)
                        assert not flag__is_nan.any()
                        new__max_index_2 = the_layer.get_max_index()
                        assert _tensor_equal(manual__max_index, new__max_index_2)
            
                        pass#for _
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "keep_output_slot      with squeeze on input" and True:
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
            the_layer = DigitalMapping_layer__2026(in_dim, out_dim, 
                    init_capacity__for_out = 9, init_capacity__for_in = 6)
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == out_dim
            assert the_layer.capacity_of_in_dim() >= in_dim
            assert the_layer.capacity_of_out_dim() >= out_dim
            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
            assert flag__is_nan.all()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
            assert flag__is_nan.all()

            #<  manually 
            max_index = the_layer.get_max_index()
            keep_which = torch.tensor([1, 0, 1])#1,1,0,0,1,1,0])
            new__out_dim = keep_which.sum()
            keep_which = keep_which.to(torch.bool)
            #prin(the_layer._raw_weight___oCAP_iCAP[0].tolist())
            the_layer.keep_output_slot(keep_which, squeeze_the_input_dim=True)#calc
            #prin(the_layer._raw_weight___oCAP_iCAP[0].tolist())
            assert the_layer.in_dim == in_dim
            assert the_layer.out_dim == new__out_dim
            assert the_layer.capacity_of_in_dim() == in_dim
            assert the_layer.capacity_of_out_dim() == new__out_dim#no useless output dim
            assert the_layer._raw_weight___oCAP_iCAP[0].shape == the_layer.get_useful_part_of_raw_weight().shape

            flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
            assert not flag__is_nan.any()
            flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0])
            assert not flag__is_nan.any()

            manual__max_index = max_index[keep_which]
            new__max_index = the_layer.get_max_index()
            assert _tensor_equal(manual__max_index, new__max_index)

            for in_dim in [5,17,53]:
                for out_dim in [7,21,67]:
                    for _ in range(15):

                        the_layer = DigitalMapping_layer__2026(in_dim, out_dim)
                        assert the_layer.in_dim == in_dim
                        assert the_layer.out_dim == out_dim
                        assert the_layer.capacity_of_in_dim() >= in_dim
                        assert the_layer.capacity_of_out_dim() >= out_dim
                        flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                        assert not flag__is_nan.any()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][:, the_layer.in_dim:])
                        assert flag__is_nan.all()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0][the_layer.out_dim:, :])
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
                        assert the_layer._raw_weight___oCAP_iCAP[0].shape == the_layer.get_useful_part_of_raw_weight().shape
            
                        flag__is_nan = torch.isnan(the_layer.get_useful_part_of_raw_weight())
                        assert not flag__is_nan.any()
                        flag__is_nan = torch.isnan(the_layer._raw_weight___oCAP_iCAP[0])
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

if "squeeze" and __DEBUG_ME__() and True:
    def ____test____squeeze():
        import random 
        if "the squeeze funciont" and True:
            for in_dim in [3,6,11, 33, ]:
                for out_dim in [2,8,15, 57]:
                    for _ in range(6):
                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, init_capacity__for_in=random.randint(in_dim+3, in_dim+111),
                                                            out_features=out_dim, init_capacity__for_out=random.randint(out_dim+3, out_dim+111))
                        assert the_layer.capacity_of_in_dim()  != the_layer.in_dim
                        assert the_layer.capacity_of_out_dim() != the_layer.out_dim
                        the_layer._squeeze(squeeze_in=True)
                        assert the_layer.capacity_of_in_dim()  == the_layer.in_dim
                        assert the_layer.capacity_of_out_dim() != the_layer.out_dim

                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, init_capacity__for_in=random.randint(in_dim+3, in_dim+111),
                                                            out_features=out_dim, init_capacity__for_out=random.randint(out_dim+3, out_dim+111))
                        assert the_layer.capacity_of_in_dim()  != the_layer.in_dim
                        assert the_layer.capacity_of_out_dim() != the_layer.out_dim
                        the_layer._squeeze(squeeze_out=True)
                        assert the_layer.capacity_of_in_dim()  != the_layer.in_dim
                        assert the_layer.capacity_of_out_dim() == the_layer.out_dim
                        
                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, init_capacity__for_in=random.randint(in_dim+3, in_dim+111),
                                                            out_features=out_dim, init_capacity__for_out=random.randint(out_dim+3, out_dim+111))
                        assert the_layer.capacity_of_in_dim()  != the_layer.in_dim
                        assert the_layer.capacity_of_out_dim() != the_layer.out_dim
                        the_layer._squeeze(squeeze_in=True, squeeze_out=True)
                        assert the_layer.capacity_of_in_dim()  == the_layer.in_dim
                        assert the_layer.capacity_of_out_dim() == the_layer.out_dim

                        pass#for _
                    pass#for out_dim
                pass#for in_dim
            pass#/ test
        return
    ____test____squeeze()
    pass
#</  all the shape related                 shape

#<  backward after reshape
if "any reshape with a new memory chunk, and then backward" and __DEBUG_ME__() and True:
    def ____test____reshape_and_then_backward():
        if "a working reference,       not the test" and True:
            for batch in [2,5,10]:
                for in_dim in [6,9,13]:
                    for out_dim in [3,7,11]:
                        if in_dim<=out_dim:
                            continue
                        for _ in range(6):
                            #<  dataset
                            input___b_i = rand_sign(size=[batch, in_dim])
                            label___b_o = rand_sign(size=[batch, out_dim])
                            #<  infra
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)

                            output___b_o:torch.Tensor = the_layer(input___b_i)
                            output___b_o.backward(gradient=label___b_o, inputs=the_layer._raw_weight___oCAP_iCAP)
                            pass
                            pass#for _
                        pass#for out_dim
                    pass#for in_dim
                pass#for batch
            pass#/ test

        # a = []   code test
        # b = a
        # assert a is b

        if "add_input_slot__to_the_tail            and then backward" and False:
            for batch in [2,5,10]:
                for in_dim in [6,9,13]:
                    for out_dim in [3,7,11]:
                        if in_dim<=out_dim:
                            continue
                        for extra___in_dim in [5,15,33,166,333]:
                            for _ in range(6):
                        
                                #<  dataset
                                label___b_o = rand_sign(size=[batch, out_dim])
                                #<  infra
                                the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                                #the_list = the_layer._raw_weight___oCAP_iCAP[0]
                                #ori_CAP_shape = the_layer._raw_weight___oCAP_iCAP[0].shape
                                #assert ori_CAP_shape == torch.Size([16, 16])

                                input___b_i = rand_sign(size=[batch, in_dim])
                                output___b_o:torch.Tensor = the_layer(input___b_i)
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                output___b_o.backward(gradient=label___b_o, inputs=the_layer._raw_weight___oCAP_iCAP)
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                                del output___b_o    
                                the_layer.zero_grad()
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                #assert the_layer._raw_weight___oCAP_iCAP[0] is the_list
                                #assert the_layer._raw_weight___oCAP_iCAP[0].shape == ori_CAP_shape



                                #assert the_layer._raw_weight___oCAP_iCAP[0] is buffer
                                the_layer.add_input_slot__to_the_tail(how_many=extra___in_dim)

                                #assert the_layer._raw_weight___oCAP_iCAP[0] is not buffer
                                #assert the_layer._raw_weight___oCAP_iCAP[0].shape == torch.Size([16, 68])
                                #assert the_layer._raw_weight___oCAP_iCAP[0].shape != torch.Size([16, 16])
                                
                                input___b_i = rand_sign(size=[batch, in_dim + extra___in_dim])
                                output___b_o:torch.Tensor = the_layer(input___b_i)
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                output___b_o.backward(gradient=label___b_o, inputs=[the_layer._raw_weight___oCAP_iCAP[0]])
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                                del output___b_o    
                                the_layer.zero_grad()
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                #assert the_layer._raw_weight___oCAP_iCAP[0] is not buffer
                                #assert the_layer._raw_weight___oCAP_iCAP[0].shape != ori_CAP_shape

                                pass#for _
                            pass#for extra___in_dim
                        pass#for out_dim
                    pass#for in_dim
                pass#for batch
            pass#/ test

        if "add_output_slot__to_the_tail            and then backward" and False:
            for batch in [2,5,10]:
                for in_dim in [6,9,13]:
                    for out_dim in [3,7,11]:
                        if in_dim<=out_dim:
                            continue
                        for extra___out_dim in [5,15,33,166,333]:
                            for _ in range(6):
                        
                                #<  dataset
                                input___b_i = rand_sign(size=[batch, in_dim])
                                #<  infra
                                the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)

                                output___b_o:torch.Tensor = the_layer(input___b_i)
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                label___b_o = rand_sign(size=[batch, out_dim])
                                output___b_o.backward(gradient=label___b_o, inputs=the_layer._raw_weight___oCAP_iCAP)
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                                del output___b_o    
                                the_layer.zero_grad()
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is None


                                the_layer.add_output_slot__to_the_tail(how_many=extra___out_dim)

                                the_layer.zero_grad()
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                output___b_o:torch.Tensor = the_layer(input___b_i)
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                label___b_o = rand_sign(size=[batch, out_dim + extra___out_dim])
                                output___b_o.backward(gradient=label___b_o, inputs=[the_layer._raw_weight___oCAP_iCAP[0]])
                                assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None

                                pass#for _
                            pass#for extra___in_dim
                        pass#for out_dim
                    pass#for in_dim
                pass#for batch
            pass#/ test

        if "keep_output_slot              and then backward" and False:
            for batch in [2,5,10]:
                for in_dim in [6,9,13]:
                    for out_dim in [3,7,11]:
                        if in_dim<=out_dim:
                            continue
                        for _ in range(6):
                    
                            #<  dataset
                            input___b_i = rand_sign(size=[batch, in_dim])
                            keep_which = torch.rand(size=[out_dim]).gt(0.5)
                            assert keep_which.dtype == torch.bool
                            new___out_dim = int(keep_which.to(torch.int32).sum().item())
                            #<  infra
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)

                            output___b_o:torch.Tensor = the_layer(input___b_i)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            label___b_o = rand_sign(size=[batch, out_dim])
                            output___b_o.backward(gradient=label___b_o, inputs=the_layer._raw_weight___oCAP_iCAP)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                            del output___b_o    
                            the_layer.zero_grad()
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None


                            the_layer.keep_output_slot(keep_which=keep_which)

                            the_layer.zero_grad()
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            output___b_o:torch.Tensor = the_layer(input___b_i)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            label___b_o = rand_sign(size=[batch, new___out_dim])
                            output___b_o.backward(gradient=label___b_o, inputs=[the_layer._raw_weight___oCAP_iCAP[0]])
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                            pass#for _
                        pass#for out_dim
                    pass#for in_dim
                pass#for batch
            pass#/ test

        if "remove_output_slot            and then backward" and False:
            for batch in [2,5,10]:
                for in_dim in [6,9,13]:
                    for out_dim in [3,7,11]:
                        if in_dim<=out_dim:
                            continue
                        for _ in range(6):
                    
                            #<  dataset
                            input___b_i = rand_sign(size=[batch, in_dim])
                            remove_which = torch.rand(size=[out_dim]).gt(0.5)
                            assert remove_which.dtype == torch.bool
                            new___out_dim = out_dim - int(remove_which.to(torch.int32).sum().item())
                            #<  infra
                            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)

                            output___b_o:torch.Tensor = the_layer(input___b_i)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            label___b_o = rand_sign(size=[batch, out_dim])
                            output___b_o.backward(gradient=label___b_o, inputs=the_layer._raw_weight___oCAP_iCAP)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                            del output___b_o    
                            the_layer.zero_grad()
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None


                            the_layer.remove_output_slot(remove_which=remove_which)

                            the_layer.zero_grad()
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            output___b_o:torch.Tensor = the_layer(input___b_i)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            label___b_o = rand_sign(size=[batch, new___out_dim])
                            output___b_o.backward(gradient=label___b_o, inputs=[the_layer._raw_weight___oCAP_iCAP[0]])
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                            pass#for _
                        pass#for out_dim
                    pass#for in_dim
                pass#for batch
            pass#/ test

        if "squeeze            and then backward" and False:
            import random
            for batch in [2,5,10]:
                for in_dim in [6,9,13]:
                    for out_dim in [3,7,11]:
                        if in_dim<=out_dim:
                            continue

                        for squeeze_in in [True, False]:
                            for squeeze_out in [True, False]:
                                if (not squeeze_in) and (not squeeze_in):
                                    continue

                                for _ in range(6):
                            
                                    #<  dataset
                                    input___b_i = rand_sign(size=[batch, in_dim])
                                    #<  infra
                                    the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim, 
                                                init_capacity__for_in  = random.randint(in_dim,  in_dim  + 100), 
                                                init_capacity__for_out = random.randint(out_dim, out_dim + 100), )

                                    output___b_o:torch.Tensor = the_layer(input___b_i)
                                    assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                    label___b_o = rand_sign(size=[batch, out_dim])
                                    output___b_o.backward(gradient=label___b_o, inputs=the_layer._raw_weight___oCAP_iCAP)
                                    assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                                    del output___b_o    
                                    the_layer.zero_grad()
                                    assert the_layer._raw_weight___oCAP_iCAP[0].grad is None

                                    the_layer._squeeze(squeeze_in=squeeze_in, squeeze_out=squeeze_out)

                                    the_layer.zero_grad()
                                    assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                    output___b_o:torch.Tensor = the_layer(input___b_i)
                                    assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                                    label___b_o = rand_sign(size=[batch, out_dim])
                                    output___b_o.backward(gradient=label___b_o, inputs=[the_layer._raw_weight___oCAP_iCAP[0]])
                                    assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                                    pass#for _
                                pass#for squeeze_out
                            pass#for squeeze_in
                        pass#for out_dim
                    pass#for in_dim
                pass#for batch
            pass#/ test

        return
    ____test____reshape_and_then_backward()
    pass













'''the optimizer'''
'''the optimizer'''
'''the optimizer'''
'''如果要用这个类，那么每一次都要新建一个optim object，用，用了丢弃。和torch的传统，一个optim一直用，会不一样。'''
def only_for_DigitalMapping_layer__2026_to_use___optim_step(raw_weight___o_i:torch.Tensor, grad_like_for_raw_weight___o_i:torch.Tensor, 
            learning_rate___s:torch.Tensor|float, safety_check = True, epsilon = torch.tensor(0.01))->torch.Tensor:
    
# pseudo_raw_weight = torch.tanh(pseudo_raw_weight___before_protection)

    if safety_check:#这两个搬到外面去
        assert raw_weight___o_i.le(0.).all()#bc of the design. No other reason.
        assert learning_rate___s > 0.
        assert epsilon > 0.
        pass
    #<  real payload
    with torch.no_grad():
        _temp___max___o:torch.Tensor = grad_like_for_raw_weight___o_i.max(dim=1).values
        _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
        inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
        del _temp___max___o, _temp___max___o_EXPANDi
        if safety_check:
            assert inner___grad_like_for_raw_weight___o_i.le(0.).all()#################
            pass

        _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
        _temp___mean_of_abs___o = _temp___mean_of_abs___o.max(epsilon)
        #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
        if safety_check:
            assert _temp___mean_of_abs___o.ge(epsilon).all()
            pass

        _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
        inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
        del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi

        new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * learning_rate___s
        if safety_check:
            assert new___raw_weight___before_tanh___o_i.le(0.).all()
            pass

        new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
        if safety_check:
            assert new___raw_weight___o_i.ge(-1.).all()##############
            assert new___raw_weight___o_i.le(-0.).all()##############
            pass
        return new___raw_weight___o_i
    #end of function.

if "optim step algo test" and __DEBUG_ME__() and True:
    def ____test____only_for_DigitalMapping_layer__2026_to_use___optim_step()->None:

        if "basic algo test" and False:
            out_dim = 2
            in_dim = 3
            #<  data
            raw_weight___o_i = torch.tensor([   [-10., -11, 0], 
                                                [-100., -11, 0]]) 
            #raw_weight___o_i = torch.rand(size=(out_dim, in_dim))*-1.
            assert raw_weight___o_i.le(0.).all()#bc of the design. No other reason.
            learning_rate___s = 1.1
            grad_like_for_raw_weight___o_i = torch.tensor([ [1231.,  1232, 1233], 
                                                            [3211.,  3213, 3215], ])

            #<  real payload
            _temp___max___o:torch.Tensor  = grad_like_for_raw_weight___o_i.max(dim=1).values
            assert _tensor_shape_check(_temp___max___o, out_dim)
            assert _tensor_equal(_temp___max___o, torch.tensor([1233, 3215]))
            _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            assert _tensor_shape_check(_temp___max___o_EXPANDi, out_dim, in_dim)
            inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
            del _temp___max___o, _temp___max___o_EXPANDi
            assert _tensor_equal(inner___grad_like_for_raw_weight___o_i, [  [-2., -1, 0], 
                                                                            [-4., -2, 0], ])

            _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
            #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
            assert _tensor_shape_check(_temp___mean_of_abs___o, out_dim)
            assert _tensor_equal(_temp___mean_of_abs___o, [1., 2])

            _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            assert _tensor_shape_check(_temp___temp___mean_of_abs___o_EXPANDi, out_dim, in_dim)
            inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
            del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi
            assert _tensor_equal(inner___grad_like_for_raw_weight___o_i, torch.tensor([ [-2., -1, 0], 
                                                                                        [-2., -1, 0], ]))

            new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * learning_rate___s

            assert _tensor_equal(new___raw_weight___before_tanh___o_i, torch.tensor([   [-12.2,  -12.1, 0], 
                                                                                        [-102.2, -12.1, 0]]))
            assert new___raw_weight___before_tanh___o_i.le(-0.).all()##############
            new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
            assert new___raw_weight___o_i.ge(-1.).all()##############
            assert new___raw_weight___o_i.le(-0.).all()##############
            #return new___raw_weight___o_i
            pass#/ test

        if "scan" and True:
            for in_dim in [3,6,11, 33, ]:
                for out_dim in [2,8,15, 57]:
                    for _ in range(6):
                        #<  data
                        raw_weight___o_i = torch.rand(size=(out_dim, in_dim))*-1.
                        assert raw_weight___o_i.le(0.).all()#bc of the design. No other reason.
                        learning_rate___s = 1.1
                        grad_like_for_raw_weight___o_i = torch.randn(size=[out_dim, in_dim])

                        #<  real payload
                        #_temp___max___o:torch.Tensor  = grad_like_for_raw_weight___o_i.max(dim=1).values
                        _temp___max___o                = grad_like_for_raw_weight___o_i.max(dim=1).values
                        assert _tensor_shape_check(_temp___max___o, out_dim)
                        _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
                        assert _tensor_shape_check(_temp___max___o_EXPANDi, out_dim, in_dim)
                        inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
                        del _temp___max___o, _temp___max___o_EXPANDi
                        assert inner___grad_like_for_raw_weight___o_i.le(0.).all()#################

                        _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
                        #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
                        assert _tensor_shape_check(_temp___mean_of_abs___o, out_dim)
                        assert _temp___mean_of_abs___o.ge(0.).all()

                        _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
                        assert _tensor_shape_check(_temp___temp___mean_of_abs___o_EXPANDi, out_dim, in_dim)
                        inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
                        del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi

                        new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * learning_rate___s
                        assert new___raw_weight___before_tanh___o_i.le(0.).all()

                        new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
                        assert new___raw_weight___o_i.ge(-1.).all()##############
                        assert new___raw_weight___o_i.le(-0.).all()##############
                        #return new___raw_weight___o_i

                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "function_version equivalence" and True:
            for in_dim in [3,6,11, 33, ]:
                for out_dim in [2,8,15, 57]:
                    for _ in range(6):
                        #<  data
                        raw_weight___o_i = torch.rand(size=(out_dim, in_dim))*-1.
                        learning_rate___s = 1.1
                        grad_like_for_raw_weight___o_i = torch.randn(size=[out_dim, in_dim])

                        #<  real payload
                        #_temp___max___o:torch.Tensor  = grad_like_for_raw_weight___o_i.max(dim=1).values
                        _temp___max___o                = grad_like_for_raw_weight___o_i.max(dim=1).values
                        _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
                        inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i.detach().clone()-_temp___max___o_EXPANDi
                        del _temp___max___o, _temp___max___o_EXPANDi

                        _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
                        #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.

                        _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
                        inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
                        del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi

                        new___raw_weight___before_tanh___o_i = raw_weight___o_i.detach().clone() + inner___grad_like_for_raw_weight___o_i * learning_rate___s

                        new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()

                        #<  function version
                        function_return_value = only_for_DigitalMapping_layer__2026_to_use___optim_step( \
                                    raw_weight___o_i = raw_weight___o_i.detach().clone(), 
                                    grad_like_for_raw_weight___o_i = grad_like_for_raw_weight___o_i.detach().clone(), 
                                    learning_rate___s = learning_rate___s)
                        #<  assert
                        assert new___raw_weight___o_i.eq(function_return_value).all()
                        pass#for _
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        return
    ____test____only_for_DigitalMapping_layer__2026_to_use___optim_step()
    pass

class 并入model的class了____optim_for___DigitalMapping_layer__2026(torch.nn.Module):#torch.optim.Optimizer):
    '''I need the useful shape information.
    The torch.optim.Optimizer only accepts torch.Tensor. It needs a lot hack
    to get this DigitalMapping_layer__2026 to work along with it.
    The reason to choose torch.nn.Module is I want the convenience 
    when I move it between devices and save/load it.'''

    learning_rate___s:torch.nn.Parameter
    digitalmapping_layers:torch.nn.ParameterList
    epsilon:torch.nn.Parameter
    def __init__(self, DigitalMapping_layers:list[DigitalMapping_layer__2026], 
                    learning_rate___s=0.01, epsilon = 0.01, device = None, dtype = None):
        super().__init__()
        #<  safety
        assert epsilon > 0., "Bad param"
        assert learning_rate___s> 0., "Bad param"
        for DigitalMapping_layer in DigitalMapping_layers:
            assert isinstance(DigitalMapping_layer, DigitalMapping_layer__2026), \
                        "this is different from the pytorch optim. It must be list[DigitalMapping_layer__2026]."
            pass

        #<  real payload
        self.digitalmapping_layers = torch.nn.ParameterList(DigitalMapping_layers)
        #learning_rate___s
        if isinstance(learning_rate___s, float):
            self.learning_rate___s = torch.nn.Parameter(torch.tensor(learning_rate___s, device=device, dtype=dtype), 
                    requires_grad = False)
            pass
        else:#torch.tensor
            learning_rate___s = learning_rate___s.to(dtype).to(device)
            self.learning_rate___s = torch.nn.Parameter(torch.tensor(learning_rate___s), 
                    requires_grad = False)
            pass
        assert self.learning_rate___s.requires_grad == False
        #epsilon
        if isinstance(epsilon, float):
            self.epsilon = torch.nn.Parameter(torch.tensor(epsilon, device=device, dtype=dtype), 
                    requires_grad = False)
            pass
        else:#torch.tensor
            epsilon = epsilon.to(dtype).to(device)
            self.epsilon = torch.nn.Parameter(torch.tensor(epsilon), 
                    requires_grad = False)
            pass
        assert self.epsilon.requires_grad == False
        return

    def forward(self):
        assert False, "This tool is designed as an optimizer, not a layer. Please search torch.optim.Optimizer for reference."
    def parameters(self, recurse = True):
        assert False, "This tool is designed as an optimizer, not a layer. Please search torch.optim.Optimizer for reference."
        return super().parameters(recurse) #unreachable!!!!!!!
    def zero_grad(self, set_to_none: bool = True) -> None:
        for digitalmapping_layer in self.digitalmapping_layers:
            assert isinstance(digitalmapping_layer, DigitalMapping_layer__2026)
            digitalmapping_layer._raw_weight___oCAP_iCAP[0].grad = None
            pass

    @torch.no_grad() # Important: disable gradient tracking within the optimizer step
    def step(self, safety_check = False)->None:#, closure=None):
        #https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-optimizers
        '''Bc I don't use this closure style, and I have no idea how it works.
        Just in case, let me turn it off. 
        Fyi, https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-optimizers'''

        for digitalmapping_layer in self.digitalmapping_layers:
            assert isinstance(digitalmapping_layer, DigitalMapping_layer__2026)

            if digitalmapping_layer._raw_weight___oCAP_iCAP[0].grad is None:
                continue # Skip parameters without gradients

            grad_like_for_raw_weight___o_i = digitalmapping_layer._get_useful_part_of_raw_weight_grad()
            if grad_like_for_raw_weight___o_i is None:
                continue
            
            # old code      new_data_for_parameter = only_for_DigitalMapping_layer__2026_to_use___optim_step( \
            #                     raw_weight___o_i = digitalmapping_layer.get_useful_part_of_raw_weight(),  
            #                     grad_like_for_raw_weight___o_i = grad_like, 
            #                     learning_rate___s = self.learning_rate___s, 
            #                     epsilon=self.epsilon)#展开
            #<  展开
            raw_weight___o_i:torch.Tensor = digitalmapping_layer.get_useful_part_of_raw_weight()

            #<  real payload
            _temp___max___o:torch.Tensor = grad_like_for_raw_weight___o_i.max(dim=1).values
            _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
            del _temp___max___o, _temp___max___o_EXPANDi
            if safety_check:
                assert inner___grad_like_for_raw_weight___o_i.le(0.).all()#################
                pass

            _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
            _temp___mean_of_abs___o = _temp___mean_of_abs___o.max(self.epsilon)
            #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
            if safety_check:
                assert _temp___mean_of_abs___o.ge(self.epsilon).all()
                pass

            _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
            del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi

            new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * self.learning_rate___s
            if safety_check:
                assert new___raw_weight___before_tanh___o_i.le(0.).all()
                pass
            new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
            #</  展开


            digitalmapping_layer.set_useful_part_of_raw_weight(new___raw_weight___o_i)
        return

if "basic behavior" and __DEBUG_ME__() and False:
    def ____test____optim_for___DigitalMapping_layer__2026()->None:
        if "zero grad function.      scan" and False:
            for batch in [2,5,10]:
                for out_dim in [3,7,11]:
                    for in_dim in [6,9,13]:
                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                        assert the_layer._raw_weight___oCAP_iCAP[0].requires_grad == True
                        assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                        assert the_layer.some_hyper_param.requires_grad == False
                        assert the_layer.some_hyper_param.grad is None
                        
                        the_optim = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=[the_layer], learning_rate___s=0.1)
                        #the_optim = optim_for___DigitalMapping_layer__2026(params=the_layer.parameters(), learning_rate___s=0.1)
                        the_optim.zero_grad()
                        assert the_layer._raw_weight___oCAP_iCAP[0].grad is None


                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                        assert the_layer._raw_weight___oCAP_iCAP[0].requires_grad == True
                        assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                        assert the_layer.some_hyper_param.requires_grad == False
                        assert the_layer.some_hyper_param.grad is None

                        the_optim = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=[the_layer], learning_rate___s=10.)

                        input___b_i = rand_sign(size=[batch, in_dim])
                        input___b_i.requires_grad_()
                        output___b_o:torch.Tensor = the_layer(input___b_i)
                        _temp_inputs = [input___b_i]
                        _temp_inputs.append(the_layer._raw_weight___oCAP_iCAP[0])
                        output___b_o.backward(gradient=torch.randn_like(output___b_o), inputs = _temp_inputs)
                        del _temp_inputs
                        assert input___b_i.grad is not None

                        assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                        assert the_layer.some_hyper_param.grad is None

                        the_optim.zero_grad()
                        assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test

        if "class equivalence         no scan" and False:
            out_dim = 2
            in_dim = 3

            the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
            learning_rate___s = 1.1
            the_optim = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=[the_layer], 
                                                        learning_rate___s=learning_rate___s)


            #<  data
            the_layer._raw_weight___oCAP_iCAP[0].grad = torch.empty_like(the_layer._raw_weight___oCAP_iCAP[0])
            the_layer._raw_weight___oCAP_iCAP[0].grad.fill_(torch.nan)
            raw_weight___o_i = torch.tensor([   [-10., -11, 0], 
                                                [-100., -11, 0]]) 
            #raw_weight___o_i = torch.rand(size=(out_dim, in_dim))*-1.
            assert raw_weight___o_i.le(0.).all()#bc of the design. No other reason.
            assert _tensor_shape_check(raw_weight___o_i, out_dim, in_dim)
            the_layer.set_useful_part_of_raw_weight(raw_weight___o_i.detach().clone())
            grad_like_for_raw_weight___o_i = torch.tensor([ [1231.,  1232, 1233], 
                                                            [3211.,  3213, 3215], ])
            assert _tensor_shape_check(grad_like_for_raw_weight___o_i, out_dim, in_dim)
            the_layer._raw_weight___oCAP_iCAP[0].grad[:out_dim, :in_dim] = grad_like_for_raw_weight___o_i.detach().clone()

            #<  real payload
            _temp___max___o:torch.Tensor  = grad_like_for_raw_weight___o_i.max(dim=1).values
            assert _tensor_shape_check(_temp___max___o, out_dim)
            assert _tensor_equal(_temp___max___o, torch.tensor([1233, 3215]))
            _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            assert _tensor_shape_check(_temp___max___o_EXPANDi, out_dim, in_dim)
            inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
            del _temp___max___o, _temp___max___o_EXPANDi
            assert _tensor_equal(inner___grad_like_for_raw_weight___o_i, [  [-2., -1, 0], 
                                                                            [-4., -2, 0], ])

            _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
            #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
            assert _tensor_shape_check(_temp___mean_of_abs___o, out_dim)
            assert _tensor_equal(_temp___mean_of_abs___o, [1., 2])

            _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            assert _tensor_shape_check(_temp___temp___mean_of_abs___o_EXPANDi, out_dim, in_dim)
            inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
            del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi
            assert _tensor_equal(inner___grad_like_for_raw_weight___o_i, torch.tensor([ [-2., -1, 0], 
                                                                                        [-2., -1, 0], ]))

            new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * learning_rate___s

            assert _tensor_equal(new___raw_weight___before_tanh___o_i, torch.tensor([   [-12.2,  -12.1, 0], 
                                                                                        [-102.2, -12.1, 0]]))
            assert new___raw_weight___before_tanh___o_i.le(-0.).all()##############
            new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
            assert new___raw_weight___o_i.ge(-1.).all()##############
            assert new___raw_weight___o_i.le(-0.).all()##############

            #<  step the layer???
            the_optim.step()
            assert the_layer.get_useful_part_of_raw_weight().eq(new___raw_weight___o_i).all()

            pass#/ test

        if "class equivalence         scan" and True:
            for out_dim in [3,7,11]:
                for in_dim in [6,9,13]:
                    for _ in range(11):
                        #<  neural net infra
                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                        learning_rate___s = 1.1
                        the_optim = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=[the_layer], 
                                                                    learning_rate___s=learning_rate___s)

                        #<  data
                        the_layer._raw_weight___oCAP_iCAP[0].grad = torch.empty_like(the_layer._raw_weight___oCAP_iCAP[0])
                        the_layer._raw_weight___oCAP_iCAP[0].grad.fill_(torch.nan)
                        raw_weight___o_i = torch.rand(size=[out_dim, in_dim ])*-1.
                        assert raw_weight___o_i.le(0.).all()#bc of the design. No other reason.
                        assert _tensor_shape_check(raw_weight___o_i, out_dim, in_dim)
                        the_layer.set_useful_part_of_raw_weight(raw_weight___o_i.detach().clone())
                        grad_like_for_raw_weight___o_i = torch.randn(size=[out_dim, in_dim ])
                        assert _tensor_shape_check(grad_like_for_raw_weight___o_i, out_dim, in_dim)
                        the_layer._raw_weight___oCAP_iCAP[0].grad[:out_dim, :in_dim] = grad_like_for_raw_weight___o_i.detach().clone()

                        #<  real payload     function version.
                        _temp___max___o:torch.Tensor  = grad_like_for_raw_weight___o_i.max(dim=1).values
                        assert _tensor_shape_check(_temp___max___o, out_dim)
                        _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
                        assert _tensor_shape_check(_temp___max___o_EXPANDi, out_dim, in_dim)
                        inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
                        del _temp___max___o, _temp___max___o_EXPANDi

                        _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
                        #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
                        assert _tensor_shape_check(_temp___mean_of_abs___o, out_dim)

                        _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
                        assert _tensor_shape_check(_temp___temp___mean_of_abs___o_EXPANDi, out_dim, in_dim)
                        inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
                        del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi

                        new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * learning_rate___s

                        assert new___raw_weight___before_tanh___o_i.le(-0.).all()##############
                        new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
                        assert new___raw_weight___o_i.ge(-1.).all()##############
                        assert new___raw_weight___o_i.le(-0.).all()##############

                        #<  layer version.
                        the_optim.step()
                        assert the_layer.get_useful_part_of_raw_weight().eq(new___raw_weight___o_i).all()
                        pass#for _ 
                    pass#for out_dim
                pass#for batch
            pass#/ test

        return
    ____test____optim_for___DigitalMapping_layer__2026()
    pass



if "integrated test" and __DEBUG_ME__() and False:
    def ____test____integrated_test()->None:
        '''modified from the backward algo test.'''

        if "prototype.    scan" and True:
            if "result" and False:
                # random rate 0.0
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.502,  0.504,  0.508,  0.519,  0.556,  0.654,  0.979,  1.000]
                # acc gain         = [ 0.001,  0.004,  0.007,  0.018,  0.055,  0.153,  0.479,  0.499]
                # random rate 0.1
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.503,  0.503,  0.509,  0.520,  0.551,  0.641,  0.932,  0.950]
                # acc gain         = [ 0.001,  0.002,  0.008,  0.018,  0.050,  0.140,  0.431,  0.450]
                # random rate 0.2
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.502,  0.502,  0.507,  0.518,  0.558,  0.639,  0.880,  0.900]
                # acc gain         = [ 0.001,  0.002,  0.006,  0.018,  0.057,  0.139,  0.379,  0.399]
                # random rate 0.3
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.501,  0.503,  0.507,  0.519,  0.547,  0.621,  0.827,  0.850]
                # acc gain         = [ 0.001,  0.003,  0.005,  0.018,  0.047,  0.120,  0.326,  0.349]
                # random rate 0.5
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.501,  0.503,  0.507,  0.519,  0.540,  0.590,  0.727,  0.750]
                # acc gain         = [ 0.001,  0.002,  0.006,  0.018,  0.040,  0.089,  0.227,  0.249]
                # random rate 0.7
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.501,  0.503,  0.508,  0.518,  0.533,  0.561,  0.628,  0.650]
                # acc gain         = [ 0.001,  0.002,  0.008,  0.017,  0.033,  0.061,  0.128,  0.150]
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
                        assert _either_1_or_neg1(target_posneg1___b_o)#debug purpose
                        #<  model param       neural net infra
                        the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                        the_optim = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=[the_layer], learning_rate___s=learning_rate)
                        backward_to_them = []
                        backward_to_them.extend(the_layer.parameters())
                        #backward_to_them.pop()#####

                        #     ori__raw_weight___o_i = torch.rand(size=[out_dim, in_dim])*-1.
                        #<  calc          forward
                        ori__raw_weight___o_i:torch.Tensor = the_layer(input_posneg1___b_i)
                        assert _tensor_shape_check(ori__raw_weight___o_i, batch, out_dim)
                        ori__raw_weight___o_i.backward(gradient=target_posneg1___b_o, inputs=backward_to_them)

                        # _, grad_like_for___raw_weight___o_i = _algo_test__backward_function( \
                        #     input_posneg1___b_i=input_posneg1___b_i, target___b_o=target_posneg1___b_o,raw_weight___o_i=ori__raw_weight___o_i)
                        # assert isinstance(grad_like_for___raw_weight___o_i, torch.Tensor)
                        # assert _tensor_shape_check(grad_like_for___raw_weight___o_i, out_dim, in_dim)

                        #<  ori   accuracy
                        ori__accuracy___s, recommended_result_value_name = \
                                _test___binary_accuracy___full_safety(target___b_o = target_posneg1___b_o, 
                                        output_posneg1___b_o = ori__raw_weight___o_i, mean_per =  'for_all', target_is_already_posneg1=True)
                        assert recommended_result_value_name == "accuracy___s"

                        # ori__output_posneg1___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i, 
                        #                                 raw_weight___o_i=ori__raw_weight___o_i, input_is_already_posneg1=True)
                        # ori__accuracy___s, recommended_result_value_name = \
                        #         _test___binary_accuracy___full_safety(target___b_o=target_posneg1___b_o, 
                        #                 output_posneg1___b_o=ori__output_posneg1___b_o, mean_per="for_all", target_is_already_posneg1=True)
                        # assert recommended_result_value_name == "accuracy___s"

                        #<  step
                        the_optim.step()
                        # new__raw_weight___o_i = _test___optimizer_algo___full_safety(ori__raw_weight___o_i = ori__raw_weight___o_i,
                        #         grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___o_i, learning_rate = learning_rate)

                        #<  new   accuracy
                        new__raw_weight___o_i:torch.Tensor = the_layer(input_posneg1___b_i)
                        new__accuracy___s, recommended_result_value_name = \
                                _test___binary_accuracy___full_safety(target___b_o = target_posneg1___b_o, 
                                        output_posneg1___b_o = new__raw_weight___o_i, mean_per =  'for_all', target_is_already_posneg1=True)
                        assert recommended_result_value_name == "accuracy___s"


                        # new__output_posneg1___b_o, _ = _test___DNN_forward___full_safety(input___b_i=input_posneg1___b_i, 
                        #                         raw_weight___o_i=new__raw_weight___o_i, input_is_already_posneg1=True)
                        # new__accuracy___s, recommended_result_value_name = \
                        #         _test___binary_accuracy___full_safety(target___b_o=target_posneg1___b_o, 
                        #                 output_posneg1___b_o=new__output_posneg1___b_o, mean_per="for_all", target_is_already_posneg1=True)
                        # assert recommended_result_value_name == "accuracy___s"

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

    ____test____integrated_test()
    pass


