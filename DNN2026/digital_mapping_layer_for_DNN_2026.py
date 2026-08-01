from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, _either_1_or_neg1, _tensor_shape_check, \
        iota
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


if "some pytorch feature test" and False:

    aaaaaaa = []

    from typing import Any
    class pytorch_customized_autograd_test(torch.autograd.Function):
        '''实际上和最早的写法，调用关系上，先forward，后setup_context。两个在forward pass都会调用。
        以前的写法就是全部都在forward函数里面。感觉区别不大。
        我以前的笔记里面写过一个，新的3段式的forward好像提供类型检查。'''
        @staticmethod
        #def forward(*args: Any, **kwargs: Any)->Any:
        def forward(*args: Any, **kwargs: Any)->Any:
            print("inside forward")
            input_0:torch.Tensor = args[0]
            return input_0 +1.

        @staticmethod
        def setup_context(ctx:torch.autograd.function.FunctionCtx, inputs, output):
            print("inside setup_context")
            input_0:torch.Tensor = inputs[0]
            input_1:torch.Tensor = inputs[1]
            
            input_0_needs_grad = torch.tensor([input_0.requires_grad])
            input_1_needs_grad = torch.tensor([input_1.requires_grad])
            ctx.save_for_backward(input_0, input_1, input_0_needs_grad, input_1_needs_grad)

            aaaaaaa.append(ctx)
            return

        @staticmethod
        def backward(ctx, g_in_b_o):
            (input_0, input_1, input_0_needs_grad, input_1_needs_grad) = ctx.saved_tensors
            return None, None

        pass  # class
    if "test" and __DEBUG_ME__() and False:
        def ____def____call_sequence_test():
            input_0 = torch.tensor([123.])
            input_1 = torch.tensor([555.])
            output:torch.Tensor = pytorch_customized_autograd_test.apply(input_0, input_1)
            aaaaaaa
            output.backward(gradient=torch.tensor([1212.]), inputs=[input_0, input_1]) 


            return 
        ____def____call_sequence_test()

        pass
    pass    

if "it looks buggy when output multiple results" and False:
    '''https://docs.pytorch.org/docs/2.13/notes/extending.html'''

    class MyCube(torch.autograd.Function):
        @staticmethod
        def forward(x):
            # We wish to save dx for backward. In order to do so, it must
            # be returned as an output.
            dx = 3 * x ** 2
            result = x ** 3
            return result, dx

        @staticmethod
        def setup_context(ctx, inputs, output):
            x, = inputs
            result, dx = output
            ctx.save_for_backward(x, dx)

        @staticmethod
        def backward(ctx, grad_output, grad_dx):
            x, dx = ctx.saved_tensors
            # In order for the autograd.Function to work with higher-order
            # gradients, we must add the gradient contribution of `dx`,
            # which is grad_dx * 6 * x.
            result = grad_output * dx + grad_dx * 6 * x
            return result

    # Wrap MyCube in a function so that it is clearer what the output is

    input = torch.tensor([2.], requires_grad=True)
    result:torch.Tensor
    result, dx = MyCube.apply(input)

    dx.backward(gradient=torch.tensor([7.]), inputs=[input])
    result.backward(gradient=torch.tensor([3.]), inputs=[input])
    pass










'''let me assume the distribution of raw weight is a uniform distribution between -1 to 0 for now.
        come back later.
        come back later.
        come back later.
'''




def _algo_test__backward_function(input_posneg1___b_i:torch.Tensor, 
            target___b_o:torch.Tensor, raw_weight___o_i:torch.Tensor, 
            SOME_HYPER_PARAM___s = 1. )->tuple[torch.Tensor|None, torch.Tensor|None]:
    '''return grad_like_for___input___b_i, grad_like_for___raw_weight___o_i

    This is algo test for the backward function.
    '''
    #<  shape
    assert input_posneg1___b_i.shape.__len__() == 2
    assert _either_1_or_neg1(input_posneg1___b_i)
    batch = input_posneg1___b_i.shape[0]  
    in_dim = input_posneg1___b_i.shape[1]  

    assert target___b_o.shape.__len__() == 2
    assert target___b_o.shape[0] == batch, "not sure if this one is wrong or the previous one?"
    out_dim = target___b_o.shape[1]

    assert _tensor_shape_check(raw_weight___o_i, out_dim, in_dim), "not sure if this one is wrong or the previous one?"
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

    return grad_like_for___input___b_i, grad_like_for___raw_weight___o_i
if "test" and True:
    def ____test_____algo_test__backward_function():
        if "xxxxxxxxxxxx" and True:
            batch = 3
            in_dim = 5
            out_dim = 2

            batch = 1000
            in_dim = 500
            out_dim = 100


            input_posneg1___b_i = rand_sign(size=[batch, in_dim], dtype=torch.int32)
            assert _either_1_or_neg1(input_posneg1___b_i)

            _index_of_max_of_raw_weight___o = torch.randint(low=0, high=in_dim, size=[out_dim])
            target___b_o = input_posneg1___b_i[:, _index_of_max_of_raw_weight___o]
            assert _either_1_or_neg1(target___b_o)
            # this is a pure random target target___b_o = torch.rand(size=[batch, out_dim])*2. -1. #  -1 to 1

            ori__raw_weight___o_i = torch.rand(size=[out_dim, in_dim])*-1.

            _, grad_like_for___raw_weight___o_i = _algo_test__backward_function( \
                input_posneg1___b_i=input_posneg1___b_i, target___b_o=target___b_o,raw_weight___o_i=ori__raw_weight___o_i)
            assert _tensor_shape_check(grad_like_for___raw_weight___o_i, out_dim, in_dim)

            #<  target into pos neg 1 form.      This is for both ori and new.
            target_posneg1___b_o = target___b_o.gt(0.)
            target_posneg1___b_o = target_posneg1___b_o.to(torch.int32)*2 -1
            assert _either_1_or_neg1(target_posneg1___b_o)


            #<  calc    ori  output
            ori__index_of_max_of_raw_weight___o = ori__raw_weight___o_i.max(dim=1).indices
            ori__output_posneg1___b_o = input_posneg1___b_i[:, ori__index_of_max_of_raw_weight___o]
            assert _either_1_or_neg1(ori__output_posneg1___b_o)
            #<  ori   accuracy
            ori__element_mul_of_target_and_output___b_o = target_posneg1___b_o * ori__output_posneg1___b_o
            ori__element_mul_of_target_and_output___b_o = ori__element_mul_of_target_and_output___b_o.to(torch.float32)

            ori__accuracy___o = ori__element_mul_of_target_and_output___b_o.mean(dim=0)
            ori__accuracy___o = ori__accuracy___o *0.5 + 0.5
            assert ori__accuracy___o.ge(0.).all()
            assert ori__accuracy___o.le(1.).all()


            #<  new raw_weight
            #new__raw_weight___o_i = ori__raw_weight___o_i+grad_like_for___raw_weight___o_i.to(torch.float32)/float(batch) * 0.3 #1w 需要一个自适应。 #没乘任何系数  可能要改？？？？？？？？
            
            adaptive___grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___o_i - grad_like_for___raw_weight___o_i.max()
            adaptive___grad_like_for___raw_weight___o_i = adaptive___grad_like_for___raw_weight___o_i.to(torch.float32)
            adaptive___grad_like_for___raw_weight___o_i /= float(batch)
            assert adaptive___grad_like_for___raw_weight___o_i.le(0.).all()
            new__raw_weight___o_i = torch.tanh(ori__raw_weight___o_i + adaptive___grad_like_for___raw_weight___o_i*0.25) #没乘任何系数  可能要改？？？？？？？？
            assert new__raw_weight___o_i.le(0.).all()

            #<  calc    ori  output
            new__index_of_max_of_raw_weight___o = new__raw_weight___o_i.max(dim=1).indices
            new__output_posneg1___b_o = input_posneg1___b_i[:, new__index_of_max_of_raw_weight___o]
            assert _either_1_or_neg1(new__output_posneg1___b_o)
            #<  ori   accuracy
            new__element_mul_of_target_and_output___b_o = target_posneg1___b_o * new__output_posneg1___b_o
            new__element_mul_of_target_and_output___b_o = new__element_mul_of_target_and_output___b_o.to(torch.float32)

            new__accuracy___o = new__element_mul_of_target_and_output___b_o.mean(dim=0)
            new__accuracy___o = new__accuracy___o *0.5 + 0.5
            assert new__accuracy___o.ge(0.).all()
            assert new__accuracy___o.le(1.).all()

            print(ori__accuracy___o.mean().item(), new__accuracy___o.mean().item())
            pass#/ test

1w
1w
1w继续





        return 
    ____test_____algo_test__backward_function()
    pass









if "backward algo test" and True:
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







assert False, "继续"
class autograd_function_class_for__DigitalMapper_layer__2026(torch.autograd.Function):
    r'''
    forward input list:
    >>> input___b_i
    >>> raw_weight___o_i (make sure this is output of get_useful())
    
    backward input list:
    >>> g_in #shape of g_in must be [batch, out_features]
    '''
    @staticmethod
    def forward(input___b_i:torch.Tensor, raw_weight___o_i:torch.Tensor)->torch.Tensor:
        # input___b_i:torch.Tensor = args[0]# shape must be [batch, in_features]
        # raw_weight___o_i:torch.Tensor = args[1]# shape must be [out_features, in_features]

        #<  real payload
        _temp_index___o = raw_weight___o_i.max(dim=1).indices
        output___b_o = input___b_i[:, _temp_index___o]
        return output___b_o

    @staticmethod
    def setup_context(ctx, inputs, output):
        input___b_i:torch.Tensor = inputs[0]
        raw_weight___o_i:torch.Tensor = inputs[1]
        #output___b_o:torch.Tensor = output
        ctx.save_for_backward(input___b_i, raw_weight___o_i, )

    @staticmethod
    def backward(ctx, target___b_o):
        #shape of g_in must be [batch, out_features]
        input___b_i:torch.Tensor
        raw_weight___o_i:torch.Tensor
        output___b_o:torch.Tensor
        (input___b_i, raw_weight___o_i) = ctx.saved_tensors

        grad__input___b_i     :tuple[torch.tensor|None] = None
        grad__raw_weight___o_i:tuple[torch.tensor|None] = None

        
        
        assert False, "在前面的测试里面"
        

        # if input___b_i.requires_grad:
        # if raw_weight___o_i.requires_grad:



        return grad__input___b_i, grad__raw_weight___o_i
    
    
    pass  # class





















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

def _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style(out_features, in_features, device, dtype) -> torch.Tensor:
    result = torch.rand(size=[out_features, in_features], device=device, dtype=dtype)
    return result
if " test" and __DEBUG_ME__() and False:
    '''感觉不用测啊。。。'''
    pass






class DigitalMapper_layer__2026(torch.nn.Module):
    in_dim         :int
    out_dim        :int
    _init_to_nan   :bool
    _raw_weight___oCAP_iCAP    :torch.nn.parameter.Parameter

    #customizable functions.
    _random_init_algo               :function
    _calc_bigger_capacity__for_in   :function
    _calc_bigger_capacity__for_out  :function
    # _calc_bigger_capacity

    def __init__(self, in_features: int, out_features: int, 
                init_capacity__for_in = 16, init_capacity__for_out = 16, init_to_nan = True, \
                    device=None, dtype=None) -> None:  
        
        #this dtype is only for a inner memory in training. It must be float point number.

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if init_capacity__for_in < in_features:
            init_capacity__for_in = in_features
            pass
        if init_capacity__for_out < out_features:
            init_capacity__for_out = out_features
            pass

        self.in_dim = in_features
        self.out_dim = out_features
        self._init_to_nan = init_to_nan
        self._raw_weight___oCAP_iCAP = torch.nn.Parameter(torch.empty(
                init_capacity__for_out, init_capacity__for_in,
                        requires_grad = False, **factory_kwargs), 
                        requires_grad = False)
        assert self._raw_weight___oCAP_iCAP.dtype in [torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16]
        if self._init_to_nan:
            self._raw_weight___oCAP_iCAP.fill_(torch.nan)
            pass

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
            with torch.no_grad():
                #raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
                out_features_s = self.raw_weight.shape[0]
                out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
                #print(out_features_iota, "out_features_iota")
                index_of_max_o = self.raw_weight.max(dim=1).indices
                #print(index_of_max_o, "index_of_max_o")

                one_hot_o_i = torch.zeros_like(self.raw_weight)
                one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
                return one_hot_o_i

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
        input___b_i = input


        assert False,"1w 要不要用 get_max_index()"

        the_useful_part___o_i = self.get_useful_part_of_raw_weight()#or do I want to squeeze it??? No for now. Untested.
        #<  real payload
        #_temp_index___o = the_useful_part___o_i.argmax(dim=1)
        _temp_index___o = self.get_max_index()
        output___b_o = input___b_i[:, _temp_index___o]
        del _temp_index___o
        assert False, "untested"
        return output___b_o
        #end of function.
    def get_max_index(self)->torch.Tensor:
        with torch.no_grad():
            _temp_useful_part = self.get_useful_part_of_raw_weight()
            the_max_index = _temp_useful_part.max(dim=1).indices
            return the_max_index

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

        _temp_new_memory = torch.empty(size=[_temp_new_out_capacity,_temp_new_in_capacity])
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




#改形状，两种改法。反向查询索引。    
if "basic reshape. " and __DEBUG_ME__() and True:
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




if "add input slot" and __DEBUG_ME__() and True:
    def ____add_input____():

        if "add input.     full assert      no class" and True:

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
            _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
            ori___output___b_o = input___b_i[:, _temp_one_hot___o]
            del _temp_one_hot___o
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
            #<  new         forward path
            _temp___new____one_hot___o = new___training_buffer___o_ii.argmax(dim=1)
            flag___same_output_as_ori___o = _temp___new____one_hot___o.lt(in_dim___ori)
            new___input___b_i = torch.empty(size=[batch, in_dim___ori+in_dim___new])
            new___input___b_i[:, :in_dim___ori] = input___b_i
            new___input___b_i[:, in_dim___ori:] = extra_input___b_ii
            new___output___b_o = new___input___b_i[:, _temp___new____one_hot___o]
            del _temp___new____one_hot___o
            #<  assert 
            assert _tensor_equal(new___output___b_o, torch.tensor([ [517,  14,  12,],  
                                                                    [527,  24,  22,],]))

            assert _tensor_equal(new___output___b_o[:, [1,2]], ori___output___b_o[:, [1,2]])

            assert _tensor_equal(new___output___b_o[:, [False, True, True]], ori___output___b_o[:, [False, True, True]])
            assert _tensor_equal(new___output___b_o[:, flag___same_output_as_ori___o], ori___output___b_o[:, flag___same_output_as_ori___o])

            pass#/ test 

        if "add input.     no assert       no class" and True:
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
                                _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
                                ori___output___b_o = input___b_i[:, _temp_one_hot___o]
                                del _temp_one_hot___o
                                assert ori___output___b_o.shape == torch.Size([batch, out_dim])

                                #<  the new shape
                                in_dim_in_total = in_dim___ori + in_dim___new
                                new___training_buffer___o_ii = torch.empty(size=[out_dim, in_dim_in_total])
                                new___training_buffer___o_ii[:, :in_dim___ori] = ori___training_buffer___o_i[:, :in_dim___ori]
                                new___training_buffer___o_ii[:, in_dim___ori:in_dim_in_total] = torch.rand(size=[out_dim, in_dim___new])
                                assert new___training_buffer___o_ii.shape == torch.Size([out_dim, in_dim_in_total])
                                #<  new         forward path
                                _temp___new____one_hot___o = new___training_buffer___o_ii.argmax(dim=1)
                                flag___same_output_as_ori___o = _temp___new____one_hot___o.lt(in_dim___ori)
                                new___input___b_i = torch.empty(size=[batch, in_dim___ori+in_dim___new])
                                new___input___b_i[:, :in_dim___ori] = input___b_i
                                new___input___b_i[:, in_dim___ori:] = extra_input___b_ii
                                new___output___b_o = new___input___b_i[:, _temp___new____one_hot___o]
                                del _temp___new____one_hot___o
                                #<  assert 
                                assert _tensor_equal(new___output___b_o[:, flag___same_output_as_ori___o], ori___output___b_o[:, flag___same_output_as_ori___o])

                                pass#for _
                            pass#for batch
                        pass#for out_dim
                    pass#for in_dim___ori
                pass#for in_dim___new

            pass#/ test 


        assert False
        if "add input.     no assert       with class" and True:
            for batch in[2, 13, 37]:
                for out_dim in[3, 14, 53]:
                    for in_dim___ori in[5, 17, 71]:
                        for in_dim___new in[7, 21, 92]:
                            for _ in range(22):

                                #<  dataset
                                input___b_i = torch.rand(size=[batch, in_dim___ori])
                                extra_input___b_ii = torch.rand(size=[batch, in_dim___new])
                                
                                #<  model param
                                assert False
                                the_layer = DigitalMapper_layer__2026()
                                ori___training_buffer___o_i =torch.rand(size=[out_dim, in_dim___ori])
                                
                                #<  original    forward path
                                _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
                                ori___output___b_o = input___b_i[:, _temp_one_hot___o]
                                del _temp_one_hot___o
                                assert ori___output___b_o.shape == torch.Size([batch, out_dim])

                                #<  the new shape
                                in_dim_in_total = in_dim___ori + in_dim___new
                                new___training_buffer___o_ii = torch.empty(size=[out_dim, in_dim_in_total])
                                new___training_buffer___o_ii[:, :in_dim___ori] = ori___training_buffer___o_i[:, :in_dim___ori]
                                new___training_buffer___o_ii[:, in_dim___ori:in_dim_in_total] = torch.rand(size=[out_dim, in_dim___new])
                                assert new___training_buffer___o_ii.shape == torch.Size([out_dim, in_dim_in_total])
                                #<  new         forward path
                                _temp___new____one_hot___o = new___training_buffer___o_ii.argmax(dim=1)
                                flag___same_output_as_ori___o = _temp___new____one_hot___o.lt(in_dim___ori)
                                new___input___b_i = torch.empty(size=[batch, in_dim___ori+in_dim___new])
                                new___input___b_i[:, :in_dim___ori] = input___b_i
                                new___input___b_i[:, in_dim___ori:] = extra_input___b_ii
                                new___output___b_o = new___input___b_i[:, _temp___new____one_hot___o]
                                del _temp___new____one_hot___o
                                #<  assert 
                                assert _tensor_equal(new___output___b_o[:, flag___same_output_as_ori___o], ori___output___b_o[:, flag___same_output_as_ori___o])

                                pass#for _
                            pass#for batch
                        pass#for out_dim
                    pass#for in_dim___ori
                pass#for in_dim___new

            pass#/ test 
















        assert False
        if "这个怎么写？？？" and True:
            for batch in [2,5,10]:
                for out_dim in [3,7,11]:
                    for in_dim in [6,9,13]:
                        for _ in range(5):
                            #<  the answer
                            keep_these_output = torch.rand(size=[out_dim])
                            keep_these_output = keep_these_output.gt(0.5)

                            new_out_dim = int(keep_these_output.sum().item())

                            #<  dataset
                            input___b_i = torch.randn(size=[batch, in_dim])
                            #<  model param
                            ori___training_buffer___o_i = torch.randn(size=[out_dim, in_dim])
                            #<  original    forward path
                            _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
                            ori___output___b_o = input___b_i[:, _temp_one_hot___o]
                            del _temp_one_hot___o
                            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
                            #<  the new shape
                            new___training_buffer___o_i = ori___training_buffer___o_i[keep_these_output,:]
                            assert new___training_buffer___o_i.shape == torch.Size([new_out_dim, in_dim])
                            _temp___new____one_hot___o = new___training_buffer___o_i.argmax(dim=1)
                            new___output___b_o = input___b_i[:, _temp___new____one_hot___o]
                            del _temp___new____one_hot___o
                            #<  assert 
                            assert _tensor_equal(new___output___b_o, ori___output___b_o[:, keep_these_output])
                            pass#for _
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test
        
        if "xxxxxxxxxxxxxxxxxx" and True:
            for batch in [2,5,10]:
                for out_dim in [3,7,11]:
                    for in_dim in [6,9,13]:
                        for _ in range(5):
                            #<  the answer
                            keep_these_output = torch.rand(size=[out_dim])
                            keep_these_output = keep_these_output.gt(0.5)

                            new_out_dim = int(keep_these_output.sum().item())

                            #<  dataset
                            input___b_i = torch.randn(size=[batch, in_dim])
                            #<  model
                            the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                            #<  original    forward path
                            ori___output___b_o = the_layer.forward(input___b_i)
                            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
                            #<  the new shape
                            the_layer.keep_output_slot(keep_these_output)
                            assert the_layer.raw_weight.shape == torch.Size([new_out_dim, in_dim])
                            new___output___b_o = the_layer.forward(input___b_i)
                            #<  assert 
                            assert _tensor_equal(new___output___b_o, ori___output___b_o[:, keep_these_output])
                            pass#for _
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test




        return 
    ____add_input____()
    pass










if "delete output slot" and __DEBUG_ME__() and False:
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
            input___b_i = torch.tensor([[1.,  1.,  1.],
                                        [1.,  1.,  1.],])
            label___b_o = torch.tensor([[1.,  1.,  1.,  1.,  1.],
                                        [1.,  1.,  1.,  1.,  1.],])
            #<  model param
            ori___training_buffer___o_i = torch.tensor([  
                                                    [0.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3],
                                                    [0.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3],
                                                    [1.1, 0.2, 0.3],])#32321
            #<  original    forward path
            _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
            ori___output___b_o = input___b_i[:, _temp_one_hot___o]
            del _temp_one_hot___o
            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
            #<  the new shape
            new___training_buffer___o_i = ori___training_buffer___o_i[keep_these_output,:]
            assert new___training_buffer___o_i.shape == torch.Size([new_out_dim, in_dim])
            #<  new         forward path
            _temp___new____one_hot___o = new___training_buffer___o_i.argmax(dim=1)
            new___output___b_o = input___b_i[:, _temp___new____one_hot___o]
            del _temp___new____one_hot___o
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
                            input___b_i = torch.randn(size=[batch, in_dim])
                            #<  model param
                            ori___training_buffer___o_i = torch.randn(size=[out_dim, in_dim])
                            #<  original    forward path
                            _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
                            ori___output___b_o = input___b_i[:, _temp_one_hot___o]
                            del _temp_one_hot___o
                            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
                            #<  the new shape
                            new___training_buffer___o_i = ori___training_buffer___o_i[keep_these_output,:]
                            assert new___training_buffer___o_i.shape == torch.Size([new_out_dim, in_dim])
                            _temp___new____one_hot___o = new___training_buffer___o_i.argmax(dim=1)
                            new___output___b_o = input___b_i[:, _temp___new____one_hot___o]
                            del _temp___new____one_hot___o
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
                        for _ in range(5):
                            #<  the answer
                            keep_these_output = torch.rand(size=[out_dim])
                            keep_these_output = keep_these_output.gt(0.5)

                            new_out_dim = int(keep_these_output.sum().item())

                            #<  dataset
                            input___b_i = torch.randn(size=[batch, in_dim])
                            #<  model
                            the_layer = DigitalMapper_layer__2026(in_features=in_dim, out_features=out_dim)
                            #<  original    forward path
                            ori___output___b_o = the_layer.forward(input___b_i)
                            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
                            #<  the new shape
                            the_layer.keep_output_slot(keep_these_output)
                            assert the_layer.raw_weight.shape == torch.Size([new_out_dim, in_dim])
                            new___output___b_o = the_layer.forward(input___b_i)
                            #<  assert 
                            assert _tensor_equal(new___output___b_o, ori___output___b_o[:, keep_these_output])
                            pass#for _
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test




        return 
    ____delete_output____()
    pass







