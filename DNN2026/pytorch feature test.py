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
'''不重要'''
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






