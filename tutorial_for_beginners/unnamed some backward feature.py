from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.Util import _tensor_equal 
import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######




'''This function is from my Util.py file. No validation here. '''
def _tensor_shape_check(the_tensor:torch.Tensor, *args)->bool:
    #assert isinstance(args, list)
    if (args.__len__() == 0) or (args.__len__() == 1 and args[0] == 1):
        if the_tensor.shape == torch.Size([1]):
            return True
        if the_tensor.shape == torch.Size([]):
            return True
        else:
            return False
        pass# if (args.__len__() == 0) or (args.__len__() == 1 and args[0] == 1):
    _temp_shape = torch.Size(args)
    return the_tensor.shape == _temp_shape












'''pytorch feature test          I found the bug. I add some assert. I believe it's solved now. So this section is not important any more'''
if "trivial test" and __DEBUG_ME__() and True:
    def ____pytorch_feature_test():
        if "buffer_0" and True:
            buffer_0 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_0_clipped = buffer_0[:3, :5]
            not_important = torch.randn(size=[2, 3], dtype=torch.float32)
            #<  forward
            x = not_important@buffer_0_clipped
            x.backward(gradient=torch.randn(size=[2, 5]), inputs=[buffer_0])

            assert buffer_0_clipped.grad is None

            assert buffer_0.grad is not None
            assert isinstance(buffer_0.grad, torch.Tensor)
            pass#/ test

        if "buffer_0_clipped" and True:
            buffer_0 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_0_clipped = buffer_0[:3, :5]
            not_important = torch.randn(size=[2, 3], dtype=torch.float32)
            #<  forward
            x = not_important@buffer_0_clipped
            x.backward(gradient=torch.randn(size=[2, 5]), inputs=[buffer_0_clipped])

            assert buffer_0_clipped.grad is not None
            assert isinstance(buffer_0_clipped.grad, torch.Tensor)

            assert buffer_0.grad is None
            pass#/ test

        if "buffer_0, buffer_1" and True:
            buffer_0 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_0_clipped = buffer_0[:3, :5]
            buffer_1 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_1_clipped = buffer_1[:5, :11]

            not_important = torch.randn(size=[2, 3], dtype=torch.float32)
            #<  forward
            x = not_important@buffer_0_clipped@buffer_1_clipped
            x.backward(gradient=torch.randn(size=[2, 11]), inputs=[buffer_0, buffer_1])

            assert buffer_0_clipped.grad is None

            assert buffer_0.grad is not None
            assert isinstance(buffer_0.grad, torch.Tensor)

            assert buffer_1_clipped.grad is None

            assert buffer_1.grad is not None
            assert isinstance(buffer_1.grad, torch.Tensor)
            pass#/ test

        if "buffer_0_clipped, buffer_1_clipped" and True:
            buffer_0 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_0_clipped = buffer_0[:3, :5]
            buffer_1 = torch.randn(size=[100,100], dtype=torch.float32, requires_grad=True)
            buffer_1_clipped = buffer_1[:5, :11]

            not_important = torch.randn(size=[2, 3], dtype=torch.float32)
            #<  forward
            x = not_important@buffer_0_clipped@buffer_1_clipped
            x.backward(gradient=torch.randn(size=[2, 11]), inputs=[buffer_0_clipped, buffer_1_clipped])

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





