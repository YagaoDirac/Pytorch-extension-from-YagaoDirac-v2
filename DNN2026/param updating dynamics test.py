from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, \
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



assert _tensor_equal(torch.tensor(-1.1).fmod(1.), [-0.1])

def 不要用____clampped_distorted_randn(size, mul_me:float|torch.Tensor, exp_me:float|torch.Tensor, 
            device = None, dtype = None, requires_grad=False)->torch.Tensor:
    assert exp_me>0.7 and exp_me<3. 
    _temp_1 = torch.randn(size=size)*mul_me
    _temp_2 = _temp_1.pow(exp_me)
    _temp_3 = _temp_2.fmod(1.)
    _temp_4 = _temp_3*rand_sign(size=size, device = device, dtype = dtype)
    result = _temp_4
    return result

if "VISUAL     clampped_distorted_randn" and False:
    def ____clampped_distorted_randn():
        from matplotlib import pyplot as plt
        
        for exp_me in [0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2, 1.5, 1.7, 2., 2.2, 2.5, 2.7, 3.]:
            display_me = clampped_distorted_randn(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn      exp_me  {exp_me}")
            plt.show()
            pass

        for exp_me in [1., 1.2, 1.5, 2., 3., 4.]:
            display_me = clampped_distorted_randn(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn      exp_me  {exp_me}")
            plt.show()
            pass
        for exp_me in [1., 0.8, 0.6, 0.5, 0.3]:
            display_me = clampped_distorted_randn(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn      exp_me  {exp_me}")
            plt.show()
            pass
        for exp_me in [-1., -1.2, -1.5, -2., -3., -4.]:
            display_me = clampped_distorted_randn(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn      exp_me  {exp_me}")
            plt.show()
            pass
        for exp_me in [-1., -0.8, -0.6, -0.5, -0.3]:
            display_me = clampped_distorted_randn(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn      exp_me  {exp_me}")
            plt.show()
            pass

        return
    ____clampped_distorted_randn()
    pass


'''rand gen for test'''
def rand_gen___clampped_distorted___v1_1(size, exp_me:float|torch.Tensor = 2., 
            device = None, dtype = None, requires_grad=False)->torch.Tensor:
    #assert exp_me>=1.5 and exp_me<=2.5
    _temp_1 = torch.rand(size=size)
    _temp_2 = _temp_1.pow(exp_me)
    _temp_3 = _temp_2*rand_sign(size=size, device = device, dtype = dtype)
    result = _temp_3
    return result

if "VISUAL     clampped_distorted_randn" and True:
    def ____rand_gen___clampped_distorted___v1_1():
        from matplotlib import pyplot as plt
        
        for exp_me in [0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2, 1.5, 1.7, 2., 2.2, 2.5, 2.7, 3.]:
            display_me = rand_gen___clampped_distorted___v1_1(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"exp_me  {exp_me}")
            plt.show()
            pass

        for exp_me in [1., 1.2, 1.5, 2., 3., 4.]:
            display_me = rand_gen___clampped_distorted___v1_1(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn___v1_1     exp_me  {exp_me}")
            plt.show()
            pass
        for exp_me in [1., 0.8, 0.6, 0.5, 0.3]:
            display_me = rand_gen___clampped_distorted___v1_1(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn___v1_1     exp_me  {exp_me}")
            plt.show()
            pass
        for exp_me in [-1., -1.2, -1.5, -2., -3., -4.]:
            display_me = rand_gen___clampped_distorted___v1_1(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn___v1_1     exp_me  {exp_me}")
            plt.show()
            pass
        for exp_me in [-1., -0.8, -0.6, -0.5, -0.3]:
            display_me = rand_gen___clampped_distorted___v1_1(size=[10000], exp_me=exp_me)
            plt.hist(display_me, bins = 50)
            plt.title(f"clampped_distorted_randn___v1_1     exp_me  {exp_me}")
            plt.show()
            pass

        return
    ____rand_gen___clampped_distorted___v1_1()
    pass





'''Im lucky. This simple formula works pretty well.
So let's keep it simple for now.
'''
if "VISUAL     protect for a lot times." and True:

    from matplotlib import pyplot as plt

    size = [10000]

    pseudo_raw_weight = torch.rand(size=size)*-1.
    assert pseudo_raw_weight.ge(-1.).all()
    assert pseudo_raw_weight.le( 0.).all()
    display_me = pseudo_raw_weight
    plt.hist(display_me, bins = 50)
    plt.title(f"raw_weight")
    plt.show()

    for ii in range(33):
        accuracy = rand_gen___clampped_distorted___v1_1(size=size)*0.5+0.5
        assert accuracy.ge(0.).all()
        assert accuracy.le(1.).all()
        display_me = accuracy
        plt.hist(display_me, bins = 50)
        plt.title(f"accuracy")
        plt.show()

        pseudo_raw_weight___before_protection = pseudo_raw_weight + accuracy -1
        display_me = pseudo_raw_weight___before_protection
        plt.hist(display_me, bins = 50)
        plt.title(f"before_protection")
        plt.show()

        pseudo_raw_weight = torch.tanh(pseudo_raw_weight___before_protection)
        assert pseudo_raw_weight.ge(-1.).all()
        assert pseudo_raw_weight.le( 0.).all()
        display_me = pseudo_raw_weight
        plt.hist(display_me, bins = 50)
        plt.title(f"after_protection")
        plt.show()
        pass


    pass







