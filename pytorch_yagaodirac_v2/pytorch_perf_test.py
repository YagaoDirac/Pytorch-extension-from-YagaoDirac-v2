import torch
from pathlib import Path
import sys
# sys.path.append(str(Path(__file__).parent))
# from timeit_yagaodirac import timeit

sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit



"a performance test. in pytorch, int tensor comparison is the same speed as tensor tensor comparison."
if "perf test." and False:
    '''result. in pytorch, int tensor comparison is the same speed as tensor tensor comparison.'''
    def func_a_t():
        input = torch.rand(size=(100,100), device='cuda')
        a_i:int = input.nelement()
        a_t:torch.Tensor = torch.tensor(a_i, device='cuda')
        the_sum_t:torch.Tensor = input.gt(2.).sum()
        for _ in range(20):
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            if a_t>the_sum_t:
                pass
            pass
        pass
    time_of_tensor_ver, _log = timeit(func_a_t,time_at_most=2., _debug__provides_log = True)

    def func_a_i():
        input = torch.rand(size=(100,100), device='cuda')
        a_i:int = input.nelement()
        a_t:torch.Tensor = torch.tensor(a_i, device='cuda')
        the_sum_t:torch.Tensor = input.gt(2.).sum()
        for _ in range(20):
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            if a_i>the_sum_t:
                pass
            pass
        pass
    time_of_int_ver,_ = timeit(func_a_i,time_at_most=2.)

    def func_empty():
        input = torch.rand(size=(100,100), device='cuda')
        a_i:int = input.nelement()
        a_t:torch.Tensor = torch.tensor(a_i, device='cuda')
        the_sum_t:torch.Tensor = input.gt(2.).sum()
        for _ in range(20):
            pass
        pass
    time_of_empty,_ = timeit(func_empty,time_at_most=2.)
    print("int ver:  ", time_of_int_ver-time_of_empty)
    print("tensor ver:  ", time_of_tensor_ver-time_of_empty)
    pass






if "falses_like init perf test" and True:
    def zero_to_false():
        for _ in range(100):
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            a = torch.zeros([100,100], dtype=torch.bool)
            pass
        pass
    def empty_to_false():
        for _ in range(100):
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            a = torch.empty(size=[100,100], dtype=torch.bool)
            a.fill_(False)
            pass
        pass
    def null_func():
        for _ in range(100):
            pass
        pass
            
    null_time = timeit(null_func,time_at_most=2.)[0]
    zero_to_false_time = timeit(zero_to_false,time_at_most=2.)[0]
    empty_to_false_time = timeit(empty_to_false,time_at_most=2.)[0]
    zero_to_false_net_time = zero_to_false_time - null_time
    empty_to_false_net_time = empty_to_false_time - null_time
    assert zero_to_false_net_time<empty_to_false_net_time
    assert zero_to_false_net_time*1.5<empty_to_false_net_time
    assert zero_to_false_net_time*2>empty_to_false_net_time
    pass






