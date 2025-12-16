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





if "falses_like init perf test" and False:
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






if "expand vs repeat, I asked this in 2024 version" and False:
    "according to what gradientlord replied to in pytorch forum in 2024, "
    "https://discuss.pytorch.org/t/how-to-avoid-repeating-a-dimention-before-some-customized-multiplication/206428"
    "notice, gradientlordj probably tested with the python's vanilla timeit module."
    "The official timeit module is better bc it's NOT A SPAM, but my version IS A SPAM!!!"

    def null_func():
        A = torch.rand([2,5,3])
        B = torch.rand([2,5])
        C = torch.rand([2,5,3])
        for _ in range(100):
            pass
        A * C
        pass
    time_of_null = timeit(null_func,time_at_most=2.)[0]#307736 epoch for 2 sec
    "6.21us"

    def Using_Expand():
        A = torch.rand([2,5,3])
        B = torch.rand([2,5])
        C = torch.rand([2,5,3])
        for _ in range(100):
            C = B.unsqueeze(dim=-1).expand(-1, -1, 3)
            #Returns a new view of the self tensor
            pass
        A * C
        pass
    time_of_Using_Expand = timeit(Using_Expand,time_at_most=2.)[0]-time_of_null#4096 epoch for 2 sec
    "166us"
    "gradientlord's result: 2.85 µs ± 16.7 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)"

    def Adding_a_dummy_dimension():
        A = torch.rand([2,5,3])
        B = torch.rand([2,5])
        C = torch.rand([2,5,3])
        for _ in range(100):
            C = B[:, :, None]
            pass
        A * C
        pass
    time_of_Adding_a_dummy_dimension = timeit(Using_Expand,time_at_most=2.)[0]-time_of_null#4096 epoch for 2 sec
    "166us"
    "gradientlord's result: 2.95 µs ± 52.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)"

    def Using_Repeat():
        A = torch.rand([2,5,3])
        B = torch.rand([2,5])
        C = torch.rand([2,5,3])
        for _ in range(100):
            C = B.unsqueeze(dim=-1).repeat(1, 1, 3)
            #Repeats this tensor along the specified dimensions.
            #Unlike ~Tensor.expand, this function copies the tensor's data.
            pass
        A * C
        pass
    time_of_Using_Repeat = timeit(Using_Repeat,time_at_most=2.)[0]-time_of_null#4096 epoch for 2 sec
    "504us"
    "gradientlord's result: 2.67 µs ± 31.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)"
    
    "but, bro, my result is a bit too clean. It may be wrong...."
    pass







if "expand vs repeat, real size version" and True:
    "details uppon"
    "result is the same. Expand is faster, while the implicit [ : :None] style is basically the same. repeat is slow."
    "it's still the view vs real object story."
    def null_func():
        A = torch.rand([10,20,100])
        B = torch.rand([10,20])
        C = torch.rand([10,20,100])
        for _ in range(10):
            pass
        A * C
        pass
    #time_of_null = timeit(null_func,time_at_most=2., _debug__provides_log = True)
    time_of_null = timeit(null_func,time_at_most=2.)[0]
    "4096 epoch for 2 sec, 105us"

    def Using_Expand():
        A = torch.rand([10,20,100])
        B = torch.rand([10,20])
        C = torch.rand([10,20,100])
        for _ in range(10):
            C = B.unsqueeze(dim=-1).expand(-1, -1, 100)
            #Returns a new view of the self tensor
            pass
        A * C
        pass
    # time_of_Using_Expand = timeit(Using_Expand,time_at_most=20., _debug__provides_log = True)
    # aaaaaa = time_of_Using_Expand[0]-time_of_null
    time_of_Using_Expand = timeit(Using_Expand,time_at_most=20.)[0]-time_of_null
    "149402 epoch for 20 sec, 19us"

    def Adding_a_dummy_dimension():
        A = torch.rand([10,20,100])
        B = torch.rand([10,20])
        C = torch.rand([10,20,100])
        for _ in range(10):
            C = B[:, :, None]
            pass
        A * C
        pass
    # time_of_Adding_a_dummy_dimension = timeit(Adding_a_dummy_dimension,time_at_most=20., _debug__provides_log = True)
    # aaaaaa = time_of_Adding_a_dummy_dimension[0]-time_of_null
    time_of_Adding_a_dummy_dimension = timeit(Adding_a_dummy_dimension,time_at_most=20.)[0]-time_of_null
    "152021 epoch for 20 sec, 17.4us"

    def Using_Repeat():
        A = torch.rand([10,20,100])
        B = torch.rand([10,20])
        C = torch.rand([10,20,100])
        for _ in range(10):
            C = B.unsqueeze(dim=-1).repeat(1, 1, 100)
            #Repeats this tensor along the specified dimensions.
            #Unlike ~Tensor.expand, this function copies the tensor's data.
            pass
        A * C
        pass
    
    # time_of_Using_Repeat = timeit(Using_Repeat,time_at_most=20., _debug__provides_log = True)
    # aaaaaa = time_of_Using_Repeat[0]-time_of_null
    time_of_Using_Repeat = timeit(Using_Repeat,time_at_most=20.)[0]-time_of_null
    "98067 epoch for 20 sec, 77us"
    pass





