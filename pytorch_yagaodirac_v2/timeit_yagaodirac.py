# ***************************************************
# Author: YagaoDirac
# find the latest version of this new timeit module in:
# https://github.com/YagaoDirac/Pytorch-extension-from-YagaoDirac-v2
# or maybe it's moved to somewhere else.
#
# feel free to do anything with this code.
#
# to find me: X.com
# ***************************************************




from typing import Optional
import time
import math

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
    
    
    
def _raw_timeit(test_this_callable, params:tuple = (), number = 10000)->tuple[float,float]:
    '''this function is not designed for you to use directly.
    
    returns avg, total. Both in seconds.'''
    assert callable(test_this_callable)
    assert number>0
    _before = time.time()
    for _ in range(number):
        test_this_callable(*params)
        #assert test_this_callable(*params) == 4950
        pass
    _since_before = time.time() - _before
    
    return _since_before/number, _since_before

if "test" and __DEBUG_ME__() and True:
    _result_tuple_ff = _raw_timeit(sum, (range(100),), number=1000000)
    assert _result_tuple_ff[1]>0.3
    assert _result_tuple_ff[1]<0.34
    
    _result_tuple_ff = _raw_timeit(sum, (range(100),), number=100000)
    assert _result_tuple_ff[1]>0.03
    assert _result_tuple_ff[1]<0.035
    _result_tuple_ff = _raw_timeit(sum, (range(100),), number=10000)
    assert _result_tuple_ff[1]>0.0025
    assert _result_tuple_ff[1]<0.004
    pass

BASIC_DELAY_1:float
BASIC_DELAY_4:float
BASIC_DELAY_16:float
BASIC_DELAY_64:float
BASIC_DELAY_256:float
BASIC_DELAY_10000:float
#__DEBUG__total_time_DELAY:dict[int,float] = {}
if "fold in vsc" and True:
    def _the_null_func():
        pass#end of function
    
    for _ in range(123):#warm up
        _raw_timeit(_the_null_func, number=1000)
        pass
    
    _result = 0.
    while _result == 0.:
        _result_tuple_ff = _raw_timeit(_the_null_func, number=1)
        _result = _result_tuple_ff[0]
        pass
    BASIC_DELAY_1 = _result
    
    _result = 0.
    while _result == 0.:
        _result_tuple_ff = _raw_timeit(_the_null_func, number=4)
        _result = _result_tuple_ff[0]
        pass
    BASIC_DELAY_4 = _result
    
    
    _result = 0.
    while _result == 0.:
        _result_tuple_ff = _raw_timeit(_the_null_func, number=16)
        _result = _result_tuple_ff[0]
        pass
    BASIC_DELAY_16 = _result
    
    
    _result = 0.
    while _result == 0.:
        _result_tuple_ff = _raw_timeit(_the_null_func, number=64)
        _result = _result_tuple_ff[0]
        pass
    BASIC_DELAY_64 = _result
    
    
    _result = 0.
    while _result == 0.:
        _result_tuple_ff = _raw_timeit(_the_null_func, number=256)
        _result = _result_tuple_ff[0]
        pass
    BASIC_DELAY_256 = _result
    
    _result = 0.
    while _result == 0.:
        _result_tuple_ff = _raw_timeit(_the_null_func, number=10000)
        _result = _result_tuple_ff[0]
        pass
    BASIC_DELAY_10000 = _result

def _calc_basic_delay(number:int)->float:
    if number == 1:
        return BASIC_DELAY_1
        
    if number<=4:
        _ratio = math.log(number/1)/math.log(4/1)
        return BASIC_DELAY_1*math.pow(BASIC_DELAY_4/BASIC_DELAY_1, _ratio)

    if number<=16:
        _ratio = math.log(number/4)/math.log(16/4)
        return BASIC_DELAY_4*math.pow(BASIC_DELAY_16/BASIC_DELAY_4, _ratio)

    if number<=64:
        _ratio = math.log(number/16)/math.log(64/16)
        return BASIC_DELAY_16*math.pow(BASIC_DELAY_64/BASIC_DELAY_16, _ratio)

    if number<=256:
        _ratio = math.log(number/64)/math.log(256/64)
        return BASIC_DELAY_64*math.pow(BASIC_DELAY_256/BASIC_DELAY_64, _ratio)
    
    if number<=10000:
        _ratio = math.log(number/256)/math.log(10000/256)
        return BASIC_DELAY_256*math.pow(BASIC_DELAY_10000/BASIC_DELAY_256, _ratio)
    
    return BASIC_DELAY_10000

if "test    it has print in it." and __DEBUG_ME__() and False:
    #for number in [10, 100, 1000, 10000, 100000]:
    number = 1.
    while number <100000:
        print(_calc_basic_delay(int(number)), "   ", number)
        number*=1.5
    
    
    
    
    
    
    
def timeit(test_this_callable, params:tuple = (), time_at_most = 0.2, _magic_ratio = 4., \
            base = 4, I_believe_it_comes_stable_at = 10000, timeout_tolerence:Optional[float] = None, \
            _debug__force_warm_up = 0, _debug__provides_log = False, \
                            )->tuple[float, Optional[list[str]]]:
    assert time_at_most>0
    assert _magic_ratio>2.
    assert base>=4 and base<=16
    if timeout_tolerence is None:
        if time_at_most<0.05:
            timeout_tolerence = 3.
            pass
        elif time_at_most<1.:
            timeout_tolerence = 1.5
            pass
        else:
            timeout_tolerence = 1.1
            pass
        pass
    assert timeout_tolerence>1.
    #assert I_believe_it_comes_stable_at>=30 and base<=10000 emmmm 
    
    _log = None
    if _debug__provides_log:
        _log = []
        pass
    
    if _debug__force_warm_up>0:
        _raw_timeit(test_this_callable, params, number=_debug__force_warm_up)
        pass
    
    
    avg_time = -1.
    start_time = time.time()
    number_to_test = 1.
    while number_to_test < I_believe_it_comes_stable_at:
    #for number_to_test in [1,8,64,512]:
        avg_time = _raw_timeit(test_this_callable, params, number=int(number_to_test))[0]
        current_time = time.time()
        time_consumed = current_time-start_time
        if _debug__provides_log:
            _log.append(f"{time_consumed:.6f}s: test for {int(number_to_test)} epoch, result: {avg_time-_calc_basic_delay(int(number_to_test))} , raw result: {avg_time}")
            pass
        
        if time_consumed > time_at_most:
            # timed out, return now.
            return avg_time-_calc_basic_delay(int(number_to_test)), _log
        
        if time_consumed > (time_at_most/base)*timeout_tolerence:
            break
        
        #tail 
        number_to_test *= base
        pass#for 
        
    _try_this_number_to_test = int(time_at_most / avg_time - I_believe_it_comes_stable_at)
    if _try_this_number_to_test<I_believe_it_comes_stable_at*_magic_ratio:
        return avg_time-_calc_basic_delay(int(number_to_test)), _log
        pass
    number_to_test = _try_this_number_to_test
    
    avg_time = _raw_timeit(test_this_callable, params, number=number_to_test)[0]
    current_time = time.time()
    time_consumed = current_time-start_time
    if _debug__provides_log:
        _log.append(f"{time_consumed:.6f}s: test for {number_to_test} epoch, result: {avg_time-_calc_basic_delay(number_to_test)} , raw result: {avg_time}")
        pass
    
    return avg_time-_calc_basic_delay(number_to_test), _log
    pass#end of function

if "test" and __DEBUG_ME__() and True:
    import random
    def _test_func_1():
        random.random()+random.random()
        pass
    avg_time, log = timeit(_test_func_1)
    assert avg_time>2.6e-8
    assert avg_time<3.5e-8
    
    def _test_func_2():
        random.random()+random.random()
        random.random()+random.random()
        pass
    avg_time, log = timeit(_test_func_2)
    assert avg_time>6e-8
    assert avg_time<7e-8
    
    def _test_func_3():
        random.random()+random.random()
        random.random()+random.random()
        random.random()+random.random()
        pass
    avg_time, log = timeit(_test_func_3)
    assert avg_time>8.7e-8
    assert avg_time<1e-7
    pass
    
if "test" and __DEBUG_ME__() and True:
    # a_list = [x*x for x in range(100)]
    # def _test_func__with_assert():
    #     assert sum(a_list) == 328350
    #     pass
    # avg_time, log = timeit(_test_func__with_assert, time_at_most=0.02, _magic_ratio = 1, _debug__provides_log = True)
    
    import random
    a_list = [random.random() for x in range(100)]
    def _test_func__without_assert_1():
        sum(a_list)
        pass
    avg_time, log = timeit(_test_func__without_assert_1)
    assert avg_time>2.5e-7
    assert avg_time<2.8e-7
    def _test_func__without_assert_2():
        sum(a_list)
        sum(a_list)
        pass
    avg_time, log = timeit(_test_func__without_assert_2)
    assert avg_time>5.1e-7
    assert avg_time<5.7e-7
    
    avg_time, log = timeit(sum, params=(a_list,))
    assert avg_time>2.3e-7
    assert avg_time<2.9e-7
    pass

