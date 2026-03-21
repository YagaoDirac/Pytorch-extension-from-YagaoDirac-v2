from typing import Any, Optional, Literal
import random
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import is_square_matrix, _tensor_equal, have_same_elements, \
    get_vector_length, vector_length_norm, iota,\
        str_the_list \

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"




"rand sign"

def rand_sign(size:list[int], dtype = torch.float32, device = 'cpu')->torch.Tensor:
    '''
    >>> for bool: true/false
    >>> for signed int: -1/1
    >>> for unsigned int: 0/1
    >>> for fp : -1./1.
    '''
    if dtype == torch.bool:
        result = torch.randint(low=0, high=2, dtype=torch.int8, size=size, device=device).eq(0)
        return result
    elif dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:#uint, 0 or 1
        result = torch.randint(low=0, high=2, dtype=dtype, size=size, device=device)
        return result
    elif dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:#int, -1 or 1
        result = torch.randint(low=0, high=2, dtype=torch.int8, size=size, device=device)*2-1
        return result.to(dtype)
    else:#float, -1. or 1.
        result = torch.randint(low=0, high=2, dtype=torch.int8, size=size, device=device)*2-1
        return result.to(dtype)
    pass#/function
if "test" and __DEBUG_ME__() and False:
    def ____test____rand_sign():
        size = torch.Size([1,2,3,4,5])
        result = rand_sign(size=size)
        assert result.shape == size
        assert result.dtype == torch.float32
        assert result.eq(1.).sum() + result.eq(-1.).sum() == 120
        
        result = rand_sign(size=[], dtype=torch.bool)
        assert result.dtype == torch.bool
        assert result.shape == torch.Size([])
        
        result = rand_sign(size=size, dtype=torch.uint16)
        assert result.dtype == torch.uint16
        assert result.eq(1).sum() + result.eq(0).sum() == 120
        
        result = rand_sign(size=size, dtype=torch.int64)
        assert result.dtype == torch.int64
        assert result.eq(1).sum() + result.eq(-1).sum() == 120
        
        return
    ____test____rand_sign()
    pass

"randn safe"

def randn_safe(size:list[int], dtype = torch.float32, device = 'cpu')->torch.Tensor:
    '''torch.randn but safer.
    
    step1 provides 68% as a randn but only the [-1,1] elements.
    step2 minus the abs of elements with abs in (1,2] with 1 and push them back into [-1,1], 13%.
    step3 re random the last 18% elements with a uniform distribution in [-1,1].'''
    result = torch.randn(size=size, dtype=dtype, device=device)
    flag__abs_gt_1 = result.abs().gt(1.)
    #aaa = flag__abs_gt_1.sum()
    result[flag__abs_gt_1] = result[flag__abs_gt_1]-1.
    flag__abs_gt_1 = result.abs().gt(1.)
    #bbb = flag__abs_gt_1.sum()
    _this_shape = result[flag__abs_gt_1].shape
    result[flag__abs_gt_1] = torch.rand(size=_this_shape, dtype=dtype)*2.-1
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____randn_safe():
        size = torch.Size([2,3,4])
        result = randn_safe(size=size)
        assert result.shape == size
        assert result.le(1.).logical_or(result.ge(-1.)).all()
        
        result = randn_safe(size=[100000])
        assert result.le(1.).logical_or(result.ge(-1.)).all()
        
        return
    ____test____randn_safe()
    pass




"random vector"

def random_standard_vector(dim:int, dtype = torch.float32, device='cpu')->torch.Tensor:
    '''generates a random vector with geometric length of 1.
    
    The random number generator is randn(normal distribution), this makes it doesn't need any further 
    randomly rotation to be uniformly on angle, but the clue is only some rough test. If you really
    need the angle to be very well uniformly distributes, you may need to make sure it by yourself.'''
    result = torch.randn(size=[dim], dtype=dtype, device=device).\
                                div(torch.tensor(dim, dtype=torch.float64).sqrt().to(dtype))
    length_sqr = result.dot(result).sum()
    while True:
        if length_sqr>0.001 and length_sqr<1000.:
            break
        #tail
        result = torch.randn(size=[dim], dtype=dtype, device=device).\
                                div(torch.tensor(dim, dtype=torch.float64).sqrt().to(dtype))
        length_sqr = result.dot(result).sum()
        pass
    result = result/(length_sqr.sqrt())
    return result
if "basic test" and __DEBUG_ME__() and False:
    def ____test_____random_standard_vector__basic_test():
        for _ in range(16):
            dim = random.randint(2,300)
            vec = random_standard_vector(dim)
            assert vec.dtype == torch.float32
            assert _tensor_equal(get_vector_length(vec), [1.]) 
            pass
        
        vec = random_standard_vector(5, torch.float64, 'cuda')
        assert vec.dtype == torch.float64
        assert vec.device.type == 'cuda'
        return
    ____test_____random_standard_vector__basic_test()
    pass
if "log10 test    copy pasted from Util.py" and __DEBUG_ME__() and False:
    #result: basically the log10 of a standard vec is -0.5*log10(dim)-0.21
    def ____test____log10_avg_safe____standard_vec():
        import math, random
        from pytorch_yagaodirac_v2.Util import log10_avg_safe
            
        if "measure the log10" and True:
            # output:
            # the_min_gt_this_list =[-1.470, -1.821, -2.285]
            # the_max_lt_this_list =[-1.170, -1.740, -2.266]
            # the_mean_eq_this_list=[-1.275, -1.775, -2.275]
            # epsilon_list       =[ 0.194,  0.046,  0.010]
            # -0.5*log10(dim)-0.275, basically from randn.
            print("the log10")
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            test_time_list = [3000,1000,300]
            #--------------------#--------------------#--------------------
            the_min_gt_this_list =  []#don't modify here.
            the_max_lt_this_list =  []
            the_mean_eq_this_list = []
            epsilon_list =          []
            for inner_iter_count in range(dim_list.__len__()):
                dim = dim_list[inner_iter_count]
                test_time = test_time_list[inner_iter_count]
                print(test_time)
            
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    vec = random_standard_vector(dim=dim)
                    assert _tensor_equal(get_vector_length(vec), [1.])
                    _this_result = log10_avg_safe(vec)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                the_min_gt_this_list.append(the_min.item())
                the_max_lt_this_list.append(the_max.item())
                the_mean_eq_this_list.append(the_mean.item())
                _delta_1 = the_mean - the_min 
                _delta_2 = the_max  - the_mean
                epsilon = max(_delta_1, _delta_2)
                epsilon_list.append(epsilon.item())    
                print(f"dim:{dim}  ///  {the_min:.3f}   {the_max:.3f}   {the_mean:.3f}   ")
                pass# for macro_iter_count
            print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
            print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
            print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
            print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
            pass#/test
        
        return 
    ____test____log10_avg_safe____standard_vec()
    pass

"   a special version for low dim, but more uniformly distributed."
if "algo perf test" and False:
    def ____test____random_standard_vector_slow____basic_also_test():
        # result
        # result_min  = [ 1.000,  1.000,  1.000,  1.000,  1.000,  1.000,  1.000,  1.000,  2.000]
        # result_max  = [ 3.000,  8.000,  15.000,  45.,  107.000,  129.000,  277.,  1048.,  3113.000]
        # result_avg  = [ 1.235,  1.920,  3.670,  6.095,  13.655,  27.690,  59.,  159.6,  399.415]
        # dim_list   = [ 2.000,  3.000,  4.000,  5.000,  6.000,  7.000,  8.000,  9.000,  10.000]
        # log_of_avg = ([0.2111, 0.6523, 1.3002, 1.8075, 2.6141, 3.3211, 4.0775, 5.0727, 5.9900])
        
        # result_avg  = torch.tensor([ 1.235,  1.920,  3.670,  6.095,  13.655,  27.690,  59.,  159.6,  399.415])
        # print(result_avg.log())
        
        result_min = []#dont modify this
        result_max = []#dont modify this
        result_avg = []#dont modify this
        #-----------------#-----------------#-----------------
        dim_list =         [2,3,4,   5,6,7,8,9,   10,]#,  50, 100]
        #test_time_list = [100,100,100,100,]
        for inner_param_set in range(dim_list.__len__()):
            dim = dim_list[inner_param_set]
            #test_time = test_time_list[////]
            test_time = 200
            threshold_len = 0.1
        #-----------------#-----------------#-----------------
            print(test_time)
            _result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                
                total_times = 0
                while True:
                    total_times +=1#tail.
                    
                    vec = torch.rand(size=[dim])#no sign at the moment.
                    the_len = get_vector_length(vec)
                    if the_len>threshold_len and the_len<1.:
                        break
                    #tail uppon.
                    pass#while
                _result[test_count] = total_times
                pass#for test_count
                
            result_min.append(_result.min())
            result_max.append(_result.max())
            result_avg.append(_result.mean())
            pass#for inner_param
        print(f"result_min  = {str_the_list(result_min, 3)}")
        print(f"result_max  = {str_the_list(result_max, 3)}")
        print(f"result_avg  = {str_the_list(result_avg, 3)}")
        print(f"dim_list   = {str_the_list(dim_list, 3)}")
        
        return 
    
    ____test____random_standard_vector_slow____basic_also_test()
    pass

def random_standard_vector__low_dim(dim:int, threshold_len = 0.1, dtype = torch.float32, device='cpu')->torch.Tensor:
    '''generates a random vector with geometric length of 1.
    
    This version is slower but more uniformly. It's a try and retry method.
    
    The random number generator is randn(normal distribution), this makes it doesn't need any further 
    randomly rotation to be uniformly on angle, but the clue is only some rough test. If you really
    need the angle to be very well uniformly distributes, you may need to make sure it by yourself.'''
    
    assert threshold_len>0.
    assert threshold_len<0.5#if you know what you are doing, this can go up to 1.
    
    assert dim <=8, "the test time is basiclly e to the power of dim*0.8."
    
    while True:
        vec = torch.rand(size=[dim], dtype = dtype, device=device)#no sign at the moment.
        the_len = get_vector_length(vec)
        if the_len>threshold_len and the_len<1.:
            normalized_vec = vector_length_norm(vec.reshape([1,-1])).reshape([-1])
            return normalized_vec*rand_sign(size=[dim], dtype = dtype, device=device)
        pass
    pass#function.

if "basic test" and __DEBUG_ME__() and False: 
    def ____test_____random_standard_vector_slow__basic_test():
        for _ in range(16):
            dim = random.randint(2,5)
            vec = random_standard_vector__low_dim(dim)
            assert vec.dtype == torch.float32
            assert _tensor_equal(get_vector_length(vec), [1.]) 
            pass
        
        vec = random_standard_vector__low_dim(5, torch.float64, 'cuda')
        assert vec.dtype == torch.float64
        assert vec.device.type == 'cuda'
        return
    ____test_____random_standard_vector_slow__basic_test()
    pass


# This is the third version. But the total_time test result is slightly worse than the "low dim version."
# so, no further work will be done for this version.
# 然后长度太小的那个到底需要不需要保护。
# 最后和之前的纯反复尝试的版本比对一下数字的分布。
if "algo perf test" and False:
    import math
    def ____test____random_standard_vector_v3____perf_test():
    # result
    # result_min  = [ 1.000,  1.000,  1.000,  1.000,  1.000,  1.000,  1.000,   2.000,   2.000]
    # result_max  = [ 6.000,  9.000,  9.000,  40.000,  49.,  178.,    313.000,  1040.,  2548.000]
    # result_avg  = [ 1.345,  1.915,  2.925,  5.775,  12.35,  27.585,  64.99,  146. ,   451.055]
    # dim_list    = [ 2.000,  3.000,  4.000,  5.000,  6.000,  7.000,  8.000,  9.000,    10.000]
    
        result_min = []#dont modify this
        result_max = []#dont modify this
        result_avg = []#dont modify this
        #-----------------#-----------------#-----------------
        dim_list =         [2,3,4,   5,6,7,8,9,   10,]#,  50, 100]
        #test_time_list = [100,100,100,100,]
        for inner_param_set in range(dim_list.__len__()):
            dim = dim_list[inner_param_set]
            #test_time = test_time_list[////]
            test_time = 200
            threshold_len = 0.1
        #-----------------#-----------------#-----------------
            print(test_time)
            _result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                
                _1_over_sqrt_of_dim = math.sqrt(1./dim)
                assert _tensor_equal([_1_over_sqrt_of_dim*_1_over_sqrt_of_dim], [1/dim])
                threshold_len = 0.1
                threshold_per_dim = threshold_len * _1_over_sqrt_of_dim
                assert _tensor_equal([threshold_per_dim*threshold_per_dim], [threshold_len*threshold_len/dim])
                
                total_times = 0
                
                while True:
                    total_times +=1#tail.
                    #-----------------#-----------------#-----------------
                    
                    vec = torch.rand(size=[dim])#, dtype = dtype, device=device)#no sign at the moment.
                    the_len = get_vector_length(vec)
                    if the_len >= 1.:
                        flag__too_large = vec.ge(_1_over_sqrt_of_dim)
                        len_of_flag = flag__too_large.sum()
                        vec[flag__too_large] = torch.rand(size=[len_of_flag])
                        continue
                    if the_len <= threshold_len:
                        flag__too_small = vec.le(threshold_per_dim)
                        len_of_flag = flag__too_small.sum()
                        vec[flag__too_small] = torch.rand(size=[len_of_flag])
                        continue
                    
                    break
                    #-----------------#-----------------#-----------------
                
                _result[test_count] = total_times
                pass#for test_count
                
            result_min.append(_result.min())
            result_max.append(_result.max())
            result_avg.append(_result.mean())
            pass#for inner_param
        print(f"result_min  = {str_the_list(result_min, 3)}")
        print(f"result_max  = {str_the_list(result_max, 3)}")
        print(f"result_avg  = {str_the_list(result_avg, 3)}")
        print(f"dim_list   = {str_the_list(dim_list, 3)}")
        
        return 
    ____test____random_standard_vector_v3____perf_test()
    pass
def ____random_standard_vector__v3____UNFINISHED(dim :int, threshold_len = 0.1):
    assert False, "继续"
    import math
    
    _1_over_sqrt_of_dim = math.sqrt(1./dim)
    threshold_per_dim = threshold_len * _1_over_sqrt_of_dim
    
    while True:
        vec = torch.rand(size=[dim])#, dtype = dtype, device=device)#no sign at the moment.
        the_len = get_vector_length(vec)
        if the_len >= 1.:
            flag__too_large = vec.ge(_1_over_sqrt_of_dim)
            len_of_flag = flag__too_large.sum()
            vec[flag__too_large] = torch.rand(size=[len_of_flag])
            continue
        if the_len <= threshold_len:
            flag__too_small = vec.le(threshold_per_dim)
            len_of_flag = flag__too_small.sum()
            vec[flag__too_small] = torch.rand(size=[len_of_flag])
            continue
        break
    
    normalized_vec = vector_length_norm(vec.reshape([1,-1])).reshape([-1])
    return normalized_vec*rand_sign(size=[dim])#, dtype = dtype, device=device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def ____test____random_standard_vector_slow____basic_also_test():
        # result
        
        result_min = []#dont modify this
        result_max = []#dont modify this
        result_avg = []#dont modify this
        #-----------------#-----------------#-----------------
        dim_list =         [2,3,4,   5,6,7,8,9,   10,]#,  50, 100]
        #test_time_list = [100,100,100,100,]
        for inner_param_set in range(dim_list.__len__()):
            dim = dim_list[inner_param_set]
            #test_time = test_time_list[////]
            test_time = 200
            threshold_len = 0.1
        #-----------------#-----------------#-----------------
            print(test_time)
            _result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                
                total_times = 0
                while True:
                    total_times +=1#tail.
                    
                    vec = torch.rand(size=[dim])#no sign at the moment.
                    the_len = get_vector_length(vec)
                    if the_len>threshold_len and the_len<1.:
                        break
                    #tail uppon.
                    pass#while
                _result[test_count] = total_times
                pass#for test_count
                
            result_min.append(_result.min())
            result_max.append(_result.max())
            result_avg.append(_result.mean())
            pass#for inner_param
        print(f"result_min  = {str_the_list(result_min, 3)}")
        print(f"result_max  = {str_the_list(result_max, 3)}")
        print(f"result_avg  = {str_the_list(result_avg, 3)}")
        print(f"dim_list   = {str_the_list(dim_list, 3)}")
        
        return 
    
    ____test____random_standard_vector_slow____basic_also_test()
    pass

if "I also found some explaination from 3b1b" and False:
    '''
    Basically, the volume of a ball in n-dimention, over the bounding box, is getting smaller
    at a significant speed when the n is big enough. 
    This makes it's very unlikely to get a set of random(torch.rand) numbers and the square sum is less than 1.
    Remember, "Your balls are just puny" by 3b1b.
    https://youtu.be/fsLh-NYhOoU?t=2979
    '''
    pass


if "maybe it's better to do it outside???" and False:

    def random_standard_vector__len_in_range(dim:int, min_length:float|torch.Tensor, dtype = torch.float32, device='cpu')->torch.Tensor:
        '''generates a random vector with geometric length between min_length and 1.
        
        The random number generator for vector is randn(normal distribution) (rand for length), 
        this makes it doesn't need any further 
        randomly rotation to be uniformly on angle, but the clue is only some rough test. If you really
        need the angle to be very well uniformly distributes, you may need to make sure it by yourself.'''
        
        assert min_length>0.01 and min_length<1.
        
        result = torch.randn(size=[dim], dtype=dtype, device=device).\
                                    div(torch.tensor(dim, dtype=torch.float64).sqrt().to(dtype))
        length_sqr = result.dot(result).sum()
        while True:
            if length_sqr>0.001 and length_sqr<1000.:
                break
            #tail
            result = torch.randn(size=[dim], dtype=dtype, device=device).\
                                    div(torch.tensor(dim, dtype=torch.float64).sqrt().to(dtype))
            length_sqr = result.dot(result).sum()
            pass
        
        target_length = torch.rand(size=[])*(1-min_length)+min_length#cpu, min_length to 1.
        mul_this = target_length/(length_sqr.sqrt())
        
        result = result*mul_this
        return result
    if "basic test" and __DEBUG_ME__() and True:
        def ____test_____random_standard_vector__len_in_range():
            for _ in range(16):
                dim = random.randint(2,300)
                min_len = random.random()*0.9+0.1
                vec = random_standard_vector__len_in_range(dim, min_length=min_len)
                assert vec.dtype == torch.float32
                assert get_vector_length(vec)>=min_len
                assert get_vector_length(vec)<=1.
                pass
            
            vec = random_standard_vector__len_in_range(5, 0.1, dtype = torch.float64, device = 'cuda')
            assert vec.dtype == torch.float64
            assert vec.device.type == 'cuda'
            return
        ____test_____random_standard_vector__len_in_range()
        pass
    if "log10 test    copy pasted from Util.py" and __DEBUG_ME__() and False:
        assert False, "unfinished."
        
        #result: basically the log10 of a standard vec is -0.5*log10(dim)-0.21
        def ____test____log10_avg_safe____standard_vec():
            import math, random
            from pytorch_yagaodirac_v2.Util import log10_avg_safe
                
            if "measure the log10" and True:
                # output:
                # the_min_gt_this_list =[-1.470, -1.821, -2.285]
                # the_max_lt_this_list =[-1.170, -1.740, -2.266]
                # the_mean_eq_this_list=[-1.275, -1.775, -2.275]
                # epsilon_list       =[ 0.194,  0.046,  0.010]
                # -0.5*log10(dim)-0.275, basically from randn.
                print("the log10")
                device = 'cuda'
                #--------------------#--------------------#--------------------
                dim_list = [100,1000,10000]
                test_time_list = [3000,1000,300]
                #--------------------#--------------------#--------------------
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                for inner_iter_count in range(dim_list.__len__()):
                    dim = dim_list[inner_iter_count]
                    test_time = test_time_list[inner_iter_count]
                    print(test_time)
                
                    _raw_result = torch.empty(size=[test_time])
                    for test_count in range(test_time):
                        #--------------------#--------------------#--------------------
                        vec = random_standard_vector(dim=dim)
                        assert _tensor_equal(get_vector_length(vec), [1.])
                        _this_result = log10_avg_safe(vec)
                        #--------------------#--------------------#--------------------
                        _raw_result[test_count] = _this_result
                        pass
                    the_min = _raw_result.min()
                    the_max = _raw_result.max()
                    the_mean = _raw_result.mean()
                    the_min_gt_this_list.append(the_min.item())
                    the_max_lt_this_list.append(the_max.item())
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min 
                    _delta_2 = the_max  - the_mean
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"dim:{dim}  ///  {the_min:.3f}   {the_max:.3f}   {the_mean:.3f}   ")
                    pass# for macro_iter_count
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass#/test
            
            return 
        ____test____log10_avg_safe____standard_vec()
        pass

    pass








"not very important."

def _get_2_diff_rand_int(dim:int, device , devide_in_this_function = 'cpu')->tuple[torch.Tensor,torch.Tensor]:
    '''return index_tensor.
    
    shape:torch.Size([2])
    
    The 2 elements are different.'''
    assert dim>=2
    rand_index_1 = torch.randint(0, dim,   size = [], device = devide_in_this_function)
    rand_index_2 = torch.randint(0, dim-1, size = [], device = devide_in_this_function)
    if rand_index_2>=rand_index_1:
        rand_index_2+=1
        pass
    stacked = torch.stack([rand_index_1,rand_index_2])
    stacked = stacked.to(device=device)
    return stacked#, rand_index_1, rand_index_2
if "test" and __DEBUG_ME__() and False:
    def ____test_____get_2_diff_rand_int():
        for _ in range(155):
            dim = random.randint(2,300)
            #-------------------#-------------------#-------------------
            index_tensor = _get_2_diff_rand_int(dim, 'cpu')
            #-------------------#-------------------#-------------------
            assert index_tensor.dtype == torch.int64
            assert index_tensor.shape == torch.Size([2])
            assert index_tensor[0]!=index_tensor[1]
            assert index_tensor.ge(0).all()
            assert index_tensor.lt(dim).all()
            pass
        return 
    ____test_____get_2_diff_rand_int()




"rotation mat 2d"

def angle_to_rotation_matrix_2d(angle:float|torch.Tensor)->torch.Tensor:
    '''maybe it's a good idea to do this purely on cpu??'''
    if isinstance(angle, float):
        angle = torch.tensor(angle)
        pass
    _cos = angle.cos()
    _sin = angle.sin()
    # result = torch.tensor( [[ _cos, _sin],
    #                         [-_sin, _cos]], device=angle.device)
    
    temp_row0 = torch.stack([_cos, _sin])
    temp_row1 = torch.stack([-_sin, _cos])
    result = torch.stack([temp_row0, temp_row1])
    return result
if "basic behavior and device adaption." and __DEBUG_ME__() and False:
    def ____test_____angle_to_rotation_matrix_2d():
        import math
        mat = angle_to_rotation_matrix_2d(0.1)
        assert _tensor_equal(mat,  [[ math.cos(0.1),math.sin(0.1)],
                                    [-math.sin(0.1),math.cos(0.1)]])
        mat = angle_to_rotation_matrix_2d(torch.tensor(torch.pi/6., device='cuda'))
        assert _tensor_equal(mat,  torch.tensor([[ math.sqrt(3.)/2,            0.5],
                                                        [   -0.5,     math.sqrt(3.)/2]], device='cuda'))
        assert mat.device.type == 'cuda'
        return
    ____test_____angle_to_rotation_matrix_2d()
    pass




"   random rotation"
# a detail about the calculating precision. 
# Check out the test in 'so called pre rotation'. I tested a lil bit. Usually this is not a problem.
# but if anything is rotated for like dim*100 times and it needs to be exactly the same length, protect it at last.

if "random_rotate algo test" and __DEBUG_ME__() and False:
    def ____test____random_rotate_algo_test():
        "column"
        dim = 3
        input = torch.linspace(0,dim*dim-1,dim*dim).reshape([dim,dim])
        #index_tensor ,_,_ = _get_2_diff_rand_int(dim,'cpu')
        index_tensor = torch.tensor([0,1])
        #angle = torch.rand(size=[])*torch.pi*2.
        angle = torch.tensor(torch.pi/6.)

        #写一个专用的旋转矩阵的函数。rotate2d一类的。

        the_only_modified_part = input.index_select(dim=1,index=index_tensor)#new tensor.
        assert the_only_modified_part.eq(torch.tensor([[0,1],[3,4],[6,7]])).all()
        the_only_modified_part = the_only_modified_part@angle_to_rotation_matrix_2d(angle)
        assert _tensor_equal(the_only_modified_part[0], torch.tensor([0.,1])@angle_to_rotation_matrix_2d(angle))
        input[:,index_tensor[0]] = the_only_modified_part[:,0]
        input[:,index_tensor[1]] = the_only_modified_part[:,1]
        
        "row"
        dim = 3
        input = torch.linspace(0,dim*dim-1,dim*dim).reshape([dim,dim])
        #index_tensor ,_,_ = _get_2_diff_rand_int(dim,'cpu')
        index_tensor = torch.tensor([0,1])
        #angle = torch.rand(size=[])*torch.pi*2.
        angle = torch.tensor(torch.pi/6.)

        #写一个专用的旋转矩阵的函数。rotate2d一类的。

        the_only_modified_part = input.index_select(dim=0,index=index_tensor)#new tensor.
        assert the_only_modified_part.eq(torch.tensor([[0,1,2],[3,4,5]])).all()
        the_only_modified_part = angle_to_rotation_matrix_2d(angle)@the_only_modified_part
        assert _tensor_equal(the_only_modified_part[:,0], (angle_to_rotation_matrix_2d(angle)@torch.tensor([[0.],[3]])).reshape([-1]))
        input[index_tensor[0]] = the_only_modified_part[0]
        input[index_tensor[1]] = the_only_modified_part[1]

        return 

    ____test____random_rotate_algo_test()
    
    pass
def randomly_rotate__matrix(input:torch.Tensor, times:int|None = None)->torch.Tensor:
    '''randomly rotates the input'''
    
    assert is_square_matrix(input)
    dim = input.shape[0]
    the_device = input.device
    the_dtype = input.dtype
    
    if times is None:
        times = dim*3
        pass
    
    with torch.no_grad():
        times_by_row = torch.randint(0,times+1,size=[])
        times_by_column = times-times_by_row
        for _ in range(times_by_row):# each ROW vec into a new rotated ROW vec
            index_tensor = _get_2_diff_rand_int(dim, device=the_device)
            angle = torch.rand(size=[], dtype = the_dtype,device=the_device)*torch.pi*2.
            the_only_modified_part = input.index_select(dim=1,index=index_tensor)#new tensor.
            the_only_modified_part = the_only_modified_part@angle_to_rotation_matrix_2d(angle)
            input[:,index_tensor[0]] = the_only_modified_part[:,0]
            input[:,index_tensor[1]] = the_only_modified_part[:,1]
            pass
        
        for _ in range(times_by_column):# each COLUMN vec into a new rotated COLUMN vec
            index_tensor  = _get_2_diff_rand_int(dim, device=the_device)
            angle = torch.rand(size=[], dtype = the_dtype,device=the_device)*torch.pi*2.
            the_only_modified_part = input.index_select(dim=0,index=index_tensor)#new tensor.
            the_only_modified_part = angle_to_rotation_matrix_2d(angle)@the_only_modified_part
            input[index_tensor[0]] = the_only_modified_part[0]
            input[index_tensor[1]] = the_only_modified_part[1]
            pass
        return input
    pass#/function
if "test" and __DEBUG_ME__() and False:
    def ____test____random_rotate():
        for _ in range(1):
            dim = random.randint(2,300)
            #-------------#-------------#-------------
            _init_eye = torch.eye(n = dim)
            mat = randomly_rotate__matrix(_init_eye)
            #-------------#-------------#-------------
            vec = random_standard_vector(dim)
            vec = vec@mat
            assert _tensor_equal(get_vector_length(vec), [1.])
            pass
    
        return
    
    ____test____random_rotate()
    pass

def randomly_rotate__vector(input:torch.Tensor, times:int|None = None)->torch.Tensor:
    '''randomly rotates the input'''
    
    assert input.shape.__len__() == 1, "Batch is not implemented. Maybe later."
    dim = input.shape[0]
    the_device = input.device
    the_dtype = input.dtype
    
    if times is None:
        times = dim*3
        pass
    
    with torch.no_grad():
        for _ in range(times):# each ROW vec into a new rotated ROW vec
            index_tensor = _get_2_diff_rand_int(dim, device=the_device)
            angle = torch.rand(size=[], dtype = the_dtype,device=the_device)*torch.pi*2.
            the_only_modified_part = input.index_select(dim=0,index=index_tensor)#new tensor.
            the_only_modified_part = the_only_modified_part@angle_to_rotation_matrix_2d(angle)
            input[index_tensor[0]] = the_only_modified_part[0]
            input[index_tensor[1]] = the_only_modified_part[1]
            pass
        
        return input
    pass#/function
if "test" and __DEBUG_ME__() and False:
    def ____test____random_rotate_this_vector():
        device = 'cpu'#cpu is faster here.
        dim_list =           [2,    3,   5,  10,100,1000]
        test_time_list = [10000,5000,1000,1000,300,30]
        for inner_param_iter in range(dim_list.__len__()):
            dim = dim_list[inner_param_iter]
            test_time = test_time_list[inner_param_iter]
            print(f"{dim}", end=" ")
            for test_count in range(test_time):
                #-------------#-------------#-------------
                _init_vec = vector_length_norm(torch.randn(size=[1,dim],device=device)).reshape([-1])
                vec = randomly_rotate__vector(_init_vec)
                #-------------#-------------#-------------
                assert _tensor_equal(get_vector_length(vec), torch.tensor([1.],device=device))
            pass
        return
    ____test____random_rotate_this_vector()
    pass

def random_rotation_matrix__by_once(dim:int, dtype = torch.float32, device = None)->torch.Tensor:
    '''Basically it's a eye_tensor, but with some rotation. This rotation only affects 2 dimentions.'''
    
    device_in_this_function = 'cpu'
    index_tensor = _get_2_diff_rand_int(dim, device=device_in_this_function)
    _angle = torch.rand(size=[], dtype = dtype,device=device_in_this_function)*torch.pi*2.
    _cos = _angle.cos()
    _sin = _angle.sin()
    result = torch.eye(n=dim, dtype=dtype, device=device_in_this_function)
    result[index_tensor[0],index_tensor[0]] = _cos
    result[index_tensor[0],index_tensor[1]] = _sin
    result[index_tensor[1],index_tensor[0]] = -_sin
    result[index_tensor[1],index_tensor[1]] = _cos
    result = result.to(device=device)
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____random_rotation_matrix__by_once():
        for _ in range(155):
            dim = random.randint(2,300)
            #-------------#-------------#-------------
            mat = random_rotation_matrix__by_once(dim)
            #-------------#-------------#-------------
            iota_of_dim = iota(dim)
            _flag_is_1 = mat.eq(1.)
            _flag_is_0 = mat.eq(0.)
            _flag_non1_and_non0 = _flag_is_1.logical_not().logical_and(_flag_is_0.logical_not())
            if _flag_non1_and_non0.sum()>0:#non special case.
                #number of non1 non0 elements.
                assert _tensor_equal(_flag_is_1.sum(), [dim-2])#2 ones are modified.
                assert _tensor_equal(_flag_is_0.sum(), [dim*dim-dim-2])#2 ones are modified.
                other_elements = mat[_flag_non1_and_non0]
                assert other_elements[0] == other_elements[3]#cos
                assert other_elements[1] == -other_elements[2]#sin
                #2 cosines equal, 2 sines equal.
                other_elements_abs = other_elements.abs()
                other_elements_abs = other_elements_abs.sort().values
                assert other_elements_abs[0] == other_elements_abs[1]
                assert other_elements_abs[0]<1.
                assert other_elements_abs[2] == other_elements_abs[3]
                assert other_elements_abs[2]<1.
                #number of sign, 3pos 1neg or 3neg 1pos
                other_elements_gt_0 = other_elements.gt(0.)
                other_elements_gt_0__sum = other_elements_gt_0.sum()
                assert other_elements_gt_0__sum == 1 or other_elements_gt_0__sum == 3
                #no 1 outside main diagonal
                _flag_is_1[iota_of_dim,iota_of_dim] = False
                assert _flag_is_1.sum() == 0
                pass
            else:#special case
                #number of 1 0 elements.
                assert _flag_is_1.sum() == dim
                assert _flag_is_0.sum() == dim*dim-dim
                #no 1 outside main diagonal
                _flag_is_1[iota_of_dim,iota_of_dim] = False
                _flag_is_1__sum = _flag_is_1.sum()
                assert _flag_is_1__sum == 0 or _flag_is_1__sum == 2
                pass
            pass
        return 
    ____test____random_rotation_matrix__by_once()

def random_rotation_matrix(dim:int, times:int|None = None, dtype = torch.float32, device = None)->torch.Tensor:
    '''generates a random rotation matrix'''
    
    device_in_this_function = 'cpu'
    init_eye = torch.eye(n=dim, dtype=dtype, device=device_in_this_function)
    result = randomly_rotate__matrix(init_eye,times=times)
    result = result.to(device=device)
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____random_rotation_matrix():
        "bc the inner function is validated, this func doesn't need to get validated again."
        return 
    ____test____random_rotation_matrix()
    pass

if "I tried the so called pre rotation, but I didn't find any clue that it helps at all." and False:
    if "torch.randperm test" and __DEBUG_ME__() and False:
        def torch_randperm_test():
            mat = torch.linspace(1,9,9).reshape([3,3])
            the_rand_index = torch.randperm(3)
            mat2 = mat[the_rand_index]
            mat3 = mat[:,the_rand_index]
            print(mat2)
            print(mat3)
            return 
        torch_randperm_test()
        pass

    if "vec @ rotation_mat length retention test" and __DEBUG_ME__() and False:
        # result. 
        # no halfway protection is needed.
        def vec_matmul_rotation_mat__length_retention_test():
            
            # fp32
            # if a vec[dim] needs at most 10*dim times rotation on a dim_pair, then no halfway length protection is needed if fp32.
            # @ dim 10    cpu , basically 4000   epochs to get the length out of [0.999998, 1.000002]. No protection needed.
            # @ dim 10    cuda, basically 500    epochs to get the length out of [0.999998, 1.000002]. No protection needed.
            # @ dim 100   cpu , basically >100k  epochs to get the length out of [0.999998, 1.000002]. No protection needed.
            # @ dim 100   cuda, basically 5000   epochs to get the length out of [0.999998, 1.000002]. No protection needed.
            # @ dim 1000  cpu , basically 3M     epochs to get the length out of [0.999998, 1.000002]. No protection needed.
            # @ dim 1000  cuda, basically 50k    epochs to get the length out of [0.999998, 1.000002]. No protection needed.
            
            #fp16
            # if a vec[dim] needs at most 10*dim times rotation on a dim_pair, then no halfway length protection is needed if fp16.
            # @ dim 10    cpu , basically 200   epochs to get the length out of [0.998, 1.002].  No protection needed.
            # @ dim 10    cuda, basically 150   epochs to get the length out of [0.998, 1.002].  No protection needed.
            # @ dim 100   cpu , basically 3000  epochs to get the length out of [0.998, 1.002].  No protection needed.
            # @ dim 100   cuda, basically 4k    epochs to get the length out of [0.998, 1.002].  No protection needed.
            # @ dim 1000  cpu , basically 60k   epochs to get the length out of [0.998, 1.002].  No protection needed.
            # @ dim 1000  cuda, basically 160k  epochs to get the length out of [0.998, 1.002].  No protection needed.
            
            
            dim = 1000
            device = 'cpu' 
            device = 'cuda' 
            dtype = torch.float16
            
            same_dim_error:list[int] = []
            too_large:list[int] = []
            too_small:list[int] = []
            
            test_time = 1000000
            for test_count in range(test_time):
                epoch = 1
                vec = random_standard_vector(dim=dim, device = device, dtype=dtype)
                while True:
                    ori_vec = vec.detach().clone()
                    vec = randomly_rotate__vector(vec, times=1)
                    assert vec.dtype == torch.float16
                    same_dim_count = ori_vec.eq(vec).sum()
                    
                    length_sqr = vec.dot(vec).sum()#-0.001
                    #if length_sqr>1.000002:
                    if length_sqr>1.002:
                        print(f"length too large   {epoch}")
                        too_large.append(epoch)
                        break
                    #if length_sqr<0.999998:
                    if length_sqr<0.998:
                        print(f"length too small   {epoch}")
                        too_small.append(epoch)
                        break
                    # if same_dim_count != dim-2:
                    #     print(f"same dim not ok   {epoch}    {same_dim_count}")
                    #     same_dim_error.append(epoch)
                    #     break
                    #tail
                    epoch += 1
                    pass#while true
                pass#for epoch
            
            return 
        
        vec_matmul_rotation_mat__length_retention_test()
                
        pass

    if "rotation algo performance test" and __DEBUG_ME__() and False:
        # result
        # cpu is always much faster(10x).
        def rotation_algo_performance_test():
            
            
            
            
            from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
            
            time_at_most = 2.
            # #--------------------#--------------------#--------------------
            # dim_list =  [10, 100, 1000] 
            # rc_list = [1000, 100, 10]
            # for inner_param_count in range(dim_list.__len__()):
            #     dim = dim_list[inner_param_count]
            #     rc = rc_list[inner_param_count]
            
            # result:
            # dim 10    rc 1000    vvvvvvvvvvvvvvvvvvv
            # fp32: 1.890 c/g 10.133
            # fp16: 1.827 c/g 11.737
            # dim 100    rc 100    vvvvvvvvvvvvvvvvvvv
            # fp32: 1.720 c/g 9.9673
            # fp16: 1.712 c/g 10.566
            # dim 1000    rc 10    vvvvvvvvvvvvvvvvvvv
            # fp32: 1.748 c/g 10.011
            # fp16: 1.719 c/g 10.175
            # #--------------------#--------------------#--------------------
            #--------------------#--------------------#--------------------
            dim = 100   
            
            rc_list = [10, 100, 1000, 10000] 
            for inner_param_count in range(rc_list.__len__()):
                rc = rc_list[inner_param_count]
                
            # dim 100    rc 10    vvvvvvvvvvvvvvvvvvv
            # fp32: 0.200 c/g 1.348
            # fp16: 0.186 c/g 1.414
            # dim 100    rc 100    vvvvvvvvvvvvvvvvvvv
            # fp32: 1.871 c/g 11.338
            # fp16: 1.876 c/g 12.902
            # dim 100    rc 1000    vvvvvvvvvvvvvvvvvvv
            # fp32: 18.196
            # fp16: 18.087
            #--------------------#--------------------#--------------------
                def _func__empty():
                    vec = random_standard_vector(dim=dim)
                    for _ in range(rc):
                        pass
                    return
                empty_time, _ = timeit(_func__empty, time_at_most=time_at_most)
                
                def _func__cpu_32():
                    vec = random_standard_vector(dim=dim, dtype = torch.float32, device = 'cpu')
                    for _ in range(rc):
                        vec = randomly_rotate__vector(vec)
                        pass
                    return
                _cpu_32_time, _ = timeit(_func__cpu_32, time_at_most=time_at_most)
                _cpu_32_time__corrected = _cpu_32_time - empty_time
                
                def _func__cuda_32():
                    vec = random_standard_vector(dim=dim, dtype = torch.float32, device = 'cuda')
                    for _ in range(rc):
                        vec = randomly_rotate__vector(vec)
                        pass
                    return
                _cuda_32_time, _ = timeit(_func__cuda_32, time_at_most=time_at_most)
                _cuda_32_time__corrected = _cuda_32_time - empty_time

                def _func__cpu_16():
                    vec = random_standard_vector(dim=dim, dtype = torch.float16, device = 'cpu')
                    for _ in range(rc):
                        vec = randomly_rotate__vector(vec)
                        pass
                    return
                _cpu_16_time, _ = timeit(_func__cpu_16, time_at_most=time_at_most)
                _cpu_16_time__corrected = _cpu_16_time - empty_time
                
                def _func__cuda_16():
                    vec = random_standard_vector(dim=dim, dtype = torch.float16, device = 'cuda')
                    for _ in range(rc):
                        vec = randomly_rotate__vector(vec)
                        pass
                    return
                _cuda_16_time, _ = timeit(_func__cuda_16, time_at_most=time_at_most)
                _cuda_16_time__corrected = _cuda_16_time - empty_time
                
                
                print(f"dim {dim}    rc {rc}    vvvvvvvvvvvvvvvvvvv")
                print(f"fp32: {_cpu_32_time__corrected:.3f} c/g {_cuda_32_time__corrected:.3f} ")
                print(f"fp16: {_cpu_16_time__corrected:.3f} c/g {_cuda_16_time__corrected:.3f} ")
                pass
            
            return 
        
        
        
        rotation_algo_performance_test()
        
        pass

    if "how many rotation is enough" and __DEBUG_ME__() and False:
        #result. I didn't see any difference with or without pre rotation.
        def how_many_rotation_is_enough():
            
            from matplotlib import pyplot as plt
            
            dim = 1000
            
            vec = random_standard_vector(dim=dim, dtype = torch.float32, device = 'cpu')
            for _ in range(100):
                #N_points = 100000
                n_bins = 20

                fig, axs = plt.subplots(1, 1, tight_layout=True)

                # We can set the number of bins with the *bins* keyword argument.
                axs.hist(vec.abs().tolist(), bins=n_bins)
                #axs.hist(_dummy_layer.weight.tolist(), bins=n_bins)

                plt.show()
                                    
                
                vec = randomly_rotate__vector(vec)
                pass
            
            
            
            ##################################
            ##################################
            #part 1 with matplotlib
            #part 2 with log10_avg_safe
            ##################################
            ##################################
            
            
            
            from pytorch_yagaodirac_v2.Util import log10_avg_safe, str_the_list
            
            
            # dim 10   vvvvvvvvvvvvvvvvvvv
            # result_max_list   = [-0.521, -0.571, -0.545, -0.545, -0.572, -0.560, -0.556, -0.535, -0.555, -0.548, -0.544, -0.571, -0.544, -0.554]
            # result_min_list   = [-1.042, -0.999, -1.182, -1.140, -1.139, -1.070, -1.100, -1.115, -1.171, -1.120, -1.205, -1.022, -1.281, -1.078]
            # result_avg_list   = [-0.769, -0.753, -0.768, -0.761, -0.763, -0.772, -0.783, -0.761, -0.764, -0.759, -0.776, -0.759, -0.764, -0.765]
            # result_delta_list = [ 0.273,  0.246,  0.414,  0.379,  0.375,  0.298,  0.317,  0.354,  0.407,  0.361,  0.429,  0.263,  0.517,  0.313]
            # rc_list           = [ 2.000,  3.000,  6.000,  10.,  15.000,  25.000,  39.000,  63.,  100.000,  158.,  251.000,  398.,  630.000,  1000]
            # dim 100   vvvvvvvvvvvvvvvvvvv
            # debug info   = 9
            # result_max_list   = [-1.226, -1.219, -1.227, -1.225, -1.226, -1.241, -1.225, -1.194, -1.216]
            # result_min_list   = [-1.352, -1.334, -1.302, -1.349, -1.366, -1.354, -1.309, -1.372, -1.309]
            # result_avg_list   = [-1.277, -1.279, -1.272, -1.295, -1.273, -1.287, -1.276, -1.270, -1.268]
            # result_delta_list = [ 0.075,  0.060,  0.045,  0.070,  0.093,  0.066,  0.051,  0.102,  0.052]
            # rc_list           = [ 2.000,  3.000,  6.000,  10.000,  15.000,  25.000,  39.000,  63.000,  100.000]
            # dim 1000   vvvvvvvvvvvvvvvvvvv
            # debug info   = 9
            # result_max_list   = [-1.763, -1.767, -1.763, -1.777, -1.761, -1.786, -1.784, -1.762, -1.768]
            # result_min_list   = [-1.771, -1.785, -1.767, -1.784, -1.773, -1.788, -1.809, -1.771, -1.774]
            # result_avg_list   = [-1.767, -1.776, -1.765, -1.781, -1.767, -1.787, -1.796, -1.767, -1.771]
            # result_delta_list = [ 0.004,  0.009,  0.002,  0.004,  0.006,  0.001,  0.012,  0.004,  0.003]
            # rc_list           = [ 2.000,  3.000,  6.000,  10.000,  15.000,  25.000,  39.000,  63.000,  100.000]
            
            #--------------------#--------------------#--------------------
            dim_list = [10, 100, 1000]
            dim_list = [100, 1000]
            test_time_list = [30,7,2]
            test_time_list = [7,2]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
                print(test_time)
            #--------------------#--------------------#--------------------
                result_max_list = []  #dont modify this
                result_min_list = []  #dont modify this
                result_avg_list = []  #dont modify this
                result_delta_list = []#dont modify this
                
                #--------------------#--------------------#--------------------
                rc_list = torch.pow(10,torch.linspace(0.4,2.,9)).to(torch.int32).tolist()
                #rc_list = [1,2,5]
                for inner_param_count in range(rc_list.__len__()):
                    rc = rc_list[inner_param_count]
                #--------------------#--------------------#--------------------
                    
                    _temp_result = torch.empty(size=[test_time])
                    for test_count in range(test_time):
                        #--------------------#--------------------#--------------------
                        vec = random_standard_vector(dim=dim, dtype = torch.float32, device = 'cpu')
                        for _ in range(rc):
                            vec = randomly_rotate__vector(vec)
                            pass
                        the_length = get_vector_length(vec)
                        assert the_length<1.00001
                        assert the_length>0.99999
                        _this_result = log10_avg_safe(vec)
                        #--------------------#--------------------#--------------------
                        _temp_result[test_count] = _this_result
                        pass# for test_count
                    __the_max = _temp_result.max()
                    __the_min = _temp_result.min()
                    __the_avg = _temp_result.mean()
                    result_max_list.append(__the_max)
                    result_min_list.append(__the_min)
                    result_avg_list.append(__the_avg)
                    __upper_delta = __the_max-__the_avg
                    __lower_delta = __the_avg-__the_min
                    __final_delta = max(__upper_delta, __lower_delta)
                    result_delta_list.append(__final_delta)
                    print(f"dim {dim}  rc {rc}  //  {__the_max.item():.3f}   {__the_min.item():.3f}   {__the_avg.item():.3f}   ")
                    pass# for inner_param
                print(f"dim {dim}   vvvvvvvvvvvvvvvvvvv")
                print(f"debug info   = {result_max_list.__len__()}")
                print(f"result_max_list   = {str_the_list(result_max_list, 3)}")
                print(f"result_min_list   = {str_the_list(result_min_list, 3)}")
                print(f"result_avg_list   = {str_the_list(result_avg_list, 3)}")
                print(f"result_delta_list = {str_the_list(result_delta_list, 3)}")
                print(f"rc_list           = {str_the_list(rc_list, 3)}")
                pass#for outter_param    
            
            
            return 
        how_many_rotation_is_enough()
        pass


    def random_standard_vector__pre_rotated(dim:int, rotate_count:int|None = None, 
                                            #_protect_length_every = 30, 
                                            shuffle = True, 
                                            dtype = torch.float32, device='cpu',
                                            device_in_this_func = 'cpu',
                                            _debug__no_final_protection = False)->torch.Tensor:
        '''generates a random rotation matrix.'''
        if rotate_count is None:
            rotate_count = int((dim+10)*1.2)
            pass
        assert rotate_count>=0
        
        # bc in my case, 
        vec = random_standard_vector(dim=dim, dtype = dtype, device = device_in_this_func)
        
        # new vvvvvvvvvv
        iter_count = 0
        for _ in range(rotate_count):
            vec = randomly_rotate__vector(vec)
            
            # #<  length protection>
            # iter_count +=1
            # if iter_count>=_protect_length_every:
            #     iter_count = 0
            #     vec = vector_length_norm(vec.reshape([1,-1])).reshape([-1])
            #     pass
            # #</ length protection>
            pass
        
        #<  before return/>
        
        assert False, "unfinished."
        if shuffle:
            torch.randperm()
            torch.randro()
            vec.s
        
        if not _debug__no_final_protection:
            vec = vector_length_norm(vec.reshape([1,-1])).reshape([-1])
            pass
        # new ^^^^^^^^^^^
        vec = vec.to(device = device)
        return vec
    if "basic test" and False:
        def ____test____random_standard_vector__pre_rotated(): 
            
            if "length accuracy test  if without the last protection before return ." and True: 
                if "result":
                    "dim 2"
                    pass# if result.
                # output:
                print("length accuracy test")
                device = 'cpu'
                #--------------------#--------------------#--------------------
                dim_list =          [2,  3,  5, 10, 100,1000]
                test_time_list = [1000,500,100,100, 30, 3]
                for outter_iter_count in range(dim_list.__len__()):
                    dim = dim_list[outter_iter_count]
                    test_time = test_time_list[outter_iter_count]
                    print(test_time)
                #--------------------#--------------------#--------------------
                    the_min_gt_this_list =  []#don't modify here.
                    the_max_lt_this_list =  []#don't modify here.
                    the_mean_eq_this_list = []#don't modify here.
                    epsilon_list =          []#don't modify here.
                    #--------------------#--------------------#--------------------
                    rc_list = [0,1,2,3,5,10,100]
                    for inner_iter_count in range(rc_list.__len__()):
                        rc = rc_list[inner_iter_count]
                    #--------------------#--------------------#--------------------
                    
                        _raw_result = torch.empty(size=[test_time])
                        for test_count in range(test_time):
                            #--------------------#--------------------#--------------------
                            vec = random_standard_vector__pre_rotated(dim=dim,rotate_count=rc, 
                                            _protect_length_every = rc+100, _debug__no_final_protection = True)
                            _this_result = get_vector_length(vec)
                            #--------------------#--------------------#--------------------
                            _raw_result[test_count] = _this_result
                            pass
                        the_min = _raw_result.min()
                        the_max = _raw_result.max()
                        the_mean = _raw_result.mean()
                        the_min_gt_this_list.append(the_min.item())
                        the_max_lt_this_list.append(the_max.item())
                        the_mean_eq_this_list.append(the_mean.item())
                        _delta_1 = the_mean - the_min 
                        _delta_2 = the_max  - the_mean
                        epsilon = max(_delta_1, _delta_2)
                        epsilon_list.append(epsilon.item())    
                        print(f"dim:{dim}  ///  {the_min:.3f}   {the_max:.3f}   {the_mean:.3f}   ")
                        pass# for macro_iter_count
                    print(f"if dim == {dim}:")
                    print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                    print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                    print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                    print(f"epsilon_list         ={str_the_list(epsilon_list, 3)}")    
                    print(f"#rc_list             ={str_the_list(rc_list,      3)}")    
                    print("pass")
                    
                    pass#for outter param
                pass#/test
                
                
                
            if "protection per ???" and True: 
                # output:
                print("protection per ???")
                device = 'cpu'
                #--------------------#--------------------#--------------------
                dim_list =          [2,  3,  5, 10, 100,1000]
                test_time_list = [1000,500,100,100, 30, 3]
                for outter_iter_count in range(dim_list.__len__()):
                    dim = dim_list[outter_iter_count]
                    test_time = test_time_list[outter_iter_count]
                    print(test_time)
                #--------------------#--------------------#--------------------
                    the_min_gt_this_list =  []#don't modify here.
                    the_max_lt_this_list =  []#don't modify here.
                    the_mean_eq_this_list = []#don't modify here.
                    epsilon_list =          []#don't modify here.
                    #--------------------#--------------------#--------------------
                    rc_list = [0,1,2,3,5,10,100]
                    for inner_iter_count in range(rc_list.__len__()):
                        rc = rc_list[inner_iter_count]
                    #--------------------#--------------------#--------------------
                    
                        _raw_result = torch.empty(size=[test_time])
                        for test_count in range(test_time):
                            #--------------------#--------------------#--------------------
                            #set the _protect_length_every to large enough to prevent the protection.
                            vec = random_standard_vector__pre_rotated(dim=dim,rotate_count=rc, _protect_length_every = rc+100)
                            _this_result = get_vector_length(vec)
                            #--------------------#--------------------#--------------------
                            _raw_result[test_count] = _this_result
                            pass
                        the_min = _raw_result.min()
                        the_max = _raw_result.max()
                        the_mean = _raw_result.mean()
                        the_min_gt_this_list.append(the_min.item())
                        the_max_lt_this_list.append(the_max.item())
                        the_mean_eq_this_list.append(the_mean.item())
                        _delta_1 = the_mean - the_min 
                        _delta_2 = the_max  - the_mean
                        epsilon = max(_delta_1, _delta_2)
                        epsilon_list.append(epsilon.item())    
                        print(f"dim:{dim}  ///  {the_min:.3f}   {the_max:.3f}   {the_mean:.3f}   ")
                        pass# for macro_iter_count
                    print(f"if dim == {dim}")
                    print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                    print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                    print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                    print(f"epsilon_list         ={str_the_list(epsilon_list, 3)}")    
                    print(f"#rc_list             ={str_the_list(rc_list,      3)}")    
                    print("pass")
                    
                    pass#for outter param
                pass#/test
                
            return 
        
        ____test____random_standard_vector__pre_rotated()
        pass
    pass





"   random permutate"

if "when I made this, I didn't know the torch.randperm function. Now I dont need this." and False:
    def random_permutate(input:torch.Tensor, times_by_row:int|None = None, times_by_column:int|None = None)->torch.Tensor:
        assert is_square_matrix(input)
        dim = input.shape[0]
        the_device = input.device
        
        if times_by_row is None:
            times_by_row = dim*3
            pass
        if times_by_column is None:
            times_by_column = dim*3
            pass
        
        with torch.no_grad():
            for _ in range(times_by_row):
                rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=the_device)
                _temp = input[rand_index_1].clone()
                input[rand_index_1] = input[rand_index_2]
                input[rand_index_2] = _temp
                pass
            for _ in range(times_by_column):
                rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=the_device)
                _temp = input[:,rand_index_1].clone()
                input[:,rand_index_1] = input[:,rand_index_2]
                input[:,rand_index_2] = _temp
            return input
        pass#/function
    if "test" and __DEBUG_ME__() and True:
        def ____test____random_permutate():
            "only row"
            for _ in range(5):
                mat = torch.tensor([[0.,1,2],[0,1,2],[0,1,2]])
                mat_after = random_permutate(mat.detach().clone(), times_by_row = random.randint(2,5),
                                                times_by_column = 0)
                #-------------------------------------------------------------------------
                assert mat.eq(mat_after).all()
                pass
            
            "only column"
            for _ in range(5):
                mat = torch.tensor([[0.,0,0],[1,1,1],[2,2,2]])
                mat_after = random_permutate(mat.detach().clone(), times_by_row = 0,
                                                times_by_column = random.randint(2,5))
                #-------------------------------------------------------------------------
                assert mat.eq(mat_after).all()
                pass
                
            "only column"
            for _ in range(115):
                dim = random.randint(2,300)
                #-------------------------------------------------------------------------
                mat = torch.randn(size=[dim,dim])
                mat_after = random_permutate(mat.detach().clone(), 
                                            times_by_row    = random.randint(dim+10,dim*3+50),
                                            times_by_column = random.randint(dim+10,dim*3+50))
                #-------------------------------------------------------------------------
                _ori_sum = mat.sum()
                _after_sum = mat_after.sum()
                assert _tensor_equal(_ori_sum, _after_sum, epsilon=_ori_sum.abs()*0.0003)
                assert _tensor_equal(mat.std(), mat_after.std())
                
                assert have_same_elements(mat, mat_after)
                pass
            return 
        ____test____random_permutate()
        pass

    def random_permutation_matrix__by_once(dim:int, dtype = torch.float32, device = 'cpu')->torch.Tensor:
        result = torch.eye(n=dim, dtype=dtype, device=device)
        rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=device)
        result[rand_index_1,rand_index_1] = 0.
        result[rand_index_2,rand_index_2] = 0.
        result[rand_index_1,rand_index_2] = 1.
        result[rand_index_2,rand_index_1] = 1.
        return result
    if "test" and __DEBUG_ME__() and True:
        def ____test____random_permutation_matrix__by_once():
            for _ in range(116):
                dim = random.randint(2,300)
                #---------------------------
                mat = random_permutation_matrix__by_once(dim)
                #---------------------------
                assert mat.sum() == dim
                
                _ref_mat = torch.zeros_like(mat)
                _ref_mat[0] = 1.
                assert have_same_elements(mat, _ref_mat)
                
                assert mat.det().abs() == 1
                pass
            
            "if multiple this thing multiply together"
            for _ in range(15):
                dim = random.randint(2,88)
                amount = random.randint(2,10)
                #---------------------------
                mat = random_permutation_matrix__by_once(dim)
                for _the_number in range(amount-1):
                    new_mat = random_permutation_matrix__by_once(dim)
                    mat = mat@new_mat
                    pass
                #---------------------------
                assert mat.sum() == dim
                
                _ref_mat = torch.zeros_like(mat)
                _ref_mat[0] = 1.
                assert have_same_elements(mat, _ref_mat)
                
                assert mat.det().abs() == 1
                pass
            
            
            return 
        ____test____random_permutation_matrix__by_once()
        pass

    def random_permutation_matrix(dim:int, times:int|None = None, dtype = torch.float32, device = 'cpu')->torch.Tensor:
        init_eye = torch.eye(n=dim, dtype=dtype, device=device)
        if times is None:
            times = dim*3
            pass
        row_times = times//2
        column_times = times-row_times
        result = random_permutate(init_eye,times_by_row=row_times,times_by_column=column_times)
        return result
    if "test" and __DEBUG_ME__() and True:
        def ____test____random_permutation_matrix():
            for _ in range(116):
                dim = random.randint(2,300)
                #---------------------------
                mat = random_permutation_matrix(dim,dim*5)
                #---------------------------
                assert mat.sum() == dim
                
                _ref_mat = torch.zeros_like(mat)
                _ref_mat[0] = 1.
                assert have_same_elements(mat, _ref_mat)
                
                assert mat.det().abs() == 1
                pass
            return 
        ____test____random_permutation_matrix()
        pass
    
    pass

"   new version below   vvvvvvvvvvvvv"

def randomly_permutate__vector(input:torch.Tensor)->torch.Tensor:
    '''randomly permutate this vector.'''
    
    dim = input.shape[-1]
    the_rand_permutation_index = torch.randperm(dim)
    result = input[..., the_rand_permutation_index]
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____randomly_permutate_this_vector():
        # vec = torch.linspace(1,5,4)
        # ori_sum = vec.sum()
        # vec = randomly_permutate_this_vector(vec)
        # assert _tensor_equal(ori_sum, vec.sum())
        
        # at least 2 ways to validate. 
        # sort and check with eq
        # or sum/mean will keep before and after.
        
        
        #a = torch.tensor([[2.,1],[4,3]])
        # a = torch.tensor([[2.,1],[0,0],[4,3]])
        # print(a)
        # b = a.sort()

        # a = torch.linspace(1,8,8).reshape([2,2,2])
        # b = a.sum(dim=1)

        
        for _ in range(112):
            dim = [random.randint(2,1000)]
            vec = torch.randn(size = dim)
            ori_sum = vec.sum()
            vec = randomly_permutate__vector(vec)
            assert _tensor_equal(ori_sum, vec.sum())
            pass
        
        for _ in range(112):
            dim = [random.randint(2,100), random.randint(2,100)]
            vec = torch.randn(size = dim)
            ori_sum_0 = vec.sum(dim=0).sort().values
            ori_sum_1 = vec.sum(dim=1)
            vec = randomly_permutate__vector(vec)
            assert _tensor_equal(ori_sum_0, vec.sum(dim=0).sort().values)
            assert _tensor_equal(ori_sum_1, vec.sum(dim=1))
            pass
        
        for _ in range(112):
            dim = [random.randint(2,30), random.randint(2,30), random.randint(2,30)]
            vec = torch.randn(size = dim)
            ori_sum_0 = vec.sum(dim=0).sort().values
            ori_sum_1 = vec.sum(dim=1).sort().values
            ori_sum_2 = vec.reshape([-1,dim[2]]).sum(dim=1)
            vec = randomly_permutate__vector(vec)
            assert _tensor_equal(ori_sum_0, vec.sum(dim=0).sort().values)
            assert _tensor_equal(ori_sum_1, vec.sum(dim=1).sort().values)
            assert _tensor_equal(ori_sum_2, vec.reshape([-1,dim[2]]).sum(dim=1))
            pass
        
        return 
    
    ____test____randomly_permutate_this_vector()
    pass

def randomly_permutate__matrix(input:torch.Tensor)->torch.Tensor:
    '''randomly permutate this vector.'''
    
    dim = input.shape[-1]
    the_rand_permutation_index = torch.randperm(dim)
    result = input[..., the_rand_permutation_index]
    
    dim = input.shape[-2]
    the_rand_permutation_index = torch.randperm(dim)
    result = result[..., the_rand_permutation_index, :]
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____randomly_permutate_this_matrix():
        # at least 2 ways to validate. 
        # sort and check with eq
        # or sum/mean will keep before and after.
        
        # a = torch.linspace(1,9,9).reshape([3,3])
        # b = randomly_permutate_this_matrix(a)
        # prin(b)
        
        
        for _ in range(112):
            dim = [random.randint(2,100), random.randint(2,100)]
            vec = torch.randn(size = dim)
            ori_sum_0 = vec.sum(dim=0).sort().values
            ori_sum_1 = vec.sum(dim=1).sort().values
            vec = randomly_permutate__matrix(vec)
            assert _tensor_equal(ori_sum_0, vec.sum(dim=0).sort().values)
            assert _tensor_equal(ori_sum_1, vec.sum(dim=1).sort().values)
            pass
        
        for _ in range(112):
            dim = [random.randint(2,30), random.randint(2,30), random.randint(2,30)]
            dim = [2,2,2]
            vec = torch.randn(size = dim)
            ori_sum_0 = vec.sum(dim=0).reshape([-1]).sort().values
            ori_sum_1 = vec.sum(dim=1).sort().values
            ori_sum_1_flattened = vec.sum(dim=1).reshape([-1]).sort().values
            ori_sum_2 = vec.reshape([dim[0], -1]).sum(dim=1).sort().values
            vec = randomly_permutate__matrix(vec)
            assert _tensor_equal(ori_sum_0, vec.sum(dim=0).reshape([-1]).sort().values)
            assert _tensor_equal(ori_sum_1, vec.sum(dim=1).sort().values)
            assert _tensor_equal(ori_sum_1_flattened, vec.sum(dim=1).reshape([-1]).sort().values)
            assert _tensor_equal(ori_sum_2, vec.reshape([dim[0], -1]).sum(dim=1).sort().values)
            pass
        
        return 
    
    ____test____randomly_permutate_this_matrix()
    pass
        
    
    # pass#/function
    #     # same elements.
        
    #     for _ in range(5):
    #         mat = torch.tensor([[0.,1,2],[0,1,2],[0,1,2]])
    #         mat_after = randomly_permutate_this_vector(mat.detach().clone(), times_by_row = random.randint(2,5),
    #                                         times_by_column = 0)
    #         #-------------------------------------------------------------------------
    #         assert mat.eq(mat_after).all()
    #         pass
        
    #     "only column"
    #     for _ in range(5):
    #         mat = torch.tensor([[0.,0,0],[1,1,1],[2,2,2]])
    #         mat_after = random_permutate(mat.detach().clone(), times_by_row = 0,
    #                                         times_by_column = random.randint(2,5))
    #         #-------------------------------------------------------------------------
    #         assert mat.eq(mat_after).all()
    #         pass
        
    #     "only column"
    #     for _ in range(115):
    #         dim = random.randint(2,300)
    #         #-------------------------------------------------------------------------
    #         mat = torch.randn(size=[dim,dim])
    #         mat_after = random_permutate(mat.detach().clone(), 
    #                                     times_by_row    = random.randint(dim+10,dim*3+50),
    #                                     times_by_column = random.randint(dim+10,dim*3+50))
    #         #-------------------------------------------------------------------------
    #         _ori_sum = mat.sum()
    #         _after_sum = mat_after.sum()
    #         assert _tensor_equal(_ori_sum, _after_sum, epsilon=_ori_sum.abs()*0.0003)
    #         assert _tensor_equal(mat.std(), mat_after.std())
            
    #         assert have_same_elements(mat, mat_after)
    #         pass
    #     return 
    # ____test____random_permutate()
    # pass










