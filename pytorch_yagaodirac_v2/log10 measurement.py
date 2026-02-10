from typing import List, Tuple, Optional, TypeGuard
import torch
import math, random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pytorch_yagaodirac_v2.Util import _raw_log10_avg_safe, log10_avg_safe__with_batch,\
    _tensor_equal,            str_the_list
from pytorch_yagaodirac_v2.training_ended_sound import play_noise

# 最后完事了来补个说明。



# 1w  里面的测试重新看看。最早的一些测试很乱。
# 两个函数的实际效果稍微对比一下。
# 另外就是，0.9和其他top ratio的结果可能不一样。

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
assert __DEBUG_ME__(), "don't import this file."


"most useful formula:"
"randn, -0.16"
"rand, rand*2-1, both -0.32"
"rand-0.5, -0.62"
"randn mat @ randn mat >>> log_a + log_b + 0.5*log10(mid_dim) + 0.12"
"vec of length 1 >>> -0.5*log10(dim)-0.21"
"vec of length 1 @ randn mat >>> -0.15"






if "programming style example" and False:
    def _example(): 
        TESTING = False#set to true, read the print. Setup the valid, set to false.
        test_time = 11
        #--------------------#--------------------#--------------------
        param_for_both_mode_list = [1,2,3]
        #--------------------#--------------------#--------------------
        if not TESTING:
            param_only_for_valid_list = [1.5,2.5,3.5]
            pass
        else:
            print(test_time)
            pass
        for param_set_count in range(param_for_both_mode_list.__len__()):
            param_for_both_mode = param_for_both_mode_list[param_set_count]
            if not TESTING:
                param_only_for_valid = param_only_for_valid_list[param_set_count]
                pass
            _raw_result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                #--------------------#--------------------#--------------------
                a = torch.rand(size=[100])
                b = a+param_for_both_mode
                #--------------------#--------------------#--------------------
                _raw_result[test_count] = b.mean()
                pass
            the_min = _raw_result.min()
            the_max = _raw_result.max()
            the_mean = _raw_result.mean()
            if TESTING:
                print(f"[{100}]   {the_min:.3f}   {the_max:.3f}   {the_mean:.3f}   ")
                pass
            else:
                assert _tensor_equal(the_mean, [param_only_for_valid], epsilon = 0.6)
                pass
        return 
    _example()
    pass


# randn    >>> -0.16
# rand     >>> -0.32
# rand*2-1 >>> -0.32
if "basic random number gen test" and False:
    # result
    # the measurement is only about the avg order of magnitude of elements.
    # it's a per element measurement.
    # the shape, or the total number of elements don't affect the result.
    # randn >>> -0.16
    # rand  >>> -0.32
    # rand*2-1 has the same abs as rand, so they also have the same log10 measurement result.
    
    def basic_random_number_gen():
        if "1d randn" and False:
            TESTING = False
            test_time = 10
            #--------------------#--------------------#--------------------
            dim_segment_list = [100,1000,10000,100000]
            #--------------------#--------------------#--------------------
            if not TESTING:
                the_min_gt_this_list =  [-0.29,-0.20,-0.18,]
                the_max_lt_this_list =  [-0.04,-0.12,-0.14,]
                the_mean_eq_this_list = [-0.16,-0.16,-0.16,]
                epsilon_list =          [ 0.15, 0.05, 0.03,]
                pass
            for param_set_count in range(dim_segment_list.__len__()-1):
                dim_from = dim_segment_list[param_set_count]
                dim_to = dim_segment_list[param_set_count+1]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                else:
                    print(test_time)
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    dim = random.randint(dim_from, dim_to)
                    the_randn = torch.randn(size=[dim])
                    _this_result = _raw_log10_avg_safe(the_randn)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    print(f"[{dim_from}, {dim_to}]   {the_min-0.01:.2f}   {the_max+0.01:.2f}   {the_mean:.2f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass
        if "2d randn" and False:
            TESTING = False
            test_time = 10
            #--------------------#--------------------#--------------------
            dim_segment_list = [10,100,500]
            #--------------------#--------------------#--------------------
            if not TESTING:
                the_min_gt_this_list =  [-0.28,-0.18]
                the_max_lt_this_list =  [-0.05,-0.14]
                the_mean_eq_this_list = [-0.16,-0.16]
                epsilon_list =          [ 0.15, 0.03]
                pass
            for param_set_count in range(dim_segment_list.__len__()-1):
                dim_from = dim_segment_list[param_set_count]
                dim_to = dim_segment_list[param_set_count+1]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                else:
                    print(test_time)
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    dim = random.randint(dim_from, dim_to)
                    the_randn = torch.randn(size=[dim, dim])
                    _this_result = _raw_log10_avg_safe(the_randn)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    print(f"[{dim_from}, {dim_to}]   {the_min-0.01:.2f}   {the_max+0.01:.2f}   {the_mean:.2f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass
            pass
        
        if "rand" and False:
            TESTING = False
            test_time = 10
            #--------------------#--------------------#--------------------
            dim_segment_list = [100,1000,10000,50000]
            #--------------------#--------------------#--------------------
            if not TESTING:
                the_min_gt_this_list =  [-0.45,-0.37,-0.34,]
                the_max_lt_this_list =  [-0.21,-0.27,-0.30,]
                the_mean_eq_this_list = [-0.32,-0.32,-0.32,]
                epsilon_list =          [ 0.15, 0.06, 0.03]
                pass
            else:
                print(test_time)
                pass
            for param_set_count in range(dim_segment_list.__len__()-1):
                dim_from = dim_segment_list[param_set_count]
                dim_to = dim_segment_list[param_set_count+1]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    dim = random.randint(dim_from, dim_to)
                    the_randn = torch.rand(size=[dim])
                    _this_result = _raw_log10_avg_safe(the_randn)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    print(f"[{dim_from}, {dim_to}]   {the_min-0.01:.2f}   {the_max+0.01:.2f}   {the_mean:.2f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass
            pass
        
        if "rand*2-1" and False:
            TESTING = False
            if TESTING:
                print("rand*2-1 >>> -0.32")
                pass
            test_time = 10
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_segment_list = [100,1000,10000,50000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                the_min_gt_this_list =[-0.430, -0.354, -0.342]
                the_max_lt_this_list =[-0.251, -0.288, -0.306]
                the_mean_eq_this_list=[-0.324, -0.323, -0.323]
                epsilon_list         =[ 0.116,  0.045,  0.029]
                pass
            
            for param_set_count in range(dim_segment_list.__len__()-1):
                dim_from = dim_segment_list[param_set_count]
                dim_to = dim_segment_list[param_set_count+1]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    dim = random.randint(dim_from, dim_to)
                    the_randn = torch.rand(size=[dim])*2.-1.
                    _this_result = _raw_log10_avg_safe(the_randn)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"[{dim_from:5}, {dim_to:5}]   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={ str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={ str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        return 
    basic_random_number_gen()
    pass



# @ means matmul.
# rand is [0,1), rand*2-1 is [-1,1)
# randn   [??,mid dim] @ randn   [mid dim,????], -0.16, -0.16, into a 0.5*log10(mid dim)-0.16
# rand*2-1[??,mid dim] @ rand*2-1[mid dim,????], -0.32, -0.32, into a 0.5*log10(mid dim)-0.63
# rand*2-1[??,mid dim] @ rand    [mid dim,????], -0.32, -0.32, into a 0.5*log10(mid dim)-0.63
# rand    [??,mid dim] @ rand    [mid dim,????], -0.32, -0.32, into a  1.*log10(mid dim)-0.60
# 2 different matrixs
# randn   [??,mid dim] @ rand*2-1[mid dim,????], -0.16, -0.32, into a 0.5*log10(mid dim)-0.40
# randn   [??,mid dim] @ rand    [mid dim,????], -0.16, -0.32, into a 0.5*log10(mid dim)-0.40

if "multiplication" and True:
    # result
    # randn @ randn, 2 -0.16s, into a 0.5*log(mid dim)-0.16.
    # so basically, the -0.16 is a global offset.
    # if you add 0.16 to all the measurement, it's regular. 
    # this applies to mat@mat, vec@mat, and vec dot vec, if it's normal distribution.
    
    # But, out side randn, all the cases are slightly different. 
    # If all the sign of elements are the same(for both matrixs), it's a 1*log10, 
    # but if half is pos and half is neg, it's always 0.5*log10.
    # the last offset varies across cases.
    
    
    def basic_multiplication():
        if "randn[dim,dim] @ randn[dim,dim]" and False:
            "randn[dim,dim] @ randn[dim,dim], 2 -0.16s, into a 0.5*log(mid dim)-0.16."
            TESTING = False
            test_time = 10
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100,316,1000,3162]#,10000,]
            #--------------------#--------------------#--------------------
            if not TESTING:
                the_min_gt_this_list =  [0.808,1.074,1.330,1.581,1.832]
                the_max_lt_this_list =  [0.871,1.108,1.354,1.602,1.852]
                the_mean_eq_this_list = [0.840,1.091,1.342,1.592,1.842]
                epsilon_list =          [0.033,0.018,0.013,0.012,0.011]
                pass
            else:
                print(test_time)
                pass
            for param_set_count in range(dim_list.__len__()):
                dim = dim_list[param_set_count]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    mat1 = torch.randn(size=[dim, dim], device = device)
                    mat2 = torch.randn(size=[dim, dim], device = device)
                    prod = mat1@mat2
                    _this_result = _raw_log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    print(f"[{dim:5}]   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            pass#/test
        if "randn[???,dim] @ randn[dim,????]" and False:
            "randn[???,dim] @ randn[dim,????], 2 -0.16s, into a 0.5*log(mid dim)-0.16."
            TESTING = False
            test_time = 1
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,316  ,1000 ,3162 ,10000,]
            rand_dim_from_list = [2000 ,2000 ,1000 ,300  ,300  ,]
            rand_dim_to_list =   [10000,10000,10000,10000,5000 ,]
            #--------------------#--------------------#--------------------
            if not TESTING:
                the_min_gt_this_list =  [0.828,1.079,1.331,1.581,1.828,]
                the_max_lt_this_list =  [0.851,1.103,1.352,1.604,1.855,]
                the_mean_eq_this_list = [0.840,1.091,1.342,1.592,1.842,]
                epsilon_list =          [0.013,0.013,0.012,0.013,0.015,]
                pass
            else:
                print(test_time)
                pass
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat1 = torch.randn(size=[__rand_dim, mid_dim  ], device = device)
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.randn(size=[mid_dim,    __rand_dim], device = device)
                    prod = mat1@mat2
                    _this_result = _raw_log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    print(f"[{mid_dim}]   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            pass#/test
        
        if "rand*2-1[???,dim] @ rand*2-1[dim,????]" and False:
            "rand*2-1[???,dim] @ rand*2-1[dim,????], 2 -0.32s, into a 0.5*log(mid dim)-0.63."
            TESTING = False
            test_time = 10
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,316  ,1000 ,3162 ,10000,]
            rand_dim_from_list = [2000 ,2000 ,1000 ,300  ,300  ,]
            rand_dim_to_list =   [10000,10000,10000,10000,5000 ,]
            #--------------------#--------------------#--------------------
            if not TESTING:
                the_min_gt_this_list =  [0.354,0.604,0.854,1.104,1.354,]
                the_max_lt_this_list =  [0.376,0.625,0.875,1.126,1.376,]
                the_mean_eq_this_list = [0.365,0.615,0.865,1.115,1.365,]
                epsilon_list =          [0.012,0.012,0.012,0.012,0.012,]#????a lil bit too clean...
                pass
            else:
                print(test_time)
                pass
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat1 = torch.rand(size=[__rand_dim, mid_dim  ], device = device)*2.-1.
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand(size=[mid_dim,    __rand_dim], device = device)*2.-1.
                    prod = mat1@mat2
                    _this_result = _raw_log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    print(f"[{mid_dim:5}]   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            pass#/test
        
        if "rand*2-1[???,dim] @ rand[dim,????]" and False:
            print("rand*2-1[???,dim] @ rand[dim,????], 2 -0.32s, into a 0.5*log(mid dim)-0.63.")
            TESTING = False
            test_time = 10
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,316  ,1000 ,3162 ,10000,]
            rand_dim_from_list = [2000 ,2000 ,1000 ,300  ,300  ,]
            rand_dim_to_list =   [10000,10000,10000,10000,5000 ,]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                the_min_gt_this_list  = [0.338, 0.590, 0.838, 1.080, 1.326]
                the_max_lt_this_list  = [0.388, 0.639, 0.889, 1.146, 1.406]
                the_mean_eq_this_list = [0.365, 0.615, 0.865, 1.115, 1.365]
                epsilon_list          = [0.037, 0.034, 0.037, 0.045, 0.051]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat1 = torch.rand(size=[__rand_dim, mid_dim  ], device = device)*2.-1.
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand(size=[mid_dim,    __rand_dim], device = device)
                    prod = mat1@mat2
                    _this_result = _raw_log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"[{mid_dim:5}]   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list={the_min_gt_this_list}")    
                print(f"the_max_lt_this_list={the_max_lt_this_list}")    
                print(f"the_mean_eq_this_list={the_mean_eq_this_list}")    
                print(f"epsilon_list={epsilon_list}")    
                pass
            pass#/test
        
        if "rand[???,dim] @ rand[dim,????]" and False:
            TESTING = False
            if TESTING:
                print("rand[???,dim] @ rand[dim,????], 2 -0.32s, into a 1.*log(mid dim)-0.60.")
                pass
            test_time = 10
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,316  ,1000 ,3162 ,10000,]
            rand_dim_from_list = [2000 ,2000 ,1000 ,300  ,300  ,]
            rand_dim_to_list =   [10000,10000,10000,10000,5000 ,]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                the_min_gt_this_list =[1.393, 1.890, 2.389, 2.888, 3.388]
                the_max_lt_this_list =[1.416, 1.913, 2.411, 2.910, 3.409]
                the_mean_eq_this_list=[1.404, 1.901, 2.400, 2.899, 3.399]
                epsilon_list         =[0.022, 0.021, 0.021, 0.021, 0.020]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat1 = torch.rand(size=[__rand_dim, mid_dim  ], device = device)
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand(size=[mid_dim,    __rand_dim], device = device)
                    prod = mat1@mat2
                    _this_result = _raw_log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"[{mid_dim:5}]   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list={ str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list={ str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "randn[???,dim] @ rand*2-1[dim,????]" and False:
            TESTING = False
            if TESTING:
                print("randn[???,dim] @ rand*2-1[dim,????], ?????????????? 2 -0.32s, into a 1.*log(mid dim)-0.60.")
                pass
            test_time = 10
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,316  ,1000 ,3162 ,10000,]
            rand_dim_from_list = [2000 ,2000 ,1000 ,300  ,300  ,]
            rand_dim_to_list =   [10000,10000,10000,10000,5000 ,]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                the_min_gt_this_list =[0.591, 0.842, 1.092, 1.342, 1.591]
                the_max_lt_this_list =[0.615, 0.864, 1.114, 1.366, 1.614]
                the_mean_eq_this_list=[0.603, 0.853, 1.103, 1.353, 1.603]
                epsilon_list         =[0.023, 0.021, 0.021, 0.023, 0.022]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat1 = torch.randn(size=[__rand_dim, mid_dim  ], device = device)
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand (size=[mid_dim,    __rand_dim], device = device)*2.-1.
                    prod = mat1@mat2
                    _this_result = _raw_log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"[{mid_dim:5}]   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={ str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={ str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "randn[??,mid dim] @ rand [mid dim,????]" and False:
            TESTING = False
            if TESTING:
                print("randn[??,mid dim] @ rand [mid dim,????], ?????????????? 2 -0.32s, into a 1.*log(mid dim)-0.60.")
                pass
            test_time = 10
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,316  ,1000 ,3162 ,10000,]
            rand_dim_from_list = [2000 ,2000 ,1000 ,300  ,300  ,]
            rand_dim_to_list =   [10000,10000,10000,10000,5000 ,]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                the_min_gt_this_list =[0.579, 0.830, 1.077, 1.315, 1.558]
                the_max_lt_this_list =[0.623, 0.877, 1.138, 1.392, 1.644]
                the_mean_eq_this_list=[0.603, 0.853, 1.103, 1.353, 1.603]
                epsilon_list         =[0.034, 0.034, 0.045, 0.049, 0.055]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat1 = torch.randn(size=[__rand_dim, mid_dim  ], device = device)
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand (size=[mid_dim,    __rand_dim], device = device)
                    prod = mat1@mat2
                    _this_result = _raw_log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"[{mid_dim:5}]   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={ str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={ str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        return 
    basic_multiplication()
    pass


# relu   (randn) >>> -0.46* . Relu provides too many 0s. I infer it as -0.16+log10(0.5), -0.46
# gelu   (randn) >>> -0.60
# sigmoid(randn) >>> -0.30
# tanh   (randn) >>> -0.27
# sin(randn*scale_factor) is complicated. For sf(scale_factor) <0.1, it's -1.*log10(dim)-0.16, the same as randn, 
# bc input is close to 0, sin(x) is very close to x. For sf>1.8, it's -0.20(or -0.197). But in between, measurement is between -1.159 and -0.197.

1w 继续。还有cos。。。

if "with activation functions" and True:
    # randn, then act, @ another randn.
    # relu,gelu,(swiss)
    # sigmoid, tanh,, 
    # sin, cos, 
    # gauss?
    def with_activation_functions():
        if "gelu(randn[100000])" and False:
            TESTING = False
            if TESTING:
                print("gelu(randn[100000]) >>> -0.60")
                pass
            test_time = 100
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.610]
                the_max_lt_this_list =[-0.581]
                the_mean_eq_this_list=[-0.596]
                epsilon_list     =[0.025]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                dim = dim_list[param_set_count]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    loss_layer = torch.nn.GELU()
                    vec = torch.randn(size=[dim], device = device)
                    vec = loss_layer(vec)
                    _this_result = _raw_log10_avg_safe(vec)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"{the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={ str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={ str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "sigmoid(randn[100000])" and False:
            TESTING = False
            if TESTING:
                print("sigmoid(randn[100000]) >>> ???")
                pass
            test_time = 100
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.309]
                the_max_lt_this_list =[-0.284]
                the_mean_eq_this_list=[-0.296]
                epsilon_list     =[0.023]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                dim = dim_list[param_set_count]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    loss_layer = torch.nn.Sigmoid()
                    vec = torch.randn(size=[dim], device = device)
                    vec = loss_layer(vec)
                    _this_result = _raw_log10_avg_safe(vec)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"{the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={ str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={ str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "tanh(randn[100000])" and False:
            TESTING = False
            if TESTING:
                print("tanh(randn[100000]) >>> ???")
                pass
            test_time = 100
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.279]
                the_max_lt_this_list =[-0.251]
                the_mean_eq_this_list=[-0.265]
                epsilon_list     =[0.024]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                dim = dim_list[param_set_count]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    loss_layer = torch.nn.Tanh()
                    vec = torch.randn(size=[dim], device = device)
                    vec = loss_layer(vec)
                    _this_result = _raw_log10_avg_safe(vec)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"{the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={ str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={ str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        
        
        if "sin(randn[100000]*scale_factor) " and True:
            TESTING = False
            if TESTING:
                print("sin(randn[100000]*scale_factor) >>> ???")
                pass
            test_time = 100
            device = 'cuda'
            #--------------------#--------------------#--------------------
            scale_factor_list = [0.00001, 0.0001, 0.001, 0.01, 0.0316,0.1,0.316,1.,1.3,1.6,1.7,1.8,2.,5.,10.]
            dim_list = [100000]*scale_factor_list.__len__()
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-5.174, -4.173, -3.173, -2.174, -1.674, -1.174, -0.681, -0.264, -0.225, -0.213, -0.212, -0.211, -0.211, -0.211, -0.210]
                the_max_lt_this_list =[-5.143, -4.143, -3.143, -2.143, -1.643, -1.144, -0.651, -0.237, -0.196, -0.186, -0.185, -0.183, -0.184, -0.183, -0.183]
                the_mean_eq_this_list=[-5.158, -4.158, -3.158, -2.158, -1.658, -1.159, -0.667, -0.251, -0.210, -0.199, -0.198, -0.197, -0.197, -0.197, -0.197]
                epsilon_list         =[ 0.026,  0.025,  0.025,  0.026,  0.025,  0.025,  0.025,  0.024,  0.025,  0.024,  0.024,  0.025,  0.024,  0.024,  0.024]
                #  scale_factor_list  0.00001, 0.0001,  0.001,   0.01, 0.0316,    0.1,  0.316,     1.,    1.3,    1.6,    1.7,    1.8,      2.,    5.,    10.
                #                                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                pass
            
            for param_set_count in range(dim_list.__len__()):
                dim = dim_list[param_set_count]
                scale_factor = scale_factor_list[param_set_count]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    #loss_layer = torch.sin(nn.sin()
                    vec = torch.randn(size=[dim], device = device)*scale_factor
                    vec = vec.sin()
                    _this_result = _raw_log10_avg_safe(vec)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test_count
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                if TESTING:
                    the_min_gt_this_list.append(the_min.item()-0.01)
                    the_max_lt_this_list.append(the_max.item()+0.01)
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min  +0.02
                    _delta_2 = the_max  - the_mean +0.02
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"dim:{dim}, sf:{scale_factor}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={ str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={ str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        return 
    with_activation_functions()
    pass
    













#if "K He init?" and True:



#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################






if "log10 standard vec related test          come back later" and __DEBUG_ME__() and False:
    def ____test____log10_avg_safe____standard_vec():
        
        #<  from Random.py   in this folder>
        def random_standard_vector(dim:int, dtype = torch.float32, device='cpu')->torch.Tensor:
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
        #</ from Random.py   in this folder>
        
        if "old" and False:
            #old, duplicated...
            # dim_list =      [1   ,  100, 10000]
            # equal_to_list = [0   ,-1.16,-2.21]
            # epsilon_list =  [1e-4, 0.16, 0.11]
            # #a better epsilong is  0.12, 0.08
            # for param_set_count in range(dim_list.__len__()):
            #     dim = dim_list[param_set_count]
            #     equal_to = equal_to_list[param_set_count]
            #     epsilon = epsilon_list[param_set_count]
            #     for _ in range(11):
            #         vec = random_standard_vector(dim)
            #         assert _tensor_equal(get_vector_length(vec), [1.])
            #         log_10_of_vec = log10_avg_safe(vec)
            #         assert vec.shape == torch.Size([dim])
            #         assert _tensor_equal(log_10_of_vec, [equal_to], epsilon=epsilon)
            #         pass
            #     pass#for param set
            pass
        
        #result: basically the log10 of a standard vec is -0.5*log10(dim)-0.21
        "directly measure. -0.5*log10(dim)-0.21"
        test_time = 6
        _ref_log10_list =      [1,     2,     3,     4,      5]
        _assert_dim_eq_list = [10,   100,  1000, 10000, 100000]
        _low_bound_list = [ -1.16, -1.37, -1.80, -2.29,  -2.78]
        _high_bound_list = [-0.45, -1.05, -1.61, -2.13,  -2.64]
        #avg of bounds             -1.21, -1.7 , -2.21,  -2.71
        for param_set_count in range(_ref_log10_list.__len__()):
            _ref_log10 = _ref_log10_list        [param_set_count]
            _assert_dim_eq = _assert_dim_eq_list[param_set_count]
            _low_bound = _low_bound_list        [param_set_count]
            _high_bound = _high_bound_list      [param_set_count]
            for _ in range(test_time):
                dim = int(math.pow(10,_ref_log10))
                assert dim == _assert_dim_eq
                vec = random_standard_vector(dim)
                assert _tensor_equal(get_vector_length(vec), [1.])#length always 1.
                log_10_of_vec = _raw_log10_avg_safe(vec)
                assert log_10_of_vec>_low_bound and log_10_of_vec<_high_bound
                pass
            
                
        if "old code" and False:
                # for _ in range(0):
                #     _ref_log10 = random.random()*0.+1.
                #     dim = int(math.pow(10,_ref_log10))
                #     assert dim == 10 
                #     vec = random_standard_vector(dim)
                #     assert _tensor_equal(get_vector_length(vec), [1.])
                #     log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
                #     assert log_10_of_vec>-1.16 and log_10_of_vec<-0.45# 0.8
                #     pass
                #
                # for _ in range(0):
                #     _ref_log10 = random.random()*0.+2.
                #     dim = int(math.pow(10,_ref_log10))
                #     assert dim == 100 
                #     vec = random_standard_vector(dim)
                #     assert _tensor_equal(get_vector_length(vec), [1.])
                #     log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
                #     assert log_10_of_vec>-1.29 and log_10_of_vec<-1.05# 1.17
                #     pass
                #
                # for _ in range(0):
                #     _ref_log10 = random.random()*0.+3.
                #     dim = int(math.pow(10,_ref_log10))
                #     assert dim == 1000 
                #     vec = random_standard_vector(dim)
                #     assert _tensor_equal(get_vector_length(vec), [1.])
                #     log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
                #     assert log_10_of_vec>-1.79 and log_10_of_vec<-1.61# 1.7
                #     pass
                #
                # for _ in range(0):
                #     _ref_log10 = random.random()*0.+4.
                #     dim = int(math.pow(10,_ref_log10))
                #     assert dim == 10000 
                #     #assert dim >= 10 and dim <= 100000
                #     vec = random_standard_vector(dim, device='cpu')
                #     assert _tensor_equal(get_vector_length(vec), torch.tensor([1.], device='cpu'))
                #     log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
                #     #_ref = (_ref_log10*0.5* 1.05 + 0.05)*-1.
                #     #assert _tensor_equal(log_10_of_vec, [-0.69], epsilon=0.22)
                #     assert log_10_of_vec>-2.29 and log_10_of_vec<-2.13# 2.21
                #     #print(log_10_of_vec, [_ref])
                #     pass
                #
                # for _ in range(1):
                #     _ref_log10 = random.random()*0.+5.
                #     dim = int(math.pow(10,_ref_log10))
                #     assert dim == 100000
                #     #assert dim >= 10 and dim <= 100000
                #     vec = random_standard_vector(dim, device='cpu')
                #     assert _tensor_equal(get_vector_length(vec), torch.tensor([1.], device='cpu'))
                #     log_10_of_vec = log10_avg_safe(vec)
                #     #_ref = (_ref_log10*0.5* 1.05 + 0.05)*-1.
                #     #assert _tensor_equal(log_10_of_vec, [-0.69], epsilon=0.22)
                #     assert log_10_of_vec>-2.78 and log_10_of_vec<-2.64# 2.71
                #     #print(log_10_of_vec, [_ref])
                #     pass
                pass
        
        "vec dot vec, -0.5*log10(dim)-0.15"
        #result, it's only about the angle, unless the least 10% are removed.
        # the top 90% is -0.5*log10(dim)-0.15
        # so 2 vecs(-0.5*log10(dim)-0.21) dot into a scalar is(-0.5*log10(dim)-0.15)
        test_time = 11
        inner_test_time = 1000
        dim_list =      [10,100,1000,10000,100000]
        _low_bound_list =   [-0.69,-1.22,-1.70,-2.21,-2.70,]
        _high_bound_list =  [-0.58,-1.11,-1.61,-2.11,-2.62,]
        #avg of bounds       -0.64,-1.16,-1.65,-2.16,-2.66,
        for param_set_count in range(_ref_log10_list.__len__()):
            dim = dim_list        [param_set_count]
            _low_bound = _low_bound_list        [param_set_count]
            _high_bound = _high_bound_list      [param_set_count]
            
            raw_result = torch.empty(size=[inner_test_time])
            for inner_test_count in range(inner_test_time):
                vec1 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                vec2 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                dot_prod = vec1.dot(vec2)
                log10_of__dot_prod = dot_prod.abs().log10()
                assert log10_of__dot_prod<=0.
                raw_result[inner_test_count] = log10_of__dot_prod
                pass
            _useful_flag = get_mask_of_top_element__rough(raw_result.reshape([1,-1]))[0].reshape([-1])
            result = raw_result[_useful_flag]
            the_mean = result.mean()
            assert the_mean>_low_bound and the_mean<_high_bound
            pass#for param set
        
        if "old" and False:
            for _ in range(0):
                test_time = 1000
                raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #dim = random.randint(300,10000)
                    dim = 10
                    vec1 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    vec2 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    dot_prod = vec1.dot(vec2)
                    log10_of__dot_prod = dot_prod.abs().log10()
                    assert log10_of__dot_prod<=0.
                    raw_result[test_count] = log10_of__dot_prod
                    pass
                _useful_flag = get_mask_of_top_element__rough(raw_result.reshape([1,-1]))[0].reshape([-1])
                result = raw_result[_useful_flag]
                the_mean = result.mean()
                assert the_mean>-0.69 and the_mean<-0.58# 0.64
            
            for _ in range(0):
                test_time = 1000
                raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #dim = random.randint(300,10000)
                    dim = 100
                    vec1 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    vec2 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    dot_prod = vec1.dot(vec2)
                    log10_of__dot_prod = dot_prod.abs().log10()
                    assert log10_of__dot_prod<=0.
                    raw_result[test_count] = log10_of__dot_prod
                    pass
                _useful_flag = get_mask_of_top_element__rough(raw_result.reshape([1,-1]))[0].reshape([-1])
                result = raw_result[_useful_flag]
                the_mean = result.mean()
                assert the_mean>-1.20 and the_mean<-1.11# 1.15
                pass
                
            for _ in range(0):
                test_time = 1000
                raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #dim = random.randint(300,10000)
                    dim = 1000
                    vec1 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    vec2 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    dot_prod = vec1.dot(vec2)
                    log10_of__dot_prod = dot_prod.abs().log10()
                    assert log10_of__dot_prod<=0.
                    raw_result[test_count] = log10_of__dot_prod
                    pass
                _useful_flag = get_mask_of_top_element__rough(raw_result.reshape([1,-1]))[0].reshape([-1])
                result = raw_result[_useful_flag]
                the_mean = result.mean()
                assert the_mean>-1.70 and the_mean<-1.61# 1.65
                pass
            
            for _ in range(0):
                test_time = 1000
                raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #dim = random.randint(300,10000)
                    dim = 10000
                    vec1 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    vec2 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    dot_prod = vec1.dot(vec2)
                    log10_of__dot_prod = dot_prod.abs().log10()
                    assert log10_of__dot_prod<=0.
                    raw_result[test_count] = log10_of__dot_prod
                    pass
                _useful_flag = get_mask_of_top_element__rough(raw_result.reshape([1,-1]))[0].reshape([-1])
                result = raw_result[_useful_flag]
                the_mean = result.mean()
                assert the_mean>-2.21 and the_mean<-2.11# 2.16
                pass
            
            for _ in range(0):
                test_time = 1000
                raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #dim = random.randint(300,10000)
                    dim = 100000
                    vec1 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    vec2 = random_standard_vector(dim)#-0.5*log10(dim)-0.21
                    dot_prod = vec1.dot(vec2)
                    log10_of__dot_prod = dot_prod.abs().log10()
                    assert log10_of__dot_prod<=0.
                    raw_result[test_count] = log10_of__dot_prod
                    pass
                _useful_flag = get_mask_of_top_element__rough(raw_result.reshape([1,-1]))[0].reshape([-1])
                result = raw_result[_useful_flag]
                the_mean = result.mean()
                assert the_mean>-2.70 and the_mean<-2.62# 2.66
                pass
            pass
        
        
        "standard vec @ randn mat, -0.15"
        device = 'cuda'
        test_time = 11
        dim_list =          [ 10,  100, 1000,10000]#[10,100,1000,10000,100000]
        _low_bound_list =   [-0.35,-0.28,-0.18,-0.17]
        _high_bound_list =  [ 0.17,-0.06,-0.11,-0.13,]
        #avg of bounds       -0.10,-0.17,-0.15,-0.16
        for param_set_count in range(dim_list.__len__()):
            dim         = dim_list          [param_set_count]
            _low_bound  = _low_bound_list   [param_set_count]
            _high_bound = _high_bound_list  [param_set_count]
            
            raw_result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                vec = random_standard_vector(dim, device=device)#-0.5*log10(dim)-0.21
                # _log10_of__vec = log10_avg_safe(vec)
                # _ref = -0.5*math.log10(dim)-0.21
                # assert _tensor_equal(_log10_of__vec, [_ref], epsilon=0.1)
                mat = torch.randn(size=[dim,dim], device=device)#-0.16
                # _log10_of__mat = log10_avg_safe(mat.reshape([-1]))
                # assert _log10_of__mat>-0.3 and _log10_of__mat<-0.14#basically -0.16, trust me.
                #assert _tensor_equal(_log10_of__mat, [-0.16], epsilon=0.02)
                prod_vec = vec@mat
                log10_of__prod_vec = _raw_log10_avg_safe(prod_vec)
                raw_result[test_count] = log10_of__prod_vec
                pass
            _useful_flag = get_mask_of_top_element__rough(raw_result.reshape([1,-1]))[0].reshape([-1])
            result = raw_result[_useful_flag]
            the_mean = result.mean()
            #prin(dim,result.max(),result.min(),the_mean)
            assert the_mean>_low_bound and the_mean<_high_bound
            pass#for param set
        
        
        return 
    ____test____log10_avg_safe____standard_vec()
    pass

if "uniform distribution matrix test          come back later" and __DEBUG_ME__() and False:
    def ____test____single_matrix_test____with_log10_avg_safe():
        "rand, [0,1), -0.33"
        test_time = 166
        dim_list =          [10,100,1000]
        _low_bound_list =  [-0.43,-0.43,-0.43]
        _high_bound_list = [-0.22,-0.31,-0.31]
        #avg                -0.32,-0.32,-0.33
        for param_set_count in range(dim_list.__len__()):
            dim         = dim_list        [param_set_count]
            _low_bound  = _low_bound_list [param_set_count]
            _high_bound = _high_bound_list[param_set_count]
            raw_result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                mat = torch.rand(size=[dim,dim])
                raw_result[test_count] = _raw_log10_avg_safe(mat.reshape([1,-1]))#.reshape([-1])
                pass
            #prin("0,1",dim,raw_result.max(),raw_result.min(),raw_result.mean())
            the_mean = raw_result.mean()
            assert the_mean>_low_bound and the_mean<_high_bound
            pass#for param set
        
        "rand-0.5, [-0.5, 0.5)"
        test_time = 166
        dim_list =         [   10,  100, 1000]
        _low_bound_list =  [-0.76,-0.74,-0.74]
        _high_bound_list = [-0.52,-0.60,-0.60]
        #avg                -0.62,-0.62,-0.63
        for param_set_count in range(dim_list.__len__()):
            dim         = dim_list        [param_set_count]
            _low_bound  = _low_bound_list [param_set_count]
            _high_bound = _high_bound_list[param_set_count]
            raw_result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                mat = torch.rand(size=[dim,dim])-0.5
                raw_result[test_count] = _raw_log10_avg_safe(mat.reshape([1,-1]))#.reshape([-1])
                pass
            #prin("-0.5,0.5",dim,raw_result.max(),raw_result.min(),raw_result.mean())
            the_mean = raw_result.mean()
            the_mean>_low_bound and the_mean<_high_bound
            pass#for param set
        
        
        "(rand-0.5)*2, [-1, 1)"
        test_time = 166
        dim_list =         [   10,  100, 1000]
        _low_bound_list =  [-0.21,-0.30,-0.30]
        _high_bound_list = [-0.45,-0.35,-0.44]
        #avg                -0.32,-0.32,-0.33
        for param_set_count in range(dim_list.__len__()):
            dim         = dim_list        [param_set_count]
            _low_bound  = _low_bound_list [param_set_count]
            _high_bound = _high_bound_list[param_set_count]
            raw_result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                mat = (torch.rand(size=[dim,dim])-0.5)*2.
                raw_result[test_count] = _raw_log10_avg_safe(mat.reshape([1,-1]))#.reshape([-1])
                pass
            #prin("-1,1",dim,raw_result.max(),raw_result.min(),raw_result.mean())
            the_mean = raw_result.mean()
            the_mean>_low_bound and the_mean<_high_bound
            pass#for param set
        return 
    ____test____single_matrix_test____with_log10_avg_safe()

if "some useful test for you to build up intuition    slow and prin" and __DEBUG_ME__() and False:
    "to save your time: the last 3 tests are the most important."
    "c = a.matmal(b), the result is log10_c = log10_a + log10_b + ??? log10(mid_dim) + k"
    "if a is the result of randn or relu(randn), k is 0 to 0.12."
    "if a is the result of sigmoid(randn), k is 0 to 0.03."
    "now you have a precise way to evaluate what the hack is going on in your model."
    
    "You don't have to read the code. Code only helps you understand the result."
    "If you don't have much time, read the commends directly."
    "about the measurement. If abs().log10().mean() doesn't works, or doesn't work stably, then use my avg_log10_safe() instead."
    #if "tested" and False:
    
    "torch.ones, scan dim*dim matmal dim*dim, factor not in this part"
    # _dim_mid=[  100],c.shape=[100,   100])],  ,log10_of_a=0.00, log10_of_b=0.00,,,log10_of_c=2.00
    # _dim_mid=[ 1000],c.shape=[1000,  1000])], ,log10_of_a=0.00, log10_of_b=0.00,,,log10_of_c=3.00        
    # _dim_mid=[10000],c.shape=[10000, 10000])],,log10_of_a=0.00, log10_of_b=0.00,,,log10_of_c=4.00
    # log_c is log10(_dim_mid)
    # but since log_a and log_b are both 0, I guess log_c is basically log_a + log_b + log10(_dim_mid)
    for _dim in [1e2, 1e3, 1e4]:
        _dim = int(_dim)
        _dim1 = _dim
        _dim2 = _dim
        _dim_mid = _dim
        
        device = 'cuda'
        a = torch.ones(size=[_dim1,    _dim_mid], device=device)
        b = torch.ones(size=[_dim_mid, _dim2   ], device=device)
        c = a.matmul(b)
        log10_of_a = a.log10().mean().cpu()
        assert _tensor_equal(log10_of_a, torch.tensor(0.), 0.0001)
        log10_of_b = b.log10().mean().cpu()
        assert _tensor_equal(log10_of_b, torch.tensor(0.), 0.0001)
        assert _tensor_equal(log10_of_a, log10_of_b, 0.0001)
        log10_of_c = c.log10().mean().cpu()
        print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape}], _factor not in this test,,,log10_of_a={log10_of_a:.4f}, log10_of_b={log10_of_b:.4f},,,log10_of_c={log10_of_c:.4f}")
        _diff = math.log10(_dim_mid)
        assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
        pass


    "torch.ones, scan dim1*dim_mid matmal dim_mid*dim_2, factor not in this part"
    # log_a and log_b are both 0. log_c is log10(_dim_mid)
    for _dim in [1e2, 1e3, 1e4]:
        for _ in range(5):
            #dim
            _dim = int(_dim)
            _dim1 = _dim
            _dim2 = _dim
            _dim_mid = random.randint(100,10000)
            #init a and b
            device = 'cuda'
            a = torch.ones(size=[_dim1,    _dim_mid], device=device)
            b = torch.ones(size=[_dim_mid, _dim2   ], device=device)
            #calc and measure.  
            c = a.matmul(b)
            log10_of_a = a.log10().mean().cpu()
            assert _tensor_equal(log10_of_a, torch.tensor(0.), 0.0001)
            log10_of_b = b.log10().mean().cpu()
            assert _tensor_equal(log10_of_b, torch.tensor(0.), 0.0001)
            assert _tensor_equal(log10_of_a, log10_of_b, 0.0001)
            log10_of_c = c.log10().mean().cpu()
            #print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape}], _factor not in this test,,,log10_of_a={log10_of_a:.4f}, log10_of_b={log10_of_b:.4f},,,log10_of_c={log10_of_c:.4f}")
            _diff = math.log10(_dim_mid)
            assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
            pass#for _
        pass#for dim
            
        "all the test uppon are only with 1, no -1. "
    
    
        
        
        "torch.ones(with altered sign), scan dim1*dim_mid matmal dim_mid*dim_2, factor not in this part"
        # p_sign_b=            0.0       /       0.25       /      0.45      /       0.5

        # p_sign_a=0.0,   2.00,3.00,4.00    1.71,2.70,3.70    1.03,2.01,3.01    0.81,1.33,1.84
        # p_sign_a=0.1,   1.91,2.90,3.90    1.61,2.61,3.60    0.92,1.91,2.91    0.83,1.33,1.84
        # p_sign_a=0.2,   1.78,2.78,3.78    1.50,2.48,3.48    0.89,1.79,2.79    0.82,1.33,1.83
        # p_sign_a=0.3,   1.61,2.61,3.60    1.31,2.31,3.31    0.86,1.60,2.61    0.83,1.33,1.79
        # p_sign_a=0.4,   1.30,2.31,3.30    1.01,2.01,3.01    0.84,1.43,2.31    0.83,1.33,1.83
        # p_sign_a=0.5,   0.85,1.36,1.74    0.82,1.34,1.85    0.83,1.32,1.85    0.83,1.33,1.84

        # a 0.5, b 0. std a bit big.
        # notice, the most right colomn and the lowest row are basically the same.

        # the formual feels like:
        # the odd_of_sign = abs(p_sign-0.5)
        # some_factor = func(odd_of_sign_a, odd_of_sign_b), and it's range is [0.51, 1]
        # log10(dim_mid)*some_factor - (0 to 0.17)
        
        
        for p_sign_b in [0., 0.25, 0.45, 0.5]:
            for p_sign_a in [0., 0.1, 0.2, 0.3, 0.4, 0.5, ]:
                for _dim in [1e2, 1e3, 1e4]:
                    for test_index in range(3):
                        #dim
                        _dim = int(_dim)
                        _dim1 = _dim
                        _dim2 = _dim
                        _dim_mid = _dim# or ? _dim_mid = random.randint(100,10000)
                        #init a and b
                        device = 'cuda'
                        a = torch.rand(size=[_dim1,    _dim_mid], device=device)
                        a = a.gt(p_sign_a)
                        a = a.to(torch.float32)*2.-1.
                        log10_of_a = a.abs().log10().mean().cpu()
                        assert _tensor_equal(log10_of_a, torch.tensor(0.), 0.0001)
                        _tensor_equal(a.mean(), torch.tensor(p_sign_a), 0.05)
                        b = torch.rand(size=[_dim_mid, _dim2   ], device=device)
                        b = b.gt(p_sign_b)
                        b = b.to(torch.float32)*2.-1.
                        log10_of_b = b.abs().log10().mean().cpu()
                        assert _tensor_equal(log10_of_b, torch.tensor(0.), 0.0001)
                        _tensor_equal(b.mean(), torch.tensor(p_sign_b), 0.05)
                        assert _tensor_equal(log10_of_a, log10_of_b, 0.0001)
                        #calc and measure.
                        c = a.matmul(b)
                        c.add_(torch.randn_like(c)*0.001)#safety
                        log10_of_c = _raw_log10_avg_safe(c.reshape([1,-1])).cpu()
                        if test_index == 0:
                            print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape[0]:5},{c.shape[1]:5}], _factor not in this test,,,log10_of_a={log10_of_a:.4f}, p_sign_a={\
                                                    p_sign_a}, log10_of_b={log10_of_b:.4f}, p_sign_b={p_sign_b},,,log10_of_c(safe)={log10_of_c:.4f}", end="")
                            pass
                        else:
                            print(f", {log10_of_c:.4f}", end="")
                            pass
                        _diff = math.log10(_dim_mid)
                        #assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
                        pass#for test_index
                    print()
                    pass#for dim
    

        
        "a*factor @ b"
        # _dim_mid=[  100],c.shape=[  100,  100], factor=1.0  ,,,log10_of_a=0.00, log10_of_b=0.00,,,log10_of_c=2.00
        # _dim_mid=[ 1000],c.shape=[ 1000, 1000], factor=1.0  ,,,log10_of_a=0.00, log10_of_b=0.00,,,log10_of_c=3.00
        # _dim_mid=[10000],c.shape=[10000,10000], factor=1.0  ,,,log10_of_a=0.00, log10_of_b=0.00,,,log10_of_c=4.00
        # _dim_mid=[  100],c.shape=[  100,  100], factor=10.0 ,,,log10_of_a=1.00, log10_of_b=0.00,,,log10_of_c=3.00
        # _dim_mid=[ 1000],c.shape=[ 1000, 1000], factor=10.0 ,,,log10_of_a=1.00, log10_of_b=0.00,,,log10_of_c=4.00
        # _dim_mid=[10000],c.shape=[10000,10000], factor=10.0 ,,,log10_of_a=1.00, log10_of_b=0.00,,,log10_of_c=5.00
        # _dim_mid=[  100],c.shape=[  100,  100], factor=100.0,,,log10_of_a=2.00, log10_of_b=0.00,,,log10_of_c=4.00
        # _dim_mid=[ 1000],c.shape=[ 1000, 1000], factor=100.0,,,log10_of_a=2.00, log10_of_b=0.00,,,log10_of_c=5.00
        # _dim_mid=[10000],c.shape=[10000,10000], factor=100.0,,,log10_of_a=2.00, log10_of_b=0.00,,,log10_of_c=6.00
        # result is very obvious
        for factor in [1., 1e1, 1e2]:
            for _dim in [1e2, 1e3, 1e4]:
                #dim
                _dim = int(_dim)
                _dim1 = _dim
                _dim2 = _dim
                _dim_mid = _dim
                #init a and b
                device = 'cuda'
                a = torch.ones(size=[_dim1,    _dim_mid], device=device)*factor
                assert _tensor_equal(a.mean(), torch.tensor(factor), 0.0001)
                log10_of_a = a.log10().mean().cpu()
                assert _float_equal(log10_of_a, torch.log10(torch.tensor(factor)).item(), 0.0001)
                b = torch.ones(size=[_dim_mid, _dim2   ], device=device)
                assert _tensor_equal(b.mean(), torch.tensor(1.), 0.0001)
                log10_of_b = b.log10().mean().cpu()
                assert _float_equal(log10_of_b, 0., 0.0001)
                #calc and measure.
                c = a.matmul(b)
                log10_of_c = c.log10().mean().cpu()
                print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape[0]:5},{c.shape[1]:5}], factor={factor},,,log10_of_a={log10_of_a:.4f}, log10_of_b={log10_of_b:.4f},,,log10_of_c={log10_of_c:.4f}")
                _diff = math.log10(_dim_mid)
                assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
                pass
            
            
            
        "now, init with randn"
        "now, init with randn"
        "now, init with randn"
        
        
        "randn, d*d matmal d*d, fixed factor"
        # _dim_mid=[  100],c.shape=[  100,  100],,,log10_of_a=-0.1537, log10_of_b=-0.1463,,,log10_of_c=0.8318, 0.8388, 0.8290
        # _dim_mid=[ 1000],c.shape=[ 1000, 1000],,,log10_of_a=-0.1577, log10_of_b=-0.1580,,,log10_of_c=1.3457, 1.3485, 1.3366
        # _dim_mid=[10000],c.shape=[10000,10000],,,log10_of_a=-0.1589, log10_of_b=-0.1587,,,log10_of_c=1.8289, 1.8417, 1.8339
        # part of the result: log10(_dim_mid)*0.5
        # log_c is basically log_a + log_b + log10(mid_dim)*0.5 + 0.12
        for _dim in [1e2, 1e3, 1e4]:
            for test_index in range(3):
                #dim
                _dim = int(_dim)
                _dim1 = _dim
                _dim2 = _dim
                _dim_mid = _dim# or ? _dim_mid = random.randint(100,10000)
                #init a and b
                device = 'cuda'
                a = torch.randn(size=[_dim1,    _dim_mid], device=device)
                #log10_of_a = log10_avg_safe(a).mean().cpu().item()
                log10_of_a = _raw_log10_avg_safe(a.reshape([1,-1])).mean().cpu()
                assert _float_equal(log10_of_a, -0.16, 0.02)
                b = torch.randn(size=[_dim_mid, _dim2   ], device=device)
                #log10_of_b = log10_avg_safe(b).mean().cpu().item()
                log10_of_b = _raw_log10_avg_safe(b.reshape([1,-1])).mean().cpu()
                assert _float_equal(log10_of_b, -0.16, 0.02)
                assert _tensor_equal(log10_of_a, log10_of_b, 0.02)
                #calc and measure.
                c = a.matmul(b)
                log10_of_c = _raw_log10_avg_safe(c.reshape([1,-1])).mean().cpu()
                if test_index == 0:
                    print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape[0]:5},{c.shape[1]:5}], _factor not in this test,,,log10_of_a={\
                                            log10_of_a:.4f}, log10_of_b={log10_of_b:.4f},,,log10_of_c(safe)={log10_of_c:.4f}", end="")
                    pass
                else:
                    print(f", {log10_of_c:.4f}", end="")
                    pass
                _diff = math.log10(_dim_mid)
                #assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
                pass#for test_index
            print()
            pass#for dim
    
    
        
        "randn, relu(d*d) matmal d*d, fixed factor"
        # _dim_mid=[  100],c.shape=[  100,  100],,,log10_of_a=-0.2710, log10_of_b=-0.1492,,,log10_of_c(safe)=0.6937, 0.6744, 0.7044
        # _dim_mid=[ 1000],c.shape=[ 1000, 1000],,,log10_of_a=-0.2764, log10_of_b=-0.1575,,,log10_of_c(safe)=1.1920, 1.1999, 1.2043
        # _dim_mid=[10000],c.shape=[10000,10000],,,log10_of_a=-0.2760, log10_of_b=-0.1588,,,log10_of_c(safe)=1.6782, 1.6754, 1.6888
        # part of the result: log10(_dim_mid)*0.5
        # log_c is basically log_a + log_b + log10(mid_dim)*0.5 + 0.12
        _relu_layer = torch.nn.ReLU(inplace=True)
        if "scan the log10 of relu(randn) first" and False:
            for _ in range(10):
                a = _relu_layer(torch.randn(size=[10000,10000], device='cuda'))
                log10_of_a = _raw_log10_avg_safe(a, top_ratio=0.5, careful_level=1).mean().cpu().item()
                assert _float_equal(log10_of_a, -0.276, 0.001)
                pass
            pass
        for _dim in [1e2, 1e3, 1e4]:
            for test_index in range(3):
                #dim
                _dim = int(_dim)
                _dim1 = _dim
                _dim2 = _dim
                _dim_mid = _dim# or ? _dim_mid = random.randint(100,10000)
                #init a and b
                device = 'cuda'
                a = _relu_layer(torch.randn(size=[_dim1,    _dim_mid], device=device))
                #a).mean().cpu().item()
                log10_of_a = _raw_log10_avg_safe(a.reshape([1,-1])).mean().cpu()
                assert _float_equal(log10_of_a, -0.276, 0.01)#relu(randn) is -0.276
                b =             torch.randn(size=[_dim_mid, _dim2   ], device=device)
                #log10_of_b = log10_avg_safe(b).mean().cpu().item()
                log10_of_b = _raw_log10_avg_safe(b.reshape([1,-1])).mean().cpu()
                assert _float_equal(log10_of_b, -0.16, 0.02)
                #calc and measure.
                c = a.matmul(b)
                log10_of_c = _raw_log10_avg_safe(c.reshape([1,-1])).mean().cpu()
                if test_index == 0:
                    print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape[0]:5},{c.shape[1]:5}], _factor not in this test,,,log10_of_a={\
                                            log10_of_a:.4f}, log10_of_b={log10_of_b:.4f},,,log10_of_c(safe)={log10_of_c:.4f}", end="")
                    pass
                else:
                    print(f", {log10_of_c:.4f}", end="")
                    pass
                _diff = math.log10(_dim_mid)
                #assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
                pass#for test_index
            print()
            pass#for dim
    
        
        
        "randn, sigmoid(d*d) matmal d*d, fixed factor"
        # _dim_mid=[  100],c.shape=[  100,  100],,,log10_of_a=-0.2927, log10_of_b=-0.1448,,,log10_of_c(safe)=0.5133, 0.6773, 0.5937
        # _dim_mid=[ 1000],c.shape=[ 1000, 1000],,,log10_of_a=-0.2962, log10_of_b=-0.1580,,,log10_of_c(safe)=1.0885, 1.0662, 1.0841
        # _dim_mid=[10000],c.shape=[10000,10000],,,log10_of_a=-0.2962, log10_of_b=-0.1589,,,log10_of_c(safe)=1.5677, 1.5899, 1.5763
        # part of the result: log10(_dim_mid)*0.5
        # log_c is basically log_a + log_b + log10(mid_dim)*0.5 + (0. to 0.03)
        
        _sigmoid_layer = torch.nn.Sigmoid()
        if "scan the log10 of sigmoid(randn) first" and False:
            for _ in range(10):
                a = _sigmoid_layer(torch.randn(size=[10000,10000], device='cuda'))
                log10_of_a = _raw_log10_avg_safe(a.reshape([1,-1])).mean().cpu().item()
                assert _float_equal(log10_of_a, -0.2963, 0.0001)
                pass
            pass
        for _dim in [1e2, 1e3, 1e4]:
            for test_index in range(3):
                #dim
                _dim = int(_dim)
                _dim1 = _dim
                _dim2 = _dim
                _dim_mid = _dim# or ? _dim_mid = random.randint(100,10000)
                #init a and b
                device = 'cuda'
                a = _sigmoid_layer(torch.randn(size=[_dim1,    _dim_mid], device=device))
                #log10_of_a = log10_avg_safe(a).mean().cpu().item()
                log10_of_a = _raw_log10_avg_safe(a.reshape([1,-1])).mean().cpu()
                assert _float_equal(log10_of_a, -0.296, 0.005)#sigmoid(randn) is -0.276
                b =                torch.randn(size=[_dim_mid, _dim2   ], device=device)
                #log10_of_b = log10_avg_safe(b).mean().cpu().item()
                log10_of_b = _raw_log10_avg_safe(b.reshape([1,-1])).mean().cpu()
                assert _float_equal(log10_of_b, -0.16, 0.02)
                #calc and measure.
                c = a.matmul(b)
                log10_of_c = _raw_log10_avg_safe(c.reshape([1,-1])).mean().cpu()
                if test_index == 0:
                    print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape[0]:5},{c.shape[1]:5}], _factor not in this test,,,log10_of_a={\
                                            log10_of_a:.4f}, log10_of_b={log10_of_b:.4f},,,log10_of_c(safe)={log10_of_c:.4f}", end="")
                    pass
                else:
                    print(f", {log10_of_c:.4f}", end="")
                    pass
                _diff = math.log10(_dim_mid)
                #assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
                pass#for test_index
            print()
            pass#for dim
    
    
    
        

        "randn, sigmoid(d*d scale with some scale_factor) matmal d*d, fixed factor"
        
        _sigmoid_layer = torch.nn.Sigmoid()
            
        #                                *****
        #     log_c measured by top ratio 0.9, while log_a measured by top ratio 0.5
        #                                *****                                      dim= 100 ,1000 ,10000
        # _scale_factor=0.01,,,log10_of_a=-0.2993, log10_of_b=-0.157,,,log10_of_c(safe)=0.512,1.047,1.545
        # _scale_factor=0.1 ,,,log10_of_a=-0.2844, log10_of_b=-0.156,,,log10_of_c(safe)=0.519,1.043,1.544
        # _scale_factor=0.3 ,,,log10_of_a=-0.2552, log10_of_b=-0.151,,,log10_of_c(safe)=0.554,1.053,1.546
        # _scale_factor=1.0 ,,,log10_of_a=-0.1789, log10_of_b=-0.158,,,log10_of_c(safe)=0.588,1.072,1.578
        # _scale_factor=3.0 ,,,log10_of_a=-0.0889, log10_of_b=-0.144,,,log10_of_c(safe)=0.630,1.141,1.637
        # _scale_factor=10.0,,,log10_of_a=-0.0378, log10_of_b=-0.151,,,log10_of_c(safe)=0.662,1.163,1.667
            
            
        #                                *****
        #     log_c measured by top ratio 0.7, while log_a measured by top ratio 0.5
        #                                *****                                        dim= 100 ,1000  ,10000
        # _scale_factor=0.01,,,log10_of_a=-0.2993, log10_of_b=-0.1542,,,log10_of_c(safe)=0.6601,1.1709,1.6703
        # _scale_factor=0.1 ,,,log10_of_a=-0.2840, log10_of_b=-0.1527,,,log10_of_c(safe)=0.6889,1.1751,1.6703
        # _scale_factor=0.3 ,,,log10_of_a=-0.2554, log10_of_b=-0.1538,,,log10_of_c(safe)=0.6497,1.1772,1.6746
        # _scale_factor=1.0 ,,,log10_of_a=-0.1795, log10_of_b=-0.1573,,,log10_of_c(safe)=0.6786,1.2031,1.7081
        # _scale_factor=3.0 ,,,log10_of_a=-0.0843, log10_of_b=-0.1503,,,log10_of_c(safe)=0.7520,1.2642,1.7624
        # _scale_factor=10.0,,,log10_of_a=-0.0389, log10_of_b=-0.1579,,,log10_of_c(safe)=0.8014,1.3051,1.8075
            
            
        #                                *****
        #     log_c measured by top ratio 0.6, while log_a measured by top ratio 0.5
        #                                *****                                     dim= 100 ,1000 ,10000
        # _scale_factor=0.01,,,log10_of_a=-0.299, log10_of_b=-0.151,,,log10_of_c(safe)=0.745,1.221,1.721,
        # _scale_factor=0.1 ,,,log10_of_a=-0.284, log10_of_b=-0.153,,,log10_of_c(safe)=0.711,1.232,1.724,
        # _scale_factor=0.3 ,,,log10_of_a=-0.254, log10_of_b=-0.151,,,log10_of_c(safe)=0.700,1.231,1.728,
        # _scale_factor=1.0 ,,,log10_of_a=-0.178, log10_of_b=-0.154,,,log10_of_c(safe)=0.752,1.252,1.760,
        # _scale_factor=3.0 ,,,log10_of_a=-0.093, log10_of_b=-0.156,,,log10_of_c(safe)=0.807,1.317,1.822,
        # _scale_factor=10.0,,,log10_of_a=-0.038, log10_of_b=-0.157,,,log10_of_c(safe)=0.869,1.356,1.860,
            
            
        #                                *****
        #     log_c measured by top ratio 0.5, while log_a measured by top ratio 0.5
        #                                *****                                       dim= 100 ,1000  ,10000
        # _scale_factor=0.01,,,log10_of_a=-0.2993, log10_of_b=-0.157,,,log10_of_c(safe)=0.7767,1.2704,1.7751
        # _scale_factor=0.1 ,,,log10_of_a=-0.2840, log10_of_b=-0.151,,,log10_of_c(safe)=0.7726,1.2707,1.7747
        # _scale_factor=0.3 ,,,log10_of_a=-0.2541, log10_of_b=-0.152,,,log10_of_c(safe)=0.7748,1.2821,1.7756
        # _scale_factor=1.0 ,,,log10_of_a=-0.1765, log10_of_b=-0.153,,,log10_of_c(safe)=0.8146,1.3037,1.8106
        # _scale_factor=3.0 ,,,log10_of_a=-0.0866, log10_of_b=-0.157,,,log10_of_c(safe)=0.8471,1.3664,1.8664
        # _scale_factor=10.0,,,log10_of_a=-0.0403, log10_of_b=-0.152,,,log10_of_c(safe)=0.8996,1.4057,1.9053
        
        # log_c is basically 0.5*log_a + log_b + 0.5*log10(mid_dim) + 0.3, 
        # condition is log_c is measured with top_ratio 0.5, if top_ratio increases by 0.1, the 0.3 also incr by 0.05(and gets 0.35)
        
        if "some extra result" and False:
            # here is some extra result about the log10(sigmoid(randn* some factor))
            #                     top_ratio=     0.9,,,               =0.5,,,
            # _scale_factor=0.01,    log10_of_a=-0.3006    log10_of_a=-0.2993
            # _scale_factor=0.1 ,    log10_of_a=-0.2972    log10_of_a=-0.2843
            # _scale_factor=0.3 ,    log10_of_a=-0.2920    log10_of_a=-0.2539
            # _scale_factor=1.0 ,    log10_of_a=-0.296     log10_of_a=-0.1766
            # _scale_factor=3.0 ,    log10_of_a=-0.419     log10_of_a=-0.085 
            # _scale_factor=10.0,    log10_of_a=-1.11      log10_of_a=-0.025 
            # the upper the smaller the std. The lower the greater the std.
            # when the top_ratio is 0.9, a lot elements are between 0 and 0.5. When the _scale_factor
            # increases, they get close to 0, and makes the log10 result smaller.
            # Thus, I also tested with top_ratio=0.5, to only consider the elements >0.5 and increases.
            pass
        if 'some basic test' and False:
            for top_ratio in [0.9, 0.5]:
                for _scale_factor in [0.01, 0.1, 0.3, 1., 3., 10.]:
                    if "scan the log10 of sigmoid(randn) first" and True:
                        for test_index in range(5):
                            a_before_sigmoid = torch.randn(size=[10000,10000], device='cuda')*_scale_factor
                            _std_of_a_before_sigmoid = a_before_sigmoid.std().cpu().item()
                            assert _float_equal(_std_of_a_before_sigmoid, _scale_factor, 0.05*_scale_factor)
                            a = _sigmoid_layer(a_before_sigmoid)
                            #log10_of_a = log10_avg_safe(a, top_ratio).mean().cpu().item()
                            log10_of_a = _raw_log10_avg_safe(a.reshape([1,-1]), top_ratio).mean().cpu().item()
                            if test_index == 0:
                                print(f"top_ratio={top_ratio},,, _scale_factor={_scale_factor}, log10_of_a={log10_of_a:.6f}", end="")
                                pass
                            else:
                                print(f", {log10_of_a:.6f}", end="")
                                pass 
                            #assert _float_equal(log10_of_a, -0.2963, 0.0001)
                            pass# test_index
                        print()
                        pass
                    pass
        
        #then the real test.
        for _scale_factor in [0.01, 0.1, 0.3, 1., 3., 10.]:
            for _dim in [1e2, 1e3, 1e4]:
                _result_list = []
                for test_index in range(5):
                    #dim
                    _dim = int(_dim)
                    _dim1 = _dim
                    _dim2 = _dim
                    _dim_mid = _dim# or ? _dim_mid = random.randint(100,10000)
                    #init a and b
                    device = 'cuda'
                    a = _sigmoid_layer(torch.randn(size=[_dim1,    _dim_mid], device=device)*_scale_factor)
                    #log10_of_a = log10_avg_safe(a, top_ratio=0.5).mean().cpu().item()
                    log10_of_a = _raw_log10_avg_safe(a.reshape([1,-1]), top_ratio=0.5).mean().cpu()
                    #assert _float_equal(log10_of_a, -0.296, 0.005)#sigmoid(randn) is -0.276
                    b =                torch.randn(size=[_dim_mid, _dim2   ], device=device)
                    #log10_of_b = log10_avg_safe(b).mean().cpu().item()
                    log10_of_b = _raw_log10_avg_safe(b.reshape([1,-1])).mean().cpu()
                    assert _float_equal(log10_of_b, -0.16, 0.02)
                    #calc and measure.
                    c = a.matmul(b)
                    log10_of_c = _raw_log10_avg_safe(c.reshape([1,-1]), top_ratio=0.6).mean().cpu()
                    if test_index == 0:
                        print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape[0]:5},{c.shape[1]:5}], _scale_factor={_scale_factor},,,log10_of_a={\
                                                log10_of_a:.3f}, log10_of_b={log10_of_b:.3f},,,log10_of_c(safe)={log10_of_c:.3f}", end="")
                        pass
                    else:
                        print(f", {log10_of_c:.4f}", end="")
                        pass
                    _result_list.append(log10_of_c)
                    #_diff = math.log10(_dim_mid)
                    #assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
                    pass#for test_index
                print(f",,,,,avg={torch.tensor(_result_list).mean().item():.3f}")
                pass#for dim
    
    
    
    
    
    
    
    
    

    "a is uniform distribution, d*d matmal d*d, fixed factor"
    # _dim_mid=[  100],c.shape=[  100,  100],,,log10_of_a=-0.319, log10_of_b=-0.153,,,log10_of_c(safe)=avg=0.786
    # _dim_mid=[ 1000],c.shape=[ 1000, 1000],,,log10_of_a=-0.323, log10_of_b=-0.157,,,log10_of_c(safe)=avg=1.282
    # _dim_mid=[10000],c.shape=[10000,10000],,,log10_of_a=-0.324, log10_of_b=-0.159,,,log10_of_c(safe)=avg=1.787
    # log_c is basically log_a + log_b + 0.5*log10(mid_dim) + 0.27, 
    
    if 'some basic test' and False:
        for test_index in range(5):
            a = torch.rand(10000, 10000)*2.-1.
            #log10_of_a = log10_avg_safe(a).mean().cpu().item()
            log10_of_a = _raw_log10_avg_safe(a.reshape([1,-1])).mean().cpu().item()
            assert _float_equal(log10_of_a, -0.3235, 0.0003)
            if test_index == 0:
                print(f"log10_of_a={log10_of_a:.6f}", end="")
                pass
            else:
                print(f", {log10_of_a:.6f}", end="")
                pass 
            pass# test_index
    
    #then the real test.
    for _dim in [1e2, 1e3, 1e4]:
        _result_list = []
        for test_index in range(7):
            #dim
            _dim = int(_dim)
            _dim1 = _dim
            _dim2 = _dim
            _dim_mid = _dim# or ? _dim_mid = random.randint(100,10000)
            #init a and b
            device = 'cuda'
            a = torch.rand (size=[_dim1,    _dim_mid], device=device)*2.-1.
            #log10_of_a = log10_avg_safe(a).mean().cpu().item()
            log10_of_a = _raw_log10_avg_safe(a.reshape([1,-1])).mean().cpu()
            assert _float_equal(log10_of_a, -0.3235, 0.05)
            b = torch.randn(size=[_dim_mid, _dim2   ], device=device)
            #log10_of_b = log10_avg_safe(b).mean().cpu().item()
            log10_of_b = _raw_log10_avg_safe(b.reshape([1,-1])).mean().cpu()
            assert _float_equal(log10_of_b, -0.16, 0.02)
            #calc and measure.
            c = a.matmul(b)
            log10_of_c = _raw_log10_avg_safe(c.reshape([1,-1]), top_ratio=0.6).mean().cpu()
            if test_index == 0:
                print(f"_dim_mid=[{_dim_mid:5}],c.shape=[{c.shape[0]:5},{c.shape[1]:5}],,,log10_of_a={\
                                        log10_of_a:.3f}, log10_of_b={log10_of_b:.3f},,,log10_of_c(safe)={log10_of_c:.3f}", end="")
                pass
            else:
                print(f", {log10_of_c:.4f}", end="")
                pass
            _result_list.append(log10_of_c)
            #_diff = math.log10(_dim_mid)
            #assert _tensor_equal(log10_of_a+_diff, log10_of_c, 0.0001)
            pass#for test_index
        print(f",,,,,avg={torch.tensor(_result_list).mean().item():.3f}")
        pass#for dim
    
    
    pass


if "xxxxxxxxxxxxx          param strength measurement functions. Maybe reopen after maintainance." and False:

    # backup code.
    # def debug_get_zero_grad_ratio(self, directly_print_out:float = False)->float:
    #     with torch.no_grad():
    #         result = 0.
    #         if not self.raw_weight_o_i.grad is None:
    #             flags = self.raw_weight_o_i.grad.eq(0.)
    #             total_amount = flags.sum().item()
    #             result = float(total_amount)/self.raw_weight_o_i.nelement()
    #         if directly_print_out:
    #             print("get_zero_grad_ratio:", result)
    #         return result


    # def debug_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01, \
    #                             print_out = False)->float:
    #     #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
    #     if self.raw_weight_o_i.grad is None:
    #         if print_out:
    #             print(0., "inside debug_micro_grad_ratio function __line 1082")
    #             pass
    #         return 0.

    #     the_device=self.raw_weight_o_i.device
    #     epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
    #     raw_weight_abs = self.raw_weight_o_i.abs()
    #     flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

    #     epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
    #     raw_weight_grad_abs = self.raw_weight_o_i.grad.abs()
    #     flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

    #     ten = torch.tensor([10.], device=the_device)
    #     log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
    #     corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
    #     flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

    #     flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
    #     result = (flag_useful_g.sum().to(torch.float32)/self.raw_weight_o_i.nelement()).item()
    #     if print_out:
    #         print(result, "inside debug_micro_grad_ratio function __line 1082")
    #         pass
    #     return result


    # def debug_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01, \
    #                             print_out = False)->float:
    #     #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
    #     if self.raw_weight_o_i.grad is None:
    #         if print_out:
    #             print(0., "inside debug_micro_grad_ratio function __line 1082")
    #             pass
    #         return 0.

    #     the_device=self.raw_weight_o_i.device
    #     epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
    #     raw_weight_abs = self.raw_weight_o_i.abs()
    #     flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

    #     epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
    #     raw_weight_grad_abs = self.raw_weight_o_i.grad.abs()
    #     flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

    #     ten = torch.tensor([10.], device=the_device)
    #     log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
    #     corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
    #     flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

    #     flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
    #     result = (flag_useful_g.sum().to(torch.float32)/self.raw_weight_o_i.nelement()).item()
    #     if print_out:
    #         print(result, "inside debug_micro_grad_ratio function __line 1082")
    #         pass
    #     return result
    pass


