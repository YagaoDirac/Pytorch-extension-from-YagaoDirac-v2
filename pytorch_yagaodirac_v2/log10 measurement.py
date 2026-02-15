from typing import List, Tuple, Optional, TypeGuard
import torch
import math, random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pytorch_yagaodirac_v2.Util import   log10_avg_safe, \
                    _tensor_equal, str_the_list
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

# new log10 measurement -0.275
# ref: raw -0.158   batch -0.270   raw batch -0.160
if "all the measurements" and False:
    def all_the_measurements():
        from pytorch_yagaodirac_v2.Util import   log10_avg_safe,      log10_avg_safe__with_batch, \
                                            _raw_log10_avg_safe, _raw_log10_avg_safe__with_batch
        test_time = 1000
        dim = 100000
        result_of______log10 = torch.empty(size=[test_time])
        result_of__raw_log10 = torch.empty(size=[test_time])
        result_of______log10_with_batch = torch.empty(size=[test_time])
        result_of__raw_log10_with_batch = torch.empty(size=[test_time])
        
        for test_count in range(test_time):
            the_randn = torch.randn(size=[dim])
            result_of______log10[test_count] =      log10_avg_safe(the_randn)
            result_of__raw_log10[test_count] = _raw_log10_avg_safe(the_randn)
            result_of______log10_with_batch[test_count] =      log10_avg_safe__with_batch(the_randn)
            result_of__raw_log10_with_batch[test_count] = _raw_log10_avg_safe__with_batch(the_randn)
            pass
        print(f". { result_of______log10.mean().item():.3f}   raw {\
                    result_of__raw_log10.mean().item():.3f}   batch {\
                    result_of______log10_with_batch.mean().item():.3f}   raw batch {\
                    result_of__raw_log10_with_batch.mean().item():.3f}")
        
        return 
    
    all_the_measurements()
    





# randn    >>> -0.275 111111111
# rand     >>> -0.422
# rand*2-1 >>> -0.422
# softmax(randn(dim)*scaling_factor) >>> @       scaling_factor(sf) < 0.4 >>> -1.*log10(dim)
# softmax(randn(dim)*scaling_factor) >>> @ 0.4 < scaling_factor(sf) < 5   >>> in between.
# softmax(randn(dim)*scaling_factor) >>> @   5 < scaling_factor(sf)       >>> -sf*(0.72+0.25*log10(dim))  (approximately)

if "basic random number gen test" and False:
    # result
    # the measurement is only about the avg order of magnitude of elements.
    # it's a per element measurement.
    # the shape, or the total number of elements don't affect the result.
    # randn >>> -0.275
    # rand  >>> -0.422
    # rand*2-1 has the same abs as rand, so they also have the same log10 measurement result.
    
    def basic_random_number_gen():
        if "1d randn" and False:
            print("1d randn")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            if TESTING:
                test_time_list = [10000,3000,3000]
                pass
            else:
                test_time_list = [100,30,30]
                pass
            dim_segment_list = [100,1000,10000,100000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.449, -0.319, -0.297]
                the_max_lt_this_list =[-0.142, -0.234, -0.253]
                the_mean_eq_this_list=[-0.276, -0.275, -0.275]
                epsilon_list         =[ 0.184,  0.053,  0.032]
                pass
            
            for param_set_count in range(test_time_list.__len__()):
                test_time = test_time_list[param_set_count]
                dim_from = dim_segment_list[param_set_count]
                dim_to = dim_segment_list[param_set_count+1]#+1
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
                    the_randn = torch.randn(size=[dim], device=device)
                    _this_result = log10_avg_safe(the_randn)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "2d randn" and False:
            print("2d randn")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            if TESTING:
                test_time_list = [1000,50]
                pass
            else:
                test_time_list = [10,1]
                pass
            dim_segment_list = [100,1000,10000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.296, -0.286]
                the_max_lt_this_list =[-0.254, -0.265]
                the_mean_eq_this_list=[-0.275, -0.275]
                epsilon_list         =[ 0.032,  0.020]
                pass
            
            for param_set_count in range(test_time_list.__len__()):
                test_time = test_time_list[param_set_count]
                dim_from = dim_segment_list[param_set_count]
                dim_to = dim_segment_list[param_set_count+1]#+1
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
                    the_randn = torch.randn(size=[dim,dim], device=device)
                    _this_result = log10_avg_safe(the_randn)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "rand" and False:
            print("rand")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            if TESTING:
                test_time_list = [10000,3000,3000]
                pass
            else:
                test_time_list = [100,30,30]
                pass
            dim_segment_list = [100,1000,10000,100000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.581, -0.472, -0.442]
                the_max_lt_this_list =[-0.282, -0.376, -0.399]
                the_mean_eq_this_list=[-0.422, -0.422, -0.422]
                epsilon_list         =[ 0.169,  0.060,  0.033]
                pass
            
            for param_set_count in range(test_time_list.__len__()):
                test_time = test_time_list[param_set_count]
                dim_from = dim_segment_list[param_set_count]
                dim_to = dim_segment_list[param_set_count+1]#+1
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
                    the_randn = torch.rand(size=[dim], device=device)
                    _this_result = log10_avg_safe(the_randn)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "rand*2-1" and False:
            print("rand*2-1")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            if TESTING:
                test_time_list = [10000,3000,3000]
                pass
            else:
                test_time_list = [100,30,30]
                pass
            dim_segment_list = [100,1000,10000,100000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.583, -0.466, -0.443]
                the_max_lt_this_list =[-0.293, -0.376, -0.400]
                the_mean_eq_this_list=[-0.423, -0.422, -0.422]
                epsilon_list         =[ 0.170,  0.056,  0.032]
                pass
            
            for param_set_count in range(test_time_list.__len__()):
                test_time = test_time_list[param_set_count]
                dim_from = dim_segment_list[param_set_count]
                dim_to = dim_segment_list[param_set_count+1]#+1
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
                    the_randn = torch.rand(size=[dim], device=device)*2.-1.
                    _this_result = log10_avg_safe(the_randn)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "softmax(dim)" and False:
            print("softmax(dim)")
            TESTING = True
            device = 'cuda'
            
            #--------------------#--------------------#--------------------
            dim_list = [1000,10000,100000]
            if TESTING:
                test_time_list = [10000,3000,500]
                pass
            else:
                test_time_list = [100,30,30]
                pass
            #--------------------#--------------------#--------------------
            for macro_iter_count in range(3):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                #--------------------#--------------------#--------------------
                #scaling_factor_list = [0.001, 0.01,  0.1, 0.2,  0.3, 0.4, 0.5, 0.6 , 0.7, 0.8, 0.9,   1.,  1.5,  2.,  2.5,  3.,  3.5,  4.,  4.5,  5.,  5.5,  6.,  6.5,  ]
                #scaling_factor_list = [0.4, 0.7,  math.log(10.,10.),math.log(20.,10.),math.log(30.,10.),math.log(40.,10.),math.log(50.,10.), math.log(100.,10.),math.log(200.,10.),math.log(300.,10.),math.log(400.,10.),math.log(500.,10.), ]
                scaling_factor_list = [0.001, 0.01,  0.1, 0.3, 0.4, 0.5, 1.,  1.5,  2.,  2.5,  3., 5., 10., 20., 30., 50.]
                #--------------------#--------------------#--------------------
                if TESTING:
                    print(test_time_list)
                    the_min_gt_this_list =  []#don't modify here.
                    the_max_lt_this_list =  []
                    the_mean_eq_this_list = []
                    epsilon_list =          []
                    pass
                else:
                    ###########################################  result paste to here.
                    if dim == 1000:
                        the_min_gt_this_list =[-3.010, -3.011, -3.018, -3.049, -3.071, -3.098, -3.324, -4.047, -4.735, -6.058, -6.934, -11.475, -24.536, -46.469, -67.667, -115.066]
                        the_max_lt_this_list =[-2.990, -2.990, -2.995, -3.016, -3.033, -3.051, -3.201, -3.423, -3.717, -4.048, -4.438, -6.161, -10.866, -21.340, -31.970, -54.477]
                        the_mean_eq_this_list=[-3.000, -3.000, -3.006, -3.032, -3.051, -3.074, -3.257, -3.547, -3.940, -4.423, -4.982, -7.553, -14.595, -29.041, -43.542, -72.430]
                        epsilon_list       =[ 0.020,  0.020,  0.022,  0.028,  0.030,  0.034,  0.077,  0.510,  0.804,  1.646,  1.961,  3.932,  9.951,  17.439,  24.135,  42.647]
                        #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.300,  0.400,  0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  5.000,  10.000,  20.000,  30.000,  50.000]
                        pass
                    if dim == 10000:
                        the_min_gt_this_list =[-4.010, -4.010, -4.017, -4.043, -4.063, -4.088, -4.284, -4.612, -5.224, -6.295, -7.670, -12.615, -24.950, -49.448, -72.679, -113.403]
                        the_max_lt_this_list =[-3.990, -3.990, -3.996, -4.020, -4.038, -4.060, -4.234, -4.501, -4.855, -5.286, -5.761, -8.128, -14.670, -27.991, -42.472, -72.221]
                        the_mean_eq_this_list=[-4.000, -4.000, -4.006, -4.032, -4.051, -4.074, -4.257, -4.548, -4.947, -5.452, -6.050, -8.993, -17.358, -34.382, -51.485, -85.462]
                        epsilon_list       =[ 0.020,  0.020,  0.020,  0.022,  0.023,  0.024,  0.037,  0.074,  0.287,  0.853,  1.631,  3.631,  7.602,  15.075,  21.205,  27.951]
                        #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.300,  0.400,  0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  5.000,  10.000,  20.000,  30.000,  50.000]
                        pass
                    if dim == 100000:
                        the_min_gt_this_list =[-5.010, -5.010, -5.016, -5.042, -5.061, -5.085, -5.270, -5.570, -5.987, -6.890, -7.356, -12.205, -25.577, -48.578, -75.383, -128.328]
                        the_max_lt_this_list =[-4.990, -4.990, -4.996, -5.021, -5.040, -5.063, -5.244, -5.528, -5.915, -6.397, -6.948, -9.686, -17.536, -34.025, -51.688, -84.894]
                        the_mean_eq_this_list=[-5.000, -5.000, -5.006, -5.032, -5.051, -5.074, -5.257, -5.548, -5.948, -6.458, -7.066, -10.290, -19.629, -39.034, -58.247, -97.628]
                        epsilon_list       =[ 0.020,  0.020,  0.020,  0.021,  0.021,  0.021,  0.024,  0.032,  0.049,  0.442,  0.300,  1.925,  5.958,  9.554,  17.146,  30.710]
                        #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.300,  0.400,  0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  5.000,  10.000,  20.000,  30.000,  50.000]
                        pass
                    
                    pass
                
                for param_set_count in range(scaling_factor_list.__len__()):
                    scaling_factor = scaling_factor_list[param_set_count]
                    if not TESTING:
                        the_min_gt_this = the_min_gt_this_list  [param_set_count]
                        the_max_lt_this = the_max_lt_this_list  [param_set_count]
                        the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                        epsilon = epsilon_list  [param_set_count]
                        pass
                    
                    _raw_result = torch.empty(size=[test_time])
                    for test_count in range(test_time):
                        #--------------------#--------------------#--------------------
                        the_randn = torch.randn(size=[dim], dtype=torch.float64)*scaling_factor
                        after_softmax = the_randn.softmax(dim=0)
                        assert after_softmax.shape == torch.Size([dim])
                        _this_result = log10_avg_safe(after_softmax)
                        #--------------------#--------------------#--------------------
                        _raw_result[test_count] = _this_result
                        pass
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
                        print(f"dim:{dim}   sf{scaling_factor}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                        pass
                    else:
                        assert the_min>the_min_gt_this
                        assert the_max<the_max_lt_this
                        assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                        pass
                    pass#for param_set_count
                if TESTING:
                    print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                    print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                    print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                    print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                    print(f"#scaling_factor_list ={str_the_list(scaling_factor_list,         3)}")    
                    pass
                pass# for dim
            pass#/test
        
        return 
    basic_random_number_gen()
    pass


# @ means matmul.
# rand is [0,1), rand*2-1 is [-1,1)
# randn   [??,mid dim] @ randn   [mid dim,????], -0.275, -0.275, into a  0.5*log10(mid dim)-0.275        
# rand*2-1[??,mid dim] @ rand*2-1[mid dim,????], -0.422, -0.422, into a  0.5*log10(mid dim)-0.753
# rand*2-1[??,mid dim] @ rand    [mid dim,????], -0.422, -0.422, into a  0.5*log10(mid dim)-0.751       
# rand    [??,mid dim] @ rand    [mid dim,????], -0.422, -0.422, into a  1. *log10(mid dim)-0.602
# 2 different matrixs  
# randn   [??,mid dim] @ rand*2-1[mid dim,????], -0.275, -0.422, into a  0.5*log10(mid dim)-0.515
# randn   [??,mid dim] @ rand    [mid dim,????], -0.275, -0.422, into a  0.5*log10(mid dim)-0.514

if "multiplication" and False:
    # result
    # randn @ randn, 2 -0.275s, into a 0.5*log(mid dim)-0.275.
    # so basically, the -0.16 is a global offset.
    # if you add 0.16 to all the measurement, it's regular. 
    # this applies to mat@mat, vec@mat, and vec dot vec, if it's normal distribution.
    
    # But, out side randn, all the cases are slightly different. 
    # If all the sign of elements are the same(for both matrixs), it's a 1*log10, 
    # but if half is pos and half is neg, it's always 0.5*log10.
    # the last offset varies across cases.
    
    
    def basic_multiplication():
        #old code
        #for config_set_count in range(dim_list.__len__()):
        
        if "randn[dim,dim] @ randn[dim,dim]" and False:
            print("randn[dim,dim] @ randn[dim,dim]")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            if TESTING:
                test_time_list = [10000,3000,10]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.682,  1.212,  1.714]
                the_max_lt_this_list =[ 0.756,  1.236,  1.735]
                the_mean_eq_this_list=[ 0.722,  1.224,  1.725]
                epsilon_list         =[ 0.050,  0.022,  0.020]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                dim = dim_list[param_set_count]
                test_time = test_time_list[param_set_count]
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
                    _this_result = log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test count
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
                    print(f"dim:{dim}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                print(f"#dim_list            ={    str_the_list(dim_list,         3)}")    
                pass
            pass#/test
        
        if "randn[???,dim] @ randn[dim,????]" and False:
            print("randn[dim,dim] @ randn[dim,dim]")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,1000 ,10000,]
            rand_dim_from_list = [200  ,200  ,1000 ,]
            rand_dim_to_list =   [5000 ,5000 ,5000 ,]
            if TESTING:
                #test_time_list = [1000,300,100]
                test_time_list = [500,500,50]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.704,  1.211,  1.714]
                the_max_lt_this_list =[ 0.737,  1.237,  1.735]
                the_mean_eq_this_list=[ 0.722,  1.224,  1.725]
                epsilon_list         =[ 0.028,  0.023,  0.020]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                test_time = test_time_list[param_set_count]
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
                    _this_result = log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test count
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
                    print(f"dim:{mid_dim}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                print(f"#mid_dim_list        ={    str_the_list(mid_dim_list,         3)}")    
                pass
            pass#/test
        
        if "rand*2-1[???,dim] @ rand*2-1[dim,????]" and False:
            print("rand*2-1[???,dim] @ rand*2-1[dim,????]")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,1000 ,10000,]
            rand_dim_from_list = [200  ,200  ,1000 ,]
            rand_dim_to_list =   [5000 ,5000 ,5000 ,]
            if TESTING:
                #test_time_list = [1000,300,100]
                test_time_list = [500,500,50]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.235,  0.735,  1.237]
                the_max_lt_this_list =[ 0.261,  0.759,  1.258]
                the_mean_eq_this_list=[ 0.247,  0.747,  1.247]
                epsilon_list         =[ 0.023,  0.023,  0.021]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                test_time = test_time_list[param_set_count]
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
                    mat1 = torch.rand(size=[__rand_dim, mid_dim   ], device = device)*2.-1.
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand(size=[mid_dim,    __rand_dim], device = device)*2.-1.
                    prod = mat1@mat2
                    _this_result = log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test count
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
                    print(f"dim:{mid_dim}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                print(f"#mid_dim_list        ={    str_the_list(mid_dim_list,         3)}")    
                pass
            pass#/test
        
        if "rand*2-1[???,dim] @ rand[dim,????]" and False:
            print("rand*2-1[???,dim] @ rand[dim,????]")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,1000 ,10000,]
            rand_dim_from_list = [200  ,200  ,1000 ,]
            rand_dim_to_list =   [5000 ,5000 ,5000 ,]
            if TESTING:
                #test_time_list = [1000,300,100]
                test_time_list = [500,500,50]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.211,  0.700,  1.230]
                the_max_lt_this_list =[ 0.281,  0.803,  1.269]
                the_mean_eq_this_list=[ 0.247,  0.748,  1.249]
                epsilon_list         =[ 0.046,  0.065,  0.030]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                test_time = test_time_list[param_set_count]
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
                    mat1 = torch.rand(size=[__rand_dim, mid_dim   ], device = device)*2.-1.
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand(size=[mid_dim,    __rand_dim], device = device)
                    prod = mat1@mat2
                    _this_result = log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test count
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
                    print(f"dim:{mid_dim}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                print(f"#mid_dim_list        ={    str_the_list(mid_dim_list,         3)}")    
                pass
            pass#/test
        
        if "rand[???,dim] @ rand[dim,????]" and False:
            print("rand*2-1[???,dim] @ rand[dim,????]")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,1000 ,10000,]
            rand_dim_from_list = [200  ,200  ,1000 ,]
            rand_dim_to_list =   [5000 ,5000 ,5000 ,]
            if TESTING:
                #test_time_list = [1000,300,100]
                test_time_list = [500,500,50]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 1.379,  2.386,  3.387]
                the_max_lt_this_list =[ 1.407,  2.408,  3.408]
                the_mean_eq_this_list=[ 1.393,  2.397,  3.398]
                epsilon_list         =[ 0.024,  0.021,  0.020]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                test_time = test_time_list[param_set_count]
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
                    mat1 = torch.rand(size=[__rand_dim, mid_dim   ], device = device)
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand(size=[mid_dim,    __rand_dim], device = device)
                    prod = mat1@mat2
                    _this_result = log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test count
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
                    print(f"dim:{mid_dim}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                print(f"#mid_dim_list        ={    str_the_list(mid_dim_list,         3)}")    
                pass
            pass#/test
        
        if "randn[???,dim] @ rand*2-1[dim,????]" and False:
            print("randn[???,dim] @ rand*2-1[dim,????]")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,1000 ,10000,]
            rand_dim_from_list = [200  ,200  ,1000 ,]
            rand_dim_to_list =   [5000 ,5000 ,5000 ,]
            if TESTING:
                #test_time_list = [1000,300,100]
                test_time_list = [500,500,50]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.470,  0.973,  1.475]
                the_max_lt_this_list =[ 0.500,  0.997,  1.496]
                the_mean_eq_this_list=[ 0.485,  0.986,  1.486]
                epsilon_list         =[ 0.025,  0.022,  0.021]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                mid_dim = mid_dim_list[param_set_count]
                rand_dim_from = rand_dim_from_list [param_set_count] 
                rand_dim_to   = rand_dim_to_list   [param_set_count] 
                test_time = test_time_list[param_set_count]
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
                    mat1 = torch.randn(size=[__rand_dim, mid_dim   ], device = device)
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand (size=[mid_dim,    __rand_dim], device = device)*2-1
                    prod = mat1@mat2
                    _this_result = log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test count
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
                    print(f"dim:{mid_dim}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                print(f"#mid_dim_list        ={    str_the_list(mid_dim_list,         3)}")    
                pass
            pass#/test
        
        if "randn[???,dim] @ rand[dim,????]" and False:
            print("randn[???,dim] @ rand[dim,????]")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            mid_dim_list =       [100  ,1000 ,10000,]
            rand_dim_from_list = [200  ,200  ,1000 ,]
            rand_dim_to_list =   [5000 ,5000 ,5000 ,]
            if TESTING:
                test_time_list = [500,500,50]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.443,  0.930,  1.463]
                the_max_lt_this_list =[ 0.520,  1.020,  1.508]
                the_mean_eq_this_list=[ 0.485,  0.986,  1.486]
                epsilon_list         =[ 0.051,  0.067,  0.033]
                pass
            
            for param_set_count in range(mid_dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    mat1 = torch.randn(size=[__rand_dim, mid_dim   ], device = device)
                    __rand_dim = random.randint(rand_dim_from, rand_dim_to)
                    mat2 = torch.rand (size=[mid_dim,    __rand_dim], device = device)
                    prod = mat1@mat2
                    _this_result = log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass#for test count
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
                    print(f"dim:{mid_dim}  ///  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                print(f"#mid_dim_list        ={    str_the_list(mid_dim_list,         3)}")    
                pass
            pass#/test
        
        return 
    basic_multiplication()
    pass





#########1111111111

# relu   (randn) >>> -0.576* . Relu provides too many 0s. I infer it as -0.275+log10(0.5), -0.576
# gelu   (randn) >>> -0.728
# sigmoid(randn) >>> -0.356
# tanh   (randn) >>> -0.353
# sin(randn*scale_factor) is complicated. For sf(scale_factor) <0.1, it's log10(sf)-0.16, the same as randn, 
#        bc input is close to 0, sin(x) is very close to x. For sf>1.8, it's -0.20(or -0.197). But in between(0.1 to 1.8), measurement is between -1.159 and -0.197.
# cos(randn*scale_factor) is the same. For sf<0.06, x is very close to 0, cos(x) is basically 1, so the measurement is 0.
#        When sf>1.9, it's -0.20(or -0.197)(same as sin). But in between(0.06 to 1.9), measurement is between 0 and -0.197.
# gauss is not tested.

# randn_vec @ relu   (randn) >>> -0.16 and -0.46*, into  0.5*log10(mid dim)-0.31
# randn_vec @ gelu   (randn) >>> -0.16 and -0.60 , into  0.5*log10(mid dim)-0.34
# randn_vec @ sigmoid(randn) >>> -0.16 and -0.30 , into  0.5*log10(mid dim)-0.42
# randn_vec @ tanh   (randn) >>> -0.16 and -0.30 , into  0.5*log10(mid dim)-0.36
# randn_vec @ sin(randn*scale_factor), @       sf < 0.1 >>> -0.16 and log10(sf)-0.16               , into  0.5*log10(mid dim) + log10(sf) - 0.16, the same as randn@randn
# randn_vec @ sin(randn*scale_factor), @ 0.1 < sf < 1.6 >>> (idk the formula, check result in code), into  0.5*log10(mid dim) - some_function(sf) but basically between log10(sf)-0.16 to -0.31
# randn_vec @ sin(randn*scale_factor), @ 1.6 < sf       >>> -0.16 and -0.197                       , into  0.5*log10(mid dim) - 0.31
# randn_vec @ cos(randn*scale_factor), @       sf < 0.1 >>> -0.16 and 0                            , into  0.5*log10(mid dim) - 0.16
# randn_vec @ cos(randn*scale_factor), @ 0.1 < sf < 1.6 >>> (idk the formula, check result in code), into  0.5*log10(mid dim) - some_function(sf) but basically between -0.16 to -0.31
# randn_vec @ cos(randn*scale_factor), @ 1.6 < sf       >>> -0.16 and -0.197                       , into  0.5*log10(mid dim) - 0.31

if "with activation functions" and True:
    # randn, then act, @ another randn.
    # relu,gelu,(swiss)
    # sigmoid, tanh,, 
    # sin, cos, 
    # gauss?
    def with_activation_functions():
        
        if "gelu(randn[100000])" and False:
            print("gelu(randn[100000])")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            if TESTING:
                test_time_list = [10000]
                pass
            else:
                test_time_list = [10]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.743]
                the_max_lt_this_list =[-0.713]
                the_mean_eq_this_list=[-0.728]
                epsilon_list         =[ 0.025]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    _this_result = log10_avg_safe(vec)
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
                print(f"epsilon_list       ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "sigmoid(randn[100000])" and False:
            print("sigmoid(randn[100000])")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            if TESTING:
                test_time_list = [10000]
                pass
            else:
                test_time_list = [10]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.369]
                the_max_lt_this_list =[-0.343]
                the_mean_eq_this_list=[-0.356]
                epsilon_list         =[ 0.023]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    _this_result = log10_avg_safe(vec)
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
                print(f"epsilon_list       ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "tanh(randn[100000])" and False:
            print("tanh(randn[100000])")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            if TESTING:
                test_time_list = [10000]
                pass
            else:
                test_time_list = [10]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-0.368]
                the_max_lt_this_list =[-0.337]
                the_mean_eq_this_list=[-0.353]
                epsilon_list         =[ 0.026]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    _this_result = log10_avg_safe(vec)
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
                print(f"epsilon_list       ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        
        
        
        if "tanh(randn[100000])" and True:
            print("tanh(randn[100000])")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            scale_factor_list = [0.00001, 0.0001, 0.001, 0.01, 0.0316,0.1,0.316,1.,1.3,1.6,1.7,1.8,2.,5.,10.]
            
            dim_list = [100000]*scale_factor_list.__len__()
            if TESTING:
                test_time_list = [10000]*scale_factor_list.__len__()
                pass
            else:
                test_time_list = [10]*scale_factor_list.__len__()  1w
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    _this_result = log10_avg_safe(vec)
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
                print(f"epsilon_list       ={         str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        
        fds=432
        
        
        
        
        
        
        
        
        
        
        
        
        
        if "sin(randn[100000]*scale_factor) " and False:
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
                    _this_result = log10_avg_safe(vec)
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
        
        if "cos(randn[100000]*scale_factor) " and False:
            TESTING = False
            if TESTING:
                print("cos(randn[100000]*scale_factor) >>> ???")
                pass
            test_time = 100
            device = 'cuda'
            #--------------------#--------------------#--------------------
            #scale_factor_list = [0.00001, 0.0001, 0.001, 0.01, 0.0316,0.1,0.316,1.,1.3,1.6,1.7,1.8,2.,5.,10.] sin
            scale_factor_list = [ 0.047,  0.056,  0.068,    0.1,  0.316,     1.,    1.6,    1.8,   1.85,    1.9,   1.95,     2.]
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
                the_min_gt_this_list =[-0.010, -0.010, -0.011, -0.011, -0.024, -0.160, -0.208, -0.210, -0.210, -0.210, -0.211, -0.210]
                the_max_lt_this_list =[ 0.010,  0.010,  0.009,  0.009, -0.004, -0.133, -0.181, -0.182, -0.183, -0.183, -0.183, -0.183]
                the_mean_eq_this_list=[-0.000, -0.000, -0.001, -0.001, -0.014, -0.147, -0.195, -0.196, -0.196, -0.197, -0.197, -0.197]
                epsilon_list         =[ 0.020,  0.020,  0.020,  0.020,  0.020,  0.024,  0.024,  0.024,  0.023,  0.024,  0.024,  0.024]
                # scale_factor_list = [ 0.047,  0.056,  0.068,    0.1,  0.316,     1.,    1.6,    1.8,   1.85,    1.9,   1.95,     2.]
                #                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                
                
                
                
                
                
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
                    vec = vec.cos()
                    _this_result = log10_avg_safe(vec)
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
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        
        if "relu(randn[mid dim]) @ randn[mid dim,???] " and False:
            TESTING = False
            if TESTING:
                print("relu(randn[mid dim]) @ randn[mid dim,???] >>> ???")
                pass
            test_time = 10
            device = 'cuda'
            #--------------------#--------------------#--------------------
            #dim_list = [100,1000]
            #dim_list = [10000]
            dim_list = [100,1000,10000]
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
                the_min_gt_this_list =[ 0.484,  1.124,  1.669]
                the_max_lt_this_list =[ 0.912,  1.271,  1.709]
                the_mean_eq_this_list=[ 0.682,  1.191,  1.691]
                epsilon_list         =[ 0.240,  0.090,  0.031]
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
                    vec = torch.randn(size=[dim])
                    loss_layer = torch.nn.ReLU()
                    vec = loss_layer(vec)
                    mat = torch.randn(size=[dim, dim])
                    prod = vec@mat
                    _this_result = log10_avg_safe(prod)
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "gelu(randn[mid dim]) @ randn[mid dim,???] " and False:
            TESTING = False
            if TESTING:
                print("gelu(randn[mid dim]) @ randn[mid dim,???] >>> ???")
                pass
            device = 'cuda'
            #test_time_list = [3000,1000,100]
            test_time_list = [30,10,5]
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.364,  1.084,  1.621]
                the_max_lt_this_list =[ 0.861,  1.241,  1.689]
                the_mean_eq_this_list=[ 0.646,  1.157,  1.656]
                epsilon_list         =[ 0.292,  0.094,  0.045]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    vec = torch.randn(size=[dim])
                    loss_layer = torch.nn.GELU()
                    vec = loss_layer(vec)
                    mat = torch.randn(size=[dim, dim])
                    prod = vec@mat
                    _this_result = log10_avg_safe(prod)
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "sigmoid(randn[mid dim]) @ randn[mid dim,???] " and False:
            TESTING = False
            if TESTING:
                print("sigmoid(randn[mid dim]) @ randn[mid dim,???] >>> ???")
                pass
            device = 'cuda'
            test_time_list = [3000,1000,100]
            test_time_list = [30,10,5]
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.394,  1.020,  1.555]
                the_max_lt_this_list =[ 0.743,  1.129,  1.596]
                the_mean_eq_this_list=[ 0.572,  1.076,  1.575]
                epsilon_list         =[ 0.187,  0.066,  0.031]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    vec = torch.randn(size=[dim])
                    loss_layer = torch.nn.Sigmoid()
                    vec = loss_layer(vec)
                    mat = torch.randn(size=[dim, dim])
                    prod = vec@mat
                    _this_result = log10_avg_safe(prod)
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "tanh(randn[mid dim]) @ randn[mid dim,???] " and False:
            TESTING = False
            if TESTING:
                print("tanh(randn[mid dim]) @ randn[mid dim,???] >>> ???")
                pass
            device = 'cuda'
            test_time_list = [3000,1000,100]
            test_time_list = [30,10,5]
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[ 0.457,  1.085,  1.618]
                the_max_lt_this_list =[ 0.799,  1.188,  1.658]
                the_mean_eq_this_list=[ 0.637,  1.140,  1.640]
                epsilon_list         =[ 0.189,  0.066,  0.031]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    vec = torch.randn(size=[dim])
                    loss_layer = torch.nn.Tanh()
                    vec = loss_layer(vec)
                    mat = torch.randn(size=[dim, dim])
                    prod = vec@mat
                    _this_result = log10_avg_safe(prod)
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list     ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "sin(randn[dim]*scale_factor) @ randn[dim,dim] " and False:
            dim = 10000
            if True:
                TESTING = False
                if TESTING:
                    print("sin(randn[dim]*scale_factor) @ randn[dim,dim] >>> ???")
                    pass
                test_time = 100
                device = 'cuda'
                #--------------------#--------------------#--------------------
                if dim == 100:
                    #the 0.1  and 1.7
                    scale_factor_list =[ 0.0001, 0.001, 0.01,  0.02,    0.1,    0.2,    0.6,   1.,  1.5,  1.6,  1.7, 1.8,1.9,  2.,   3.]
                    pass
                if dim == 1000:
                    #the 0.1  and 1.6
                    scale_factor_list =[ 0.0001, 0.001, 0.01,  0.02,    0.1,    0.2,    0.6,   1.,  1.5,  1.6,  1.7,   2.,   3.]
                    pass
                if dim == 10000:
                    #????
                    scale_factor_list =[ 0.0001, 0.001, 0.01,  0.02,    0.1,    0.2,    0.6,   1.,  1.5,  1.6,  1.7, 1.8,1.9,  2.,   3.]
                    pass
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
                    if dim == 100:
                        the_min_gt_this_list =[-3.372, -2.399, -1.374, -1.074, -0.377, -0.095,  0.359,  0.452,  0.456,  0.492,  0.503,  0.539,  0.501,  0.495,  0.513]
                        the_max_lt_this_list =[-2.992, -1.975, -0.980, -0.675,  0.000,  0.307,  0.720,  0.796,  0.838,  0.856,  0.842,  0.847,  0.863,  0.832,  0.837]
                        the_mean_eq_this_list=[-3.161, -2.160, -1.162, -0.861, -0.164,  0.129,  0.544,  0.658,  0.685,  0.687,  0.688,  0.688,  0.689,  0.687,  0.689]
                        epsilon_list         =[ 0.221,  0.248,  0.223,  0.223,  0.223,  0.235,  0.195,  0.217,  0.239,  0.206,  0.195,  0.169,  0.198,  0.202,  0.186]
                        #scale_factor_list   =[0.0001,  0.001,   0.01,   0.02,    0.1,    0.2,    0.6,     1.,    1.5,    1.6,    1.7,    1.8,    1.9,  2.,   3.]
                        #                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    pass
                    if dim == 1000:
                        the_min_gt_this_list =[-2.718, -1.729, -0.725, -0.419,  0.277,  0.575,  0.984,  1.104,  1.133,  1.138,  1.135,  1.127,  1.141]
                        the_max_lt_this_list =[-2.599, -1.597, -0.602, -0.298,  0.402,  0.696,  1.105,  1.214,  1.241,  1.250,  1.241,  1.244,  1.243]
                        the_mean_eq_this_list=[-2.658, -1.658, -0.659, -0.358,  0.339,  0.634,  1.046,  1.159,  1.189,  1.191,  1.191,  1.191,  1.191]
                        epsilon_list         =[ 0.070,  0.081,  0.076,  0.071,  0.073,  0.072,  0.072,  0.065,  0.066,  0.070,  0.066,  0.074,  0.062]
                        #scale_factor_list  =[ 0.0001,  0.001,   0.01,   0.02,    0.1,    0.2,    0.6,     1.,    1.5,    1.6,    1.7,     2.,     3.]
                        #                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass
                    if dim == 10000:
                        the_min_gt_this_list =[-2.182, -1.183, -0.179,  0.119,  0.816,  1.112,  1.527,  1.636,  1.667,  1.669,  1.669,  1.672,  1.671,  1.669,  1.670]
                        the_max_lt_this_list =[-2.138, -1.134, -0.133,  0.162,  0.861,  1.153,  1.566,  1.683,  1.708,  1.710,  1.711,  1.712,  1.709,  1.712,  1.711]
                        the_mean_eq_this_list=[-2.158, -1.158, -0.158,  0.143,  0.838,  1.134,  1.546,  1.660,  1.689,  1.691,  1.690,  1.691,  1.691,  1.691,  1.691]
                        epsilon_list         =[ 0.034,  0.035,  0.035,  0.034,  0.032,  0.032,  0.030,  0.034,  0.033,  0.031,  0.031,  0.031,  0.030,  0.032,  0.031]
                        #scale_factor_list   =[0.0001,  0.001,   0.01,   0.02,    0.1,    0.2,    0.6,     1.,    1.5,    1.6,    1.7,    1.8,    1.9,  2.,   3.]
                        #                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass
                    pass
                
                for param_set_count in range(scale_factor_list.__len__()):
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
                        vec = torch.randn(size=[dim])*scale_factor
                        vec = vec.sin()
                        #loss_layer = torch.nn.Tanh()
                        #vec = loss_layer(vec)
                        mat = torch.randn(size=[dim, dim])
                        prod = vec@mat
                        _this_result = log10_avg_safe(prod)
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
        
        if "cos(randn[dim]*scale_factor) @ randn[dim,dim] " and True:
            dim = 10000
            if True:
                TESTING = True
                if TESTING:
                    print("cos(randn[dim]*scale_factor) @ randn[dim,dim] >>> ???")
                    pass
                test_time = 200
                device = 'cuda'
                #--------------------#--------------------#--------------------
                if dim == 100:
                    #the 0.1  and 1.4
                    scale_factor_list =[0.001,  0.01,   0.06,    0.1,    0.2,     1.,   1.3,   1.4,   1.5,    2.,     5]
                    pass
                if dim == 1000:
                    #the 0.1  and 1.4
                    scale_factor_list =[0.001,  0.01,   0.06,    0.1,    0.2,     1.,   1.3,   1.4,   1.5,   1.6,   1.7,    2.,     5]
                    pass
                if dim == 10000:
                    #the 0.1  and 1.4
                    scale_factor_list =[0.001,  0.01,   0.06,    0.1,    0.2,     1.,   1.3,   1.4,   1.5,   1.6,   1.7,    2.,     5]
                    pass
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
                    if dim == 100:
                        the_min_gt_this_list =[ 0.683,  0.659,  0.688,  0.673,  0.678,  0.532,  0.522,  0.527,  0.521,  0.479,  0.530]
                        the_max_lt_this_list =[ 0.982,  0.971,  0.976,  0.975,  0.978,  0.852,  0.841,  0.880,  0.839,  0.844,  0.857]
                        the_mean_eq_this_list=[ 0.839,  0.839,  0.840,  0.837,  0.831,  0.716,  0.696,  0.693,  0.691,  0.688,  0.689]
                        epsilon_list         =[ 0.166,  0.190,  0.162,  0.174,  0.162,  0.194,  0.184,  0.197,  0.180,  0.219,  0.178]
                        #scale_factor_list   =[ 0.001,  0.010,  0.060,  0.100,  0.200,  1.000,  1.300,  1.400,  1.500,  2.000,  5.000]
                        #                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass
                    if dim == 1000:
                        the_min_gt_this_list =[ 1.293,  1.292,  1.287,  1.280,  1.285,  1.168,  1.148,  1.139,  1.135,  1.133,  1.137,  1.134,  1.136]
                        the_max_lt_this_list =[ 1.394,  1.394,  1.385,  1.392,  1.396,  1.280,  1.263,  1.251,  1.244,  1.249,  1.248,  1.249,  1.249]
                        the_mean_eq_this_list=[ 1.342,  1.342,  1.341,  1.340,  1.333,  1.218,  1.198,  1.195,  1.193,  1.193,  1.191,  1.192,  1.192]
                        epsilon_list         =[ 0.062,  0.062,  0.063,  0.070,  0.073,  0.071,  0.076,  0.067,  0.068,  0.070,  0.067,  0.067,  0.067]
                        #scale_factor_list   =[ 0.001,  0.010,  0.060,  0.100,  0.200,  1.000,  1.300,  1.400,  1.500,  1.600,  1.700,  2.000,  5.000]
                        #                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass
                    if dim == 10000:
                        the_min_gt_this_list =[ 1.823,  1.823,  1.819,  1.820,  1.811,  1.694,  1.676,  1.669,  1.673,  1.671,  1.670,  1.668,  1.669]
                        the_max_lt_this_list =[ 1.864,  1.862,  1.862,  1.861,  1.855,  1.741,  1.720,  1.717,  1.715,  1.715,  1.712,  1.712,  1.711]
                        the_mean_eq_this_list=[ 1.842,  1.841,  1.842,  1.840,  1.833,  1.719,  1.699,  1.695,  1.694,  1.693,  1.692,  1.691,  1.692]
                        epsilon_list         =[ 0.032,  0.031,  0.033,  0.031,  0.032,  0.035,  0.033,  0.036,  0.032,  0.032,  0.032,  0.033,  0.032]
                        #scale_factor_list   =[ 0.001,  0.010,  0.060,  0.100,  0.200,  1.000,  1.300,  1.400,  1.500,  1.600,  1.700,  2.000,  5.000]
                        #                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass
                    pass
                
                for param_set_count in range(scale_factor_list.__len__()):
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
                        vec = torch.randn(size=[dim])*scale_factor
                        vec = vec.cos()
                        #loss_layer = torch.nn.Tanh()
                        #vec = loss_layer(vec)
                        mat = torch.randn(size=[dim, dim])
                        prod = vec@mat
                        _this_result = log10_avg_safe(prod)
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
                    print(f"epsilon_list        ={     str_the_list(epsilon_list,         3)}")    
                    print(f"scale_factor_list    ={str_the_list(scale_factor_list,         3)}")    
                    pass
                pass#/test
            
        return 
    with_activation_functions()
    pass



# softmax(randn[mid dim]*scaling_factor) @ randn[mid dim,??]
# if scaling_factor < 0.1 >>> -0.5 *log10(dim) - 0.16 (it feels like the randn@randn but the dim makes it smaller.)
# if scaling_factor == 1. >>> -0.5 *log10(dim) + 0.04 
# if scaling_factor == 2. >>> -0.36*log10(dim) + 0.07
# if scaling_factor > 5   >>> -0.02*log10(dim) - 0.20  (basically -0.24 to -0.28)

if "softmax @ randn" and False:
    def softmax_matmul_randn():
        if "softmax @ randn" and True:
            dim = 10000
            TESTING = True
            if TESTING:
                print("softmax @ randn >>> ???")
                pass
            device = 'cuda'
            #--------------------#--------------------#--------------------
            if dim == 100:
                # 0.1, ~5
                test_time = 3000
                scaling_factor_list = [0.001, 0.01,  0.1,  0.4,  0.5,   1.,   2.,   3.,  5.,  10.,]
                #scaling_factor_list = [0.1,  0.4,  0.5,   1.,   2.,   3.,  5.,  10.,]
                pass
            if dim == 1000:
                test_time = 300
                scaling_factor_list = [0.001, 0.01,  0.1,  0.4,  0.5,   1.,   2.,   3.,  5.,  10.,]
                pass
            if dim == 10000:
                test_time = 100
                scaling_factor_list = [0.001, 0.01,  0.1,  0.4,  0.5,   1.,   2.,   3.,  5.,  10.,]
                pass
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                _softmax_ref_list =          []
                pass
            else:
                ###########################################  result paste to here.
                if dim == 100:
                    the_min_gt_this_list =[-1.331, -1.334, -1.344, -1.301, -1.289, -1.203, -1.034, -0.877, -0.742, -0.715]
                    the_max_lt_this_list =[-1.002, -1.010, -0.993, -0.977, -0.966, -0.468, -0.120, -0.074, -0.045, -0.009]
                    the_mean_eq_this_list=[-1.160, -1.160, -1.158, -1.126, -1.107, -0.964, -0.655, -0.477, -0.333, -0.237]
                    epsilon_list       =[ 0.181,  0.184,  0.196,  0.185,  0.192,  0.507,  0.545,  0.413,  0.419,  0.488]
                    #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.400,  0.500,  1.000,  2.000,  3.000,  5.000,  10.000]
                    #_softmax_ref_list   =[-2.000, -1.999, -1.994, -2.001, -2.012, -2.131, -2.652, -3.422, -5.245, -10.150]
                    pass
                if dim == 1000:
                    the_min_gt_this_list =[-1.708, -1.702, -1.708, -1.669, -1.653, -1.524, -1.253, -1.042, -0.791, -0.625]
                    the_max_lt_this_list =[-1.618, -1.614, -1.603, -1.577, -1.553, -1.279, -0.436, -0.217, -0.133, -0.116]
                    the_mean_eq_this_list=[-1.660, -1.658, -1.656, -1.624, -1.605, -1.444, -0.990, -0.663, -0.414, -0.265]
                    epsilon_list       =[ 0.058,  0.054,  0.063,  0.057,  0.062,  0.175,  0.564,  0.455,  0.387,  0.370]
                    #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.400,  0.500,  1.000,  2.000,  3.000,  5.000,  10.000]
                    #_softmax_ref_list   =[-3.000, -2.999, -2.994, -3.001, -3.012, -3.132, -3.689, -4.603, -6.913, -13.379]
                    pass
                if dim == 10000:
                    the_min_gt_this_list =[-2.177, -2.179, -2.175, -2.142, -2.125, -1.976, -1.590, -1.312, -0.885, -0.605]
                    the_max_lt_this_list =[-2.140, -2.139, -2.136, -2.102, -2.078, -1.896, -0.481, -0.233, -0.155, -0.143]
                    the_mean_eq_this_list=[-2.158, -2.158, -2.156, -2.123, -2.104, -1.944, -1.372, -0.923, -0.511, -0.279]
                    epsilon_list       =[ 0.029,  0.031,  0.030,  0.031,  0.036,  0.058,  0.900,  0.700,  0.384,  0.336]
                    #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.400,  0.500,  1.000,  2.000,  3.000,  5.000,  10.000]
                    #_softmax_ref_list   =[-4.000, -3.999, -3.994, -4.001, -4.012, -4.132, -4.699, -5.674, -8.335, -16.100]
                    pass
                pass
            
            for param_set_count in range(scaling_factor_list.__len__()):
                scaling_factor = scaling_factor_list[param_set_count]
                #test_time = test_time_list[param_set_count]
                #dim = dim_list[param_set_count]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                
                _raw_result_of__softmax = torch.empty(size=[test_time])
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    the_randn = torch.randn(size=[dim], dtype=torch.float64, device=device)*scaling_factor
                    softmax_vec = the_randn.softmax(dim=0)
                    mat = torch.randn(size=[dim,dim], dtype=torch.float64, device=device)
                    prod = softmax_vec@mat
                    assert prod.dtype == torch.float64
                    _this_result = log10_avg_safe(prod)
                    #--------------------#--------------------#--------------------
                    _raw_result_of__softmax[test_count] = log10_avg_safe(softmax_vec)
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    
                    _softmax_ref_list.append(_raw_result_of__softmax.mean().item())    
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                print(f"#_softmax_ref_list   ={str_the_list(_softmax_ref_list,         3)}")    
                print(f"#scaling_factor_list ={str_the_list(scaling_factor_list,         3)}")    
                pass
            pass#/test
        
        return
    
    softmax_matmul_randn()
    pass









# K He init >>> -0.5*log10(dim) -0.32


if "K He init" and True:
    def K_He_init():
        
        if "K He init [dim]" and False:
            TESTING = False
            if TESTING:
                print("K He init [dim] >>> ???")
                pass
            device = 'cuda'
            #--------------------#--------------------#--------------------
            test_time_list = [10000,3000,2000]
            dim_list = [1000,10000,100000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-1.876, -2.348, -2.836]
                the_max_lt_this_list =[-1.775, -2.301, -2.810]
                the_mean_eq_this_list=[-1.823, -2.323, -2.823]
                epsilon_list         =[ 0.063,  0.034,  0.023]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    _dummy_layer = torch.nn.Linear(in_features=dim, out_features=1, bias=False, device=device)
                    _this_result = log10_avg_safe(_dummy_layer.weight.data)
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "K He init [dim,dim]" and True:
            TESTING = False
            if TESTING:
                print("K He init [dim,dim] >>> ???")
                pass
            device = 'cuda'
            #--------------------#--------------------#--------------------
            test_time_list = [5000,1000,30]
            dim_list = [100,1000,10000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-1.345, -1.834, -2.333]
                the_max_lt_this_list =[-1.300, -1.812, -2.313]
                the_mean_eq_this_list=[-1.323, -1.823, -2.323]
                epsilon_list         =[ 0.033,  0.021,  0.020]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
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
                    _dummy_layer = torch.nn.Linear(in_features=dim, out_features=dim, bias=False, device=device)
                    _this_result = log10_avg_safe(_dummy_layer.weight.data)
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        
        
        
        
        
        
        
        if "randn and randn @ K He init" and True:
            TESTING = True
            if TESTING:
                print("K He init [dim,dim] >>> ???")
                pass
            device = 'cuda'
            #--------------------#--------------------#--------------------
            test_time_list = [5000,1000,30]
            dim_list = [100,1000,10000]
            #--------------------#--------------------#--------------------
            if TESTING:
                print(test_time_list)
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []
                the_mean_eq_this_list = []
                epsilon_list =          []
                pass
            else:
                ###########################################  result paste to here.
                the_min_gt_this_list =[-1.345, -1.834, -2.333]
                the_max_lt_this_list =[-1.300, -1.812, -2.313]
                the_mean_eq_this_list=[-1.323, -1.823, -2.323]
                epsilon_list         =[ 0.033,  0.021,  0.020]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                test_time = test_time_list[param_set_count]
                dim = dim_list[param_set_count]
                if not TESTING:
                    the_min_gt_this = the_min_gt_this_list  [param_set_count]
                    the_max_lt_this = the_max_lt_this_list  [param_set_count]
                    the_mean_eq_this = the_mean_eq_this_list[param_set_count]
                    epsilon = epsilon_list  [param_set_count]
                    pass
                
                _raw_ref = torch.empty(size=[test_time])
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    vec_before = torch.randn(size=[dim])
                    _dummy_layer = torch.nn.Linear(in_features=dim, out_features=dim, bias=False, device=device)
                    vec_after = _dummy_layer(vec_before)
                    _this_result = log10_avg_safe(_dummy_layer.weight.data)
                    #--------------------#--------------------#--------------------
                    assert False, "jixu."
                    _raw_ref[test_count] = log10_avg_safe(vec_before)
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
                    print(f"dim:{dim},   {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
                    pass
                else:
                    assert the_min>the_min_gt_this
                    assert the_max<the_max_lt_this
                    assert _tensor_equal(the_mean, [the_mean_eq_this], epsilon = epsilon)
                    pass
                pass#for param_set_count
            if TESTING:
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        
        return
    K_He_init()













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


