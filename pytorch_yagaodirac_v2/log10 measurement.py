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



# randn    >>> -0.275
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
            for macro_iter_count in range(dim_list.__len__()):
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



# sf = scaling_factor
# relu   (randn) >>> -0.576* . Relu provides too many 0s. I infer it as -0.275+log10(0.5), -0.576

# gelu   (randn) >>>       sf < 0.1 >>> log10(sf) -0.577   (< -1.58)
# gelu   (randn) >>> 0.1 < sf < 4   >>> between -1.58 and -0.82
# gelu   (randn) >>> 4   < sf       >>> 1.9*log10(sf) -2.1 (> -0.82)

# sigmoid(randn) >>>       sf < 0.1 >>>  -0.301         (log10(0.5))
# sigmoid(randn) >>> 0.1 < sf < 10  >>>  between -0.301 and -1.732
# sigmoid(randn) >>> 10  < sf       >>>  sf * -0.171    (<-1.732)

# notice that, the measurement of tanh and sigmoid is very different.
# tanh   (randn) >>>       sf < 0.1 >>>  log10(sf)-0.275   (<-1.277)
# tanh   (randn) >>> 0.1 < sf < 10  >>>  between -1.277 to 0
# tanh   (randn) >>> 10  < sf       >>>  0

# sin(randn*sf) >>>       sf < 0.1 >>> 1.*log10(sf) -0.275 (-inf to -1.276)(the same as randn)
# sin(randn*sf) >>> 0.1 < sf < 1.8 >>> from -1.276 to -0.277
# sin(randn*sf) >>> 1.8 < sf       >>> -0.277
# For sf<0.1, input is close to 0, sin(x) is very close to x.

# cos(randn*sf) >>>       sf < 0.05 >>> 0
# cos(randn*sf) >>> 0.05 < sf < 1.8 >>> from 0 to -0.277
# cos(randn*sf) >>> 1.8 < sf        >>> -0.277
# For sf<0.05, x is very close to 0, cos(x) is basically 1, so the measurement is 0.

# gauss is not tested.
if "with activation functions" and False:
    def with_activation_functions():
        
        if "gelu(randn[100000])" and False:
            print("gelu(randn[100000])")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            if TESTING:
                test_time_list = [1000]
                pass
            else:
                test_time_list = [10]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                scaling_factor_list = [0.0005,0.001,0.002,0.005,0.01, 0.02, 0.1,0.2, 0.5,   1.,  3.16,4,5,10,31.6,40,50, 100.]
                #                            0.1                           4
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
                    if dim == 100000:
                        the_min_gt_this_list =[-3.892, -3.592, -3.290, -2.893, -2.591, -2.291, -1.592, -1.296, -0.924, -0.742, -0.849, -0.849, -0.781, -0.225,  0.946,  1.126,  1.286,  1.655]
                        the_max_lt_this_list =[-3.862, -3.561, -3.261, -2.863, -2.562, -2.261, -1.561, -1.265, -0.894, -0.713, -0.792, -0.790, -0.722, -0.148,  1.002,  1.186,  1.331,  1.694]
                        the_mean_eq_this_list=[-3.877, -3.576, -3.275, -2.878, -2.577, -2.276, -1.578, -1.280, -0.909, -0.728, -0.821, -0.819, -0.752, -0.186,  0.972,  1.156,  1.308,  1.675]
                        epsilon_list         =[ 0.025,  0.026,  0.025,  0.025,  0.025,  0.025,  0.026,  0.026,  0.025,  0.024,  0.039,  0.040,  0.040,  0.048,  0.040,  0.041,  0.033,  0.030]
                        #scaling_factor_list =[0.0005,  0.001,  0.002,  0.005,  0.01 ,  0.02 ,  0.1  ,  0.2  ,  0.5  ,  1.   ,  3.16 ,  4.   ,  5.   ,  10.  ,  31.6 ,  40.  ,  50.  ,  100. ]
                        
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
                        loss_layer = torch.nn.GELU()
                        vec = torch.randn(size=[dim],device=device)*scaling_factor
                        vec = loss_layer(vec)
                        _this_result = log10_avg_safe(vec)
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
        
        if "sigmoid(randn[100000])" and False:
            print("sigmoid(randn[100000])")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            if TESTING:
                test_time_list = [1000]
                pass
            else:
                test_time_list = [10]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                scaling_factor_list = [0.0005,0.001,0.002,0.005,0.01, 0.02, 0.1,0.2, 0.5,   1.,  2,3,4,5,7,10,20,30,40,50,70,100]
                #                            0.1                           4
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
                    if dim == 100000:
                        #fp64
                        the_min_gt_this_list =[-0.311, -0.311, -0.311, -0.311, -0.311, -0.311, -0.314, -0.317, -0.331, -0.369, -0.478, -0.617, -0.769, -0.931, -1.262, -1.774, -3.531, -5.229, -6.968, -8.734, -12.230, -17.393]
                        the_max_lt_this_list =[-0.291, -0.291, -0.291, -0.291, -0.291, -0.291, -0.293, -0.296, -0.309, -0.344, -0.448, -0.580, -0.727, -0.876, -1.198, -1.684, -3.340, -5.016, -6.711, -8.388, -11.693, -16.797]
                        the_mean_eq_this_list=[-0.301, -0.301, -0.301, -0.301, -0.301, -0.301, -0.303, -0.307, -0.320, -0.356, -0.463, -0.598, -0.747, -0.905, -1.230, -1.732, -3.430, -5.135, -6.842, -8.553, -11.969, -17.100]
                        epsilon_list         =[ 0.020,  0.020,  0.020,  0.020,  0.020,  0.020,  0.020,  0.021,  0.022,  0.023,  0.025,  0.029,  0.032,  0.038,  0.042,  0.057,  0.111,  0.129,  0.141,  0.191,   0.285,   0.313]
                        #scaling_factor_list =[0.0005,  0.001,  0.002,  0.005,  0.010,  0.020,  0.100,  0.200,  0.500,  1.000,  2.000,  3.000,  4.000,  5.000,  7.000,  10.00,  20.00,  30.00,  40.00,  50.00,  70.000,  100.00]
                        # the_mean_eq_this_list / scaling_factor_list :
                        #                    [-602.00, -301.0, -150.5, -60.20, -30.10, -15.050, -3.030, -1.535, -0.640, -0.356, -0.231, -0.199, -0.187, -0.181, -0.176, -0.173, -0.171, -0.171, -0.171, -0.171, -0.171, -0.171]
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
                        loss_layer = torch.nn.Sigmoid()
                        vec = torch.randn(size=[dim],device=device, dtype=torch.float64)*scaling_factor
                        vec = loss_layer(vec)
                        _this_result = log10_avg_safe(vec)
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
        
        if "tanh(randn[100000])" and False:
            print("tanh(randn[100000])")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100000]
            if TESTING:
                test_time_list = [1000]
                pass
            else:
                test_time_list = [10]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                scaling_factor_list = [0.0005,0.001,0.002,0.005,0.01, 0.02, 0.1,0.2, 0.5,   1.,  2,3,4,5,7,10,20,30]
                #                          ??????????  0.1                           10
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
                    if dim == 100000:
                        #fp64
                        
                        
                        the_min_gt_this_list =[-3.592, -3.291, -2.990, -2.592, -2.290, -1.989, -1.291, -0.993, -0.616, -0.368, -0.180, -0.105, -0.067, -0.046, -0.026, -0.015, -0.010, -0.010]
                        the_max_lt_this_list =[-3.562, -3.260, -2.959, -2.562, -2.260, -1.960, -1.261, -0.964, -0.587, -0.339, -0.153, -0.080, -0.043, -0.023, -0.004,  0.005,  0.010,  0.010]
                        the_mean_eq_this_list=[-3.576, -3.275, -2.974, -2.576, -2.275, -1.974, -1.277, -0.979, -0.602, -0.353, -0.166, -0.092, -0.055, -0.035, -0.015, -0.005, -0.000, -0.000]
                        epsilon_list         =[ 0.025,  0.025,  0.026,  0.025,  0.025,  0.025,  0.025,  0.025,  0.024,  0.025,  0.024,  0.023,  0.022,  0.022,  0.021,  0.020,  0.020,  0.020]
                        #scaling_factor_list =[0.0005,  0.001,  0.002,  0.005,  0.010,  0.020,  0.100,  0.200,  0.500,  1.000,  2.000,  3.000,  4.000,  5.000,  7.000,  10.00,  20.00,  30.00]
                        
                        
                        
                        
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
                        loss_layer = torch.nn.Tanh()
                        vec = torch.randn(size=[dim],device=device, dtype=torch.float64)*scaling_factor
                        vec = loss_layer(vec)
                        _this_result = log10_avg_safe(vec)
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
        
        if "sin(randn[100000]*scaling_factor)" and False:
            print("sin(randn[100000]*scaling_factor)")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            scale_factor_list = [0.00001, 0.0001, 0.001, 0.01, 0.0316,0.1,0.316,1.,1.3,1.6,1.7,1.8,2.,5.,10.]
            
            dim_list = [100000]*scale_factor_list.__len__()
            if TESTING:
                test_time_list = [2000]*scale_factor_list.__len__()
                pass
            else:
                test_time_list = [10]*scale_factor_list.__len__()
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
                the_min_gt_this_list =[-5.292, -4.290, -3.291, -2.290, -1.791, -1.291, -0.797, -0.359, -0.309, -0.295, -0.293, -0.293, -0.292, -0.292, -0.291]
                the_max_lt_this_list =[-5.260, -4.260, -3.260, -2.260, -1.759, -1.259, -0.766, -0.328, -0.280, -0.266, -0.264, -0.264, -0.263, -0.263, -0.263]
                the_mean_eq_this_list=[-5.275, -4.275, -3.275, -2.275, -1.776, -1.276, -0.782, -0.343, -0.294, -0.280, -0.279, -0.278, -0.277, -0.277, -0.277]
                epsilon_list         =[ 0.026,  0.026,  0.026,  0.025,  0.027,  0.027,  0.025,  0.025,  0.025,  0.025,  0.025,  0.025,  0.025,  0.025,  0.025]
                #scale_factor_list   =[ 0.000,  0.000,  0.001,  0.010,  0.032,  0.100,  0.316,  1.000,  1.300,  1.600,  1.700,  1.800,  2.000,  5.000,  10.000]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                scale_factor = scale_factor_list[param_set_count]
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
                    print(f"dim{dim}  sf{scale_factor}  //  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
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
                print(f"#scale_factor_list   ={str_the_list(scale_factor_list,         3)}")    
                pass
            pass#/test
        
        if "cos(randn[100000]*scaling_factor)" and False:
            print("cos(randn[100000]*scaling_factor)")
            TESTING = True
            device = 'cuda'
            #--------------------#--------------------#--------------------
            scale_factor_list = [ 0.01,  0.047,  0.056,  0.068,    0.1,  0.316,     1.,    1.6,    1.8,   1.85,    1.9,   1.95,     2.]
            
            dim_list = [100000]*scale_factor_list.__len__()
            if TESTING:
                test_time_list = [2000]*scale_factor_list.__len__()
                pass
            else:
                test_time_list = [10]*scale_factor_list.__len__()
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
                the_min_gt_this_list =[-0.010, -0.010, -0.011, -0.011, -0.012, -0.030, -0.224, -0.289, -0.291, -0.291, -0.291, -0.291, -0.292]
                the_max_lt_this_list =[ 0.010,  0.010,  0.009,  0.009,  0.008, -0.009, -0.196, -0.260, -0.262, -0.262, -0.261, -0.263, -0.262]
                the_mean_eq_this_list=[-0.000, -0.000, -0.001, -0.001, -0.002, -0.020, -0.210, -0.274, -0.276, -0.277, -0.277, -0.277, -0.277]
                epsilon_list         =[ 0.020,  0.020,  0.020,  0.020,  0.020,  0.020,  0.024,  0.025,  0.024,  0.024,  0.026,  0.024,  0.025]
                #scale_factor_list   =[ 0.010,  0.047,  0.056,  0.068,  0.100,  0.316,  1.000,  1.600,  1.800,  1.850,  1.900,  1.950,  2.000]
                pass
            
            for param_set_count in range(dim_list.__len__()):
                scale_factor = scale_factor_list[param_set_count]
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
                    print(f"dim{dim}  sf{scale_factor}  //  {the_min-0.01:.3f}   {the_max+0.01:.3f}   {the_mean:.3f}   ")
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
                print(f"#scale_factor_list   ={str_the_list(scale_factor_list,         3)}")    
                pass
            pass#/test
        
        return 
    with_activation_functions()
    pass



# randn_vec @ relu   (randn) >>> -0.275 and -0.576*, into  0.5*log10(mid dim)-0.426

# randn_vec @ gelu(randn*sf) >>>       sf < 0.1 >>> -0.275 and log10(sf) -0.577 (< -1.58)    into  +0.5*log10(dim) + 1.*log10(sf) -0.572
# randn_vec @ gelu(randn*sf) >>> 0.1 < sf < 4   >>> -0.275 and between -1.58 and -0.82
# randn_vec @ gelu(randn*sf) >>> 4   < sf       >>> -0.275 and 1.9*log10(sf) -2.1 (> -0.82)  into  +0.5*log10(dim) + 1.*log10(sf) -0.427

# randn_vec @ sigmoid(randn) >>> sf < 0.1 >>> -0.275 and -0.301             , into  0.5*log10(dim) -0.577
# randn_vec @ sigmoid(randn) >>> 0.1 < sf >>> -0.275 and (-0.301 and -1.732), into  a lil larger. basically 0.5*log10(dim) -0.577 to 0.5*log10(dim) -0.428

# randn_vec @ tanh   (randn) >>> sf < 0.1 >>> -0.275 and log10(sf)-0.275 (<-1.277),   into    0.5*log10(dim) + 1.*log10(sf) -0.276
# randn_vec @ tanh   (randn) >>> 0.1 < sf >>> -0.275 and between -1.277 to 0      , into    0.5*log10(dim) + (-0.276 to +0.723)
# bc when x is very close to 0, tanh(x) is basically x. When sf < 0.1, it's basically a randn @ randn.

# randn_vec @ sin(randn*scale_factor) >>>       sf < 0.1 >>> -0.275 and 1.*log10(sf) -0.275    , into  0.5*log10(dim) + 1.*log10(sf) -0.275, the same as randn@randn
# randn_vec @ sin(randn*scale_factor) >>> 0.1 < sf < 1.6 >>> -0.275 and (from -1.276 to -0.277), into  0.5*log10(dim) - some_function(sf) but basically between log10(sf)-0.16 to -0.31
# randn_vec @ sin(randn*scale_factor) >>> 1.6 < sf       >>> -0.275 and -0.277                 , into  0.5*log10(dim) -0.426

# randn_vec @ cos(randn*scale_factor) >>>        sf < 0.05 >>> -0.275 and 0            , into  0.5 log10(dim) -0.276 (similar to a randn@constant[dim])
# randn_vec @ cos(randn*scale_factor) >>> 0.05 < sf < 1.8  >>> -0.275 and (0 to -0.277), into  in between ^^^ and vvv
# randn_vec @ cos(randn*scale_factor) >>> 1.8  < sf        >>> -0.275 and -0.277       , into  0.5 log10(dim) -0.426

# gauss is not tested. Maybe later?

if "matmul after activation functions" and False:
    def matmul_after_activation_functions():
        
        if "relu(randn[dim]) @ randn[dim,dim]" and False:
            TESTING = True
            if TESTING:
                print("relu(randn[dim]) @ randn[dim,dim]")
                pass
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            if TESTING:
                test_time_list = [20000,5000,500]
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
                the_min_gt_this_list =[ 0.227,  0.984,  1.543]
                the_max_lt_this_list =[ 0.823,  1.170,  1.605]
                the_mean_eq_this_list=[ 0.567,  1.073,  1.574]
                epsilon_list         =[ 0.350,  0.107,  0.041]
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
                    vec = torch.randn(size=[dim],device=device)
                    loss_layer = torch.nn.ReLU()
                    vec = loss_layer(vec)
                    mat = torch.randn(size=[dim, dim],device=device)
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
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass
            pass#/test
        
        if "gelu(randn[dim]*sf) @ randn[dim,dim]" and False:
            print("gelu(randn[dim]) @ randn[dim,dim]")
            TESTING = True
            device = 'cuda'
            
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #dim_list = [10000]
            if TESTING:
                test_time_list = [2000,2000,200]
                #test_time_list = [20]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                #scaling_factor_list = [0.0005,0.001,0.002,0.005,0.01, 0.02, 0.1,0.2, 0.5,   1.,  3.16,4,5,10,31.6,40,50, 100.]
                scaling_factor_list = [0.001,0.01, 0.02, 0.1, 0.5,   1.,  3.16,4,5,10,31.6,40,50, 100.]
                #                            0.1                           4
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
                        the_min_gt_this_list =[-2.776, -1.785, -1.497, -0.838, -0.119,  0.260,  0.804,  0.899,  0.990,  1.241,  1.826,  1.914,  1.990,  2.216]
                        the_max_lt_this_list =[-2.366, -1.360, -1.058, -0.318,  0.439,  0.773,  1.302,  1.388,  1.481,  1.847,  2.292,  2.384,  2.499,  2.802]
                        the_mean_eq_this_list=[-2.580, -1.581, -1.279, -0.579,  0.176,  0.529,  1.062,  1.170,  1.266,  1.568,  2.067,  2.168,  2.267,  2.567]
                        epsilon_list       =[ 0.224,  0.231,  0.231,  0.271,  0.305,  0.280,  0.268,  0.282,  0.285,  0.337,  0.251,  0.264,  0.287,  0.361]
                        #scaling_factor_list =[ 0.001,  0.010,  0.020,  0.100,  0.500,  1.000,  3.160,  4.000,  5.000,  10.000, 31.600, 40.000, 50.000, 100.000]
                        pass
                    if dim == 1000:
                        the_min_gt_this_list =[-2.166, -1.157, -0.857, -0.141,  0.587,  0.944,  1.486,  1.591,  1.683,  1.992,  2.489,  2.594,  2.680,  2.979]
                        the_max_lt_this_list =[-2.000, -1.004, -0.708,  0.004,  0.773,  1.126,  1.661,  1.752,  1.851,  2.166,  2.654,  2.754,  2.866,  3.152]
                        the_mean_eq_this_list=[-2.077, -1.077, -0.776, -0.073,  0.685,  1.037,  1.572,  1.674,  1.771,  2.073,  2.573,  2.676,  2.773,  3.073]
                        epsilon_list         =[ 0.098,  0.090,  0.092,  0.086,  0.107,  0.103,  0.099,  0.093,  0.098,  0.103,  0.094,  0.092,  0.103,  0.104]
                        #scaling_factor_list =[ 0.001,  0.010,  0.020,  0.100,  0.500,  1.000,  3.160,  4.000,  5.000,  10.000, 31.600, 40.000, 50.000, 100.000]
                        pass
                    if dim == 10000:
                        the_min_gt_this_list =[-1.605, -0.601, -0.298,  0.404,  1.155,  1.505,  2.047,  2.142,  2.243,  2.547,  3.049,  3.148,  3.238,  3.541]
                        the_max_lt_this_list =[-1.552, -0.550, -0.251,  0.452,  1.211,  1.574,  2.098,  2.201,  2.302,  2.607,  3.096,  3.203,  3.306,  3.604]
                        the_mean_eq_this_list=[-1.577, -0.576, -0.275,  0.428,  1.185,  1.538,  2.072,  2.175,  2.272,  2.573,  3.074,  3.176,  3.274,  3.573]
                        epsilon_list         =[ 0.038,  0.036,  0.034,  0.035,  0.039,  0.045,  0.036,  0.043,  0.040,  0.044,  0.034,  0.038,  0.046,  0.041]
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
                        vec = torch.randn(size=[dim],device=device)*scaling_factor
                        loss_layer = torch.nn.GELU()
                        vec = loss_layer(vec)
                        mat = torch.randn(size=[dim, dim],device=device)
                        prod = vec@mat
                        _this_result = log10_avg_safe(prod)
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
        
        if "sigmoid(randn[dim]*sf) @ randn[dim,dim]" and False:
            print("sigmoid(randn[dim]) @ randn[dim,dim]")
            TESTING = True
            device = 'cuda'
            
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #dim_list = [10000]
            if TESTING:
                test_time_list = [2000,2000,200]
                #test_time_list = [20]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                #scaling_factor_list = [0.001,0.01, 0.02, 0.1, 0.5,   1.,  3.16,4,5,10,31.6,40,50, 100.]
                scaling_factor_list = [0.0005,0.001,0.002,0.005,0.01, 0.02, 0.1,0.2, 0.5,   1.,  2,3,4,5,7,10,20,30,40,50,70,100]
                #                                                           0.1                      4
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
                        the_min_gt_this_list =[ 0.219,  0.222,  0.227,  0.233,  0.248,  0.230,  0.251,  0.237,  0.262,  0.216,  0.279,  0.323,  0.340,  0.327,  0.328,  0.346,  0.361,  0.344,  0.359,  0.300,  0.339,  0.359]
                        the_max_lt_this_list =[ 0.625,  0.579,  0.577,  0.571,  0.594,  0.574,  0.597,  0.586,  0.591,  0.639,  0.680,  0.722,  0.704,  0.717,  0.738,  0.733,  0.734,  0.754,  0.753,  0.758,  0.736,  0.776]
                        the_mean_eq_this_list=[ 0.421,  0.423,  0.422,  0.423,  0.421,  0.421,  0.420,  0.422,  0.435,  0.455,  0.493,  0.515,  0.527,  0.537,  0.544,  0.554,  0.562,  0.567,  0.567,  0.568,  0.570,  0.570]
                        epsilon_list       =[ 0.214,  0.210,  0.205,  0.199,  0.183,  0.200,  0.187,  0.195,  0.184,  0.249,  0.224,  0.217,  0.197,  0.220,  0.226,  0.218,  0.211,  0.232,  0.218,  0.278,  0.241,  0.221]
                        #scaling_factor_list =[0.0005,  0.001,  0.002,  0.005,  0.010,  0.020,  0.100,  0.200,  0.500,  1.000,  2.000,  3.000,  4.000,  5.000,  7.000,  10.000,  20.000,  30.000,  40.000,  50.000,  70.000,  100.000]
                        pass
                    if dim == 1000:
                        the_min_gt_this_list =[ 0.859,  0.859,  0.863,  0.858,  0.855,  0.864,  0.862,  0.853,  0.873,  0.884,  0.927,  0.944,  0.966,  0.970,  0.977,  0.995,  1.000,  0.994,  0.998,  1.000,  0.984,  1.004]
                        the_max_lt_this_list =[ 0.988,  0.977,  0.982,  0.985,  0.984,  0.979,  0.998,  0.985,  0.993,  1.021,  1.056,  1.081,  1.108,  1.110,  1.105,  1.120,  1.141,  1.136,  1.136,  1.138,  1.132,  1.138]
                        the_mean_eq_this_list=[ 0.923,  0.923,  0.923,  0.924,  0.924,  0.924,  0.923,  0.925,  0.935,  0.958,  0.995,  1.017,  1.031,  1.039,  1.048,  1.056,  1.064,  1.068,  1.069,  1.070,  1.072,  1.072]
                        epsilon_list       =[ 0.075,  0.074,  0.070,  0.076,  0.079,  0.070,  0.084,  0.082,  0.072,  0.084,  0.079,  0.082,  0.088,  0.081,  0.081,  0.074,  0.087,  0.084,  0.081,  0.080,  0.098,  0.077]
                        #scaling_factor_list =[0.0005,  0.001,  0.002,  0.005,  0.010,  0.020,  0.100,  0.200,  0.500,  1.000,  2.000,  3.000,  4.000,  5.000,  7.000,  10.000,  20.000,  30.000,  40.000,  50.000,  70.000,  100.000]
                        pass
                    if dim == 10000:
                        the_min_gt_this_list =[ 1.398,  1.401,  1.402,  1.399,  1.399,  1.400,  1.402,  1.402,  1.414,  1.438,  1.470,  1.492,  1.507,  1.512,  1.524,  1.527,  1.538,  1.545,  1.544,  1.548,  1.546,  1.544]
                        the_max_lt_this_list =[ 1.450,  1.451,  1.446,  1.446,  1.445,  1.447,  1.448,  1.452,  1.460,  1.482,  1.517,  1.541,  1.555,  1.564,  1.574,  1.580,  1.591,  1.596,  1.593,  1.593,  1.595,  1.599]
                        the_mean_eq_this_list=[ 1.423,  1.424,  1.423,  1.424,  1.423,  1.424,  1.423,  1.426,  1.436,  1.458,  1.496,  1.517,  1.530,  1.538,  1.549,  1.556,  1.565,  1.568,  1.570,  1.571,  1.572,  1.572]
                        epsilon_list       =[ 0.036,  0.037,  0.032,  0.035,  0.034,  0.034,  0.035,  0.036,  0.034,  0.034,  0.036,  0.036,  0.035,  0.037,  0.036,  0.039,  0.037,  0.038,  0.035,  0.033,  0.036,  0.038]
                        #scaling_factor_list =[0.0005,  0.001,  0.002,  0.005,  0.010,  0.020,  0.100,  0.200,  0.500,  1.000,  2.000,  3.000,  4.000,  5.000,  7.000,  10.000,  20.000,  30.000,  40.000,  50.000,  70.000,  100.000]
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
                        vec = torch.randn(size=[dim],device=device)*scaling_factor
                        loss_layer = torch.nn.Sigmoid()
                        vec = loss_layer(vec)
                        mat = torch.randn(size=[dim, dim],device=device)
                        prod = vec@mat
                        _this_result = log10_avg_safe(prod)
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
        
        if "tanh(randn[dim]*sf) @ randn[dim,dim]" and False:
            print("tanh(randn[dim]) @ randn[dim,dim]")
            TESTING = True
            device = 'cuda'
            
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #dim_list = [10000]
            if TESTING:
                test_time_list = [2000,2000,200]
                #test_time_list = [20]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                #scaling_factor_list = [0.001,0.01, 0.02, 0.1, 0.5,   1.,  3.16,4,5,10,31.6,40,50, 100.]
                scaling_factor_list = [0.0005,0.001,0.002,0.005,0.01, 0.02, 0.1,0.2, 0.5,   1.,  2,3,4,5,7,10,20,30,40,50,70,100]
                #                                                           0.1                      4
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
                        the_min_gt_this_list =[-2.852, -2.503, -2.219, -1.795, -1.540, -1.197, -0.492, -0.215,  0.141,  0.293,  0.409,  0.461,  0.511,  0.497,  0.477,  0.509,  0.515,  0.506,  0.510,  0.531,  0.505,  0.534]
                        the_max_lt_this_list =[-2.372, -2.034, -1.801, -1.363, -1.078, -0.798, -0.090,  0.220,  0.537,  0.702,  0.801,  0.828,  0.839,  0.840,  0.856,  0.861,  0.866,  0.869,  0.874,  0.889,  0.907,  0.869]
                        the_mean_eq_this_list=[-2.581, -2.281, -1.979, -1.583, -1.283, -0.977, -0.283,  0.004,  0.342,  0.519,  0.623,  0.660,  0.675,  0.683,  0.697,  0.704,  0.714,  0.717,  0.718,  0.719,  0.721,  0.721]
                        epsilon_list         =[ 0.281,  0.257,  0.250,  0.230,  0.267,  0.229,  0.219,  0.229,  0.211,  0.236,  0.224,  0.209,  0.174,  0.197,  0.229,  0.205,  0.209,  0.221,  0.218,  0.198,  0.226,  0.197]
                        #scaling_factor_list = 0.0005,  0.001,  0.002,  0.005,  0.010,  0.020,  0.100,  0.200,  0.500,  1.000,  2.000,  3.000,  4.000,  5.000,  7.000,  10.00,  20.00,  30.00,  40.00,  50.00,  70.00,  100.0]
                        pass
                    if dim == 1000:
                        the_min_gt_this_list =[-2.165, -1.853, -1.546, -1.151, -0.848, -0.552,  0.145,  0.434,  0.776,  0.943,  1.057,  1.092,  1.102,  1.125,  1.137,  1.140,  1.133,  1.146,  1.154,  1.153,  1.161,  1.162]
                        the_max_lt_this_list =[-2.002, -1.697, -1.402, -1.009, -0.711, -0.406,  0.295,  0.575,  0.908,  1.084,  1.196,  1.226,  1.230,  1.244,  1.257,  1.272,  1.274,  1.280,  1.295,  1.287,  1.286,  1.284]
                        the_mean_eq_this_list=[-2.077, -1.776, -1.475, -1.076, -0.777, -0.475,  0.220,  0.509,  0.845,  1.022,  1.126,  1.161,  1.177,  1.187,  1.199,  1.207,  1.215,  1.218,  1.220,  1.221,  1.222,  1.222]
                        epsilon_list         =[ 0.098,  0.090,  0.083,  0.084,  0.082,  0.087,  0.085,  0.085,  0.079,  0.089,  0.080,  0.079,  0.086,  0.072,  0.072,  0.077,  0.092,  0.082,  0.085,  0.078,  0.074,  0.072]
                        #scaling_factor_list = 0.0005,  0.001,  0.002,  0.005,  0.010,  0.020,  0.100,  0.200,  0.500,  1.000,  2.000,  3.000,  4.000,  5.000,  7.000,  10.00,  20.00,  30.00,  40.00,  50.00,  70.00,  100.0]
                        
                        pass
                    if dim == 10000:
                        
                        the_min_gt_this_list =[-1.603, -1.301, -1.001, -0.605, -0.306, -0.001,  0.696,  0.983,  1.320,  1.498,  1.599,  1.635,  1.652,  1.663,  1.677,  1.685,  1.689,  1.693,  1.697,  1.699,  1.694,  1.700]
                        the_max_lt_this_list =[-1.549, -1.251, -0.944, -0.552, -0.246,  0.049,  0.744,  1.042,  1.371,  1.544,  1.649,  1.684,  1.700,  1.712,  1.724,  1.729,  1.739,  1.740,  1.744,  1.750,  1.747,  1.747]
                        the_mean_eq_this_list=[-1.577, -1.275, -0.974, -0.577, -0.276,  0.025,  0.720,  1.009,  1.345,  1.523,  1.626,  1.661,  1.678,  1.687,  1.698,  1.706,  1.715,  1.719,  1.720,  1.721,  1.722,  1.723]
                        epsilon_list         =[ 0.037,  0.036,  0.040,  0.038,  0.040,  0.036,  0.034,  0.043,  0.036,  0.035,  0.037,  0.036,  0.035,  0.035,  0.035,  0.032,  0.036,  0.035,  0.034,  0.039,  0.038,  0.034]
                        #scaling_factor_list = 0.0005,  0.001,  0.002,  0.005,  0.010,  0.020,  0.100,  0.200,  0.500,  1.000,  2.000,  3.000,  4.000,  5.000,  7.000,  10.00,  20.00,  30.00,  40.00,  50.00,  70.00,  100.0]
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
                        vec = torch.randn(size=[dim],device=device)*scaling_factor
                        loss_layer = torch.nn.Tanh()
                        vec = loss_layer(vec)
                        mat = torch.randn(size=[dim, dim],device=device)
                        prod = vec@mat
                        _this_result = log10_avg_safe(prod)
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
        
        if "sin(randn[dim]*sf) @ randn[dim,dim]" and False:
            print("sin(randn[dim]) @ randn[dim,dim]")
            TESTING = True
            device = 'cuda'
            
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #dim_list = [10000]
            if TESTING:
                test_time_list = [2000,2000,200]
                #test_time_list = [20]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                scaling_factor_list =[ 0.0001, 0.001, 0.01,  0.02,    0.1,    0.2,    0.6,   1.,  1.5,  1.6,  1.7, 1.8,1.9,  2.,   3.]
                #the 0.1  and 1.6
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
                        the_min_gt_this_list =[-3.550, -2.511, -1.509, -1.189, -0.473, -0.193,  0.194,  0.338,  0.365,  0.383,  0.338,  0.384,  0.360,  0.374,  0.392]
                        the_max_lt_this_list =[-3.090, -2.077, -1.053, -0.801, -0.095,  0.226,  0.631,  0.711,  0.738,  0.743,  0.740,  0.739,  0.770,  0.744,  0.739]
                        the_mean_eq_this_list=[-3.280, -2.279, -1.281, -0.977, -0.280,  0.014,  0.426,  0.539,  0.569,  0.570,  0.570,  0.572,  0.571,  0.571,  0.573]
                        epsilon_list         =[ 0.280,  0.243,  0.239,  0.221,  0.203,  0.222,  0.242,  0.212,  0.214,  0.197,  0.242,  0.198,  0.221,  0.206,  0.191]
                        #scaling_factor_list =[0.0001,  0.001,  0.010,  0.020,  0.100,  0.200,  0.600,  1.000,  1.500,  1.600,  1.700,  1.800,  1.900,  2.000,  3.000]
                        pass
                    if dim == 1000:
                        the_min_gt_this_list =[-2.858, -1.843, -0.843, -0.557,  0.141,  0.439,  0.857,  0.977,  1.009,  0.992,  0.993,  1.007,  1.002,  1.007,  1.000]
                        the_max_lt_this_list =[-2.702, -1.703, -0.702, -0.409,  0.290,  0.591,  0.992,  1.105,  1.130,  1.147,  1.133,  1.132,  1.144,  1.135,  1.134]
                        the_mean_eq_this_list=[-2.777, -1.775, -0.775, -0.475,  0.222,  0.517,  0.930,  1.042,  1.071,  1.073,  1.074,  1.073,  1.073,  1.074,  1.074]
                        epsilon_list         =[ 0.091,  0.082,  0.083,  0.092,  0.091,  0.088,  0.083,  0.075,  0.071,  0.091,  0.091,  0.076,  0.082,  0.077,  0.083]
                        #scaling_factor_list =[0.0001,  0.001,  0.010,  0.020,  0.100,  0.200,  0.600,  1.000,  1.500,  1.600,  1.700,  1.800,  1.900,  2.000,  3.000]
                        pass
                    if dim == 10000:
                        the_min_gt_this_list =[-2.300, -1.302, -0.307, -0.004,  0.697,  0.990,  1.404,  1.519,  1.548,  1.550,  1.545,  1.548,  1.548,  1.549,  1.545]
                        the_max_lt_this_list =[-2.247, -1.250, -0.251,  0.053,  0.746,  1.044,  1.452,  1.568,  1.596,  1.599,  1.597,  1.597,  1.599,  1.597,  1.596]
                        the_mean_eq_this_list=[-2.276, -1.277, -0.275,  0.025,  0.723,  1.017,  1.429,  1.543,  1.571,  1.572,  1.573,  1.573,  1.574,  1.573,  1.574]
                        epsilon_list         =[ 0.038,  0.037,  0.042,  0.039,  0.036,  0.038,  0.035,  0.035,  0.034,  0.037,  0.038,  0.035,  0.036,  0.035,  0.039]
                        #scaling_factor_list =[0.0001,  0.001,  0.010,  0.020,  0.100,  0.200,  0.600,  1.000,  1.500,  1.600,  1.700,  1.800,  1.900,  2.000,  3.000]
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
                        vec = torch.randn(size=[dim],device=device)*scaling_factor
                        vec = vec.sin()
                        mat = torch.randn(size=[dim, dim],device=device)
                        prod = vec@mat
                        _this_result = log10_avg_safe(prod)
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
        
        if "cos(randn[dim]*sf) @ randn[dim,dim]" and False:
            print("cos(randn[dim]) @ randn[dim,dim]")
            TESTING = True
            device = 'cuda'
            
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #dim_list = [10000]
            if TESTING:
                test_time_list = [2000,2000,200]
                #test_time_list = [20]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                scaling_factor_list = [ 0.01,  0.047,  0.056,  0.068,    0.1,  0.316,     1.,    1.6,    1.8,   1.85,    1.9,   1.95,     2.]
                #the 0.05 < sf < 1.8  ???
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
                        the_min_gt_this_list =[ 0.546,  0.544,  0.508,  0.522,  0.528,  0.521,  0.405,  0.374,  0.396,  0.348,  0.386,  0.390,  0.353]
                        the_max_lt_this_list =[ 0.889,  0.907,  0.878,  0.880,  0.910,  0.869,  0.760,  0.738,  0.733,  0.749,  0.726,  0.764,  0.761]
                        the_mean_eq_this_list=[ 0.722,  0.724,  0.722,  0.722,  0.720,  0.702,  0.601,  0.572,  0.573,  0.572,  0.570,  0.573,  0.571]
                        epsilon_list         =[ 0.186,  0.193,  0.224,  0.210,  0.202,  0.192,  0.206,  0.209,  0.187,  0.233,  0.194,  0.201,  0.228]
                        #scaling_factor_list =[ 0.010,  0.047,  0.056,  0.068,  0.100,  0.316,  1.000,  1.600,  1.800,  1.850,  1.900,  1.950,  2.000]
                        pass
                    if dim == 1000:
                        the_min_gt_this_list =[ 1.163,  1.153,  1.156,  1.166,  1.162,  1.144,  1.038,  1.007,  1.015,  1.009,  1.001,  1.010,  1.007]
                        the_max_lt_this_list =[ 1.281,  1.289,  1.290,  1.284,  1.277,  1.262,  1.164,  1.146,  1.145,  1.130,  1.135,  1.139,  1.136]
                        the_mean_eq_this_list=[ 1.225,  1.224,  1.224,  1.224,  1.222,  1.204,  1.102,  1.075,  1.074,  1.074,  1.074,  1.074,  1.074]
                        epsilon_list         =[ 0.072,  0.081,  0.078,  0.070,  0.070,  0.071,  0.074,  0.081,  0.081,  0.076,  0.083,  0.075,  0.077]
                        #scaling_factor_list =[ 0.010,  0.047,  0.056,  0.068,  0.100,  0.316,  1.000,  1.600,  1.800,  1.850,  1.900,  1.950,  2.000]
                        pass
                    if dim == 10000:
                        the_min_gt_this_list =[ 1.701,  1.694,  1.703,  1.701,  1.699,  1.677,  1.578,  1.549,  1.548,  1.551,  1.548,  1.551,  1.546]
                        the_max_lt_this_list =[ 1.750,  1.746,  1.747,  1.746,  1.751,  1.726,  1.625,  1.602,  1.598,  1.598,  1.598,  1.603,  1.597]
                        the_mean_eq_this_list=[ 1.724,  1.724,  1.724,  1.723,  1.722,  1.704,  1.601,  1.576,  1.575,  1.574,  1.574,  1.574,  1.574]
                        epsilon_list         =[ 0.036,  0.039,  0.033,  0.033,  0.039,  0.037,  0.034,  0.036,  0.037,  0.034,  0.036,  0.038,  0.038]
                        #scaling_factor_list =[ 0.010,  0.047,  0.056,  0.068,  0.100,  0.316,  1.000,  1.600,  1.800,  1.850,  1.900,  1.950,  2.000]
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
                        vec = torch.randn(size=[dim],device=device)*scaling_factor
                        vec = vec.cos()
                        mat = torch.randn(size=[dim, dim],device=device)
                        prod = vec@mat
                        _this_result = log10_avg_safe(prod)
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
    matmul_after_activation_functions()
    pass



# softmax(randn[mid dim]*scaling_factor) @ randn[mid dim,??]

# sf < 0.4      >>> -0.275 and -1.*log10(dim)            , into 0.5*log10(dim) -0.275 (similar to randn @ constant[dim])
# 0.4 < sf < 30 >>> -0.275 and (in between)              , into between ^^^ and vvv
# 30 < sf       >>> -0.275 and -sf*(0.72+0.25*log10(dim)), into -0.296

# old note. probably wrong.
# if scaling_factor < 0.1 >>> -0.5 *log10(dim) - 0.16 (it feels like the randn@randn but the dim makes it smaller.)
# if scaling_factor == 1. >>> -0.5 *log10(dim) + 0.04 
# if scaling_factor == 2. >>> -0.36*log10(dim) + 0.07
# if scaling_factor > 5   >>> -0.02*log10(dim) - 0.20  (basically -0.24 to -0.28)

if "softmax @ randn" and True:
    def softmax_matmul_randn():
        if "softmax(randn[dim]*sf) @ randn[dim,dim]" and True:
            print("softmax(randn[dim]*sf) @ randn[dim,dim]")
            TESTING = True
            device = 'cuda'
            
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            #dim_list = [10000]
            if TESTING:
                test_time_list = [2000,2000,200]
                test_time_list = [200,200,5]
                pass
            else:
                test_time_list = [100,30,5]
                pass
            for macro_iter_count in range(dim_list.__len__()):
                dim = dim_list[macro_iter_count]
                test_time = test_time_list[macro_iter_count]
                
                scaling_factor_list = [0.001, 0.01,  0.1, 0.3, 0.4, 0.5, 1.,  1.5,  2.,  2.5,  3., 5., 10., 20., 30., 50.]
                #the   0.4 < scaling_factor(sf) < 5???
            #--------------------#--------------------#--------------------
                if TESTING:
                    print(test_time)
                    the_min_gt_this_list =  []#don't modify here.
                    the_max_lt_this_list =  []
                    the_mean_eq_this_list = []
                    epsilon_list =          []
                    
                    _softmax_ref_list =          []##################################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    pass
                else:
                    ###########################################  result paste to here.
                    if dim == 100:
                        the_min_gt_this_list =[-1.507, -1.496, -1.440, -1.449, -1.439, -1.425, -1.314, -1.221, -1.139, -1.051, -1.050, -0.875, -0.810, -0.657, -0.628, -0.644]
                        the_max_lt_this_list =[-1.115, -1.116, -1.105, -1.099, -1.059, -1.059, -0.535, -0.408, -0.260, -0.186, -0.152, -0.153, -0.127, -0.122, -0.132, -0.113]
                        the_mean_eq_this_list=[-1.279, -1.278, -1.272, -1.257, -1.242, -1.223, -1.083, -0.905, -0.771, -0.668, -0.603, -0.449, -0.352, -0.316, -0.304, -0.294]
                        #softmax_ref         =[-3.000, -3.000, -3.006, -3.032, -3.051, -3.074, -3.257, -3.547, -3.940, -4.423, -4.982, -7.553,-14.595,-29.041,-43.542, -72.430]
                        epsilon_list         =[ 0.237,  0.229,  0.178,  0.202,  0.207,  0.212,  0.557,  0.507,  0.520,  0.492,  0.460,  0.436,  0.468,  0.351,  0.334,  0.360]
                        #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.300,  0.400,  0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  5.000,  10.00,  20.00,  30.00,  50.00]
                        pass
                    if dim == 1000:
                        the_min_gt_this_list =[-1.839, -1.831, -1.834, -1.822, -1.802, -1.791, -1.654, -1.522, -1.373, -1.265, -1.182, -0.997, -0.727, -0.717, -0.619, -0.584]
                        the_max_lt_this_list =[-1.715, -1.720, -1.707, -1.694, -1.673, -1.663, -1.355, -0.713, -0.344, -0.304, -0.283, -0.228, -0.222, -0.216, -0.216, -0.223]
                        the_mean_eq_this_list=[-1.776, -1.776, -1.773, -1.756, -1.740, -1.722, -1.563, -1.333, -1.106, -0.923, -0.776, -0.534, -0.378, -0.325, -0.305, -0.293]
                        #softmax_ref         =[-4.000, -4.000, -4.006, -4.032, -4.051, -4.074, -4.257, -4.548, -4.947, -5.452, -6.050, -8.993,-17.358,-34.382,-51.485, -85.462]
                        epsilon_list         =[ 0.073,  0.066,  0.077,  0.075,  0.078,  0.079,  0.218,  0.630,  0.772,  0.629,  0.502,  0.473,  0.359,  0.402,  0.324,  0.302]
                        #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.300,  0.400,  0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  5.000,  10.00,  20.00,  30.00,  50.00]
                        pass
                    if dim == 10000:
                        the_min_gt_this_list =[-2.299, -2.299, -2.297, -2.280, -2.263, -2.244, -2.100, -1.891, -1.685, -1.512, -1.349, -0.999, -0.747, -0.646, -0.510, -0.513]
                        the_max_lt_this_list =[-2.251, -2.252, -2.251, -2.234, -2.220, -2.196, -1.930, -1.201, -0.686, -0.696, -0.365, -0.269, -0.256, -0.256, -0.253, -0.254]
                        the_mean_eq_this_list=[-2.275, -2.276, -2.273, -2.256, -2.241, -2.221, -2.058, -1.800, -1.510, -1.244, -1.001, -0.604, -0.406, -0.338, -0.307, -0.296]
                        #softmax_ref         =[-5.000, -5.000, -5.006, -5.032, -5.051, -5.074, -5.257, -5.548, -5.948, -6.458, -7.066,-10.290,-19.629,-39.034,-58.247, -97.628]
                        epsilon_list         =[ 0.034,  0.034,  0.034,  0.034,  0.032,  0.035,  0.138,  0.609,  0.835,  0.558,  0.646,  0.404,  0.350,  0.318,  0.213,  0.228]
                        #scaling_factor_list =[ 0.001,  0.010,  0.100,  0.300,  0.400,  0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  5.000,  10.00,  20.00,  30.00,  50.00]
                        pass
                    
                    pass
                
                # sf<0.4    >>> -0.275 and ... , into 0.5*log10(dim) -0.275 (similar to randn @ constant[dim])
                # 0.4<sf<30 >>> -0.275 and ... , into idk.
                # 30<sf     >>> -0.275 and ... , into -0.296
                
                1w 继续找规律。
                
                for param_set_count in range(scaling_factor_list.__len__()):
                    scaling_factor = scaling_factor_list[param_set_count]
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
                    print(f"#_softmax_ref_list   ={str_the_list(_softmax_ref_list,         3)}")  #####################################
                    print(f"#scaling_factor_list ={str_the_list(scaling_factor_list,         3)}")    
                    pass
                pass# for dim
            pass#/test
        
        return   
    
    softmax_matmul_randn()
    pass



# K He init >>> -0.5*log10(dim) -0.32


if "K He init" and True:
    def K_He_init():
        
        if "K He init [dim]" and False:
            print("K He init [dim] >>> ???")
            TESTING = False
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
    pass













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


