assert False, '''this project was stopped. I didn't expect it to work, and I didn't plan to use it.
I don't use xgb at all.

The basic idea is, xgb is a bit tend to seperate the extreme case, which can lead to overfitting on 
some of the special cases. And the formula for "final_score" doesn't look correct.

My version is:
branch_score = (sum/2), sqr
leaf_score = sum, sqr   for both.
final_score = branch*some_factor - avg(leaf_score)
The some_factor is at least 1., and designed to stop building too small sub branches, 
and also to help preventing the overfitting. 

According to the measurement. My verion is slightly worse than the original version with the best lambda possible.
(search: measure with partly sorted data)
So, I guess the 2 versions are similar. 
Then, no plan to continue this project.
'''


import random
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal

import time
def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######
from pytorch_yagaodirac_v2.Util import str_the_list





def calc_ori_similarity_score(input:torch.Tensor, the_lambda = 1.)->torch.Tensor:
    _temp1 = input.sum()
    _temp2 = _temp1*_temp1
    ori_similarity_score = _temp2/(input.shape[-1] + the_lambda)
    return ori_similarity_score
if "basic behavior" and False:
    def ____calc_ori_similarity_score____basic():
        assert _tensor_equal(calc_ori_similarity_score(torch.tensor([0.])), [0.])
        assert _tensor_equal(calc_ori_similarity_score(torch.tensor([1.])), [0.5])
        assert _tensor_equal(calc_ori_similarity_score(torch.tensor([2.])), [2.])
        assert _tensor_equal(calc_ori_similarity_score(torch.tensor([1., 0])), [0.333333])
        assert _tensor_equal(calc_ori_similarity_score(torch.tensor([1.]), the_lambda=2.), [0.333333])
        return 
    ____calc_ori_similarity_score____basic()
    pass

def calc_ori_final_score(input:torch.Tensor, leaf_1:torch.Tensor, leaf_2:torch.Tensor, 
                        the_lambda = 1.)->torch.Tensor:
    ori_score = calc_ori_similarity_score(input, the_lambda)
    ori_score_leaf_1 = calc_ori_similarity_score(leaf_1, the_lambda)
    ori_score_leaf_2 = calc_ori_similarity_score(leaf_2, the_lambda)
    ori_final_score = (ori_score_leaf_1 + ori_score_leaf_2) - ori_score
    return ori_final_score
if "basic behavior" and False:
    def ____calc_ori_final_score____basic():
        branch = torch.tensor([0., 0])
        leaf_1 = branch[0:1]
        leaf_2 = branch[1:2]
        final_score = calc_ori_final_score(branch, leaf_1, leaf_2)
        assert _tensor_equal(final_score, [0.])
        
        branch = torch.tensor([1., -1])
        leaf_1 = branch[0:1]
        leaf_2 = branch[1:2]
        final_score = calc_ori_final_score(branch, leaf_1, leaf_2)
        assert _tensor_equal(final_score, [1.])
        
        branch = torch.tensor([2., -1])
        leaf_1 = branch[0:1]
        leaf_2 = branch[1:2]
        final_score = calc_ori_final_score(branch, leaf_1, leaf_2)
        assert _tensor_equal(final_score, [2. + 0.5 - 0.3333333])
        
        return
    ____calc_ori_final_score____basic()
    pass    

if "scan xgb in random number" and True:
    def ____ori_in_random_number____():
        
        while "sorted data" and False:
            size = 1000
            branch = torch.randn(size=[size]).sort().values
            all_final_scores = torch.empty(size=[size-1])
            all_final_scores.fill_(torch.nan)
            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
            for cut_pos in all_cut_pos:
                leaf_1 = branch[:cut_pos]
                leaf_2 = branch[cut_pos:]
                this_final_score = calc_ori_final_score(branch, leaf_1, leaf_2)
                all_final_scores[cut_pos-1] = this_final_score
                pass
            
            import matplotlib.pyplot as plt
            the_max_of_score = all_final_scores.max()
            plt.plot(all_cut_pos, all_final_scores)
            plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1))
            plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.title("ori xgb")
            plt.show()
            pass# /test
        
        # they are indeed much smaller than the previous test.
        while "unsorted data" and False:
            size = 1000
            branch = torch.randn(size=[size])
            all_final_scores = torch.empty(size=[size-1])
            all_final_scores.fill_(torch.nan)
            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
            for cut_pos in all_cut_pos:
                leaf_1 = branch[:cut_pos]
                leaf_2 = branch[cut_pos:]
                this_final_score = calc_ori_final_score(branch, leaf_1, leaf_2)
                all_final_scores[cut_pos-1] = this_final_score
                pass
            
            import matplotlib.pyplot as plt
            the_max_of_score = all_final_scores.max()
            plt.plot(all_cut_pos, all_final_scores)
            plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1), alpha = 0.5)
            plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.title("ori xgb")
            plt.show()
            pass# /test
        
        
        while "partly sorted data" and False:
            size = 1000
            branch = torch.randn(size=[size])
            
            sorted_section_len = int(size*0.1)################
            #sorted_section_len = 3
            ii_sorted_section__start_pos = random.randint(0,size-sorted_section_len)
            ii_sorted_section__mid_pos = ii_sorted_section__start_pos+sorted_section_len//2
            ii_sorted_section__end_pos = ii_sorted_section__start_pos+sorted_section_len
            assert ii_sorted_section__end_pos <= size
            branch[ii_sorted_section__start_pos:ii_sorted_section__mid_pos] = -1.
            branch[ii_sorted_section__mid_pos:ii_sorted_section__end_pos] = 1.
            
            all_final_scores = torch.empty(size=[size-1])
            all_final_scores.fill_(torch.nan)
            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
            for cut_pos in all_cut_pos:
                leaf_1 = branch[:cut_pos]
                leaf_2 = branch[cut_pos:]
                this_final_score = calc_ori_final_score(branch, leaf_1, leaf_2)
                all_final_scores[cut_pos-1] = this_final_score
                pass
            
            #so the best score is expected to be in [start_pos: end_pos-1]
            
            import matplotlib.pyplot as plt
            the_max_of_score = all_final_scores.max()
            plt.plot(all_cut_pos-0.75, all_final_scores)
            plt.plot(torch.linspace(ii_sorted_section__start_pos,ii_sorted_section__end_pos-1,sorted_section_len-1)+0.25, 
                    [the_max_of_score]*(sorted_section_len-1), alpha = 0.6)
            plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1), alpha = 0.5)
            plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.title("ori xgb")
            plt.show()
            pass# /test
        
        if "measure with partly sorted data. 10% size as the accepted zone." and True:
            
            
            # size 100    lambda 1.0
            # the_max__min = [ 0.720,  0.936,  0.832,  0.948,  1.187,  2.937,  6.994,  13.842]
            # the_max__max = [ 13.248,  13.139,  12.705,  18.254,  18.359,  25.713,  38.004,  46.143]
            # the_max__avg = [ 4.058,  4.135,  3.987,  4.403,  6.431,  11.762,  18.533,  27.698]
            # expected_ratio=[ 0.08,   0.06,   0.14,   0.26,   0.68,   0.88,   0.99,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]

            # size 100    lambda 3.0
            # the_max__min = [ 0.669,  0.625,  0.790,  0.824,  1.496,  3.430,  5.191,  7.619]
            # the_max__max = [ 14.982,  11.815,  9.416,  9.291,  23.872,  24.859,  32.346,  41.756]
            # the_max__avg = [ 3.600,  3.321,  3.513,  3.626,  6.371,  11.064,  17.610,  26.318]
            # expected_ratio=[ 0.08,   0.16,   0.16,   0.32,   0.73,   0.92,   0.99,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]

            # size 100    lambda 10.0
            # the_max__min = [ 0.388,  0.494,  0.553,  0.468,  1.034,  2.437,  4.082,  10.804]
            # the_max__max = [ 10.345,  8.642,  12.787,  10.734,  13.699,  22.981,  32.468,  35.390]
            # the_max__avg = [ 2.895,  2.770,  2.750,  3.093,  4.982,  8.524,  15.594,  22.715]
            # expected_ratio=[ 0.06,   0.09,   0.17,   0.32,   0.79,   0.94,   0.99,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]

            # size 100    lambda 33.0
            # the_max__min = [ 0.328,  0.135,  0.343,  0.319,  0.266,  1.496,  3.165,  5.271]
            # the_max__max = [ 8.506,  6.837,  5.917,  7.908,  11.007,  14.178,  20.198,  25.199]
            # the_max__avg = [ 1.796,  1.845,  1.756,  1.962,  3.330,  5.698,  10.299,  15.172]
            # expected_ratio=[ 0.09,   0.09,   0.15,   0.34,   0.77,   0.93,   0.98,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]

            # size 100    lambda 100.0
            # the_max__min = [-0.225, -0.027,  0.019,  0.096,  0.238,  0.538,  1.440,  3.814]
            # the_max__max = [ 4.072,  3.895,  4.371,  3.536,  5.370,  7.178,  11.081,  13.983]
            # the_max__avg = [ 0.881,  0.937,  0.974,  0.994,  1.693,  3.059,  5.556,  8.310]
            # expected_ratio=[ 0.09,   0.10,   0.17,   0.31,   0.75,   0.93,   1.00,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]


            # size 100    lambda 1.0
            # expected_ratio=[ 0.08,   0.06,   0.14,   0.26,   0.68,   0.88,   0.99,   1.00]
            # size 100    lambda 3.0
            # expected_ratio=[ 0.08,   0.16,   0.16,   0.32,   0.73,   0.92,   0.99,   1.00]
            # size 100    lambda 10.0
            # expected_ratio=[ 0.06,   0.09,   0.17,   0.32,   0.79,   0.94,   0.99,   1.00]
            # size 100    lambda 33.0
            # expected_ratio=[ 0.09,   0.09,   0.15,   0.34,   0.77,   0.93,   0.98,   1.00]
            # size 100    lambda 100.0
            # expected_ratio=[ 0.09,   0.10,   0.17,   0.31,   0.75,   0.93,   1.00,   1.00]

            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]

                        
                        
                        
                        
                        
                        
            # size 1000    lambda 1.0
            # the_max__min = [ 1.602,  2.293,  2.054,  5.369,  21.436,  71.407,  132.413,  211.601]
            # the_max__max = [ 12.000,  11.301,  19.190,  56.273,  107.725,  176.831,  251.223,  348.665]
            # the_max__avg = [ 5.396,  6.060,  7.378,  17.610,  56.082,  108.343,  187.146,  282.224]
            # expected_ratio=[ 0.08,   0.34,   0.70,   0.88,   1.00,   1.00,   1.00,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]
                        
            # size 1000    lambda 3.0
            # the_max__min = [ 1.835,  1.112,  2.585,  5.026,  19.379,  61.402,  131.168,  215.481]
            # the_max__max = [ 12.758,  15.401,  22.670,  50.042,  100.568,  175.050,  250.419,  352.533]
            # the_max__avg = [ 5.212,  5.768,  7.333,  16.191,  56.526,  114.831,  188.275,  276.128]
            # expected_ratio=[ 0.08,   0.40,   0.60,   0.80,   1.00,   1.00,   1.00,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]
                        
            # size 1000    lambda 10.0
            # the_max__min = [ 1.383,  1.332,  3.200,  5.021,  25.681,  73.374,  131.701,  209.059]
            # the_max__max = [ 13.597,  21.054,  34.229,  50.455,  102.028,  166.372,  244.727,  323.258]
            # the_max__avg = [ 4.752,  6.259,  8.233,  17.695,  51.693,  114.796,  178.660,  257.981]
            # expected_ratio=[ 0.18,   0.40,   0.60,   0.96,   1.00,   1.00,   1.00,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]
                        
            # size 1000    lambda 33.0
            # the_max__min = [ 1.177,  1.691,  2.621,  4.033,  22.915,  66.442,  113.765,  191.123]
            # the_max__max = [ 8.313,  13.531,  15.930,  34.709,  95.018,  144.538,  223.130,  300.830]
            # the_max__avg = [ 4.207,  5.004,  6.614,  16.634,  48.676,  100.938,  165.981,  249.588]
            # expected_ratio=[ 0.18,   0.48,   0.62,   1.00,   1.00,   1.00,   1.00,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]
                        
            # size 1000    lambda 100.0
            # the_max__min = [ 0.605,  1.453,  1.336,  3.157,  19.112,  49.553,  100.092,  178.654]
            # the_max__max = [ 9.537,  7.756,  17.807,  26.243,  64.015,  128.813,  200.527,  272.308]
            # the_max__avg = [ 2.880,  3.746,  5.918,  11.409,  39.523,  86.621,  149.419,  227.926]
            # expected_ratio=[ 0.18,   0.30,   0.66,   0.90,   1.00,   1.00,   1.00,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]
                        

            # size 1000    lambda 1.0
            # expected_ratio=[ 0.08,   0.34,   0.70,   0.88,   1.00,   1.00,   1.00,   1.00]
            # size 1000    lambda 3.0
            # expected_ratio=[ 0.08,   0.40,   0.60,   0.80,   1.00,   1.00,   1.00,   1.00]
            # size 1000    lambda 10.0
            # expected_ratio=[ 0.18,   0.40,   0.60,   0.96,   1.00,   1.00,   1.00,   1.00]
            # size 1000    lambda 33.0
            # expected_ratio=[ 0.18,   0.48,   0.62,   1.00,   1.00,   1.00,   1.00,   1.00]
            # size 1000    lambda 100.0
            # expected_ratio=[ 0.18,   0.30,   0.66,   0.90,   1.00,   1.00,   1.00,   1.00]
            # lambda 333.
            # expected_ratio=[ worse, worse,   0.74,  

            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]
                        
            
            
            
            
            
            print(f"{_line_()}    measure with partly sorted data")
            
            
            for the_lambda in [1.,3., 10., 33., 100.]:
                
                #------------------#------------------#------------------
                size_list =                          [ 100, 1000]
                number_of_tests_list = torch.tensor([  200,  50])
                number_of_tests_list = number_of_tests_list.mul(1).to(torch.int32)
                for ii_outter_param_set in range(size_list.__len__()):
                    size = size_list[ii_outter_param_set]
                    # iota_of_dim = iota(dim)
                    number_of_tests = number_of_tests_list[ii_outter_param_set]
                    device = 'cpu'
                    # if dim>100:
                    #     device = 'cuda'
                    #     pass
                    #print(f"dim {size}   test_time {number_of_tests}    device {device}")
                #------------------#------------------#------------------
                    
                    the_max__min = []#don't modify this
                    the_max__max = []
                    the_max__avg = []
                    expected_ratio = []
                    
                    _when_start = time.perf_counter()
                    
                    half__proportion_of_sorted_part_list = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, ]
                    for half__proportion_of_sorted_part in half__proportion_of_sorted_part_list:
                        _raw_result__max = torch.empty(size=[number_of_tests])
                        _raw_result__max.fill_(torch.nan)
                        _raw_result__is_expected =torch.empty(size=[number_of_tests], dtype=torch.bool)
                        _raw_result__is_expected.fill_(False)
                        for ii__test in range(number_of_tests):
                            
                            #------------------#------------------#------------------
                            #<  init           
                            branch = torch.randn(size=[size], device=device)
                            
                            half__sorted_section_len = int(size*half__proportion_of_sorted_part)
                            half__detection_section_len = int(size*0.05)#10% accepted zone.
                            assert half__sorted_section_len > 0
                            assert half__detection_section_len > 0
                            max_of_both_half_length = max(half__sorted_section_len, half__detection_section_len)
                            #sorted_section_len = 3
                            ii_section__mid_pos = random.randint(max_of_both_half_length, size-max_of_both_half_length)
                            ii_sorted_section__start_pos    = ii_section__mid_pos - half__sorted_section_len
                            ii_sorted_section__end_pos      = ii_section__mid_pos + half__sorted_section_len
                            ii_detection_section__start_pos = ii_section__mid_pos - half__detection_section_len
                            ii_detection_section__end_pos   = ii_section__mid_pos + half__detection_section_len
                            assert ii_sorted_section__end_pos <= size
                            branch[ii_sorted_section__start_pos:ii_section__mid_pos       ] = -1.
                            branch[ii_section__mid_pos         :ii_sorted_section__end_pos] = 1.
                            
                            all_final_scores = torch.empty(size=[size-1])
                            all_final_scores.fill_(torch.nan)
                            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
                            for cut_pos in all_cut_pos:
                                leaf_1 = branch[:cut_pos]
                                leaf_2 = branch[cut_pos:]
                                this_final_score = calc_ori_final_score(branch, leaf_1, leaf_2, the_lambda=the_lambda)
                                all_final_scores[cut_pos-1] = this_final_score
                                pass
                            
                            _the_max_of_score = all_final_scores.max()
                            #<  measure
                            
                            #so the best score is expected to be in [start_pos: end_pos-1]
                            _raw_result__max[ii__test] = _the_max_of_score
                            
                            the_index_of_max_score = all_final_scores.argmax()
                            if the_index_of_max_score>=ii_detection_section__start_pos and the_index_of_max_score<ii_detection_section__end_pos -1:
                                _raw_result__is_expected[ii__test] = True
                                pass
                            #------------------#------------------#------------------
                            pass#for ii__test
                        
                            
                        the_max__min  .append(_raw_result__max.min())
                        the_max__max  .append(_raw_result__max.max())
                        the_max__avg  .append(_raw_result__max.mean())
                        expected_ratio.append(_raw_result__is_expected.sum().to(torch.float32)/number_of_tests)
                        
                        pass#for scanned_param
                    _when_end = time.perf_counter()
                    
                    #print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                    print(f"size {size}    lambda {the_lambda}")
                    print(f"the_max__min = {str_the_list(the_max__min, 3)}")#########################
                    print(f"the_max__max = {str_the_list(the_max__max, 3)}")#########################
                    print(f"the_max__avg = {str_the_list(the_max__avg, 3)}")#########################
                    print(f"expected_ratio={str_the_list(expected_ratio, 2, segment=",  ")}")#########################
                    print(f"half the param={str_the_list(half__proportion_of_sorted_part_list, 3)}")#########################
                    
                    pass#for ii_outter_param_set
            
            
                pass#/ the labmda
            
            pass#/ test
            
        # emmm is this good or bad???
        while "extreme case overfitting" and False:
            size = 100
            branch = torch.randn(size=[size])
            branch[0] = -3.7
            
            all_final_scores = torch.empty(size=[size-1])
            all_final_scores.fill_(torch.nan)
            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
            for cut_pos in all_cut_pos:
                leaf_1 = branch[:cut_pos]
                leaf_2 = branch[cut_pos:]
                this_final_score = calc_ori_final_score(branch, leaf_1, leaf_2)
                all_final_scores[cut_pos-1] = this_final_score
                pass
            
            #so the best score is expected to be in [start_pos: end_pos-1]
            
            import matplotlib.pyplot as plt
            the_max_of_score = all_final_scores.max()
            plt.plot(all_cut_pos-0.75, all_final_scores, label = 'score')
            plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1), alpha = 0.5, label = "data")
            plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.legend()
            plt.show()
            pass# /test
        
        
        
        
        
        
        while "score accumulation" and True:
            the_lambda=100
            size = 1000
            score_accumulation = torch.zeros(size=[size-1])
            test_time = 200
            for _ii in range(test_time):
                #< init 
                branch = torch.randn(size=[size])
                
                #< calc
                all_final_scores = torch.empty(size=[size-1])
                all_final_scores.fill_(torch.nan)
                all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
                for cut_pos in all_cut_pos:
                    leaf_1 = branch[:cut_pos]
                    leaf_2 = branch[cut_pos:]
                    this_final_score = calc_ori_final_score(branch, leaf_1, leaf_2, the_lambda=the_lambda)
                    all_final_scores[cut_pos-1] = this_final_score
                    pass
                
                #< measure
                score_accumulation += all_final_scores
                pass
            
            score_accumulation /= test_time
            score_accumulation += score_accumulation.flip(-1)
            
            import matplotlib.pyplot as plt
            the_max_of_score = score_accumulation.max()
            plt.plot(all_cut_pos-0.75, score_accumulation, label = 'score')
            #plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1), alpha = 0.5, label = "data")
            #plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.legend()
            plt.title(f"the_lambda {the_lambda}")
            plt.show()
            pass# /test
        
        return 
        
    ____ori_in_random_number____()
    pass




    
def calc_my_version_score(branch:torch.Tensor, leaf_1:torch.Tensor, 
                        threshold_factor_k:torch.Tensor|float = 1.1, 
                        threshold_factor_b:torch.Tensor|float = 0., )->torch.Tensor:
    if isinstance(threshold_factor_k, float):
        threshold_factor_k = torch.tensor(threshold_factor_k)
        pass
    assert threshold_factor_k>=1.
    if isinstance(threshold_factor_b, float):
        threshold_factor_b = torch.tensor(threshold_factor_b)
        pass
    assert threshold_factor_b>=0.
    
    _branch_sum = branch.sum()
    branch_sum__over_2 = _branch_sum/2.
    leaf_1_sum = leaf_1.sum()
    leaf_2_sum = _branch_sum - leaf_1_sum
    
    leaf_1__proportion = torch.tensor(float(leaf_1.shape[-1])/float(branch.shape[-1]))
    assert leaf_1__proportion >= 0. and leaf_1__proportion <= 1.
    leaf_2__proportion = 1. - leaf_1__proportion
    leaf_entropy = -1.*(    leaf_1__proportion*(leaf_1__proportion.log2()) 
                        +   leaf_2__proportion*(leaf_2__proportion.log2()) )
    
    branch_score = threshold_factor_k * (branch_sum__over_2*branch_sum__over_2) + threshold_factor_b
    leaf_score__raw = (leaf_1_sum*leaf_1_sum + leaf_2_sum*leaf_2_sum)/2.
    leaf_score = leaf_score__raw* leaf_entropy
    
    final_score = leaf_score - branch_score
    return final_score
if "basic behavior" and False:
    def ____calc_my_version_score____basic():
        branch = torch.tensor([0., 0])
        leaf_1 = branch[0:1]
        final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
        assert _tensor_equal(final_score, [0.])
        
        branch = torch.tensor([1., -1])
        leaf_1 = branch[0:1]
        final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
        assert _tensor_equal(final_score, [1.])
        
        branch = torch.tensor([2., -1])
        leaf_1 = branch[0:1]
        final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
        assert _tensor_equal(final_score, [(4. + 1)/2. - 0.25])
        
        # k and b
        branch = torch.tensor([2., -1])
        leaf_1 = branch[0:1]
        final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=12.)
        assert _tensor_equal(final_score, [(4. + 1)/2. - 0.25*12.])
        
        branch = torch.tensor([2., -1])
        leaf_1 = branch[0:1]
        final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1., threshold_factor_b = 2.345)
        assert _tensor_equal(final_score, [(4. + 1)/2. - 0.25- 2.345])
        
        return
    ____calc_my_version_score____basic()
    pass

    assert False

if "scan MY xgb in random number" and True:
    def ____MY_in_random_number____():
        
        while "sorted data" and False:
            size = 100
            branch = torch.randn(size=[size]).sort().values
            all_final_scores = torch.empty(size=[size-1])
            all_final_scores.fill_(torch.nan)
            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
            for cut_pos in all_cut_pos:
                leaf_1 = branch[:cut_pos]
                this_final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
                all_final_scores[cut_pos-1] = this_final_score
                pass
            
            import matplotlib.pyplot as plt
            the_max_of_score = all_final_scores.max()
            plt.plot(all_cut_pos, all_final_scores)
            plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1))
            plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.title("MY xgb")
            plt.show()
            pass# /test
        
        # they are indeed much smaller than the previous test.
        while "unsorted data" and False:
            size = 1000
            branch = torch.randn(size=[size])
            all_final_scores = torch.empty(size=[size-1])
            all_final_scores.fill_(torch.nan)
            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
            for cut_pos in all_cut_pos:
                leaf_1 = branch[:cut_pos]
                this_final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
                all_final_scores[cut_pos-1] = this_final_score
                pass
            
            import matplotlib.pyplot as plt
            the_max_of_score = all_final_scores.max()
            plt.plot(all_cut_pos, all_final_scores)
            plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1), alpha = 0.5)
            plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.title("MY xgb")
            plt.show()
            pass# /test
        
        
        while "partly sorted data" and False:
            size = 1000
            branch = torch.randn(size=[size])
            
            sorted_section_len = int(size*0.1)################
            #sorted_section_len = 3
            ii_sorted_section__start_pos = random.randint(0,size-sorted_section_len)
            ii_sorted_section__mid_pos = ii_sorted_section__start_pos+sorted_section_len//2
            ii_sorted_section__end_pos = ii_sorted_section__start_pos+sorted_section_len
            assert ii_sorted_section__end_pos <= size
            branch[ii_sorted_section__start_pos:ii_sorted_section__mid_pos] = -1.
            branch[ii_sorted_section__mid_pos:ii_sorted_section__end_pos] = 1.
            
            all_final_scores = torch.empty(size=[size-1])
            all_final_scores.fill_(torch.nan)
            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
            for cut_pos in all_cut_pos:
                leaf_1 = branch[:cut_pos]
                this_final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
                all_final_scores[cut_pos-1] = this_final_score
                pass
            
            #so the best score is expected to be in [start_pos: end_pos-1]
            
            import matplotlib.pyplot as plt
            the_max_of_score = all_final_scores.max()
            plt.plot(all_cut_pos-0.75, all_final_scores)
            plt.plot(torch.linspace(ii_sorted_section__start_pos,ii_sorted_section__end_pos-1,sorted_section_len-1)+0.25, 
                    [the_max_of_score]*(sorted_section_len-1), alpha = 0.6)
            plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1), alpha = 0.5)
            plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.title("MY xgb")
            plt.show()
            pass# /test
        
        if "MY ver. measure with partly sorted data. 10% size as the accepted zone." and False:
            
            # cpu   7.556877 , or 0.075569 per test
            # size 100
            # the_max__min = [ 7.099,  13.983,  12.507,  7.698,  20.724,  40.500,  142.126,  162.671]
            # the_max__max = [ 192.803,  177.048,  297.443,  277.213,  398.163,  747.584,  1002.107,  1066.696]
            # the_max__avg = [ 64.365,  65.038,  69.630,  72.417,  104.809,  214.945,  407.801,  614.280]
            # expected_ratio=[ 0.08,   0.12,   0.18,   0.26,   0.68,   0.94,   0.99,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]

            # cpu   14.160022 , or 0.708001 per test
            # size 1000
            # the_max__min = [ 318.672,  239.459,  198.375,  1040.209,  2350.057,  13779.809,  22669.254,  48424.082]
            # the_max__max = [ 1248.177,  1995.918,  2707.107,  6792.234,  15734.655,  34534.871,  49629.215,  69628.453]
            # the_max__avg = [ 672.207,  790.074,  1068.318,  2667.869,  7744.818,  20203.631,  35556.176,  58862.625]
            # expected_ratio=[ 0.20,   0.30,   0.70,   0.85,   0.95,   1.00,   1.00,   1.00]
            # half the param=[ 0.010,  0.020,  0.030,  0.050,  0.100,  0.150,  0.200,  0.250]
                        
            print(f"{_line_()}    MY ver   measure with partly sorted data")
            
            #------------------#------------------#------------------
            size_list =                          [ 100, 1000]
            number_of_tests_list = torch.tensor([  100,  20])
            number_of_tests_list = number_of_tests_list.mul(1).to(torch.int32)
            for ii_outter_param_set in range(size_list.__len__()):
                size = size_list[ii_outter_param_set]
                # iota_of_dim = iota(dim)
                number_of_tests = number_of_tests_list[ii_outter_param_set]
                device = 'cpu'
                # if dim>100:
                #     device = 'cuda'
                #     pass
                print(f"dim {size}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------
                
                the_max__min = []#don't modify this
                the_max__max = []
                the_max__avg = []
                expected_ratio = []
                
                _when_start = time.perf_counter()
                
                half__proportion_of_sorted_part_list = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, ]
                for half__proportion_of_sorted_part in half__proportion_of_sorted_part_list:
                    _raw_result__max = torch.empty(size=[number_of_tests])
                    _raw_result__max.fill_(torch.nan)
                    _raw_result__is_expected =torch.empty(size=[number_of_tests], dtype=torch.bool)
                    _raw_result__is_expected.fill_(False)
                    for ii__test in range(number_of_tests):
                        
                        #------------------#------------------#------------------
                        #<  init           
                        branch = torch.randn(size=[size], device=device)
                        
                        half__sorted_section_len = int(size*half__proportion_of_sorted_part)
                        half__detection_section_len = int(size*0.05)#10% accepted zone.
                        assert half__sorted_section_len > 0
                        assert half__detection_section_len > 0
                        max_of_both_half_length = max(half__sorted_section_len, half__detection_section_len)
                        #sorted_section_len = 3
                        ii_section__mid_pos = random.randint(max_of_both_half_length, size-max_of_both_half_length)
                        ii_sorted_section__start_pos    = ii_section__mid_pos - half__sorted_section_len
                        ii_sorted_section__end_pos      = ii_section__mid_pos + half__sorted_section_len
                        ii_detection_section__start_pos = ii_section__mid_pos - half__detection_section_len
                        ii_detection_section__end_pos   = ii_section__mid_pos + half__detection_section_len
                        assert ii_sorted_section__end_pos <= size
                        branch[ii_sorted_section__start_pos:ii_section__mid_pos       ] = -1.
                        branch[ii_section__mid_pos         :ii_sorted_section__end_pos] = 1.
                        
                        all_final_scores = torch.empty(size=[size-1])
                        all_final_scores.fill_(torch.nan)
                        all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
                        for cut_pos in all_cut_pos:
                            leaf_1 = branch[:cut_pos]
                            this_final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
                            all_final_scores[cut_pos-1] = this_final_score
                            pass
                        
                        _the_max_of_score = all_final_scores.max()
                        #<  measure
                        
                        #so the best score is expected to be in [start_pos: end_pos-1]
                        _raw_result__max[ii__test] = _the_max_of_score
                        
                        the_index_of_max_score = all_final_scores.argmax()
                        if the_index_of_max_score>=ii_detection_section__start_pos and the_index_of_max_score<ii_detection_section__end_pos -1:
                            _raw_result__is_expected[ii__test] = True
                            pass
                        #------------------#------------------#------------------
                        pass#for ii__test
                    
                        
                    the_max__min  .append(_raw_result__max.min())
                    the_max__max  .append(_raw_result__max.max())
                    the_max__avg  .append(_raw_result__max.mean())
                    expected_ratio.append(_raw_result__is_expected.sum().to(torch.float32)/number_of_tests)
                    
                    pass#for scanned_param
                _when_end = time.perf_counter()
                
                print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                print(f"size {size}")
                print(f"the_max__min = {str_the_list(the_max__min, 3)}")#########################
                print(f"the_max__max = {str_the_list(the_max__max, 3)}")#########################
                print(f"the_max__avg = {str_the_list(the_max__avg, 3)}")#########################
                print(f"expected_ratio={str_the_list(expected_ratio, 2, segment=",  ")}")#########################
                print(f"half the param={str_the_list(half__proportion_of_sorted_part_list, 3)}")#########################
                
                pass#for ii_outter_param_set
            pass#/ test
        
        # emmm is this good or bad???
        while "extreme case overfitting" and False:
            size = 100
            branch = torch.randn(size=[size])
            branch[0] = -3.7
            
            all_final_scores = torch.empty(size=[size-1])
            all_final_scores.fill_(torch.nan)
            all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
            for cut_pos in all_cut_pos:
                leaf_1 = branch[:cut_pos]
                this_final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
                all_final_scores[cut_pos-1] = this_final_score
                pass
            
            #so the best score is expected to be in [start_pos: end_pos-1]
            
            import matplotlib.pyplot as plt
            the_max_of_score = all_final_scores.max()
            plt.plot(all_cut_pos-0.75, all_final_scores, label = 'score')
            plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1), alpha = 0.5, label = "data")
            plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.legend()
            plt.title("MY xgb")
            plt.show()
            pass# /test
        
        
        while "score accumulation" and False:
            size = 1000
            score_accumulation = torch.zeros(size=[size-1])
            test_time = 200
            for _ii in range(test_time):
                #< init 
                branch = torch.randn(size=[size])
                
                #< calc
                all_final_scores = torch.empty(size=[size-1])
                all_final_scores.fill_(torch.nan)
                all_cut_pos = torch.linspace(1, size-1, steps=size-1, dtype=torch.int64)
                for cut_pos in all_cut_pos:
                    leaf_1 = branch[:cut_pos]
                    this_final_score = calc_my_version_score(branch, leaf_1, threshold_factor_k=1.)
                    all_final_scores[cut_pos-1] = this_final_score
                    pass
                
                #< measure
                score_accumulation += all_final_scores
                pass
            
            score_accumulation /= test_time
            score_accumulation += score_accumulation.flip(-1)
            
            import matplotlib.pyplot as plt
            the_max_of_score = score_accumulation.max()
            plt.plot(all_cut_pos-0.75, score_accumulation, label = 'score')
            #plt.plot(torch.linspace(0,size-1,size), (branch+2.5)*(the_max_of_score*0.1), alpha = 0.5, label = "data")
            #plt.plot(torch.linspace(0,size-1,size), [2.5*the_max_of_score*0.1]*size, alpha = 0.2)
            plt.legend()
            plt.title(f"My xgb")
            plt.show()
            pass# /test
        
        return 
        
    ____MY_in_random_number____()
    pass