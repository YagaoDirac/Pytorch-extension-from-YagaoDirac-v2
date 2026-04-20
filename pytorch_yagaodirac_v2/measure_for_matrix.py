

from pathlib import Path
import math
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal, \
    iota, is_square_matrix, \
    vector_length_norm, get_vector_length,\
    log10_avg_safe, get_mask_of_top_element__rough,\
    str_the_list
from pytorch_yagaodirac_v2.ParamMo import GradientModification__mean_len_of_something_to_1
from pytorch_yagaodirac_v2.Random import random_standard_vector, randomly_permutate__matrix, randomly_rotate__matrix
    


def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######







def LOSS__vec_len_retention__of_a_mat_in_matmul(matrix:torch.Tensor, test_time:int|None = None, 
                    at_least = -7., _debug__needs_log = False)->tuple[torch.Tensor, tuple[torch.Tensor]|None]:
    
    '''return length_retention_score, None
    
    if needs log, then return length_retention_score, (no_abs__length_retention_score,)
    
    
    The score_log10_div2 and score_log10_div2__abs maybe empty. If so, the length_retention_score is nan.
    
    the 2nd and 3rd output size can be smaller than the 1st one if any of them is smaller than the at_least
    
    The result is always >=0. The smaller the better.'''
    assert is_square_matrix(matrix)
    dim = matrix.shape[0]
    if test_time is None:
        test_time=int((dim+30)*1.2)
        if test_time>100:
            test_time = 100
            pass
        pass
    assert test_time>=1
    assert at_least<0.
    
    with torch.no_grad():
        the_device = matrix.device
        score__len_sqr___test_time = torch.empty([test_time], device = the_device)
        for epoch in range(test_time):
            vec = random_standard_vector(dim=dim, dtype=matrix.dtype, device=the_device)
            
            #vec = vec.reshape(shape=[1,-1])
            after_matmul = vec@matrix
            new_len_sqr = after_matmul.dot(after_matmul)
            
            score__len_sqr___test_time[epoch] = new_len_sqr
            
            #old
            #score_of_this_epoch = new_len_sqr.log10().abs()/2.
            #all_score[epoch] = score_of_this_epoch
            pass#/ for
        
        raw_score_log10_div2___test_time = score__len_sqr___test_time.log10()/2.
        score_log10_div2___maybe_test_time = raw_score_log10_div2___test_time[raw_score_log10_div2___test_time>=at_least]
        score_log10_div2__abs___maybe_test_time = score_log10_div2___maybe_test_time.abs()
        length_retention_score = score_log10_div2__abs___maybe_test_time.mean()
        
        if _debug__needs_log:
            no_abs__length_retention_score = score_log10_div2___maybe_test_time.mean()
            return length_retention_score, (no_abs__length_retention_score,)
        return length_retention_score, (None,)
            #(score__len_sqr___test_time, score_log10_div2___maybe_test_time, score_log10_div2__abs___maybe_test_time)
    pass#/function

if "basic test" and __DEBUG_ME__() and False:
    def ____test____LOSS__vec_len_retention__of_a_mat_in_matmul():
        if "when no halfway result is big enough, they are all eliminated. Then the result is nan" and True:
            mat = torch.eye(n=3)/10000.
            length_retention_score, _ = \
                        LOSS__vec_len_retention__of_a_mat_in_matmul(mat,test_time=2, at_least= -2)
            assert length_retention_score.isnan()
            pass
        
        if "eye is perfect" and True:
            for dim in range(3, 15):
                test_time = 10
                for test_count in range(test_time):
                    mat = torch.eye(n=dim)  
                    length_retention_score, (no_abs__length_retention_score,) = \
                                LOSS__vec_len_retention__of_a_mat_in_matmul(mat,test_time=100, _debug__needs_log = True)
                    assert _tensor_equal(no_abs__length_retention_score, [0.], epsilon=1e-6)
                    pass
                pass#for dim
            pass#/test
        if "scaled a lil bit" and True:
            for dim in range(3, 15):
                test_time = 10
                for test_count in range(test_time):
                    _to_the_power = (torch.rand(size=[])*2-1)*3.
                    _factor = torch.pow(10, _to_the_power)
                    mat = torch.eye(n=dim)*_factor
                    length_retention_score, (no_abs__length_retention_score,) = \
                                LOSS__vec_len_retention__of_a_mat_in_matmul(mat, test_time=100, _debug__needs_log = True)
                    assert _tensor_equal(no_abs__length_retention_score, [_to_the_power], epsilon=1e-6)
                    pass
                pass#for dim
            pass#/test
        
        if "rotation and permutation are all perfect" and True:
            for dim in range(3, 15):
                test_time = 10
                for test_count in range(test_time):
                    mat = torch.eye(n=dim)
                    #<  prepare the vec>
                    for _ in range(5):
                        mat = randomly_permutate__matrix(mat)
                        mat = randomly_rotate__matrix(mat)
                        pass
                    #</ prepare the vec>
                    _to_the_power = (torch.rand(size=[])*2.-1.)*3.
                    _factor = torch.pow(10, _to_the_power)
                    mat.mul_(_factor)
                    length_retention_score, (no_abs__length_retention_score,) = \
                                LOSS__vec_len_retention__of_a_mat_in_matmul(mat, test_time=100, _debug__needs_log = True)
                    assert _tensor_equal(no_abs__length_retention_score, [_to_the_power], epsilon=1e-6)
                    pass
                pass#for dim
            pass#/test
            #rotation should also be perfect. This test is done in the rand_basic_ratation_matrix's test
        
        
        
        if "randn and randn but rotated and permuted should be similar" and True:
            #result
            # result_of__length_diff_mean    = [ 0.009,  0.006, -0.006, -0.001]
            # dim_list                       = [ 2.000,  5.000,  10.000,  100.000]
            
            result_of__diff = []#don't modify this.
            #----------------#----------------#----------------
            dim_list =         [2,  5, 10, 100]
            test_time_list = [100, 50, 30, 15]
            test_time_list = [10, 5, 3, 1]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                _raw_result_of__diff = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    mat_1 = torch.randn(size=[dim,dim])
                    mat_2 = mat_1.detach().clone()
                    #<  prepare the vec>
                    for _ in range(5):
                        mat_2 = randomly_permutate__matrix(mat_2)
                        mat_2 = randomly_rotate__matrix(mat_2)
                        pass
                    #</ prepare the vec>
                    _, (mat_1__no_abs__length_retention_score,) = \
                            LOSS__vec_len_retention__of_a_mat_in_matmul(mat_1, test_time=100, _debug__needs_log = True)
                    _, (mat_2__no_abs__length_retention_score,) = \
                            LOSS__vec_len_retention__of_a_mat_in_matmul(mat_2, test_time=100, _debug__needs_log = True)
                    _this_result = mat_1__no_abs__length_retention_score-mat_2__no_abs__length_retention_score
                    #----------------#----------------#----------------
                    _raw_result_of__diff[test_count] = _this_result
                    pass
                result_of__diff.append(_raw_result_of__diff.mean())
                pass#for dim
            print(f"result_of__length_diff_mean = {str_the_list(result_of__diff    , 3)}")
            print(f"dim_list                    = {str_the_list(dim_list     , 3)}")
            pass#/test
        
        
        return 
    
    ____test____LOSS__vec_len_retention__of_a_mat_in_matmul()
    pass

# All the measurement are for square matrix. So, [dim] means [dim,dim]
# randn[dim,dim] >>> 0.5*log10(dim)
# randn@randn    >>>  1.*log10(dim) 
# randn[dim,dim]/sqrt(dim) >>> 0.

# rand [dim,dim] >>> 0.5*log10(dim) -0.32
# rand @rand     >>> 1.5*log10(dim) -0.86

# rand*2-1       >>> 0.5*log10(dim)-0.24
# rand*2-1 @ rand*2-1 >>> 1.*log10(dim)-0.26
if "measure the random init" and __DEBUG_ME__() and True:
    def ____test____measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10___2():
        
        if "uniformed test of all above" and False:
            
            if True:
                # randn
                # the_min_gt_this_list  = [ 0.337,  0.983,  1.497]
                # the_max_lt_this_list  = [ 0.588,  1.012,  1.502]
                # the_mean_eq_this_list = [ 0.478,  0.998,  1.500]
                # dim_list               = [ 10,    100,    1000]
                # 0.5*log10(dim)
                
                # randn / sqrt(dim)
                # the_min_gt_this_list  = [-0.137, -0.017, -0.003]
                # the_max_lt_this_list  = [ 0.089,  0.015,  0.002]
                # the_mean_eq_this_list = [-0.021, -0.002, -0.000]
                # dim_list               = [ 10,    100,    1000]
                # 0.*log10(dim), yeah, 0.
                # higher print precision below.
                
                # randn @ randn
                # the_min_gt_this_list  = [ 0.771,  1.976,  2.995]
                # the_max_lt_this_list  = [ 1.109,  2.017,  3.003]
                # the_mean_eq_this_list = [ 0.956,  1.996,  3.000]
                # dim_list               = [ 10,    100,    1000]
                # 1.*log10(dim)
                
                # rand
                # the_min_gt_this_list  = [ 0.059,  0.617,  1.140]
                # the_max_lt_this_list  = [ 0.283,  0.737,  1.230]
                # the_mean_eq_this_list = [ 0.174,  0.679,  1.181]
                # dim_list               = [ 10,    100,    1000]
                # 0.5*log10(dim) -0.32
                
                
                # rand @ rand
                # the_min_gt_this_list  = [ 0.507,  2.011,  3.515]
                # the_max_lt_this_list  = [ 0.901,  2.272,  3.747]
                # the_mean_eq_this_list = [ 0.699,  2.147,  3.631]
                # dim_list               = [ 10,    100,    1000]
                # 1.5*log10(dim)-0.86
                
                # rand*2-1
                # the_min_gt_this_list  = [ 0.170,  0.744,  1.259]
                # the_max_lt_this_list  = [ 0.326,  0.772,  1.265]
                # the_mean_eq_this_list = [ 0.243,  0.759,  1.261]
                # dim_list               = [ 10,    100,    1000]
                # 0.5*log10(dim)-0.24
                
                # rand*2-1 @ rand*2-1
                # the_min_gt_this_list  = [ 0.529,  1.674,  2.702]
                # the_max_lt_this_list  = [ 1.005,  1.819,  2.789]
                # the_mean_eq_this_list = [ 0.751,  1.743,  2.740]
                # dim_list               = [ 10,    100,    1000]
                # 1.*log10(dim)-0.26
                pass
            
            from typing import Literal
            
            #--------------------#--------------------#--------------------
            how_to_gen_the_mat__list:Literal["randn", "randn / sqrt(dim)", "randn @ randn", 
                    "rand", "rand @ rand", "rand*2-1", "rand*2-1 @ rand*2-1", ] \
                = [
                    #"randn",
                    # rand_mat = torch.randn(size=[dim,dim], device=device)
                    
                    #"randn / sqrt(dim)", 
                    # rand_mat = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)
                    
                    #"randn @ randn",
                    # rand_mat = torch.randn(size=[dim,dim], device=device)
                    # rand_mat = rand_mat@torch.randn(size=[dim,dim], device=device)
                    
                    #"rand",
                    # rand_mat = torch.rand (size=[dim,dim], device=device)#rand
                    
                    "rand @ rand",
                    # rand_mat = torch.rand (size=[dim,dim], device=device)#rand
                    # rand_mat = rand_mat @ torch.rand (size=[dim,dim], device=device)#rand
                    
                    "rand*2-1",
                    # rand_mat = torch.rand (size=[dim,dim], device=device) *2.-1.#rand*2-1
                    
                    "rand*2-1 @ rand*2-1",
                    # rand_mat = torch.rand (size=[dim,dim], device=device) *2.-1.#rand*2-1
                    # rand_mat = rand_mat @ torch.rand (size=[dim,dim], device=device) *2.-1.#rand*2-1
                    
                    ]
            for how_to_gen_the_mat in how_to_gen_the_mat__list:
                print(f"{how_to_gen_the_mat}")
            #--------------------#--------------------#--------------------
            
                mean_of__score_log10_div2__min =  []#don't modify here.
                mean_of__score_log10_div2__max =  []#don't modify here.
                mean_of__score_log10_div2__avg =  []#don't modify here.
                
                #--------------------#--------------------#--------------------
                dim_list = [10,100,1000]
                test_time_list = torch.tensor([2000,1000,200])
                test_time_list = test_time_list.mul(1).to(torch.int64)
                with_div_sqrt_of_dim = False
                for inner_iter_count in range(dim_list.__len__()):
                    dim = dim_list[inner_iter_count]
                    test_time = test_time_list[inner_iter_count]
                    device = 'cpu'
                    if dim>100:
                        device = 'cuda'
                        pass
                    #print(f"test_time {test_time}    device {device}")
                #--------------------#--------------------#--------------------

                    _raw_result = torch.empty(size=[test_time])
                    for test_count in range(test_time):
                        
                        #--------------------#--------------------#--------------------
                        if how_to_gen_the_mat == "randn":
                            rand_mat = torch.randn(size=[dim,dim], device=device)
                            pass
                        elif how_to_gen_the_mat == "randn / sqrt(dim)":
                            rand_mat = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)
                            pass
                        elif how_to_gen_the_mat == "randn @ randn":
                            rand_mat = torch.randn(size=[dim,dim], device=device)
                            rand_mat = rand_mat@torch.randn(size=[dim,dim], device=device)
                            pass
                        elif how_to_gen_the_mat == "rand":
                            rand_mat = torch.rand (size=[dim,dim], device=device)#rand
                            pass
                        elif how_to_gen_the_mat == "rand @ rand":
                            rand_mat = torch.rand (size=[dim,dim], device=device)#rand
                            rand_mat = rand_mat @ torch.rand (size=[dim,dim], device=device)#rand
                            pass
                        elif how_to_gen_the_mat == "rand*2-1":
                            rand_mat = torch.rand (size=[dim,dim], device=device) *2.-1.#rand*2-1
                            pass
                        elif how_to_gen_the_mat == "rand*2-1 @ rand*2-1":
                            rand_mat = torch.rand (size=[dim,dim], device=device) *2.-1.#rand*2-1
                            rand_mat = rand_mat @ torch.rand (size=[dim,dim], device=device) *2.-1.#rand*2-1
                            pass
                        else:
                            assert False, "bad param: how_to_gen_the_mat"
                        
                        _, (no_abs__length_retention_score,) = \
                                    LOSS__vec_len_retention__of_a_mat_in_matmul(rand_mat, _debug__needs_log = True)
                        
                        _this_result = no_abs__length_retention_score
                        #--------------------#--------------------#--------------------
                        _raw_result[test_count] = _this_result
                        
                        pass
                    mean_of__score_log10_div2__min.append(_raw_result.min() )
                    mean_of__score_log10_div2__max.append(_raw_result.max() )
                    mean_of__score_log10_div2__avg.append(_raw_result.mean())
                    pass# for macro_iter_count
                
                print(f"the_min_gt_this_list  = {str_the_list(mean_of__score_log10_div2__min, 3)}")    
                print(f"the_max_lt_this_list  = {str_the_list(mean_of__score_log10_div2__max, 3)}")    
                print(f"the_mean_eq_this_list = {str_the_list(mean_of__score_log10_div2__avg, 3)}")    
                print(f"dim_list               = {str_the_list(dim_list, 0, ",   ")}")
                pass#for how_to_gen_the_mat
            
            pass#/test
        
        if "the higher print precision here. randn[dim,dim] / sqrt(dim)" and False:
            # output:
            # the_min_gt_this_list  = [-0.14032, -0.01610, -0.00370]
            # the_max_lt_this_list  = [ 0.07667,  0.01356,  0.00182]
            # the_mean_eq_this_list = [-0.02223, -0.00216, -0.00042]
            # dim_list               = [ 10,      100,      1000]
            
            print("randn[dim,dim] with or without /sqrt(dim)")
            
            mean_of__score_log10_div2__min =  []#don't modify here.
            mean_of__score_log10_div2__max =  []#don't modify here.
            mean_of__score_log10_div2__avg =  []#don't modify here.
            
            #--------------------#--------------------#--------------------
            dim_list = [10,100,1000]
            test_time_list = torch.tensor([2000,1000,200])
            test_time_list = test_time_list.mul(1).to(torch.int64)
            with_div_sqrt_of_dim = False
            for inner_iter_count in range(dim_list.__len__()):
                dim = dim_list[inner_iter_count]
                test_time = test_time_list[inner_iter_count]
                device = 'cpu'
                if dim>100:
                    device = 'cuda'
                    pass
                print(f"test_time {test_time}    device {device}")
            #--------------------#--------------------#--------------------

                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    
                    #--------------------#--------------------#--------------------
                    rand_mat = torch.randn(size=[dim,dim], device=device)
                    if with_div_sqrt_of_dim:
                        rand_mat = rand_mat/math.sqrt(dim)
                        pass
                    _, (no_abs__length_retention_score,) = \
                                LOSS__vec_len_retention__of_a_mat_in_matmul(rand_mat, _debug__needs_log = True)
                    
                    _this_result = no_abs__length_retention_score
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    
                    pass
                mean_of__score_log10_div2__min.append(_raw_result.min() )
                mean_of__score_log10_div2__max.append(_raw_result.max() )
                mean_of__score_log10_div2__avg.append(_raw_result.mean())
                pass# for macro_iter_count
            print(f"with_div_sqrt_of_dim {with_div_sqrt_of_dim}")    
            print(f"the_min_gt_this_list  = {str_the_list(mean_of__score_log10_div2__min, 5)}")    
            print(f"the_max_lt_this_list  = {str_the_list(mean_of__score_log10_div2__max, 5)}")    
            print(f"the_mean_eq_this_list = {str_the_list(mean_of__score_log10_div2__avg, 5)}")    
            print(f"dim_list               = {str_the_list(dim_list, 0, ",     ")}")
            pass#/test
        
        return 
    
    ____test____measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10___2()
    pass

if "I still don't understand what happened in this test. No plan now."and False:
    '''
    if True:
        
        assert False,"  1w"
        assert False,"  1w"
        assert False,"  1w"
        assert False,"  1w"
        "unstable. idk why...."
        accumulate_score = torch.tensor([0.])
        for _ in range(116):
            dim = random.randint(2,300)
            rand = random.random()*0.1+0.1
            rand_mat = torch.randn(size=[dim,dim])*rand
            mat1 = torch.eye(n=dim)*math.cos(0.1)+rand_mat*math.sin(0.1)
            mat2 = torch.eye(n=dim)*math.cos(0.2)+rand_mat*math.sin(0.2)
            result_for_1 = LOSS__vec_len_retention__of_a_mat_in_matmul(mat1,test_time=100)[0]
            result_for_2 = LOSS__vec_len_retention__of_a_mat_in_matmul(mat2,test_time=100)[0]
            assert result_for_1 < result_for_2
            accumulate_score += result_for_2 - result_for_1
            pass
        
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        #1w
        
        #some affine matrix.
        "unstable. idk why...."
        for _ in range(155):
            dim = random.randint(2,300)
            rand = random.random()*0.1+0.1
            mat1 = torch.eye(n=dim)
            mat2 = torch.eye(n=dim)
            mat1[0,1] = rand
            mat2[0,1] = rand*1.1
            result_for_1 = LOSS__vec_len_retention__of_a_mat_in_matmul(mat1,test_time=100)[0]
            result_for_2 = LOSS__vec_len_retention__of_a_mat_in_matmul(mat2,test_time=100)[0]
            assert result_for_1<result_for_2
            pass
        #return 
    #____test____measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10()
    '''
    pass



def LOSS__behavior_similarity(input:torch.Tensor, target:torch.Tensor,
                            test_time:int|None = None, cap = 5., _debug__needs_ref_result = False,) \
                ->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    '''if _debug__needs_ref_result is True, return length__log_10__diff__abs_avg, angle_diff__avg, 
    (length__log_10__diff__ori, length__log_10__diff__avg, angle_diff, angle_diff__avg). 
    
    else: return length__log_10__diff__abs_avg, angle_diff__avg, None
    
    return length__log_10__diff__ori, length__log_10__diff__avg, length__log_10__diff__abs_avg, angle_diff, angle_diff__avg
    
    The result is always >=0. The smaller the better.
    
    This tool measures how much the matmul behavior is similar between 2 matrixs.'''
    #<  check dim>
    assert is_square_matrix(input)
    assert is_square_matrix(target)
    dim = input.shape[0]
    #import math
    #_1_over_sqrt_of_dim = math.pow(dim,-0.5)
    assert target.shape[0] == dim
    #</ check dim>
    #<  test time>
    if test_time is None:
        test_time=dim+30
        pass
    assert test_time>=1
    assert cap>1.#at least >0
    #</ test time>
    
    with torch.no_grad():
        the_device = input.device
        length__log_10__diff = torch.empty([test_time], device = the_device)
        angle_diff           = torch.empty([test_time], device = the_device)
        for epoch in range(test_time):
            vec = random_standard_vector(dim, dtype=input.dtype, device=input.device)
            
            to_test = vec@input
            as_ref = vec@target
            
            length_of_test = get_vector_length(to_test)
            length_of_ref = get_vector_length(as_ref)
            
            ____temp = length_of_test.log10() - length_of_ref.log10()
            length__log_10__diff[epoch] = ____temp
            del ____temp
            
            to_test.div_(length_of_test)
            as_ref.div_(length_of_ref)
            assert _tensor_equal(get_vector_length(to_test), torch.tensor([1.],device=input.device))
            assert _tensor_equal(get_vector_length(as_ref), torch.tensor([1.],device=input.device))
            
            ____temp = to_test.dot(as_ref)
            angle_diff[epoch] = 1.-____temp
            
            pass#/ for
        
        if not _debug__needs_ref_result:
            
            length__log_10__diff = length__log_10__diff.clamp_max_(cap)
            length__log_10__diff = length__log_10__diff.clamp_min_(-cap)
            #all_score = all_score.cpu()
            
            length__log_10__diff__abs_avg = length__log_10__diff.abs().mean()
            
            angle_diff__avg = angle_diff.mean()
            
            return length__log_10__diff__abs_avg, angle_diff__avg, None
        
        else:
            length__log_10__diff__ori = length__log_10__diff.detach().clone()
            length__log_10__diff = length__log_10__diff.clamp_max_(cap)
            length__log_10__diff = length__log_10__diff.clamp_min_(-cap)
            #all_score = all_score.cpu()
            
            length__log_10__diff__avg = length__log_10__diff.mean()
            length__log_10__diff__abs_avg = length__log_10__diff.abs().mean()
            
            angle_diff__avg = angle_diff.mean()
            
            return length__log_10__diff__abs_avg, angle_diff__avg, (length__log_10__diff__ori, length__log_10__diff__avg, angle_diff, angle_diff__avg)
            #old code 
            # return length__log_10__diff__ori, length__log_10__diff__avg, length__log_10__diff__abs_avg, angle_diff, angle_diff__avg
        
    pass#/function

if "basic behavior test" and __DEBUG_ME__() and False:
    def ____test____basic____LOSS__behavior_similarity():
        # set up breakpoint and read.
        mat_1 = torch.tensor([  
                                [1., 0 ],
                                [0,  1.],
                                ])
        mat_2 = torch.tensor([  
                                [0,  1.],
                                [1., 0 ],
                                ])
        length__log_10__diff__abs_avg, angle_diff__avg, (length__log_10__diff__ori, length__log_10__diff__avg, angle_diff, angle_diff__avg) = \
                LOSS__behavior_similarity(mat_1, mat_2,test_time=1, _debug__needs_ref_result = True)
        
        return
    
    ____test____basic____LOSS__behavior_similarity()
    pass

if "test" and __DEBUG_ME__() and True:
    def ____test____LOSS__behavior_similarity():
        import random
        "part 1, 2 similar matrixs."
        if "self and self*?, the same direction." and False:
            #----------------#----------------#----------------
            dim_list =         [2,  5, 10, 100]
            test_time_list = [100, 50, 30, 15]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                for _ in range(test_time):
                    #----------------#----------------#----------------
                    mat_ref = torch.randn(size=[dim,dim])
                    to_the_power = (random.random()-0.5)*5.
                    _factor = math.pow(10,to_the_power)
                    mat = mat_ref.detach()*_factor
                    length__log_10__diff__abs_avg, angle_diff__avg, (length__log_10__diff__ori, length__log_10__diff__avg, angle_diff, angle_diff__avg) = \
                            LOSS__behavior_similarity(mat, mat_ref, _debug__needs_ref_result = True)
                    assert _tensor_equal(length__log_10__diff__ori, torch.ones(size=length__log_10__diff__ori.shape)*to_the_power)
                    assert _tensor_equal(angle_diff__avg,           torch.zeros(size=angle_diff__avg.shape))
                    #----------------#----------------#----------------
                    pass
                pass#for outter
            pass#/ test
        
        if "manually make 2 very similar mats." and False:
            #result
            # 1 vs 1.1
            # result_of__length_diff_min     = [-0.0285, -0.0063, -0.0025, -0.0001]
            # result_of__length_diff_max     = [ 0.0520,  0.0148,  0.0048,  0.0002]
            # result_of__length_diff_avg     = [ 0.0149,  0.0021,  0.0005,  0.0000]
            # result_of__length_diff_abs_avg = [ 0.0202,  0.0035,  0.0010,  0.0000]
            # result_of__angle_diff_max      = [ 0.0325,  0.0019,  0.0004,  0.0000]
            # result_of__angle_diff_avg      = [ 0.0022,  0.0003,  0.0001,  0.0000]
            # dim_list                       = [ 2.0000,  5.0000,  10.000,  100.00]
            
            # 0 vs 0.1
            # result_of__length_diff_min     = [-0.0795, -0.0104, -0.0034, -0.0002]
            # result_of__length_diff_max     = [ 0.0928,  0.0114,  0.0036,  0.0002]
            # result_of__length_diff_avg     = [ 0.0025,  0.0001,  0.0000,  0.0000]
            # result_of__length_diff_abs_avg = [ 0.0255,  0.0029,  0.0009,  0.0000]
            # result_of__angle_diff_max      = [ 0.1546,  0.0025,  0.0004,  0.0000]
            # result_of__angle_diff_avg      = [ 0.0143,  0.0003,  0.0001,  0.0000]
            # dim_list                       = [ 2.0000,  5.0000,  10.0000,  100.0000]
            
            #maybe the angle result is more different.
            
            result_of__length_diff_min     = []
            result_of__length_diff_max     = []
            result_of__length_diff_avg     = []
            result_of__length_diff_abs_avg = []
            result_of__angle_diff_max      = []
            result_of__angle_diff_avg      = []
            #----------------#----------------#----------------
            dim_list =         [2,  5, 10, 100]
            test_time_list = [1000, 500, 300, 100]
            #test_time_list = [10, 5, 3, 1]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                _raw_result_of__length_diff_min     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_max     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_avg     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_abs_avg = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_max      = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_avg      = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    mat_1 = torch.randn(size=[dim,dim])
                    mat_1[0,0] = 0.#1.
                    mat_2 = mat_1.detach().clone()
                    mat_1[0,0] = 0.1#1.1
                    
                    length__log_10__diff__abs_avg, angle_diff__avg, (length__log_10__diff__ori, length__log_10__diff__avg, angle_diff, angle_diff__avg) = \
                            LOSS__behavior_similarity(mat_1, mat_2, _debug__needs_ref_result = True)
                    
                    _raw_result_of__length_diff_min[test_count] = length__log_10__diff__ori.min()
                    _raw_result_of__length_diff_max[test_count] = length__log_10__diff__ori.max()
                    _raw_result_of__length_diff_avg[test_count] = length__log_10__diff__avg
                    _raw_result_of__length_diff_abs_avg[test_count] = length__log_10__diff__abs_avg
                    _raw_result_of__angle_diff_max[test_count] = angle_diff.max()
                    _raw_result_of__angle_diff_avg[test_count] = angle_diff__avg
                    #----------------#----------------#----------------
                    pass
                result_of__length_diff_min    .append(_raw_result_of__length_diff_min    .mean())
                result_of__length_diff_max    .append(_raw_result_of__length_diff_max    .mean())
                result_of__length_diff_avg    .append(_raw_result_of__length_diff_avg    .mean())
                result_of__length_diff_abs_avg.append(_raw_result_of__length_diff_abs_avg.mean())
                result_of__angle_diff_max     .append(_raw_result_of__angle_diff_max     .mean())
                result_of__angle_diff_avg     .append(_raw_result_of__angle_diff_avg     .mean())
                pass#for outter
            print(f"result_of__length_diff_min     = {str_the_list(result_of__length_diff_min    , 4)}")
            print(f"result_of__length_diff_max     = {str_the_list(result_of__length_diff_max    , 4)}")
            print(f"result_of__length_diff_avg     = {str_the_list(result_of__length_diff_avg    , 4)}")
            print(f"result_of__length_diff_abs_avg = {str_the_list(result_of__length_diff_abs_avg, 4)}")
            print(f"result_of__angle_diff_max      = {str_the_list(result_of__angle_diff_max     , 4)}")
            print(f"result_of__angle_diff_avg     = {str_the_list(result_of__angle_diff_avg      , 4)}")
            print(f"dim_list                       = {str_the_list(dim_list     , 4)}")
            pass#/ test
        
        if "scan the difference_factor in the last test" and False:
            #result
            #base 0.
            # result_of__length_diff_min     = [-0.0000, -0.0004, -0.0033, -0.0112, -0.0314]
            # result_of__length_diff_max     = [ 0.0000,  0.0004,  0.0037,  0.0121,  0.0401]
            # result_of__length_diff_avg     = [ 0.0000,  0.0000,  0.0000,  0.0002,  0.0019]
            # result_of__length_diff_abs_avg = [ 0.0000,  0.0001,  0.0009,  0.0029,  0.0097]
            # result_of__angle_diff_max      = [ 0.0000,  0.0000,  0.0004,  0.0035,  0.0362]
            # result_of__angle_diff_avg      = [ 0.0000,  0.0000,  0.0001,  0.0005,  0.0058]
            # diff_list                      = [ 0.0010,  0.0100,  0.1000,  0.3000,  1.0000]
            
            #base 1.
            # result_of__length_diff_min     = [-0.0000, -0.0003, -0.0025, -0.0073, -0.0210]
            # result_of__length_diff_max     = [ 0.0000,  0.0005,  0.0049,  0.0148,  0.0525]
            # result_of__length_diff_avg     = [ 0.0000,  0.0000,  0.0005,  0.0017,  0.0065]
            # result_of__length_diff_abs_avg = [ 0.0000,  0.0001,  0.0010,  0.0032,  0.0106]
            # result_of__angle_diff_max      = [ 0.0000,  0.0000,  0.0004,  0.0033,  0.0337]
            # result_of__angle_diff_avg      = [ 0.0000,  0.0000,  0.0001,  0.0005,  0.0054]
            # diff_list                      = [ 0.0010,  0.0100,  0.1000,  0.3000,  1.0000]
            
            result_of__length_diff_min     = []
            result_of__length_diff_max     = []
            result_of__length_diff_avg     = []
            result_of__length_diff_abs_avg = []
            result_of__angle_diff_max      = []
            result_of__angle_diff_avg      = []
            #----------------#----------------#----------------
            _base = 1.
            dim = 10
            test_time = 100
            diff_list = [0.001,0.01,0.1,0.3,1.]
            #test_time_list = [10, 5, 3, 1]
            for outter_param_count in range(diff_list.__len__()):
                diff = diff_list[outter_param_count]
            #----------------#----------------#----------------
                _raw_result_of__length_diff_min     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_max     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_avg     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_abs_avg = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_max      = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_avg      = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    mat_1 = torch.randn(size=[dim,dim])
                    mat_1[0,0] = _base
                    mat_2 = mat_1.detach().clone()
                    mat_1[0,0] = _base + diff
                    
                    length__log_10__diff__abs_avg, angle_diff__avg, (length__log_10__diff__ori, length__log_10__diff__avg, angle_diff, angle_diff__avg) = \
                            LOSS__behavior_similarity(mat_1, mat_2, _debug__needs_ref_result = True)
                    
                    _raw_result_of__length_diff_min[test_count] = length__log_10__diff__ori.min()
                    _raw_result_of__length_diff_max[test_count] = length__log_10__diff__ori.max()
                    _raw_result_of__length_diff_avg[test_count] = length__log_10__diff__avg
                    _raw_result_of__length_diff_abs_avg[test_count] = length__log_10__diff__abs_avg
                    _raw_result_of__angle_diff_max[test_count] = angle_diff.max()
                    _raw_result_of__angle_diff_avg[test_count] = angle_diff__avg
                    #----------------#----------------#----------------
                    pass
                result_of__length_diff_min    .append(_raw_result_of__length_diff_min    .mean())
                result_of__length_diff_max    .append(_raw_result_of__length_diff_max    .mean())
                result_of__length_diff_avg    .append(_raw_result_of__length_diff_avg    .mean())
                result_of__length_diff_abs_avg.append(_raw_result_of__length_diff_abs_avg.mean())
                result_of__angle_diff_max     .append(_raw_result_of__angle_diff_max     .mean())
                result_of__angle_diff_avg     .append(_raw_result_of__angle_diff_avg     .mean())
                pass#for outter
            print(f"result_of__length_diff_min     = {str_the_list(result_of__length_diff_min    , 4)}")
            print(f"result_of__length_diff_max     = {str_the_list(result_of__length_diff_max    , 4)}")
            print(f"result_of__length_diff_avg     = {str_the_list(result_of__length_diff_avg    , 4)}")
            print(f"result_of__length_diff_abs_avg = {str_the_list(result_of__length_diff_abs_avg, 4)}")
            print(f"result_of__angle_diff_max      = {str_the_list(result_of__angle_diff_max     , 4)}")
            print(f"result_of__angle_diff_avg     = {str_the_list(result_of__angle_diff_avg      , 4)}")
            print(f"diff_list                      = {str_the_list(diff_list     , 4)}")
            pass#/ test
        
        
        
        "part 2, 2 randn matrixs."
        if "2 randn?" and False:
            #result
            # result_of__length_diff_min     = [-0.4867, -0.4228, -0.3007, -0.1128]
            # result_of__length_diff_max     = [ 0.4763,  0.4071,  0.2989,  0.1144]
            # result_of__length_diff_avg     = [-0.0029, -0.0013, -0.0013, -0.0003]
            # result_of__length_diff_abs_avg = [ 0.2956,  0.1686,  0.1144,  0.0353]
            # result_of__angle_diff_min      = [ 0.2677,  0.2356,  0.3839,  0.7453]
            # result_of__angle_diff_max      = [ 1.7229,  1.7608,  1.6046,  1.2548]
            # result_of__angle_diff_avg      = [ 0.9972,  0.9991,  1.0022,  1.0011]
            # dim_list                       = [ 2.000,  5.000,    10.000,  100.000]
            # the angle is basically orthogonal, the same as expected.
            
            print("2 randn?")
            result_of__length_diff_min     = []
            result_of__length_diff_max     = []
            result_of__length_diff_avg     = []
            result_of__length_diff_abs_avg = []
            result_of__angle_diff_min      = []
            result_of__angle_diff_max      = []
            result_of__angle_diff_avg      = []
            #----------------#----------------#----------------
            dim_list =         [2,  5, 10, 100]
            test_time_list = [1000, 500, 300, 100]
            #test_time_list = [10, 5, 3, 1]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                _raw_result_of__length_diff_min     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_max     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_avg     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_abs_avg = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_min      = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_max      = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_avg      = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    mat_1 = torch.randn(size=[dim,dim])
                    mat_2 = torch.randn(size=[dim,dim])
                    
                    length__log_10__diff__abs_avg, angle_diff__avg, (length__log_10__diff__ori, length__log_10__diff__avg, angle_diff, angle_diff__avg) = \
                            LOSS__behavior_similarity(mat_1, mat_2, _debug__needs_ref_result = True)
                    
                    _raw_result_of__length_diff_min[test_count] = length__log_10__diff__ori.min()
                    _raw_result_of__length_diff_max[test_count] = length__log_10__diff__ori.max()
                    _raw_result_of__length_diff_avg[test_count] = length__log_10__diff__avg
                    _raw_result_of__length_diff_abs_avg[test_count] = length__log_10__diff__abs_avg
                    _raw_result_of__angle_diff_min[test_count] = angle_diff.min()
                    _raw_result_of__angle_diff_max[test_count] = angle_diff.max()
                    _raw_result_of__angle_diff_avg[test_count] = angle_diff__avg
                    #----------------#----------------#----------------
                    pass
                result_of__length_diff_min    .append(_raw_result_of__length_diff_min    .mean())
                result_of__length_diff_max    .append(_raw_result_of__length_diff_max    .mean())
                result_of__length_diff_avg    .append(_raw_result_of__length_diff_avg    .mean())
                result_of__length_diff_abs_avg.append(_raw_result_of__length_diff_abs_avg.mean())
                result_of__angle_diff_min     .append(_raw_result_of__angle_diff_min     .mean())
                result_of__angle_diff_max     .append(_raw_result_of__angle_diff_max     .mean())
                result_of__angle_diff_avg     .append(_raw_result_of__angle_diff_avg     .mean())
                pass#for outter
            print(f"result_of__length_diff_min     = {str_the_list(result_of__length_diff_min    , 4)}")
            print(f"result_of__length_diff_max     = {str_the_list(result_of__length_diff_max    , 4)}")
            print(f"result_of__length_diff_avg     = {str_the_list(result_of__length_diff_avg    , 4)}")
            print(f"result_of__length_diff_abs_avg = {str_the_list(result_of__length_diff_abs_avg, 4)}")
            print(f"result_of__angle_diff_min      = {str_the_list(result_of__angle_diff_min     , 4)}")
            print(f"result_of__angle_diff_max      = {str_the_list(result_of__angle_diff_max     , 4)}")
            print(f"result_of__angle_diff_avg     = {str_the_list(result_of__angle_diff_avg     ,  4)}")
            print(f"dim_list                       = {str_the_list(dim_list     , 3)}")
            pass#/ test
        
        if "ref for 2 randn?" and False:
            #result
            # result_of__angle_diff_min  = [ 0.0000,  0.0666,  0.2562,  0.7783,  0.9294]
            # result_of__angle_diff_max  = [ 2.0000,  1.9464,  1.8249,  1.2406,  1.0955]
            # result_of__angle_diff_avg  = [ 1.0099,  1.0024,  0.9994,  0.9927,  0.9982]
            # dim_list                   = [ 2.000,  5.000,  10.000,  100.000,  1000.000]
            
            print("ref for 2 randn?")
            result_of__angle_diff_min= []
            result_of__angle_diff_max= []
            result_of__angle_diff_avg= []
            #----------------#----------------#----------------
            dim_list =         [2,  5, 10, 100, 1000]
            test_time_list = [1000, 500, 300, 100, 100]
            #test_time_list = [10, 5, 3, 1]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                _raw_result      = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    dot_product = random_standard_vector(dim=dim).dot(random_standard_vector(dim=dim))
                    _this_result = 1. - dot_product
                    
                    _raw_result[test_count] = _this_result
                    #----------------#----------------#----------------
                    pass
                result_of__angle_diff_min.append(_raw_result.min())
                result_of__angle_diff_max.append(_raw_result.max())
                result_of__angle_diff_avg.append(_raw_result.mean())
                pass#for outter
            print(f"result_of__angle_diff_min  = {str_the_list(result_of__angle_diff_min     , 4)}")
            print(f"result_of__angle_diff_max  = {str_the_list(result_of__angle_diff_max     , 4)}")
            print(f"result_of__angle_diff_avg = {str_the_list(result_of__angle_diff_avg     ,  4)}")
            print(f"dim_list                   = {str_the_list(dim_list     , 3)}")
            pass#/ test
        
        
        
        "part 3, 2 rand matrixs."
        if "2 rand ?" and False:
            #result
            # result_of__length_diff_min     = [-0.4191, -0.3433, -0.2460, -0.0968]
            # result_of__length_diff_max     = [ 0.4565,  0.3339,  0.2534,  0.0992]
            # result_of__length_diff_avg     = [ 0.0171, -0.0009,  0.0021,  0.0002]
            # result_of__length_diff_abs_avg = [ 0.2250,  0.1231,  0.0844,  0.0265]
            # result_of__angle_diff_min      = [ 0.0126,  0.0315,  0.0420,  0.0406]
            # result_of__angle_diff_max      = [ 1.4124,  1.5155,  1.3866,  1.1511]
            # result_of__angle_diff_avg      = [ 0.3538,  0.4444,  0.4619,  0.4858]
            # dim_list                       = [ 2.000,  5.000,    10.000,  100.000]
            
            # angle is getting closer to 0.5. I don't believe it's a mathmatical 0.5.
            # some other ref tests below.
            
            print("2 rand ?")
            result_of__length_diff_min     = []
            result_of__length_diff_max     = []
            result_of__length_diff_avg     = []
            result_of__length_diff_abs_avg = []
            result_of__angle_diff_min      = []
            result_of__angle_diff_max      = []
            result_of__angle_diff_avg      = []
            #----------------#----------------#----------------
            dim_list =         [2,  5, 10, 100]
            test_time_list = [1000, 500, 300, 100]
            #test_time_list = [10, 5, 3, 1]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                _raw_result_of__length_diff_min     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_max     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_avg     = torch.empty(size=[test_time])
                _raw_result_of__length_diff_abs_avg = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_min      = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_max      = torch.empty(size=[test_time])
                _raw_result_of__angle_diff_avg      = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    mat_1 = torch.rand (size=[dim,dim])#rand
                    mat_2 = torch.rand (size=[dim,dim])#rand
                    
                    length__log_10__diff__abs_avg, angle_diff__avg, (length__log_10__diff__ori, length__log_10__diff__avg, angle_diff, angle_diff__avg) = \
                            LOSS__behavior_similarity(mat_1, mat_2, _debug__needs_ref_result = True)
                    
                    _raw_result_of__length_diff_min[test_count] = length__log_10__diff__ori.min()
                    _raw_result_of__length_diff_max[test_count] = length__log_10__diff__ori.max()
                    _raw_result_of__length_diff_avg[test_count] = length__log_10__diff__avg
                    _raw_result_of__length_diff_abs_avg[test_count] = length__log_10__diff__abs_avg
                    _raw_result_of__angle_diff_min[test_count] = angle_diff.min()
                    _raw_result_of__angle_diff_max[test_count] = angle_diff.max()
                    _raw_result_of__angle_diff_avg[test_count] = angle_diff__avg
                    #----------------#----------------#----------------
                    pass
                result_of__length_diff_min    .append(_raw_result_of__length_diff_min    .mean())
                result_of__length_diff_max    .append(_raw_result_of__length_diff_max    .mean())
                result_of__length_diff_avg    .append(_raw_result_of__length_diff_avg    .mean())
                result_of__length_diff_abs_avg.append(_raw_result_of__length_diff_abs_avg.mean())
                result_of__angle_diff_min     .append(_raw_result_of__angle_diff_min     .mean())
                result_of__angle_diff_max     .append(_raw_result_of__angle_diff_max     .mean())
                result_of__angle_diff_avg     .append(_raw_result_of__angle_diff_avg     .mean())
                pass#for outter
            print(f"result_of__length_diff_min     = {str_the_list(result_of__length_diff_min    , 4)}")
            print(f"result_of__length_diff_max     = {str_the_list(result_of__length_diff_max    , 4)}")
            print(f"result_of__length_diff_avg     = {str_the_list(result_of__length_diff_avg    , 4)}")
            print(f"result_of__length_diff_abs_avg = {str_the_list(result_of__length_diff_abs_avg, 4)}")
            print(f"result_of__angle_diff_min      = {str_the_list(result_of__angle_diff_min     , 4)}")
            print(f"result_of__angle_diff_max      = {str_the_list(result_of__angle_diff_max     , 4)}")
            print(f"result_of__angle_diff_avg     = {str_the_list(result_of__angle_diff_avg     ,  4)}")
            print(f"dim_list                       = {str_the_list(dim_list     , 3)}")
            pass#/ test
        
        if "some test on the algo itself as a ref." and False:
            dim = 1000
            
            vec_1 = torch.randn(size=[1,dim])
            #vec_1 -= vec_1.mean()
            #vec_1 += offset
            vec_1 = vector_length_norm(vec_1)
            vec_1_mean = vec_1.mean()
            if vec_1_mean<0:
                vec_1 *= -1.
                vec_1_mean *= -1.
                pass
            vec_1 = vec_1.reshape([-1])
            
            vec_2 = torch.randn(size=[1,dim])
            #vec_2 -= vec_2.mean()
            #vec_2 += offset
            vec_2 = vector_length_norm(vec_2)
            vec_2_mean = vec_2.mean()
            if vec_2_mean<0:
                vec_2 *= -1.
                vec_2_mean *= -1.
                pass
            vec_2 = vec_2.reshape([-1])
            
            dot_prod = vec_1.dot(vec_2)
            pass#/ test
        
        if "vec_randn @ mat_rand   the sign is not very random. same_sign/_diff_sign is between 2.9 to 3.4" and False:
            dim = 10000
            same_sign = 0
            diff_sign = 0
            for _ in range(50):
                vec = torch.randn(size=[dim])
                if vec.mean()<0:
                    vec *= -1.
                    pass
                
                mat_1 = torch.rand(size=[dim,dim])
                vec_1 = vec@mat_1
                sign_1:torch.Tensor = vec_1.gt(0.)
                
                mat_2 = torch.rand(size=[dim,dim])
                vec_2 = vec@mat_2
                sign_2:torch.Tensor = vec_2.gt(0.)
                
                both_pos = sign_1.logical_and(sign_2).sum()
                both_neg = dim - sign_1.logical_or(sign_2).sum()
                pos_1_2_neg = sign_1.logical_and(sign_2.logical_not()).sum()
                neg_1_2_pos = sign_1.logical_not().logical_and(sign_2).sum()
                assert both_pos+both_neg+pos_1_2_neg+neg_1_2_pos == dim
                #print(f"++{both_pos:4} --{both_neg:4}  {both_pos+both_neg:4}//{pos_1_2_neg+neg_1_2_pos:4}  +-{pos_1_2_neg:4} -+{neg_1_2_pos:4} ")
                same_sign += both_pos+both_neg
                diff_sign += pos_1_2_neg+neg_1_2_pos
                pass
            print(f"same_sign {same_sign:6}     diff_sign {diff_sign:6}")
            pass#/ test
        
        if "The v_randn @ M_rand 's mean is affected by the v's mean.    for each case     VISUALISATION!    v_randn is not very accurate." and False:
            #<  The visualisation version>
            
            # 用不同的rand mat去乘同一个，看看是不是都有类似的偏移。
            # 现在已知的结论是，这个vec的mean会控制最终乘出来的vec的整体的分布，所以是一个mean=最开始的mean*可能乘以维度吧，的一个正态。
            # 应该是这个事情导致了最终的角度相似性的很神奇的行为。
            # 可能是dim/sqrt2
            
            #vec = random_standard_vector(dim=1000)
            vec = torch.randn(size=[1000])
            print(f"vec.mean()   {vec.mean()}")
            for _ in range(111):
                mat = torch.rand (size=[1000,1000])
                result_vec = vec@mat
                print(result_vec.mean())
                from matplotlib import pyplot as plt
            
                n_bins = 30############### modify this.

                fig, axs = plt.subplots(1, 1, tight_layout=True)

                # We can set the number of bins with the *bins* keyword argument.
                axs.hist(result_vec.tolist(), bins=n_bins)
                #axs.hist(mat.reshape([-1]).tolist(), bins=n_bins)
                
                plt.show()
                
                axs.clear()
                pass
            #</ The visualisation version>
        
        if "The v_randn @ M_rand 's mean is affected by the v's mean.    the entire thing  VISUALISATION!    v_randn is not very accurate." and False:
            # 用不同的rand mat去乘同一个，看看是不是都有类似的偏移。
            # 现在已知的结论是，这个vec的mean会控制最终乘出来的vec的整体的分布，所以是一个mean=最开始的mean*可能乘以维度吧，的一个正态。
            # 应该是这个事情导致了最终的角度相似性的很神奇的行为。
            # 可能是dim/sqrt2
            
            #vec = random_standard_vector(dim=1000)
            test_time = 300
            test_range = 0.2
            dim = 1000
            
            x_data = torch.linspace(-test_range, test_range, test_time).tolist()
            y_data = []
            
            for the_x in x_data:
                #<  prepare the vec>
                vec = torch.randn(size=[dim])
                vec -= vec.mean()
                assert _tensor_equal(vec.mean(), [0.])
                vec += the_x
                assert _tensor_equal(vec.mean(), [the_x])
                #</ prepare the vec>
                #<  calc>
                mat = torch.rand (size=[dim,dim])
                result_vec = vec@mat
                #</ calc>
                y_data.append(result_vec.mean())
                pass
            
            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(1, 1, tight_layout=True)
            axs.plot(x_data,y_data)
            plt.show()
                
            pass#/ test
        
        if "The v_randn @ M_rand 's mean is affected by the v's mean.    the entire thing                    v_randn is not very accurate." and False:
            #result
            # dim 100       test_time1000
            # result_min = [-10.935, -5.947, -0.915,  3.986,  9.077]
            # result_max = [-9.176, -4.107,  1.039,  5.902,  10.828]
            # result_avg = [-9.998, -5.011, -0.006,  5.005,  10.003]
            # the_x_list = [-0.200, -0.100,  0.000,  0.100,  0.200]
            # dim 1000       test_time100
            # result_min = [-100.67, -50.697, -0.646,  49.088,  99.110]
            # result_max = [-99.075, -49.356,  0.660,  50.730,  100.778]
            # result_avg = [-99.985, -49.998, -0.017,  49.999,  99.984]
            # the_x_list = [-0.200,  -0.100,  0.000,   0.100,   0.200]
            # dim 10000       test_time10
            # result_min = [-1000.40, -500.337, -0.449,  499.564,  999.076]
            # result_max = [-999.470, -499.441,  0.249,  500.536,  1000.414]
            # result_avg = [-999.891, -500.026, -0.147,  499.986,  999.962]
            # the_x_list = [-0.200,   -0.100,    0.000,  0.100,    0.200]
            
            # 0.5*the_x*dim
            
            # 用不同的rand mat去乘同一个，看看是不是都有类似的偏移。
            # 现在已知的结论是，这个vec的mean会控制最终乘出来的vec的整体的分布，所以是一个mean=最开始的mean*可能乘以维度吧，的一个正态。
            # 应该是这个事情导致了最终的角度相似性的很神奇的行为。
            
            #vec = random_standard_vector(dim=1000)
            dim_list = [100, 1000, 10000]
            test_time_list = [1000,100,10]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
                
                result_min = []#dont modify this
                result_max = []#dont modify this
                result_avg = []#dont modify this
                
                the_x_list = [-0.2, -0.1, 0., 0.1, 0.2,]
                for inner_param_count in range(the_x_list.__len__()):
                    the_x = the_x_list[inner_param_count]
                    
                    _raw_result = torch.empty(size=[test_time])
                    for test_count in range(test_time):
                        #<  prepare the vec>
                        vec = torch.randn(size=[dim])
                        vec -= vec.mean()
                        assert _tensor_equal(vec.mean(), [0.])
                        vec += the_x
                        assert _tensor_equal(vec.mean(), [the_x])
                        #</ prepare the vec>
                        #<  calc>
                        mat = torch.rand (size=[dim,dim])
                        result_vec = vec@mat
                        #</ calc>
                        _raw_result[test_count] = result_vec.mean()
                        pass
                    
                    
                    result_min.append(_raw_result.min() )
                    result_max.append(_raw_result.max() )
                    result_avg.append(_raw_result.mean())
                    pass#for test_range
                
                print(f"dim {dim}       test_time{test_time}")
                print(f"result_min = {str_the_list(result_min, 3)}")
                print(f"result_max = {str_the_list(result_max, 3)}")
                print(f"result_avg = {str_the_list(result_avg, 3)}")
                print(f"the_x_list = {str_the_list(the_x_list, 3)}")
                pass#for outter_param
            pass#/ test
        
        
        
        if "irrelevant test, just for fun. vec_rand dot vec_rand" and False:
            #result
            # result_of__angle_diff_min  = [ 0.0000,  0.0017,  0.0367,  0.1828,  0.2261]
            # result_of__angle_diff_max  = [ 0.9163,  0.7605,  0.5372,  0.3549,  0.2700]
            # result_of__angle_diff_avg = [ 0.1570,  0.2268,  0.2405,  0.2501,  0.2484]
            # dim_list                   = [ 2.000,  5.000,  10.000,  100.000,  1000.000]
            # I dirived the formula, the result is 3/4. The measurement is 1-the result, so it's 1/4.
            # This result is correct.
            
            
            print("ref for 2 rand ?")
            result_of__angle_diff_min= []
            result_of__angle_diff_max= []
            result_of__angle_diff_avg= []
            #----------------#----------------#----------------
            dim_list =         [2,  5, 10, 100, 1000]
            test_time_list = [1000, 500, 300, 100, 100]
            #test_time_list = [10, 5, 3, 1]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                _raw_result      = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    vec_1 = torch.rand(size=[1, dim])
                    vec_1 = vector_length_norm(vec_1).reshape([-1])
                    vec_2 = torch.rand(size=[1, dim])
                    vec_2 = vector_length_norm(vec_2).reshape([-1])
                    
                    dot_product = vec_1.dot(vec_2)
                    _this_result = 1. - dot_product
                    
                    _raw_result[test_count] = _this_result
                    #----------------#----------------#----------------
                    pass
                result_of__angle_diff_min.append(_raw_result.min())
                result_of__angle_diff_max.append(_raw_result.max())
                result_of__angle_diff_avg.append(_raw_result.mean())
                pass#for outter
            print(f"result_of__angle_diff_min  = {str_the_list(result_of__angle_diff_min     , 4)}")
            print(f"result_of__angle_diff_max  = {str_the_list(result_of__angle_diff_max     , 4)}")
            print(f"result_of__angle_diff_avg = {str_the_list(result_of__angle_diff_avg     ,  4)}")
            print(f"dim_list                   = {str_the_list(dim_list     , 3)}")
            pass#/ test
        
        if "irrelevant test, vec_randn.abs @ vec_randn.abs " and False:
            #result
            # result_of__angle_diff_min  = [ 0.0000,  0.0144,  0.0785,  0.2601,  0.3199]
            # result_of__angle_diff_max  = [ 0.9702,  0.8939,  0.6802,  0.4946,  0.4023]
            # result_of__angle_diff_avg = [ 0.1801,  0.3039,  0.3379,  0.3645,  0.3639]
            # dim_list                   = [ 2.000,  5.000,  10.000,  100.000,  1000.000]
            # randn.abs is very different from rand.
            
            print("irrelevant test, vec_randn.abs @ vec_randn.abs ")
            result_of__angle_diff_min= []
            result_of__angle_diff_max= []
            result_of__angle_diff_avg= []
            #----------------#----------------#----------------
            dim_list =         [2,  5,   10,  100, 1000]
            test_time_list = [1000, 500, 300, 100, 100]
            #test_time_list = [10, 5, 3, 1]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                _raw_result      = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    dot_product = random_standard_vector(dim=dim).abs().dot(random_standard_vector(dim=dim).abs())
                    _this_result = 1. - dot_product
                    
                    _raw_result[test_count] = _this_result
                    #----------------#----------------#----------------
                    pass
                result_of__angle_diff_min.append(_raw_result.min())
                result_of__angle_diff_max.append(_raw_result.max())
                result_of__angle_diff_avg.append(_raw_result.mean())
                pass#for outter
            print(f"result_of__angle_diff_min  = {str_the_list(result_of__angle_diff_min     , 4)}")
            print(f"result_of__angle_diff_max  = {str_the_list(result_of__angle_diff_max     , 4)}")
            print(f"result_of__angle_diff_avg = {str_the_list(result_of__angle_diff_avg     ,  4)}")
            print(f"dim_list                   = {str_the_list(dim_list     , 3)}")
            pass#/ test
        
        return
    
    ____test____LOSS__behavior_similarity()
    pass



def LOSS__mat_is_standard_orthogonal(matrix:torch.Tensor, _result_log10_at_least = -10., 
                                _correct_offset_for_angle_score = -0.4, 
                                _debug__needs_log = False)\
                    -> tuple[torch.Tensor,torch.Tensor, list[tuple[str, torch.Tensor]]|None]:
    '''return  len_loss, angle_loss, _log
    
    The result is always >=0. The smaller the better.'''
    assert is_square_matrix(matrix)
    assert _result_log10_at_least < -2., "if you know what you are doing, modify this line."
    
    assert _correct_offset_for_angle_score in [0., -1., -0.4], "if you know what you are doing..."
    #I tested with torch.randn. -0.4 is recommended. It helps low dim cases to result the same result as high dim cases.
    
    dim = matrix.shape[0]
    
    mat_sqr = matrix*matrix
    assert mat_sqr.shape == matrix.shape
    
    if _debug__needs_log:
        _log:list[tuple[str, torch.Tensor]]|None = [("****** if you see this, the code is wrong.", torch.empty(size=[]))]*2
        pass
    else:
        _log = None
        pass
    
    
    #<  horizontal length score>
    #所以要输出的东西，这个应该是1的原始值，log10之后的原始值，再abs再mean的实际分数。
    hor_sum__as_len_sqr___1_is_good__dim = mat_sqr.sum(dim=1)# if it's 1, it's good. 
    hor_sum__as_len_log10___0_is_good__dim = hor_sum__as_len_sqr___1_is_good__dim.log10()/2.# if it's 0, it's good. 
    
    hor_len_score__raw = hor_sum__as_len_log10___0_is_good__dim[hor_sum__as_len_log10___0_is_good__dim>_result_log10_at_least]
    hor_len_score__raw_mean_without_abs   = hor_len_score__raw.mean()# as a ref. 
    hor_len_score = hor_len_score__raw.abs().mean()# [almost RETURN VALUE]
    if _log is not None:
        _log.append(("hor_sum__as_len_sqr___1_is_good__dim", hor_sum__as_len_sqr___1_is_good__dim))#[2]
        _log.append(("hor_sum__as_len_log10___0_is_good__dim", hor_sum__as_len_log10___0_is_good__dim))#[3]
        _log.append(("hor_len_score__raw", hor_len_score__raw))#[4]
        _log.append(("hor_len_score__raw_mean_without_abs", hor_len_score__raw_mean_without_abs))#[5]
        _log.append(("hor_len_score", hor_len_score))#[6]
        pass
    
    assert hor_len_score.ge(_result_log10_at_least)
    #assert hor_sum__as_len_log10___0_is_good__dim.ge(_result_log10_at_least)#??????????????
    
    #old code
    #hor_len_score__mse = hor_sum__as_len_sqr__should_be_1.mean().sqrt()
    #hor_len_score__mae = hor_len.mean()
    #</ horizontal length score>
    
    
    #<  vertical length score>
    ver_sum__as_len_sqr___1_is_good__dim = mat_sqr.sum(dim=0)# if it's 1, it's good.
    ver_sum__as_len_log10___0_is_good__dim = ver_sum__as_len_sqr___1_is_good__dim.log10()/2.# if it's 0, it's good. 
    
    ver_len_score__raw = ver_sum__as_len_log10___0_is_good__dim[ver_sum__as_len_log10___0_is_good__dim>_result_log10_at_least]
    ver_len_score__raw_mean_without_abs   = ver_len_score__raw.mean()# as a ref. 
    ver_len_score = ver_len_score__raw.abs().mean()# [almost RETURN VALUE]
    if _log is not None:
        _log.append(("ver_sum__as_len_sqr___1_is_good__dim", ver_sum__as_len_sqr___1_is_good__dim))#[7]
        _log.append(("ver_sum__as_len_log10___0_is_good__dim", ver_sum__as_len_log10___0_is_good__dim))
        _log.append(("ver_len_score__raw", ver_len_score__raw))
        _log.append(("ver_len_score__raw_mean_without_abs", ver_len_score__raw_mean_without_abs))
        _log.append(("ver_len_score", ver_len_score))#[11]
        
        pass
    
    assert ver_len_score.ge(_result_log10_at_least)
    #assert ver_sum__as_len_log10___0_is_good__dim.ge(_result_log10_at_least)#???????????
    
    #old code
    #ver_len_score__mse = ver_sum__as_len_sqr.mean().sqrt()
    #ver_len_score__mae = ver_len.mean()
    #</ vertical length score>
    
    
    iota_of_dim = iota(dim, device=matrix.device)
    
    #<  horizontal sub vectors   angle score>  
    hor_len = hor_sum__as_len_sqr___1_is_good__dim.sqrt()
    matrix_into_hor_len_1 = matrix/(hor_len.reshape([-1,1]).expand([-1,dim]))
    if _log is not None:
        _log.append(("matrix_into_hor_len_1 before nan_to_num", matrix_into_hor_len_1.detach().clone()))#[10]
        pass
    matrix_into_hor_len_1.nan_to_num_(0.)# just in case, if any /0, there's nan.
    #assert _tensor_equal(get_vector_length(matrix_into_hor_len_1[0]), [1.]) or _tensor_equal(get_vector_length(matrix_into_hor_len_1[0]), [0.])
    hor_vec_angle_test = matrix_into_hor_len_1 @ (matrix_into_hor_len_1.T)                                    
    _assert_only___diagonal_of_hor = hor_vec_angle_test[iota_of_dim, iota_of_dim]
    assert _assert_only___diagonal_of_hor.abs().lt(0.0001).logical_or(_assert_only___diagonal_of_hor.sub(1.).abs().lt(0.0001)).all()#near 0 or 1.
    hor_vec_angle_test[iota_of_dim, iota_of_dim] = 0.
    hor_angle_score = hor_vec_angle_test.abs().mean()*dim/(dim-1)# [almost RETURN VALUE]
    if _log is not None:
        _log.append(("hor_vec_angle_test", hor_vec_angle_test))#[11]
        _log.append(("hor_angle_score", hor_angle_score))#[12]
        pass
    #</ horizontal sub vectors   angle score>    
    
    
    #<  vertical sub vectors   angle score>    
    ver_len = ver_sum__as_len_sqr___1_is_good__dim.sqrt()
    matrix_into_ver_len_1 = matrix/(ver_len.reshape([1,-1]).expand([dim,-1]))
    if _log is not None:
        _log.append(("matrix_into_ver_len_1 before nan_to_num", matrix_into_ver_len_1.detach().clone()))#[13]
        pass
    matrix_into_ver_len_1.nan_to_num_(0.)# just in case, if any /0, there's nan.
    #assert _tensor_equal(get_vector_length(matrix_into_ver_len_1[:,0]), [1.]) or _tensor_equal(get_vector_length(matrix_into_ver_len_1[:,0]), [0.])
    ver_vec_angle_test = (matrix_into_ver_len_1.T) @ matrix_into_ver_len_1                              
    _assert_only___diagonal_of_ver = ver_vec_angle_test[iota_of_dim, iota_of_dim]
    assert _assert_only___diagonal_of_ver.abs().lt(0.0001).logical_or(_assert_only___diagonal_of_ver.sub(1.).abs().lt(0.0001)).all()#near 0 or 1.
    ver_vec_angle_test[iota_of_dim, iota_of_dim] = 0.
    ver_angle_score = ver_vec_angle_test.abs().mean()*dim/(dim-1)# [almost RETURN VALUE]
    if _log is not None:
        _log.append(("ver_vec_angle_test", ver_vec_angle_test))#[14]
        _log.append(("ver_angle_score", ver_angle_score))#[15]
        pass
    #</ vertical sub vectors   angle score>    
    
    
    len_score = (hor_len_score+ver_len_score)/2.
    angle_score_raw = hor_angle_score+ver_angle_score
    angle_score = angle_score_raw*math.sqrt(dim + _correct_offset_for_angle_score)
    #              If you don't want to scan this _correct_offset_for_angle_score yourself
    # I scanned it for you. -0.4 works as I expected and simplifies your life.
    # So, yeah, simply don't touch it.
    
    
    if _log is not None:
        _log[0] = ("sum of two len_score__raw_mean_without_abs", \
            (hor_len_score__raw_mean_without_abs + ver_len_score__raw_mean_without_abs)/2.)#[0]
        _log[1] = ("sum of two raw angle_score", angle_score_raw)#[1]
        
        pass
    return  len_score, angle_score, _log

if "old code" and False:
    # old code.
    # _temp___all_should_near_0 = matrix@(matrix.T)-torch.eye(n=dim, device=matrix.device)
    # iota_of_dim = iota(dim, device=matrix.device)
    # __debug_only__temp = _temp___all_should_near_0[iota_of_dim, iota_of_dim]
    # __debug_only__temp = __debug_only__temp.abs()
    # __debug_only__temp = __debug_only__temp.mean()
    # length_score__mse = _temp___all_should_near_0[iota_of_dim, iota_of_dim].abs().mean().sqrt()
    # length_score__mae = _temp___all_should_near_0[iota_of_dim, iota_of_dim].abs().sqrt().mean()
    
    # _temp___all_should_near_0[iota_of_dim, iota_of_dim] = 0.
    # angle_score_part_1 = _temp___all_should_near_0.abs().mean()
    
    # _temp___all_should_near_0 = (matrix.T)@matrix-torch.eye(n=dim, device=matrix.device)
    # _temp___all_should_near_0[iota_of_dim, iota_of_dim] = 0.
    # angle_score_part_2 = _temp___all_should_near_0.abs().mean()
    # angle_score = angle_score*dim/(dim-1)
    # return length_score__mse, length_score__mae, angle_score
    pass

if "test" and __DEBUG_ME__() and True:
    "for a perfect matrix of this test, the rotation and permutation of it is also perfect. "
    def ____test____basic_behavior____LOSS__the_mat_is_standard_orthogonal():
        if "some manually written values." and True:
            #1 1
            #0 0
            mat = torch.tensor([[1.,1],
                                [0, 0]])
            len_score, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
            assert _log[2][0] == 'hor_sum__as_len_sqr___1_is_good__dim'
            assert _tensor_equal(_log[2][1], [2., 0])
            assert _log[3][0] == 'hor_sum__as_len_log10___0_is_good__dim'
            assert _tensor_equal(   _log[3][1][0], [math.log10(2.)/2.])
            assert                  _log[3][1][1].isneginf()
            assert _log[6][0] == 'hor_len_score'
            assert _tensor_equal(_log[6][1], [math.log10(2.)/2.])# the neginf is not counted.
            
            assert _log[7][0] == 'ver_sum__as_len_sqr___1_is_good__dim'
            assert _tensor_equal(_log[7][1], [1., 1])
            assert _log[8][0] == 'ver_sum__as_len_log10___0_is_good__dim'
            assert _tensor_equal(_log[8][1], [0., 0])
            assert _log[11][0] == 'ver_len_score'
            assert _tensor_equal(_log[11][1], [0.])
            
            assert _tensor_equal(len_score, [math.log10(2.)/4.])# not updated?
            assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
            assert _tensor_equal(_log[0][1], [math.log10(2.)/4.])# not updated?
            
            assert _log[14][0] == 'hor_angle_score'
            assert _tensor_equal(_log[14][1], [0.])
            assert _log[17][0] == 'ver_angle_score'
            assert _tensor_equal(_log[17][1], [1.])
            assert _log[1][0] == 'sum of two raw angle_score'
            assert _tensor_equal(_log[1][1], [1.])
            
            
            # this doesn't pass
            # for _ in range(7):
            #     mat = torch.tensor([[1.,1],
            #                         [0, 0]])
            #     mat = randomly_rotate__matrix(mat)
            #     mat = randomly_permutate__matrix(mat)
            #     len_score, _, _log = LOSS__the_mat_is_standard_orthogonal(mat)
            #     assert _tensor_equal(len_score, [math.log10(2.)/2.])
            #     assert _log[1][0] == 'sum of two raw angle_score'
            #     assert _tensor_equal(_log[1][1], [1.])
            #     pass
            
            mat = torch.tensor([[0.1, 0.1],
                                [0,   0]])
            len_score, _, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
            
            assert _tensor_equal(len_score, [abs(math.log10(2.)/4.-1)])# not updated?
            assert _log[1][0] == 'sum of two raw angle_score'
            assert _tensor_equal(_log[1][1], [1.])
            assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
            assert _tensor_equal(_log[0][1], [math.log10(2.)/4.-1])# not updated?
            
            #1 0
            #1 0
            mat = torch.tensor([[1.,0],
                                [1, 0]])
            len_score, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
            assert _log[2][0] == 'hor_sum__as_len_sqr___1_is_good__dim'
            assert _tensor_equal(_log[2][1], [1., 1])
            assert _log[3][0] == 'hor_sum__as_len_log10___0_is_good__dim'
            assert _tensor_equal(_log[3][1], [0., 0])
            assert _log[6][0] == 'hor_len_score'
            assert _tensor_equal(_log[6][1], [0.])
            
            assert _log[7][0] == 'ver_sum__as_len_sqr___1_is_good__dim'
            assert _tensor_equal(_log[7][1], [2., 0])
            assert _log[8][0] == 'ver_sum__as_len_log10___0_is_good__dim'
            assert _tensor_equal(   _log[8][1][0], [math.log10(2.)/2.])
            assert                  _log[8][1][1].isneginf()
            assert _log[11][0] == 'ver_len_score'
            assert _tensor_equal(_log[11][1], [math.log10(2.)/2.])# the neginf is not counted.
            
            assert _tensor_equal(len_score, [math.log10(2.)/4.])# not updated?
            assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
            assert _tensor_equal(_log[0][1], [math.log10(2.)/4.])# not updated?
            
            assert _log[14][0] == 'hor_angle_score'
            assert _tensor_equal(_log[14][1], [1.])
            assert _log[17][0] == 'ver_angle_score'
            assert _tensor_equal(_log[17][1], [0.])
            assert _log[1][0] == 'sum of two raw angle_score'
            assert _tensor_equal(_log[1][1], [1.])
            
            
            # this doesn't pass
            # for _ in range(7):
            #     mat = torch.tensor([[1.,0],
            #                         [1, 0]])
            #     mat = randomly_rotate__matrix(mat)
            #     mat = randomly_permutate__matrix(mat)
            #     len_score, _, _log = LOSS__the_mat_is_standard_orthogonal(mat)
            #     assert _tensor_equal(len_score, [math.log10(2.)/2.])
            #     assert _log[1][0] == 'sum of two raw angle_score'
            #     assert _tensor_equal(_log[1][1], [1.])
            #     pass
            
            #1 -1
            #1 -1
            mat = torch.tensor([[1.,-1],
                                [1, -1]])
            len_score, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
            assert _log[2][0] == 'hor_sum__as_len_sqr___1_is_good__dim'
            assert _tensor_equal(_log[2][1], [2., 2])
            assert _log[3][0] == 'hor_sum__as_len_log10___0_is_good__dim'
            assert _tensor_equal(_log[3][1], [math.log10(2.)/2., math.log10(2.)/2.])
            assert _log[6][0] == 'hor_len_score'
            assert _tensor_equal(_log[6][1], [math.log10(2.)/2.])
            
            assert _log[7][0] == 'ver_sum__as_len_sqr___1_is_good__dim'
            assert _tensor_equal(_log[7][1], [2., 2])
            assert _log[8][0] == 'ver_sum__as_len_log10___0_is_good__dim'
            assert _tensor_equal(_log[8][1], [math.log10(2.)/2., math.log10(2.)/2.])
            assert _log[11][0] == 'ver_len_score'
            assert _tensor_equal(_log[11][1], [math.log10(2.)/2.])
            
            assert _tensor_equal(len_score, [math.log10(2.)/2.])# not updated?
            assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
            assert _tensor_equal(_log[0][1], [math.log10(2.)/2.])# not updated?
            
            assert _log[14][0] == 'hor_angle_score'
            assert _tensor_equal(_log[14][1], [1.])
            assert _log[17][0] == 'ver_angle_score'
            assert _tensor_equal(_log[17][1], [1.])
            assert _log[1][0] == 'sum of two raw angle_score'
            assert _tensor_equal(_log[1][1], [2.])
            
            # this doesn't pass
            # for _ in range(7):
            #     mat = torch.tensor([[1.,-1],
            #                         [1, -1]])
            #     mat = randomly_rotate__matrix(mat)
            #     mat = randomly_permutate__matrix(mat)
            #     len_score, _, _log = LOSS__the_mat_is_standard_orthogonal(mat)
            #     assert _tensor_equal(len_score, [math.log10(2.)])
            #     assert _log[1][0] == 'sum of two raw angle_score'
            #     assert _tensor_equal(_log[1][1], [2.])
            #     pass
            
            
            #I forgot the name of the matrix
            mat = torch.tensor([[1., 1, 1, 1],
                                [1 , 1,-1,-1],
                                [1 ,-1,-1, 1],
                                [1 ,-1, 1,-1],])
            len_score, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
            assert _log[2][0] == 'hor_sum__as_len_sqr___1_is_good__dim'
            assert _tensor_equal(_log[2][1], [4.]*4)
            assert _log[3][0] == 'hor_sum__as_len_log10___0_is_good__dim'
            assert _tensor_equal(_log[3][1], [math.log10(4.)/2.]*4)
            assert _log[6][0] == 'hor_len_score'
            assert _tensor_equal(_log[6][1], [math.log10(4.)/2.])
            
            assert _log[7][0] == 'ver_sum__as_len_sqr___1_is_good__dim'
            assert _tensor_equal(_log[7][1], [4.]*4)
            assert _log[8][0] == 'ver_sum__as_len_log10___0_is_good__dim'
            assert _tensor_equal(_log[8][1], [math.log10(4.)/2.]*4)
            assert _log[11][0] == 'ver_len_score'
            assert _tensor_equal(_log[11][1], [math.log10(4.)/2.])
            
            assert _tensor_equal(len_score, [math.log10(4.)/2.])# not updated?
            assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
            assert _tensor_equal(_log[0][1], [math.log10(4./2.)])# not updated?
            
            assert _log[14][0] == 'hor_angle_score'
            assert _tensor_equal(_log[14][1], [0.])
            assert _log[17][0] == 'ver_angle_score'
            assert _tensor_equal(_log[17][1], [0.])
            assert _log[1][0] == 'sum of two raw angle_score'
            assert _tensor_equal(_log[1][1], [0.])
            
            for _ in range(17):
                mat = torch.tensor([[1., 1, 1, 1],
                                    [1 , 1,-1,-1],
                                    [1 ,-1,-1, 1],
                                    [1 ,-1, 1,-1],])
                mat = randomly_rotate__matrix(mat)
                mat = randomly_permutate__matrix(mat)
                len_score, _, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
                assert _tensor_equal(len_score, [math.log10(4.)/2.])# not updated?
                assert _log[1][0] == 'sum of two raw angle_score'
                assert _tensor_equal(_log[1][1], [0.])
                pass
            
            pass#/test group.
        
        
        if "rotation+permutation is perfect in this test" and True:
            #------------------#------------------#------------------
            dim_list =       [2,     10, 100, 1000]
            test_time_list = [1000,1000, 100,   10]
            for inner_param_set in range(dim_list.__len__()):
                dim = dim_list[inner_param_set]
                test_time = test_time_list[inner_param_set]
                print(test_time)
            #------------------#------------------#------------------
                for test_count in range(test_time):
                    #<  init a standard orthogonal matrix.
                    mat = torch.eye(n=dim)
                    mat = randomly_rotate__matrix(mat)
                    mat = randomly_permutate__matrix(mat)
                    #</ init a standard orthogonal matrix.
                    
                    len_score, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
                    assert _log[2][0] == 'hor_sum__as_len_sqr___1_is_good__dim'
                    assert _tensor_equal(_log[2][1], [1.]*dim)
                    assert _log[3][0] == 'hor_sum__as_len_log10___0_is_good__dim'
                    assert _tensor_equal(_log[3][1], [0.]*dim)
                    assert _log[6][0] == 'hor_len_score'
                    assert _tensor_equal(_log[6][1], [0.])# the neginf is not counted.
                    
                    assert _log[7][0] == 'ver_sum__as_len_sqr___1_is_good__dim'
                    assert _tensor_equal(_log[7][1], [1.]*dim)
                    assert _log[8][0] == 'ver_sum__as_len_log10___0_is_good__dim'
                    assert _tensor_equal(_log[8][1], [0.]*dim)
                    assert _log[11][0] == 'ver_len_score'
                    assert _tensor_equal(_log[11][1], [0.])# the neginf is not counted.
                    
                    assert _tensor_equal(len_score, [0.])
                    assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                    assert _tensor_equal(_log[0][1], [0.])
                    
                    assert _log[14][0] == 'hor_angle_score'
                    assert _tensor_equal(_log[14][1], [0.])
                    assert _log[17][0] == 'ver_angle_score'
                    assert _tensor_equal(_log[17][1], [0.])
                    assert _log[1][0] == 'sum of two raw angle_score'
                    assert _tensor_equal(_log[1][1], [0.])
                    pass
                pass#for inner param set
            pass#/test
        
        return 
    
    #____test____basic_behavior____LOSS__the_mat_is_standard_orthogonal()
    
    
    
    def ____test____measure_the_random_init____LOSS__the_mat_is_standard_orthogonal():
        if "the reason to correct the angle_score" and False:
            print("the reason to correct the angle_score")
            #result
            # log10_of__angle_score_min = [-1.260, -1.305, -1.778]#the first 3 lines are log10(per element, read the code.)
            # log10_of__angle_score_max = [-0.501, -1.238, -1.773]
            # log10_of__angle_score_avg = [-0.784, -1.273, -1.775]
            # corrected_angle_score_min = [ 1.127,  1.546,  1.592]#the following 3 lines are abs.
            # corrected_angle_score_max = [ 2.299,  1.648,  1.598]
            # corrected_angle_score_avg = [ 1.605,  1.597,  1.596]
            # dim_list                  = [ 10.0,  100.0,  1000.0]
            
            #btw, corrected_angle_score for randn is 1.60
        
            log10_of__angle_score_min = []#don't modify this
            log10_of__angle_score_max = []#don't modify this
            log10_of__angle_score_avg = []#don't modify this
            corrected_angle_score_min = []#don't modify this
            corrected_angle_score_max = []#don't modify this
            corrected_angle_score_avg = []#don't modify this
            #------------------#------------------#------------------
            dim_list =       [   10,  100, 1000]
            test_time_list = [2000, 1000,  100]
            #test_time_list = [200, 100,  10]
            for inner_param_set in range(dim_list.__len__()):
                dim = dim_list[inner_param_set]
                test_time = test_time_list[inner_param_set]
                print(test_time)
            #------------------#------------------#------------------
            
                raw_corrected_angle_score = torch.empty(size=[test_time])
                log10_of__raw_angle_score = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #------------------#------------------#------------------
                    mat = torch.randn(size=[dim,dim])/math.sqrt(dim)
                    len_score, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
                    assert _log[13][0] == "hor_vec_angle_test"
                    _tensor_in_log = _log[13][1]
                    
                    part_1 = _tensor_in_log[dim//2:,:dim//2]
                    #assert not part_1.eq(0.).any()#may not stable.
                    log10_avg__for_part_1 = log10_avg_safe(part_1)
                    
                    part_2 = _tensor_in_log[:dim//2,dim//2:]
                    assert not part_2.eq(0.).any()#may not stable.
                    log10_avg__for_part_2 = log10_avg_safe(part_2)
                    
                    _this_result = (log10_avg__for_part_1+log10_avg__for_part_2)/2.
                    #------------------#------------------#------------------
                    log10_of__raw_angle_score[test_count] = _this_result
                    raw_corrected_angle_score[test_count] = angle_score
                    pass
                log10_of__angle_score_min .append(log10_of__raw_angle_score.min ())
                log10_of__angle_score_max .append(log10_of__raw_angle_score.max ())
                log10_of__angle_score_avg .append(log10_of__raw_angle_score.mean())
                corrected_angle_score_min .append(raw_corrected_angle_score.min ())
                corrected_angle_score_max .append(raw_corrected_angle_score.max ())
                corrected_angle_score_avg .append(raw_corrected_angle_score.mean())
                pass#for inner param set
            print(f"log10_of__angle_score_min = {str_the_list(log10_of__angle_score_min, 3)}")
            print(f"log10_of__angle_score_max = {str_the_list(log10_of__angle_score_max, 3)}")
            print(f"log10_of__angle_score_avg = {str_the_list(log10_of__angle_score_avg, 3)}")
            print(f"corrected_angle_score_min = {str_the_list(corrected_angle_score_min, 3)}")
            print(f"corrected_angle_score_max = {str_the_list(corrected_angle_score_max, 3)}")
            print(f"corrected_angle_score_avg = {str_the_list(corrected_angle_score_avg, 3)}")
            print(f"dim_list                = {str_the_list(dim_list, 1)}")
            
            pass#/test
        
        if "randn" and False:
            print("randn")
            assert False, "new len score is 0.5* old score. result is old score."
            #the angle part is repeating. But anyway.
            # randn(size=[dim,dim])/sqrt(dim)
            #result
            # len_score_min   = [-2.438, -0.295, -0.027, -0.002]
            # len_score_max   = [ 0.705,  0.204,  0.019,  0.001]
            # len_score_avg   = [-0.250, -0.046, -0.004, -0.000]
            # angle_score_min = [ 0.020,  1.129,  1.550,  1.591]
            # angle_score_max = [ 2.530,  2.348,  1.650,  1.599]
            # angle_score_avg = [ 1.616,  1.605,  1.596,  1.596]
            # dim_list         = [ 2.00,  10.00,  100.00,  1000.]
            
            # angle_score_avg = [ 1.801,  1.636,  1.601,  1.596]# if the correction offset if 0.
            # angle_score_avg = [ 1.272,  1.548,  1.592,  1.595]# if the correction offset if -1.
            
            #throritically, rotating and permutating a matrix should not change the result too much ?
            
            len_score_min = []#don't modify this
            len_score_max = []#don't modify this
            len_score_avg = []#don't modify this
            angle_score_min = []#don't modify this
            angle_score_max = []#don't modify this
            angle_score_avg = []#don't modify this
            #------------------#------------------#------------------
            dim_list =       [2,       10,  100, 1000]
            test_time_list = [10000, 3000, 1000,  200]
            #test_time_list = [1000, 300, 100,  20]
            for inner_param_set in range(dim_list.__len__()):
                dim = dim_list[inner_param_set]
                test_time = test_time_list[inner_param_set]
                print(test_time)
            #------------------#------------------#------------------
            
                raw_len_score = torch.empty(size=[test_time])
                raw_angle_score = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #------------------#------------------#------------------
                    mat = torch.randn(size=[dim,dim])/math.sqrt(dim)
                    _, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)#,_correct_offset_for_angle_score = -1.)
                    assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                    len_score = _log[0][1]
                    #------------------#------------------#------------------
                    
                    raw_len_score[test_count] = len_score
                    raw_angle_score[test_count] = angle_score
                    pass
                len_score_min   .append(raw_len_score.min ())
                len_score_max   .append(raw_len_score.max ())
                len_score_avg   .append(raw_len_score.mean())
                angle_score_min .append(raw_angle_score.min ())
                angle_score_max .append(raw_angle_score.max ())
                angle_score_avg .append(raw_angle_score.mean())
                
                pass#for inner param set
            print(f"len_score_min   = {str_the_list(len_score_min  , 3)}")
            print(f"len_score_max   = {str_the_list(len_score_max  , 3)}")
            print(f"len_score_avg   = {str_the_list(len_score_avg  , 3)}")
            print(f"angle_score_min = {str_the_list(angle_score_min, 3)}")
            print(f"angle_score_max = {str_the_list(angle_score_max, 3)}")
            print(f"angle_score_avg = {str_the_list(angle_score_avg, 3)}")
            print(f"dim_list      = {str_the_list(dim_list, 2)}")
            
            pass#/test
        
        if "rand*2-1" and False:
            #the angle part is very weird. It looks the correction in that function is wrong?
            assert False, "new len score is 0.5* old score. result is old score."
            # randn(size=[dim,dim])/sqrt(dim)
            #result
            # len_score_min               = [-2.519, -0.653, -0.492, -0.478]
            # len_score_max               = [-0.075, -0.367, -0.464, -0.476]
            # len_score_avg               = [-0.621, -0.496, -0.479, -0.477]
            # angle_score_min             = [ 0.008,  1.137,  1.546,  1.592]
            # angle_score_max             = [ 2.530,  2.303,  1.658,  1.601]
            # angle_score_avg             = [ 1.611,  1.600,  1.595,  1.596]
            # dim_list                  = [ 2.00,  10.00,  100.00,  1000.00]
                        
            # angle_score_avg             = [ 1.796,  1.627,  1.599,  1.596]# if the correction offset if 0.
            # angle_score_avg             = [ 1.279,  1.547,  1.592,  1.595]# if the correction offset if -1.
            
            #throritically, rotating and permutating a matrix should not change the result too much ?
            print("rand*2-1")
            len_score_min = []#don't modify this
            len_score_max = []#don't modify this
            len_score_avg = []#don't modify this
            angle_score_min = []#don't modify this
            angle_score_max = []#don't modify this
            angle_score_avg = []#don't modify this
            #------------------#------------------#------------------
            dim_list =       [2,      10,  100, 1000]
            test_time_list = [5000, 3000, 1000,  200]
            test_time_list = [500, 300, 100,  20]
            for inner_param_set in range(dim_list.__len__()):
                dim = dim_list[inner_param_set]
                test_time = test_time_list[inner_param_set]
                print(test_time)
            #------------------#------------------#------------------
            
                raw_len_score = torch.empty(size=[test_time])
                raw_angle_score = torch.empty(size=[test_time])
                raw_angle_score_uncorrected = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #------------------#------------------#------------------
                    mat = (torch.rand (size=[dim,dim])*2.-1.)/math.sqrt(dim)
                    _, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)#,_correct_offset_for_angle_score = 0.)
                    assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                    len_score = _log[0][1]
                    #------------------#------------------#------------------
                    
                    raw_len_score[test_count] = len_score
                    raw_angle_score[test_count] = angle_score
                    pass
                len_score_min   .append(raw_len_score.min ())
                len_score_max   .append(raw_len_score.max ())
                len_score_avg   .append(raw_len_score.mean())
                angle_score_min .append(raw_angle_score.min ())
                angle_score_max .append(raw_angle_score.max ())
                angle_score_avg .append(raw_angle_score.mean())
                
                pass#for inner param set
            print(f"len_score_min               = {str_the_list(len_score_min  , 3)}")
            print(f"len_score_max               = {str_the_list(len_score_max  , 3)}")
            print(f"len_score_avg               = {str_the_list(len_score_avg  , 3)}")
            print(f"angle_score_min             = {str_the_list(angle_score_min, 3)}")
            print(f"angle_score_max             = {str_the_list(angle_score_max, 3)}")
            print(f"angle_score_avg             = {str_the_list(angle_score_avg, 3)}")
            print(f"dim_list                  = {str_the_list(dim_list, 2)}")
            
            pass#/test
        
        if "rand_sign as an extreme case for rand*2-1" and False:
            # rand_sign(size=[dim,dim])/sqrt(dim)
            assert False, "new len score is 0.5* old score. result is old score."
            #result
            # len_score_min               = [-0.000,  0.000,  0.000,  0.000]
            # len_score_max               = [-0.000,  0.000,  0.000,  0.000]
            # len_score_avg               = [-0.000,  0.000,  0.000,  0.000]
            # angle_score_min             = [ 0.000,  0.991,  1.541,  1.591]
            # angle_score_max             = [ 2.530,  2.424,  1.641,  1.600]
            # angle_score_avg             = [ 1.253,  1.531,  1.589,  1.595]
            # dim_list                  = [ 2.00,  10.00,  100.00,  1000.00]
            
            # angle_score_avg             = [ 1.419,  1.563,  1.593,  1.595]# if the correction offset if 0.
            # angle_score_avg             = [ 0.989,  1.478,  1.584,  1.595]# if the correction offset if -1.
            
            #throritically, rotating and permutating a matrix should not change the result too much ?
            print("rand_sign as an extreme case for rand*2-1")
            from pytorch_yagaodirac_v2.Random import rand_sign
            len_score_min = []#don't modify this
            len_score_max = []#don't modify this
            len_score_avg = []#don't modify this
            angle_score_min = []#don't modify this
            angle_score_max = []#don't modify this
            angle_score_avg = []#don't modify this
            #------------------#------------------#------------------
            dim_list =       [2,      10,  100, 1000]
            test_time_list = [5000, 3000, 1000,  200]
            #test_time_list = [500, 300, 100,  20]
            for inner_param_set in range(dim_list.__len__()):
                dim = dim_list[inner_param_set]
                test_time = test_time_list[inner_param_set]
                print(test_time)
            #------------------#------------------#------------------
            
                raw_len_score = torch.empty(size=[test_time])
                raw_angle_score = torch.empty(size=[test_time])
                raw_angle_score_uncorrected = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #------------------#------------------#------------------
                    mat = rand_sign(size=[dim,dim])/math.sqrt(dim)
                    assert not mat.eq(0.).any()
                    _, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)#,_correct_offset_for_angle_score = -1.)
                    assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                    len_score = _log[0][1]
                    #------------------#------------------#------------------
                    
                    raw_len_score[test_count] = len_score
                    raw_angle_score[test_count] = angle_score
                    pass
                len_score_min   .append(raw_len_score.min ())
                len_score_max   .append(raw_len_score.max ())
                len_score_avg   .append(raw_len_score.mean())
                angle_score_min .append(raw_angle_score.min ())
                angle_score_max .append(raw_angle_score.max ())
                angle_score_avg .append(raw_angle_score.mean())
                
                pass#for inner param set
            print(f"len_score_min               = {str_the_list(len_score_min  , 3)}")
            print(f"len_score_max               = {str_the_list(len_score_max  , 3)}")
            print(f"len_score_avg               = {str_the_list(len_score_avg  , 3)}")
            print(f"angle_score_min             = {str_the_list(angle_score_min, 3)}")
            print(f"angle_score_max             = {str_the_list(angle_score_max, 3)}")
            print(f"angle_score_avg             = {str_the_list(angle_score_avg, 3)}")
            print(f"dim_list                  = {str_the_list(dim_list, 2)}")
            
            pass#/test
        
        if "permutation doesn't affect the result." and False:
            if "some trivial test" and False:
                dim = 3
                mat = torch.randn(size=[dim,dim])/math.sqrt(dim)
                mat = randomly_rotate__matrix(mat)
                mat = randomly_permutate__matrix(mat)
                before__len_score, before__angle_score, before___log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
                
                #mat = randomly_rotate__matrix(mat)#no... this is tested later.
                mat = randomly_permutate__matrix(mat)*10.
                after__len_score, after__angle_score, after___log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)

                #assert _tensor_equal(before__len_score, after__len_score)# no.
                assert _tensor_equal(before__angle_score, after__angle_score)
                
                for ii in range(before___log.__len__()):
                    name = before___log[ii][0]
                    before_entry = before___log[ii][1]
                    after_entry = after___log[ii][1]
                    #print(f"{name} {before_entry} {after_entry}")
                    #assert _tensor_equal(before_entry.flatten().sort().values, after_entry.flatten().sort().values)
                    if name in [
                        "sum of two len_score__raw_mean_without_abs" # 0.1400 2.1400
                    ]:
                        assert _tensor_equal(before_entry+2., after_entry)# new version is 0.5*old verion?????
                        pass
                    if name in [
                        "sum of two raw angle_score" # 0.8766 0.8766
                        "hor_angle_score" #            0.4468 0.4468
                        "ver_angle_score" #            0.4297 0.4297
                    ]:
                        assert _tensor_equal(before_entry, after_entry)
                        pass
                    if name in [
                        "hor_sum__as_len_sqr___1_is_good__dim" # ([2.6050, 0.6573, 1.3940]) tensor([ 65.7252, 260.5022, 139.3987])
                        "ver_sum__as_len_sqr___1_is_good__dim" # ([0.8186, 2.2877, 1.5500]) tensor([155.0018,  81.8580, 228.7663])
                    ]:
                        assert _tensor_equal(before_entry.flatten().sort().values*100., after_entry.flatten().sort().values)# new version is 0.5*old verion?????
                        pass
                    if name in [
                        "hor_sum__as_len_log10___0_is_good__dim" #([ 0.2079, -0.0911,  0.0721]) tensor([0.9089, 1.2079, 1.0721])
                        "hor_len_score__raw" #              tensor([ 0.2079, -0.0911,  0.0721]) tensor([0.9089, 1.2079, 1.0721])
                        "hor_len_score__raw_mean_without_abs" #0.0629 1.0629
                        "ver_sum__as_len_log10___0_is_good__dim" #([-0.0435,  0.1797,  0.0952]) tensor([1.0952, 0.9565, 1.1797])
                        "ver_len_score__raw" #              tensor([-0.0435,  0.1797,  0.0952]) tensor([1.0952, 0.9565, 1.1797])
                        "ver_len_score__raw_mean_without_abs" #0.0771 1.0771
                    ]:
                        assert _tensor_equal(before_entry.flatten().sort().values+1., after_entry.flatten().sort().values)# new version is 0.5*old verion?????
                    if name in [
                        "matrix_into_hor_len_1 before nan_to_num"
                        # tensor([[-0.1599,  0.7765, -0.6095],
                        #         [ 0.6166,  0.0616,  0.7849],
                        #         [-0.6001, -0.7159,  0.3568]]) 
                        # tensor([[ 0.7849,  0.6166,  0.0616],
                        #         [-0.6095, -0.1599,  0.7765],
                        #         [ 0.3568, -0.6001, -0.7159]])
                        "matrix_into_ver_len_1 before nan_to_num"
                        # tensor([[-0.2853,  0.8286, -0.7901],
                        #         [ 0.5525,  0.0330,  0.5111],
                        #         [-0.7832, -0.5588,  0.3384]]) 
                        # tensor([[ 0.5111,  0.5525,  0.0330],
                        #         [-0.7901, -0.2853,  0.8286],
                        #         [ 0.3384, -0.7832, -0.5588]])
                        ]:
                        import random
                        _some_random_number = random.random()
                        assert _tensor_equal(  before_entry.nan_to_num(_some_random_number).flatten().sort().values, 
                                                after_entry.nan_to_num(_some_random_number).flatten().sort().values)
                    if name in [
                        "hor_vec_angle_test"
                        # tensor([[ 0.0000, -0.5291, -0.6774],
                        #         [-0.5291,  0.0000, -0.1341],
                        #         [-0.6774, -0.1341,  0.0000]])
                        # tensor([[ 0.0000, -0.5291, -0.1341],
                        #         [-0.5291,  0.0000, -0.6774],
                        #         [-0.1341, -0.6774,  0.0000]])
                        "ver_vec_angle_test"
                        # tensor([[ 0.0000,  0.2195,  0.2428],
                        #         [ 0.2195,  0.0000, -0.8269],
                        #         [ 0.2428, -0.8269,  0.0000]]) 
                        # tensor([[ 0.0000,  0.2428, -0.8269],
                        #         [ 0.2428,  0.0000,  0.2195],
                        #         [-0.8269,  0.2195,  0.0000]])
                        ]:
                        import random
                        _some_random_number = random.random()
                        assert _tensor_equal(  before_entry.flatten().sort().values, 
                                                after_entry.flatten().sort().values)
                        pass
                    if name in [
                        "hor_len_score" # 0.12372 1.06296 # this is some .abs().sum. Dont check it here.
                        "ver_len_score" # 0.10611 1.07713 # this is some .abs().sum. Dont check it here.
                        ]:
                        #nothing here. This is not checked for some reason.
                        pass
                    pass
            
            #it's a only assertion test. No print.
            #throritically, rotating and permutating a matrix should not change the result too much ?
            #------------------#------------------#------------------
            dim_list =       [2,   10,   30, 100, 1000]
            test_time_list = [100, 500, 200, 100,   10]
            #test_time_list = [10, 50, 20, 10,   2]
            for inner_param_set in range(dim_list.__len__()):
                dim = dim_list[inner_param_set]
                test_time = test_time_list[inner_param_set]
                print(test_time)
            #------------------#------------------#------------------
                for test_count in range(test_time):
                    mat = torch.randn(size=[dim,dim])/math.sqrt(dim)
                    mat = randomly_rotate__matrix(mat)
                    mat = randomly_permutate__matrix(mat)
                    before__len_score, before__angle_score, before___log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
                    
                    #mat = randomly_rotate__matrix(mat)#no... this is tested later.
                    mat = randomly_permutate__matrix(mat)
                    after__len_score, after__angle_score, after___log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)

                    assert _tensor_equal(before__len_score, after__len_score)
                    assert _tensor_equal(before__angle_score, after__angle_score)
                    
                    #for ii in range(before___log.__len__()):
                    for ii in [0, 1]:
                        before_entry = before___log[ii][1]
                        after_entry = after___log[ii][1]
                        assert _tensor_equal(before_entry.flatten().sort().values, after_entry.flatten().sort().values)
                        pass
                    
                    mat = mat*10
                    after__len_score, after__angle_score, after___log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
                    
                    #assert _tensor_equal(before__len_score, after__len_score)# no. wrong.
                    assert _tensor_equal(before__angle_score, after__angle_score)
                    
                    for ii in range(before___log.__len__()):
                        name = before___log[ii][0]
                        before_entry = before___log[ii][1]
                        after_entry = after___log[ii][1]
                        #print(f"{name} {before_entry} {after_entry}")
                        #assert _tensor_equal(before_entry.flatten().sort().values, after_entry.flatten().sort().values)
                        if name in [
                            "sum of two len_score__raw_mean_without_abs" # 0.1400 2.1400
                        ]:
                            assert _tensor_equal(before_entry+2., after_entry)
                            pass
                        if name in [
                            "sum of two raw angle_score" # 0.8766 0.8766
                            "hor_angle_score" #            0.4468 0.4468
                            "ver_angle_score" #            0.4297 0.4297
                        ]:
                            assert _tensor_equal(before_entry, after_entry)
                            pass
                        if name in [
                            "hor_sum__as_len_sqr___1_is_good__dim" # ([2.6050, 0.6573, 1.3940]) tensor([ 65.7252, 260.5022, 139.3987])
                            "ver_sum__as_len_sqr___1_is_good__dim" # ([0.8186, 2.2877, 1.5500]) tensor([155.0018,  81.8580, 228.7663])
                        ]:
                            assert _tensor_equal(before_entry.flatten().sort().values*100., after_entry.flatten().sort().values)
                            pass
                        if name in [
                            "hor_sum__as_len_log10___0_is_good__dim" #([ 0.2079, -0.0911,  0.0721]) tensor([0.9089, 1.2079, 1.0721])
                            "hor_len_score__raw" #              tensor([ 0.2079, -0.0911,  0.0721]) tensor([0.9089, 1.2079, 1.0721])
                            "hor_len_score__raw_mean_without_abs" #0.0629 1.0629
                            "ver_sum__as_len_log10___0_is_good__dim" #([-0.0435,  0.1797,  0.0952]) tensor([1.0952, 0.9565, 1.1797])
                            "ver_len_score__raw" #              tensor([-0.0435,  0.1797,  0.0952]) tensor([1.0952, 0.9565, 1.1797])
                            "ver_len_score__raw_mean_without_abs" #0.0771 1.0771
                        ]:
                            assert _tensor_equal(before_entry.flatten().sort().values+1., after_entry.flatten().sort().values)
                        if name in [
                            "matrix_into_hor_len_1 before nan_to_num"
                            # tensor([[-0.1599,  0.7765, -0.6095],
                            #         [ 0.6166,  0.0616,  0.7849],
                            #         [-0.6001, -0.7159,  0.3568]]) 
                            # tensor([[ 0.7849,  0.6166,  0.0616],
                            #         [-0.6095, -0.1599,  0.7765],
                            #         [ 0.3568, -0.6001, -0.7159]])
                            "matrix_into_ver_len_1 before nan_to_num"
                            # tensor([[-0.2853,  0.8286, -0.7901],
                            #         [ 0.5525,  0.0330,  0.5111],
                            #         [-0.7832, -0.5588,  0.3384]]) 
                            # tensor([[ 0.5111,  0.5525,  0.0330],
                            #         [-0.7901, -0.2853,  0.8286],
                            #         [ 0.3384, -0.7832, -0.5588]])
                            ]:
                            import random
                            _some_random_number = random.random()
                            assert _tensor_equal(  before_entry.nan_to_num(_some_random_number).flatten().sort().values, 
                                                    after_entry.nan_to_num(_some_random_number).flatten().sort().values)
                        if name in [
                            "hor_vec_angle_test"
                            # tensor([[ 0.0000, -0.5291, -0.6774],
                            #         [-0.5291,  0.0000, -0.1341],
                            #         [-0.6774, -0.1341,  0.0000]])
                            # tensor([[ 0.0000, -0.5291, -0.1341],
                            #         [-0.5291,  0.0000, -0.6774],
                            #         [-0.1341, -0.6774,  0.0000]])
                            "ver_vec_angle_test"
                            # tensor([[ 0.0000,  0.2195,  0.2428],
                            #         [ 0.2195,  0.0000, -0.8269],
                            #         [ 0.2428, -0.8269,  0.0000]]) 
                            # tensor([[ 0.0000,  0.2428, -0.8269],
                            #         [ 0.2428,  0.0000,  0.2195],
                            #         [-0.8269,  0.2195,  0.0000]])
                            ]:
                            import random
                            _some_random_number = random.random()
                            assert _tensor_equal(  before_entry.flatten().sort().values, 
                                                    after_entry.flatten().sort().values)
                            pass
                        if name in [
                            "hor_len_score" # 0.12372 1.06296 # this is some .abs().sum. Dont check it here.
                            "ver_len_score" # 0.10611 1.07713 # this is some .abs().sum. Dont check it here.
                            ]:
                            #nothing here. This is not checked for some reason.
                            pass
                    
                    
                    pass
                pass#for inner param set
            pass#/test
        
        if "how much does the randomly rotation affect the score." and False:
            # randn(size=[dim,dim])/math.sqrt(dim) * 1000
            assert False, "# new version is 0.5*old verion, here is old version"
            
            #result is the abs(diff)
            #theoritically, rotating a matrix should not change the result too much ?
            # diff_of__len_score_min   = [ 0.00000,  0.00000,  0.00000,  0.00000]
            # diff_of__len_score_max   = [ 1.26512,  0.06083,  0.00122,  0.00003]
            # diff_of__len_score_avg   = [ 0.09150,  0.01082,  0.00031,  0.00001]
            # diff_of__angle_score_min = [ 0.00000,  0.00002,  0.00004,  0.00005]
            # diff_of__angle_score_max = [ 2.36598,  0.42716,  0.01646,  0.00098]
            # diff_of__angle_score_avg = [ 0.31417,  0.06749,  0.00462,  0.00036]
            # dim_list               = [ 2.0000,  10.0000,  100.0000,  1000.0000]#this is scaled score, and it also goes into 0 when dim incr.
            # dim_list                = [  2.0000,   10.0000,  100.0000,  1000.]
            # if dim is big enough, the error is small enough.
            
            diff_of__len_score_min = []#don't modify this
            diff_of__len_score_max = []#don't modify this
            diff_of__len_score_avg = []#don't modify this
            diff_of__angle_score_min = []#don't modify this
            diff_of__angle_score_max = []#don't modify this
            diff_of__angle_score_avg = []#don't modify this
            #------------------#------------------#------------------
            dim_list =       [2,       10,  100, 1000]
            test_time_list = [10000,10000,  300,  20]
            for inner_param_set in range(dim_list.__len__()):
                dim = dim_list[inner_param_set]
                test_time = test_time_list[inner_param_set]
                print(test_time)
            #------------------#------------------#------------------
            
                raw_len_score = torch.empty(size=[test_time])
                raw_angle_score = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #------------------#------------------#------------------
                    mat = torch.randn(size=[dim,dim])/math.sqrt(dim)
                    _, before__angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
                    assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                    before__len_score = _log[0][1]
                    
                    #this assertion is specially for this test. They must be >0 before the abs().
                    # assert all_the_other_results[0][1].ge(0.).all()
                    # assert all_the_other_results[1][1].ge(0.).all()
                    
                    mat = randomly_rotate__matrix(mat)
                    _, after__angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
                    assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                    after__len_score = _log[0][1]
                    #------------------#------------------#------------------
                    
                    raw_len_score[test_count] = (before__len_score-after__len_score).abs()
                    raw_angle_score[test_count] = (before__angle_score-after__angle_score).abs()
                    pass
                diff_of__len_score_min   .append(raw_len_score.min ())
                diff_of__len_score_max   .append(raw_len_score.max ())
                diff_of__len_score_avg   .append(raw_len_score.mean())
                diff_of__angle_score_min .append(raw_angle_score.min ())
                diff_of__angle_score_max .append(raw_angle_score.max ())
                diff_of__angle_score_avg .append(raw_angle_score.mean())
                
                pass#for inner param set
            print(f"diff_of__len_score_min   = {str_the_list(diff_of__len_score_min  , 5)}")
            print(f"diff_of__len_score_max   = {str_the_list(diff_of__len_score_max  , 5)}")
            print(f"diff_of__len_score_avg   = {str_the_list(diff_of__len_score_avg  , 5)}")
            print(f"diff_of__angle_score_min = {str_the_list(diff_of__angle_score_min, 5)}")
            print(f"diff_of__angle_score_max = {str_the_list(diff_of__angle_score_max, 5)}")
            print(f"diff_of__angle_score_avg = {str_the_list(diff_of__angle_score_avg, 5)}")
            print(f"dim_list               = {str_the_list(dim_list, 4)}")
            
            pass#/test
        
        if "a perfect blends with a randn" and False:
            # nothing special. The result looks normal. It's a weak evidence for the tool itself.
            # rand_sign(size=[dim,dim])/sqrt(dim)
            #result
            assert False, "# new version is 0.5*old verion, here is old version"
            
            # if dim == 2:
            # len_score_min       = [-0.000, -0.323, -0.823, -1.597, -1.643, -1.735, -2.038, -1.484, -2.103, -1.523, -1.569]
            # len_score_max       = [ 0.000,  0.059,  0.204,  0.240,  0.324,  0.459,  0.537,  0.795,  0.715,  0.799,  0.833]
            # len_score_avg       = [-0.000, -0.090, -0.205, -0.294, -0.389, -0.376, -0.328, -0.212, -0.130, -0.034,  0.055]
            # angle_score_min     = [ 0.000,  0.009,  0.006,  0.037,  0.008,  0.083,  0.092,  0.040,  0.017,  0.046,  0.039]
            # angle_score_max     = [ 0.000,  1.229,  2.210,  2.523,  2.530,  2.530,  2.530,  2.530,  2.530,  2.530,  2.530]
            # angle_score_avg     = [ 0.000,  0.320,  0.698,  1.142,  1.491,  1.542,  1.635,  1.586,  1.644,  1.654,  1.637]
            # randn_factor_list   = [ 0.000,  0.100,  0.200,  0.300,  0.400,  0.500,  0.600,  0.700,  0.800,  0.900,  1.000]
            #                                 ^^^^^
            # pass
            # if dim == 10:
            # len_score_min      = [-0.000, -0.067, -0.134, -0.158, -0.088,  0.079,  0.201,  0.345,  0.437,  0.597,  0.690,  0.770]
            # len_score_max      = [ 0.000,  0.008,  0.039,  0.126,  0.258,  0.434,  0.543,  0.679,  0.834,  0.919,  1.040,  1.120]
            # len_score_avg      = [-0.000, -0.035, -0.051, -0.008,  0.101,  0.246,  0.389,  0.530,  0.653,  0.766,  0.868,  0.957]
            # angle_score_min    = [ 0.000,  0.266,  0.532,  0.973,  1.116,  1.241,  1.276,  1.239,  1.220,  1.185,  1.257,  1.290]
            # angle_score_max    = [ 0.000,  0.538,  0.968,  1.648,  1.928,  2.144,  2.206,  2.231,  2.110,  2.027,  2.027,  2.051]
            # angle_score_avg    = [ 0.000,  0.365,  0.719,  1.271,  1.515,  1.586,  1.606,  1.599,  1.581,  1.607,  1.614,  1.613]
            # randn_factor_list  = [ 0.000,  0.050,  0.100,  0.200,  0.300,  0.400,  0.500,  0.600,  0.700,  0.800,  0.900,  1.000]
            #                                        ^^^^^
            # pass
            # if dim == 100:
            # len_score_min     = [-0.000, -0.006,  0.025,  0.082,  0.159,  0.240,  0.646,  0.956,  1.382,  1.676,  1.981]
            # len_score_max     = [ 0.000,  0.004,  0.042,  0.100,  0.180,  0.271,  0.675,  0.989,  1.413,  1.702,  2.014]
            # len_score_avg     = [-0.000, -0.000,  0.033,  0.092,  0.169,  0.255,  0.662,  0.973,  1.399,  1.687,  1.996]
            # angle_score_min   = [ 0.000,  0.435,  0.809,  1.100,  1.288,  1.401,  1.542,  1.542,  1.567,  1.557,  1.557]
            # angle_score_max   = [ 0.000,  0.454,  0.859,  1.151,  1.345,  1.465,  1.626,  1.634,  1.637,  1.640,  1.635]
            # angle_score_avg   = [ 0.000,  0.445,  0.835,  1.123,  1.313,  1.427,  1.582,  1.594,  1.599,  1.596,  1.596]
            # randn_factor_list = [ 0.000,  0.020,  0.040,  0.060,  0.080,  0.100,  0.200,  0.300,  0.500,  0.700,  1.000]
            #                                                               ^^^^^
            # pass
            # if dim == 1000:
            # len_score_min     = [-0.000,  0.006,  0.033,  0.076,  0.133,  0.264,  0.401,  0.530,  0.650,  0.860,  1.033,  1.608,  2.398,  2.999]
            # len_score_max     = [-0.000,  0.007,  0.034,  0.078,  0.134,  0.265,  0.402,  0.532,  0.652,  0.861,  1.035,  1.609,  2.399,  3.000]
            # len_score_avg     = [-0.000,  0.006,  0.033,  0.077,  0.133,  0.265,  0.401,  0.531,  0.651,  0.860,  1.034,  1.608,  2.398,  3.000]
            # angle_score_min   = [ 0.000,  0.352,  0.669,  0.930,  1.129,  1.369,  1.484,  1.537,  1.561,  1.580,  1.588,  1.593,  1.594,  1.592]
            # angle_score_max   = [ 0.000,  0.352,  0.671,  0.932,  1.133,  1.373,  1.487,  1.540,  1.568,  1.587,  1.592,  1.599,  1.598,  1.598]
            # angle_score_avg   = [ 0.000,  0.352,  0.670,  0.931,  1.130,  1.372,  1.485,  1.539,  1.564,  1.584,  1.590,  1.596,  1.596,  1.595]
            # randn_factor_list = [ 0.000,  0.005,  0.010,  0.015,  0.020,  0.030,  0.040,  0.050,  0.060,  0.080,  0.100,  0.200,  0.500,  1.000]
            #                                                                                                       ^^^^^
            # pass
            
            
            #throritically, rotating and permutating a matrix should not change the result too much ?
            print("a perfect blends with a randn")
            
            #------------------#------------------#------------------
            dim_list =       [2,    10, 100, 1000]
            test_time_list = [500, 300, 100,  5]
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                test_time = test_time_list[outter_param_set]
                
                if dim <1000:
                    continue
                
                print(test_time)
            #------------------#------------------#------------------
                len_score_min = []#don't modify this
                len_score_max = []#don't modify this
                len_score_avg = []#don't modify this
                angle_score_min = []#don't modify this
                angle_score_max = []#don't modify this
                angle_score_avg = []#don't modify this
                
                #------------------#------------------#------------------
                if dim ==2 :
                    randn_factor_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
                    pass
                elif dim == 10:
                    randn_factor_list = [0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
                    pass
                elif dim == 100:
                    randn_factor_list = [0., 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.7, 1.]
                    pass
                elif dim == 1000:
                    randn_factor_list = [0., 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5, 1.]
                    pass
                    
                for inner_param_set in range(randn_factor_list.__len__()):
                    randn_factor = randn_factor_list[inner_param_set]
                #------------------#------------------#------------------
                
                    raw_len_score = torch.empty(size=[test_time])
                    raw_angle_score = torch.empty(size=[test_time])
                    raw_angle_score_uncorrected = torch.empty(size=[test_time])
                    for test_count in range(test_time):
                        #------------------#------------------#------------------
                        perfect_mat = torch.eye(n=dim)
                        perfect_mat = randomly_rotate__matrix(perfect_mat)
                        perfect_mat = randomly_permutate__matrix(perfect_mat)
                        
                        randn_mat = torch.randn(size=[dim,dim])
                        
                        test_mat = perfect_mat*(1.-randn_factor)+randn_mat*randn_factor
                        
                        _, angle_score, _log = LOSS__mat_is_standard_orthogonal(test_mat, _debug__needs_log = True)#,_correct_offset_for_angle_score = -1.)
                        assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                        len_score = _log[0][1]
                        #------------------#------------------#------------------
                        
                        raw_len_score[test_count] = len_score
                        raw_angle_score[test_count] = angle_score
                        pass
                    len_score_min   .append(raw_len_score.min ())
                    len_score_max   .append(raw_len_score.max ())
                    len_score_avg   .append(raw_len_score.mean())
                    angle_score_min .append(raw_angle_score.min ())
                    angle_score_max .append(raw_angle_score.max ())
                    angle_score_avg .append(raw_angle_score.mean())
                    
                    pass#for inner param set
                print(f"if dim == {dim}:")
                print(f"len_score_min   = {str_the_list(len_score_min  , 3)}")
                print(f"len_score_max   = {str_the_list(len_score_max  , 3)}")
                print(f"len_score_avg   = {str_the_list(len_score_avg  , 3)}")
                print(f"angle_score_min = {str_the_list(angle_score_min, 3)}")
                print(f"angle_score_max = {str_the_list(angle_score_max, 3)}")
                print(f"angle_score_avg = {str_the_list(angle_score_avg, 3)}")
                print(f"randn_factor_list = {str_the_list(randn_factor_list, 3)}")
                print("pass")
                
                pass#for outter param set
            
            pass#/test
        
        return 
    
    #____test____measure_the_random_init____LOSS__the_mat_is_standard_orthogonal()
    
    
    
    def ____test____device_adaption____LOSS__the_mat_is_standard_orthogonal():
        mat = torch.randn(size=[2,2], device='cuda')
        
        len_score, angle_score, _log = LOSS__mat_is_standard_orthogonal(mat, _debug__needs_log = True)
        assert len_score.device.type == 'cuda'
        assert angle_score.device.type == 'cuda'
        for entry in _log:
            assert entry[1].device.type == 'cuda'
            pass
        
        return 
    #____test____device_adaption____LOSS__the_mat_is_standard_orthogonal()
    pass




if "angle_similarity but the assuption is a bit wrong." and False:
    #simply use other measurements
        
    def ____xxxx____LOSS__angle_similarity(input:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        '''return score
        
        The result is always >=0. The smaller the better.
        
        The score between 2 random matrix is 1.85 to 2.3, for any dimention. Tests below.
        
        If any length of any row or column is too small, it's not measured, 
        and it's a fake good score for that detail.
        '''
        assert is_square_matrix(input)
        assert is_square_matrix(target)
        with torch.no_grad():
            
            norm_ed_by_row_mat_1 = vector_length_norm(input)
            norm_ed_by_row_mat_2 = vector_length_norm(target)
            ___temp = norm_ed_by_row_mat_1*norm_ed_by_row_mat_2
            ___temp = ___temp.sum(dim=1)
            ___temp = ___temp.mean()
            score_part_1 = 1.-___temp
            
            norm_ed_by_column_mat_1 = vector_length_norm(input.T)
            norm_ed_by_column_mat_2 = vector_length_norm(target.T)
            ___temp = norm_ed_by_column_mat_1*norm_ed_by_column_mat_2
            ___temp = ___temp.sum(dim=1)
            ___temp = ___temp.mean()
            score_part_2 = 1.-___temp
            
            return (score_part_1+score_part_2).mean()
        pass#/function
    if "test" and __DEBUG_ME__() and False:
        def ____test____LOSS__angle_similarity():
            "The score between 2 random matrix is 1.85 to 2.3, for any dimention. Tests below."
            for dim in [10,100,1000,10000]:
                mat1 = torch.randn(size=[dim,dim])
                mat2 = torch.randn(size=[dim,dim])
                score = ____xxxx____LOSS__angle_similarity(mat1,mat2)
                print(dim, score)
                pass
            
            import math, random
            for _ in range(166):
                dim = random.randint(2,300)
                mat1 = torch.randn(size=[dim,dim])
                mat2 = mat1.detach().clone().mul(1.3)
                score = ____xxxx____LOSS__angle_similarity(mat1,mat2)
                assert _tensor_equal(score, [0.])
                pass
            
            mat0 = torch.tensor([[1.,0],[0,1]])
            mat1 = torch.tensor([[1.,0.1],[0,1]])
            mat2 = torch.tensor([[1.,0.2],[0,1]])
            score_small = ____xxxx____LOSS__angle_similarity(mat0,mat1)
            score_large = ____xxxx____LOSS__angle_similarity(mat0,mat2)
            assert score_small<score_large
            
            for _ in range(116):
                rand = random.random()+0.001# >0.
                dim = random.randint(2,300)
                mat0 = torch.eye(n=dim)
                mat0[0,1] = rand
                mat1 = torch.eye(n=dim)
                mat1[0,1] = rand*1.1
                mat2 = torch.eye(n=dim)
                mat2[0,1] = rand*1.2
                score_small = ____xxxx____LOSS__angle_similarity(mat0,mat1)
                score_large = ____xxxx____LOSS__angle_similarity(mat0,mat2)
                assert score_small<=score_large
                pass
            return
        ____test____LOSS__angle_similarity()
        pass
    
    pass





def full_length_info_test__in_log10(input:torch.Tensor, needs_no_abs_result = True) \
                ->tuple[torch.Tensor, torch.Tensor|None, torch.Tensor, torch.Tensor|None]:
    '''return len_loss, _raw_result__no_abs__len_loss, length_retention_loss, no_abs__length_retention_score  
    
    or 
    
    return len_loss, None, length_retention_loss, None'''

    len_loss, _, _log = LOSS__mat_is_standard_orthogonal(input, _debug__needs_log = needs_no_abs_result)
    len_loss#################################
    
    length_retention_loss, (no_abs__length_retention_score,) = LOSS__vec_len_retention__of_a_mat_in_matmul(\
                                                input, _debug__needs_log = needs_no_abs_result)
    if needs_no_abs_result:
        assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
        _raw_result__no_abs__len_loss = _log[0][1]###############################
        return len_loss, _raw_result__no_abs__len_loss, length_retention_loss, no_abs__length_retention_score
    
    return len_loss, None, length_retention_loss, no_abs__length_retention_score
















