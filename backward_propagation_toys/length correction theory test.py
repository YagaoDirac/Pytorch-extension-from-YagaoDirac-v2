from typing import Literal
import time
from pathlib import Path
import math, random
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal, \
    iota, is_square_matrix, \
    vector_length_norm, get_vector_length, get_full_info_of_vector_length__2d, expand_vec_to_matrix,\
    log10_avg_safe, log10_avg__how_similar, get_mask_of_top_element__rough,\
    str_the_list, str_the_list__probability
        
from pytorch_yagaodirac_v2.Interpolation import \
    interpolation_of_list, interpolation_of_list_2d, reverse_interpolation_of_list__list_must_sorted
    
from pytorch_yagaodirac_v2.ParamMo import GradientModification__mean_len_of_something_to_1
from pytorch_yagaodirac_v2.Random import random_standard_vector, randomly_permutate__matrix, randomly_rotate__matrix
from pytorch_yagaodirac_v2.measure_for_matrix import LOSS__behavior_similarity, \
    LOSS__mat_is_standard_orthogonal, LOSS__vec_len_retention__of_a_mat_in_matmul, full_length_info_test__in_log10

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######

def get_device(dim:int, threshold = 101)->Literal['cpu', 'cuda']:
    if dim>threshold:
        device:Literal['cpu', 'cuda'] = 'cuda'
        pass
    else:
        device = 'cpu'
        pass
    return device





def random_dummy_mat__v2(dim:int, noise_strength:float, div_sqrt_1_plus_ns_sqr:bool,
                    device='cpu', iota_of_dim:torch.Tensor|None = None, 
                    )->torch.Tensor:
    '''docs????
    '''
    if iota_of_dim is None:
        iota_of_dim = iota(dim)
        pass
    
    #<  real job
    mat = torch.eye(n = dim, device=device)
    mat = randomly_rotate__matrix(mat)
    mat = randomly_permutate__matrix(mat)
    
    #<  some noise to mimic the learning update.
    _mul_me = noise_strength/math.sqrt(dim)
    mat += torch.randn_like(mat)*_mul_me
    del _mul_me
    
    if div_sqrt_1_plus_ns_sqr:
        _div_me = math.sqrt(1+noise_strength*noise_strength)
        _mul_me = 1./_div_me
        del _div_me
        mat *= _mul_me
        pass
    return mat
#copied from the angle test part 2.py. Already tested behavior there. No basic behavior test here.

if "measure the random gen algo" and False:
    def ____test____measure_the_random_gen_algo():
        
        # pow(10,0.024) is 1.057
        # pow(10,0.08) is 1.2
        # pow(10,0.15) is 1.41
        # at least, when dim >=100, the noise is small enough.
        if " measure the random gen algo  v2 with length correction." and False:
            print(" measure the random gen algo  v2 with length correction.")
            if True:
                #some other ref
                #                         noise_strength_list = torch.tensor(
                #                             [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.10,    1.30,    1.80,     ])
                #                         angle_loss_list__full = torch.tensor([
                #                             [ 0.0000,  0.2184,  0.4341,  0.6253,  0.7969,  0.9537,  1.0867,  1.1974,  1.3483,  1.4409,  1.5145,  1.5496,   ],#dim 10
                #                             [ 0.0000,  0.2234,  0.4377,  0.6346,  0.8075,  0.9575,  1.0838,  1.1826,  1.3292,  1.4228,  1.4859,  1.5540,   ],#dim 100
                #                             [ 0.0000,  0.2238,  0.4381,  0.6346,  0.8080,  0.9577,  1.0820,  1.1828,  1.3303,  1.4248,  1.4814,  1.5506,   ],])# dim 1000
                
                
                # dim = 10
                # noise_strength_list         = [ 0.0,     0.1,     0.2,     0.3,     0.4,     0.5,     0.6,     0.7,     0.9,     1.1,     1.3,     1.8]
                # len_score__max            = [ 0.0000,  0.0168,  0.0318,  0.0609,  0.0659,  0.0750,  0.0801,  0.0987,  0.1066,  0.1207,  0.1255,  0.1247]
                # len_score__avg            = [ 0.0000,  0.0112,  0.0216,  0.0312,  0.0395,  0.0479,  0.0534,  0.0604,  0.0680,  0.0728,  0.0760,  0.0790]
                # len_score__no_abs__min    = [-0.0000, -0.0122, -0.0261, -0.0355, -0.0577, -0.0681, -0.0581, -0.0642, -0.0980, -0.0966, -0.1010, -0.0973]
                # len_score__no_abs__max    = [ 0.0000,  0.0096,  0.0231,  0.0276,  0.0301,  0.0402,  0.0389,  0.0688,  0.0618,  0.0802,  0.0634,  0.0599]
                # len_score__no_abs__avg    = [-0.0000, -0.0003, -0.0016, -0.0044, -0.0067, -0.0079, -0.0100, -0.0114, -0.0170, -0.0173, -0.0166, -0.0232]

                # len_retention__max        = [ 0.0000,  0.0160,  0.0327,  0.0477,  0.0657,  0.0738,  0.0875,  0.0964,  0.1075,  0.1032,  0.1451,  0.1333]
                # len_retention__avg        = [ 0.0000,  0.0110,  0.0211,  0.0313,  0.0410,  0.0481,  0.0547,  0.0599,  0.0678,  0.0718,  0.0753,  0.0797]
                # len_retention__no_abs__min= [-0.0000, -0.0127, -0.0293, -0.0390, -0.0629, -0.0687, -0.0727, -0.0682, -0.0979, -0.0872, -0.1274, -0.1309]
                # len_retention__no_abs__max= [ 0.0000,  0.0140,  0.0234,  0.0358,  0.0365,  0.0472,  0.0469,  0.0651,  0.0555,  0.0762,  0.0855,  0.0776]
                # len_retention__no_abs__avg= [-0.0000, -0.0002, -0.0016, -0.0045, -0.0076, -0.0089, -0.0109, -0.0121, -0.0170, -0.0168, -0.0145, -0.0226]
                
                # dim = 100
                # noise_strength_list         = [ 0.0,     0.1,     0.2,     0.3,     0.4,     0.5,     0.6,     0.7,     0.9,     1.1,     1.3,     1.8]
                # len_score__max            = [ 0.0000,  0.0039,  0.0077,  0.0111,  0.0143,  0.0173,  0.0196,  0.0206,  0.0235,  0.0250,  0.0263,  0.0284]
                # len_score__avg            = [ 0.0000,  0.0034,  0.0068,  0.0098,  0.0125,  0.0147,  0.0167,  0.0182,  0.0205,  0.0223,  0.0230,  0.0241]
                # len_score__no_abs__min    = [-0.0000, -0.0009, -0.0023, -0.0044, -0.0050, -0.0044, -0.0071, -0.0088, -0.0093, -0.0087, -0.0087, -0.0104]
                # len_score__no_abs__max    = [ 0.0000,  0.0010,  0.0021,  0.0023,  0.0044,  0.0046,  0.0034,  0.0034,  0.0040,  0.0047,  0.0072,  0.0054]
                # len_score__no_abs__avg    = [-0.0000,  0.0000, -0.0002, -0.0004, -0.0008, -0.0006, -0.0010, -0.0012, -0.0017, -0.0016, -0.0014, -0.0024]

                # len_retention__max        = [ 0.0000,  0.0041,  0.0085,  0.0115,  0.0143,  0.0186,  0.0200,  0.0212,  0.0254,  0.0268,  0.0273,  0.0289]
                # len_retention__avg        = [ 0.0000,  0.0034,  0.0067,  0.0098,  0.0126,  0.0146,  0.0167,  0.0182,  0.0205,  0.0221,  0.0225,  0.0238]
                # len_retention__no_abs__min= [-0.0000, -0.0011, -0.0037, -0.0052, -0.0057, -0.0067, -0.0099, -0.0098, -0.0118, -0.0141, -0.0140, -0.0138]
                # len_retention__no_abs__max= [ 0.0000,  0.0013,  0.0025,  0.0029,  0.0062,  0.0051,  0.0061,  0.0057,  0.0077,  0.0103,  0.0077,  0.0095]
                # len_retention__no_abs__avg= [-0.0000,  0.0000, -0.0004, -0.0004, -0.0007, -0.0007, -0.0011, -0.0010, -0.0020, -0.0013, -0.0016, -0.0023]
                
                # dim = 1000
                # noise_strength_list         = [ 0.0,     0.1,     0.2,     0.3,     0.4,     0.5,     0.6,     0.7,     0.9,     1.1,     1.3,     1.8]
                # len_score__max            = [ 0.0000,  0.0011,  0.0022,  0.0032,  0.0041,  0.0047,  0.0053,  0.0059,  0.0066,  0.0071,  0.0074,  0.0077]
                # len_score__avg            = [ 0.0000,  0.0011,  0.0021,  0.0031,  0.0039,  0.0046,  0.0052,  0.0057,  0.0065,  0.0069,  0.0072,  0.0075]
                # len_score__no_abs__min    = [ 0.0000, -0.0001, -0.0002, -0.0002, -0.0002, -0.0004, -0.0004, -0.0003, -0.0005, -0.0004, -0.0004, -0.0005]
                # len_score__no_abs__max    = [ 0.0000,  0.0000,  0.0001,  0.0002,  0.0002,  0.0003,  0.0003,  0.0004,  0.0001,  0.0001,  0.0004, -0.0000]
                # len_score__no_abs__avg    = [ 0.0000,  0.0000, -0.0001, -0.0000, -0.0000,  0.0000, -0.0001, -0.0000, -0.0002, -0.0002, -0.0001, -0.0003]

                # len_retention__max        = [ 0.0000,  0.0012,  0.0024,  0.0038,  0.0046,  0.0051,  0.0059,  0.0065,  0.0068,  0.0075,  0.0085,  0.0086]
                # len_retention__avg        = [ 0.0000,  0.0011,  0.0022,  0.0032,  0.0038,  0.0046,  0.0054,  0.0056,  0.0063,  0.0068,  0.0073,  0.0073]
                # len_retention__no_abs__min= [ 0.0000, -0.0003, -0.0003, -0.0007, -0.0003, -0.0012, -0.0009, -0.0008, -0.0013, -0.0013, -0.0012, -0.0009]
                # len_retention__no_abs__max= [ 0.0000,  0.0002, -0.0001,  0.0006,  0.0007,  0.0017,  0.0011,  0.0009,  0.0015,  0.0008,  0.0017,  0.0011]
                # len_retention__no_abs__avg= [ 0.0000,  0.0000, -0.0002,  0.0000,  0.0002,  0.0000, -0.0004,  0.0001, -0.0002, -0.0002,  0.0001, -0.0001]
                
                pass            
            
            
            
            #------------------#------------------#------------------
            dim_list =       [ 10, 100,1000]
            test_time_list = torch.tensor([ 200, 100, 10])
            #test_time_list = (test_time_list*0.1).to(torch.int64)
            for outter_iter_count in range(dim_list.__len__()):
                dim = dim_list[outter_iter_count]
                test_time = test_time_list[outter_iter_count]
                iota_of_dim = iota(dim)
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                print(f"dim {dim}    test_time {test_time}    device {device}")
            #------------------#------------------#------------------
                
                len_score__max           = []  # dont modity this
                len_score__avg           = []  # dont modity this
                len_score__no_abs__min   = []  # dont modity this
                len_score__no_abs__max   = []  # dont modity this
                len_score__no_abs__avg   = []  # dont modity this
                len_retention_score__max = []  # dont modity this
                len_retention_score__avg = []  # dont modity this
                len_retention_score__no_abs__min = []  # dont modity this
                len_retention_score__no_abs__max = []  # dont modity this
                len_retention_score__no_abs__avg = []  # dont modity this
                #------------------#------------------#------------------
                noise_strength_list = [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.10,    1.30,    1.80,     ]
                for noise_strength in noise_strength_list:
                #------------------#------------------#------------------

                    _raw_result__len_loss            = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__no_abs__len_loss    = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__len_retention_score = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__no_abs__len_retention_score = torch.empty(size=[test_time])  # dont modity this
                    
                    for _test_count in range(test_time):
                        
                        #----------------#----------------#----------------#----------------
                        ori_mat = random_dummy_mat__v2(dim=dim, noise_strength=noise_strength, div_sqrt_1_plus_ns_sqr=True, 
                                                            device=device, iota_of_dim=iota_of_dim)
                        
                        #<  measure the init>
                        len_loss, _, _log = LOSS__mat_is_standard_orthogonal(ori_mat, _debug__needs_log = True)
                        _raw_result__len_loss[_test_count] = len_loss#################################
                        assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                        _raw_result__no_abs__len_loss[_test_count] = _log[0][1]###############################
                        
                        length_retention_loss, (no_abs__length_retention_score,) = \
                                LOSS__vec_len_retention__of_a_mat_in_matmul(ori_mat, _debug__needs_log = True)
                        _raw_result__len_retention_score[_test_count] = length_retention_loss
                        _raw_result__no_abs__len_retention_score[_test_count] = no_abs__length_retention_score
                        #----------------#----------------#----------------#----------------
                        pass#for test_count
                    
                    len_score__max                  .append(_raw_result__len_loss.max ())
                    len_score__avg                  .append(_raw_result__len_loss.mean())
                    len_score__no_abs__min          .append(_raw_result__no_abs__len_loss.min ())
                    len_score__no_abs__max          .append(_raw_result__no_abs__len_loss.max ())
                    len_score__no_abs__avg          .append(_raw_result__no_abs__len_loss.mean())
                    len_retention_score__max        .append(_raw_result__len_retention_score.max ())
                    len_retention_score__avg        .append(_raw_result__len_retention_score.mean())
                    len_retention_score__no_abs__min.append(_raw_result__no_abs__len_retention_score.min ())
                    len_retention_score__no_abs__max.append(_raw_result__no_abs__len_retention_score.max ())
                    len_retention_score__no_abs__avg.append(_raw_result__no_abs__len_retention_score.mean())
                    
                    pass# for expansion_factor(y axis)
                
                print(f"dim = {dim}")
                print(f"noise_strength_list         = {str_the_list(noise_strength_list, 1, segment=",    ")}")
                print(f"len_score__max            = {str_the_list(len_score__max                  , 4)}")
                print(f"len_score__avg            = {str_the_list(len_score__avg                  , 4)}")
                print(f"len_score__no_abs__min    = {str_the_list(len_score__no_abs__min          , 4)}")
                print(f"len_score__no_abs__max    = {str_the_list(len_score__no_abs__max          , 4)}")
                print(f"len_score__no_abs__avg    = {str_the_list(len_score__no_abs__avg          , 4)}")
                print()
                print(f"len_retention__max        = {str_the_list(len_retention_score__max        , 4)}")
                print(f"len_retention__avg        = {str_the_list(len_retention_score__avg        , 4)}")
                print(f"len_retention__no_abs__min= {str_the_list(len_retention_score__no_abs__min, 4)}")
                print(f"len_retention__no_abs__max= {str_the_list(len_retention_score__no_abs__max, 4)}")
                print(f"len_retention__no_abs__avg= {str_the_list(len_retention_score__no_abs__avg, 4)}")
                
                pass#for outter_iter_count
            
            pass#/ test
        
        
        return 
    
    ____test____measure_the_random_gen_algo()
    pass







def full_test_version_of_length_correction__by_row(input:torch.Tensor, 
                        raw_power_me_to_protect_length:float|torch.Tensor, iota_of_dim:torch.Tensor|None = None, 
                        )->torch.Tensor:
    
    # if cap_to is None:
    #     cap_to__s = calc__cap_to____ver_1(input.shape[0], expansion_factor__s)
    #     pass
    if isinstance(raw_power_me_to_protect_length, float):
        raw_power_me_to_protect_length__s = torch.tensor(raw_power_me_to_protect_length)
        pass
    elif isinstance(raw_power_me_to_protect_length, torch.Tensor):
        raw_power_me_to_protect_length__s = raw_power_me_to_protect_length.detach().clone()
        pass
    else:
        assert False, "bad type."
        pass
    assert isinstance(raw_power_me_to_protect_length__s, torch.Tensor)
    assert raw_power_me_to_protect_length__s>-1. and raw_power_me_to_protect_length__s<1.
    
    if iota_of_dim is None:
        iota_of_dim = iota(input.shape[0])
        pass
    
    with torch.no_grad():
        normalized_row_vec, length_of_row_vec__n = get_full_info_of_vector_length__2d(input)
        target_length__n = length_of_row_vec__n.pow(raw_power_me_to_protect_length__s)
        target_length__n_1EXPANDn = expand_vec_to_matrix(target_length__n,each_element_to='row')
        result = normalized_row_vec.mul(target_length__n_1EXPANDn)
        pass#no grad
    return result


if "test" and True:
    def ____test____full_test_version_of_length_correction__by_row____basic():
        
        if "it doesn't touch perfect matrix" and False:
            print("it doesn't touch perfect matrix")
            
            #------------------#------------------#------------------
            dim_list =                          [ 2,  10,100, 1000]
            number_of_tests_list = torch.tensor([100,100, 50,  10])
            number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                iota_of_dim = iota(dim)
                number_of_tests = number_of_tests_list[outter_param_set]
                device = 'cpu'
                if dim>100:
                    device = 'cuda'
                    pass
                print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------

                _when_start = time.perf_counter()
                
                power_me_to_protect_length_list = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
                for power_me_to_protect_length in power_me_to_protect_length_list:
                    
                    for ii__test in range(number_of_tests):
                        
                        #------------------#------------------#------------------
                        #<  init                orthogonal mat.
                        mat = torch.eye(n=dim)
                        mat = randomly_rotate__matrix(mat)
                        
                        #<  perfect should score 0.
                        len_loss, no_abs_len_loss, length_retention_loss, \
                                no_abs__length_retention_loss = full_length_info_test__in_log10(mat)
                        assert _tensor_equal(len_loss                     , [0.])
                        assert _tensor_equal(no_abs_len_loss              , [0.])
                        assert _tensor_equal(length_retention_loss        , [0.])
                        assert _tensor_equal(no_abs__length_retention_loss, [0.])
                        
                        #<  calc
                        ori_mat = mat.detach().clone()
                        mat = full_test_version_of_length_correction__by_row(mat, \
                                power_me_to_protect_length = power_me_to_protect_length, iota_of_dim = iota_of_dim)
                        
                        #<  assertion
                        assert _tensor_equal(ori_mat, mat)
                        
                        pass#for ii__test
                    
                    pass#for power_me_to_protect_length
                _when_end = time.perf_counter()
                
                print(f"{_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                
                pass#for outter_param_set
            pass#/ test
        
        if "randn rand and protected by the func" and False:
            
            if "x1" and True:
                # randn    dim 10                                                       
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.078,  0.072,  0.091,  0.118]  0.127]
                # aft__len_loss__avg                = [ 0.037,  0.041,  0.058,  0.076]  0.080]
                # aft__no_abs_len_loss              = [-0.010, -0.011, -0.017, -0.022] -0.022]
                # aft__length_retention_loss__max   = [ 0.099,  0.098,  0.109,  0.116]  0.114]
                # aft__length_retention_loss__avg   = [ 0.073,  0.073,  0.075,  0.080]  0.081]
                # aft__no_abs__length_retention_loss= [-0.018, -0.021, -0.023, -0.023] -0.023]
                # randn    dim 100
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.015,  0.016,  0.021,  0.026]  0.027]
                # aft__len_loss__avg                = [ 0.012,  0.013,  0.018,  0.023]  0.025]
                # aft__no_abs_len_loss              = [-0.001, -0.001, -0.001, -0.002] -0.002]
                # aft__length_retention_loss__max   = [ 0.029,  0.028,  0.028,  0.030]  0.029]
                # aft__length_retention_loss__avg   = [ 0.024,  0.024,  0.024,  0.024]  0.025]
                # aft__no_abs__length_retention_loss= [-0.002, -0.002, -0.003, -0.002] -0.002]
                # randn    dim 1000
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.004,  0.004,  0.006,  0.007]  0.008]
                # aft__len_loss__avg                = [ 0.004,  0.004,  0.006,  0.007]  0.008]
                # aft__no_abs_len_loss              = [-0.000, -0.000, -0.000, -0.000] -0.000]
                # aft__length_retention_loss__max   = [ 0.009,  0.009,  0.009,  0.008]  0.009]
                # aft__length_retention_loss__avg   = [ 0.008,  0.008,  0.008,  0.008]  0.008]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.000, -0.000] -0.000]
                
                # ^^^^^ randn ^^^^^
                # vvvvv rand  vvvvv
                
                # rand    dim 10
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.045,  0.062,  0.164,  0.284]  0.315]
                # aft__len_loss__avg                = [ 0.027,  0.039,  0.127,  0.223]  0.247]
                # aft__no_abs_len_loss              = [-0.007, -0.029, -0.127, -0.223] -0.247]
                # aft__length_retention_loss__max   = [ 0.212,  0.219,  0.313,  0.393]  0.428]
                # aft__length_retention_loss__avg   = [ 0.174,  0.182,  0.235,  0.306]  0.327]
                # aft__no_abs__length_retention_loss= [-0.087, -0.107, -0.205, -0.298] -0.320]
                # rand    dim 100
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.010,  0.026,  0.122,  0.220]  0.244]
                # aft__len_loss__avg                = [ 0.009,  0.025,  0.120,  0.215]  0.239]
                # aft__no_abs_len_loss              = [-0.003, -0.024, -0.120, -0.215] -0.239]
                # aft__length_retention_loss__max   = [ 0.199,  0.200,  0.267,  0.339]  0.369]
                # aft__length_retention_loss__avg   = [ 0.173,  0.182,  0.233,  0.306]  0.330]
                # aft__no_abs__length_retention_loss= [-0.084, -0.107, -0.201, -0.295] -0.323]
                # rand    dim 1000
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.004,  0.024,  0.120,  0.215]  0.239]
                # aft__len_loss__avg                = [ 0.004,  0.024,  0.119,  0.215]  0.239]
                # aft__no_abs_len_loss              = [-0.002, -0.024, -0.119, -0.215] -0.239]
                # aft__length_retention_loss__max   = [ 0.181,  0.203,  0.259,  0.327]  0.349]
                # aft__length_retention_loss__avg   = [ 0.168,  0.183,  0.232,  0.301]  0.335]
                # aft__no_abs__length_retention_loss= [-0.075, -0.109, -0.204, -0.290] -0.329]
                
                pass
            
            if "x10" and True:
                # randn x10    dim 10
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.066,  0.113,  0.516,  0.942]  1.049]
                # aft__len_loss__avg                = [ 0.040,  0.100,  0.483,  0.877]  0.976]
                # aft__no_abs_len_loss              = [ 0.001,  0.089,  0.483,  0.877]  0.976]
                # aft__length_retention_loss__max   = [ 0.100,  0.136,  0.514,  0.947]  1.082]
                # aft__length_retention_loss__avg   = [ 0.072,  0.102,  0.475,  0.875]  0.976]
                # aft__no_abs__length_retention_loss= [-0.006,  0.080,  0.475,  0.875]  0.976]
                # randn x10    dim 100
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.020,  0.100,  0.501,  0.904]  1.004]
                # aft__len_loss__avg                = [ 0.018,  0.099,  0.498,  0.898]  0.998]
                # aft__no_abs_len_loss              = [ 0.009,  0.099,  0.498,  0.898]  0.998]
                # aft__length_retention_loss__max   = [ 0.029,  0.102,  0.503,  0.907]  1.007]
                # aft__length_retention_loss__avg   = [ 0.025,  0.097,  0.497,  0.898]  0.998]
                # aft__no_abs__length_retention_loss= [ 0.008,  0.097,  0.497,  0.898]  0.998]
                # randn x10    dim 1000
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.011,  0.100,  0.500,  0.900]  1.000]
                # aft__len_loss__avg                = [ 0.011,  0.100,  0.500,  0.900]  1.000]
                # aft__no_abs_len_loss              = [ 0.010,  0.100,  0.500,  0.900]  1.000]
                # aft__length_retention_loss__max   = [ 0.012,  0.101,  0.501,  0.901]  1.001]
                # aft__length_retention_loss__avg   = [ 0.011,  0.100,  0.500,  0.900]  1.000]
                # aft__no_abs__length_retention_loss= [ 0.010,  0.100,  0.500,  0.900]  1.000]
                
                # ^^^^^ randn x10 ^^^^^
                # vvvvv rand x10  vvvvv
                
                # rand x10    dim 10
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.054,  0.092,  0.395,  0.723]  0.804]
                # aft__len_loss__avg                = [ 0.031,  0.078,  0.373,  0.678]  0.754]
                # aft__no_abs_len_loss              = [ 0.002,  0.070,  0.373,  0.678]  0.754]
                # aft__length_retention_loss__max   = [ 0.216,  0.203,  0.363,  0.666]  0.763]
                # aft__length_retention_loss__avg   = [ 0.176,  0.159,  0.306,  0.596]  0.680]
                # aft__no_abs__length_retention_loss= [-0.078, -0.010,  0.299,  0.596]  0.680]
                # rand x10    dim 100
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.014,  0.076,  0.382,  0.689]  0.765]
                # aft__len_loss__avg                = [ 0.012,  0.076,  0.380,  0.684]  0.760]
                # aft__no_abs_len_loss              = [ 0.007,  0.076,  0.380,  0.684]  0.760]
                # aft__length_retention_loss__max   = [ 0.193,  0.176,  0.346,  0.650]  0.722]
                # aft__length_retention_loss__avg   = [ 0.170,  0.152,  0.300,  0.604]  0.681]
                # aft__no_abs__length_retention_loss= [-0.076, -0.006,  0.300,  0.604]  0.681]
                # rand x10    dim 1000
                # raw_power_me_to_protect_length   = [ 0.010,  0.100,  0.500,  0.900]   1.000
                # aft__len_loss__max                = [ 0.008,  0.076,  0.381,  0.686]  0.762]
                # aft__len_loss__avg                = [ 0.008,  0.076,  0.381,  0.685]  0.761]
                # aft__no_abs_len_loss              = [ 0.008,  0.076,  0.381,  0.685]  0.761]
                # aft__length_retention_loss__max   = [ 0.193,  0.179,  0.310,  0.638]  0.700]
                # aft__length_retention_loss__avg   = [ 0.172,  0.157,  0.289,  0.605]  0.672]
                # aft__no_abs__length_retention_loss= [-0.077, -0.001,  0.289,  0.605]  0.672]
                pass
            
            print("it doesn't touch perfect matrix")
            
            random_generator_style_list = ["randn","rand", "randn x10","rand x10"]
            random_generator_style_list = ["randn","rand"]
            for random_generator_style in random_generator_style_list:
                
                #------------------#------------------#------------------
                dim_list =                          [ 10, 100, 1000]
                number_of_tests_list = torch.tensor([100,  50,  10])
                number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
                for outter_param_set in range(dim_list.__len__()):
                    dim = dim_list[outter_param_set]
                    iota_of_dim = iota(dim)
                    number_of_tests = number_of_tests_list[outter_param_set]
                    device = 'cuda'
                    if dim>10001:
                        device = 'cuda'
                        pass
                    print(f"dim {dim}   test_time {number_of_tests}    device {device}")
                #------------------#------------------#------------------
                    raw_power_me_to_protect_length_list = [0.01, 0.1, 0.5, 0.9]

                    ori__len_loss__max                  = torch.empty(size=[raw_power_me_to_protect_length_list.__len__()])#don't modify this
                    ori__len_loss__avg                  = torch.empty(size=[raw_power_me_to_protect_length_list.__len__()])#don't modify this
                    ori__no_abs_len_loss                = torch.empty(size=[raw_power_me_to_protect_length_list.__len__()])#don't modify this
                    
                    ori__length_retention_loss__max     = torch.empty(size=[raw_power_me_to_protect_length_list.__len__()])#don't modify this
                    ori__length_retention_loss__avg     = torch.empty(size=[raw_power_me_to_protect_length_list.__len__()])#don't modify this
                    ori__no_abs__length_retention_loss  = torch.empty(size=[raw_power_me_to_protect_length_list.__len__()])#don't modify this
                    
                    after__len_loss__max                = []#don't modify this
                    after__len_loss__avg                = []#don't modify this
                    after__no_abs_len_loss              = []#don't modify this
                    
                    after__length_retention_loss__max   = []#don't modify this
                    after__length_retention_loss__avg   = []#don't modify this
                    after__no_abs__length_retention_loss= []#don't modify this
                    
                    _when_start = time.perf_counter()
                    
                    for raw_power_me_to_protect_length in raw_power_me_to_protect_length_list:
                        
                        _raw_result__ori__len_loss                      = torch.empty(size=[number_of_tests])
                        _raw_result__ori__no_abs_len_loss               = torch.empty(size=[number_of_tests])
                        _raw_result__ori__length_retention_loss         = torch.empty(size=[number_of_tests])
                        _raw_result__ori__no_abs__length_retention_loss = torch.empty(size=[number_of_tests])
                        _raw_result__after__len_loss                      = torch.empty(size=[number_of_tests])
                        _raw_result__after__no_abs_len_loss               = torch.empty(size=[number_of_tests])
                        _raw_result__after__length_retention_loss         = torch.empty(size=[number_of_tests])
                        _raw_result__after__no_abs__length_retention_loss = torch.empty(size=[number_of_tests])
                        
                        
                        for ii__test in range(number_of_tests):
                            
                            #------------------#------------------#------------------
                            #<  init                orthogonal mat.
                            if random_generator_style == "randn":
                                mat = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)
                                pass
                            elif random_generator_style == "rand":
                                mat = torch.rand (size=[dim,dim], device=device)/math.sqrt(dim)
                                pass
                            elif random_generator_style == "rand x10":
                                mat = torch.rand (size=[dim,dim], device=device)/math.sqrt(dim)*10.
                                pass
                            elif random_generator_style == "rand x10":
                                mat = torch.rand (size=[dim,dim], device=device)/math.sqrt(dim)*10.
                                pass
                            # elif random_generator_style == "rand_dummy_v2":
                            #     mat = random_dummy_mat__v2(dim=dim, device=device)/math.sqrt(dim)*10.
                            #     pass
                            # random_dummy_mat__v2
                            else:
                                assert False, "bad param"
                            
                            #<  perfect should score 0.
                            len_loss, no_abs_len_loss, length_retention_loss, \
                                    no_abs__length_retention_loss = full_length_info_test__in_log10(mat)
                            _raw_result__ori__len_loss                     [ii__test] = len_loss                     
                            _raw_result__ori__no_abs_len_loss              [ii__test] = no_abs_len_loss              
                            _raw_result__ori__length_retention_loss        [ii__test] = length_retention_loss        
                            _raw_result__ori__no_abs__length_retention_loss[ii__test] = no_abs__length_retention_loss
                            
                            #<  calc
                            mat = full_test_version_of_length_correction__by_row(mat, \
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length, iota_of_dim = iota_of_dim)
                            
                            #<  measure
                            len_loss, no_abs_len_loss, length_retention_loss, \
                                    no_abs__length_retention_loss = full_length_info_test__in_log10(mat)
                            _raw_result__after__len_loss                     [ii__test] = len_loss                     
                            _raw_result__after__no_abs_len_loss              [ii__test] = no_abs_len_loss              
                            _raw_result__after__length_retention_loss        [ii__test] = length_retention_loss        
                            _raw_result__after__no_abs__length_retention_loss[ii__test] = no_abs__length_retention_loss
                            
                            pass#for ii__test
                        
                        
                        ori__len_loss__max                  .append(_raw_result__ori__len_loss       .max())
                        ori__len_loss__avg                  .append(_raw_result__ori__len_loss       .mean())
                        ori__no_abs_len_loss                .append(_raw_result__ori__no_abs_len_loss.mean())

                        ori__length_retention_loss__max     .append(_raw_result__ori__length_retention_loss        .max())
                        ori__length_retention_loss__avg     .append(_raw_result__ori__length_retention_loss        .mean())
                        ori__no_abs__length_retention_loss  .append(_raw_result__ori__no_abs__length_retention_loss.mean())

                        after__len_loss__max                .append(_raw_result__after__len_loss       .max())
                        after__len_loss__avg                .append(_raw_result__after__len_loss       .mean())
                        after__no_abs_len_loss              .append(_raw_result__after__no_abs_len_loss.mean())

                        after__length_retention_loss__max   .append(_raw_result__after__length_retention_loss        .max())
                        after__length_retention_loss__avg   .append(_raw_result__after__length_retention_loss        .mean())
                        after__no_abs__length_retention_loss.append(_raw_result__after__no_abs__length_retention_loss.mean())
                        
                        pass#for power_me_to_protect_length
                    _when_end = time.perf_counter()
                    
                    print(f"{_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                    print(f"{random_generator_style}    dim {dim}")
                    # print(f"ori__len_loss__max                = {str_the_list(ori__len_loss__max                  , 3)}")
                    # print(f"ori__len_loss__avg                = {str_the_list(ori__len_loss__avg                  , 3)}")
                    # print(f"ori__no_abs_len_loss              = {str_the_list(ori__no_abs_len_loss                , 3)}")
                    # print(f"ori__length_retention_loss__max   = {str_the_list(ori__length_retention_loss__max     , 3)}")
                    # print(f"ori__length_retention_loss__avg   = {str_the_list(ori__length_retention_loss__avg     , 3)}")
                    # print(f"ori__no_abs__length_retention_loss= {str_the_list(ori__no_abs__length_retention_loss  , 3)}")
                    print(f"power_me_to_protect_length       = {str_the_list(raw_power_me_to_protect_length_list  , 3)}   1.000")
                    print(f"aft__len_loss__max                = {str_the_list(after__len_loss__max                 , 3)}   {ori__len_loss__max                .mean()}")
                    print(f"aft__len_loss__avg                = {str_the_list(after__len_loss__avg                 , 3)}   {ori__len_loss__avg                .mean()}")
                    print(f"aft__no_abs_len_loss              = {str_the_list(after__no_abs_len_loss               , 3)}   {ori__no_abs_len_loss              .mean()}")
                    print(f"aft__length_retention_loss__max   = {str_the_list(after__length_retention_loss__max    , 3)}   {ori__length_retention_loss__max   .mean()}")
                    print(f"aft__length_retention_loss__avg   = {str_the_list(after__length_retention_loss__avg    , 3)}   {ori__length_retention_loss__avg   .mean()}")
                    print(f"aft__no_abs__length_retention_loss= {str_the_list(after__no_abs__length_retention_loss , 3)}   {ori__no_abs__length_retention_loss.mean()}")
                    #print(f"dim        = {str_the_list(dim_list, 0, ",    ")}")
                    
                    pass#for outter_param_set
            
                pass#for random_generator_style
            pass#/ test
        
        
        
        
        
        
        if "rand dummy v2   protected." and True:
            if True:
                # when noise strength is 0, it's a perfect orthogonal matrix. All measure will be 0.
            
            1w
            1w
            1w
            1w来读
                            
                # noise_strength 0.1     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.010,  0.008,  0.007,  0.008,  0.012,  0.016]
                # aft__len_loss__avg                = [ 0.007,  0.005,  0.004,  0.005,  0.007,  0.011]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.000, -0.001]
                # aft__length_retention_loss__max   = [ 0.014,  0.014,  0.013,  0.014,  0.013,  0.017]
                # aft__length_retention_loss__avg   = [ 0.010,  0.009,  0.009,  0.010,  0.010,  0.011]
                # aft__no_abs__length_retention_loss= [ 0.000, -0.000, -0.000, -0.000, -0.000, -0.001]
                # noise_strength 0.2     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.022,  0.016,  0.016,  0.016,  0.020,  0.035]
                # aft__len_loss__avg                = [ 0.015,  0.010,  0.009,  0.010,  0.014,  0.022]
                # aft__no_abs_len_loss              = [ 0.001, -0.000, -0.001, -0.001, -0.001, -0.001]
                # aft__length_retention_loss__max   = [ 0.026,  0.027,  0.026,  0.027,  0.027,  0.031]
                # aft__length_retention_loss__avg   = [ 0.019,  0.019,  0.018,  0.018,  0.019,  0.021]
                # aft__no_abs__length_retention_loss= [ 0.000, -0.001, -0.002, -0.002, -0.002, -0.002]
                # noise_strength 0.3     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.033,  0.030,  0.019,  0.023,  0.031,  0.049]
                # aft__len_loss__avg                = [ 0.021,  0.014,  0.012,  0.014,  0.020,  0.031]
                # aft__no_abs_len_loss              = [ 0.001, -0.001, -0.001, -0.002, -0.002, -0.005]
                # aft__length_retention_loss__max   = [ 0.038,  0.035,  0.037,  0.036,  0.041,  0.053]
                # aft__length_retention_loss__avg   = [ 0.028,  0.027,  0.027,  0.027,  0.028,  0.031]
                # aft__no_abs__length_retention_loss= [-0.000, -0.003, -0.003, -0.003, -0.002, -0.006]
                # noise_strength 0.4     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.039,  0.028,  0.025,  0.029,  0.038,  0.066]
                # aft__len_loss__avg                = [ 0.027,  0.018,  0.016,  0.019,  0.028,  0.040]
                # aft__no_abs_len_loss              = [ 0.002, -0.001, -0.002, -0.003, -0.004, -0.006]
                # aft__length_retention_loss__max   = [ 0.051,  0.051,  0.046,  0.050,  0.051,  0.059]
                # aft__length_retention_loss__avg   = [ 0.037,  0.036,  0.035,  0.035,  0.037,  0.041]
                # aft__no_abs__length_retention_loss= [-0.000, -0.003, -0.004, -0.006, -0.006, -0.006]
                # noise_strength 0.5     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.052,  0.033,  0.033,  0.034,  0.052,  0.070]
                # aft__len_loss__avg                = [ 0.033,  0.022,  0.021,  0.023,  0.032,  0.049]
                # aft__no_abs_len_loss              = [ 0.001, -0.002, -0.003, -0.004, -0.006, -0.007]
                # aft__length_retention_loss__max   = [ 0.065,  0.061,  0.057,  0.055,  0.062,  0.068]
                # aft__length_retention_loss__avg   = [ 0.044,  0.041,  0.041,  0.042,  0.044,  0.048]
                # aft__no_abs__length_retention_loss= [-0.001, -0.004, -0.006, -0.006, -0.010, -0.006]
                # noise_strength 0.6     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.064,  0.043,  0.035,  0.043,  0.058,  0.081]
                # aft__len_loss__avg                = [ 0.037,  0.025,  0.022,  0.026,  0.037,  0.055]
                # aft__no_abs_len_loss              = [ 0.002, -0.003, -0.003, -0.005, -0.007, -0.014]
                # aft__length_retention_loss__max   = [ 0.075,  0.079,  0.068,  0.065,  0.067,  0.077]
                # aft__length_retention_loss__avg   = [ 0.050,  0.048,  0.047,  0.048,  0.049,  0.054]
                # aft__no_abs__length_retention_loss= [-0.001, -0.008, -0.007, -0.009, -0.010, -0.012]
                # noise_strength 0.7     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.068,  0.046,  0.045,  0.046,  0.071,  0.096]
                # aft__len_loss__avg                = [ 0.041,  0.028,  0.026,  0.029,  0.041,  0.059]
                # aft__no_abs_len_loss              = [ 0.004, -0.003, -0.005, -0.006, -0.009, -0.009]
                # aft__length_retention_loss__max   = [ 0.079,  0.070,  0.070,  0.070,  0.077,  0.082]
                # aft__length_retention_loss__avg   = [ 0.055,  0.051,  0.054,  0.053,  0.055,  0.059]
                # aft__no_abs__length_retention_loss= [-0.000, -0.008, -0.010, -0.012, -0.014, -0.007]
                # noise_strength 0.9     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.081,  0.056,  0.054,  0.051,  0.071,  0.094]
                # aft__len_loss__avg                = [ 0.048,  0.032,  0.029,  0.032,  0.047,  0.067]
                # aft__no_abs_len_loss              = [ 0.002, -0.004, -0.006, -0.007, -0.013, -0.012]
                # aft__length_retention_loss__max   = [ 0.086,  0.081,  0.085,  0.087,  0.098,  0.105]
                # aft__length_retention_loss__avg   = [ 0.062,  0.060,  0.060,  0.059,  0.061,  0.068]
                # aft__no_abs__length_retention_loss= [-0.001, -0.010, -0.014, -0.015, -0.017, -0.014]
                # noise_strength 1.3     dim 10
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.078,  0.062,  0.062,  0.064,  0.081,  0.126]
                # aft__len_loss__avg                = [ 0.053,  0.036,  0.033,  0.038,  0.053,  0.078]
                # aft__no_abs_len_loss              = [ 0.005, -0.005, -0.008, -0.010, -0.016, -0.021]
                # aft__length_retention_loss__max   = [ 0.094,  0.092,  0.098,  0.094,  0.099,  0.128]
                # aft__length_retention_loss__avg   = [ 0.070,  0.067,  0.067,  0.068,  0.070,  0.077]
                # aft__no_abs__length_retention_loss= [-0.002, -0.013, -0.014, -0.018, -0.022, -0.019]
                
                # noise_strength 0.1     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.003,  0.002,  0.002,  0.002,  0.003,  0.004]
                # aft__len_loss__avg                = [ 0.002,  0.002,  0.001,  0.002,  0.002,  0.003]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000,  0.000,  0.000]
                # aft__length_retention_loss__max   = [ 0.004,  0.004,  0.004,  0.004,  0.004,  0.004]
                # aft__length_retention_loss__avg   = [ 0.003,  0.003,  0.003,  0.003,  0.003,  0.003]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000,  0.000, -0.000,  0.000,  0.000]
                # noise_strength 0.2     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.005,  0.004,  0.003,  0.004,  0.005,  0.008]
                # aft__len_loss__avg                = [ 0.005,  0.003,  0.003,  0.003,  0.005,  0.007]
                # aft__no_abs_len_loss              = [-0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.008,  0.008,  0.008,  0.009,  0.008,  0.008]
                # aft__length_retention_loss__avg   = [ 0.007,  0.007,  0.007,  0.007,  0.007,  0.007]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # noise_strength 0.3     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.008,  0.006,  0.005,  0.006,  0.008,  0.011]
                # aft__len_loss__avg                = [ 0.007,  0.005,  0.004,  0.005,  0.007,  0.010]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.012,  0.012,  0.011,  0.012,  0.011,  0.012]
                # aft__length_retention_loss__avg   = [ 0.010,  0.010,  0.010,  0.010,  0.010,  0.010]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.000, -0.000, -0.001, -0.000]
                # noise_strength 0.4     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.010,  0.007,  0.006,  0.007,  0.010,  0.015]
                # aft__len_loss__avg                = [ 0.009,  0.006,  0.005,  0.006,  0.009,  0.013]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.001, -0.001]
                # aft__length_retention_loss__max   = [ 0.016,  0.014,  0.015,  0.015,  0.014,  0.015]
                # aft__length_retention_loss__avg   = [ 0.012,  0.012,  0.012,  0.013,  0.012,  0.013]
                # aft__no_abs__length_retention_loss= [ 0.000, -0.000, -0.000, -0.001, -0.001, -0.001]
                # noise_strength 0.5     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.012,  0.008,  0.008,  0.009,  0.012,  0.017]
                # aft__len_loss__avg                = [ 0.010,  0.007,  0.006,  0.007,  0.011,  0.015]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.001, -0.001]
                # aft__length_retention_loss__max   = [ 0.017,  0.017,  0.018,  0.016,  0.017,  0.017]
                # aft__length_retention_loss__avg   = [ 0.015,  0.014,  0.015,  0.014,  0.015,  0.015]
                # aft__no_abs__length_retention_loss= [-0.000, -0.001, -0.001, -0.001, -0.001, -0.002]
                # noise_strength 0.6     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.014,  0.009,  0.009,  0.009,  0.013,  0.019]
                # aft__len_loss__avg                = [ 0.012,  0.008,  0.008,  0.008,  0.012,  0.017]
                # aft__no_abs_len_loss              = [-0.000, -0.000, -0.000, -0.000, -0.001, -0.001]
                # aft__length_retention_loss__max   = [ 0.019,  0.019,  0.019,  0.020,  0.019,  0.020]
                # aft__length_retention_loss__avg   = [ 0.016,  0.017,  0.016,  0.016,  0.017,  0.017]
                # aft__no_abs__length_retention_loss= [-0.000, -0.001, -0.000, -0.002, -0.001, -0.001]
                # noise_strength 0.7     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.014,  0.011,  0.010,  0.011,  0.015,  0.020]
                # aft__len_loss__avg                = [ 0.013,  0.009,  0.008,  0.010,  0.013,  0.018]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.001, -0.001, -0.001, -0.001]
                # aft__length_retention_loss__max   = [ 0.021,  0.021,  0.021,  0.022,  0.021,  0.022]
                # aft__length_retention_loss__avg   = [ 0.018,  0.018,  0.018,  0.018,  0.018,  0.018]
                # aft__no_abs__length_retention_loss= [-0.000, -0.001, -0.001, -0.001, -0.001, -0.001]
                # noise_strength 0.9     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.017,  0.012,  0.011,  0.012,  0.017,  0.023]
                # aft__len_loss__avg                = [ 0.015,  0.011,  0.009,  0.011,  0.015,  0.020]
                # aft__no_abs_len_loss              = [ 0.001, -0.001, -0.001, -0.001, -0.001, -0.002]
                # aft__length_retention_loss__max   = [ 0.023,  0.025,  0.025,  0.025,  0.025,  0.025]
                # aft__length_retention_loss__avg   = [ 0.021,  0.020,  0.020,  0.020,  0.020,  0.021]
                # aft__no_abs__length_retention_loss= [ 0.000, -0.001, -0.001, -0.001, -0.002, -0.002]
                # noise_strength 1.3     dim 100
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.018,  0.014,  0.013,  0.014,  0.020,  0.026]
                # aft__len_loss__avg                = [ 0.016,  0.012,  0.011,  0.012,  0.017,  0.023]
                # aft__no_abs_len_loss              = [ 0.000, -0.001, -0.001, -0.001, -0.002, -0.001]
                # aft__length_retention_loss__max   = [ 0.026,  0.026,  0.026,  0.026,  0.026,  0.027]
                # aft__length_retention_loss__avg   = [ 0.022,  0.022,  0.022,  0.023,  0.023,  0.023]
                # aft__no_abs__length_retention_loss= [-0.001, -0.001, -0.002, -0.001, -0.002, -0.001]
                
                # noise_strength 0.1     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.001,  0.001,  0.000,  0.001,  0.001,  0.001]
                # aft__len_loss__avg                = [ 0.001,  0.001,  0.000,  0.001,  0.001,  0.001]
                # aft__no_abs_len_loss              = [ 0.000,  0.000, -0.000, -0.000, -0.000,  0.000]
                # aft__length_retention_loss__max   = [ 0.001,  0.001,  0.001,  0.001,  0.001,  0.001]
                # aft__length_retention_loss__avg   = [ 0.001,  0.001,  0.001,  0.001,  0.001,  0.001]
                # aft__no_abs__length_retention_loss= [-0.000,  0.000, -0.000, -0.000, -0.000,  0.000]
                # noise_strength 0.2     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.002,  0.001,  0.001,  0.001,  0.002,  0.002]
                # aft__len_loss__avg                = [ 0.002,  0.001,  0.001,  0.001,  0.001,  0.002]
                # aft__no_abs_len_loss              = [-0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.002,  0.002,  0.002,  0.002,  0.002,  0.002]
                # aft__length_retention_loss__avg   = [ 0.002,  0.002,  0.002,  0.002,  0.002,  0.002]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.000,  0.000,  0.000,  0.000]
                # noise_strength 0.3     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.002,  0.002,  0.001,  0.002,  0.002,  0.003]
                # aft__len_loss__avg                = [ 0.002,  0.002,  0.001,  0.002,  0.002,  0.003]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.004,  0.004,  0.004,  0.004,  0.003,  0.003]
                # aft__length_retention_loss__avg   = [ 0.003,  0.003,  0.003,  0.003,  0.003,  0.003]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.000, -0.000,  0.000, -0.000]
                # noise_strength 0.4     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.003,  0.002,  0.002,  0.002,  0.003,  0.004]
                # aft__len_loss__avg                = [ 0.003,  0.002,  0.002,  0.002,  0.003,  0.004]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.005,  0.004,  0.004,  0.004,  0.004,  0.004]
                # aft__length_retention_loss__avg   = [ 0.004,  0.004,  0.004,  0.004,  0.004,  0.004]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000,  0.000, -0.000, -0.000, -0.000]
                # noise_strength 0.5     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.003,  0.002,  0.002,  0.002,  0.003,  0.005]
                # aft__len_loss__avg                = [ 0.003,  0.002,  0.002,  0.002,  0.003,  0.005]
                # aft__no_abs_len_loss              = [-0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.005,  0.005,  0.005,  0.005,  0.005,  0.005]
                # aft__length_retention_loss__avg   = [ 0.005,  0.005,  0.005,  0.005,  0.005,  0.005]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.000, -0.000,  0.000,  0.000]
                # noise_strength 0.6     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.004,  0.003,  0.003,  0.003,  0.004,  0.005]
                # aft__len_loss__avg                = [ 0.004,  0.003,  0.002,  0.003,  0.004,  0.005]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.006,  0.006,  0.006,  0.006,  0.006,  0.006]
                # aft__length_retention_loss__avg   = [ 0.005,  0.005,  0.005,  0.005,  0.005,  0.005]
                # aft__no_abs__length_retention_loss= [-0.000,  0.000, -0.000, -0.001, -0.000, -0.000]
                # noise_strength 0.7     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.004,  0.003,  0.003,  0.003,  0.004,  0.006]
                # aft__len_loss__avg                = [ 0.004,  0.003,  0.003,  0.003,  0.004,  0.006]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.006,  0.006,  0.007,  0.006,  0.007,  0.006]
                # aft__length_retention_loss__avg   = [ 0.006,  0.006,  0.006,  0.006,  0.006,  0.006]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # noise_strength 0.9     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.005,  0.003,  0.003,  0.003,  0.005,  0.007]
                # aft__len_loss__avg                = [ 0.005,  0.003,  0.003,  0.003,  0.005,  0.006]
                # aft__no_abs_len_loss              = [-0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.007,  0.007,  0.007,  0.007,  0.007,  0.007]
                # aft__length_retention_loss__avg   = [ 0.006,  0.006,  0.006,  0.006,  0.007,  0.007]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.001, -0.001, -0.000, -0.000]
                # noise_strength 1.3     dim 1000
                # power_me_to_protect_length       = [-0.500, -0.100,  0.001,  0.100,  0.500,  0.999]
                # aft__len_loss__max                = [ 0.005,  0.004,  0.004,  0.004,  0.005,  0.007]
                # aft__len_loss__avg                = [ 0.005,  0.004,  0.003,  0.004,  0.005,  0.007]
                # aft__no_abs_len_loss              = [ 0.000, -0.000, -0.000, -0.000, -0.000, -0.000]
                # aft__length_retention_loss__max   = [ 0.008,  0.008,  0.008,  0.008,  0.008,  0.008]
                # aft__length_retention_loss__avg   = [ 0.007,  0.007,  0.007,  0.007,  0.007,  0.007]
                # aft__no_abs__length_retention_loss= [-0.000, -0.000, -0.000, -0.000,  0.000,  0.001]
            
            
            
            
            
            
            
            
                pass
            
            print("rand dummy v2   protected.")
        
            #------------------#------------------#------------------
            dim_list =                          [ 10, 100, 1000]
            number_of_tests_list = torch.tensor([100,  50,  10])
            number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                iota_of_dim = iota(dim)
                number_of_tests = number_of_tests_list[outter_param_set]
                device = 'cpu'
                if dim>10001:
                    device = 'cuda'
                    pass
                print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------
                #some other ref
                #                         noise_strength_list = torch.tensor(
                #                             [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.10,    1.30,    1.80,     ])
                #                         angle_loss_list__full = torch.tensor([
                #                             [ 0.0000,  0.2184,  0.4341,  0.6253,  0.7969,  0.9537,  1.0867,  1.1974,  1.3483,  1.4409,  1.5145,  1.5496,   ],#dim 10
                #                             [ 0.0000,  0.2234,  0.4377,  0.6346,  0.8075,  0.9575,  1.0838,  1.1826,  1.3292,  1.4228,  1.4859,  1.5540,   ],#dim 100
                #                             [ 0.0000,  0.2238,  0.4381,  0.6346,  0.8080,  0.9577,  1.0820,  1.1828,  1.3303,  1.4248,  1.4814,  1.5506,   ],])# dim 1000
                
                noise_strength_list = [ 0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.30,]
                for kk in range(noise_strength_list.__len__()):
                    noise_strength = noise_strength_list[kk]
            
                    raw_power_me_to_protect_length_list = [-0.5, -0.1, 0.001, 0.1, 0.5, 0.999]
                    #_size = raw_power_me_to_protect_length_list.__len__()

                    # ori__len_loss__max                  = torch.empty(size=[_size])#don't modify this
                    # ori__len_loss__avg                  = torch.empty(size=[_size])#don't modify this
                    # ori__no_abs_len_loss                = torch.empty(size=[_size])#don't modify this
                    
                    # ori__length_retention_loss__max     = torch.empty(size=[_size])#don't modify this
                    # ori__length_retention_loss__avg     = torch.empty(size=[_size])#don't modify this
                    # ori__no_abs__length_retention_loss  = torch.empty(size=[_size])#don't modify this
                    #del _size
                    
                    after__len_loss__max                = []#don't modify this
                    after__len_loss__avg                = []#don't modify this
                    after__no_abs_len_loss              = []#don't modify this
                    
                    after__length_retention_loss__max   = []#don't modify this
                    after__length_retention_loss__avg   = []#don't modify this
                    after__no_abs__length_retention_loss= []#don't modify this
                    
                    _when_start = time.perf_counter()
                    
                    for jj_x_axis in range(raw_power_me_to_protect_length_list.__len__()):# x axis
                        raw_power_me_to_protect_length =  raw_power_me_to_protect_length_list[jj_x_axis]
                        
                        #y axis
                        _raw_result__ori__len_loss                      = torch.empty(size=[number_of_tests])
                        _raw_result__ori__no_abs_len_loss               = torch.empty(size=[number_of_tests])
                        _raw_result__ori__length_retention_loss         = torch.empty(size=[number_of_tests])
                        _raw_result__ori__no_abs__length_retention_loss = torch.empty(size=[number_of_tests])
                        _raw_result__after__len_loss                      = torch.empty(size=[number_of_tests])
                        _raw_result__after__no_abs_len_loss               = torch.empty(size=[number_of_tests])
                        _raw_result__after__length_retention_loss         = torch.empty(size=[number_of_tests])
                        _raw_result__after__no_abs__length_retention_loss = torch.empty(size=[number_of_tests])
                        
                        
                        for ii__test in range(number_of_tests):
                            
                            #------------------#------------------#------------------
                            #<  init                orthogonal mat.
                            mat = random_dummy_mat__v2(dim=dim, noise_strength = noise_strength,
                                    div_sqrt_1_plus_ns_sqr = True, device=device, iota_of_dim = iota_of_dim)
                            
                            #<  perfect should score 0.
                            len_loss, no_abs_len_loss, length_retention_loss, \
                                    no_abs__length_retention_loss = full_length_info_test__in_log10(mat)
                            _raw_result__ori__len_loss                     [ii__test] = len_loss                     
                            _raw_result__ori__no_abs_len_loss              [ii__test] = no_abs_len_loss              
                            _raw_result__ori__length_retention_loss        [ii__test] = length_retention_loss        
                            _raw_result__ori__no_abs__length_retention_loss[ii__test] = no_abs__length_retention_loss
                            
                            #<  calc
                            mat = full_test_version_of_length_correction__by_row(mat, \
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length, iota_of_dim = iota_of_dim)
                            
                            #<  measure
                            len_loss, no_abs_len_loss, length_retention_loss, \
                                    no_abs__length_retention_loss = full_length_info_test__in_log10(mat)
                            _raw_result__after__len_loss                     [ii__test] = len_loss                     
                            _raw_result__after__no_abs_len_loss              [ii__test] = no_abs_len_loss              
                            _raw_result__after__length_retention_loss        [ii__test] = length_retention_loss        
                            _raw_result__after__no_abs__length_retention_loss[ii__test] = no_abs__length_retention_loss
                            
                            pass#for ii__test
                        
                        
                        # ori__len_loss__max                  [jj_x_axis] = _raw_result__ori__len_loss       .max()
                        # ori__len_loss__avg                  [jj_x_axis] = _raw_result__ori__len_loss       .mean()
                        # ori__no_abs_len_loss                [jj_x_axis] = _raw_result__ori__no_abs_len_loss.mean()

                        # ori__length_retention_loss__max     [jj_x_axis] = _raw_result__ori__length_retention_loss        .max()
                        # ori__length_retention_loss__avg     [jj_x_axis] = _raw_result__ori__length_retention_loss        .mean()
                        # ori__no_abs__length_retention_loss  [jj_x_axis] = _raw_result__ori__no_abs__length_retention_loss.mean()

                        after__len_loss__max                .append(_raw_result__after__len_loss       .max())
                        after__len_loss__avg                .append(_raw_result__after__len_loss       .mean())
                        after__no_abs_len_loss              .append(_raw_result__after__no_abs_len_loss.mean())

                        after__length_retention_loss__max   .append(_raw_result__after__length_retention_loss        .max())
                        after__length_retention_loss__avg   .append(_raw_result__after__length_retention_loss        .mean())
                        after__no_abs__length_retention_loss.append(_raw_result__after__no_abs__length_retention_loss.mean())
                        
                        pass#for power_me_to_protect_length
                    _when_end = time.perf_counter()
                    
                    print(f"{_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                    print(f"noise_strength {noise_strength}     dim {dim}")
                    # print(f"ori__len_loss__max                = {str_the_list(ori__len_loss__max                  , 3)}")
                    # print(f"ori__len_loss__avg                = {str_the_list(ori__len_loss__avg                  , 3)}")
                    # print(f"ori__no_abs_len_loss              = {str_the_list(ori__no_abs_len_loss                , 3)}")
                    # print(f"ori__length_retention_loss__max   = {str_the_list(ori__length_retention_loss__max     , 3)}")
                    # print(f"ori__length_retention_loss__avg   = {str_the_list(ori__length_retention_loss__avg     , 3)}")
                    # print(f"ori__no_abs__length_retention_loss= {str_the_list(ori__no_abs__length_retention_loss  , 3)}")
                    print(f"power_me_to_protect_length       = {str_the_list(raw_power_me_to_protect_length_list  , 3)}")
                    print(f"aft__len_loss__max                = {str_the_list(after__len_loss__max                 , 3)}")
                    print(f"aft__len_loss__avg                = {str_the_list(after__len_loss__avg                 , 3)}")
                    print(f"aft__no_abs_len_loss              = {str_the_list(after__no_abs_len_loss               , 3)}")
                    print(f"aft__length_retention_loss__max   = {str_the_list(after__length_retention_loss__max    , 3)}")
                    print(f"aft__length_retention_loss__avg   = {str_the_list(after__length_retention_loss__avg    , 3)}")
                    print(f"aft__no_abs__length_retention_loss= {str_the_list(after__no_abs__length_retention_loss , 3)}")
                    #print(f"dim        = {str_the_list(dim_list, 0, ",    ")}")
                    
                    pass#for outter_param_set
                
                pass#for random_generator_style
            pass#/ test
        
        
        
        
        
        return
                            
    ____test____full_test_version_of_length_correction__by_row____basic()
    pass
        
        











