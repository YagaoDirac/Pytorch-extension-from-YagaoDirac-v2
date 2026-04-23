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


if "test" and False:
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
        
        #rough result. length protection helps protect length, but only in one direction. 
        # if protect the row, the col still score poorly.
        # so the length_loss can be halfed. But the retention_loss is besically the same. Maybe 10% better.
        # but very little effect on length_retention( in matmul)
        if "rand dummy v2   protected 1 step." and False:
                # when noise strength is 0, it's a perfect orthogonal matrix. All measure will be 0.
            if True:
                
                # noise_strength 0.7     dim 10
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.06710,  0.04227,  0.04167,  0.04954,  0.06577,  0.08943]
                # aft__len_loss__avg                = [ 0.04053,  0.02858,  0.02617,  0.02790,  0.04204,  0.05887]
                # aft__no_abs_len_loss              = [ 0.00138, -0.00325, -0.00493, -0.00544, -0.00922, -0.01277]
                # aft__length_retention_loss__max   = [ 0.08098,  0.07266,  0.07415,  0.07541,  0.07511,  0.08175]
                # aft__length_retention_loss__avg   = [ 0.05643,  0.05323,  0.05362,  0.05262,  0.05366,  0.05966]
                # aft__no_abs__length_retention_loss= [-0.00233, -0.00886, -0.00974, -0.01103, -0.01254, -0.01487]
                # 2.457835 , or 0.024578 per test
                # noise_strength 0.9     dim 10
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.07935,  0.05172,  0.05005,  0.06018,  0.07469,  0.10100]
                # aft__len_loss__avg                = [ 0.04760,  0.03267,  0.02936,  0.03288,  0.04723,  0.06731]
                # aft__no_abs_len_loss              = [ 0.00331, -0.00436, -0.00623, -0.00735, -0.01222, -0.01792]
                # aft__length_retention_loss__max   = [ 0.09084,  0.08209,  0.08183,  0.08971,  0.09137,  0.09724]
                # aft__length_retention_loss__avg   = [ 0.06274,  0.06135,  0.06050,  0.06069,  0.06192,  0.06823]
                # aft__no_abs__length_retention_loss= [-0.00116, -0.01166, -0.01345, -0.01467, -0.01660, -0.01780]
                # 2.453181 , or 0.024532 per test
                # noise_strength 1.3     dim 10
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.07692,  0.06530,  0.05283,  0.06230,  0.07838,  0.11794]
                # aft__len_loss__avg                = [ 0.05361,  0.03842,  0.03216,  0.03928,  0.05286,  0.07463]
                # aft__no_abs_len_loss              = [ 0.00210, -0.00634, -0.00721, -0.01016, -0.01230, -0.01688]
                # aft__length_retention_loss__max   = [ 0.09522,  0.10334,  0.09078,  0.09271,  0.09429,  0.11439]
                # aft__length_retention_loss__avg   = [ 0.06880,  0.06913,  0.06705,  0.06806,  0.06927,  0.07533]
                # aft__no_abs__length_retention_loss= [-0.00453, -0.01568, -0.01605, -0.01756, -0.01720, -0.01910]
                # dim 100   test_time 50    device cpu
                # 8.020616 , or 0.160412 per test
                # noise_strength 0.7     dim 100
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.01394,  0.01072,  0.00959,  0.01072,  0.01490,  0.02011]
                # aft__len_loss__avg                = [ 0.01298,  0.00922,  0.00836,  0.00931,  0.01317,  0.01821]
                # aft__no_abs_len_loss              = [ 0.00014, -0.00041, -0.00051, -0.00064, -0.00094, -0.00134]
                # aft__length_retention_loss__max   = [ 0.02236,  0.02269,  0.02197,  0.02154,  0.02050,  0.02144]
                # aft__length_retention_loss__avg   = [ 0.01791,  0.01825,  0.01804,  0.01799,  0.01813,  0.01854]
                # aft__no_abs__length_retention_loss= [-0.00037, -0.00095, -0.00151, -0.00145, -0.00086, -0.00092]
                # 7.929951 , or 0.158599 per test
                # noise_strength 0.9     dim 100
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.01707,  0.01212,  0.01100,  0.01218,  0.01690,  0.02478]
                # aft__len_loss__avg                = [ 0.01477,  0.01057,  0.00968,  0.01060,  0.01482,  0.02060]
                # aft__no_abs_len_loss              = [ 0.00070, -0.00053, -0.00067, -0.00078, -0.00125, -0.00124]
                # aft__length_retention_loss__max   = [ 0.02264,  0.02350,  0.02307,  0.02330,  0.02360,  0.02299]
                # aft__length_retention_loss__avg   = [ 0.02020,  0.02021,  0.01990,  0.02025,  0.02062,  0.02051]
                # aft__no_abs__length_retention_loss= [ 0.00055, -0.00112, -0.00054, -0.00144, -0.00193, -0.00123]
                # 7.898305 , or 0.157966 per test
                # noise_strength 1.3     dim 100
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.01818,  0.01406,  0.01286,  0.01383,  0.01909,  0.02701]
                # aft__len_loss__avg                = [ 0.01667,  0.01191,  0.01097,  0.01233,  0.01686,  0.02323]
                # aft__no_abs_len_loss              = [ 0.00047, -0.00064, -0.00087, -0.00109, -0.00165, -0.00169]
                # aft__length_retention_loss__max   = [ 0.02636,  0.02774,  0.02637,  0.02622,  0.02690,  0.02665]
                # aft__length_retention_loss__avg   = [ 0.02247,  0.02272,  0.02240,  0.02274,  0.02259,  0.02284]
                # aft__no_abs__length_retention_loss= [-0.00014, -0.00219, -0.00216, -0.00173, -0.00256, -0.00195]
                # dim 1000   test_time 10    device cpu
                # 15.929535 , or 1.592954 per test
                # noise_strength 0.7     dim 1000
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.00423,  0.00301,  0.00272,  0.00304,  0.00423,  0.00598]
                # aft__len_loss__avg                = [ 0.00412,  0.00292,  0.00264,  0.00294,  0.00416,  0.00574]
                # aft__no_abs_len_loss              = [ 0.00001, -0.00005, -0.00005, -0.00007, -0.00009, -0.00005]
                # aft__length_retention_loss__max   = [ 0.00642,  0.00629,  0.00626,  0.00657,  0.00656,  0.00644]
                # aft__length_retention_loss__avg   = [ 0.00587,  0.00570,  0.00573,  0.00577,  0.00595,  0.00587]
                # aft__no_abs__length_retention_loss= [-0.00014, -0.00027,  0.00051,  0.00018, -0.00063,  0.00019]
                # 16.098578 , or 1.609858 per test
                # noise_strength 0.9     dim 1000
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.00475,  0.00355,  0.00313,  0.00349,  0.00480,  0.00661]
                # aft__len_loss__avg                = [ 0.00463,  0.00338,  0.00304,  0.00338,  0.00472,  0.00642]
                # aft__no_abs_len_loss              = [ 0.00004, -0.00004, -0.00007, -0.00009, -0.00010, -0.00012]
                # aft__length_retention_loss__max   = [ 0.00731,  0.00773,  0.00736,  0.00714,  0.00705,  0.00727]
                # aft__length_retention_loss__avg   = [ 0.00644,  0.00633,  0.00672,  0.00638,  0.00629,  0.00640]
                # aft__no_abs__length_retention_loss= [-0.00014, -0.00007, -0.00004, -0.00002, -0.00043, -0.00014]
                # 16.137738 , or 1.613774 per test
                # noise_strength 1.3     dim 1000
                # power_me_to_protect_length       = [-0.500,   -0.100,    0.001,    0.100,    0.500,    0.999]
                # aft__len_loss__max                = [ 0.00537,  0.00405,  0.00355,  0.00407,  0.00551,  0.00729]
                # aft__len_loss__avg                = [ 0.00528,  0.00391,  0.00348,  0.00386,  0.00534,  0.00718]
                # aft__no_abs_len_loss              = [ 0.00010, -0.00006, -0.00009, -0.00010, -0.00017,  0.00002]
                # aft__length_retention_loss__max   = [ 0.00879,  0.00771,  0.00794,  0.00807,  0.00813,  0.00803]
                # aft__length_retention_loss__avg   = [ 0.00728,  0.00708,  0.00718,  0.00716,  0.00719,  0.00739]
                # aft__no_abs__length_retention_loss= [-0.00022, -0.00006, -0.00033, -0.00034, -0.00007,  0.00001]
                
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
                
                #noise_strength_list = [ 0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.30,]
                noise_strength_list = [ 0.70,    0.90,    1.30,]
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
                        # _raw_result__ori__len_loss                      = torch.empty(size=[number_of_tests])
                        # _raw_result__ori__no_abs_len_loss               = torch.empty(size=[number_of_tests])
                        # _raw_result__ori__length_retention_loss         = torch.empty(size=[number_of_tests])
                        # _raw_result__ori__no_abs__length_retention_loss = torch.empty(size=[number_of_tests])
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
                            # len_loss, no_abs_len_loss, length_retention_loss, \
                            #         no_abs__length_retention_loss = full_length_info_test__in_log10(mat)
                            # _raw_result__ori__len_loss                     [ii__test] = len_loss                     
                            # _raw_result__ori__no_abs_len_loss              [ii__test] = no_abs_len_loss              
                            # _raw_result__ori__length_retention_loss        [ii__test] = length_retention_loss        
                            # _raw_result__ori__no_abs__length_retention_loss[ii__test] = no_abs__length_retention_loss
                            
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
                    print(f"power_me_to_protect_length       = {str_the_list(raw_power_me_to_protect_length_list  , 3, ",   ")}")
                    print(f"aft__len_loss__max                = {str_the_list(after__len_loss__max                 , 5)}")
                    print(f"aft__len_loss__avg                = {str_the_list(after__len_loss__avg                 , 5)}")
                    print(f"aft__no_abs_len_loss              = {str_the_list(after__no_abs_len_loss               , 5)}")
                    print(f"aft__length_retention_loss__max   = {str_the_list(after__length_retention_loss__max    , 5)}")
                    print(f"aft__length_retention_loss__avg   = {str_the_list(after__length_retention_loss__avg    , 5)}")
                    print(f"aft__no_abs__length_retention_loss= {str_the_list(after__no_abs__length_retention_loss , 5)}")
                    #print(f"dim        = {str_the_list(dim_list, 0, ",    ")}")
                    
                    pass#for outter_param_set
                
                pass#for random_generator_style
            pass#/ test
        
        
        
        
        return 
    ____test____full_test_version_of_length_correction__by_row____basic()
    
    pass

if "similar test from the angle part2.  Scan the hyper param" and True:
    def ____test____full_test_version_of_length_correction__by_row____scan_the_hyperparam():
        # the style means "r", "rr", "rc", "rrr", "rrc". 
        # result is manually extracted form visualization.
        # it's a rough test.
        if "style doesn't matter......a rough test. To build up intuition." and True:
            #result
            # r,0.1 == (r,0.316)x2
            
            
            style_list = ["r", "rr", "rc", "rcr", "rCr" ]
            for style in style_list:
                raw_power_me_to_protect_length = torch.tensor(0.1)
                dim = 100
                init_pow = 1.
                
                steps = 5
                iota_of_dim = iota(dim)
                
                device = 'cpu'
                if dim>10001:
                    device = 'cuda'
                    pass
                
                test_time = 20###########################################################
                
                print(test_time)
                _raw_result__len_loss                     = torch.empty(size=[test_time, steps])
                _raw_result__no_abs__len_loss             = torch.empty(size=[test_time, steps])
                _raw_result__length_retention_loss        = torch.empty(size=[test_time, steps])
                _raw_result__no_abs__length_retention_loss= torch.empty(size=[test_time, steps])
                
                for ii_test_count in range(test_time):
                    
                    #<  init
                    mat = torch.randn(size=[dim,dim])/math.sqrt(dim)*math.pow(10., init_pow)
                    len_loss__in_the_beginning, no_abs__len_loss__in_the_beginning, length_retention_loss__in_the_beginning, \
                            no_abs__length_retention_loss__in_the_beginning = full_length_info_test__in_log10(mat)
                    
                    _raw_result__len_loss                     [ii_test_count, 0] = len_loss__in_the_beginning
                    _raw_result__no_abs__len_loss             [ii_test_count, 0] = no_abs__len_loss__in_the_beginning
                    _raw_result__length_retention_loss        [ii_test_count, 0] = length_retention_loss__in_the_beginning
                    _raw_result__no_abs__length_retention_loss[ii_test_count, 0] = no_abs__length_retention_loss__in_the_beginning
                    
                    #<  calc
                    for ii_step_count in range(1, steps):
                        #----------------#----------------#----------------
                        #only use one of them.
                        if style == "r":
                            mat = full_test_version_of_length_correction__by_row(mat,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length, iota_of_dim=iota_of_dim)
                            pass
                        elif style == "rr":
                            mat = full_test_version_of_length_correction__by_row(mat,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(0.5), iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_length_correction__by_row(mat,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(0.5), iota_of_dim=iota_of_dim)
                            pass
                        elif style == "rc":
                            mat = full_test_version_of_length_correction__by_row(mat,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(0.5), iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_length_correction__by_row(mat.T,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(0.5), iota_of_dim=iota_of_dim).T
                            pass
                        elif style == "rcr":
                            mat = full_test_version_of_length_correction__by_row(mat,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(1./3.), iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_length_correction__by_row(mat.T,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(1./3.), iota_of_dim=iota_of_dim).T
                            mat = full_test_version_of_length_correction__by_row(mat,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(1./3.), iota_of_dim=iota_of_dim)
                            pass
                        elif style == "rCr":
                            mat = full_test_version_of_length_correction__by_row(mat,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(0.25), iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_length_correction__by_row(mat.T,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(0.5), iota_of_dim=iota_of_dim).T
                            mat = full_test_version_of_length_correction__by_row(mat,
                                    raw_power_me_to_protect_length = raw_power_me_to_protect_length.pow(0.25), iota_of_dim=iota_of_dim)
                            pass
                        else:
                            assert False, "bad param: style"
                        #----------------#----------------#----------------
                        
                        #<  measure
                        len_loss, no_abs__len_loss, length_retention_loss, no_abs__length_retention_loss = \
                                                    full_length_info_test__in_log10(mat)
                        
                        _raw_result__len_loss                     [ii_test_count, ii_step_count] = len_loss
                        _raw_result__no_abs__len_loss             [ii_test_count, ii_step_count] = no_abs__len_loss
                        _raw_result__length_retention_loss        [ii_test_count, ii_step_count] = length_retention_loss
                        _raw_result__no_abs__length_retention_loss[ii_test_count, ii_step_count] = no_abs__length_retention_loss
                        
                                #angle_loss__in_the_beginning - angle_loss__of_this_step
                        # _raw_result__step__score_incr [_test_count, _step_count] = \
                        #         angle_loss__last_step        - angle_loss__of_this_step
                        
                        #tail
                        #angle_loss__last_step = angle_loss__of_this_step
                        
                        pass#for _step_count
                    
                    pass#for _test_count
                    
                plot_me__len_loss                      = _raw_result__len_loss                     .mean(dim=0)
                plot_me__no_abs__len_loss              = _raw_result__no_abs__len_loss             .mean(dim=0)
                plot_me__length_retention_loss         = _raw_result__length_retention_loss        .mean(dim=0)
                plot_me__no_abs__length_retention_loss = _raw_result__no_abs__length_retention_loss.mean(dim=0)
                assert plot_me__len_loss.shape.__len__() == 1
                assert plot_me__len_loss.shape[0] == steps
                
                x_axis = torch.linspace(0, steps-1, steps )
                from matplotlib import pyplot as plt
                plt.plot(x_axis[1:], plot_me__len_loss                     [1:], "b--,", label='len_loss                     ')#, x_axis, step__score_incr)
                plt.plot(x_axis[1:], plot_me__no_abs__len_loss             [1:], "r--,", label='no_abs__len_loss             ')#, x_axis, step__score_incr)
                plt.plot(x_axis[1:], plot_me__length_retention_loss        [1:], "b-." , label='length_retention_loss        ')#, x_axis, step__score_incr)
                plt.plot(x_axis[1:], plot_me__no_abs__length_retention_loss[1:], "r-." , label='no_abs__length_retention_loss')#, x_axis, step__score_incr)
                plt.plot(x_axis[1:], torch.zeros_like(x_axis[1:]), 'k-')#, x_axis, step__score_incr)
                #plt.title(f"safe_f {safe_factor}     {style}      power_me {raw_power_me_to_protect_length:.2f}")
                plt.title(f"{style}      power_me {raw_power_me_to_protect_length:.2f}")
                plt.legend()
                plt.show()
                
                pass#for style
            
            pass#/ test
        
        
        1w 看看数值。感觉都不用看了。
        后面看看保护长度和角度，对另外一个指标有没有影响。
        然后就去综合搜了。
        
        
        # if "style test, r, rc, rcr." and True:
            
            
        #     print(f"{_line_()}        style test, r, rc, rcr.")
        
        #     #------------------#------------------#------------------
        #     dim_list =                          [ 10, 100, 1000]
        #     number_of_tests_list = torch.tensor([100,  50,  10])
        #     number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
        #     for outter_param_set in range(dim_list.__len__()):
        #         dim = dim_list[outter_param_set]
        #         iota_of_dim = iota(dim)
        #         number_of_tests = number_of_tests_list[outter_param_set]
        #         device = 'cpu'
        #         if dim>10001:
        #             device = 'cuda'
        #             pass
        #         print(f"dim {dim}   test_time {number_of_tests}    device {device}")
        #     #------------------#------------------#------------------
        #         #some other ref
        #         #                         noise_strength_list = torch.tensor(
        #         #                             [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.10,    1.30,    1.80,     ])
        #         #                         angle_loss_list__full = torch.tensor([
        #         #                             [ 0.0000,  0.2184,  0.4341,  0.6253,  0.7969,  0.9537,  1.0867,  1.1974,  1.3483,  1.4409,  1.5145,  1.5496,   ],#dim 10
        #         #                             [ 0.0000,  0.2234,  0.4377,  0.6346,  0.8075,  0.9575,  1.0838,  1.1826,  1.3292,  1.4228,  1.4859,  1.5540,   ],#dim 100
        #         #                             [ 0.0000,  0.2238,  0.4381,  0.6346,  0.8080,  0.9577,  1.0820,  1.1828,  1.3303,  1.4248,  1.4814,  1.5506,   ],])# dim 1000
                
        #         1w
        #         for kk in range(noise_strength_list.__len__()):
        #             noise_strength = noise_strength_list[kk]
            
        #             raw_power_me_to_protect_length_list = [-0.5, -0.1, 0.001, 0.1, 0.5, 0.999]
        #             #_size = raw_power_me_to_protect_length_list.__len__()

        #             # ori__len_loss__max                  = torch.empty(size=[_size])#don't modify this
        #             # ori__len_loss__avg                  = torch.empty(size=[_size])#don't modify this
        #             # ori__no_abs_len_loss                = torch.empty(size=[_size])#don't modify this
                    
        #             # ori__length_retention_loss__max     = torch.empty(size=[_size])#don't modify this
        #             # ori__length_retention_loss__avg     = torch.empty(size=[_size])#don't modify this
        #             # ori__no_abs__length_retention_loss  = torch.empty(size=[_size])#don't modify this
        #             #del _size
                    
        #             after__len_loss__max                = []#don't modify this
        #             after__len_loss__avg                = []#don't modify this
        #             after__no_abs_len_loss              = []#don't modify this
                    
        #             after__length_retention_loss__max   = []#don't modify this
        #             after__length_retention_loss__avg   = []#don't modify this
        #             after__no_abs__length_retention_loss= []#don't modify this
                    
        #             _when_start = time.perf_counter()
                    
        #             for jj_x_axis in range(raw_power_me_to_protect_length_list.__len__()):# x axis
        #                 raw_power_me_to_protect_length =  raw_power_me_to_protect_length_list[jj_x_axis]
                        
        #                 #y axis
        #                 # _raw_result__ori__len_loss                      = torch.empty(size=[number_of_tests])
        #                 # _raw_result__ori__no_abs_len_loss               = torch.empty(size=[number_of_tests])
        #                 # _raw_result__ori__length_retention_loss         = torch.empty(size=[number_of_tests])
        #                 # _raw_result__ori__no_abs__length_retention_loss = torch.empty(size=[number_of_tests])
        #                 _raw_result__after__len_loss                      = torch.empty(size=[number_of_tests])
        #                 _raw_result__after__no_abs_len_loss               = torch.empty(size=[number_of_tests])
        #                 _raw_result__after__length_retention_loss         = torch.empty(size=[number_of_tests])
        #                 _raw_result__after__no_abs__length_retention_loss = torch.empty(size=[number_of_tests])
                        
                        
        #                 for ii__test in range(number_of_tests):
                            
        #                     #------------------#------------------#------------------
        #                     #<  init                orthogonal mat.
        #                     mat = random_dummy_mat__v2(dim=dim, noise_strength = noise_strength,
        #                             div_sqrt_1_plus_ns_sqr = True, device=device, iota_of_dim = iota_of_dim)
                            
        #                     #<  perfect should score 0.
        #                     # len_loss, no_abs_len_loss, length_retention_loss, \
        #                     #         no_abs__length_retention_loss = full_length_info_test__in_log10(mat)
        #                     # _raw_result__ori__len_loss                     [ii__test] = len_loss                     
        #                     # _raw_result__ori__no_abs_len_loss              [ii__test] = no_abs_len_loss              
        #                     # _raw_result__ori__length_retention_loss        [ii__test] = length_retention_loss        
        #                     # _raw_result__ori__no_abs__length_retention_loss[ii__test] = no_abs__length_retention_loss
                            
        #                     #<  calc
        #                     mat = full_test_version_of_length_correction__by_row(mat, \
        #                             raw_power_me_to_protect_length = raw_power_me_to_protect_length, iota_of_dim = iota_of_dim)
                            
        #                     #<  measure
        #                     len_loss, no_abs_len_loss, length_retention_loss, \
        #                             no_abs__length_retention_loss = full_length_info_test__in_log10(mat)
        #                     _raw_result__after__len_loss                     [ii__test] = len_loss                     
        #                     _raw_result__after__no_abs_len_loss              [ii__test] = no_abs_len_loss              
        #                     _raw_result__after__length_retention_loss        [ii__test] = length_retention_loss        
        #                     _raw_result__after__no_abs__length_retention_loss[ii__test] = no_abs__length_retention_loss
                            
        #                     pass#for ii__test
                        
                        
        #                 # ori__len_loss__max                  [jj_x_axis] = _raw_result__ori__len_loss       .max()
        #                 # ori__len_loss__avg                  [jj_x_axis] = _raw_result__ori__len_loss       .mean()
        #                 # ori__no_abs_len_loss                [jj_x_axis] = _raw_result__ori__no_abs_len_loss.mean()

        #                 # ori__length_retention_loss__max     [jj_x_axis] = _raw_result__ori__length_retention_loss        .max()
        #                 # ori__length_retention_loss__avg     [jj_x_axis] = _raw_result__ori__length_retention_loss        .mean()
        #                 # ori__no_abs__length_retention_loss  [jj_x_axis] = _raw_result__ori__no_abs__length_retention_loss.mean()

        #                 after__len_loss__max                .append(_raw_result__after__len_loss       .max())
        #                 after__len_loss__avg                .append(_raw_result__after__len_loss       .mean())
        #                 after__no_abs_len_loss              .append(_raw_result__after__no_abs_len_loss.mean())

        #                 after__length_retention_loss__max   .append(_raw_result__after__length_retention_loss        .max())
        #                 after__length_retention_loss__avg   .append(_raw_result__after__length_retention_loss        .mean())
        #                 after__no_abs__length_retention_loss.append(_raw_result__after__no_abs__length_retention_loss.mean())
                        
        #                 pass#for power_me_to_protect_length
        #             _when_end = time.perf_counter()
                    
        #             print(f"{_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
        #             print(f"noise_strength {noise_strength}     dim {dim}")
        #             # print(f"ori__len_loss__max                = {str_the_list(ori__len_loss__max                  , 3)}")
        #             # print(f"ori__len_loss__avg                = {str_the_list(ori__len_loss__avg                  , 3)}")
        #             # print(f"ori__no_abs_len_loss              = {str_the_list(ori__no_abs_len_loss                , 3)}")
        #             # print(f"ori__length_retention_loss__max   = {str_the_list(ori__length_retention_loss__max     , 3)}")
        #             # print(f"ori__length_retention_loss__avg   = {str_the_list(ori__length_retention_loss__avg     , 3)}")
        #             # print(f"ori__no_abs__length_retention_loss= {str_the_list(ori__no_abs__length_retention_loss  , 3)}")
        #             print(f"power_me_to_protect_length       = {str_the_list(raw_power_me_to_protect_length_list  , 3, ",   ")}")
        #             print(f"aft__len_loss__max                = {str_the_list(after__len_loss__max                 , 5)}")
        #             print(f"aft__len_loss__avg                = {str_the_list(after__len_loss__avg                 , 5)}")
        #             print(f"aft__no_abs_len_loss              = {str_the_list(after__no_abs_len_loss               , 5)}")
        #             print(f"aft__length_retention_loss__max   = {str_the_list(after__length_retention_loss__max    , 5)}")
        #             print(f"aft__length_retention_loss__avg   = {str_the_list(after__length_retention_loss__avg    , 5)}")
        #             print(f"aft__no_abs__length_retention_loss= {str_the_list(after__no_abs__length_retention_loss , 5)}")
        #             #print(f"dim        = {str_the_list(dim_list, 0, ",    ")}")
                    
        #             pass#for outter_param_set
                
        #         pass#for random_generator_style
        #     pass#/ test
        
        
        
        
        
        return
    ____test____full_test_version_of_length_correction__by_row____scan_the_hyperparam()                            
    pass
        
        











