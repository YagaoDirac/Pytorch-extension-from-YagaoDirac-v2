from typing import Literal
import datetime
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
from pytorch_yagaodirac_v2.measure_for_matrix import LOSS__behavior_similarity, LOSS__mat_is_standard_orthogonal, LOSS__vec_len_retention__of_a_mat_in_matmul

from pytorch_yagaodirac_v2.measure_for_matrix import LOSS__behavior_similarity, \
    LOSS__mat_is_standard_orthogonal, LOSS__vec_len_retention__of_a_mat_in_matmul

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
                        power_me_to_protect_length:float|torch.Tensor, iota_of_dim:torch.Tensor|None = None, 
                        )->torch.Tensor:
    
    # if cap_to is None:
    #     cap_to__s = calc__cap_to____ver_1(input.shape[0], expansion_factor__s)
    #     pass
    if isinstance(power_me_to_protect_length, float):
        power_me_to_protect_length__s = torch.tensor(power_me_to_protect_length)
        pass
    elif isinstance(power_me_to_protect_length, torch.Tensor):
        power_me_to_protect_length__s = power_me_to_protect_length.detach().clone()
        pass
    else:
        assert False, "bad type."
        pass
    assert isinstance(power_me_to_protect_length__s, torch.Tensor)
    assert power_me_to_protect_length__s<0.
    
    if iota_of_dim is None:
        iota_of_dim = iota(input.shape[0])
        pass
    
    with torch.no_grad():
        normalized_row_vec, length_of_row_vec__n = get_full_info_of_vector_length__2d(input)
        target_length__n = length_of_row_vec__n.pow(power_me_to_protect_length__s)
        target_length__n_1EXPANDn = expand_vec_to_matrix(target_length__n,each_element_to='row')
        result = normalized_row_vec.mul(target_length__n_1EXPANDn)
        pass#no grad
    return result


if "test" and True:
    def ____test____full_test_version_of_length_correction__by_row____basic():
        if "it doesn't touch perfect matrix":
            for dim in [2,10,100,1000]:
                iota_of_dim = iota(dim)
                for _ in range(11):
                    #<  init
                    mat = torch.eye(n=dim)
                    mat = randomly_rotate__matrix(mat)
                    #<  calc
                    mat = full_test_version_of_length_correction__by_row(mat)
                    
        
        
        
基本行为测试











