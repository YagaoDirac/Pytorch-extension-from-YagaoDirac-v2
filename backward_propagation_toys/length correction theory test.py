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
        device = 'cuda'
        pass
    else:
        device = 'cpu'
        pass
    return device





def random_dummy_mat__v2(dim:int, noise_strength:float,
                    device='cpu', iota_of_dim:torch.Tensor|None = None)->torch.Tensor:
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
    return mat
#copied from the angle test part 2.py. Already tested behavior there. No basic behavior test here.



if "measure the random gen algo":
    def ____test____measure_the_random_gen_algo():
        
        if " measure the random gen algo" and True:
            if True:
                # dim = 10
                # noise_strength_list          = [ 0.0,     0.1,     0.2,     0.3,     0.4,     0.5,     0.6,     0.7,     0.9,     1.1,     1.3,     1.8]
                # len_score__max            = [ 0.0000,  0.0329,  0.0690,  0.1160,  0.1521,  0.2122,  0.2301,  0.2934,  0.3697,  0.4441,  0.5467,  0.7482]
                # len_score__avg            = [ 0.0000,  0.0223,  0.0439,  0.0698,  0.0918,  0.1214,  0.1504,  0.1791,  0.2452,  0.3199,  0.3994,  0.5848]
                # len_score__no_abs__min    = [-0.0000, -0.0190, -0.0390, -0.0423, -0.0492, -0.0284, -0.0166, -0.0173,  0.0389,  0.1766,  0.2092,  0.4366]
                # len_score__no_abs__max    = [ 0.0000,  0.0243,  0.0561,  0.1000,  0.1449,  0.2122,  0.2118,  0.2934,  0.3597,  0.4405,  0.5467,  0.7482]
                # len_score__no_abs__avg    = [-0.0000,  0.0033,  0.0104,  0.0321,  0.0521,  0.0821,  0.1158,  0.1474,  0.2235,  0.3085,  0.3928,  0.5833]
                # len_retention__max        = [ 0.0000,  0.0158,  0.0332,  0.0534,  0.0713,  0.1059,  0.1154,  0.1459,  0.1904,  0.2443,  0.2888,  0.3728]
                # len_retention__avg        = [ 0.0000,  0.0109,  0.0220,  0.0345,  0.0455,  0.0604,  0.0756,  0.0900,  0.1234,  0.1609,  0.1984,  0.2933]
                # len_retention__no_abs__min= [-0.0000, -0.0122, -0.0236, -0.0208, -0.0175, -0.0140, -0.0105, -0.0164,  0.0131,  0.0704,  0.1132,  0.2016]
                # len_retention__no_abs__max= [ 0.0000,  0.0126,  0.0294,  0.0471,  0.0713,  0.1033,  0.1108,  0.1418,  0.1851,  0.2443,  0.2876,  0.3728]
                # len_retention__no_abs__avg= [-0.0000,  0.0017,  0.0054,  0.0165,  0.0254,  0.0410,  0.0582,  0.0736,  0.1120,  0.1545,  0.1948,  0.2929]
                # dim = 100
                # noise_strength_list          = [ 0.0,     0.1,     0.2,     0.3,     0.4,     0.5,     0.6,     0.7,     0.9,     1.1,     1.3,     1.8]
                # len_score__max            = [ 0.0000,  0.0089,  0.0229,  0.0447,  0.0716,  0.1058,  0.1435,  0.1853,  0.2712,  0.3528,  0.4422,  0.6364]
                # len_score__avg            = [ 0.0000,  0.0077,  0.0197,  0.0382,  0.0635,  0.0950,  0.1318,  0.1699,  0.2547,  0.3411,  0.4267,  0.6237]
                # len_score__no_abs__min    = [-0.0000,  0.0010,  0.0131,  0.0281,  0.0565,  0.0826,  0.1184,  0.1589,  0.2441,  0.3295,  0.4154,  0.6049]
                # len_score__no_abs__max    = [ 0.0000,  0.0065,  0.0206,  0.0446,  0.0715,  0.1057,  0.1435,  0.1853,  0.2712,  0.3528,  0.4422,  0.6364]
                # len_score__no_abs__avg    = [-0.0000,  0.0041,  0.0168,  0.0366,  0.0629,  0.0948,  0.1317,  0.1699,  0.2547,  0.3411,  0.4267,  0.6237]
                # len_retention__max        = [ 0.0000,  0.0047,  0.0123,  0.0220,  0.0364,  0.0531,  0.0739,  0.0928,  0.1383,  0.1793,  0.2229,  0.3199]
                # len_retention__avg        = [ 0.0000,  0.0038,  0.0099,  0.0192,  0.0319,  0.0474,  0.0661,  0.0850,  0.1276,  0.1706,  0.2131,  0.3112]
                # len_retention__no_abs__min= [-0.0000,  0.0002,  0.0063,  0.0120,  0.0272,  0.0411,  0.0596,  0.0775,  0.1193,  0.1609,  0.2028,  0.2980]
                # len_retention__no_abs__max= [ 0.0000,  0.0035,  0.0111,  0.0216,  0.0364,  0.0530,  0.0739,  0.0928,  0.1383,  0.1793,  0.2229,  0.3199]
                # len_retention__no_abs__avg= [-0.0000,  0.0020,  0.0084,  0.0184,  0.0316,  0.0473,  0.0660,  0.0850,  0.1276,  0.1706,  0.2131,  0.3112]
                # dim = 1000
                # noise_strength_list          = [ 0.0,     0.1,     0.2,     0.3,     0.4,     0.5,     0.6,     0.7,     0.9,     1.1,     1.3,     1.8]
                # len_score__max            = [ 0.0000,  0.0047,  0.0172,  0.0379,  0.0648,  0.0976,  0.1345,  0.1739,  0.2582,  0.3445,  0.4303,  0.6277]
                # len_score__avg            = [ 0.0000,  0.0045,  0.0169,  0.0374,  0.0644,  0.0968,  0.1333,  0.1730,  0.2573,  0.3440,  0.4294,  0.6270]
                # len_score__no_abs__min    = [ 0.0000,  0.0042,  0.0167,  0.0370,  0.0639,  0.0963,  0.1324,  0.1722,  0.2565,  0.3431,  0.4285,  0.6265]
                # len_score__no_abs__max    = [ 0.0000,  0.0045,  0.0172,  0.0379,  0.0648,  0.0976,  0.1345,  0.1739,  0.2582,  0.3445,  0.4303,  0.6277]
                # len_score__no_abs__avg    = [ 0.0000,  0.0044,  0.0169,  0.0374,  0.0644,  0.0968,  0.1333,  0.1730,  0.2573,  0.3440,  0.4294,  0.6270]
                # len_retention__max        = [ 0.0000,  0.0026,  0.0088,  0.0196,  0.0332,  0.0499,  0.0685,  0.0880,  0.1303,  0.1727,  0.2161,  0.3153]
                # len_retention__avg        = [ 0.0000,  0.0022,  0.0085,  0.0187,  0.0321,  0.0485,  0.0664,  0.0865,  0.1285,  0.1721,  0.2146,  0.3133]
                # len_retention__no_abs__min= [ 0.0000,  0.0017,  0.0081,  0.0182,  0.0315,  0.0477,  0.0651,  0.0855,  0.1271,  0.1704,  0.2135,  0.3117]
                # len_retention__no_abs__max= [ 0.0000,  0.0025,  0.0088,  0.0196,  0.0332,  0.0499,  0.0685,  0.0880,  0.1303,  0.1727,  0.2161,  0.3153]
                # len_retention__no_abs__avg= [ 0.0000,  0.0021,  0.0085,  0.0187,  0.0321,  0.0485,  0.0664,  0.0865,  0.1285,  0.1721,  0.2146,  0.3133]
                
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
                        ori_mat = random_dummy_mat__v2(dim=dim, noise_strength=noise_strength,
                                                            device=device, iota_of_dim=iota_of_dim)
                        
                        #<  measure the init>
                        len_loss, _, _log = LOSS__mat_is_standard_orthogonal(ori_mat, _debug__needs_log = True)
                        _raw_result__len_loss[_test_count] = len_loss#################################
                        assert _log[0][0] == "sum of two len_score__raw_mean_without_abs"
                        _raw_result__no_abs__len_loss[_test_count] = _log[0][1]###############################
                        
                        length_retention_loss, (no_abs__length_retention_score,) = LOSS__vec_len_retention__of_a_mat_in_matmul(ori_mat)
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
                print(f"noise_strength_list          = {str_the_list(noise_strength_list, 1, segment=",    ")}")
                print(f"len_score__max            = {str_the_list(len_score__max                  , 4)}")
                print(f"len_score__avg            = {str_the_list(len_score__avg                  , 4)}")
                print(f"len_score__no_abs__min    = {str_the_list(len_score__no_abs__min          , 4)}")
                print(f"len_score__no_abs__max    = {str_the_list(len_score__no_abs__max          , 4)}")
                print(f"len_score__no_abs__avg    = {str_the_list(len_score__no_abs__avg          , 4)}")
                print(f"len_retention__max        = {str_the_list(len_retention_score__max        , 4)}")
                print(f"len_retention__avg        = {str_the_list(len_retention_score__avg        , 4)}")
                print(f"len_retention__no_abs__min= {str_the_list(len_retention_score__no_abs__min, 4)}")
                print(f"len_retention__no_abs__max= {str_the_list(len_retention_score__no_abs__max, 4)}")
                print(f"len_retention__no_abs__avg= {str_the_list(len_retention_score__no_abs__avg, 4)}")
                
                pass#for outter_iter_count
            
            pass#/ test
        
        
        1w 今天 4 17的。角度里面有一个除以最大长度。重新看看分布，限制一下，感觉1.1就够了，防止出现个别的超级无敌大的。
        就是自适应里面。
        
        return 
    
    ____test____measure_the_random_gen_algo()
    pass







def full_test_version_of_length_correction__by_row(input:torch.Tensor, 
                        expansion:float, iota_of_dim:torch.Tensor|None = None, 
                        )->torch.Tensor:
    
    # if cap_to is None:
    #     cap_to__s = calc__cap_to____ver_1(input.shape[0], expansion_factor__s)
    #     pass
    if isinstance(expansion, float):
        expansion__s = torch.tensor(expansion)
        pass
    elif isinstance(expansion, torch.Tensor):
        expansion__s = expansion.detach().clone()
        pass
    else:
        assert False, "bad type."
        pass
    assert isinstance(expansion__s, torch.Tensor)
    
    if iota_of_dim is None:
        iota_of_dim = iota(input.shape[0])
        pass
    
    with torch.no_grad():
        #assert False, "is it mean or sum  vvv  here ????"
        normalized_row_vec, length_of_row_vec__n = get_full_info_of_vector_length__2d(input)
        max_length_of_input = length_of_row_vec__n.max().values()
        length_of_row_vec__over_max_len__n = length_of_row_vec__n.div(max_length_of_input)
        target_length__n = length_of_row_vec__over_max_len__n.pow(expansion__s)
        target_length__n_1EXPANDn = expand_vec_to_matrix(target_length__n,each_element_to='row')
        result = normalized_row_vec.mul(target_length__n_1EXPANDn)
        
        pass#no grad
    return result

基本行为测试











