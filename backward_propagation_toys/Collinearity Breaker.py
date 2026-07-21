from typing import Literal
import datetime
from pathlib import Path
import math, random
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal, \
    iota, is_square_matrix, \
    vector_length_norm, get_vector_length, expand_vec_to_matrix,\
    log10_avg_safe, log10_avg__how_similar, get_mask_of_top_element__rough,\
    str_the_list, str_the_list__probability
        
from pytorch_yagaodirac_v2.Interpolation import \
    interpolation_of_list, interpolation_of_list_2d, reverse_interpolation_of_list__list_must_sorted
    
from pytorch_yagaodirac_v2.ParamMo import GradientModification__mean_len_of_something_to_1
from pytorch_yagaodirac_v2.Random import random_standard_vector, randomly_permutate__matrix, randomly_rotate__matrix
from pytorch_yagaodirac_v2.measure_for_matrix import LOSS__behavior_similarity, LOSS__mat_is_standard_orthogonal, LOSS__vec_len_retention__of_a_mat_in_matmul
    



def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
import sys
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######








# a =  torch.tensor([
#     [False, False, False, False, ],
#     [False, False, False, False,  ],
#     [False, True, False, False,  ],
#     ], dtype = torch.uint8)
# b = a.argmax(keepdim=True)   #[[9]],,,, 
# fds=432



# a =  torch.tensor([
#     [False, False, False, False, ],
#     [False, False, False, False,  ],
#     [False, False, False, False,  ],
#     ], dtype = torch.uint8)
# b = a.argmax()
# index1 = torch.floor_divide(b, a.shape[1])
# index2 = b - index1*a.shape[1]
# #int8 and uint8 both work.
# fds=432


# a =  torch.tensor([
#     [False, False, False, False, ],
#     [False, False, False, False,  ],
#     [False, False, False, False,  ],
#     ])
# b = a.any()
# fds=432








def colinearity_break____pure_random____no_batch(input:torch.Tensor, threshold:float|torch.Tensor = 0.999, max_steps = 5,
                        #iota_of_dim:torch.Tensor|None = None,
                    input_already_standardized = False, transpose = False, Im_confident_about_the_dim = False, _debug__needs_log = False) \
        ->tuple[torch.Tensor, list[tuple[str, torch.Tensor|None]]|None]:
    '''By row. If you need by colomn, set transpose to True.
    
    this tool modifies later(with greater index number) vectors, to break the colinearity.
    '''
    if transpose:
        return colinearity_break____pure_random____no_batch(input, threshold=threshold, input_already_standardized=input_already_standardized, transpose=False)
        
    if isinstance(threshold, float):
        threshold = torch.tensor(threshold, device = input.device, dtype = input.dtype)
        pass
    assert threshold > 0.9 and threshold < 1.

    assert input.shape.__len__() == 2
    
    '''if you have 3 of 1d vectors, they are always colinear. When the number of vectors is greater than the dim, 
    Im not sure about the behavior. But if you know what you are doing, feel free to do anything here.'''
    if not Im_confident_about_the_dim:
        assert input.shape[0] <= input.shape[1]
        pass

    # if iota_of_dim is None:
    #     iota_of_dim = iota(input.shape[0], device=input.device)
    #     pass

    if input_already_standardized:
        mat = input
        pass
    else:
        mat = vector_length_norm(input)
        pass

    _log:list[tuple[str, torch.Tensor|None]]|None = None
    if _debug__needs_log:
        _log = []
        pass

    #<  real payload
    _log_only__with_extra_round = True
    for ii_iter in range(max_steps):
        mat_mat_T = mat@(mat.T)#diag is removed.
        mat_mat_T = mat_mat_T.triu(diagonal=1)#diag is removed.
        #mat_mat_T[iota_of_dim, iota_of_dim] = 0.
        mat_mat_T = mat_mat_T.abs()
        bad_flag = mat_mat_T.gt(threshold)

        if _log is not None:
            _log.append((f"bad_flag of step {ii_iter}", bad_flag.to(torch.int8)))#.detach().clone()))
            pass
        
        if not bad_flag.any():
            _log_only__with_extra_round = False
            break

        while bad_flag.any():
            bad_flag__in_int = bad_flag.to(torch.uint8)#int8 also works.
            position_in_1_number = bad_flag__in_int.argmax()
            index_host = torch.floor_divide(position_in_1_number, bad_flag__in_int.shape[1])
            index_guest = position_in_1_number - index_host*bad_flag__in_int.shape[1]
            assert index_host<index_guest #debug code. 
            #<  random the vector.
            bad_flag[index_host, index_guest] = False
            del index_host
            mat[index_guest] = random_standard_vector(bad_flag__in_int.shape[1], dtype=mat.dtype, device=mat.device)
            if _log is not None:
                _log.append((f"vector {index_guest} randomed", None))
                pass
            #tail of infi loop.
            bad_flag[index_guest, index_guest+1:].fill_(False)
            pass# while bad_flag.any()
        pass # for ii_iter
    if _log is not None:
        if _log_only__with_extra_round:
            _log.append((f"{ii_iter+1} round(s) in total", None))
            pass
        else:
            _log.append((f"{ii_iter} round(s) in total", None))
            pass
        pass#if _log
    return mat, _log

if "test" and __DEBUG_ME__() and True:
    def ____test____basic_behavior____colinearity_break____pure_random____no_batch():
        if "no need to touch anything case" and False:
            input = torch.tensor([  [1., 0], 
                                    [0., 1], ])
            output, _log = colinearity_break____pure_random____no_batch(input, _debug__needs_log = True)
            assert _tensor_equal(input, output)# nothing is touched.
            assert _log.__len__() == 2         # nothing is touched.

            for dim in [2,3,5,10,100]:
                input = torch.eye(n=dim)
                output, _log = colinearity_break____pure_random____no_batch(input, _debug__needs_log = True)
                assert _tensor_equal(input, output)# nothing is touched.
                assert _log.__len__() == 2         # nothing is touched.
                pass
            pass#/ test

        if "a manually read case" and True:
            '''
            this case shows what the following line in the function does.
            >>> #tail of infi loop.
            >>> bad_flag[index_guest, index_guest+1:].fill_(False)
            '''

            import math
            input = vector_length_norm(torch.tensor([
                [0.,   0,   0,1],
                [math.cos(0.), math.sin(0.),   0,0],
                [math.cos(0.1), math.sin(0.1),   0,0],
                [math.cos(0.2), math.sin(0.2),   0,0],
                ]))
            output, _log = colinearity_break____pure_random____no_batch(input, 
                        threshold=0.99, _debug__needs_log = True)
            assert _log[0][0] == "bad_flag of step 0"
            assert _tensor_equal(_log[0][1], torch.tensor([ [0, 0, 0, 0],
                                                            [0, 0, 1, 0],
                                                            [0, 0, 0, 1],
                                                            [0, 0, 0, 0]]))
            assert _log[-1][0] == "1 round(s) in total"#may be unstalbe. But likely.
            pass#/ test

        if "repeating vectors. The previous one is untouched." and True:
            input = torch.tensor([  [1., 0], 
                                    [1., 0], ])
            output, _log = colinearity_break____pure_random____no_batch(input, _debug__needs_log = True)
            assert _tensor_equal(input[0], output[0])#the first vector is untouched.
            #read the log manually. it should be a flag a random a flag.
            pass
            
            for _ in range(33):
                input = vector_length_norm(torch.randn(size=[2,2]))
                input[1] = input[0]
                output, _log = colinearity_break____pure_random____no_batch(input)
                assert _tensor_equal(input[0], output[0])#the first vector is untouched.
                pass
            pass#/ test
            
        if "repeating vectors. The worst case. does it work?" and True:
            for dim in [2,3,5,10,100]:
                for _ in range(11):
                    input = vector_length_norm(torch.randn(size=[dim,dim]))
                    for ii in range(1, dim):
                        input[ii] = input[0]
                        pass
                    output, _ = colinearity_break____pure_random____no_batch(input)

                    #<  assert
                    output_2, _ = colinearity_break____pure_random____no_batch(output)
                    assert _tensor_equal(output, output_2)
                    pass# for _
                pass# for dim
            pass#/ test

        return
    ____test____basic_behavior____colinearity_break____pure_random____no_batch()

扫一下次数和阈值。


    pass












assert False,"准备放到向量工具里面去"
