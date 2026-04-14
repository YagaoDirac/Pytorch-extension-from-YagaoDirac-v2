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





'''
a bit readme.

The formula is very clear. 

manually__mul_me = 4./(dim*dim)
manually__mat_matmul_mat__d_d = mat@(mat.T)
manually__mat_matmul_mat__d_d[iota_of_dim, iota_of_dim] = 0.
manually__grad = manually__mat_matmul_mat__d_d@mat
manually__grad *= manually__mul_me

The maximum possible value for grad is 4*(dim-1)/(dim*dim) or something similar.
But the actual cap value is 1.41/(dim)  *  4*(dim-1)/(dim*dim).( or not 1.41, but 2.82???)
When dim is big, the training dynamics is not useful.

Then the visualization shows that if the init_mat is random, except for length of 
row vectors are all scaled to 1, the length of the row vectors of grad are very similar.
Most of them are 0.9 to 1.1 times the average length.

To correct this, and also to add some adaptive expansion, I wrote the 2nd visualization.

'''

from pytorch_yagaodirac_v2.measure_for_matrix import LOSS__behavior_similarity, \
    LOSS__mat_is_standard_orthogonal, LOSS__vec_len_retention__of_a_mat_in_matmul







def ____test____basic_order_of_magnitude_test():
    
    "0.58 to 0.82   when accumulating_time == dim-1"    
    if "random len1 vec dot and accululate. It's similar to forward pass." and False:
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.]
        # cpu
        # dim == 2 
        # sum__min           = [ 0.01500,  0.00665,  0.01969,  0.02072,  0.53518]
        # sum__max           = [ 0.99994,  3.52393,  5.98439,  16.85637,  59.49195]
        # sum__avg           = [ 0.62658,  1.16668,  1.63911,  5.41536,  17.60757]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.]
        # dim == 5              
        # sum__min           = [ 0.00500,  0.01848,  0.00316,  0.07038,  0.12127]
        # sum__max           = [ 0.92337,  2.45363,  3.40919,  11.74703,  33.48606]
        # sum__avg           = [ 0.37021,  0.69509,  1.00356,  3.53499,  10.81130]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.]
        # dim == 10              
        # sum__min           = [ 0.01107,  0.01215,  0.01691,  0.03488,  0.17180]
        # sum__max           = [ 0.73128,  1.53533,  2.58554,  9.77455,  27.93412]
        # sum__avg           = [ 0.26163,  0.49004,  0.70206,  2.53179,  8.72839]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.]
        # dim == 100              
        # sum__min           = [ 0.00100,  0.00047,  0.00655,  0.00050,  0.00964]
        # sum__max           = [ 0.26407,  0.41101,  0.80717,  2.95427,  10.22717]
        # sum__avg           = [ 0.06868,  0.14782,  0.24610,  0.80938,  2.47465]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.]
        # dim == 1000              
        # sum__min           = [ 0.00001,  0.00026,  0.00012,  0.00048,  0.01112]
        # sum__max           = [ 0.09448,  0.22957,  0.29213,  0.75334,  2.77561]
        # sum__avg           = [ 0.02458,  0.05065,  0.07587,  0.23639,  0.75746]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.]
        
        # sum__min               = [ 0.01500, 0.01848,   0.01691,  0.00050, 0.01112]
        # sum__max               = [ 0.99994, 2.45363,   2.58554,  2.95427, 2.77561]
        # sum__avg               = [ 0.62658, 0.69509,   0.70206,  0.80938, 0.75746]
        # accumulate_time and also dim     2, 5.0000,   10.0000,  100.0000, 1000.]
        
        
        # cuda
        # dim == 2             
        # sum__min           = [ 0.02648,  0.00699,  0.12668,  0.10067,  0.01249]
        # sum__max           = [ 0.99999,  3.18379,  5.16254,  20.20586,  61.41320]
        # sum__avg           = [ 0.59455,  1.17904,  1.75762,  5.61126,  19.14854]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.0000]
        # dim == 5             
        # sum__min           = [ 0.00786,  0.00116,  0.01273,  0.08277,  0.01728]
        # sum__max           = [ 0.92342,  1.68872,  3.39152,  10.45560,  39.85177]
        # sum__avg           = [ 0.40593,  0.58058,  1.01392,  3.39034,  11.03279]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.0000]
        # dim == 10            
        # sum__min           = [ 0.00073,  0.00403,  0.00656,  0.01735,  0.00390]
        # sum__max           = [ 0.81276,  1.76747,  2.90984,  9.46086,  29.39728]
        # sum__avg           = [ 0.27530,  0.58429,  0.80673,  2.61582,  6.80672]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.0000]
        # dim == 100           
        # sum__min           = [ 0.00043,  0.00081,  0.00018,  0.00079,  0.02054]
        # sum__max           = [ 0.25698,  0.60732,  1.09038,  2.84871,  7.70734]
        # sum__avg           = [ 0.07961,  0.16004,  0.25928,  0.85063,  2.36571]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.0000]
        # dim == 1000          
        # sum__min           = [ 0.00011,  0.00039,  0.00049,  0.00388,  0.01125]
        # sum__max           = [ 0.09817,  0.17504,  0.24787,  0.82352,  3.88339]
        # sum__avg           = [ 0.02563,  0.04981,  0.07104,  0.26357,  0.81579]
        # accumulate_time_list= [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.0000]
        
        # sum__min               = [ 0.02648, 0.00116,   0.00656,  0.00079, 0.01125]
        # sum__max               = [ 0.99999, 1.68872,   2.90984,  2.84871, 3.88339]
        # sum__avg               = [ 0.59455, 0.58058,   0.80673,  0.85063, 0.81579]
        # accumulate_time and also dim     2, 5.0000,   10.0000,  100.0000, 1000.]
        
        
        #--------------------#--------------------#--------------------
        device = 'cuda'
        dim_list =        [ 2,  5, 10,100,1000]
        test_time_list = [100,100,100,100, 100]
        for outter_iter_count in range(dim_list.__len__()):
            dim = dim_list[outter_iter_count]
            test_time = test_time_list[outter_iter_count]
            print(f"dim {dim}   test_time {test_time}  {device}")
        #--------------------#--------------------#--------------------
            
            sum__min = []#dont modify this
            sum__max = []#dont modify this
            sum__avg = []#dont modify this
            
            start_time = datetime.datetime.now()
            
            #--------------------#--------------------#--------------------
            accumulate_time_list = [2,  5,  10,100,1000]
            for inner_iter_count in range(accumulate_time_list.__len__()):
                accumulate_time = accumulate_time_list[inner_iter_count] -1
            #--------------------#--------------------#--------------------
                
                _raw_result = torch.empty(size=[test_time])#dont modify this
                for test_count in range(test_time):
                    
                    #--------------------#--------------------#--------------------
                    host = random_standard_vector(dim=dim)
                    _sum = 0.
                    for accumulate_count in range(accumulate_time):
                        _sum += host.dot(random_standard_vector(dim=dim))
                        pass
                    #--------------------#--------------------#--------------------
                    
                    _raw_result[test_count] = _sum
                    pass#for accumulate_count
                sum__min.append(_raw_result.abs().min() )
                sum__max.append(_raw_result.abs().max() )
                sum__avg.append(_raw_result.abs().mean())
                pass# for macro_iter_count
            
            end_time = datetime.datetime.now()
            delta_time = end_time-start_time
            print(f"dim == {dim}              total_time {delta_time}   {delta_time/test_time}/per test")    
            print(f"sum__min           = {str_the_list(sum__min, 5)}")    
            print(f"sum__max           = {str_the_list(sum__max, 5)}")    
            print(f"sum__avg           = {str_the_list(sum__avg, 5)}")    
            print(f"accumulate_time_list= {str_the_list(accumulate_time_list, 4)}")
            pass#for outter_iter_count
        pass#/test
    
    # if all the row vec are len_1(generated by some func I wrote),
    # if the dot_product of row i and row j is DotP_ij
    # the formula for the grad of a row(grad_row_i), affected by row_j is
    # [[[    DotP_ij*(4/(dim*dim))*row_j    ]]]\
    # DotP_ij is between -1 and 1.
    # the row_j is also a vec (len_1), 
    # so the length of grad_row_i(from j) is 0 to 4/(dim*dim)
    # And in total, there are n-1 rows that provides grad.
    # so the grad_row_i is between 0 and 4(dim-1)/(dim*dim), if dim is big, then 4/dim
    if "formula test" and False:
        #<  test infra>
        for dim in [2,3,5,10,100,1000]:
            iota_of_dim = iota(dim)
            _1000000_ref_vec = torch.zeros(size=[dim])
            _1000000_ref_vec[0] = 1.
            #prin(_1000000_ref_vec)
            #<  init
            mat = torch.zeros(size=[dim,dim])
            mat[:,0] = 1.
            assert mat[0].eq(_1000000_ref_vec).all()
            mat.requires_grad_()
            
            #<  neural net infra>
            train_it = [mat]
            optim = torch.optim.SGD(params=train_it, lr=0.1)
            loss_func = torch.nn.MSELoss()#mse, otherwise the formula for grad is a bit weird.

            #<  forward
            should_be_eye = mat@(mat.T)#.T is on the right
            should_be_eye[iota_of_dim, iota_of_dim] = 0.#this op also cuts the grad chain partly.
            loss = loss_func(should_be_eye, torch.zeros_like(should_be_eye))
            
            #<  backward
            optim.zero_grad()
            loss.backward(inputs = train_it)
            
            #<  assertion>
            assert _tensor_equal(mat.grad[0], _1000000_ref_vec*4*(dim-1)/(dim*dim), epsilon=1e-7)
            
            pass#for dim
        
        del mat
        pass#/ test
    
    if "len=1 vec, what is the grad*(dim*dim)/(4*(dim-1))" and False:
        # result
        # length__min         = [ 0.00021,  0.08963,  0.07170,  0.01312,  0.00141]
        # length__max         = [ 1.00000,  0.67096,  0.27833,  0.01517,  0.00142]
        # length__avg         = [ 0.63007,  0.26032,  0.13508,  0.01406,  0.00141]
        # log10_of_length__min= [-3.68059, -1.04756, -1.14450, -1.88192, -2.85189]
        # log10_of_length__max= [ 0.00000, -0.17330, -0.55544, -1.81894, -2.84737]
        # log10_of_length__avg= [-0.30635, -0.60295, -0.87554, -1.85211, -2.84978]
        # dim_list             = [ 2.0000,  5.0000,  10.0000,  100.0000,  1000.0000]
        #log10 is -1.*log10(dim)+0.15
        
        length__min          = []#dont modify this
        length__max          = []#dont modify this
        length__avg          = []#dont modify this
        log10_of_length__min = []#dont modify this
        log10_of_length__max = []#dont modify this
        log10_of_length__avg = []#dont modify this
        
        #--------------------#--------------------#--------------------
        device = 'cpu'
        dim_list =        [ 2,  5, 10,100,1000]
        test_time_list = [10000,5000,5000,5000, 100]
        for outter_iter_count in range(dim_list.__len__()):
            dim = dim_list[outter_iter_count]
            _div_me = 4*(dim-1)/(dim*dim)
            iota_of_dim = iota(dim)
            test_time = test_time_list[outter_iter_count]
            print(f"dim {dim}   test_time {test_time}  {device}")
        #--------------------#--------------------#--------------------
            
            _raw_result_of__scaled_avg_grad_len = torch.empty(size=[test_time])#dont modify this
            _raw_result_of__log10__scaled_avg_grad_len = torch.empty(size=[test_time])#dont modify this
            for test_count in range(test_time):
                
                #--------------------#--------------------#--------------------
                mat = torch.randn(size=[dim,dim], device=device)
                mat = vector_length_norm(mat)
                # old code mat = ____init_mat_with_row_len_is_1(dim)
                
                #<  neural net infra>
                train_it = [mat]
                optim = torch.optim.SGD(params=train_it, lr=0.1)
                loss_func = torch.nn.MSELoss()#mse, otherwise the formula for grad is a bit weird.
                
                #<  forward
                should_be_eye = mat@(mat.T)#.T is on the right
                should_be_eye[iota_of_dim, iota_of_dim] = 0.#this op also cuts the grad chain partly.
                loss = loss_func(should_be_eye, torch.zeros_like(should_be_eye))
                
                #<  backward
                optim.zero_grad()
                loss.backward(inputs = train_it)
                
                #<  measure>
                grad_len_sqr = mat.grad.mul(mat.grad).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
                avg_grad_len = grad_len_sqr.sqrt().mean()
                scaled_avg_grad_len = avg_grad_len/_div_me
                #--------------------#--------------------#--------------------
                
                _raw_result_of__scaled_avg_grad_len[test_count] = scaled_avg_grad_len
                _raw_result_of__log10__scaled_avg_grad_len[test_count] = scaled_avg_grad_len.log10()
                pass#for accumulate_count
            
            length__min         .append(_raw_result_of__scaled_avg_grad_len.min() )
            length__max         .append(_raw_result_of__scaled_avg_grad_len.max() )
            length__avg         .append(_raw_result_of__scaled_avg_grad_len.mean() )
            log10_of_length__min.append(_raw_result_of__log10__scaled_avg_grad_len.min() )
            log10_of_length__max.append(_raw_result_of__log10__scaled_avg_grad_len.max() )
            log10_of_length__avg.append(_raw_result_of__log10__scaled_avg_grad_len.mean() )
            pass# for macro_iter_count
        
        print(f"length__min         = {str_the_list(length__min,          5)}")    
        print(f"length__max         = {str_the_list(length__max,          5)}")    
        print(f"length__avg         = {str_the_list(length__avg,          5)}")    
        print(f"log10_of_length__min= {str_the_list(log10_of_length__min, 5)}")    
        print(f"log10_of_length__max= {str_the_list(log10_of_length__max, 5)}")    
        print(f"log10_of_length__avg= {str_the_list(log10_of_length__avg, 5)}")    
        print(f"dim_list             = {str_the_list(dim_list, 4)}")
        
        pass#/test
    
    if "to replace the error propagation style" and False:
        "assertion only test. No print"
        #device??
        #-----------------#-----------------#-----------------
        dim_list = [2,10,100]
        test_time_list = [100,100,100]
        for outter_param_list in range(dim_list.__len__()):
            dim = dim_list[outter_param_list]
            test_time = test_time_list[outter_param_list]
            iota_of_dim = iota(dim)
            manually__mul_me = 4./(dim*dim)
        #-----------------#-----------------#-----------------
            
            for test_count in range(test_time):
                
                #-----------------#-----------------#-----------------
                #mat = torch.randn(size=[dim,dim], device=device)
                mat = vector_length_norm(mat)
                # old code mat = ____init_mat_with_row_len_is_1(dim)
                
                #<  neural net infra>
                train_it = [mat]
                optim = torch.optim.SGD(params=train_it, lr=0.1)
                loss_func = torch.nn.MSELoss()#mse, otherwise the formula for grad is a bit weird.
                
                #<  forward
                should_be_eye = mat@(mat.T)#.T is on the right
                should_be_eye[iota_of_dim, iota_of_dim] = 0.#this op also cuts the grad chain partly.
                loss = loss_func(should_be_eye, torch.zeros_like(should_be_eye))
                
                #<  backward
                optim.zero_grad()
                loss.backward(inputs = train_it)
                
                #<  manually repeat this process>
                with torch.no_grad():
                    manually__mat_matmul_mat__d_d = mat@(mat.T)
                    manually__mat_matmul_mat__d_d[iota_of_dim, iota_of_dim] = 0.
                    manually__grad = manually__mat_matmul_mat__d_d@mat
                    manually__grad *= manually__mul_me
                    pass
                
                #<  are they the same??>
                assert _tensor_equal(mat.grad, manually__grad)
                #-----------------#-----------------#-----------------
                
                pass#for test_count
            
            pass#for outter_param
        
        pass#/ test
    
    if "visualization of origin" and False:
        dim = 1000
        _div_me = 4*(dim-1)/(dim*dim)    *    (dim/1.42)    * 2.
        #    part 1111111111111111111    part 2222222222      ??
        #part 1 is from formula, part 2 is from tested distribution.
        # so the result should be around 1.
        iota_of_dim = iota(dim)
        for test_count in range(123123123):
            
            #--------------------#--------------------#--------------------
            mat = torch.randn(size=[dim,dim], device=device)
            mat = vector_length_norm(mat)
            # old code mat = ____init_mat_with_row_len_is_1(dim)
            
            #<  neural net infra>
            train_it = [mat]
            optim = torch.optim.SGD(params=train_it, lr=0.1)
            loss_func = torch.nn.MSELoss()#mse, otherwise the formula for grad is a bit weird.
            
            #<  forward
            should_be_eye = mat@(mat.T)#.T is on the right
            should_be_eye[iota_of_dim, iota_of_dim] = 0.#this op also cuts the grad chain partly.
            loss = loss_func(should_be_eye, torch.zeros_like(should_be_eye))
            
            #<  backward
            optim.zero_grad()
            loss.backward(inputs = train_it)
            
            #<  scale it a bit>
            grad_len_sqr = mat.grad.mul(mat.grad).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
            grad_len = grad_len_sqr.sqrt()
            scaled_grad_len = grad_len/_div_me
            
            #<  visualize>
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(tight_layout=True)
            n_bins = dim//20
            if n_bins<=3:
                n_bins = 3
                pass
            ax.hist(scaled_grad_len.tolist(), bins=n_bins)
            plt.show()
            
            pass#for test_count 
        
        pass#/ test
    
    if "visualization of adaptive expansion" and False:
        expansion_factor = 10.
        dim = 1000
        #_div_me = 4*(dim-1)/(dim*dim)    *    (dim/1.42)    * 2.
        #    part 1111111111111111111    part 2222222222      ??
        #part 1 is from formula, part 2 is from tested distribution.
        # so the result should be around 1.
        iota_of_dim = iota(dim)
        for test_count in range(123123123):
            
            #--------------------#--------------------#--------------------
            mat = torch.randn(size=[dim,dim], device=device)
            mat = vector_length_norm(mat)
            # old code mat = ____init_mat_with_row_len_is_1(dim)
            
            #<  neural net infra>
            train_it = [mat]
            optim = torch.optim.SGD(params=train_it, lr=0.1)
            loss_func = torch.nn.MSELoss()#mse, otherwise the formula for grad is a bit weird.
            
            #<  forward
            should_be_eye = mat@(mat.T)#.T is on the right
            should_be_eye[iota_of_dim, iota_of_dim] = 0.#this op also cuts the grad chain partly.
            loss = loss_func(should_be_eye, torch.zeros_like(should_be_eye))
            
            #<  backward
            optim.zero_grad()
            loss.backward(inputs = train_it)
            
            #<  scale it a bit>
            grad_len_sqr__dim = mat.grad.mul(mat.grad).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
            max_of__grad_len_sqr__s = grad_len_sqr__dim.max()
            scaled__grad_len_sqr__dim = grad_len_sqr__dim/max_of__grad_len_sqr__s
            assert scaled__grad_len_sqr__dim.le(1.).all()
            visualize_this = scaled__grad_len_sqr__dim.pow(expansion_factor/2.)
            #old code
            # grad_len = grad_len_sqr.sqrt()
            # scaled_grad_len = grad_len/_div_me
            
            #<  visualize>
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(tight_layout=True)
            n_bins = dim//20
            if n_bins<=3:
                n_bins = 3
                pass
            ax.hist(visualize_this.tolist(), bins=n_bins)
            plt.show()
            
            pass#for test_count 
        
        pass#/ test

#____test____basic_order_of_magnitude_test()


def ____test____correction_method_test():
    #this is a reference. Not a real test.
    # highlight is manually added. It's the best result from the later test.
    if "measure of random 1 to n rows in a standard orthogonal matrix." and False:
        #result
        # dim = 10
        # random_vec_count_list       = [ 1        2        3        5        7]
        # len_loss__opt_value     = [-0.03110, -0.05500, -0.06091, -0.07188, -0.07495]
        # len_loss__opt_prob      = [****    , XX      , XX      , XX      , XX      ]
        # angle_loss__opt_value   = [-0.45963, -0.75733, -0.97159, -1.28427, -1.46266]
        # angle_loss__opt_prob    = [   ********X      , XX      , XX      , XX      ]
        # length_retention_loss__v= [-0.02681, -0.04221, -0.05154, -0.06207, -0.06753]
        # length_retention_loss__p= [******* , XX      , XX      , XX      , XX      ]
        # ------------->>>>>>>>>>>>>
        # 30
        # dim = 100
        # random_vec_count_list       = [ 1        2        3        5        7        10        12        15        20        50]
        # len_loss__opt_value     = [-0.00410, -0.00782, -0.00996, -0.01504, -0.01998, -0.02504, -0.02771, -0.03144, -0.03566, -0.04837]
        # len_loss__opt_prob      = [XX      , XX      ,******** , XX      , XX      , XX      , XX      , XX      , XX      , XX      ]
        # angle_loss__opt_value   = [-0.09192, -0.15175, -0.20392, -0.28873, -0.36323, -0.46238, -0.52304, -0.61114, -0.73726, -1.27269]
        # angle_loss__opt_prob    = [XX      , XX      , XX      , XX      , XX      , XX      , XX  ********      , XX      , XX      ]
        # length_retention_loss__v= [-0.00256, -0.00430, -0.00541, -0.00729, -0.00883, -0.01027, -0.01116, -0.01248, -0.01463, -0.02081]
        # length_retention_loss__p= [XX      , XX      , XX      , XX      ,******** , XX      , XX      , XX      , XX      , XX      ]
        # ------------->>>>>>>>>>>>>
        # 10
        # dim = 1000
        # random_vec_count_list       = [ 1        2        3        5        7        10        12        15        20        50        100        150        200]
        # len_loss__opt_value     = [-0.00047, -0.00085, -0.00136, -0.00214, -0.00297, -0.00394, -0.00447, -0.00553, -0.00729, -0.01471, -0.02306, -0.03020, -0.03421]
        # len_loss__opt_prob      = [XX      , XX      , XX      , XX    **********  , XX      , XX      , XX      , XX      , XX      , XX      , XX      , XX      ]
        # angle_loss__opt_value   = [-0.02254, -0.03633, -0.04720, -0.06464, -0.08011, -0.09971, -0.11192, -0.12837, -0.15456, -0.28012, -0.44550, -0.58566, -0.71163]
        # angle_loss__opt_prob    = [XX      , XX      , XX      , XX      , XX      , XX      , XX      , XX      , XX      , XX      , XX   **********   , XX      ]
        # length_retention_loss__v= [-0.00030, -0.00042, -0.00055, -0.00072, -0.00089, -0.00104, -0.00120, -0.00129, -0.00157, -0.00248, -0.00338, -0.00421, -0.00445]
        # length_retention_loss__p= [XX      , XX      , XX      , XX      , XX      , XX      , XX      , XX      , XX      , XX  **********    , XX      , XX      ]
        
        
        #------------------#------------------#------------------
        # dim_list =        [ 2, 10,100,1000]
        # test_time_list = [100,100,100, 20]
        # test_time_list = [10,10,10, 2]
        
        dim_list =       [ 10, 100,1000]
        test_time_list = [ 50, 30, 10]
        #test_time_list = [ 5, 3, 1]
        for outter_iter_count in range(dim_list.__len__()):
            dim = dim_list[outter_iter_count]
            test_time = test_time_list[outter_iter_count]
            print(test_time)
            iota_of_dim = iota(dim)
            if dim>100:
                device = 'cuda'
                pass
            else:
                device = 'cpu'
                pass
        #------------------#------------------#------------------
        
            len_loss__opt_value              = []  # dont modity this
            len_loss__opt_prob               = []  # dont modity this
            angle_loss__opt_value            = []  # dont modity this
            angle_loss__opt_prob             = []  # dont modity this
            length_retention_loss__opt_value = []  # dont modity this
            length_retention_loss__opt_prob  = []  # dont modity this
            #neg_behavior_similar_loss            = []  # dont modity this
                
            #------------------#------------------#------------------
            random_vec_count_list__raw = [1,2,3,5,7,10,12,15,20,50,100,150,200]
            random_vec_count_list = []
            for vec_count in random_vec_count_list__raw:
                if vec_count < dim:
                    random_vec_count_list.append(vec_count)
                    pass
                pass
            del random_vec_count_list__raw
            
            for random_vec_count in random_vec_count_list:
                if random_vec_count>=dim:
                    continue
            #------------------#------------------#------------------
            
                _raw_result__len_loss__after_sub_before                 = torch.empty(size=[test_time])  # dont modity this
                _raw_result__angle_loss__after_sub_before               = torch.empty(size=[test_time])  # dont modity this
                _raw_result__length_retention_loss__after_sub_before    = torch.empty(size=[test_time])  # dont modity this
                #_raw_result__neg_behavior_similar_loss                  = torch.empty(size=[test_time])  # dont modity this
                
                for test_count in range(test_time):
                    #----------------#----------------#----------------#----------------
                    #<  init
                    mat = torch.eye(n=dim)
                    mat = randomly_rotate__matrix(mat)
                    
                    #<  measure the init>
                    before__len_loss, before__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                    before__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
                    
                    #<  random some rows.
                    for ii in range(random_vec_count):
                        mat[ii] = random_standard_vector(dim=dim)
                        pass
                    
                    #<  measure the protected.>
                    after__len_loss, after__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                    after__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
                    #----------------#----------------#----------------#----------------
                    
                    _raw_result__len_loss__after_sub_before[test_count] = before__len_loss - after__len_loss
                    _raw_result__angle_loss__after_sub_before[test_count] = before__angle_loss - after__angle_loss
                    _raw_result__length_retention_loss__after_sub_before[test_count] = \
                        before__length_retention_loss - after__length_retention_loss
                    #_raw_result__neg_behavior_similar_loss[test_count] = -LOSS__behavior_similarity(mat, ori_mat)
                    pass#for test_count
                
                len_loss__opt_value             .append(_raw_result__len_loss__after_sub_before.mean().item())
                len_loss__opt_prob              .append(_raw_result__len_loss__after_sub_before.gt(0.).sum().item()/test_time)
                angle_loss__opt_value           .append(_raw_result__angle_loss__after_sub_before.mean().item())
                angle_loss__opt_prob            .append(_raw_result__angle_loss__after_sub_before.gt(0.).sum().item()/test_time)
                length_retention_loss__opt_value.append(_raw_result__length_retention_loss__after_sub_before.mean().item())
                length_retention_loss__opt_prob .append(_raw_result__length_retention_loss__after_sub_before.gt(0.).sum().item()/test_time)
                #neg_behavior_similar_loss       .append(_raw_result__neg_behavior_similar_loss.mean().item())
                
                pass# for expansion_factor(y axis)
            
            print(f"dim = {dim}")
            # print(f"expansion_factor_list   = {str_the_list(expansion_factor_list, 2, segment=",    ")}")
            # print(f"cap_to = {cap_to}")
            print(f"random_vec_count_list       = {str_the_list(random_vec_count_list,0,segment="       ")}")
            print(f"len_loss__opt_value     = {str_the_list(len_loss__opt_value, 5)}")
            print(f"len_loss__opt_prob      = {str_the_list__probability(len_loss__opt_prob, 3, flag__offset_by50=True, flag__mul_2_after_offset=True)}")
            print(f"angle_loss__opt_value   = {str_the_list(angle_loss__opt_value, 5)}")
            print(f"angle_loss__opt_prob    = {str_the_list__probability(angle_loss__opt_prob, 3, flag__offset_by50=True, flag__mul_2_after_offset=True)}")
            print(f"length_retention_loss__v= {str_the_list(length_retention_loss__opt_value, 5)}")
            print(f"length_retention_loss__p= {str_the_list__probability(length_retention_loss__opt_prob, 3, 
                                                                                                flag__offset_by50=True, flag__mul_2_after_offset=True)}")
            #print(f"neg_behavior_similar_loss= {str_the_list(neg_behavior_similar_loss, 5)}")
            print("------------->>>>>>>>>>>>>")
            
            pass#for outter_iter_count
        del mat
        pass#/ test
    
    if "measure of random 1 to n rows in a standard orthogonal matrix.  behavior similarity." and False:
        #result
        
        # dim = 10
        # random_vec_count_list       = [ 1        2        3        5        7]
        # len_loss__opt_value     = [-0.03220, -0.04662, -0.05882, -0.07484, -0.07177]
        # angle_loss__opt_value   = [-0.46431, -0.74140, -0.95821, -1.25765, -1.43906]
        # length_retention_loss__v= [-0.02690, -0.04119, -0.05007, -0.06016, -0.06941]
        # neg_BS__loss__length    = [-0.02629, -0.04101, -0.05247, -0.06168, -0.06725]
        # neg_BS__loss__angle     = [-0.09518, -0.20794, -0.30665, -0.52239, -0.73191]
        # ------------->>>>>>>>>>>>>
        # 30
        # dim = 100
        # random_vec_count_list       = [ 1        2        3        5        7        10        12        15        20        50]
        # len_loss__opt_value     = [-0.00475, -0.00715, -0.00994, -0.01516, -0.01935, -0.02494, -0.02645, -0.03232, -0.03650, -0.04754]
        # angle_loss__opt_value   = [-0.09108, -0.15443, -0.20360, -0.29002, -0.36331, -0.46354, -0.52510, -0.60958, -0.73831, -1.27326]
        # length_retention_loss__v= [-0.00276, -0.00439, -0.00541, -0.00729, -0.00873, -0.01027, -0.01141, -0.01265, -0.01427, -0.02077]
        # neg_BS__loss__length    = [-0.00276, -0.00435, -0.00559, -0.00711, -0.00859, -0.01054, -0.01161, -0.01268, -0.01430, -0.02095]
        # neg_BS__loss__angle     = [-0.01012, -0.01995, -0.02966, -0.04941, -0.07074, -0.09921, -0.12095, -0.15158, -0.19889, -0.49991]
        # ------------->>>>>>>>>>>>>
        # 10
        # dim = 1000
        # random_vec_count_list       = [ 1        2        3        5        7        10        12        15        20        50        100        150        200]
        # len_loss__opt_value     = [-0.00048, -0.00089, -0.00130, -0.00223, -0.00296, -0.00401, -0.00477, -0.00540, -0.00725, -0.01482, -0.02343, -0.03010, -0.03428]
        # angle_loss__opt_value   = [-0.02216, -0.03612, -0.04746, -0.06517, -0.08007, -0.09981, -0.11192, -0.12866, -0.15416, -0.27940, -0.44542, -0.58688, -0.71061]
        # length_retention_loss__v= [-0.00030, -0.00045, -0.00055, -0.00070, -0.00089, -0.00109, -0.00116, -0.00129, -0.00157, -0.00248, -0.00346, -0.00405, -0.00446]
        # neg_BS__loss__length    = [-0.00028, -0.00043, -0.00055, -0.00073, -0.00088, -0.00107, -0.00117, -0.00132, -0.00152, -0.00242, -0.00337, -0.00407, -0.00461]
        # neg_BS__loss__angle     = [-0.00101, -0.00198, -0.00299, -0.00494, -0.00701, -0.00999, -0.01193, -0.01501, -0.01999, -0.04996, -0.10003, -0.15020, -0.19996]
        # ------------->>>>>>>>>>>>>
        
        # conclusion
        # the length loss is not linear.
        # the angle loss looks very linear... it's basically, n/dim
        
        
        #------------------#------------------#------------------
        # dim_list =        [ 2, 10,100,1000]
        # test_time_list = [100,100,100, 20]
        # test_time_list = [10,10,10, 2]
        
        dim_list =       [ 10, 100,1000]
        test_time_list = [ 50, 30, 10]
        #test_time_list = [ 5, 3, 1]
        for outter_iter_count in range(dim_list.__len__()):
            dim = dim_list[outter_iter_count]
            test_time = test_time_list[outter_iter_count]
            print(test_time)
            iota_of_dim = iota(dim)
            if dim>100:
                device = 'cuda'
                pass
            else:
                device = 'cpu'
                pass
        #------------------#------------------#------------------
        
            len_loss__opt_value              = []  # dont modity this
            angle_loss__opt_value            = []  # dont modity this
            length_retention_loss__opt_value = []  # dont modity this
            neg_BS__loss__length             = []  # dont modity this
            neg_BS__loss__angle              = []  # dont modity this
                
            #------------------#------------------#------------------
            random_vec_count_list__raw = [1,2,3,5,7,10,12,15,20,50,100,150,200]
            random_vec_count_list = []
            for vec_count in random_vec_count_list__raw:
                if vec_count < dim:
                    random_vec_count_list.append(vec_count)
                    pass
                pass
            del random_vec_count_list__raw
            
            for random_vec_count in random_vec_count_list:
                if random_vec_count>=dim:
                    continue
            #------------------#------------------#------------------

                _raw_result__len_loss__after_sub_before             = torch.empty(size=[test_time])  # dont modity this
                _raw_result__angle_loss__after_sub_before           = torch.empty(size=[test_time])  # dont modity this
                _raw_result__length_retention_loss__after_sub_before= torch.empty(size=[test_time])  # dont modity this
                _raw_result__neg_BS__loss__length                   = torch.empty(size=[test_time])  # dont modity this
                _raw_result__neg_BS__loss__angle                    = torch.empty(size=[test_time])  # dont modity this
                
                for test_count in range(test_time):
                    #----------------#----------------#----------------#----------------
                    #<  init
                    ori_mat = torch.eye(n=dim)
                    ori_mat = randomly_rotate__matrix(ori_mat   )
                    mat = ori_mat.detach().clone()
                    
                    #<  measure the init>
                    before__len_loss, before__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                    before__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
                    
                    #<  random some rows.
                    for ii in range(random_vec_count):
                        mat[ii] = random_standard_vector(dim=dim)
                        pass
                    
                    #<  measure the protected.>
                    after__len_loss, after__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                    after__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
                    
                    length__log_10__diff__abs_avg, angle_diff__avg,_ = LOSS__behavior_similarity(mat, ori_mat)
                    #----------------#----------------#----------------#----------------
                    
                    _raw_result__len_loss__after_sub_before[test_count] = before__len_loss - after__len_loss
                    _raw_result__angle_loss__after_sub_before[test_count] = before__angle_loss - after__angle_loss
                    _raw_result__length_retention_loss__after_sub_before[test_count] = \
                        before__length_retention_loss - after__length_retention_loss
                    _raw_result__neg_BS__loss__length[test_count] = -length__log_10__diff__abs_avg
                    _raw_result__neg_BS__loss__angle [test_count] = -angle_diff__avg
                    pass#for test_count
                
                len_loss__opt_value             .append(_raw_result__len_loss__after_sub_before             .mean().item())
                angle_loss__opt_value           .append(_raw_result__angle_loss__after_sub_before           .mean().item())
                length_retention_loss__opt_value.append(_raw_result__length_retention_loss__after_sub_before.mean().item())
                neg_BS__loss__length            .append(_raw_result__neg_BS__loss__length                   .mean().item())
                neg_BS__loss__angle             .append(_raw_result__neg_BS__loss__angle                    .mean().item())
                
                pass# for expansion_factor(y axis)
            
            print(f"dim = {dim}")
            # print(f"expansion_factor_list   = {str_the_list(expansion_factor_list, 2, segment=",    ")}")
            # print(f"cap_to = {cap_to}")
            print(f"random_vec_count_list         = {str_the_list(random_vec_count_list,0,segment="       ")}")
            print(f"len_loss__opt_value     = {str_the_list(len_loss__opt_value  , 5)}")
            print(f"angle_loss__opt_value   = {str_the_list(angle_loss__opt_value, 5)}")
            print(f"length_retention_loss__v= {str_the_list(length_retention_loss__opt_value, 5)}")
            print(f"neg_BS__loss__length    = {str_the_list(neg_BS__loss__length , 5)}")
            print(f"neg_BS__loss__angle     = {str_the_list(neg_BS__loss__angle  , 5)}")
            
            print("------------->>>>>>>>>>>>>")
            
            pass#for outter_iter_count
        del mat
        pass#/ test
    
    if "roughly scan 2 hyperparams." and False:
        # greater is better
        
        if "result of   expansion_factor -2 to 0" and False:
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   -2.0
            # len_loss__opt_value     = [ 0.00210,  0.00355,  0.00780,  0.01163,  0.01176,  0.01406,  0.01144,  0.00710,  0.00422,  0.00718, -0.00121,  0.00112]
            # len_loss__opt_prob      = [   0.920,    0.800,    0.840,    0.560,    0.520,    0.600,    0.480,    0.240,    0.120,    0.280,    0.040,    0.160]
            # angle_loss__opt_value   = [ 0.04472,  0.09412,  0.18088,  0.26151,  0.30328,  0.25637,  0.24347,  0.13511,  0.07418,  0.05273,  0.01339,  0.01331]
            # angle_loss__opt_prob    = [ v      ,  v      ,    0.920,    0.840,    0.880,    0.680,    0.680,    0.400,    0.160,    0.280,    0.160, XX 0.000]
            # length_retention_loss__v= [ 0.00235,  0.00429,  0.00549,  0.01235,  0.01397,  0.01391,  0.00929,  0.00782,  0.00245,  0.00639,  0.00052,  0.00206]
            # length_retention_loss__p= [   0.200,    0.320,    0.360,    0.640,    0.680,    0.680,    0.480,    0.440,    0.120,    0.440,    0.120,    0.040]
            # avg_abs_final_grad__min = [ 0.00347,  0.00762,  0.01831,  0.03965,  0.05913,  0.07577,  0.07821,  0.11449,  0.12042,  0.15299,  0.18555,  0.17476]
            # avg_abs_final_grad__max = [ 0.01518,  0.02419,  0.14391,  0.16137,  0.21709,  0.27190,  0.27019,  0.32727,  0.51019,  0.37109,  0.76498,  0.77428]
            # avg_abs_final_grad__avg = [ 0.00616,  0.01302,  0.03417,  0.07263,  0.09787,  0.12574,  0.14597,  0.19410,  0.23097,  0.24153,  0.31814,  0.31361]
            # ------------->>>>>>>>>>>>>                                           *****      ***
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.00128,  0.00233,  0.00561,  0.01110,  0.01673,  0.01833,  0.01738,  0.02247,  0.02000,  0.01508,  0.01408,  0.01163]
            # len_loss__opt_prob      = [ v      ,  v      ,    0.920,    0.920,    0.840,    0.840,    0.680,    0.800,    0.680,    0.520,    0.520,    0.320]
            # angle_loss__opt_value   = [ 0.02723,  0.05739,  0.13793,  0.26582,  0.38905,  0.43926,  0.44876,  0.43642,  0.38024,  0.32288,  0.28999,  0.17337]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.960,    0.960,  v      ,    0.800,    0.840,    0.600]
            # length_retention_loss__v= [ 0.00120,  0.00215,  0.00860,  0.01212,  0.01716,  0.01604,  0.02234,  0.02168,  0.01763,  0.01664,  0.01444,  0.00901]
            # length_retention_loss__p= [   0.120,    0.240,    0.560,    0.640,    0.880,    0.920,    0.960,  v      ,    0.760,    0.880,    0.800,    0.520]
            # avg_abs_final_grad__min = [ 0.00306,  0.00634,  0.01550,  0.02923,  0.04556,  0.05833,  0.07379,  0.08392,  0.10116,  0.11367,  0.13809,  0.15306]
            # avg_abs_final_grad__max = [ 0.00540,  0.01030,  0.02914,  0.05668,  0.07420,  0.12355,  0.12818,  0.16027,  0.15343,  0.20751,  0.24431,  0.24963]
            # avg_abs_final_grad__avg = [ 0.00380,  0.00764,  0.01957,  0.03858,  0.05680,  0.07880,  0.09714,  0.11728,  0.13181,  0.15821,  0.17365,  0.18711]
            # ------------->>>>>>>>>>>>>                                                               *****      ***
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.00083,  0.00166,  0.00412,  0.00852,  0.01258,  0.01490,  0.01796,  0.02198,  0.02339,  0.02401,  0.01989,  0.02208]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,    0.960,  v      ,    0.960,  v      ,    0.840,    0.920,    0.840,    0.640,    0.760]
            # angle_loss__opt_value   = [ 0.01821,  0.03961,  0.09884,  0.18600,  0.28396,  0.38079,  0.43410,  0.49693,  0.52991,  0.55681,  0.55353,  0.52760]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [-0.00003,  0.00404,  0.00428,  0.00859,  0.01204,  0.01596,  0.01885,  0.02191,  0.02330,  0.02547,  0.02512,  0.02595]
            # length_retention_loss__p= [   0.040,    0.360,    0.160,    0.560,    0.760,    0.920,    0.960,    0.960,    0.960,    0.960,  v      ,    0.960]
            # avg_abs_final_grad__min = [ 0.00238,  0.00445,  0.01191,  0.02272,  0.03443,  0.04273,  0.05659,  0.06942,  0.08399,  0.09691,  0.10246,  0.12023]
            # avg_abs_final_grad__max = [ 0.00276,  0.00556,  0.01385,  0.02814,  0.04185,  0.05486,  0.06952,  0.08215,  0.09609,  0.11005,  0.12712,  0.13901]
            # avg_abs_final_grad__avg = [ 0.00258,  0.00519,  0.01295,  0.02577,  0.03869,  0.05186,  0.06443,  0.07729,  0.09128,  0.10396,  0.11760,  0.12966]
            # ------------->>>>>>>>>>>>>                                                                                             *****                ***
            
            
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   -2.0
            # len_loss__opt_value     = [ 0.00059,  0.00109,  0.00290,  0.00555,  0.00706,  0.00854,  0.00822,  0.00720,  0.00682,  0.00588,  0.00538,  0.00454]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.933,  v      ,    0.933]
            # angle_loss__opt_value   = [ 0.03760,  0.07374,  0.18632,  0.35827,  0.48242,  0.54237,  0.53434,  0.48302,  0.44354,  0.40484,  0.36695,  0.31901]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00039,  0.00118,  0.00293,  0.00575,  0.00726,  0.00766,  0.00782,  0.00790,  0.00648,  0.00607,  0.00539,  0.00422]
            # length_retention_loss__p= [   0.200,    0.267,    0.600,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00117,  0.00226,  0.00559,  0.01246,  0.01769,  0.02356,  0.02999,  0.03597,  0.04048,  0.04708,  0.05348,  0.05782]
            # avg_abs_final_grad__max = [ 0.00179,  0.00318,  0.00800,  0.01690,  0.02511,  0.03333,  0.04162,  0.06175,  0.05808,  0.06477,  0.08061,  0.09089]
            # avg_abs_final_grad__avg = [ 0.00138,  0.00272,  0.00689,  0.01386,  0.02066,  0.02776,  0.03463,  0.04200,  0.04789,  0.05405,  0.06303,  0.06978]
            # ------------->>>>>>>>>>>>>                                                     *****                ***                   
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.00043,  0.00086,  0.00205,  0.00440,  0.00610,  0.00736,  0.00903,  0.00817,  0.00806,  0.00802,  0.00691,  0.00672]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.02877,  0.05722,  0.14215,  0.27791,  0.39652,  0.50011,  0.55020,  0.56068,  0.53973,  0.50830,  0.45808,  0.42941]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [-0.00006,  0.00045,  0.00124,  0.00454,  0.00683,  0.00816,  0.00835,  0.00886,  0.00863,  0.00795,  0.00673,  0.00646]
            # length_retention_loss__p= [   0.067, XX 0.000,    0.267,    0.933,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00094,  0.00194,  0.00496,  0.00979,  0.01441,  0.01973,  0.02420,  0.02931,  0.03406,  0.03872,  0.04338,  0.04834]
            # avg_abs_final_grad__max = [ 0.00123,  0.00236,  0.00575,  0.01206,  0.01652,  0.02375,  0.02852,  0.03471,  0.03833,  0.04537,  0.05077,  0.05967]
            # avg_abs_final_grad__avg = [ 0.00105,  0.00209,  0.00523,  0.01047,  0.01548,  0.02118,  0.02594,  0.03167,  0.03633,  0.04115,  0.04679,  0.05259]
            # ------------->>>>>>>>>>>>>                                                                ***      *****
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.00033,  0.00067,  0.00169,  0.00325,  0.00492,  0.00591,  0.00704,  0.00788,  0.00860,  0.00823,  0.00835,  0.00753]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.02196,  0.04377,  0.10993,  0.21656,  0.31783,  0.40984,  0.48202,  0.53416,  0.55988,  0.56357,  0.54405,  0.51577]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00003,  0.00006,  0.00210,  0.00299,  0.00416,  0.00636,  0.00715,  0.00753,  0.00925,  0.00771,  0.00811,  0.00779]
            # length_retention_loss__p= [   0.133, XX 0.000,    0.533,    0.867,    0.933,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00079,  0.00159,  0.00397,  0.00795,  0.01193,  0.01594,  0.01984,  0.02387,  0.02785,  0.03176,  0.03572,  0.03974]
            # avg_abs_final_grad__max = [ 0.00080,  0.00161,  0.00402,  0.00804,  0.01208,  0.01610,  0.02010,  0.02412,  0.02815,  0.03220,  0.03614,  0.04028]
            # avg_abs_final_grad__avg = [ 0.00080,  0.00160,  0.00400,  0.00800,  0.01199,  0.01601,  0.02000,  0.02398,  0.02799,  0.03198,  0.03597,  0.04003]
            # ------------->>>>>>>>>>>>>                                                                                   *****      ***
            
            
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   -2.0
            # len_loss__opt_value     = [ 0.00014,  0.00027,  0.00069,  0.00130,  0.00192,  0.00238,  0.00262,  0.00262,  0.00267,  0.00245,  0.00238,  0.00230]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.02828,  0.05669,  0.14051,  0.27117,  0.39126,  0.48780,  0.54552,  0.56359,  0.55227,  0.51691,  0.48737,  0.46871]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00028,  0.00008,  0.00104,  0.00091,  0.00196,  0.00278,  0.00289,  0.00253,  0.00214,  0.00310,  0.00217,  0.00222]
            # length_retention_loss__p= [XX 0.000,    0.200,    0.800,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00030,  0.00061,  0.00155,  0.00303,  0.00459,  0.00626,  0.00775,  0.00913,  0.01069,  0.01244,  0.01392,  0.01523]
            # avg_abs_final_grad__max = [ 0.00033,  0.00066,  0.00167,  0.00324,  0.00487,  0.00657,  0.00840,  0.01018,  0.01148,  0.01338,  0.01486,  0.01636]
            # avg_abs_final_grad__avg = [ 0.00032,  0.00064,  0.00159,  0.00313,  0.00472,  0.00636,  0.00795,  0.00964,  0.01099,  0.01283,  0.01432,  0.01579]
            # ------------->>>>>>>>>>>>>                                                                          ***       ***       ***
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.00013,  0.00024,  0.00061,  0.00117,  0.00173,  0.00228,  0.00244,  0.00265,  0.00277,  0.00271,  0.00242,  0.00236]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.02521,  0.05039,  0.12541,  0.24618,  0.35579,  0.45024,  0.51840,  0.55685,  0.56362,  0.54805,  0.51942,  0.49160]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [-0.00011, -0.00010,  0.00013,  0.00122,  0.00172,  0.00210,  0.00199,  0.00270,  0.00256,  0.00252,  0.00251,  0.00200]
            # length_retention_loss__p= [XX 0.000, XX-0.200,    0.400,    0.800,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00028,  0.00056,  0.00139,  0.00278,  0.00417,  0.00557,  0.00691,  0.00834,  0.00977,  0.01106,  0.01249,  0.01392]
            # avg_abs_final_grad__max = [ 0.00029,  0.00058,  0.00146,  0.00289,  0.00430,  0.00579,  0.00731,  0.00865,  0.01010,  0.01139,  0.01284,  0.01439]
            # avg_abs_final_grad__avg = [ 0.00028,  0.00057,  0.00141,  0.00282,  0.00422,  0.00565,  0.00706,  0.00848,  0.00990,  0.01128,  0.01268,  0.01411]
            # ------------->>>>>>>>>>>>>                                                                          ***      *****          
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.00011,  0.00022,  0.00055,  0.00108,  0.00161,  0.00197,  0.00239,  0.00264,  0.00275,  0.00276,  0.00272,  0.00258]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.02250,  0.04494,  0.11195,  0.22136,  0.32301,  0.41301,  0.48594,  0.53561,  0.56060,  0.56290,  0.54671,  0.52193]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [-0.00024,  0.00011,  0.00048,  0.00130,  0.00146,  0.00253,  0.00197,  0.00237,  0.00309,  0.00296,  0.00281,  0.00270]
            # length_retention_loss__p= [XX-0.400, XX 0.000,    0.800,    0.800,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00025,  0.00050,  0.00126,  0.00252,  0.00378,  0.00505,  0.00631,  0.00757,  0.00883,  0.01009,  0.01135,  0.01261]
            # avg_abs_final_grad__max = [ 0.00025,  0.00050,  0.00126,  0.00252,  0.00379,  0.00505,  0.00631,  0.00757,  0.00884,  0.01010,  0.01136,  0.01262]
            # avg_abs_final_grad__avg = [ 0.00025,  0.00050,  0.00126,  0.00252,  0.00379,  0.00505,  0.00631,  0.00757,  0.00883,  0.01009,  0.01136,  0.01262]
            # ------------->>>>>>>>>>>>>                                                                                    ***      *****
            pass
        
        # conclusions:
        # when dim is 1000, no bit difference.
        # when dim is 10 or 100, result of expansion_factor 0 is much better than -1 and -2.
        
        if "result of   expansion_factor -1 to 4" and False:
            
            # dim = 10
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,  
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.01788,  0.01995,  0.01573,  0.00905, 
            # len_loss__opt_prob      = [   0.720,    0.640,    0.480,    0.440, 
            # angle_loss__opt_value   = [ 0.41741,  0.37804,  0.28461,  0.18456, 
            # angle_loss__opt_prob    = [ v      ,    0.920,    0.920,    0.640, 
            # length_retention_loss__v= [ 0.01924,  0.01767,  0.01453,  0.00931, 
            # length_retention_loss__p= [   0.840,    0.920,    0.720,    0.480, 
            # avg_abs_final_grad__min = [ 0.08931,  0.10773,  0.12396,  0.13913, 
            # avg_abs_final_grad__max = [ 0.16860,  0.17279,  0.19468,  0.23154, 
            # avg_abs_final_grad__avg = [ 0.11538,  0.13502,  0.15372,  0.17691, 
            # ------------->>>>>>>>>>>>>    maybe the best is to the left???
            # dim = 10
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.02152,  0.02609,  0.02555,  0.02686,  0.02459,  0.01412,  0.01168,  0.00709,  0.01074,  0.00981]
            # len_loss__opt_prob      = [   0.960,    0.880,    0.800,    0.920,    0.840,    0.360,    0.280,    0.240,    0.360,    0.320]
            # angle_loss__opt_value   = [ 0.50678,  0.53851,  0.52253,  0.52456,  0.52181,  0.38611,  0.25907,  0.17450,  0.22029,  0.17500]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.840,    0.520,    0.600,    0.520]
            # length_retention_loss__v= [ 0.02252,  0.02504,  0.02176,  0.02595,  0.02245,  0.01910,  0.01101,  0.00756,  0.00741,  0.00991]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,    0.960,    0.880,    0.920,    0.760,    0.440,    0.280,    0.480]
            # avg_abs_final_grad__min = [ 0.06897,  0.08274,  0.09327,  0.10540,  0.11693,  0.11828,  0.16401,  0.18378,  0.23128,  0.28030]
            # avg_abs_final_grad__max = [ 0.08227,  0.09607,  0.10942,  0.12625,  0.13741,  0.16680,  0.19109,  0.22826,  0.27750,  0.33054]
            # avg_abs_final_grad__avg = [ 0.07750,  0.09061,  0.10265,  0.11606,  0.12948,  0.15457,  0.18011,  0.20576,  0.25956,  0.30959]
            # ------------->>>>>>>>>>>>>             *****                ***         
            # dim = 10
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.01901,  0.02328,  0.02153,  0.02423,  0.02214,  0.02158,  0.02501,  0.01921,  0.01530,  0.02259]
            # len_loss__opt_prob      = [ v      ,  v      ,    0.920,    0.880,    0.920,    0.760,    0.680,    0.560,    0.440,    0.760]
            # angle_loss__opt_value   = [ 0.39691,  0.45736,  0.49680,  0.52124,  0.52111,  0.50981,  0.43512,  0.35327,  0.33380,  0.43148]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.920,    0.960]
            # length_retention_loss__v= [ 0.01752,  0.02092,  0.02180,  0.02133,  0.02707,  0.02189,  0.02219,  0.01466,  0.01514,  0.01914]
            # length_retention_loss__p= [   0.840,    0.880,    0.920,    0.920,  v      ,    0.960,    0.960,    0.800,    0.840,    0.840]
            # avg_abs_final_grad__min = [ 0.04381,  0.04883,  0.06056,  0.06688,  0.07040,  0.09068,  0.11155,  0.13194,  0.15082,  0.19142]
            # avg_abs_final_grad__max = [ 0.06888,  0.08572,  0.09049,  0.10068,  0.12141,  0.13410,  0.15060,  0.18353,  0.22279,  0.26907]
            # avg_abs_final_grad__avg = [ 0.05595,  0.06498,  0.07498,  0.08516,  0.09795,  0.11400,  0.13334,  0.15478,  0.19358,  0.22871]
            # ------------->>>>>>>>>>>>>                                 *****       ***        
            # dim = 10
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.01458,  0.01714,  0.01718,  0.02285,  0.02076,  0.02378,  0.02359,  0.01943,  0.01428,  0.01689]
            # len_loss__opt_prob      = [   0.960,    0.960,    0.920,    0.920,    0.800,    0.800,    0.880,    0.640,    0.560,    0.600]
            # angle_loss__opt_value   = [ 0.32077,  0.37758,  0.43092,  0.47022,  0.51880,  0.50180,  0.48742,  0.43924,  0.34806,  0.43398]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.960,    0.960]
            # length_retention_loss__v= [ 0.01595,  0.01856,  0.01880,  0.02019,  0.02652,  0.01943,  0.02250,  0.02300,  0.01561,  0.02085]
            # length_retention_loss__p= [   0.880,    0.880,    0.880,    0.960,    0.960,  v      ,    0.960,    0.960,    0.680,    0.880]
            # avg_abs_final_grad__min = [ 0.03093,  0.03708,  0.04332,  0.04949,  0.05770,  0.05759,  0.08225,  0.08491,  0.10784,  0.13556]
            # avg_abs_final_grad__max = [ 0.05935,  0.07034,  0.09188,  0.08659,  0.10856,  0.11072,  0.14256,  0.16562,  0.19340,  0.23859]
            # avg_abs_final_grad__avg = [ 0.04375,  0.05220,  0.05888,  0.06684,  0.07697,  0.08818,  0.10566,  0.12044,  0.14840,  0.17754]
            # ------------->>>>>>>>>>>>>                                           *****       *         *


            # dim = 100
            # cap_to_list      =           0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.00832,  0.00824,  0.00809,  0.00714,  0.00693,  0.00575,  0.00591,  0.00357, -0.00675, -0.01473]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.867, XX      , XX      ]
            # angle_loss__opt_value   = [ 0.56055,  0.54147,  0.49819,  0.46064,  0.42876,  0.40834,  0.34091,  0.19518, -0.48927, -1.07745]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.933,    0.933, XX      , XX      ]
            # length_retention_loss__v= [ 0.00876,  0.00867,  0.00779,  0.00692,  0.00683,  0.00599,  0.00487,  0.00309, -0.00579, -0.01485]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.933,    0.867, XX      , XX      ]
            # avg_abs_final_grad__min = [ 0.02812,  0.03414,  0.03918,  0.04412,  0.04898,  0.05830,  0.06807,  0.07788,  0.09658,  0.11357]
            # avg_abs_final_grad__max = [ 0.03301,  0.03952,  0.04617,  0.04918,  0.05643,  0.06770,  0.08711,  0.08958,  0.11065,  0.13456]
            # avg_abs_final_grad__avg = [ 0.03078,  0.03622,  0.04203,  0.04669,  0.05232,  0.06212,  0.07320,  0.08290,  0.10419,  0.12398]
            # ------------->>>>>>>>>>>>>    maybe the best is to the left???
            # dim = 100
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.00829,  0.00901,  0.00915,  0.00834,  0.00795,  0.00654,  0.00674,  0.00654,  0.00390, -0.00334]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.867,  v      ,    0.933, XX-0.733]
            # angle_loss__opt_value   = [ 0.53456,  0.56278,  0.56545,  0.54463,  0.52120,  0.45810,  0.44541,  0.43033,  0.21494, -0.28898]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      ]
            # length_retention_loss__v= [ 0.00780,  0.00867,  0.00850,  0.00875,  0.00786,  0.00619,  0.00668,  0.00627,  0.00371, -0.00393]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.600, XX-0.733]
            # avg_abs_final_grad__min = [ 0.02374,  0.02782,  0.03179,  0.03572,  0.03980,  0.04785,  0.05570,  0.06363,  0.07950,  0.09549]
            # avg_abs_final_grad__max = [ 0.02411,  0.02817,  0.03226,  0.03623,  0.04046,  0.04837,  0.05637,  0.06432,  0.08055,  0.09641]
            # avg_abs_final_grad__avg = [ 0.02397,  0.02800,  0.03201,  0.03596,  0.04004,  0.04805,  0.05603,  0.06400,  0.08002,  0.09596]
            # ------------->>>>>>>>>>>>>                      *******
            # dim = 100
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.00733,  0.00764,  0.00806,  0.00829,  0.00867,  0.00811,  0.00757,  0.00678,  0.00667,  0.00486]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.933]
            # angle_loss__opt_value   = [ 0.46278,  0.50647,  0.53688,  0.55846,  0.55717,  0.52736,  0.49705,  0.46640,  0.45357,  0.31395]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00774,  0.00786,  0.00846,  0.00845,  0.00850,  0.00823,  0.00772,  0.00804,  0.00629,  0.00509]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.933]
            # avg_abs_final_grad__min = [ 0.01728,  0.01987,  0.02166,  0.02633,  0.02815,  0.03307,  0.03884,  0.04613,  0.05555,  0.06776]
            # avg_abs_final_grad__max = [ 0.02019,  0.02336,  0.02670,  0.03069,  0.03349,  0.04006,  0.04709,  0.05344,  0.06595,  0.08018]
            # avg_abs_final_grad__avg = [ 0.01878,  0.02166,  0.02478,  0.02817,  0.03117,  0.03709,  0.04330,  0.04967,  0.06214,  0.07429]
            # ------------->>>>>>>>>>>>>                                  ***      *****
            # dim = 100
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.00578,  0.00645,  0.00740,  0.00835,  0.00779,  0.00874,  0.00844,  0.00736,  0.00769,  0.00687]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.38025,  0.42834,  0.46059,  0.50126,  0.52172,  0.54349,  0.52966,  0.51771,  0.48739,  0.45078]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00653,  0.00682,  0.00685,  0.00725,  0.00826,  0.00815,  0.00781,  0.00828,  0.00768,  0.00680]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.01206,  0.01326,  0.01621,  0.01900,  0.01927,  0.02446,  0.02866,  0.02980,  0.04424,  0.04821]
            # avg_abs_final_grad__max = [ 0.01722,  0.01957,  0.02139,  0.02450,  0.02738,  0.03406,  0.03950,  0.04408,  0.05831,  0.06616]
            # avg_abs_final_grad__avg = [ 0.01473,  0.01709,  0.01902,  0.02174,  0.02430,  0.02923,  0.03510,  0.03912,  0.04953,  0.05916]
            # ------------->>>>>>>>>>>>>                                            ****      ****
            # dim = 100
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   3.0
            # len_loss__opt_value     = [ 0.00490,  0.00562,  0.00606,  0.00638,  0.00682,  0.00814,  0.00812,  0.00829,  0.00721,  0.00756]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.30980,  0.36351,  0.39909,  0.42163,  0.45576,  0.49925,  0.51943,  0.52173,  0.51358,  0.49532]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00451,  0.00549,  0.00601,  0.00648,  0.00628,  0.00750,  0.00731,  0.00830,  0.00777,  0.00694]
            # length_retention_loss__p= [   0.933,    0.933,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00778,  0.01126,  0.01207,  0.01190,  0.01394,  0.01564,  0.01987,  0.02115,  0.02785,  0.03784]
            # avg_abs_final_grad__max = [ 0.01452,  0.01747,  0.01794,  0.02124,  0.02274,  0.02732,  0.03384,  0.03721,  0.04725,  0.05703]
            # avg_abs_final_grad__avg = [ 0.01164,  0.01402,  0.01567,  0.01690,  0.01920,  0.02261,  0.02667,  0.03017,  0.03888,  0.04532]
            # ------------->>>>>>>>>>>>>                                                                        *******
            
            
            # dim = 1000
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   -2.0
            # len_loss__opt_value     = [ 0.00275,  0.00260,  0.00256,  0.00236,  0.00221,  0.00229,  0.00191,  0.00069, -0.00320, -0.00664]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      ]
            # angle_loss__opt_value   = [ 0.56356,  0.55114,  0.52253,  0.48896,  0.47018,  0.45515,  0.38554,  0.14119, -0.66227, -1.36867]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      ]
            # length_retention_loss__v= [ 0.00256,  0.00274,  0.00215,  0.00246,  0.00219,  0.00241,  0.00219,  0.00034, -0.00321, -0.00661]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.600, XX      , XX      ]
            # avg_abs_final_grad__min = [ 0.00931,  0.01079,  0.01239,  0.01365,  0.01514,  0.01854,  0.02161,  0.02483,  0.03088,  0.03678]
            # avg_abs_final_grad__max = [ 0.01001,  0.01200,  0.01299,  0.01474,  0.01709,  0.01963,  0.02297,  0.02580,  0.03462,  0.03996]
            # avg_abs_final_grad__avg = [ 0.00959,  0.01107,  0.01263,  0.01425,  0.01571,  0.01913,  0.02217,  0.02539,  0.03199,  0.03872]
            # ------------->>>>>>>>>>>>>        or to the left???
            # dim = 1000
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.00271,  0.00280,  0.00264,  0.00252,  0.00231,  0.00224,  0.00229,  0.00174, -0.00104, -0.00463]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      ]
            # angle_loss__opt_value   = [ 0.55590,  0.56321,  0.54685,  0.52038,  0.49060,  0.46218,  0.45542,  0.36416, -0.18731, -0.94564]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      ]
            # length_retention_loss__v= [ 0.00234,  0.00302,  0.00281,  0.00247,  0.00280,  0.00225,  0.00213,  0.00186, -0.00033, -0.00505]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.200, XX      ]
            # avg_abs_final_grad__min = [ 0.00831,  0.00975,  0.01122,  0.01240,  0.01394,  0.01661,  0.01946,  0.02226,  0.02774,  0.03372]
            # avg_abs_final_grad__max = [ 0.00858,  0.01017,  0.01161,  0.01289,  0.01443,  0.01706,  0.02010,  0.02363,  0.02910,  0.03525]
            # avg_abs_final_grad__avg = [ 0.00845,  0.00990,  0.01137,  0.01267,  0.01419,  0.01688,  0.01977,  0.02272,  0.02812,  0.03420]
            # ------------->>>>>>>>>>>>>            *******  or to the left???
            # dim = 1000
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.00259,  0.00279,  0.00270,  0.00265,  0.00259,  0.00245,  0.00230,  0.00219,  0.00084, -0.00231]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      ]
            # angle_loss__opt_value   = [ 0.53595,  0.56062,  0.56309,  0.54740,  0.52238,  0.47661,  0.46435,  0.45610,  0.14637, -0.49425]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      ]
            # length_retention_loss__v= [ 0.00247,  0.00284,  0.00269,  0.00236,  0.00243,  0.00230,  0.00234,  0.00250, -0.00002, -0.00205]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.200, XX      ]
            # avg_abs_final_grad__min = [ 0.00757,  0.00883,  0.01009,  0.01135,  0.01261,  0.01514,  0.01766,  0.02017,  0.02523,  0.03027]
            # avg_abs_final_grad__max = [ 0.00758,  0.00884,  0.01010,  0.01136,  0.01262,  0.01515,  0.01768,  0.02019,  0.02525,  0.03030]
            # avg_abs_final_grad__avg = [ 0.00757,  0.00883,  0.01009,  0.01136,  0.01262,  0.01514,  0.01767,  0.02019,  0.02524,  0.03028]
            # ------------->>>>>>>>>>>>>             *****      ***    or to the left???
            # dim = 1000
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.00247,  0.00265,  0.00266,  0.00276,  0.00264,  0.00255,  0.00226,  0.00220,  0.00180, -0.00056]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.800]
            # angle_loss__opt_value   = [ 0.50521,  0.54385,  0.56200,  0.56165,  0.54908,  0.50473,  0.47135,  0.46598,  0.36046, -0.09057]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      ]
            # length_retention_loss__v= [ 0.00236,  0.00299,  0.00268,  0.00253,  0.00253,  0.00241,  0.00231,  0.00255,  0.00199, -0.00055]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.600]
            # avg_abs_final_grad__min = [ 0.00655,  0.00779,  0.00882,  0.00980,  0.01110,  0.01336,  0.01550,  0.01779,  0.02225,  0.02691]
            # avg_abs_final_grad__max = [ 0.00686,  0.00799,  0.00921,  0.01035,  0.01146,  0.01371,  0.01604,  0.01831,  0.02308,  0.02735]
            # avg_abs_final_grad__avg = [ 0.00674,  0.00791,  0.00901,  0.01013,  0.01123,  0.01349,  0.01583,  0.01804,  0.02264,  0.02717]
            # ------------->>>>>>>>>>>>>              ***       ***       ***      
            # dim = 1000
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.00221,  0.00257,  0.00265,  0.00272,  0.00270,  0.00255,  0.00234,  0.00221,  0.00226,  0.00095]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.47334,  0.51693,  0.54737,  0.56031,  0.56118,  0.52876,  0.49238,  0.47233,  0.45632,  0.20942]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00259,  0.00266,  0.00270,  0.00262,  0.00280,  0.00251,  0.00271,  0.00226,  0.00214,  0.00128]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.800]
            # avg_abs_final_grad__min = [ 0.00571,  0.00679,  0.00779,  0.00868,  0.00981,  0.01186,  0.01297,  0.01590,  0.01868,  0.02322]
            # avg_abs_final_grad__max = [ 0.00620,  0.00725,  0.00832,  0.00944,  0.01052,  0.01252,  0.01483,  0.01666,  0.02058,  0.02517]
            # avg_abs_final_grad__avg = [ 0.00607,  0.00705,  0.00808,  0.00899,  0.01008,  0.01226,  0.01419,  0.01621,  0.01976,  0.02448]
        # ------------->>>>>>>>>>>>>                                     *****      ***
            # dim = 1000
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   3.0
            # len_loss__opt_value     = [ 0.00211,  0.00227,  0.00261,  0.00256,  0.00275,  0.00263,  0.00238,  0.00236,  0.00220,  0.00180]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.42928,  0.48628,  0.52552,  0.54594,  0.55915,  0.55093,  0.51624,  0.49305,  0.46993,  0.38294]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00198,  0.00223,  0.00268,  0.00291,  0.00275,  0.00275,  0.00222,  0.00239,  0.00271,  0.00178]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00507,  0.00594,  0.00715,  0.00760,  0.00867,  0.01006,  0.01220,  0.01328,  0.01544,  0.02028]
            # avg_abs_final_grad__max = [ 0.00559,  0.00671,  0.00771,  0.00859,  0.00951,  0.01137,  0.01346,  0.01533,  0.01900,  0.02305]
            # avg_abs_final_grad__avg = [ 0.00531,  0.00635,  0.00733,  0.00810,  0.00904,  0.01085,  0.01291,  0.01438,  0.01791,  0.02191]
            # ------------->>>>>>>>>>>>>                                  ***      *****
            # dim = 1000
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.00194,  0.00226,  0.00246,  0.00257,  0.00259,  0.00278,  0.00253,  0.00252,  0.00220,  0.00216]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.40029,  0.46028,  0.49811,  0.52305,  0.54304,  0.55765,  0.54322,  0.50802,  0.47859,  0.45219]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00218,  0.00220,  0.00222,  0.00247,  0.00254,  0.00285,  0.00272,  0.00207,  0.00184,  0.00199]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00462,  0.00518,  0.00617,  0.00696,  0.00711,  0.00933,  0.01046,  0.01264,  0.01523,  0.01839]
            # avg_abs_final_grad__max = [ 0.00524,  0.00614,  0.00696,  0.00777,  0.00866,  0.01033,  0.01183,  0.01417,  0.01752,  0.02056]
            # avg_abs_final_grad__avg = [ 0.00486,  0.00585,  0.00663,  0.00731,  0.00808,  0.00979,  0.01126,  0.01335,  0.01621,  0.01949]
            # ------------->>>>>>>>>>>>>                                                    *******
            pass
        
        # conclusion from test 2:
        # best cap_to and best expansion_factor affects mutually. 
        # if expansion_factor is 0, best cap_to is 0.4. But if expansion_factor is -1, best cap_to maybe is < 0.4 ( details in test3)
        # when dim is 100, best expansion_factor is probably between -1 and 1, but when dim is 10 or 1000, the results are similar with expansion_factor from -1 to 4.
        
        if "result of   cap_to 0.2 to 10, expansion_factor 1 to 8" and False:
            
            # dim = 10
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,     7.50,     10.00]
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.01164,  0.01579,  0.01917,  0.02057,  0.02173,  0.02423,  0.02577,  0.02124,  0.01530,  0.01587,  0.01441,  0.01786,  0.01439, -0.00605, -0.04215, -0.06417, -0.07283, -0.07720]
            # len_loss__opt_prob      = [   0.450,    0.470,    0.470,    0.480,    0.440,    0.450,    0.410,    0.390,    0.230,    0.300,    0.230,    0.290,    0.230, XX-0.050, XX-0.450, XX      , XX      , XX      ]
            # angle_loss__opt_value   = [ 0.27801,  0.34959,  0.40836,  0.45435,  0.51215,  0.51399,  0.54180,  0.51788,  0.42814,  0.35589,  0.32596,  0.42812,  0.29643, -0.24837, -0.96358, -1.42133, -1.62618, -1.55587]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.490,    0.440,    0.480,    0.420, XX-0.250, XX      , XX      , XX      , XX      ]
            # length_retention_loss__v= [ 0.01382,  0.01603,  0.01900,  0.01984,  0.02209,  0.02267,  0.02644,  0.02610,  0.01957,  0.01485,  0.01376,  0.01840,  0.01551, -0.00708, -0.04166, -0.06403, -0.07372, -0.07635]
            # length_retention_loss__p= [   0.340,    0.400,  v      ,    0.480,    0.480,    0.490,  v      ,  v      ,    0.460,    0.400,    0.360,    0.440,    0.370, XX-0.130, XX-0.480, XX      , XX      , XX      ]
            # avg_abs_final_grad__min = [ 0.02972,  0.03809,  0.04469,  0.04884,  0.05697,  0.06810,  0.07653,  0.09383,  0.10172,  0.11974,  0.15116,  0.16554,  0.22781,  0.28343,  0.44672,  0.76077,  1.13813,  1.49791]
            # avg_abs_final_grad__max = [ 0.04715,  0.05600,  0.07018,  0.07823,  0.09610,  0.10199,  0.11995,  0.13514,  0.16041,  0.18526,  0.22614,  0.26984,  0.35273,  0.46555,  0.65840,  1.19961,  1.68233,  2.38583]
            # avg_abs_final_grad__avg = [ 0.03816,  0.04739,  0.05686,  0.06604,  0.07619,  0.08463,  0.09441,  0.11333,  0.13361,  0.15291,  0.19005,  0.22407,  0.28418,  0.38037,  0.56235,  0.93972,  1.41254,  1.91537]
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>                                                         *******
            # dim = 10
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,     7.50,     10.00]
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.01118,  0.01184,  0.01510,  0.01786,  0.01916,  0.01982,  0.02119,  0.02515,  0.01875,  0.01727,  0.01090,  0.02171,  0.02088,  0.01193, -0.01411, -0.05050, -0.05879, -0.06828]
            # len_loss__opt_prob      = [ v      ,    0.490,    0.490,    0.480,    0.470,    0.470,    0.450,    0.410,    0.340,    0.300,    0.140,    0.300,    0.310,    0.210, XX-0.260, XX-0.490, XX      , XX-0.490]
            # angle_loss__opt_value   = [ 0.21503,  0.28282,  0.32762,  0.39300,  0.43128,  0.44641,  0.48533,  0.51295,  0.47757,  0.41515,  0.35327,  0.43384,  0.45605,  0.27982, -0.38660, -1.06657, -1.28834, -1.44257]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.480,    0.490,  v      ,    0.420, XX-0.380, XX      , XX      , XX      ]
            # length_retention_loss__v= [ 0.00829,  0.01468,  0.01382,  0.01638,  0.01992,  0.01953,  0.02113,  0.02370,  0.02235,  0.01838,  0.01521,  0.02021,  0.02282,  0.01405, -0.01737, -0.04443, -0.06065, -0.06867]
            # length_retention_loss__p= [   0.230,    0.430,    0.380,    0.420,    0.460,    0.460,    0.490,  v      ,    0.490,    0.410,    0.410,    0.470,    0.490,    0.350, XX-0.340, XX      , XX      , XX      ]
            # avg_abs_final_grad__min = [ 0.02120,  0.02459,  0.03068,  0.03637,  0.04038,  0.04769,  0.04977,  0.05674,  0.07888,  0.08076,  0.10295,  0.13013,  0.15045,  0.21229,  0.29679,  0.53182,  0.76054,  1.14826]
            # avg_abs_final_grad__max = [ 0.04011,  0.04696,  0.06272,  0.07362,  0.08385,  0.09150,  0.09880,  0.12659,  0.14286,  0.16772,  0.21012,  0.25231,  0.28915,  0.37401,  0.58040,  1.02037,  1.45372,  1.99222]
            # avg_abs_final_grad__avg = [ 0.02907,  0.03705,  0.04449,  0.05238,  0.05921,  0.06744,  0.07364,  0.08895,  0.10583,  0.12324,  0.14599,  0.17986,  0.22148,  0.29425,  0.43442,  0.73843,  1.11606,  1.48318]
            # >>>>>>>>>>>>>>>>>>>>>>>>>                                                                         *******
            # dim = 10
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,     7.50,     10.00]
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.00737,  0.00971,  0.01099,  0.01228,  0.01503,  0.01685,  0.01772,  0.01796,  0.02071,  0.01701,  0.01479,  0.01617,  0.01832,  0.01755,  0.00795, -0.01226, -0.02981, -0.03786]
            # len_loss__opt_prob      = [   0.480,  v      ,    0.480,    0.450,    0.440,    0.470,    0.430,    0.400,    0.420,    0.320,    0.280,    0.300,    0.360,    0.330,    0.180, XX-0.170, XX-0.390, XX-0.420]
            # angle_loss__opt_value   = [ 0.15301,  0.19044,  0.22427,  0.28292,  0.31564,  0.35881,  0.38552,  0.43442,  0.45439,  0.43338,  0.37619,  0.35409,  0.37164,  0.35013,  0.16756, -0.21546, -0.59355, -0.82213]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.490,  v      ,    0.490,  v      ,    0.300, XX-0.200, XX-0.490, XX      ]
            # length_retention_loss__v= [ 0.00722,  0.01168,  0.01056,  0.01374,  0.01474,  0.01670,  0.01739,  0.01940,  0.02158,  0.01742,  0.01813,  0.01711,  0.01669,  0.01612,  0.00807, -0.01151, -0.02767, -0.03885]
            # length_retention_loss__p= [   0.160,    0.360,    0.330,    0.360,    0.400,    0.410,    0.460,    0.460,    0.470,    0.460,    0.410,    0.400,    0.400,    0.420,    0.240, XX-0.240, XX-0.440, XX-0.490]
            # avg_abs_final_grad__min = [ 0.01294,  0.01363,  0.01762,  0.02205,  0.02461,  0.02915,  0.02928,  0.03641,  0.04128,  0.05194,  0.07075,  0.07609,  0.09263,  0.11635,  0.18284,  0.31461,  0.53848,  0.68581]
            # avg_abs_final_grad__max = [ 0.03118,  0.03773,  0.05080,  0.05516,  0.06343,  0.08165,  0.07800,  0.09575,  0.11547,  0.14081,  0.17413,  0.23178,  0.26170,  0.35622,  0.53749,  0.83115,  1.25999,  1.68579]
            # avg_abs_final_grad__avg = [ 0.02102,  0.02548,  0.03013,  0.03620,  0.04133,  0.04960,  0.05266,  0.06086,  0.07267,  0.08333,  0.10446,  0.12881,  0.15480,  0.20823,  0.31623,  0.53180,  0.77798,  1.07722]
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>                                                                        *******
            # dim = 10
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,     7.50,     10.00]
            # expansion_factor =   6.0
            # len_loss__opt_value     = [ 0.00596,  0.00609,  0.00875,  0.00963,  0.01127,  0.01460,  0.01469,  0.01730,  0.01810,  0.01758,  0.01736,  0.01276,  0.01419,  0.00962,  0.00950,  0.00081, -0.01325, -0.02137]
            # len_loss__opt_prob      = [ v      ,    0.470,    0.470,    0.470,    0.420,    0.470,    0.440,    0.440,    0.440,    0.390,    0.310,    0.260,    0.240,    0.240,    0.210,    0.010, XX-0.180, XX-0.330]
            # angle_loss__opt_value   = [ 0.11900,  0.15166,  0.18389,  0.22066,  0.25257,  0.28311,  0.32491,  0.36899,  0.40341,  0.38926,  0.35048,  0.33117,  0.31538,  0.25138,  0.20115,  0.01916, -0.24034, -0.41070]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.460,    0.380,    0.110, XX-0.290, XX-0.390]
            # length_retention_loss__v= [ 0.00603,  0.00825,  0.00962,  0.01121,  0.01187,  0.01548,  0.01460,  0.01662,  0.01793,  0.01647,  0.01684,  0.01396,  0.01168,  0.01096,  0.00921,  0.00101, -0.01255, -0.02001]
            # length_retention_loss__p= [   0.220,    0.250,    0.310,    0.310,    0.360,    0.430,    0.370,    0.430,    0.440,    0.450,    0.420,    0.360,    0.340,    0.310,    0.230,    0.030, XX-0.240, XX-0.350]
            # avg_abs_final_grad__min = [ 0.00942,  0.01148,  0.01259,  0.01548,  0.01890,  0.01912,  0.01879,  0.02824,  0.03092,  0.03684,  0.03892,  0.04506,  0.06769,  0.07901,  0.11638,  0.18277,  0.25368,  0.45278]
            # avg_abs_final_grad__max = [ 0.03484,  0.03481,  0.04622,  0.04862,  0.05432,  0.06281,  0.06970,  0.09256,  0.11057,  0.12563,  0.14783,  0.22307,  0.20990,  0.28418,  0.46654,  0.73806,  1.04797,  1.60214]
            # avg_abs_final_grad__avg = [ 0.01660,  0.02007,  0.02421,  0.02829,  0.03264,  0.03588,  0.04213,  0.04804,  0.05804,  0.06601,  0.08020,  0.09704,  0.12285,  0.15706,  0.23692,  0.39573,  0.61491,  0.86198]
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>                                                                        *******
            # dim = 10
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,     7.50,     10.00]
            # expansion_factor =   8.0
            # len_loss__opt_value     = [ 0.00498,  0.00615,  0.00736,  0.00860,  0.01009,  0.01207,  0.01276,  0.01516,  0.01910,  0.01795,  0.01658,  0.01343,  0.00945,  0.01177,  0.00565,  0.00191, -0.00226, -0.00666]
            # len_loss__opt_prob      = [   0.490,    0.460,    0.490,    0.480,    0.470,    0.480,    0.450,    0.450,    0.440,    0.450,    0.360,    0.240,    0.230,    0.260,    0.160,    0.040, XX-0.030, XX-0.120]
            # angle_loss__opt_value   = [ 0.09933,  0.13150,  0.15422,  0.19157,  0.21416,  0.26135,  0.28228,  0.32150,  0.35119,  0.37192,  0.32780,  0.27463,  0.23272,  0.19610,  0.15606, -0.00297, -0.03448, -0.16778]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.490,    0.480,    0.470,    0.420,    0.360,    0.020, XX-0.050, XX-0.160]
            # length_retention_loss__v= [ 0.00202,  0.00500,  0.00739,  0.00706,  0.01112,  0.01462,  0.01078,  0.01531,  0.01691,  0.01621,  0.01327,  0.01294,  0.00986,  0.00990,  0.00764,  0.00219, -0.00466, -0.00689]
            # length_retention_loss__p= [   0.070,    0.180,    0.240,    0.240,    0.330,    0.390,    0.360,    0.420,    0.430,    0.440,    0.400,    0.350,    0.310,    0.260,    0.230,    0.060, XX-0.120, XX-0.150]
            # avg_abs_final_grad__min = [ 0.00694,  0.00904,  0.00849,  0.01187,  0.01348,  0.01645,  0.01739,  0.02046,  0.02493,  0.02781,  0.03333,  0.04158,  0.05105,  0.06860,  0.09340,  0.13886,  0.23494,  0.33179]
            # avg_abs_final_grad__max = [ 0.02566,  0.03025,  0.03281,  0.03991,  0.05398,  0.05551,  0.06613,  0.06715,  0.08710,  0.09098,  0.14328,  0.19206,  0.18758,  0.24737,  0.40934,  0.64439,  0.96552,  1.41980]
            # avg_abs_final_grad__avg = [ 0.01383,  0.01723,  0.02017,  0.02402,  0.02706,  0.03135,  0.03402,  0.04002,  0.04859,  0.05508,  0.06595,  0.08714,  0.10022,  0.13673,  0.20803,  0.35482,  0.50995,  0.69587]
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>                                                                         *****      ***
            
            # ^^^^^^^^^^^^^^^^^^^^^^^^   dim 10   ^^^^^^^^^^^^^^^^^^^^^^^^
            # vvvvvvvvvvvvvvvvvvvvvvvv   dim 100   vvvvvvvvvvvvvvvvvvvvvvvv
            
            # dim = 100
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00, ]
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.00531,  0.00639,  0.00702,  0.00777,  0.00813,  0.00867,  0.00877,  0.00784,  0.00786,  0.00714,  0.00745,  0.00394, -0.00405, -0.01693,]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.450, XX-0.400, XX      ,]
            # angle_loss__opt_value   = [ 0.32610,  0.39960,  0.45416,  0.50896,  0.53862,  0.55430,  0.55641,  0.53964,  0.49515,  0.46565,  0.45054,  0.31295, -0.27287, -1.22885,]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      ,]
            # length_retention_loss__v= [ 0.00558,  0.00513,  0.00796,  0.00762,  0.00807,  0.00841,  0.00822,  0.00850,  0.00799,  0.00693,  0.00636,  0.00467, -0.00234, -0.01757,]
            # length_retention_loss__p= [   0.450,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.200, XX      ,]
            # sum_abs_final_grad__min = [ 103.665,  144.334,  162.930,  203.948,  206.604,  264.925,  276.857,  319.651,  393.942,  470.318,  563.763,  676.151,  788.385,  1167.079, 
            # sum_abs_final_grad__max = [ 132.024,  164.900,  197.517,  234.450,  266.375,  292.188,  326.322,  396.491,  467.425,  540.259,  656.188,  777.548,  1002.53,  1340.531,
            # sum_abs_final_grad__avg = [ 123.484,  155.601,  186.084,  220.252,  249.806,  278.590,  306.429,  370.092,  430.936,  502.125,  614.497,  742.283,  928.402,  1247.680, 
            # >>>>>>>>>>>>>>>>>>>>>>>>                                                        ***     *******          
            # 
            # dim = 100
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,  
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.00412,  0.00472,  0.00550,  0.00641,  0.00715,  0.00747,  0.00759,  0.00925,  0.00804,  0.00785,  0.00705,  0.00746,  0.00437, -0.00606, -0.02200, 
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.400, XX-0.450, XX      , 
            # angle_loss__opt_value   = [ 0.25650,  0.32715,  0.38267,  0.42765,  0.48522,  0.49949,  0.51824,  0.54543,  0.53608,  0.51375,  0.48570,  0.46317,  0.29492, -0.41312, -1.49551, 
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.450, XX      , 
            # length_retention_loss__v= [ 0.00365,  0.00434,  0.00507,  0.00684,  0.00725,  0.00730,  0.00748,  0.00830,  0.00816,  0.00855,  0.00737,  0.00815,  0.00467, -0.00600, -0.02056, 
            # length_retention_loss__p= [   0.450,    0.450,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.400, XX-0.400, XX      , 
            # sum_abs_final_grad__min = [  77.349,  108.899,  130.214,  148.067,  173.985,  187.999,  193.512,  244.734,  302.840,  343.882,  425.406,  482.684,  641.974,  819.776,  1254.258, 
            # sum_abs_final_grad__max = [  110.20,  134.086,  171.274,  196.204,  239.439,  242.181,  272.744,  321.964,  394.602,  454.112,  571.942,  663.823,  822.772,  1098.61,  1585.487
            # sum_abs_final_grad__avg = [  95.061,  121.270,  147.864,  170.932,  204.211,  217.810,  235.821,  294.361,  344.607,  398.872,  496.715,  595.058,  735.123,  972.574,  1443.407, 
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>                                                                       *******               ***  
            # 
            # dim = 100
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,     7.50,  
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.00268,  0.00309,  0.00375,  0.00450,  0.00549,  0.00565,  0.00576,  0.00651,  0.00798,  0.00786,  0.00768,  0.00810,  0.00733,  0.00546, -0.00196, -0.01984, -0.03064, 
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX 0.000, XX      , XX      , 
            # angle_loss__opt_value   = [ 0.17490,  0.21006,  0.25372,  0.28615,  0.34355,  0.36353,  0.39025,  0.43584,  0.48204,  0.49834,  0.50206,  0.50940,  0.47069,  0.32301, -0.11735, -1.34532, -1.96145, 
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX 0.000, XX      , XX      , 
            # length_retention_loss__v= [ 0.00171,  0.00289,  0.00294,  0.00507,  0.00574,  0.00499,  0.00613,  0.00598,  0.00694,  0.00789,  0.00800,  0.00770,  0.00680,  0.00466, -0.00284, -0.02017, -0.02967, 
            # length_retention_loss__p= [   0.250,    0.300,    0.450,    0.450,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.400, XX-0.050, XX      , XX      , 
            # sum_abs_final_grad__min = [  51.587,   38.298,   62.503,   84.746,   95.491,  103.138,  111.165,  126.922,  176.445,  152.385,  212.042,  284.540,  351.446,  509.463,  563.782,  1095.63,  1652.23,
            # sum_abs_final_grad__max = [  78.502,  106.215,  118.078,  134.294,  160.051,  182.446,  188.569,  225.505,  290.580,  315.111,  433.269,  438.318,  632.582,  770.776,  1247.28,  1977.93,  2919.74,
            # sum_abs_final_grad__avg = [  63.741,   76.381,   93.190,  106.946,  130.967,  141.462,  152.836,  183.305,  225.955,  249.952,  316.229,  370.980,  489.876,  650.878,  884.270,  1554.74,  2374.41,
            # >>>>>>>>>>>>>>>>>>>>>                                                                                         ***                         *******          
            # 
            # dim = 100
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,     7.50,     10.00]
            # expansion_factor =   6.0
            # len_loss__opt_value     = [ 0.00190,  0.00211,  0.00277,  0.00340,  0.00338,  0.00392,  0.00483,  0.00523,  0.00600,  0.00625,  0.00680,  0.00706,  0.00733,  0.00641,  0.00252, -0.00557, -0.01810, -0.02242]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.300, XX-0.400, XX      , XX      ]
            # angle_loss__opt_value   = [ 0.12260,  0.13550,  0.18028,  0.21481,  0.22226,  0.25868,  0.29588,  0.33445,  0.37955,  0.39405,  0.43440,  0.44601,  0.46452,  0.41746,  0.19353, -0.27145, -1.12785, -1.37758]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.250, XX-0.350, XX      , XX      ]
            # length_retention_loss__v= [ 0.00289,  0.00235,  0.00235,  0.00376,  0.00372,  0.00395,  0.00335,  0.00457,  0.00620,  0.00528,  0.00711,  0.00718,  0.00733,  0.00618,  0.00206, -0.00590, -0.01876, -0.02236]
            # length_retention_loss__p= [   0.300,    0.350,    0.250,    0.450,    0.400,    0.450,    0.450,    0.450,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.200, XX-0.400, XX      , XX      ]
            # sum_abs_final_grad__min = [  29.415,   21.476,   42.377,   56.145,   45.703,   61.694,   66.206,   77.319,   82.416,  109.994,  137.270,  144.134,  205.968,  329.391,  440.863,  670.776,  1111.696,  1518.173]
            # sum_abs_final_grad__max = [  68.335,   71.283,   98.425,  104.827,  112.048,  128.016,  163.775,  166.118,  201.696,  238.245,  328.211,  318.480,  422.570,  642.479,  950.430,  1451.298,  2477.145,  2782.618]
            # sum_abs_final_grad__avg = [  43.898,   48.214,   65.192,   77.790,   80.605,   94.667,  111.032,  129.836,  154.279,  171.654,  223.976,  243.266,  321.749,  465.030,  686.222,  1047.725,  1738.126,  2026.958]
            # >>>>>>>>>>>>>>>>>>                                                                                                                                  *******
            # 
            # dim = 100
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,     7.50,     10.00]
            # expansion_factor =   8.0
            # len_loss__opt_value     = [ 0.00148,  0.00166,  0.00220,  0.00235,  0.00252,  0.00261,  0.00307,  0.00346,  0.00388,  0.00445,  0.00501,  0.00520,  0.00621,  0.00634,  0.00545,  0.00144, -0.00557, -0.01065]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.450,    0.250, XX-0.350, XX      ]
            # angle_loss__opt_value   = [ 0.09367,  0.10789,  0.14315,  0.14478,  0.17180,  0.18075,  0.21089,  0.24113,  0.25917,  0.31241,  0.35883,  0.35261,  0.37959,  0.41230,  0.35537,  0.14155, -0.25349, -0.61001]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.300, XX-0.250, XX      ]
            # length_retention_loss__v= [ 0.00223,  0.00109,  0.00326,  0.00134,  0.00395,  0.00334,  0.00276,  0.00319,  0.00444,  0.00569,  0.00520,  0.00475,  0.00502,  0.00641,  0.00566,  0.00117, -0.00608, -0.01035]
            # length_retention_loss__p= [   0.250,    0.200,    0.400,    0.100,    0.400,  v      ,    0.350,    0.300,    0.450,  v      ,  v      ,  v      ,  v      ,  v      ,    0.450,    0.150, XX-0.300, XX-0.450]
            # sum_abs_final_grad__min = [  17.596,   22.116,   29.222,   21.854,   34.854,   32.643,   25.995,   43.535,   53.154,   76.356,   64.279,   50.553,  106.497,  249.364,  236.345,  393.416,  497.155,  979.496]
            # sum_abs_final_grad__max = [  49.282,   64.331,   79.973,   89.916,  100.908,   99.299,  137.419,  118.452,  159.804,  166.795,  233.958,  271.947,  329.944,  512.973,  732.596,  1090.89,  1918.24,  2040.43]
            # sum_abs_final_grad__avg = [  33.322,   37.929,   50.796,   51.277,   61.140,   64.452,   75.638,   88.235,   98.274,  124.741,  166.700,  176.787,  225.458,  347.928,  447.356,  709.225,  1102.75,  1473.75]
            # >>>>>>>>>>>>>>>>>>>                                                                                                                                           *******
            
            # when the sum_abs_final_grad__avg is 300, the result is the best.
            # pass
            
            
            # ^^^^^^^^^^^^^^^^^^^^^^^^   dim 100   ^^^^^^^^^^^^^^^^^^^^^^^^
            # vvvvvvvvvvvvvvvvvvvvvvvv   dim 1000   vvvvvvvvvvvvvvvvvvvvvvvv
            
            # dim = 1000
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,  
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.00181,  0.00222,  0.00242,  0.00260,  0.00270,  0.00270,  0.00254,  0.00256,  0.00233,  0.00232,  0.00183, -0.00038, -0.00473, -0.00885, 
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.400, XX      , XX      , 
            # angle_loss__opt_value   = [ 0.37685,  0.44856,  0.50682,  0.54433,  0.56179,  0.56150,  0.54776,  0.50218,  0.47211,  0.46633,  0.36395, -0.07634, -0.95392, -1.83078, 
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.450, XX      , XX      , 
            # length_retention_loss__v= [ 0.00175,  0.00251,  0.00262,  0.00251,  0.00255,  0.00273,  0.00253,  0.00268,  0.00237,  0.00214,  0.00174, -0.00029, -0.00496, -0.00888, 
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.100, XX      , XX      , 
            # sum_abs_final_grad__min = [ 4371.33,  5494.22,  6616.97,  7757.25,  8691.33,  9913.74,  10960.7,  13095.7,  15507.8,  17363.8,  21940.9,  25652.0,  33302.6,  44150.7, 
            # sum_abs_final_grad__max = [ 4595.87,  5737.50,  6892.19,  8064.23,  9204.55,  10433.0,  11536.3,  13788.6,  16218.3,  18415.0,  23100.7,  27586.7,  34744.8,  46273.8, 
            # sum_abs_final_grad__avg = [ 4517.89,  5622.47,  6773.16,  7919.39,  9013.26,  10169.6,  11274.0,  13575.9,  15822.2,  17975.7,  22630.3,  27096.3,  33984.7,  45051.3, 
            # >>>>>>>>>>>>>>>>>>>>>>>                                              *****     *****
            # dim = 1000
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,  
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.00167,  0.00200,  0.00229,  0.00244,  0.00266,  0.00266,  0.00274,  0.00257,  0.00234,  0.00233,  0.00213,  0.00116, -0.00258, -0.00739, 
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      , 
            # angle_loss__opt_value   = [ 0.34288,  0.41298,  0.47391,  0.51680,  0.54707,  0.56154,  0.55992,  0.53064,  0.49302,  0.47235,  0.45035,  0.23352, -0.51315, -1.54074, 
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      , 
            # length_retention_loss__v= [ 0.00149,  0.00200,  0.00222,  0.00250,  0.00245,  0.00296,  0.00274,  0.00280,  0.00239,  0.00222,  0.00210,  0.00118, -0.00241, -0.00762, 
            # length_retention_loss__p= [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.350, XX      , XX      , 
            # sum_abs_final_grad__min = [ 3840.25,  4822.65,  5787.17,  6696.48,  7689.88,  8891.54,  9791.54,  11728.4,  13408.6,  15486.0,  18745.1,  23499.5,  29322.8,  39039.5,
            # sum_abs_final_grad__max = [ 4195.26,  5234.78,  6283.95,  7294.73,  8384.51,  9411.39,  10493.5,  12478.3,  14675.6,  16516.2,  20941.9,  24930.3,  31786.3,  41846.4,
            # sum_abs_final_grad__avg = [ 4045.65,  5044.54,  6090.57,  7047.53,  8088.57,  9125.28,  10143.2,  12145.9,  14175.6,  16081.6,  20219.0,  24247.8,  30413.6,  40402.3,
            # >>>>>>>>>>>>>>>>>>>>>>>                                                         ***       ***       ***
            # dim = 1000
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,  
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.00132,  0.00165,  0.00200,  0.00218,  0.00237,  0.00253,  0.00262,  0.00275,  0.00269,  0.00242,  0.00227,  0.00219,  0.00096, -0.00367, -0.00953, 
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      , 
            # angle_loss__opt_value   = [ 0.27627,  0.34173,  0.40671,  0.44924,  0.48985,  0.52526,  0.54214,  0.55657,  0.54127,  0.51479,  0.47741,  0.44862,  0.18843, -0.74235, -1.96615, 
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      , 
            # length_retention_loss__v= [ 0.00123,  0.00163,  0.00168,  0.00224,  0.00244,  0.00289,  0.00274,  0.00264,  0.00240,  0.00239,  0.00243,  0.00233,  0.00091, -0.00377, -0.00983, 
            # length_retention_loss__p= [   0.450,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.300, XX      , XX      , 
            # sum_abs_final_grad__min = [ 2621.67,  3597.70,  4548.20,  5311.88,  5855.25,  6631.55,  7306.34,  8052.29,  10407.6,  11829.8,  14838.1,  18140.2,  21942.3,  28919.6,  44281.0,
            # sum_abs_final_grad__max = [ 3404.60,  4330.07,  5230.94,  6047.81,  7007.55,  7960.61,  8605.15,  10497.0,  12177.2,  13870.4,  17659.0,  20867.4,  25943.5,  34789.2,  52297.4,
            # sum_abs_final_grad__avg = [ 3188.52,  4026.61,  4959.09,  5651.40,  6447.50,  7376.31,  8041.14,  9792.27,  11414.4,  12961.2,  16251.2,  19689.2,  24431.2,  32552.2,  49350.1,
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>                                                     ***                *****
            # dim = 1000
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00,  
            # expansion_factor =   6.0
            # len_loss__opt_value     = [ 0.00113,  0.00140,  0.00163,  0.00190,  0.00210,  0.00215,  0.00240,  0.00260,  0.00262,  0.00261,  0.00248,  0.00224,  0.00199, -0.00050, -0.00651, -0.01127, 
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.200, XX      , XX      , 
            # angle_loss__opt_value   = [ 0.22992,  0.28782,  0.33659,  0.38634,  0.42467,  0.46246,  0.49645,  0.53277,  0.55031,  0.54441,  0.50632,  0.48148,  0.42037, -0.05933, -1.32751, -2.30729, 
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.100, XX      , XX      , 
            # length_retention_loss__v= [ 0.00122,  0.00122,  0.00168,  0.00191,  0.00205,  0.00218,  0.00233,  0.00243,  0.00279,  0.00256,  0.00238,  0.00216,  0.00220, -0.00021, -0.00627, -0.01076, 
            # length_retention_loss__p= [   0.450,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX-0.100, XX      , XX      , 
            # sum_abs_final_grad__min = [ 2281.09,  2897.20,  3386.22,  4123.92,  4195.22,  5168.89,  5718.24,  6910.15,  8168.85,  9074.61,  12640.0,  12608.2,  17419.8,  24750.1,  35918.3,  60220.152,  
            # sum_abs_final_grad__max = [ 2931.33,  3768.44,  4271.30,  5273.65,  5859.53,  6619.19,  7380.40,  9046.09,  10052.9,  12212.7,  14878.7,  17827.8,  21824.3,  29171.5,  42975.5,  71926.797,
            # sum_abs_final_grad__avg = [ 2621.81,  3332.17,  3961.21,  4659.69,  5269.25,  5936.35,  6672.84,  7866.28,  9196.18,  10589.3,  13539.2,  16094.4,  19900.2,  26838.1,  39729.9,  66115.234, 
            # >>>>>>>>>>>>>>>>>>>>>                                                                                       *******
            # dim = 1000
            # cap_to_list               = [ 0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20,     1.50,     2.00,     3.00,     5.00, 
            # expansion_factor =   8.0
            # len_loss__opt_value     = [ 0.00093,  0.00111,  0.00135,  0.00157,  0.00178,  0.00189,  0.00206,  0.00230,  0.00247,  0.00257,  0.00263,  0.00254,  0.00239,  0.00140, -0.00283, -0.00955,
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      ,
            # angle_loss__opt_value   = [ 0.19189,  0.23497,  0.28255,  0.32300,  0.36566,  0.39141,  0.43245,  0.47732,  0.51744,  0.53641,  0.53707,  0.51274,  0.47263,  0.30197, -0.51298, -1.98053,
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      , XX      , XX      ,
            # length_retention_loss__v= [ 0.00069,  0.00110,  0.00116,  0.00148,  0.00181,  0.00212,  0.00207,  0.00266,  0.00257,  0.00274,  0.00248,  0.00254,  0.00249,  0.00139, -0.00276, -0.00982,
            # length_retention_loss__p= [   0.350,    0.450,  v      ,  v      ,    0.450,    0.450,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.450, XX      , XX      ,
            # sum_abs_final_grad__min = [ 1750.43,  2248.20,  2639.03,  2538.02,  3500.89,  3556.32,  4343.38,  5418.68,  6348.57,  7553.30,  8057.51,  11053.3,  13069.7,  17267.3,  28016.6,  46066.7,
            # sum_abs_final_grad__max = [ 2460.02,  3241.37,  3717.02,  4190.39,  4966.41,  5499.70,  6055.83,  7336.27,  8322.90,  9591.62,  12166.8,  14754.2,  19849.6,  24870.3,  36203.5,  64152.0,
            # sum_abs_final_grad__avg = [ 2170.56,  2680.08,  3261.24,  3785.25,  4369.53,  4754.30,  5433.47,  6319.39,  7481.03,  8508.69,  10337.5,  12875.7,  16267.9,  21789.1,  31650.2,  55209.2,
            # >>>>>>>>>>>>>>>>>>>>>                                                                                                   ***      *****
            
            pass
        
        # conclusion from test 1:
        # best expansion_factor is <4. No idea about the lower bound.(more in test 2)
        # all 3 measurements show very similar result.
        
        
        # summary:
        # Anyway, the expansion_factor as 0 fits into most cases.. But I'll scan again in the next test.
        # the avg_abs_final_grad__avg shows some clue. The next test will contain a adaptive algo based on this.
        
        
        
        
        #------------------#------------------#------------------
        # dim_list =        [ 2, 10,100,1000]
        # test_time_list = [100,100,100, 20]
        # test_time_list = [10,10,10, 2]
        
        dim_list =       [ 10, 100,1000]
        test_time_list = [ 50, 30, 10]
        # dim_list =        [100]
        # test_time_list = [ 50]
        for outter_iter_count in range(dim_list.__len__()):
            dim = dim_list[outter_iter_count]
            test_time = test_time_list[outter_iter_count]
            print(test_time)
            iota_of_dim = iota(dim)
            if dim>100:
                device = 'cuda'
                pass
            else:
                device = 'cpu'
                pass
        #------------------#------------------#------------------
        
            #------------------#------------------#------------------
            expansion_factor_list = [-2.,-1., 0.]#,1.,2.,3.,4.]
            for expansion_factor in expansion_factor_list:
            #------------------#------------------#------------------
            
                len_loss__opt_value              = []  # dont modity this
                len_loss__opt_prob               = []  # dont modity this
                angle_loss__opt_value            = []  # dont modity this
                angle_loss__opt_prob             = []  # dont modity this
                length_retention_loss__opt_value = []  # dont modity this
                length_retention_loss__opt_prob  = []  # dont modity this
                avg_of_abs_of_final_grad__min    = []  # dont modity this
                avg_of_abs_of_final_grad__max    = []  # dont modity this
                avg_of_abs_of_final_grad__avg    = []  # dont modity this
                #neg_behavior_similar_loss            = []  # dont modity this
                    
                #------------------#------------------#------------------
                #cap_to_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1., 1.2, 1.5, 2.]
                #cap_to_list = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1., 1.2]
                cap_to_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,]
                for cap_to in cap_to_list:
                #------------------#------------------#------------------
                
                    _raw_result__len_loss__after_sub_before                 = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__angle_loss__after_sub_before               = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__length_retention_loss__after_sub_before    = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__avg_of_abs_of_final_grad                   = torch.empty(size=[test_time])  # dont modity this
                    #_raw_result__neg_behavior_similar_loss                  = torch.empty(size=[test_time])  # dont modity this
                    
                    for test_count in range(test_time):
                        #----------------#----------------#----------------#----------------
                        ori_mat = torch.randn(size=[dim,dim], device=device)
                        ori_mat = vector_length_norm(ori_mat)
                        # old code ori_mat = ____init_mat_with_row_len_is_1(dim, device=device)
                        
                        #<  measure the init>
                        before__len_loss, before__angle_loss, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                        before__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(ori_mat)
                        
                        #<  calc grad>
                        mat = ori_mat.detach().clone()
                        
                        manual__mat_matmul_mat__d_d = mat@(mat.T)
                        manual__mat_matmul_mat__d_d[iota_of_dim, iota_of_dim] = 0.
                        
                        #<  original grad>
                        ori__grad__d_d = manual__mat_matmul_mat__d_d@mat
                        ori__grad_len__dim = get_vector_length(ori__grad__d_d)
                        #<  original grad, but len into 1>
                        _temp_reverse_of__ori__grad_len__dim = ori__grad_len__dim.pow(-1.)
                        len_1__ori_grad__d_d = ori__grad__d_d.mul( \
                                expand_vec_to_matrix(_temp_reverse_of__ori__grad_len__dim, each_element_to="row"))
                        del _temp_reverse_of__ori__grad_len__dim # just in case.
                        assert _tensor_equal(get_vector_length(len_1__ori_grad__d_d), torch.ones(size=[dim], device=device))
                        
                        #<  scale it a bit, to make the distribution a bit wider>
                        #otherwise, they are a lot 0.8,0.9. Wider means most are 0.3 to 0.8.
                        #<old code/>xx__grad_len_sqr__dim = ori__grad__d_d.mul(ori__grad__d_d).sum(dim=1)#mul and then sum, it's a dot.
                        max_of__ori__grad_len__dim = ori__grad_len__dim.max()
                        ratio_of__grad_len__dim = ori__grad_len__dim/max_of__ori__grad_len__dim
                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this scaled_to_1__grad_len_sqr__dim is still one 1.0 and a lot 0.xx
                        assert ratio_of__grad_len__dim.le(1.).all()
                        assert ratio_of__grad_len__dim.eq(1.).sum() == 1
                        
                        grad_len__after_expansion__dim = ratio_of__grad_len__dim.pow(expansion_factor)
                        # assert grad_len__after_expansion__dim.le(1.).all() when expansion_factor is neg, this is wrong.
                        # assert grad_len__after_expansion__dim.eq(1.).sum() == 1 when expansion_factor is 0., this is wrong.
                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this grad_len__after_expansion__dim is still one 1.0 and a lot 0.xx
                        
                        if "visualization" and False:
                            #prin(expansion_factor)
                            visualize_this = grad_len__after_expansion__dim
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(tight_layout=True)
                            n_bins = dim//5
                            if n_bins<=3:
                                n_bins = 3
                                pass
                            ax.hist(visualize_this.tolist(), bins=n_bins)
                            plt.show()
                            pass
                        
                        #<  target length>
                        grad_length_target__dim = grad_len__after_expansion__dim*cap_to# cap_to is scalar
                        grad_length_target__dim__expand_dim = expand_vec_to_matrix(grad_length_target__dim, each_element_to="row")
                        
                        #<  finally 
                        useful_grad__d_d = len_1__ori_grad__d_d.mul(grad_length_target__dim__expand_dim)#row
                        # # assert _tensor_equal(get_vector_length(useful_grad__d_d), grad_length_target__dim)
                        
                        # debug only  vvvvvvvvvvvvvvvvvvvv
                        log10_of__mat = log10_avg_safe(mat)
                        log10_of__grad = log10_avg_safe(useful_grad__d_d)
                        # debug only  ^^^^^^^^^^^^^^^^^^^^
                        
                        #<  update mat
                        mat -= useful_grad__d_d
                        
                        # debug only  vvvvvvvvvvvvvvvvvvvv
                        log10_of__mat_after_modify = log10_avg_safe(mat)
                        log10_similarity = log10_avg__how_similar(mat, ori_mat)
                        # debug only  ^^^^^^^^^^^^^^^^^^^^
                        
                        #<  protect the length after updating
                        _temp_len_sqr = mat.mul(mat).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
                        mul_me___to_get_unit_length = (_temp_len_sqr).pow(-0.5)
                        mat *= expand_vec_to_matrix(mul_me___to_get_unit_length, each_element_to="row")
                        #<  a bit check>
                        temp_vec_len = get_vector_length(mat)
                        # # assert _tensor_equal(temp_vec_len, [1.]*dim)
                        
                        #<  measure the protected.>
                        after__len_loss, after__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                        after__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
                        #----------------#----------------#----------------#----------------
                        
                        _raw_result__len_loss__after_sub_before[test_count] = before__len_loss - after__len_loss
                        _raw_result__angle_loss__after_sub_before[test_count] = before__angle_loss - after__angle_loss
                        _raw_result__length_retention_loss__after_sub_before[test_count] = \
                            before__length_retention_loss - after__length_retention_loss
                        _raw_result__avg_of_abs_of_final_grad[test_count] = useful_grad__d_d.abs().mean()
                        #_raw_result__neg_behavior_similar_loss[test_count] = -LOSS__behavior_similarity(mat, ori_mat)
                        pass#for test_count
                    
                    len_loss__opt_value             .append(_raw_result__len_loss__after_sub_before.mean().item())
                    len_loss__opt_prob              .append(_raw_result__len_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    angle_loss__opt_value           .append(_raw_result__angle_loss__after_sub_before.mean().item())
                    angle_loss__opt_prob            .append(_raw_result__angle_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    length_retention_loss__opt_value.append(_raw_result__length_retention_loss__after_sub_before.mean().item())
                    length_retention_loss__opt_prob .append(_raw_result__length_retention_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    avg_of_abs_of_final_grad__min   .append(_raw_result__avg_of_abs_of_final_grad.min ())
                    avg_of_abs_of_final_grad__max   .append(_raw_result__avg_of_abs_of_final_grad.max ())
                    avg_of_abs_of_final_grad__avg   .append(_raw_result__avg_of_abs_of_final_grad.mean())
                    #neg_behavior_similar_loss       .append(_raw_result__neg_behavior_similar_loss.mean().item())
                    
                    pass# for expansion_factor(y axis)
                
                print(f"dim = {dim}")
                # print(f"expansion_factor_list   = {str_the_list(expansion_factor_list, 2, segment=",    ")}")
                # print(f"cap_to = {cap_to}")
                print(f"cap_to_list               = {str_the_list(cap_to_list, 2, segment=",    ")}")
                print(f"expansion_factor =   {expansion_factor}")
                print(f"len_loss__opt_value     = {str_the_list(len_loss__opt_value, 5)}")
                print(f"len_loss__opt_prob      = {str_the_list__probability(len_loss__opt_prob, 3, flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                print(f"angle_loss__opt_value   = {str_the_list(angle_loss__opt_value, 5)}")
                print(f"angle_loss__opt_prob    = {str_the_list__probability(angle_loss__opt_prob, 3, flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                print(f"length_retention_loss__v= {str_the_list(length_retention_loss__opt_value, 5)}")
                print(f"length_retention_loss__p= {str_the_list__probability(length_retention_loss__opt_prob, 3, 
                                                                                                    flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                print(f"avg_abs_final_grad__min = {str_the_list(avg_of_abs_of_final_grad__min, 5)}")
                print(f"avg_abs_final_grad__max = {str_the_list(avg_of_abs_of_final_grad__max, 5)}")
                print(f"avg_abs_final_grad__avg = {str_the_list(avg_of_abs_of_final_grad__avg, 5)}")
                #print(f"neg_behavior_similar_loss= {str_the_list(neg_behavior_similar_loss, 5)}")
                print("------------->>>>>>>>>>>>>")
                
                pass#for cap_to(x axis)
            
            pass#for outter_iter_count
        
        pass#/ test
    
    #it feels like, the expansion_factor is better to be 0 or around.
    # behavior similarity doesn't show anything.
    if "behavior similarity" and False:
        # positive is better, greater is better
        
        if "original data" and False:
        
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15, 
            # expansion_factor =   -4
            # angle_loss__opt_value   = [ 0.06591,  0.04063,  0.09504, -0.07959, -0.05965,
            # length_retention_loss__v= [ 0.00609,  0.00204,  0.00598, -0.00235, -0.00238,
            # avg_abs_final_grad__avg = [ 0.03366,  0.06056,  0.13488,  0.38360,  0.58498,
            # neg_BS__length__min     = [-0.07788, -0.12619, -0.11769, -0.15377, -0.15975,
            # neg_BS__length__avg     = [-0.02141, -0.03836, -0.06170, -0.09594, -0.10406,
            # neg_BS__angle__min      = [-0.20432, -0.46301, -0.56125, -0.77375, -0.65506,
            # neg_BS__angle__avg      = [-0.02945, -0.07072, -0.13825, -0.33622, -0.37087,
            # ------------->>>>>>>>>>>>>                       *****
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,  
            # expansion_factor =   -2.0
            # angle_loss__opt_value   = [ 0.05071,  0.08677,  0.19680,  0.28555,  0.29444,  0.28933,  0.18580,  0.20505,  0.10955,  0.02049,  0.02755, -0.06364, -0.08956, 
            # length_retention_loss__v= [ 0.00277,  0.00161,  0.00876,  0.01339,  0.00925,  0.01171,  0.00931,  0.01171,  0.00619,  0.00066,  0.00226, -0.00329, -0.00315, 
            # avg_abs_final_grad__avg = [ 0.00664,  0.01268,  0.03173,  0.06319,  0.09832,  0.12378,  0.17774,  0.19132,  0.23653,  0.25198,  0.28196,  0.34078,  0.39400, 
            # neg_BS__length__min     = [-0.01060, -0.01722, -0.04715, -0.08206, -0.11637, -0.13077, -0.13457, -0.13498, -0.16032, -0.13369, -0.15235, -0.16152, -0.17822, 
            # neg_BS__length__avg     = [-0.00385, -0.00766, -0.01930, -0.04119, -0.05836, -0.07374, -0.08922, -0.09453, -0.10449, -0.10818, -0.11373, -0.11597, -0.11869, 
            # neg_BS__angle__min      = [-0.00229, -0.00898, -0.06467, -0.24557, -0.30139, -0.45774, -0.53841, -0.57299, -0.68968, -0.82272, -0.81490, -0.92272, -0.94645, 
            # neg_BS__angle__avg      = [-0.00040, -0.00149, -0.00958, -0.04458, -0.09407, -0.14332, -0.23412, -0.25929, -0.36218, -0.44487, -0.47322, -0.57606, -0.70201, 
            # ------------->>>>>>>>>>>>>                                    **      **
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   -1.0
            # angle_loss__opt_value   = [ 0.02985,  0.05887,  0.13949,  0.26648,  0.38193,  0.43431,  0.45450,  0.44971,  0.40546,  0.31581,  0.22247,  0.19815,  0.06037,  0.05341,  0.10051]
            # length_retention_loss__v= [ 0.00140,  0.00457,  0.00855,  0.01148,  0.01391,  0.02031,  0.02152,  0.02014,  0.01858,  0.01540,  0.00923,  0.01280,  0.00208, -0.00134,  0.00874]
            # avg_abs_final_grad__avg = [ 0.00392,  0.00782,  0.01930,  0.03842,  0.05996,  0.07409,  0.09505,  0.11526,  0.13154,  0.15354,  0.17390,  0.19689,  0.22133,  0.27115,  0.30741]
            # neg_BS__length__min     = [-0.00306, -0.00685, -0.01594, -0.04344, -0.06514, -0.06509, -0.08158, -0.10630, -0.11230, -0.11324, -0.13119, -0.13787, -0.14084, -0.15349, -0.13813]
            # neg_BS__length__avg     = [-0.00196, -0.00399, -0.00996, -0.02043, -0.03484, -0.04189, -0.05515, -0.06832, -0.07762, -0.08929, -0.09796, -0.10284, -0.11337, -0.11519, -0.10931]
            # neg_BS__angle__min      = [-0.00018, -0.00075, -0.00456, -0.03970, -0.11152, -0.07004, -0.15604, -0.19616, -0.26087, -0.39646, -0.39845, -0.49224, -0.60579, -0.73364, -0.90409]
            # neg_BS__angle__avg      = [-0.00007, -0.00027, -0.00171, -0.00770, -0.02240, -0.03222, -0.06216, -0.09318, -0.12660, -0.18521, -0.23809, -0.29712, -0.39491, -0.56394, -0.69079]
            # ------------->>>>>>>>>>>>>                                                                ****
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   0.0
            # angle_loss__opt_value   = [ 0.01867,  0.03957,  0.09382,  0.19157,  0.29026,  0.36784,  0.43545,  0.49362,  0.52423,  0.52228,  0.55013,  0.46986,  0.35994,  0.21461,  0.20571]
            # length_retention_loss__v= [ 0.00186, -0.00107,  0.00567,  0.00453,  0.01438,  0.01233,  0.02025,  0.02193,  0.02508,  0.02506,  0.02688,  0.02194,  0.01714,  0.00914,  0.01028]
            # avg_abs_final_grad__avg = [ 0.00257,  0.00516,  0.01294,  0.02582,  0.03876,  0.05156,  0.06438,  0.07733,  0.09001,  0.10394,  0.11661,  0.12854,  0.15546,  0.18083,  0.20750]
            # neg_BS__length__min     = [-0.00164, -0.00316, -0.00801, -0.01637, -0.02818, -0.03518, -0.04094, -0.05269, -0.06404, -0.07871, -0.09039, -0.09382, -0.12242, -0.13489, -0.14787]
            # neg_BS__length__max     = [-0.00083, -0.00190, -0.00443, -0.00761, -0.01403, -0.01729, -0.02690, -0.03379, -0.03005, -0.04899, -0.04571, -0.05885, -0.07046, -0.07837, -0.07832]
            # neg_BS__length__avg     = [-0.00121, -0.00251, -0.00618, -0.01259, -0.02054, -0.02669, -0.03355, -0.04204, -0.04918, -0.05924, -0.06669, -0.07412, -0.08989, -0.10390, -0.10767]
            # neg_BS__angle__min      = [-0.00003, -0.00011, -0.00072, -0.00301, -0.00692, -0.01348, -0.02254, -0.03280, -0.05047, -0.06694, -0.09645, -0.11710, -0.21330, -0.30688, -0.44001]
            # neg_BS__angle__max      = [-0.00001, -0.00006, -0.00039, -0.00171, -0.00414, -0.00820, -0.01210, -0.02060, -0.02643, -0.04494, -0.05664, -0.07292, -0.13306, -0.21023, -0.25849]
            # neg_BS__angle__avg      = [-0.00002, -0.00009, -0.00055, -0.00232, -0.00544, -0.01024, -0.01790, -0.02662, -0.03932, -0.05487, -0.07374, -0.09665, -0.16523, -0.25637, -0.35612]
            # ------------->>>>>>>>>>>>>                                                                                                        ****
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   1.0
            # angle_loss__opt_value   = [ 0.01426,  0.02799,  0.07069,  0.14348,  0.20059,  0.27240,  0.33624,  0.40125,  0.48031,  0.50322,  0.52595,  0.53192,  0.52038,  0.44395,  0.34606]
            # length_retention_loss__v= [ 0.00067,  0.00431,  0.00045,  0.00658,  0.00963,  0.01107,  0.01599,  0.01714,  0.02209,  0.02274,  0.02397,  0.02583,  0.02439,  0.02187,  0.01666]
            # avg_abs_final_grad__avg = [ 0.00188,  0.00371,  0.00955,  0.01889,  0.02792,  0.03747,  0.04690,  0.05698,  0.06773,  0.07590,  0.08438,  0.09282,  0.11498,  0.13324,  0.14806]
            # neg_BS__length__min     = [-0.00121, -0.00246, -0.00624, -0.01299, -0.01851, -0.02626, -0.03335, -0.03626, -0.05062, -0.05388, -0.07020, -0.07594, -0.09278, -0.10399, -0.11782]
            # neg_BS__length__max     = [-0.00059, -0.00109, -0.00290, -0.00585, -0.00933, -0.01146, -0.01740, -0.02415, -0.02638, -0.02964, -0.02498, -0.03499, -0.04901, -0.05329, -0.05396]
            # neg_BS__length__avg     = [-0.00089, -0.00177, -0.00445, -0.00913, -0.01337, -0.01814, -0.02400, -0.02917, -0.03720, -0.04226, -0.04891, -0.05332, -0.06710, -0.08109, -0.08562]
            # neg_BS__angle__min      = [-0.00002, -0.00006, -0.00045, -0.00160, -0.00460, -0.00772, -0.01259, -0.01858, -0.02858, -0.03732, -0.05133, -0.06877, -0.11707, -0.17198, -0.26174]
            # neg_BS__angle__max      = [-0.00001, -0.00003, -0.00018, -0.00075, -0.00166, -0.00314, -0.00586, -0.00908, -0.01384, -0.01666, -0.01904, -0.02886, -0.04648, -0.08482, -0.10042]
            # neg_BS__angle__avg      = [-0.00001, -0.00004, -0.00028, -0.00117, -0.00263, -0.00484, -0.00839, -0.01335, -0.01979, -0.02604, -0.03524, -0.04411, -0.07686, -0.12119, -0.17291]
            # ------------->>>>>>>>>>>>>                                                                                                                *****
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   2.0
            # angle_loss__opt_value   = [ 0.01054,  0.02165,  0.05348,  0.10528,  0.16823,  0.22960,  0.27015,  0.32464,  0.38445,  0.43178,  0.45507,  0.49110,  0.51242,  0.47152,  0.46228]
            # length_retention_loss__v= [ 0.00021, -0.00066,  0.00358,  0.00469,  0.00768,  0.01139,  0.01291,  0.01340,  0.01660,  0.01893,  0.01978,  0.02271,  0.02126,  0.02157,  0.02012]
            # avg_abs_final_grad__avg = [ 0.00147,  0.00298,  0.00756,  0.01453,  0.02332,  0.03029,  0.03606,  0.04341,  0.05201,  0.05982,  0.06725,  0.07519,  0.08905,  0.10361,  0.11391]
            # neg_BS__length__min     = [-0.00100, -0.00212, -0.00501, -0.01020, -0.01688, -0.02234, -0.02545, -0.03444, -0.03917, -0.04761, -0.05625, -0.06116, -0.08390, -0.09778, -0.10185]
            # neg_BS__length__max     = [-0.00038, -0.00089, -0.00239, -0.00387, -0.00605, -0.00875, -0.01276, -0.01530, -0.01570, -0.01971, -0.02119, -0.02631, -0.03296, -0.04161, -0.04574]
            # neg_BS__length__avg     = [-0.00067, -0.00138, -0.00356, -0.00671, -0.01107, -0.01549, -0.01816, -0.02366, -0.02746, -0.03412, -0.03750, -0.04648, -0.05337, -0.06613, -0.07246]
            # neg_BS__angle__min      = [-0.00001, -0.00005, -0.00030, -0.00120, -0.00395, -0.00637, -0.00833, -0.01555, -0.01961, -0.02724, -0.04381, -0.06612, -0.08481, -0.16930, -0.20532]
            # neg_BS__angle__max      = [-0.00000, -0.00002, -0.00008, -0.00036, -0.00073, -0.00186, -0.00296, -0.00398, -0.00596, -0.00892, -0.01030, -0.01075, -0.02063, -0.02976, -0.06693]
            # neg_BS__angle__avg      = [-0.00001, -0.00003, -0.00019, -0.00073, -0.00187, -0.00358, -0.00531, -0.00813, -0.01189, -0.01753, -0.02308, -0.03154, -0.04812, -0.07888, -0.11009]
            # ------------->>>>>>>>>>>>>                                                                                                                  ***       ***
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   3.0
            # angle_loss__opt_value   = [ 0.00856,  0.01712,  0.04344,  0.08720,  0.13123,  0.18459,  0.22171,  0.29376,  0.31387,  0.36477,  0.39743,  0.39979,  0.45805,  0.48597,  0.40312]
            # length_retention_loss__v= [ 0.00334,  0.00094,  0.00218,  0.00289,  0.00888,  0.00760,  0.01119,  0.01278,  0.01660,  0.01170,  0.01987,  0.01909,  0.02138,  0.02325,  0.01635]
            # avg_abs_final_grad__avg = [ 0.00118,  0.00240,  0.00616,  0.01209,  0.01842,  0.02399,  0.03015,  0.03763,  0.04319,  0.04821,  0.05350,  0.05947,  0.07143,  0.08830,  0.09589]
            # neg_BS__length__min     = [-0.00088, -0.00196, -0.00434, -0.00805, -0.01472, -0.02412, -0.02519, -0.03317, -0.03910, -0.04700, -0.05060, -0.06003, -0.07605, -0.10651, -0.09975]
            # neg_BS__length__max     = [-0.00027, -0.00054, -0.00136, -0.00309, -0.00544, -0.00787, -0.00923, -0.01180, -0.01367, -0.01619, -0.02029, -0.01848, -0.02565, -0.02980, -0.04016]
            # neg_BS__length__avg     = [-0.00055, -0.00114, -0.00293, -0.00569, -0.00894, -0.01267, -0.01513, -0.02088, -0.02365, -0.02833, -0.03132, -0.03520, -0.04483, -0.05752, -0.06256]
            # neg_BS__angle__min      = [-0.00001, -0.00004, -0.00025, -0.00102, -0.00249, -0.00430, -0.00978, -0.01303, -0.02129, -0.02345, -0.03302, -0.05305, -0.07640, -0.13209, -0.15249]
            # neg_BS__angle__max      = [-0.00000, -0.00001, -0.00006, -0.00023, -0.00067, -0.00118, -0.00168, -0.00281, -0.00328, -0.00521, -0.00883, -0.00786, -0.01590, -0.02605, -0.04091]
            # neg_BS__angle__avg      = [-0.00000, -0.00002, -0.00013, -0.00058, -0.00136, -0.00246, -0.00414, -0.00683, -0.00960, -0.01302, -0.01648, -0.02272, -0.03708, -0.06349, -0.08679]
            # ------------->>>>>>>>>>>>>                                                                                                                                    ******
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   4.0
            # angle_loss__opt_value   = [ 0.00730,  0.01459,  0.03567,  0.07786,  0.11895,  0.15604,  0.19505,  0.25184,  0.29183,  0.29193,  0.35870,  0.39954,  0.41313,  0.44387,  0.41207]
            # length_retention_loss__v= [-0.00264, -0.00146,  0.00047,  0.00465,  0.00714,  0.00853,  0.01059,  0.01332,  0.01358,  0.01289,  0.01766,  0.01665,  0.01979,  0.02159,  0.01930]
            # avg_abs_final_grad__avg = [ 0.00101,  0.00206,  0.00494,  0.01075,  0.01626,  0.02104,  0.02586,  0.03214,  0.03798,  0.03879,  0.04580,  0.05275,  0.06130,  0.07177,  0.08192]
            # neg_BS__length__min     = [-0.00079, -0.00227, -0.00344, -0.00958, -0.01342, -0.01747, -0.02367, -0.03718, -0.03436, -0.04138, -0.05034, -0.05949, -0.06220, -0.07282, -0.08997]
            # neg_BS__length__max     = [-0.00019, -0.00055, -0.00115, -0.00269, -0.00389, -0.00568, -0.00816, -0.01008, -0.00953, -0.01275, -0.01371, -0.01366, -0.02271, -0.02638, -0.03721]
            # neg_BS__length__avg     = [-0.00047, -0.00097, -0.00236, -0.00511, -0.00787, -0.01038, -0.01365, -0.01798, -0.02145, -0.02267, -0.02908, -0.03337, -0.04008, -0.04949, -0.05740]
            # neg_BS__angle__min      = [-0.00001, -0.00004, -0.00019, -0.00090, -0.00254, -0.00509, -0.00690, -0.01080, -0.01804, -0.02156, -0.02497, -0.04107, -0.07609, -0.09016, -0.15417]
            # neg_BS__angle__max      = [-0.00000, -0.00001, -0.00004, -0.00014, -0.00046, -0.00078, -0.00124, -0.00222, -0.00270, -0.00512, -0.00469, -0.00830, -0.01220, -0.02671, -0.03665]
            # neg_BS__angle__avg      = [-0.00000, -0.00002, -0.00010, -0.00050, -0.00120, -0.00219, -0.00347, -0.00549, -0.00856, -0.00987, -0.01446, -0.01907, -0.03157, -0.04883, -0.07525]
            # ------------->>>>>>>>>>>>>                                                                                                                                    ******
            # dim = 10
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   6
            # angle_loss__opt_value   = [ 0.00537,  0.01143,  0.02905,  0.05817,  0.08823,  0.12082,  0.14599,  0.18736,  0.23017,  0.24413,  0.28371,  0.29818,  0.35975,  0.41628,  0.40677]
            # length_retention_loss__v= [ 0.00080,  0.00217, -0.00062,  0.00335,  0.00222,  0.00690,  0.01075,  0.00785,  0.00912,  0.01246,  0.01466,  0.01378,  0.01423,  0.01705,  0.01736]
            # avg_abs_final_grad__avg = [ 0.00078,  0.00165,  0.00408,  0.00821,  0.01201,  0.01595,  0.01957,  0.02455,  0.02982,  0.03234,  0.03558,  0.03731,  0.04648,  0.05698,  0.06338]
            # neg_BS__length__min     = [-0.00060, -0.00131, -0.00299, -0.00720, -0.01000, -0.01314, -0.01732, -0.02476, -0.03791, -0.03499, -0.03994, -0.05024, -0.05879, -0.07375, -0.07493]
            # neg_BS__length__max     = [-0.00013, -0.00038, -0.00119, -0.00170, -0.00230, -0.00369, -0.00379, -0.00614, -0.00789, -0.01013, -0.01253, -0.01227, -0.01719, -0.02582, -0.02320]
            # neg_BS__length__avg     = [-0.00036, -0.00079, -0.00208, -0.00418, -0.00622, -0.00857, -0.01048, -0.01360, -0.01716, -0.01927, -0.02311, -0.02624, -0.03278, -0.04324, -0.05120]
            # neg_BS__angle__min      = [-0.00001, -0.00003, -0.00020, -0.00074, -0.00187, -0.00307, -0.00569, -0.00903, -0.01634, -0.01784, -0.02770, -0.02775, -0.05914, -0.07061, -0.14157]
            # neg_BS__angle__max      = [-0.00000, -0.00000, -0.00004, -0.00010, -0.00017, -0.00056, -0.00050, -0.00108, -0.00135, -0.00338, -0.00447, -0.00450, -0.01019, -0.01827, -0.02789]
            # neg_BS__angle__avg      = [-0.00000, -0.00001, -0.00009, -0.00039, -0.00085, -0.00162, -0.00248, -0.00413, -0.00661, -0.00821, -0.01187, -0.01450, -0.02582, -0.04092, -0.06075]
            # ------------->>>>>>>>>>>>>
            
            
            
            # 30
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   -4
            # angle_loss__opt_value   = [ 0.06969,  0.13031,  0.33043,  0.46900,  0.45934,  0.36852,  0.28830,  0.18268,  0.01549, -0.15126, -0.36464, -0.40202, -0.81864, -1.18799, -1.42677]
            # length_retention_loss__v= [ 0.00105,  0.00213,  0.00405,  0.00713,  0.00721,  0.00521,  0.00435,  0.00264, -0.00028, -0.00245, -0.00625, -0.00606, -0.01262, -0.01692, -0.02126]
            # avg_abs_final_grad__avg = [ 0.00257,  0.00482,  0.01312,  0.02619,  0.03705,  0.05099,  0.06224,  0.07504,  0.09047,  0.10252,  0.11619,  0.12004,  0.14755,  0.17348,  0.19586]
            # neg_BS__length__min     = [-0.00246, -0.00517, -0.01578, -0.02857, -0.03490, -0.03649, -0.03843, -0.03927, -0.03751, -0.03952, -0.03931, -0.03938, -0.03826, -0.04471, -0.04811]
            # neg_BS__length__max     = [-0.00112, -0.00224, -0.00550, -0.01382, -0.01708, -0.02143, -0.02824, -0.02992, -0.03009, -0.03027, -0.02853, -0.02852, -0.02680, -0.02826, -0.03184]
            # neg_BS__length__avg     = [-0.00163, -0.00310, -0.00902, -0.01865, -0.02543, -0.03020, -0.03319, -0.03394, -0.03376, -0.03392, -0.03270, -0.03284, -0.03411, -0.03534, -0.03629]
            # neg_BS__angle__min      = [-0.00074, -0.00380, -0.02991, -0.15389, -0.25632, -0.52025, -0.72460, -0.81138, -0.92024, -1.20557, -1.23416, -1.18392, -1.31595, -1.41244, -1.35538]
            # neg_BS__angle__max      = [-0.00015, -0.00064, -0.00513, -0.02560, -0.04636, -0.14567, -0.23275, -0.30444, -0.43749, -0.52641, -0.62442, -0.66300, -0.89144, -0.93416, -1.12846]
            # neg_BS__angle__avg      = [-0.00035, -0.00132, -0.01235, -0.06392, -0.13689, -0.25868, -0.37223, -0.50338, -0.63686, -0.74038, -0.84703, -0.88615, -1.04015, -1.16038, -1.24822]
            # ------------->>>>>>>>>>>>>                                  ***       ***
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   -2.0
            # angle_loss__opt_value   = [ 0.03737,  0.07334,  0.18499,  0.36177,  0.47964,  0.54293,  0.53146,  0.46953,  0.43794,  0.40177,  0.36683,  0.33344,  0.17530, -0.17218, -0.55541]
            # length_retention_loss__v= [ 0.00041,  0.00050,  0.00269,  0.00553,  0.00665,  0.00798,  0.00876,  0.00760,  0.00641,  0.00624,  0.00527,  0.00475,  0.00273, -0.00260, -0.00778]
            # avg_abs_final_grad__avg = [ 0.00136,  0.00269,  0.00686,  0.01391,  0.02052,  0.02767,  0.03462,  0.04332,  0.04858,  0.05525,  0.06297,  0.06878,  0.08281,  0.09717,  0.11180]
            # neg_BS__length__min     = [-0.00105, -0.00207, -0.00515, -0.01202, -0.01772, -0.02267, -0.02859, -0.03294, -0.03291, -0.03595, -0.03637, -0.03911, -0.03684, -0.03447, -0.03729]
            # neg_BS__length__max     = [-0.00064, -0.00132, -0.00349, -0.00733, -0.01204, -0.01472, -0.01835, -0.02424, -0.02564, -0.02656, -0.02878, -0.02647, -0.02630, -0.02583, -0.02584]
            # neg_BS__length__avg     = [-0.00081, -0.00162, -0.00424, -0.00915, -0.01402, -0.01875, -0.02304, -0.02837, -0.02992, -0.03102, -0.03270, -0.03196, -0.03092, -0.02984, -0.03035]
            # neg_BS__angle__min      = [-0.00012, -0.00043, -0.00328, -0.01361, -0.03916, -0.07412, -0.16879, -0.21586, -0.35763, -0.40947, -0.50934, -0.60422, -0.79755, -1.04911, -1.16670]
            # neg_BS__angle__max      = [-0.00005, -0.00022, -0.00170, -0.00733, -0.01894, -0.03297, -0.06074, -0.11299, -0.15631, -0.20608, -0.29187, -0.33233, -0.48017, -0.59066, -0.77897]
            # neg_BS__angle__avg      = [-0.00008, -0.00031, -0.00217, -0.01025, -0.02564, -0.05273, -0.09301, -0.16136, -0.21306, -0.28325, -0.37272, -0.45105, -0.61942, -0.77967, -0.91996]
            # ------------->>>>>>>>>>>>>                                                        **      **
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   -1.0
            # angle_loss__opt_value   = [ 0.02865,  0.05740,  0.14136,  0.28052,  0.40065,  0.49090,  0.55100,  0.56094,  0.54063,  0.50260,  0.45486,  0.42375,  0.40914,  0.36182,  0.13228]
            # length_retention_loss__v= [-0.00041,  0.00085,  0.00191,  0.00437,  0.00600,  0.00731,  0.00874,  0.00892,  0.00819,  0.00722,  0.00727,  0.00707,  0.00635,  0.00547,  0.00166]
            # avg_abs_final_grad__avg = [ 0.00104,  0.00211,  0.00519,  0.01054,  0.01570,  0.02071,  0.02650,  0.03167,  0.03665,  0.04180,  0.04735,  0.05235,  0.06141,  0.07262,  0.08481]
            # neg_BS__length__min     = [-0.00068, -0.00156, -0.00387, -0.00866, -0.01150, -0.01635, -0.02041, -0.02362, -0.02686, -0.03120, -0.03279, -0.03582, -0.03585, -0.03323, -0.03184]
            # neg_BS__length__max     = [-0.00053, -0.00108, -0.00256, -0.00562, -0.00851, -0.01126, -0.01513, -0.01828, -0.02165, -0.02367, -0.02602, -0.02580, -0.02794, -0.02468, -0.02336]
            # neg_BS__length__avg     = [-0.00060, -0.00126, -0.00310, -0.00663, -0.00984, -0.01350, -0.01734, -0.02105, -0.02428, -0.02687, -0.02959, -0.03043, -0.03190, -0.03005, -0.02804]
            # neg_BS__angle__min      = [-0.00005, -0.00026, -0.00153, -0.00730, -0.01654, -0.03090, -0.05255, -0.08858, -0.12357, -0.17164, -0.21950, -0.29617, -0.41759, -0.58698, -0.82924]
            # neg_BS__angle__max      = [-0.00004, -0.00014, -0.00094, -0.00406, -0.01046, -0.02177, -0.03484, -0.05765, -0.08270, -0.10823, -0.16198, -0.19870, -0.31168, -0.42246, -0.57270]
            # neg_BS__angle__avg      = [-0.00004, -0.00018, -0.00114, -0.00515, -0.01252, -0.02403, -0.04390, -0.06778, -0.09853, -0.13763, -0.18872, -0.24086, -0.34975, -0.50469, -0.66645]
            # ------------->>>>>>>>>>>>>                                                                        *******
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   0.0
            # angle_loss__opt_value   = [ 0.02189,  0.04375,  0.10893,  0.21756,  0.31991,  0.40763,  0.48205,  0.53358,  0.55891,  0.56033,  0.54231,  0.51770,  0.46707,  0.42958,  0.42908]
            # length_retention_loss__v= [ 0.00089, -0.00018,  0.00178,  0.00293,  0.00477,  0.00546,  0.00724,  0.00731,  0.00905,  0.00857,  0.00840,  0.00837,  0.00730,  0.00627,  0.00658]
            # avg_abs_final_grad__avg = [ 0.00080,  0.00160,  0.00399,  0.00800,  0.01200,  0.01600,  0.01999,  0.02399,  0.02802,  0.03200,  0.03601,  0.03998,  0.04804,  0.05604,  0.06396]
            # neg_BS__length__min     = [-0.00053, -0.00104, -0.00281, -0.00534, -0.00857, -0.01187, -0.01498, -0.01723, -0.02090, -0.02480, -0.02606, -0.02904, -0.03202, -0.03363, -0.03360]
            # neg_BS__length__max     = [-0.00041, -0.00083, -0.00209, -0.00424, -0.00663, -0.00925, -0.01160, -0.01352, -0.01611, -0.01813, -0.02040, -0.02168, -0.02514, -0.02592, -0.02603]
            # neg_BS__length__avg     = [-0.00047, -0.00093, -0.00241, -0.00490, -0.00756, -0.01021, -0.01335, -0.01557, -0.01813, -0.02152, -0.02365, -0.02563, -0.02883, -0.03021, -0.03054]
            # neg_BS__angle__min      = [-0.00002, -0.00010, -0.00066, -0.00293, -0.00691, -0.01338, -0.02226, -0.03539, -0.05124, -0.06983, -0.09472, -0.12381, -0.19853, -0.29299, -0.40586]
            # neg_BS__angle__max      = [-0.00002, -0.00009, -0.00062, -0.00269, -0.00632, -0.01224, -0.02046, -0.03161, -0.04691, -0.06316, -0.08704, -0.11077, -0.18165, -0.26413, -0.37018]
            # neg_BS__angle__avg      = [-0.00002, -0.00010, -0.00064, -0.00276, -0.00665, -0.01268, -0.02143, -0.03307, -0.04814, -0.06711, -0.09111, -0.11859, -0.19060, -0.27993, -0.38718]
            # ------------->>>>>>>>>>>>>                                                                                     **        **
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   1.0
            # angle_loss__opt_value   = [ 0.01707,  0.03421,  0.08641,  0.17034,  0.25326,  0.32835,  0.39724,  0.45577,  0.50869,  0.53919,  0.55263,  0.55470,  0.52675,  0.49563,  0.45752]
            # length_retention_loss__v= [ 0.00019,  0.00030,  0.00099,  0.00222,  0.00400,  0.00491,  0.00631,  0.00704,  0.00775,  0.00801,  0.00890,  0.00854,  0.00794,  0.00779,  0.00713]
            # avg_abs_final_grad__avg = [ 0.00062,  0.00125,  0.00313,  0.00623,  0.00937,  0.01246,  0.01544,  0.01844,  0.02207,  0.02495,  0.02796,  0.03113,  0.03736,  0.04296,  0.04974]
            # neg_BS__length__min     = [-0.00043, -0.00084, -0.00226, -0.00459, -0.00676, -0.00881, -0.01170, -0.01386, -0.01675, -0.02059, -0.02171, -0.02416, -0.02785, -0.03204, -0.03324]
            # neg_BS__length__max     = [-0.00030, -0.00066, -0.00168, -0.00330, -0.00470, -0.00696, -0.00837, -0.01069, -0.01268, -0.01438, -0.01704, -0.01810, -0.02186, -0.02317, -0.02377]
            # neg_BS__length__avg     = [-0.00037, -0.00074, -0.00189, -0.00377, -0.00590, -0.00795, -0.00997, -0.01219, -0.01456, -0.01681, -0.01885, -0.02077, -0.02450, -0.02667, -0.02938]
            # neg_BS__angle__min      = [-0.00002, -0.00007, -0.00045, -0.00197, -0.00457, -0.00871, -0.01426, -0.02225, -0.03235, -0.04428, -0.05342, -0.07690, -0.11895, -0.18426, -0.24726]
            # neg_BS__angle__max      = [-0.00001, -0.00005, -0.00031, -0.00141, -0.00302, -0.00622, -0.00949, -0.01402, -0.02253, -0.02959, -0.04038, -0.04726, -0.08192, -0.11567, -0.18168]
            # neg_BS__angle__avg      = [-0.00001, -0.00006, -0.00039, -0.00163, -0.00388, -0.00731, -0.01191, -0.01801, -0.02748, -0.03652, -0.04890, -0.06507, -0.10285, -0.14815, -0.21561]
            # ------------->>>>>>>>>>>>>                                                                                                        **        **
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   2.0
            # angle_loss__opt_value   = [ 0.01342,  0.02752,  0.06693,  0.13603,  0.19289,  0.26232,  0.31988,  0.37944,  0.42143,  0.46252,  0.50162,  0.51859,  0.54694,  0.53307,  0.51619]
            # length_retention_loss__v= [-0.00023,  0.00183,  0.00054,  0.00237,  0.00262,  0.00390,  0.00609,  0.00632,  0.00649,  0.00713,  0.00788,  0.00827,  0.00843,  0.00732,  0.00773]
            # avg_abs_final_grad__avg = [ 0.00049,  0.00100,  0.00243,  0.00493,  0.00708,  0.00971,  0.01209,  0.01455,  0.01665,  0.01917,  0.02196,  0.02414,  0.03005,  0.03492,  0.03812]
            # neg_BS__length__min     = [-0.00036, -0.00071, -0.00172, -0.00369, -0.00525, -0.00807, -0.00924, -0.01117, -0.01385, -0.01532, -0.01755, -0.02059, -0.02327, -0.02663, -0.03050]
            # neg_BS__length__max     = [-0.00022, -0.00051, -0.00119, -0.00252, -0.00353, -0.00499, -0.00587, -0.00722, -0.00891, -0.01050, -0.01060, -0.01209, -0.01567, -0.01993, -0.02114]
            # neg_BS__length__avg     = [-0.00029, -0.00061, -0.00146, -0.00309, -0.00448, -0.00619, -0.00775, -0.00966, -0.01121, -0.01280, -0.01494, -0.01658, -0.02049, -0.02339, -0.02481]
            # neg_BS__angle__min      = [-0.00001, -0.00005, -0.00033, -0.00135, -0.00310, -0.00595, -0.01021, -0.01397, -0.02184, -0.02690, -0.04238, -0.05688, -0.08375, -0.12236, -0.16117]
            # neg_BS__angle__max      = [-0.00001, -0.00003, -0.00016, -0.00072, -0.00127, -0.00297, -0.00481, -0.00746, -0.00859, -0.01360, -0.01893, -0.01675, -0.03690, -0.06548, -0.06435]
            # neg_BS__angle__avg      = [-0.00001, -0.00004, -0.00024, -0.00104, -0.00224, -0.00441, -0.00715, -0.01094, -0.01502, -0.02076, -0.02911, -0.03673, -0.06251, -0.09240, -0.11725]
            # ------------->>>>>>>>>>>>>                                                                                                                            *****
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   3.0
            # angle_loss__opt_value   = [ 0.01041,  0.02140,  0.05230,  0.10943,  0.16019,  0.21069,  0.26008,  0.30644,  0.35687,  0.38667,  0.41853,  0.46594,  0.50577,  0.52347,  0.52309]
            # length_retention_loss__v= [-0.00060, -0.00070,  0.00121,  0.00116,  0.00229,  0.00301,  0.00294,  0.00433,  0.00503,  0.00645,  0.00584,  0.00654,  0.00744,  0.00726,  0.00812]
            # avg_abs_final_grad__avg = [ 0.00038,  0.00077,  0.00188,  0.00396,  0.00583,  0.00772,  0.00956,  0.01152,  0.01372,  0.01504,  0.01685,  0.01955,  0.02349,  0.02790,  0.03028]
            # neg_BS__length__min     = [-0.00031, -0.00069, -0.00163, -0.00315, -0.00482, -0.00652, -0.00823, -0.00954, -0.01152, -0.01299, -0.01529, -0.01694, -0.02085, -0.02804, -0.02618]
            # neg_BS__length__max     = [-0.00015, -0.00035, -0.00080, -0.00188, -0.00293, -0.00332, -0.00453, -0.00593, -0.00691, -0.00858, -0.00816, -0.01011, -0.01096, -0.01600, -0.01565]
            # neg_BS__length__avg     = [-0.00023, -0.00047, -0.00117, -0.00247, -0.00376, -0.00495, -0.00632, -0.00755, -0.00906, -0.01041, -0.01156, -0.01361, -0.01637, -0.01943, -0.02091]
            # neg_BS__angle__min      = [-0.00001, -0.00004, -0.00023, -0.00092, -0.00261, -0.00447, -0.00683, -0.01042, -0.01532, -0.02471, -0.02869, -0.03679, -0.05720, -0.09488, -0.14183]
            # neg_BS__angle__max      = [-0.00000, -0.00001, -0.00008, -0.00035, -0.00097, -0.00122, -0.00260, -0.00427, -0.00699, -0.00835, -0.00900, -0.01486, -0.01908, -0.03565, -0.03816]
            # neg_BS__angle__avg      = [-0.00001, -0.00002, -0.00015, -0.00069, -0.00157, -0.00292, -0.00459, -0.00696, -0.01048, -0.01288, -0.01708, -0.02410, -0.03849, -0.05835, -0.07479]
            # ------------->>>>>>>>>>>>>
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   4.0
            # angle_loss__opt_value   = [ 0.00868,  0.01773,  0.04459,  0.08747,  0.13199,  0.17903,  0.21975,  0.26365,  0.29395,  0.32637,  0.36203,  0.39963,  0.45015,  0.48244,  0.49963]
            # length_retention_loss__v= [ 0.00028, -0.00072,  0.00146,  0.00193,  0.00181,  0.00332,  0.00364,  0.00495,  0.00478,  0.00436,  0.00562,  0.00589,  0.00626,  0.00768,  0.00730]
            # avg_abs_final_grad__avg = [ 0.00031,  0.00064,  0.00160,  0.00315,  0.00477,  0.00651,  0.00796,  0.00977,  0.01096,  0.01234,  0.01395,  0.01600,  0.01912,  0.02274,  0.02524]
            # neg_BS__length__min     = [-0.00028, -0.00054, -0.00149, -0.00269, -0.00385, -0.00549, -0.00672, -0.00933, -0.00928, -0.01170, -0.01399, -0.01608, -0.01957, -0.02049, -0.02453]
            # neg_BS__length__max     = [-0.00013, -0.00030, -0.00081, -0.00117, -0.00177, -0.00279, -0.00322, -0.00450, -0.00588, -0.00584, -0.00663, -0.00771, -0.01007, -0.01044, -0.01441]
            # neg_BS__length__avg     = [-0.00020, -0.00040, -0.00102, -0.00202, -0.00316, -0.00432, -0.00513, -0.00648, -0.00739, -0.00866, -0.00988, -0.01148, -0.01384, -0.01630, -0.01786]
            # neg_BS__angle__min      = [-0.00001, -0.00003, -0.00018, -0.00071, -0.00160, -0.00311, -0.00550, -0.00921, -0.01111, -0.01598, -0.02187, -0.02634, -0.04457, -0.05569, -0.09683]
            # neg_BS__angle__max      = [-0.00000, -0.00001, -0.00008, -0.00024, -0.00031, -0.00092, -0.00112, -0.00207, -0.00395, -0.00513, -0.00528, -0.00765, -0.01378, -0.01637, -0.03236]
            # neg_BS__angle__avg      = [-0.00000, -0.00002, -0.00012, -0.00047, -0.00114, -0.00215, -0.00339, -0.00528, -0.00697, -0.00923, -0.01244, -0.01728, -0.02617, -0.04012, -0.05283]
            # ------------->>>>>>>>>>>>>
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   6
            # angle_loss__opt_value   = [ 0.00584,  0.01165,  0.02831,  0.05821,  0.08948,  0.11601,  0.15177,  0.18146,  0.20751,  0.22871,  0.24309,  0.29122,  0.31039,  0.34490,  0.40183]
            # length_retention_loss__v= [ 0.00041,  0.00061,  0.00146,  0.00131,  0.00149,  0.00174,  0.00204,  0.00306,  0.00308,  0.00330,  0.00376,  0.00408,  0.00530,  0.00499,  0.00638]
            # avg_abs_final_grad__avg = [ 0.00021,  0.00042,  0.00102,  0.00207,  0.00319,  0.00413,  0.00544,  0.00657,  0.00748,  0.00835,  0.00893,  0.01090,  0.01184,  0.01380,  0.01746]
            # neg_BS__length__min     = [-0.00021, -0.00036, -0.00095, -0.00191, -0.00354, -0.00437, -0.00561, -0.00720, -0.00775, -0.00994, -0.01158, -0.01171, -0.01366, -0.01556, -0.02148]
            # neg_BS__length__max     = [-0.00009, -0.00018, -0.00030, -0.00083, -0.00150, -0.00165, -0.00230, -0.00276, -0.00335, -0.00305, -0.00346, -0.00511, -0.00557, -0.00666, -0.00904]
            # neg_BS__length__avg     = [-0.00014, -0.00027, -0.00066, -0.00140, -0.00222, -0.00293, -0.00383, -0.00464, -0.00549, -0.00629, -0.00659, -0.00819, -0.00911, -0.01073, -0.01375]
            # neg_BS__angle__min      = [-0.00000, -0.00002, -0.00012, -0.00041, -0.00119, -0.00195, -0.00357, -0.00585, -0.00795, -0.01114, -0.01650, -0.02152, -0.02643, -0.03674, -0.05458]
            # neg_BS__angle__max      = [-0.00000, -0.00000, -0.00001, -0.00009, -0.00031, -0.00035, -0.00061, -0.00103, -0.00151, -0.00147, -0.00179, -0.00337, -0.00417, -0.00746, -0.01364]
            # neg_BS__angle__avg      = [-0.00000, -0.00001, -0.00006, -0.00024, -0.00060, -0.00106, -0.00190, -0.00288, -0.00398, -0.00538, -0.00620, -0.00937, -0.01219, -0.01800, -0.02926]
            # ------------->>>>>>>>>>>>>
            # dim = 100
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   8
            # angle_loss__opt_value   = [ 0.00426,  0.00889,  0.02252,  0.04095,  0.06331,  0.08316,  0.10896,  0.12118,  0.15914,  0.16869,  0.19491,  0.18705,  0.23744,  0.27794,  0.28749]
            # length_retention_loss__v= [-0.00022,  0.00023,  0.00134,  0.00104,  0.00074,  0.00087,  0.00221,  0.00310,  0.00269,  0.00211,  0.00289,  0.00215,  0.00331,  0.00375,  0.00504]
            # avg_abs_final_grad__avg = [ 0.00015,  0.00032,  0.00081,  0.00146,  0.00223,  0.00295,  0.00386,  0.00428,  0.00565,  0.00596,  0.00690,  0.00663,  0.00875,  0.01077,  0.01134]
            # neg_BS__length__min     = [-0.00018, -0.00030, -0.00100, -0.00148, -0.00249, -0.00363, -0.00416, -0.00580, -0.00662, -0.00687, -0.01035, -0.00844, -0.01289, -0.01448, -0.01443]
            # neg_BS__length__max     = [-0.00005, -0.00007, -0.00036, -0.00042, -0.00068, -0.00132, -0.00145, -0.00230, -0.00248, -0.00301, -0.00292, -0.00318, -0.00472, -0.00555, -0.00608]
            # neg_BS__length__avg     = [-0.00010, -0.00022, -0.00057, -0.00103, -0.00167, -0.00220, -0.00293, -0.00340, -0.00442, -0.00492, -0.00572, -0.00579, -0.00745, -0.00952, -0.01017]
            # neg_BS__angle__min      = [-0.00000, -0.00001, -0.00014, -0.00034, -0.00076, -0.00142, -0.00244, -0.00451, -0.00557, -0.00572, -0.01716, -0.00917, -0.01871, -0.02939, -0.03534]
            # neg_BS__angle__max      = [-0.00000, -0.00000, -0.00002, -0.00004, -0.00007, -0.00029, -0.00034, -0.00068, -0.00093, -0.00131, -0.00155, -0.00162, -0.00387, -0.00464, -0.00683]
            # neg_BS__angle__avg      = [-0.00000, -0.00001, -0.00005, -0.00016, -0.00040, -0.00068, -0.00122, -0.00160, -0.00285, -0.00339, -0.00497, -0.00515, -0.00858, -0.01464, -0.01743]
            # ------------->>>>>>>>>>>>>
            
            
            
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   -4
            # angle_loss__opt_value   = [ 0.03527,  0.06872,  0.18063,  0.35081,  0.48506,  0.54218,  0.55624,  0.53525,  0.47986,  0.46343,  0.43913,  0.43836,  0.25093, -0.06074, -0.57835]
            # length_retention_loss__v= [ 0.00037, -0.00055,  0.00072,  0.00180,  0.00241,  0.00298,  0.00243,  0.00276,  0.00219,  0.00219,  0.00207,  0.00192,  0.00090, -0.00017, -0.00236]
            # avg_abs_final_grad__avg = [ 0.00040,  0.00077,  0.00205,  0.00416,  0.00636,  0.00795,  0.01026,  0.01187,  0.01469,  0.01631,  0.01886,  0.01949,  0.02415,  0.02750,  0.03194]
            # neg_BS__length__min     = [-0.00026, -0.00051, -0.00148, -0.00308, -0.00464, -0.00571, -0.00751, -0.00852, -0.00947, -0.01001, -0.00954, -0.01005, -0.00935, -0.00957, -0.01040]
            # neg_BS__length__max     = [-0.00023, -0.00046, -0.00126, -0.00254, -0.00397, -0.00506, -0.00674, -0.00736, -0.00873, -0.00926, -0.00923, -0.00915, -0.00907, -0.00889, -0.00912]
            # neg_BS__length__avg     = [-0.00024, -0.00049, -0.00131, -0.00276, -0.00434, -0.00542, -0.00698, -0.00792, -0.00904, -0.00954, -0.00947, -0.00972, -0.00918, -0.00909, -0.00963]
            # neg_BS__angle__min      = [-0.00007, -0.00028, -0.00245, -0.01009, -0.02752, -0.04540, -0.09181, -0.12712, -0.19879, -0.26185, -0.45038, -0.40842, -0.62260, -0.77578, -0.94785]
            # neg_BS__angle__max      = [-0.00006, -0.00022, -0.00173, -0.00770, -0.01860, -0.03503, -0.06436, -0.09875, -0.16945, -0.21262, -0.27421, -0.32428, -0.52176, -0.65714, -0.78145]
            # neg_BS__angle__avg      = [-0.00006, -0.00025, -0.00191, -0.00886, -0.02358, -0.04016, -0.07565, -0.10890, -0.18657, -0.24062, -0.34017, -0.36361, -0.56252, -0.69928, -0.86075]
            # ------------->>>>>>>>>>>>>                                                              *******
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   -2.0
            # angle_loss__opt_value   = [ 0.02818,  0.05619,  0.14114,  0.27443,  0.39203,  0.48851,  0.54573,  0.56319,  0.55315,  0.51619,  0.49085,  0.46830,  0.45780,  0.39599,  0.14980]
            # length_retention_loss__v= [ 0.00022,  0.00031,  0.00069,  0.00129,  0.00211,  0.00265,  0.00308,  0.00284,  0.00218,  0.00208,  0.00204,  0.00225,  0.00214,  0.00157,  0.00098]
            # avg_abs_final_grad__avg = [ 0.00032,  0.00063,  0.00159,  0.00317,  0.00474,  0.00637,  0.00795,  0.00953,  0.01097,  0.01290,  0.01408,  0.01580,  0.01889,  0.02201,  0.02534]
            # neg_BS__length__min     = [-0.00020, -0.00039, -0.00105, -0.00213, -0.00320, -0.00451, -0.00571, -0.00695, -0.00772, -0.00871, -0.00898, -0.00965, -0.00972, -0.00937, -0.00879]
            # neg_BS__length__max     = [-0.00019, -0.00038, -0.00095, -0.00198, -0.00310, -0.00410, -0.00527, -0.00611, -0.00707, -0.00811, -0.00870, -0.00911, -0.00935, -0.00889, -0.00851]
            # neg_BS__length__avg     = [-0.00019, -0.00039, -0.00100, -0.00207, -0.00314, -0.00429, -0.00548, -0.00648, -0.00743, -0.00845, -0.00880, -0.00940, -0.00949, -0.00921, -0.00865]
            # neg_BS__angle__min      = [-0.00004, -0.00017, -0.00117, -0.00493, -0.01188, -0.02435, -0.04134, -0.06982, -0.09047, -0.14011, -0.16903, -0.22961, -0.35729, -0.48630, -0.62848]
            # neg_BS__angle__max      = [-0.00004, -0.00016, -0.00101, -0.00457, -0.01126, -0.02185, -0.03721, -0.05706, -0.08163, -0.12385, -0.15789, -0.20033, -0.32362, -0.45999, -0.59838]
            # neg_BS__angle__avg      = [-0.00004, -0.00016, -0.00109, -0.00473, -0.01158, -0.02293, -0.03909, -0.06108, -0.08664, -0.13086, -0.16282, -0.21812, -0.33636, -0.47133, -0.61688]
            # ------------->>>>>>>>>>>>>                                                                ***       ***
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   -1.0
            # angle_loss__opt_value   = [ 0.02535,  0.05032,  0.12399,  0.24551,  0.35706,  0.44979,  0.51956,  0.55586,  0.56279,  0.54686,  0.52158,  0.49317,  0.46220,  0.45051,  0.37396]
            # length_retention_loss__v= [ 0.00057,  0.00021,  0.00028,  0.00191,  0.00168,  0.00216,  0.00226,  0.00307,  0.00292,  0.00250,  0.00285,  0.00267,  0.00229,  0.00269,  0.00146]
            # avg_abs_final_grad__avg = [ 0.00028,  0.00057,  0.00140,  0.00281,  0.00424,  0.00565,  0.00707,  0.00846,  0.00993,  0.01137,  0.01262,  0.01411,  0.01690,  0.02010,  0.02254]
            # neg_BS__length__min     = [-0.00018, -0.00036, -0.00090, -0.00188, -0.00289, -0.00390, -0.00496, -0.00591, -0.00717, -0.00763, -0.00850, -0.00900, -0.00958, -0.00982, -0.00930]
            # neg_BS__length__max     = [-0.00017, -0.00034, -0.00085, -0.00174, -0.00269, -0.00374, -0.00461, -0.00567, -0.00654, -0.00732, -0.00779, -0.00872, -0.00911, -0.00942, -0.00885]
            # neg_BS__length__avg     = [-0.00018, -0.00035, -0.00088, -0.00181, -0.00278, -0.00382, -0.00478, -0.00577, -0.00686, -0.00753, -0.00819, -0.00885, -0.00945, -0.00964, -0.00902]
            # neg_BS__angle__min      = [-0.00003, -0.00014, -0.00085, -0.00381, -0.00905, -0.01743, -0.02972, -0.04663, -0.06990, -0.09622, -0.12400, -0.17262, -0.26677, -0.40575, -0.50502]
            # neg_BS__angle__max      = [-0.00003, -0.00013, -0.00081, -0.00357, -0.00874, -0.01684, -0.02851, -0.04354, -0.06485, -0.09078, -0.12062, -0.15805, -0.24984, -0.36549, -0.48253]
            # neg_BS__angle__avg      = [-0.00003, -0.00013, -0.00083, -0.00364, -0.00894, -0.01718, -0.02927, -0.04505, -0.06689, -0.09404, -0.12252, -0.16279, -0.25563, -0.38663, -0.49382]
            # ------------->>>>>>>>>>>>>                                                                          ***       ***          
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   0.0
            # angle_loss__opt_value   = [ 0.02251,  0.04494,  0.11211,  0.22123,  0.32288,  0.41328,  0.48559,  0.53611,  0.56105,  0.56302,  0.54726,  0.52258,  0.47577,  0.46661,  0.45312]
            # length_retention_loss__v= [ 0.00014,  0.00046,  0.00033,  0.00056,  0.00227,  0.00220,  0.00199,  0.00214,  0.00215,  0.00218,  0.00308,  0.00229,  0.00171,  0.00179,  0.00246]
            # avg_abs_final_grad__avg = [ 0.00025,  0.00050,  0.00126,  0.00252,  0.00379,  0.00505,  0.00631,  0.00757,  0.00883,  0.01009,  0.01136,  0.01262,  0.01515,  0.01767,  0.02019]
            # neg_BS__length__min     = [-0.00016, -0.00032, -0.00081, -0.00166, -0.00247, -0.00335, -0.00435, -0.00535, -0.00622, -0.00701, -0.00789, -0.00836, -0.00951, -0.00987, -0.00947]
            # neg_BS__length__max     = [-0.00015, -0.00031, -0.00077, -0.00158, -0.00237, -0.00328, -0.00396, -0.00493, -0.00587, -0.00654, -0.00736, -0.00812, -0.00916, -0.00954, -0.00900]
            # neg_BS__length__avg     = [-0.00015, -0.00031, -0.00079, -0.00162, -0.00243, -0.00332, -0.00420, -0.00509, -0.00603, -0.00680, -0.00766, -0.00820, -0.00931, -0.00968, -0.00927]
            # neg_BS__angle__min      = [-0.00003, -0.00010, -0.00067, -0.00287, -0.00696, -0.01329, -0.02223, -0.03434, -0.05016, -0.06958, -0.09402, -0.12299, -0.19524, -0.28678, -0.39204]
            # neg_BS__angle__max      = [-0.00003, -0.00010, -0.00067, -0.00286, -0.00693, -0.01319, -0.02208, -0.03410, -0.04972, -0.06935, -0.09301, -0.12239, -0.19406, -0.28462, -0.38829]
            # neg_BS__angle__avg      = [-0.00003, -0.00010, -0.00067, -0.00287, -0.00695, -0.01324, -0.02218, -0.03424, -0.04990, -0.06945, -0.09362, -0.12277, -0.19474, -0.28590, -0.39055]
            # ------------->>>>>>>>>>>>>                                                                                              ***       ***
            # 1.0
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   1.0
            # angle_loss__opt_value   = [ 0.02008,  0.04010,  0.10129,  0.19897,  0.28919,  0.37638,  0.44935,  0.50789,  0.54502,  0.56267,  0.56071,  0.54565,  0.50188,  0.47204,  0.46597]
            # length_retention_loss__v= [-0.00065,  0.00026,  0.00081,  0.00062,  0.00167,  0.00187,  0.00217,  0.00268,  0.00254,  0.00257,  0.00179,  0.00251,  0.00247,  0.00270,  0.00229]
            # avg_abs_final_grad__avg = [ 0.00023,  0.00045,  0.00114,  0.00226,  0.00335,  0.00451,  0.00564,  0.00680,  0.00794,  0.00911,  0.01020,  0.01134,  0.01355,  0.01585,  0.01798]
            # neg_BS__length__min     = [-0.00014, -0.00028, -0.00073, -0.00147, -0.00223, -0.00308, -0.00383, -0.00476, -0.00552, -0.00636, -0.00724, -0.00792, -0.00883, -0.00940, -0.01007]
            # neg_BS__length__max     = [-0.00013, -0.00027, -0.00068, -0.00141, -0.00214, -0.00300, -0.00373, -0.00449, -0.00527, -0.00605, -0.00663, -0.00750, -0.00858, -0.00886, -0.00921]
            # neg_BS__length__avg     = [-0.00014, -0.00028, -0.00070, -0.00144, -0.00217, -0.00303, -0.00379, -0.00461, -0.00544, -0.00615, -0.00697, -0.00766, -0.00865, -0.00923, -0.00949]
            # neg_BS__angle__min      = [-0.00002, -0.00008, -0.00056, -0.00230, -0.00546, -0.01059, -0.01755, -0.02697, -0.03996, -0.05550, -0.07236, -0.09702, -0.15201, -0.22544, -0.30559]
            # neg_BS__angle__max      = [-0.00002, -0.00008, -0.00051, -0.00223, -0.00514, -0.00979, -0.01664, -0.02607, -0.03750, -0.05133, -0.06920, -0.08980, -0.14001, -0.21364, -0.29170]
            # neg_BS__angle__avg      = [-0.00002, -0.00008, -0.00054, -0.00227, -0.00531, -0.01023, -0.01713, -0.02654, -0.03852, -0.05393, -0.07144, -0.09352, -0.14707, -0.21903, -0.29819]
            # ------------->>>>>>>>>>>>>                                                                            ***             ***
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   2.0
            # angle_loss__opt_value   = [ 0.01780,  0.03601,  0.08992,  0.17622,  0.26349,  0.33937,  0.40982,  0.47407,  0.51717,  0.54804,  0.56227,  0.55850,  0.52888,  0.49655,  0.47506]
            # length_retention_loss__v= [-0.00029, -0.00020,  0.00007,  0.00050,  0.00073,  0.00145,  0.00196,  0.00231,  0.00244,  0.00275,  0.00276,  0.00246,  0.00233,  0.00183,  0.00217]
            # avg_abs_final_grad__avg = [ 0.00020,  0.00040,  0.00101,  0.00200,  0.00303,  0.00400,  0.00500,  0.00609,  0.00706,  0.00811,  0.00920,  0.01024,  0.01228,  0.01399,  0.01592]
            # neg_BS__length__min     = [-0.00013, -0.00026, -0.00067, -0.00130, -0.00210, -0.00268, -0.00357, -0.00429, -0.00504, -0.00579, -0.00659, -0.00710, -0.00831, -0.00938, -0.00974]
            # neg_BS__length__max     = [-0.00012, -0.00024, -0.00058, -0.00123, -0.00194, -0.00244, -0.00318, -0.00399, -0.00449, -0.00557, -0.00600, -0.00672, -0.00767, -0.00839, -0.00911]
            # neg_BS__length__avg     = [-0.00012, -0.00025, -0.00062, -0.00126, -0.00202, -0.00259, -0.00334, -0.00409, -0.00480, -0.00566, -0.00629, -0.00687, -0.00802, -0.00889, -0.00947]
            # neg_BS__angle__min      = [-0.00002, -0.00007, -0.00046, -0.00187, -0.00450, -0.00826, -0.01406, -0.02192, -0.03114, -0.04173, -0.05898, -0.07623, -0.11665, -0.17116, -0.24228]
            # neg_BS__angle__max      = [-0.00002, -0.00006, -0.00038, -0.00162, -0.00404, -0.00721, -0.01180, -0.01807, -0.02580, -0.03981, -0.05192, -0.07142, -0.11012, -0.14610, -0.19184]
            # neg_BS__angle__avg      = [-0.00002, -0.00007, -0.00042, -0.00175, -0.00429, -0.00786, -0.01305, -0.02057, -0.02919, -0.04088, -0.05565, -0.07273, -0.11501, -0.16135, -0.22310]
            # ------------->>>>>>>>>>>>>                                                                                                      *******
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   3.0
            # angle_loss__opt_value   = [ 0.01647,  0.03205,  0.08098,  0.16428,  0.23700,  0.31904,  0.37552,  0.44338,  0.48914,  0.52158,  0.55052,  0.55805,  0.55394,  0.52349,  0.48903]
            # length_retention_loss__v= [ 0.00029, -0.00032,  0.00039,  0.00082,  0.00136,  0.00177,  0.00143,  0.00184,  0.00250,  0.00292,  0.00254,  0.00269,  0.00270,  0.00260,  0.00184]
            # avg_abs_final_grad__avg = [ 0.00018,  0.00036,  0.00091,  0.00186,  0.00271,  0.00373,  0.00449,  0.00554,  0.00641,  0.00720,  0.00830,  0.00909,  0.01064,  0.01245,  0.01455]
            # neg_BS__length__min     = [-0.00012, -0.00025, -0.00058, -0.00126, -0.00184, -0.00256, -0.00305, -0.00385, -0.00455, -0.00511, -0.00609, -0.00666, -0.00758, -0.00869, -0.00923]
            # neg_BS__length__max     = [-0.00011, -0.00021, -0.00053, -0.00112, -0.00163, -0.00234, -0.00288, -0.00362, -0.00421, -0.00468, -0.00539, -0.00582, -0.00681, -0.00772, -0.00867]
            # neg_BS__length__avg     = [-0.00011, -0.00023, -0.00056, -0.00119, -0.00173, -0.00245, -0.00295, -0.00374, -0.00438, -0.00488, -0.00568, -0.00636, -0.00721, -0.00816, -0.00900]
            # neg_BS__angle__min      = [-0.00001, -0.00005, -0.00039, -0.00171, -0.00368, -0.00713, -0.01085, -0.01839, -0.02403, -0.03277, -0.04526, -0.06035, -0.08837, -0.13594, -0.19333]
            # neg_BS__angle__max      = [-0.00001, -0.00005, -0.00032, -0.00135, -0.00299, -0.00630, -0.00972, -0.01384, -0.02251, -0.02886, -0.04144, -0.04525, -0.07344, -0.10864, -0.16845]
            # neg_BS__angle__avg      = [-0.00001, -0.00005, -0.00034, -0.00151, -0.00338, -0.00678, -0.01031, -0.01670, -0.02345, -0.03098, -0.04357, -0.05496, -0.08108, -0.12080, -0.17985]
            # ------------->>>>>>>>>>>>>                                                                                              ***                  ***   
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   4.0
            # angle_loss__opt_value   = [ 0.01425,  0.02819,  0.07409,  0.14594,  0.20870,  0.28410,  0.35648,  0.39856,  0.44876,  0.48973,  0.52711,  0.54336,  0.55630,  0.54020,  0.51544]
            # length_retention_loss__v= [ 0.00062, -0.00018,  0.00002,  0.00044,  0.00065,  0.00177,  0.00207,  0.00195,  0.00190,  0.00254,  0.00282,  0.00236,  0.00266,  0.00266,  0.00209]
            # avg_abs_final_grad__avg = [ 0.00016,  0.00032,  0.00083,  0.00165,  0.00237,  0.00329,  0.00423,  0.00484,  0.00564,  0.00644,  0.00740,  0.00813,  0.00975,  0.01152,  0.01289]
            # neg_BS__length__min     = [-0.00010, -0.00021, -0.00056, -0.00110, -0.00168, -0.00231, -0.00308, -0.00342, -0.00408, -0.00450, -0.00522, -0.00612, -0.00700, -0.00782, -0.00895]
            # neg_BS__length__max     = [-0.00009, -0.00019, -0.00048, -0.00097, -0.00149, -0.00202, -0.00271, -0.00292, -0.00356, -0.00420, -0.00486, -0.00524, -0.00613, -0.00724, -0.00812]
            # neg_BS__length__avg     = [-0.00010, -0.00019, -0.00053, -0.00104, -0.00155, -0.00214, -0.00284, -0.00320, -0.00387, -0.00437, -0.00504, -0.00561, -0.00659, -0.00762, -0.00851]
            # neg_BS__angle__min      = [-0.00001, -0.00005, -0.00031, -0.00127, -0.00302, -0.00617, -0.01015, -0.01435, -0.01941, -0.02684, -0.03461, -0.05101, -0.07783, -0.11129, -0.15184]
            # neg_BS__angle__max      = [-0.00001, -0.00004, -0.00026, -0.00106, -0.00229, -0.00448, -0.00819, -0.01012, -0.01550, -0.02228, -0.03297, -0.03406, -0.05393, -0.09408, -0.12020]
            # neg_BS__angle__avg      = [-0.00001, -0.00004, -0.00029, -0.00118, -0.00257, -0.00522, -0.00914, -0.01241, -0.01767, -0.02404, -0.03358, -0.04249, -0.06649, -0.10104, -0.13453]
            # ------------->>>>>>>>>>>>>                                                                                                        ***                 ***
            # dim = 1000
            # cap_to_list               = [ 0.01,     0.02,     0.05,     0.10,     0.15,     0.20,     0.25,     0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80]
            # expansion_factor =   6
            # angle_loss__opt_value   = [ 0.01195,  0.02353,  0.06005,  0.11397,  0.17777,  0.23211,  0.28569,  0.34528,  0.38533,  0.43072,  0.46380,  0.49437,  0.52882,  0.55026,  0.54603]
            # length_retention_loss__v= [ 0.00025, -0.00051,  0.00017,  0.00000,  0.00075,  0.00156,  0.00173,  0.00199,  0.00230,  0.00208,  0.00254,  0.00262,  0.00271,  0.00290,  0.00288]
            # avg_abs_final_grad__avg = [ 0.00013,  0.00026,  0.00067,  0.00128,  0.00201,  0.00265,  0.00330,  0.00408,  0.00464,  0.00535,  0.00602,  0.00664,  0.00762,  0.00948,  0.01039]
            # neg_BS__length__min     = [-0.00009, -0.00018, -0.00046, -0.00087, -0.00138, -0.00183, -0.00242, -0.00315, -0.00331, -0.00387, -0.00468, -0.00518, -0.00558, -0.00692, -0.00755]
            # neg_BS__length__max     = [-0.00008, -0.00016, -0.00039, -0.00070, -0.00126, -0.00150, -0.00187, -0.00249, -0.00293, -0.00356, -0.00341, -0.00426, -0.00519, -0.00615, -0.00677]
            # neg_BS__length__avg     = [-0.00008, -0.00016, -0.00042, -0.00081, -0.00131, -0.00173, -0.00219, -0.00276, -0.00312, -0.00371, -0.00415, -0.00462, -0.00537, -0.00653, -0.00720]
            # neg_BS__angle__min      = [-0.00001, -0.00003, -0.00021, -0.00083, -0.00205, -0.00392, -0.00657, -0.01108, -0.01374, -0.01889, -0.02829, -0.03470, -0.04551, -0.07405, -0.10061]
            # neg_BS__angle__max      = [-0.00001, -0.00003, -0.00016, -0.00053, -0.00172, -0.00239, -0.00394, -0.00735, -0.00982, -0.01427, -0.01368, -0.02251, -0.03360, -0.05379, -0.06728]
            # neg_BS__angle__avg      = [-0.00001, -0.00003, -0.00019, -0.00072, -0.00186, -0.00339, -0.00548, -0.00878, -0.01175, -0.01627, -0.02176, -0.02710, -0.03788, -0.06519, -0.08214]
            # ------------->>>>>>>>>>>>>                                                                                                                                    *******
        
            pass
        
        if "result in short" and False:
            # dim = 10
            #           -4                -2.0               expansion_factor =              -1    
            #          0.05,            0.10,     0.15,      cap_to_list                     0.25, 
            #        0.09504,         0.28555,  0.29444      angle_loss__opt_value         0.45450,
            #        0.00598,         0.01339,  0.00925      length_retention_loss__v      0.02152,
            #        0.13488,         0.06319,  0.09832      avg_abs_final_grad__avg       0.09505,
            #       -0.06170,        -0.04119, -0.05836      neg_BS__length__avg          -0.05515,
            #       -0.13825,        -0.04458, -0.09407      neg_BS__angle__avg           -0.06216,
            # dim = 10
            #           0              1              2         2         expansion_factor =               3          4
            #          0.45,          0.50,          0.50,     0.60,      cap_to_list                     0.70,      0.70, 
            #        0.55013,       0.53192,       0.49110,  0.51242,     angle_loss__opt_value         0.48597,   0.44387,
            #        0.02688,       0.02583,       0.02271,  0.02126,     length_retention_loss__v      0.02325,   0.02159,
            #        0.11661,       0.09282,       0.07519,  0.08905,     avg_abs_final_grad__avg       0.08830,   0.07177,
            #       -0.06669,      -0.05332,      -0.04648, -0.05337,     neg_BS__length__avg          -0.05752,  -0.04949,
            #       -0.07374,      -0.04411,      -0.03154, -0.04812,     neg_BS__angle__avg           -0.06349,  -0.04883,
            
            
            # dim = 100
            #             -4      -4                 -2       -2             expansion_factor =                -1      
            #           0.10,     0.15,             0.20,     0.25,          cap_to_list                      0.30,    
            #         0.46900,  0.45934,          0.54293,  0.53146,         angle_loss__opt_value          0.56094,   
            #         0.00713,  0.00721,          0.00798,  0.00876,         length_retention_loss__v       0.00892,   
            #         0.02619,  0.03705,          0.02767,  0.03462,         avg_abs_final_grad__avg        0.03167,   
            #        -0.01865, -0.02543,          0.01875, -0.02304,         neg_BS__length__avg           -0.02105,   
            #        -0.06392, -0.13689,          0.05273, -0.09301,         neg_BS__angle__avg            -0.06778,   
            # dim = 100
            #           0         0             1         1         expansion_factor =            2
            #         0.35,     0.40,         0.45,     0.50,       cap_to_list                  0.60, 
            #         0.55891,  0.56033,      0.55263,  0.55470,    angle_loss__opt_value   =   0.54694,
            #         0.00905,  0.00857,      0.00890,  0.00854,    length_retention_loss__v=   0.00843,
            #         0.02802,  0.03200,      0.02796,  0.03113,    avg_abs_final_grad__avg =   0.03005,
            #        -0.01813, -0.02152,     -0.01885, -0.02077,    neg_BS__length__avg     =  -0.02049,
            #        -0.04814, -0.06711,     -0.04890, -0.06507,    neg_BS__angle__avg      =  -0.06251,
            
            
            # dim = 1000
            #       -4             -2        -2      expansion_factor =             -1         -1             0        0          expansion_factor =             1         1
            #      0.25,          0.25,     0.30,    cap_to_list               =    0.30,     0.35,         0.40,     0.45,       cap_to_list               =   0.30,     0.40,  
            #    0.55624,       0.54573,  0.56319,   angle_loss__opt_value   = [  0.55586,  0.56279,      0.56302,  0.54726,      angle_loss__opt_value   = [ 0.50789,  0.56267, 
            #    0.00243,       0.00308,  0.00284,   length_retention_loss__v= [  0.00307,  0.00292,      0.00218,  0.00308,      length_retention_loss__v= [ 0.00268,  0.00257, 
            #    0.01026,       0.00795,  0.00953,   avg_abs_final_grad__avg = [  0.00846,  0.00993,      0.01009,  0.01136,      avg_abs_final_grad__avg = [ 0.00680,  0.00911, 
            #   -0.00698,      -0.00548, -0.00648,   neg_BS__length__avg     = [ -0.00577, -0.00686,     -0.00680, -0.00766,      neg_BS__length__avg     = [-0.00461, -0.00615, 
            #   -0.07565,      -0.03909, -0.06108,   neg_BS__angle__avg      = [ -0.04505, -0.06689,     -0.06945, -0.09362,      neg_BS__angle__avg      = [-0.02654, -0.05393, 
            # dim = 1000
            #      2            expansion_factor =             3         3                   4          4         expansion_factor =              6
            #     0.45,         cap_to_list               =   0.40,     0.50,               0.45,      0.60,      cap_to_list                    0.70,  
            #   0.56227,        angle_loss__opt_value   = [ 0.52158,  0.55805,            0.52711,   0.55630,     angle_loss__opt_value   = ,  0.55026, 
            #   0.00276,        length_retention_loss__v= [ 0.00292,  0.00269,            0.00282,   0.00266,     length_retention_loss__v= ,  0.00290, 
            #   0.00920,        avg_abs_final_grad__avg = [ 0.00720,  0.00909,            0.00740,   0.00975,     avg_abs_final_grad__avg = ,  0.00948, 
            #  -0.00629,        neg_BS__length__avg     = [ 0.00488, -0.00636,           -0.00504,  -0.00659,     neg_BS__length__avg     = , -0.00653, 
            #  -0.05565,        neg_BS__angle__avg      = [ 0.03098, -0.05496,           -0.03358,  -0.06649,     neg_BS__angle__avg      = , -0.06519, 
            # ------------->>>>>>>>>>>>>                                                                                                      
            pass
        
        #------------------#------------------#------------------
        # dim_list =        [ 2, 10,100,1000]
        # test_time_list = [100,100,100, 20]
        # test_time_list = [10,10,10, 2]
        
        dim_list =       [ 10, 100,1000]
        test_time_list = [ 50, 30, 10]
        #test_time_list = [ 10, 6, 3]
        dim_list =       [1000]
        test_time_list = [ 5]
        for outter_iter_count in range(dim_list.__len__()):
            dim = dim_list[outter_iter_count]
            test_time = test_time_list[outter_iter_count]
            print(test_time)
            iota_of_dim = iota(dim)
            if dim>100:
                device = 'cuda'
                pass
            else:
                device = 'cpu'
                pass
        #------------------#------------------#------------------
        
            #------------------#------------------#------------------
            expansion_factor_list = [-4,-2.,-1., 0.,1.,2.,3.,4.,6,8]
            for expansion_factor in expansion_factor_list:
            #------------------#------------------#------------------

                # I chose some of the most useful measurement from the last test as ref.
                
                #len_loss__opt_value              = []  # dont modity this
                #len_loss__opt_prob               = []  # dont modity this
                angle_loss__opt_value            = []  # dont modity this
                #angle_loss__opt_prob             = []  # dont modity this
                length_retention_loss__opt_value = []  # dont modity this
                #length_retention_loss__opt_prob  = []  # dont modity this
                #avg_of_abs_of_final_grad__min    = []  # dont modity this
                #avg_of_abs_of_final_grad__max    = []  # dont modity this
                avg_of_abs_of_final_grad__avg    = []  # dont modity this
                
                neg_BS__length__min              = []  # dont modity this
                neg_BS__length__max              = []  # dont modity this
                neg_BS__length__avg              = []  # dont modity this
                neg_BS__angle__min               = []  # dont modity this
                neg_BS__angle__max               = []  # dont modity this
                neg_BS__angle__avg               = []  # dont modity this
                #------------------#------------------#------------------
                #cap_to_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1., 1.2, 1.5, 2.]
                #cap_to_list = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1., 1.2]
                cap_to_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8,]
                for cap_to in cap_to_list:
                #------------------#------------------#------------------
                
                    #_raw_result__len_loss__after_sub_before                 = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__angle_loss__after_sub_before             = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__length_retention_loss__after_sub_before  = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__avg_of_abs_of_final_grad                 = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__neg_BS__length                           = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__neg_BS__angle                           = torch.empty(size=[test_time])  # dont modity this
                    
                    for test_count in range(test_time):
                        #----------------#----------------#----------------#----------------
                        ori_mat = torch.randn(size=[dim,dim], device=device)
                        ori_mat = vector_length_norm(ori_mat)
                        # old code ori_mat = ____init_mat_with_row_len_is_1(dim, device=device)
                        
                        #<  measure the init>
                        before__len_loss, before__angle_loss, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                        before__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(ori_mat)
                        
                        #<  calc grad>
                        mat = ori_mat.detach().clone()
                        
                        manual__mat_matmul_mat__d_d = mat@(mat.T)
                        manual__mat_matmul_mat__d_d[iota_of_dim, iota_of_dim] = 0.
                        
                        #<  original grad>
                        ori__grad__d_d = manual__mat_matmul_mat__d_d@mat
                        ori__grad_len__dim = get_vector_length(ori__grad__d_d)
                        #<  original grad, but len into 1>
                        _temp_reverse_of__ori__grad_len__dim = ori__grad_len__dim.pow(-1.)
                        len_1__ori_grad__d_d = ori__grad__d_d.mul( \
                                expand_vec_to_matrix(_temp_reverse_of__ori__grad_len__dim, each_element_to="row"))
                        del _temp_reverse_of__ori__grad_len__dim # just in case.
                        assert _tensor_equal(get_vector_length(len_1__ori_grad__d_d), torch.ones(size=[dim], device=device))
                        
                        #<  scale it a bit, to make the distribution a bit wider>
                        #otherwise, they are a lot 0.8,0.9. Wider means most are 0.3 to 0.8.
                        #<old code/>xx__grad_len_sqr__dim = ori__grad__d_d.mul(ori__grad__d_d).sum(dim=1)#mul and then sum, it's a dot.
                        max_of__ori__grad_len__dim = ori__grad_len__dim.max()
                        ratio_of__grad_len__dim = ori__grad_len__dim/max_of__ori__grad_len__dim
                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this scaled_to_1__grad_len_sqr__dim is still one 1.0 and a lot 0.xx
                        assert ratio_of__grad_len__dim.le(1.).all()
                        assert ratio_of__grad_len__dim.eq(1.).sum() == 1
                        
                        grad_len__after_expansion__dim = ratio_of__grad_len__dim.pow(expansion_factor)
                        # assert grad_len__after_expansion__dim.le(1.).all() when expansion_factor is neg, this is wrong.
                        # assert grad_len__after_expansion__dim.eq(1.).sum() == 1 when expansion_factor is 0., this is wrong.
                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this grad_len__after_expansion__dim is still one 1.0 and a lot 0.xx
                        
                        if "visualization" and False:
                            #prin(expansion_factor)
                            visualize_this = grad_len__after_expansion__dim
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(tight_layout=True)
                            n_bins = dim//5
                            if n_bins<=3:
                                n_bins = 3
                                pass
                            ax.hist(visualize_this.tolist(), bins=n_bins)
                            plt.show()
                            pass
                        
                        #<  target length>
                        grad_length_target__dim = grad_len__after_expansion__dim*cap_to# cap_to is scalar
                        grad_length_target__dim__expand_dim = expand_vec_to_matrix(grad_length_target__dim, each_element_to="row")
                        
                        #<  finally 
                        useful_grad__d_d = len_1__ori_grad__d_d.mul(grad_length_target__dim__expand_dim)#row
                        # # assert _tensor_equal(get_vector_length(useful_grad__d_d), grad_length_target__dim)
                        
                        # debug only  vvvvvvvvvvvvvvvvvvvv
                        log10_of__mat = log10_avg_safe(mat)
                        log10_of__grad = log10_avg_safe(useful_grad__d_d)
                        # debug only  ^^^^^^^^^^^^^^^^^^^^
                        
                        #<  update mat
                        mat -= useful_grad__d_d
                        
                        # debug only  vvvvvvvvvvvvvvvvvvvv
                        log10_of__mat_after_modify = log10_avg_safe(mat)
                        log10_similarity = log10_avg__how_similar(mat, ori_mat)
                        # debug only  ^^^^^^^^^^^^^^^^^^^^
                        
                        #<  protect the length after updating
                        _temp_len_sqr = mat.mul(mat).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
                        mul_me___to_get_unit_length = (_temp_len_sqr).pow(-0.5)
                        mat *= expand_vec_to_matrix(mul_me___to_get_unit_length, each_element_to="row")
                        #<  a bit check>
                        temp_vec_len = get_vector_length(mat)
                        assert _tensor_equal(temp_vec_len, torch.ones(size=[dim], device= device))
                        
                        #<  measure the protected.>
                        after__len_loss, after__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                        after__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
                        
                        BS__length, BS__angle, _ = LOSS__behavior_similarity(mat, ori_mat)
                        #----------------#----------------#----------------#----------------
                        
                        #_raw_result__len_loss__after_sub_before[test_count] = before__len_loss - after__len_loss
                        _raw_result__angle_loss__after_sub_before[test_count] = before__angle_loss - after__angle_loss
                        _raw_result__length_retention_loss__after_sub_before[test_count] = \
                            before__length_retention_loss - after__length_retention_loss
                        _raw_result__avg_of_abs_of_final_grad[test_count] = useful_grad__d_d.abs().mean()
                        _raw_result__neg_BS__length[test_count] = -BS__length
                        _raw_result__neg_BS__angle[test_count] = -BS__angle
                        pass#for test_count
                    
                    #len_loss__opt_value             .append(_raw_result__len_loss__after_sub_before.mean().item())
                    #len_loss__opt_prob              .append(_raw_result__len_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    angle_loss__opt_value           .append(_raw_result__angle_loss__after_sub_before.mean().item())
                    #angle_loss__opt_prob            .append(_raw_result__angle_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    length_retention_loss__opt_value.append(_raw_result__length_retention_loss__after_sub_before.mean().item())
                    #length_retention_loss__opt_prob .append(_raw_result__length_retention_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    #avg_of_abs_of_final_grad__min   .append(_raw_result__avg_of_abs_of_final_grad.min ())
                    #avg_of_abs_of_final_grad__max   .append(_raw_result__avg_of_abs_of_final_grad.max ())
                    avg_of_abs_of_final_grad__avg   .append(_raw_result__avg_of_abs_of_final_grad.mean())
                    neg_BS__length__min       .append(_raw_result__neg_BS__length.min ().item())
                    neg_BS__length__max       .append(_raw_result__neg_BS__length.max ().item())
                    neg_BS__length__avg       .append(_raw_result__neg_BS__length.mean().item())
                    neg_BS__angle__min        .append(_raw_result__neg_BS__angle .min ().item())
                    neg_BS__angle__max        .append(_raw_result__neg_BS__angle .max ().item())
                    neg_BS__angle__avg        .append(_raw_result__neg_BS__angle .mean().item())
                    
                    pass# for expansion_factor(y axis)
                
                print(f"dim = {dim}")
                # print(f"expansion_factor_list   = {str_the_list(expansion_factor_list, 2, segment=",    ")}")
                # print(f"cap_to = {cap_to}")
                print(f"cap_to_list               = {str_the_list(cap_to_list, 2, segment=",    ")}")
                print(f"expansion_factor =   {expansion_factor}")
                #print(f"len_loss__opt_value     = {str_the_list(len_loss__opt_value, 5)}")
                #print(f"len_loss__opt_prob      = {str_the_list__probability(len_loss__opt_prob, 3, flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                print(f"angle_loss__opt_value   = {str_the_list(angle_loss__opt_value, 5)}")
                #print(f"angle_loss__opt_prob    = {str_the_list__probability(angle_loss__opt_prob, 3, flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                print(f"length_retention_loss__v= {str_the_list(length_retention_loss__opt_value, 5)}")
                #print(f"length_retention_loss__p= {str_the_list__probability(length_retention_loss__opt_prob, 3, 
                                                                                                    # flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                #print(f"avg_abs_final_grad__min = {str_the_list(avg_of_abs_of_final_grad__min, 5)}")
                #print(f"avg_abs_final_grad__max = {str_the_list(avg_of_abs_of_final_grad__max, 5)}")
                print(f"avg_abs_final_grad__avg = {str_the_list(avg_of_abs_of_final_grad__avg, 5)}")
                print(f"neg_BS__length__min     = {str_the_list(neg_BS__length__min, 5)}")
                print(f"neg_BS__length__max     = {str_the_list(neg_BS__length__max, 5)}")
                print(f"neg_BS__length__avg     = {str_the_list(neg_BS__length__avg, 5)}")
                print(f"neg_BS__angle__min      = {str_the_list(neg_BS__angle__min , 5)}")
                print(f"neg_BS__angle__max      = {str_the_list(neg_BS__angle__max , 5)}")
                print(f"neg_BS__angle__avg      = {str_the_list(neg_BS__angle__avg , 5)}")
                print("------------->>>>>>>>>>>>>")
                
                pass#for cap_to(x axis)
            
            pass#for outter_iter_count
        
        del mat, ori_mat
        pass#/ test
    
    # useful.
    if "accurate scan for the best param combination." and False:
        # greater is better
        
        if "result" and False:
            # dim = 10
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.16849,  0.17533,  0.18723,  0.19949,  0.21258,  0.22154,  0.23268,  0.24334,  0.25475,  0.26566,  0.27609,  0.29249,  0.30043,  0.31010,  0.32064,  0.33390,  0.34268,  0.35047]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.01924,  0.01858,  0.01996,  0.01919,  0.02242,  0.01989,  0.02059,  0.02158,  0.02310,  0.02317,  0.02019,  0.02055,  0.02107,  0.02064,  0.02017,  0.01989,  0.02085,  0.01778]
            # angle_loss__opt_value   = [ 0.41444,  0.41379,  0.43264,  0.44491,  0.45806,  0.46014,  0.46533,  0.45736,  0.47393,  0.45452,  0.45580,  0.45810,  0.43987,  0.42997,  0.41582,  0.40986,  0.38417,  0.38468]
            # length_retention_loss__v= [ 0.01768,  0.01743,  0.01831,  0.01992,  0.02026,  0.02210,  0.02051,  0.02101,  0.02132,  0.02057,  0.02177,  0.02129,  0.01938,  0.01861,  0.01904,  0.01880,  0.01873,  0.01740]
            # avg_abs_final_grad__avg = [ 0.06300,  0.06724,  0.07147,  0.07571,  0.07994,  0.08418,  0.08841,  0.09265,  0.09688,  0.10112,  0.10535,  0.10959,  0.11382,  0.11806,  0.12229,  0.12653,  0.13076,  0.13500]
            # ------------->>>>>>>>>>>>>                                                       **                            **        **
            # dim = 10
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.24347,  0.26043,  0.27628,  0.29302,  0.30973,  0.32575,  0.34148,  0.36021,  0.37390,  0.39206,  0.40823,  0.42362,  0.44049,  0.45772,  0.47174,  0.49023,  0.50767,  0.52104]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.02023,  0.02007,  0.02198,  0.02126,  0.02320,  0.02476,  0.02374,  0.02635,  0.02443,  0.02628,  0.02554,  0.02465,  0.02414,  0.02352,  0.02409,  0.02670,  0.02241,  0.02259]
            # angle_loss__opt_value   = [ 0.44158,  0.45813,  0.48507,  0.49752,  0.51658,  0.52213,  0.53737,  0.52732,  0.55993,  0.55268,  0.53475,  0.54222,  0.53300,  0.51615,  0.51809,  0.49811,  0.48260,  0.48486]
            # length_retention_loss__v= [ 0.01859,  0.02000,  0.02108,  0.02176,  0.02395,  0.02370,  0.02426,  0.02410,  0.02558,  0.02512,  0.02369,  0.02469,  0.02535,  0.02475,  0.02268,  0.02315,  0.02273,  0.02212]
            # avg_abs_final_grad__avg = [ 0.06300,  0.06724,  0.07147,  0.07571,  0.07994,  0.08418,  0.08841,  0.09265,  0.09688,  0.10112,  0.10535,  0.10959,  0.11382,  0.11806,  0.12229,  0.12653,  0.13076,  0.13500]
            # ------------->>>>>>>>>>>>>                                                                                    ****                                                                      **    
            # dim = 10
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.33545,  0.35707,  0.37892,  0.40076,  0.42470,  0.44817,  0.47248,  0.49561,  0.51279,  0.53822,  0.55614,  0.58379,  0.60280,  0.62003,  0.64823,  0.67303,  0.69363,  0.71715]
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.02057,  0.02043,  0.02176,  0.02246,  0.02513,  0.02329,  0.02311,  0.02602,  0.02433,  0.02617,  0.02501,  0.02373,  0.02190,  0.02054,  0.02124,  0.02191,  0.02071,  0.01778]
            # angle_loss__opt_value   = [ 0.44451,  0.47244,  0.48760,  0.51584,  0.52518,  0.53221,  0.53350,  0.54073,  0.53762,  0.53465,  0.53510,  0.53143,  0.51251,  0.49986,  0.48191,  0.45630,  0.42995,  0.43197]
            # length_retention_loss__v= [ 0.01977,  0.02131,  0.02341,  0.02319,  0.02301,  0.02395,  0.02471,  0.02586,  0.02499,  0.02533,  0.02477,  0.02454,  0.02350,  0.02380,  0.02117,  0.02113,  0.02013,  0.01894]
            # avg_abs_final_grad__avg = [ 0.06300,  0.06724,  0.07147,  0.07571,  0.07994,  0.08418,  0.08841,  0.09265,  0.09688,  0.10112,  0.10535,  0.10959,  0.11382,  0.11806,  0.12229,  0.12653,  0.13076,  0.13500]
            # ------------->>>>>>>>>>>>>                                                                           ****                 **
            # dim = 10
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.43656,  0.46439,  0.49016,  0.52771,  0.54726,  0.59133,  0.60628,  0.63769,  0.67070,  0.68983,  0.71214,  0.74395,  0.77178,  0.81628,  0.83430,  0.86820,  0.90718,  0.93359]
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.02209,  0.02124,  0.02287,  0.02295,  0.02368,  0.02227,  0.02180,  0.02462,  0.02093,  0.02258,  0.01966,  0.02163,  0.02138,  0.02169,  0.02041,  0.01837,  0.01820,  0.01291]
            # angle_loss__opt_value   = [ 0.45888,  0.47859,  0.48167,  0.50368,  0.51395,  0.52079,  0.51779,  0.49929,  0.49427,  0.50469,  0.48170,  0.47162,  0.46163,  0.43913,  0.40923,  0.40613,  0.39370,  0.38449]
            # length_retention_loss__v= [ 0.01995,  0.02131,  0.02227,  0.02214,  0.02226,  0.02260,  0.02381,  0.02239,  0.02363,  0.02235,  0.02177,  0.02249,  0.02053,  0.01834,  0.01815,  0.01630,  0.01768,  0.01648]
            # avg_abs_final_grad__avg = [ 0.06300,  0.06724,  0.07147,  0.07571,  0.07994,  0.08418,  0.08841,  0.09265,  0.09688,  0.10112,  0.10535,  0.10959,  0.11382,  0.11806,  0.12229,  0.12653,  0.13076,  0.13500]
            # ------------->>>>>>>>>>>>>                                                       **        **        **
            # dim = 10
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.53277,  0.58285,  0.61695,  0.64819,  0.67229,  0.71676,  0.74937,  0.79346,  0.82156,  0.84566,  0.90041,  0.92603,  0.99598,  0.98056,  1.03392,  1.09721,  1.08395,  1.13501]
            # expansion_factor =   3.0
            # len_loss__opt_value     = [ 0.01838,  0.02031,  0.02290,  0.02083,  0.02065,  0.02204,  0.02139,  0.01907,  0.01937,  0.01875,  0.02017,  0.01716,  0.02036,  0.01728,  0.01581,  0.01413,  0.01542,  0.01782]
            # angle_loss__opt_value   = [ 0.44857,  0.46782,  0.46999,  0.47571,  0.46905,  0.45995,  0.45223,  0.46643,  0.44503,  0.43875,  0.42042,  0.42443,  0.39709,  0.41749,  0.41076,  0.38324,  0.40456,  0.39153]
            # length_retention_loss__v= [ 0.01960,  0.02062,  0.02205,  0.02209,  0.02293,  0.02080,  0.02049,  0.02141,  0.02005,  0.01989,  0.01877,  0.01930,  0.01864,  0.01760,  0.01776,  0.01645,  0.01774,  0.01671]
            # avg_abs_final_grad__avg = [ 0.06300,  0.06724,  0.07147,  0.07571,  0.07994,  0.08418,  0.08841,  0.09265,  0.09688,  0.10112,  0.10535,  0.10959,  0.11382,  0.11806,  0.12229,  0.12653,  0.13076,  0.13500]
            # ------------->>>>>>>>>>>>>                        **        **        **                                                                                                        
            # dim = 10
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.63716,  0.68111,  0.71290,  0.77213,  0.81297,  0.86275,  0.89510,  0.94832,  0.98189,  1.03355,  1.07930,  1.11903,  1.16246,  1.18052,  1.22268,  1.25986,  1.33375,  1.38034]
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.02067,  0.02076,  0.02273,  0.01840,  0.02020,  0.02001,  0.01989,  0.02037,  0.01812,  0.01884,  0.01580,  0.01645,  0.01568,  0.01878,  0.01528,  0.01624,  0.01592,  0.01445]
            # angle_loss__opt_value   = [ 0.42613,  0.45167,  0.43542,  0.44348,  0.42860,  0.41514,  0.41254,  0.39793,  0.38389,  0.38717,  0.37766,  0.38013,  0.36969,  0.36992,  0.36978,  0.37216,  0.36318,  0.37730]
            # length_retention_loss__v= [ 0.01997,  0.02053,  0.02012,  0.01860,  0.01922,  0.01911,  0.01794,  0.01808,  0.01725,  0.01735,  0.01652,  0.01801,  0.01665,  0.01602,  0.01475,  0.01793,  0.01769,  0.01623]
            # avg_abs_final_grad__avg = [ 0.06300,  0.06724,  0.07147,  0.07571,  0.07994,  0.08418,  0.08841,  0.09265,  0.09688,  0.10112,  0.10535,  0.10959,  0.11382,  0.11806,  0.12229,  0.12653,  0.13076,  0.13500]
            # ------------->>>>>>>>>>>>>              ****         **                                         
            
            
            
            #1w 继续来读
            # dim = 100
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.20004,  0.21501,  0.22845,  0.24189,  0.25543,  0.26924,  0.28403,  0.29712,  0.31052,  0.32509,  0.33668,  0.35036,  0.36644,  0.37885,  0.39053,  0.40521,  0.41806,  0.43473]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.00757,  0.00766,  0.00839,  0.00839,  0.00848,  0.00855,  0.00866,  0.00840,  0.00832,  0.00828,  0.00849,  0.00866,  0.00838,  0.00806,  0.00749,  0.00780,  0.00736,  0.00731]
            # angle_loss__opt_value   = [ 0.49589,  0.51480,  0.53092,  0.54289,  0.55098,  0.55746,  0.56178,  0.56329,  0.55986,  0.55579,  0.54730,  0.54125,  0.52788,  0.51683,  0.50587,  0.49560,  0.48284,  0.46953]
            # length_retention_loss__v= [ 0.00765,  0.00806,  0.00815,  0.00815,  0.00811,  0.00838,  0.00847,  0.00884,  0.00872,  0.00818,  0.00822,  0.00867,  0.00747,  0.00787,  0.00756,  0.00766,  0.00748,  0.00709]
            # avg_abs_final_grad__avg = [ 0.02100,  0.02241,  0.02382,  0.02524,  0.02665,  0.02806,  0.02947,  0.03088,  0.03229,  0.03371,  0.03512,  0.03653,  0.03794,  0.03935,  0.04076,  0.04218,  0.04359,  0.04500]
            # ------------->>>>>>>>>>>>>                                                                 *        ****                                     *          
            # dim = 100
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.26266,  0.28029,  0.29795,  0.31554,  0.33322,  0.35063,  0.36858,  0.38629,  0.40371,  0.42148,  0.43894,  0.45688,  0.47454,  0.49201,  0.50965,  0.52689,  0.54464,  0.56260]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.00735,  0.00782,  0.00805,  0.00855,  0.00853,  0.00837,  0.00873,  0.00910,  0.00838,  0.00841,  0.00820,  0.00852,  0.00841,  0.00818,  0.00781,  0.00691,  0.00747,  0.00751]
            # angle_loss__opt_value   = [ 0.49679,  0.51665,  0.53108,  0.54382,  0.55410,  0.55977,  0.56253,  0.56418,  0.56083,  0.55613,  0.54650,  0.54218,  0.52883,  0.51988,  0.50862,  0.49885,  0.48688,  0.48128]
            # length_retention_loss__v= [ 0.00773,  0.00783,  0.00820,  0.00799,  0.00831,  0.00845,  0.00851,  0.00856,  0.00844,  0.00836,  0.00824,  0.00820,  0.00802,  0.00806,  0.00779,  0.00763,  0.00746,  0.00758]
            # avg_abs_final_grad__avg = [ 0.02100,  0.02241,  0.02382,  0.02524,  0.02665,  0.02806,  0.02947,  0.03088,  0.03229,  0.03371,  0.03512,  0.03653,  0.03794,  0.03935,  0.04076,  0.04218,  0.04359,  0.04500]
            # ------------->>>>>>>>>>>>>                                                                         ******                                                          
            # dim = 100
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.33763,  0.36013,  0.38376,  0.40772,  0.42712,  0.45347,  0.47521,  0.49948,  0.52104,  0.54337,  0.56184,  0.58535,  0.60772,  0.63536,  0.65840,  0.67952,  0.70487,  0.72967]
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.00772,  0.00790,  0.00796,  0.00824,  0.00848,  0.00852,  0.00850,  0.00835,  0.00827,  0.00885,  0.00829,  0.00845,  0.00856,  0.00756,  0.00801,  0.00766,  0.00744,  0.00732]
            # angle_loss__opt_value   = [ 0.49393,  0.51338,  0.52987,  0.54291,  0.55130,  0.55679,  0.55796,  0.55857,  0.55626,  0.55437,  0.54630,  0.53844,  0.52952,  0.51689,  0.51159,  0.50050,  0.49432,  0.48856]
            # length_retention_loss__v= [ 0.00772,  0.00748,  0.00822,  0.00822,  0.00799,  0.00818,  0.00863,  0.00891,  0.00859,  0.00865,  0.00855,  0.00816,  0.00818,  0.00788,  0.00817,  0.00778,  0.00768,  0.00693]
            # avg_abs_final_grad__avg = [ 0.02100,  0.02241,  0.02382,  0.02524,  0.02665,  0.02806,  0.02947,  0.03088,  0.03229,  0.03371,  0.03512,  0.03653,  0.03794,  0.03935,  0.04076,  0.04218,  0.04359,  0.04500]
            # ------------->>>>>>>>>>>>>                                                                          ****                 **                   
            # dim = 100
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.42804,  0.45758,  0.49073,  0.50968,  0.54768,  0.57388,  0.60843,  0.64003,  0.66700,  0.69970,  0.70899,  0.74347,  0.77958,  0.80940,  0.84204,  0.85534,  0.91187,  0.93215]
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.00772,  0.00788,  0.00798,  0.00840,  0.00852,  0.00860,  0.00819,  0.00841,  0.00860,  0.00844,  0.00812,  0.00775,  0.00767,  0.00796,  0.00767,  0.00790,  0.00750,  0.00724]
            # angle_loss__opt_value   = [ 0.49263,  0.51000,  0.52307,  0.53345,  0.54084,  0.54478,  0.54547,  0.54769,  0.54619,  0.54200,  0.53507,  0.53073,  0.52140,  0.51417,  0.51014,  0.50256,  0.49677,  0.49385]
            # length_retention_loss__v= [ 0.00745,  0.00794,  0.00817,  0.00814,  0.00788,  0.00825,  0.00825,  0.00815,  0.00822,  0.00813,  0.00790,  0.00777,  0.00786,  0.00768,  0.00755,  0.00743,  0.00754,  0.00723]
            # avg_abs_final_grad__avg = [ 0.02100,  0.02241,  0.02382,  0.02524,  0.02665,  0.02806,  0.02947,  0.03088,  0.03229,  0.03371,  0.03512,  0.03653,  0.03794,  0.03935,  0.04076,  0.04218,  0.04359,  0.04500]
            # ------------->>>>>>>>>>>>>                                                     *  *        *        **           *                                                        
            # dim = 100
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.54435,  0.57699,  0.62593,  0.66368,  0.68788,  0.74574,  0.76789,  0.80049,  0.83745,  0.86154,  0.91005,  0.95281,  0.98514,  1.01575,  1.05865,  1.09510,  1.14804,  1.16923]
            # expansion_factor =   3.0
            # len_loss__opt_value     = [ 0.00759,  0.00741,  0.00765,  0.00773,  0.00806,  0.00766,  0.00789,  0.00793,  0.00802,  0.00817,  0.00800,  0.00818,  0.00818,  0.00774,  0.00775,  0.00731,  0.00700,  0.00761]
            # angle_loss__opt_value   = [ 0.48504,  0.50124,  0.51332,  0.52224,  0.52900,  0.53168,  0.53270,  0.53268,  0.53335,  0.53005,  0.52509,  0.52155,  0.51979,  0.51458,  0.50787,  0.50641,  0.49863,  0.50043]
            # length_retention_loss__v= [ 0.00764,  0.00767,  0.00746,  0.00794,  0.00789,  0.00835,  0.00823,  0.00810,  0.00803,  0.00845,  0.00771,  0.00791,  0.00781,  0.00810,  0.00772,  0.00768,  0.00783,  0.00771]
            # avg_abs_final_grad__avg = [ 0.02100,  0.02241,  0.02382,  0.02524,  0.02665,  0.02806,  0.02947,  0.03088,  0.03229,  0.03371,  0.03512,  0.03653,  0.03794,  0.03935,  0.04076,  0.04218,  0.04359,  0.04500]
            # ------------->>>>>>>>>>>>>                                                                                    **        **                   *          *                 
            # dim = 100
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.69164,  0.73525,  0.77860,  0.81635,  0.87567,  0.93048,  0.95738,  1.02242,  1.03556,  1.09853,  1.15621,  1.18575,  1.23748,  1.27130,  1.31346,  1.38905,  1.43010,  1.49088]
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.00737,  0.00723,  0.00765,  0.00794,  0.00783,  0.00771,  0.00797,  0.00825,  0.00808,  0.00794,  0.00753,  0.00767,  0.00760,  0.00765,  0.00767,  0.00733,  0.00763,  0.00701]
            # angle_loss__opt_value   = [ 0.47529,  0.48831,  0.49793,  0.50621,  0.51173,  0.51460,  0.51858,  0.51926,  0.51758,  0.51649,  0.51637,  0.51276,  0.51013,  0.50606,  0.50205,  0.49862,  0.49679,  0.48877]
            # length_retention_loss__v= [ 0.00729,  0.00736,  0.00732,  0.00793,  0.00779,  0.00795,  0.00801,  0.00806,  0.00772,  0.00733,  0.00800,  0.00773,  0.00769,  0.00775,  0.00761,  0.00766,  0.00733,  0.00734]
            # avg_abs_final_grad__avg = [ 0.02100,  0.02241,  0.02382,  0.02524,  0.02665,  0.02806,  0.02947,  0.03088,  0.03229,  0.03371,  0.03512,  0.03653,  0.03794,  0.03935,  0.04076,  0.04218,  0.04359,  0.04500]
            # ------------->>>>>>>>>>>>>                                                                         ******            
            
            
            
            # dim = 1000
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.24796,  0.26453,  0.28074,  0.29788,  0.31385,  0.33068,  0.34668,  0.36476,  0.38013,  0.39754,  0.41405,  0.42894,  0.44847,  0.46359,  0.48059,  0.49821,  0.51446,  0.53249]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.00252,  0.00256,  0.00262,  0.00271,  0.00272,  0.00273,  0.00277,  0.00268,  0.00267,  0.00263,  0.00266,  0.00262,  0.00260,  0.00246,  0.00241,  0.00239,  0.00230,  0.00225]
            # angle_loss__opt_value   = [ 0.51600,  0.53248,  0.54543,  0.55502,  0.56127,  0.56430,  0.56404,  0.56150,  0.55608,  0.54913,  0.54059,  0.53121,  0.52114,  0.51115,  0.50152,  0.49230,  0.48465,  0.47732]
            # length_retention_loss__v= [ 0.00254,  0.00261,  0.00270,  0.00277,  0.00252,  0.00266,  0.00282,  0.00269,  0.00266,  0.00248,  0.00249,  0.00240,  0.00273,  0.00251,  0.00247,  0.00230,  0.00233,  0.00235]
            # avg_abs_final_grad__avg = [ 0.00700,  0.00747,  0.00794,  0.00841,  0.00888,  0.00935,  0.00982,  0.01029,  0.01076,  0.01124,  0.01171,  0.01218,  0.01265,  0.01312,  0.01359,  0.01406,  0.01453,  0.01500]
            # ------------->>>>>>>>>>>>>                                                        **      ****
            # dim = 1000
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.27738,  0.29600,  0.31466,  0.33332,  0.35195,  0.37059,  0.38928,  0.40788,  0.42653,  0.44516,  0.46379,  0.48251,  0.50110,  0.51975,  0.53837,  0.55707,  0.57571,  0.59432]
            # expansion_factor =   0.0
            # len_loss__opt_value     = [ 0.00253,  0.00256,  0.00269,  0.00268,  0.00270,  0.00275,  0.00269,  0.00273,  0.00268,  0.00267,  0.00265,  0.00253,  0.00253,  0.00253,  0.00248,  0.00238,  0.00245,  0.00228]
            # angle_loss__opt_value   = [ 0.51608,  0.53281,  0.54548,  0.55519,  0.56107,  0.56426,  0.56373,  0.56122,  0.55644,  0.54903,  0.54066,  0.53135,  0.52156,  0.51161,  0.50228,  0.49373,  0.48432,  0.47834]
            # length_retention_loss__v= [ 0.00270,  0.00269,  0.00259,  0.00255,  0.00286,  0.00253,  0.00307,  0.00275,  0.00263,  0.00269,  0.00269,  0.00241,  0.00240,  0.00231,  0.00260,  0.00256,  0.00227,  0.00210]
            # avg_abs_final_grad__avg = [ 0.00700,  0.00747,  0.00794,  0.00841,  0.00888,  0.00935,  0.00982,  0.01029,  0.01076,  0.01124,  0.01171,  0.01218,  0.01265,  0.01312,  0.01359,  0.01406,  0.01453,  0.01500]
            # ------------->>>>>>>>>>>>>                                                      ****     **     
            # dim = 1000
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.31070,  0.33028,  0.35221,  0.37169,  0.39232,  0.41425,  0.43444,  0.45644,  0.47593,  0.49806,  0.51992,  0.53998,  0.56246,  0.58277,  0.60195,  0.62154,  0.64401,  0.66581]
            # expansion_factor =   1.0
            # len_loss__opt_value     = [ 0.00250,  0.00258,  0.00265,  0.00270,  0.00271,  0.00266,  0.00272,  0.00272,  0.00266,  0.00275,  0.00261,  0.00255,  0.00253,  0.00251,  0.00241,  0.00239,  0.00240,  0.00232]
            # angle_loss__opt_value   = [ 0.51581,  0.53213,  0.54506,  0.55440,  0.56079,  0.56317,  0.56365,  0.56060,  0.55537,  0.54829,  0.53992,  0.53049,  0.52182,  0.51122,  0.50244,  0.49382,  0.48661,  0.47918]
            # length_retention_loss__v= [ 0.00248,  0.00246,  0.00297,  0.00257,  0.00278,  0.00263,  0.00261,  0.00250,  0.00285,  0.00281,  0.00261,  0.00259,  0.00264,  0.00258,  0.00225,  0.00254,  0.00238,  0.00231]
            # avg_abs_final_grad__avg = [ 0.00700,  0.00747,  0.00794,  0.00841,  0.00888,  0.00935,  0.00982,  0.01029,  0.01076,  0.01124,  0.01171,  0.01218,  0.01265,  0.01312,  0.01359,  0.01406,  0.01453,  0.01500]
            # ------------->>>>>>>>>>>>>                        **                                      **                              **
            # dim = 1000
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.34499,  0.36952,  0.39232,  0.41694,  0.43872,  0.46609,  0.48552,  0.51054,  0.53545,  0.55729,  0.57738,  0.60511,  0.62702,  0.64752,  0.67365,  0.69416,  0.72340,  0.74212]
            # expansion_factor =   2.0
            # len_loss__opt_value     = [ 0.00250,  0.00258,  0.00268,  0.00270,  0.00271,  0.00279,  0.00273,  0.00273,  0.00269,  0.00260,  0.00255,  0.00252,  0.00251,  0.00247,  0.00252,  0.00238,  0.00236,  0.00239]
            # angle_loss__opt_value   = [ 0.51501,  0.53150,  0.54424,  0.55353,  0.55947,  0.56237,  0.56164,  0.55930,  0.55430,  0.54720,  0.53934,  0.53039,  0.52102,  0.51157,  0.50351,  0.49399,  0.48784,  0.48103]
            # length_retention_loss__v= [ 0.00250,  0.00262,  0.00280,  0.00294,  0.00272,  0.00281,  0.00278,  0.00258,  0.00270,  0.00288,  0.00236,  0.00247,  0.00240,  0.00248,  0.00241,  0.00256,  0.00218,  0.00223]
            # avg_abs_final_grad__avg = [ 0.00700,  0.00747,  0.00794,  0.00841,  0.00888,  0.00935,  0.00982,  0.01029,  0.01076,  0.01124,  0.01171,  0.01218,  0.01265,  0.01312,  0.01359,  0.01406,  0.01453,  0.01500]
            # ------------->>>>>>>>>>>>>                                   **                 ****                         
            # dim = 1000
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.38444,  0.40729,  0.44372,  0.46347,  0.48705,  0.51719,  0.54070,  0.56784,  0.59038,  0.62849,  0.64673,  0.67886,  0.70064,  0.72072,  0.75160,  0.77673,  0.79977,  0.82594]
            # expansion_factor =   3.0
            # len_loss__opt_value     = [ 0.00250,  0.00258,  0.00263,  0.00265,  0.00274,  0.00272,  0.00270,  0.00270,  0.00270,  0.00266,  0.00258,  0.00264,  0.00254,  0.00249,  0.00243,  0.00249,  0.00238,  0.00234]
            # angle_loss__opt_value   = [ 0.51442,  0.53017,  0.54339,  0.55246,  0.55795,  0.56061,  0.56028,  0.55740,  0.55249,  0.54629,  0.53819,  0.53019,  0.52015,  0.51084,  0.50364,  0.49522,  0.48950,  0.48336]
            # length_retention_loss__v= [ 0.00232,  0.00272,  0.00248,  0.00249,  0.00270,  0.00264,  0.00275,  0.00278,  0.00250,  0.00246,  0.00270,  0.00272,  0.00262,  0.00263,  0.00265,  0.00235,  0.00241,  0.00247]
            # avg_abs_final_grad__avg = [ 0.00700,  0.00747,  0.00794,  0.00841,  0.00888,  0.00935,  0.00982,  0.01029,  0.01076,  0.01124,  0.01171,  0.01218,  0.01265,  0.01312,  0.01359,  0.01406,  0.01453,  0.01500]
            # ------------->>>>>>>>>>>>>                                            **          ****
            # dim = 1000
            # scan_factor_list           = [ 0.70,     0.75,     0.79,     0.84,     0.89,     0.94,     0.98,     1.03,     1.08,     1.12,     1.17,     1.22,     1.26,     1.31,     1.36,     1.41,     1.45,     1.50]
            # actual_cap_to           = [ 0.43994,  0.46192,  0.49567,  0.51737,  0.54753,  0.57881,  0.60351,  0.63381,  0.66525,  0.68743,  0.71791,  0.74862,  0.77697,  0.80493,  0.83680,  0.86424,  0.88287,  0.91579]
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.00250,  0.00258,  0.00262,  0.00266,  0.00271,  0.00271,  0.00271,  0.00267,  0.00267,  0.00270,  0.00260,  0.00258,  0.00250,  0.00244,  0.00244,  0.00246,  0.00237,  0.00235]
            # angle_loss__opt_value   = [ 0.51344,  0.52926,  0.54141,  0.55061,  0.55604,  0.55826,  0.55778,  0.55509,  0.55020,  0.54445,  0.53590,  0.52871,  0.52035,  0.51219,  0.50455,  0.49644,  0.49072,  0.48580]
            # length_retention_loss__v= [ 0.00267,  0.00235,  0.00273,  0.00262,  0.00268,  0.00272,  0.00280,  0.00258,  0.00256,  0.00281,  0.00258,  0.00236,  0.00246,  0.00258,  0.00232,  0.00247,  0.00269,  0.00243]
            # avg_abs_final_grad__avg = [ 0.00700,  0.00747,  0.00794,  0.00841,  0.00888,  0.00935,  0.00982,  0.01029,  0.01076,  0.01124,  0.01171,  0.01218,  0.01265,  0.01312,  0.01359,  0.01406,  0.01453,  0.01500]
            # ------------->>>>>>>>>>>>>                                                      ****                                    **
            
            pass
        
        #------------------#------------------#------------------
        # dim_list =        [ 2, 10,100,1000]
        # test_time_list = [100,100,100, 20]
        # test_time_list = [10,10,10, 2]
        
        dim_list =       [ 10, 100,1000]
        test_time_list = [200, 100, 30]
        dim_list =        [100]
        test_time_list = [ 100]
        for outter_iter_count in range(dim_list.__len__()):
            dim = dim_list[outter_iter_count]
            test_time = test_time_list[outter_iter_count]
            print(test_time)
            iota_of_dim = iota(dim)
            #<  device
            if dim>100:
                device = 'cuda'
                pass
            else:
                device = 'cpu'
                pass
            #<  core_ref_of__avg_abs__capped_grad
            if dim == 10:
                core_ref_of__avg_abs__capped_grad = 0.095#expansion<=1. otherwise   -0.009* (expansion_factor-1) + 0.095
                pass
            elif dim == 100:
                core_ref_of__avg_abs__capped_grad = 0.031
                pass
            elif dim == 1000:
                core_ref_of__avg_abs__capped_grad = 0.0095
                pass
            else:
                assert False, "bad param: dim"
        #------------------#------------------#------------------
        
            #------------------#------------------#------------------
            expansion_factor_list = [-1., 0.,1.,2.,3.,4.]
            #expansion_factor_list = [ 1.,2.,3.,4.]
            for expansion_factor in expansion_factor_list:
            #------------------#------------------#------------------
            
                len_loss__opt_value              = []  # dont modity this
                len_loss__opt_prob               = []  # dont modity this
                angle_loss__opt_value            = []  # dont modity this
                angle_loss__opt_prob             = []  # dont modity this
                length_retention_loss__opt_value = []  # dont modity this
                length_retention_loss__opt_prob  = []  # dont modity this
                avg_of_abs_of_final_grad__min    = []  # dont modity this
                avg_of_abs_of_final_grad__max    = []  # dont modity this
                avg_of_abs_of_final_grad__avg    = []  # dont modity this
                actual_cap_to                    = []  # dont modity this
                    
                #------------------#------------------#------------------
                # cap_to in this test is calculated.

                # scan_factor_list = []
                # for _raw_scan_factor in range(16):
                #     scan_factor = _raw_scan_factor*0.02# 0 to 0.4
                #     assert scan_factor >= 0. and scan_factor <= 0.3
                #     scan_factor += 0.9 
                #     assert scan_factor >= 0.9 and scan_factor <= 1.2
                #     scan_factor_list.append(scan_factor)
                #     pass
                
                #scan_factor_list = [ 0.90,     0.92,     0.94,     0.96,     0.98,     1.00,     1.02,     1.04,     1.06,     1.08,     1.10,     1.12,     1.14,     1.16,     1.18,     1.20]
                scan_factor_list = torch.linspace(0.7,1.5,18)
                for scan_factor in scan_factor_list:
                #------------------#------------------#------------------
                
                    _raw_result__len_loss__after_sub_before             = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__angle_loss__after_sub_before           = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__length_retention_loss__after_sub_before= torch.empty(size=[test_time])  # dont modity this
                    _raw_result__avg_of_abs_of_final_grad               = torch.empty(size=[test_time])  # dont modity this
                    _raw_result__actual_cap_to                          = torch.empty(size=[test_time])  # dont modity this
                    
                    for test_count in range(test_time):
                        #----------------#----------------#----------------#----------------
                        ori_mat = torch.randn(size=[dim,dim], device=device)
                        ori_mat = vector_length_norm(ori_mat)
                        # old code ori_mat = ____init_mat_with_row_len_is_1(dim, device=device)
                        
                        #<  measure the init>
                        before__len_loss, before__angle_loss, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                        before__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(ori_mat)
                        
                        #<  calc grad>
                        mat = ori_mat.detach().clone()
                        
                        manual__mat_matmul_mat__d_d = mat@(mat.T)
                        manual__mat_matmul_mat__d_d[iota_of_dim, iota_of_dim] = 0.
                        
                        #<  original grad>
                        ori__grad__d_d = manual__mat_matmul_mat__d_d@mat
                        ori__grad_len__dim = get_vector_length(ori__grad__d_d)
                        #<  original grad, but len into 1>
                        _temp_reverse_of__ori__grad_len__dim = ori__grad_len__dim.pow(-1.)
                        len_1__ori_grad__d_d = ori__grad__d_d.mul( \
                                expand_vec_to_matrix(_temp_reverse_of__ori__grad_len__dim, each_element_to="row"))
                        del _temp_reverse_of__ori__grad_len__dim # just in case.
                        assert _tensor_equal(get_vector_length(len_1__ori_grad__d_d), torch.ones(size=[dim], device=device))
                        
                        #<  scale it a bit, to make the distribution a bit wider>
                        #otherwise, they are a lot 0.8,0.9. Wider means most are 0.3 to 0.8.
                        #<old code/>xx__grad_len_sqr__dim = ori__grad__d_d.mul(ori__grad__d_d).sum(dim=1)#mul and then sum, it's a dot.
                        max_of__ori__grad_len__dim = ori__grad_len__dim.max()
                        ratio_of__grad_len__dim = ori__grad_len__dim/max_of__ori__grad_len__dim
                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this scaled_to_1__grad_len_sqr__dim is still one 1.0 and a lot 0.xx
                        assert ratio_of__grad_len__dim.le(1.).all()
                        assert ratio_of__grad_len__dim.eq(1.).sum() == 1
                        
                        grad_len__after_expansion__dim = ratio_of__grad_len__dim.pow(expansion_factor)
                        # assert grad_len__after_expansion__dim.le(1.).all() when expansion_factor is neg, this is wrong.
                        # assert grad_len__after_expansion__dim.eq(1.).sum() == 1 when expansion_factor is 0., this is wrong.
                        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this grad_len__after_expansion__dim is still one 1.0 and a lot 0.xx
                        
                        if "visualization" and False:
                            #prin(expansion_factor)
                            visualize_this = grad_len__after_expansion__dim
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(tight_layout=True)
                            n_bins = dim//5
                            if n_bins<=3:
                                n_bins = 3
                                pass
                            ax.hist(visualize_this.tolist(), bins=n_bins)
                            plt.show()
                            pass
                        
                        #<  in this test the cap_to is calculated.>
                        #old code
                        # in old code the cap_to is set manually.
                        # #<  target length>
                        # grad_length_target__dim = grad_len__after_expansion__dim*cap_to# cap_to is scalar
                        # grad_length_target__dim__expand_dim = expand_vec_to_matrix(grad_length_target__dim, each_element_to="row")
                        
                        #new code, the cap_to is calculated now.
                        
                        
                        
                        avg_of_abs_of__un_capped_grad__per_row___dim = len_1__ori_grad__d_d.abs().mean(dim=1)
                        avg_of_abs_of__un_capped_grad__s = avg_of_abs_of__un_capped_grad__per_row___dim.dot(grad_len__after_expansion__dim)/dim
                        
                        target_of__avg_abs__capped_grad__s = scan_factor*core_ref_of__avg_abs__capped_grad
                        mul_me___similar_to_old__cap_to__s = target_of__avg_abs__capped_grad__s/avg_of_abs_of__un_capped_grad__s
                        grad_length_target__dim = grad_len__after_expansion__dim*mul_me___similar_to_old__cap_to__s# cap_to is calculated in this test
                        grad_length_target__dim__expand_dim = expand_vec_to_matrix(grad_length_target__dim, each_element_to="row")
                        #</ in this test the cap_to is calculated.>
                        
                        #<  finally 
                        useful_grad__d_d = len_1__ori_grad__d_d.mul(grad_length_target__dim__expand_dim)#row
                        # # assert _tensor_equal(get_vector_length(useful_grad__d_d), grad_length_target__dim)
                        
                        # debug only  vvvvvvvvvvvvvvvvvvvv
                        # log10_of__mat  = log10_avg_safe(mat)
                        # log10_of__grad = log10_avg_safe(useful_grad__d_d)
                        # debug only  ^^^^^^^^^^^^^^^^^^^^
                        
                        #<  update mat
                        mat -= useful_grad__d_d
                        
                        # debug only  vvvvvvvvvvvvvvvvvvvv
                        # log10_of__mat_after_modify = log10_avg_safe(mat)
                        # log10_similarity = log10_avg__how_similar(mat, ori_mat)
                        # debug only  ^^^^^^^^^^^^^^^^^^^^
                        
                        #<  protect the length after updating
                        _temp_len_sqr = mat.mul(mat).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
                        mul_me___to_get_unit_length = (_temp_len_sqr).pow(-0.5)
                        mat *= expand_vec_to_matrix(mul_me___to_get_unit_length, each_element_to="row")
                        #<  a bit check>
                        temp_vec_len = get_vector_length(mat)
                        # # assert _tensor_equal(temp_vec_len, [1.]*dim)
                        
                        #<  measure the protected.>
                        after__len_loss, after__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                        after__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
                        #----------------#----------------#----------------#----------------
                        
                        _raw_result__len_loss__after_sub_before[test_count] = before__len_loss - after__len_loss
                        _raw_result__angle_loss__after_sub_before[test_count] = before__angle_loss - after__angle_loss
                        _raw_result__length_retention_loss__after_sub_before[test_count] = \
                            before__length_retention_loss - after__length_retention_loss
                        _raw_result__avg_of_abs_of_final_grad[test_count] = useful_grad__d_d.abs().mean()
                        _raw_result__actual_cap_to[test_count] = mul_me___similar_to_old__cap_to__s
                        pass#for test_count
                    
                    len_loss__opt_value             .append(_raw_result__len_loss__after_sub_before.mean().item())
                    len_loss__opt_prob              .append(_raw_result__len_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    angle_loss__opt_value           .append(_raw_result__angle_loss__after_sub_before.mean().item())
                    angle_loss__opt_prob            .append(_raw_result__angle_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    length_retention_loss__opt_value.append(_raw_result__length_retention_loss__after_sub_before.mean().item())
                    length_retention_loss__opt_prob .append(_raw_result__length_retention_loss__after_sub_before.gt(0.).sum().item()/test_time)
                    avg_of_abs_of_final_grad__min   .append(_raw_result__avg_of_abs_of_final_grad.min ())
                    avg_of_abs_of_final_grad__max   .append(_raw_result__avg_of_abs_of_final_grad.max ())
                    avg_of_abs_of_final_grad__avg   .append(_raw_result__avg_of_abs_of_final_grad.mean())
                    actual_cap_to                   .append(_raw_result__actual_cap_to.mean())
                    #neg_behavior_similar_loss       .append(_raw_result__neg_behavior_similar_loss.mean().item())
                    
                    pass# for expansion_factor(y axis)
                
                print(f"dim = {dim}")
                # print(f"expansion_factor_list   = {str_the_list(expansion_factor_list, 2, segment=",    ")}")
                # print(f"cap_to = {cap_to}")
                print(f"scan_factor_list           = {str_the_list(scan_factor_list, 2, segment=",    ")}")
                print(f"actual_cap_to           = {str_the_list(actual_cap_to, 5)}")
                print(f"expansion_factor =   {expansion_factor}")
                print(f"len_loss__opt_value     = {str_the_list(len_loss__opt_value, 5)}")
                #print(f"len_loss__opt_prob      = {str_the_list__probability(len_loss__opt_prob, 3, flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                print(f"angle_loss__opt_value   = {str_the_list(angle_loss__opt_value, 5)}")
                #print(f"angle_loss__opt_prob    = {str_the_list__probability(angle_loss__opt_prob, 3, flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                print(f"length_retention_loss__v= {str_the_list(length_retention_loss__opt_value, 5)}")
                #print(f"length_retention_loss__p= {str_the_list__probability(length_retention_loss__opt_prob, 3, 
                                                                                                #    flag__offset_by50=True, flag__mul_2_after_offset=True)}")
                #print(f"avg_abs_final_grad__min = {str_the_list(avg_of_abs_of_final_grad__min, 5)}")
                #print(f"avg_abs_final_grad__max = {str_the_list(avg_of_abs_of_final_grad__max, 5)}")
                print(f"avg_abs_final_grad__avg = {str_the_list(avg_of_abs_of_final_grad__avg, 5)}")
                #print(f"neg_behavior_similar_loss= {str_the_list(neg_behavior_similar_loss, 5)}")
                print("------------->>>>>>>>>>>>>")
                
                pass#for cap_to(x axis)
            
            pass#for outter_iter_count
        
        pass#/ test
    
    return 

#____test____correction_method_test()






def calc__cap_to____ver_1(dim:int, expansion_factor:torch.Tensor, _debug__auto_clamp = False)->torch.Tensor:
    '''
    >>>             -1      0       1       2      #    3       4
    >>>             expansion_factor
    >>> dim 
    >>> 10          0.25    0.38    0.5     0.6     #   0.65    0.69
    >>> 100         0.3     0.39    0.5     0.62    #   0.86    1.02
    >>> 1000        0.34    0.38    0.42    0.46    #   0.5     0.6
    '''
    
    expansion_factor__inner = expansion_factor.detach().clone()
    
    if _debug__auto_clamp:
        #<  clamp the dim
        if dim<10:
            dim = 10
            pass
        if dim>1000:
            dim = 1000
            pass
        #</ clamp the dim
        
        expansion_factor__inner.clamp_(-1, 2)
        pass
    
    assert dim >=10 and dim <=1000, "only in between 10 and 1000 is implemented now."
    assert expansion_factor__inner>=-1. and expansion_factor__inner <=2
    
    lookup_table   = torch.tensor([[ 0.25, 0.38, 0.5 , 0.6 ],   # dim 10
                                    [0.3 , 0.39, 0.5 , 0.62],   # dim 100
                                    [0.34, 0.38, 0.42, 0.46]])  # dim 1000
    
    log10_of_dim = torch.tensor(dim, dtype=torch.float32).log10()
    
    result = interpolation_of_list_2d(lookup_table, log10_of_dim-1, expansion_factor__inner+1)
    return result
if "test" and False:
    result = calc__cap_to____ver_1(10, torch.tensor(-1))
    assert result.eq(0.25)
    assert result.shape.__len__() == 0
    assert calc__cap_to____ver_1(10, torch.tensor(0)) .eq(0.38)
    assert calc__cap_to____ver_1(10, torch.tensor(1)) .eq(0.5 )
    assert calc__cap_to____ver_1(10, torch.tensor(2)) .eq(0.6 )
    result = calc__cap_to____ver_1(10, torch.tensor(2))
    assert result.eq(0.6)
    assert result.shape.__len__() == 0
    
    assert calc__cap_to____ver_1(100, torch.tensor(0)).eq(0.39)
    assert calc__cap_to____ver_1(100, torch.tensor(1)).eq(0.5)
    assert calc__cap_to____ver_1(100, torch.tensor(2)).eq(0.62)
    
    assert calc__cap_to____ver_1(1000, torch.tensor(0)).eq(0.38)
    result = calc__cap_to____ver_1(1000, torch.tensor(1))
    assert result.eq(0.42)
    assert result.shape.__len__() == 0
    
    dim = 1001
    expansion_factor__outter = torch.tensor(-5)
    assert calc__cap_to____ver_1(1000, expansion_factor=expansion_factor__outter, _debug__auto_clamp = True).eq(0.34)
    assert dim == 1001
    assert expansion_factor__outter == torch.tensor(-5)
    
    pass    


def full_test_version_of_angle_correction__by_row(input:torch.Tensor, expansion_factor = 1., 
                        cap_to:float|None = None, iota_of_dim:torch.Tensor|None = None, 
                        #_debug__allow_any_param = False
                        )->torch.Tensor:
    '''The length of row vector of input can be anything. I'm not sure if it handles 0 vector correctly.
    
    row vectors of return value are 1, unless the input is too close to 0. I believe it's called standard vector???
    '''
    
    if isinstance(expansion_factor, float):
        expansion_factor__s = torch.tensor(expansion_factor)
        pass
    elif isinstance(expansion_factor, torch.Tensor):
        expansion_factor__s = expansion_factor.detach().clone()
        pass
    else:
        assert False, "bad type: expansion_factor"
    
    
    if cap_to is None:
        cap_to__s = calc__cap_to____ver_1(input.shape[0], expansion_factor__s)
        pass
    elif isinstance(cap_to, float):
        cap_to__s = torch.tensor(cap_to)
        pass
    elif isinstance(cap_to, torch.Tensor):
        cap_to__s = cap_to.detach().clone()
        pass
    else:
        assert False, "bad type."
        pass
    assert isinstance(cap_to__s, torch.Tensor)
    
    # if not _debug__allow_any_param:
    #     assert cap_to__s>=0.2 and cap_to__s <=0.8
    #     assert expansion_factor__s >=-1. and expansion_factor__s <=2# repeated?
    #     pass
    
    if iota_of_dim is None:
        iota_of_dim = iota(input.shape[0])
        pass
    
    #<  real payload
    mat = vector_length_norm(input)
    # old code ori_mat = ____init_mat_with_row_len_is_1(dim, device=device)

    #<  calc grad>
    manual__mat_matmul_mat__d_d = mat@(mat.T)
    manual__mat_matmul_mat__d_d[iota_of_dim, iota_of_dim] = 0.

    #<  original grad>
    ori__grad__d_d = manual__mat_matmul_mat__d_d@mat
    #<  original grad, but len into 1>
    len_1__ori_grad__d_d = vector_length_norm(ori__grad__d_d)
    #assert _tensor_equal(get_vector_length(len_1__ori_grad__d_d), torch.ones(size=[input.shape[0]], device=input.device))

    #<  scale it a bit, to make the distribution a bit wider>
    #otherwise, they are a lot 0.8,0.9. Wider means most are 0.3 to 0.8.
    #<old code/>xx__grad_len_sqr__dim = ori__grad__d_d.mul(ori__grad__d_d).sum(dim=1)#mul and then sum, it's a dot.
    ori__grad_len__dim = get_vector_length(ori__grad__d_d)
    max_of__ori__grad_len__dim = ori__grad_len__dim.max()
    ratio_of__grad_len__dim = ori__grad_len__dim/max_of__ori__grad_len__dim
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this scaled_to_1__grad_len_sqr__dim is still one 1.0 and a lot 0.xx
    #assert ratio_of__grad_len__dim.le(1.).all()
    #assert ratio_of__grad_len__dim.eq(1.).sum() == 1

    grad_len__after_expansion__dim = ratio_of__grad_len__dim.pow(expansion_factor__s)
    # assert grad_len__after_expansion__dim.le(1.).all() when expansion_factor__s is neg, this is wrong.
    # assert grad_len__after_expansion__dim.eq(1.).sum() == 1 when expansion_factor__s is 0., this is wrong.
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this grad_len__after_expansion__dim is still one 1.0 and a lot 0.xx

    # if "visualization" and False:
    #     #prin(expansion_factor__s)
    #     visualize_this = grad_len__after_expansion__dim
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(tight_layout=True)
    #     n_bins = dim//5
    #     if n_bins<=3:
    #         n_bins = 3
    #         pass
    #     ax.hist(visualize_this.tolist(), bins=n_bins)
    #     plt.show()
    #     pass

    #<  target length>
    grad_length_target__dim = grad_len__after_expansion__dim*cap_to__s
    grad_length_target__dim__expand_dim = expand_vec_to_matrix(grad_length_target__dim, each_element_to="row")
#1w
    #<  finally 
    useful_grad__d_d = len_1__ori_grad__d_d.mul(grad_length_target__dim__expand_dim)#row
    # # assert _tensor_equal(get_vector_length(useful_grad__d_d), grad_length_target__dim)

    # debug only  vvvvvvvvvvvvvvvvvvvv
    # log10_of__mat = log10_avg_safe(mat)
    # log10_of__grad = log10_avg_safe(useful_grad__d_d)
    # debug only  ^^^^^^^^^^^^^^^^^^^^

    #<  update mat
    mat -= useful_grad__d_d

    # debug only  vvvvvvvvvvvvvvvvvvvv
    # log10_of__mat_after_modify = log10_avg_safe(mat)
    # log10_similarity = log10_avg__how_similar(mat, ori_mat)
    # debug only  ^^^^^^^^^^^^^^^^^^^^

    #<  protect the length after updating
    mat = vector_length_norm(mat)
    
    return mat

def random_dummy_mat(dim:int, init__cap_to = 0.2, noise_strength = 0.2, 
                    device='cpu', iota_of_dim:torch.Tensor|None = None)->torch.Tensor:
    '''
    init__cap_to in[0.1, 0.5], recommended[0.2, 0.3]
    noise_strength in[0, 3], recommended[0.2, 0.5]
    or if you know what you are doing.
    '''
    if iota_of_dim is None:
        iota_of_dim = iota(dim)
        pass
    
    #<  real job
    mat = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)
    
    #<  dummy op it a bit.
    mat = full_test_version_of_angle_correction__by_row(mat, 
                cap_to=init__cap_to/2., iota_of_dim=iota_of_dim)
    mat = full_test_version_of_angle_correction__by_row(mat.T,
                cap_to=init__cap_to/2., iota_of_dim=iota_of_dim).T# .T in and .T out, this is col-wise
    
    #<  some noise to mimic the learning update.
    
    assert False,"new code untested."
    #new code
    _mul_me = noise_strength/math.sqrt(dim)
    mat += torch.randn_like(mat)*_mul_me
    
    #old code
    #mat += torch.randn_like(mat)/math.sqrt(dim)*noise_strength
    return mat

"maybe a random_dummy_mat from target??"
def _param_for__random_dummy_mat(dim:int, target_angle_score:float)->tuple[float, float]:
    '''return init__cap_to, noise_strength
    
    it's a rough reverse lookup for 
    >>> mat = random_dummy_mat(dim=dim,init__cap_to = init__cap_to, noise_strength = noise_strength)
    >>> _, angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
    '''
    assert target_angle_score>=0.9 and target_angle_score<=1.55

    if dim < 316: # returns the result of dim == 100    
        if target_angle_score>1.53:#1.55
            return 0.05, 0.7
        elif target_angle_score>1.45:#1.5
            return 0.08, 0.5
        elif target_angle_score>1.35:#1.4
            return 0.16, 0.45
        elif target_angle_score>1.25:#1.3
            return 0.22, 0.32
        elif target_angle_score>1.15:#1.2
            return 0.27, 0.31
        elif target_angle_score>1.05:#1.1
            return 0.33, 0.16
        elif target_angle_score>0.95:#1.0
            return 0.42, 0.16
        else:#0.9
            return 0.47, 0.05
        pass#if dim
    else:# returns the result of dim == 100    
        if target_angle_score>1.53:#1.55
            return 0.05, 0.7
        elif target_angle_score>1.45:#1.5
            return 0.08, 0.5
        elif target_angle_score>1.35:#1.4
            return 0.16, 0.48
        elif target_angle_score>1.25:#1.3
            return 0.22, 0.39
        elif target_angle_score>1.15:#1.2
            return 0.27, 0.32
        elif target_angle_score>1.05:#1.1
            return 0.33, 0.27
        elif target_angle_score>0.95:#1.0
            return 0.42, 0.26
        else:#0.9
            return 0.47, 0.17
        pass#if dim
    pass# end of function

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

def random_dummy_mat__v2__from_target(dim:int, target_angle_loss:torch.Tensor,
                    device='cpu', iota_of_dim:torch.Tensor|None = None)->torch.Tensor:
    '''
    target_angle_loss inside [0., 1.6]
    '''
    assert target_angle_loss >= 0. and target_angle_loss <= 1.6
    
    noise_strength_list = torch.tensor(
        [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.10,    1.30,    1.80,     ])
    angle_loss_list__full = torch.tensor([
        [ 0.0000,  0.2184,  0.4341,  0.6253,  0.7969,  0.9537,  1.0867,  1.1974,  1.3483,  1.4409,  1.5145,  1.5496,   ],#dim 10
        [ 0.0000,  0.2234,  0.4377,  0.6346,  0.8075,  0.9575,  1.0838,  1.1826,  1.3292,  1.4228,  1.4859,  1.5540,   ],#dim 100
        [ 0.0000,  0.2238,  0.4381,  0.6346,  0.8080,  0.9577,  1.0820,  1.1828,  1.3303,  1.4248,  1.4814,  1.5506,   ],])# dim 1000
    
    log10_of_dim = torch.tensor(dim).log10()
    log10_of_dim.clamp_(1., 3.)
    angle_loss_list__for_this_dim = interpolation_of_list(angle_loss_list__full, log10_of_dim -1.)# a row
    assert angle_loss_list__for_this_dim.shape[0] == angle_loss_list__full.shape[-1]
    
    _index_float = reverse_interpolation_of_list__list_must_sorted(angle_loss_list__for_this_dim, list_is_Ascending = True, 
                                                        the_input=target_angle_loss, Im_sure_the_list_is_sorted=True)
    assert _index_float <= angle_loss_list__for_this_dim.nelement()-1
    
    noise_strength = interpolation_of_list(noise_strength_list, _index_float)
    
    return random_dummy_mat__v2(dim = dim, noise_strength = noise_strength, device = device, iota_of_dim = iota_of_dim)




if "test   random_dummy_mat__v2" and False:
    def ____test____random_dummy_mat__v2():
        
        if "random_dummy_mat__v2  reverse lookup table" and False:
            
            if True:
                # dim 10      (x axis is noise_strength, y axis is cap_to)                                                                                                 
                # noise_stre=[ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.10,    1.30,    1.80,     ]
                # angle_loss [ 0.0000,  0.2184,  0.4341,  0.6253,  0.7969,  0.9537,  1.0867,  1.1974,  1.3483,  1.4409,  1.5145,  1.5496,   ]
                # dim 100      (x axis is noise_strength, y axis is cap_to)                                                                               
                # noise_stre=[ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.10,    1.30,    1.80,     ]
                # angle_loss [ 0.0000,  0.2234,  0.4377,  0.6346,  0.8075,  0.9575,  1.0838,  1.1826,  1.3292,  1.4228,  1.4859,  1.5540,   ]
                # dim 1000      (x axis is noise_strength, y axis is cap_to)                                                                               
                # noise_stre=[ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.90,    1.10,    1.30,    1.80,     ]
                # angle_loss [ 0.0000,  0.2238,  0.4381,  0.6346,  0.8080,  0.9577,  1.0820,  1.1828,  1.3303,  1.4248,  1.4814,  1.5506,   ]
                
                pass
            
            #result 
            
            #-------------------#-------------------#-------------------
            dim_list =       [ 10,100,1000]
            test_time_list = [200,100,5]#500 for dim100 is too slow...
            for outter_iter_count in range(dim_list.__len__()):
                dim = dim_list[outter_iter_count]
                test_time = test_time_list[outter_iter_count]
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                print(f"dim {dim}   test_time {test_time}  {device}")
            #-------------------#-------------------#-------------------
            
                x_axis___dim = 14
                x_axis___dim = 4#########################
            
                angle_loss__avg = torch.empty(size=[x_axis___dim])#dont modify this.
                angle_loss__avg.fill_(torch.nan)
                
                #-------------------#-------------------#-------------------
                noise_strength_list = torch.linspace(0., 1.3, x_axis___dim) #x axis
                noise_strength_list = torch.linspace(1.6, 1.9, x_axis___dim) ##########################
                for _init__x_axis___dim in range(x_axis___dim):
                    noise_strength = noise_strength_list[_init__x_axis___dim]
                #-------------------#-------------------#-------------------
                    
                    _raw_result__angle_loss = torch.empty(size=[test_time])#dont modify this.
                    _raw_result__angle_loss.fill_(torch.nan)
                    
                    for _test_count in range(test_time):
                        
                        #-------------------#-------------------#-------------------
                        mat = random_dummy_mat__v2(dim=dim, noise_strength=noise_strength,
                                                device=device, iota_of_dim=iota_of_dim)
                        _, angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                        #-------------------#-------------------#-------------------
                        
                        _raw_result__angle_loss[_test_count] = angle_loss
                        pass# for _test_count
                    
                    angle_loss__avg[_init__x_axis___dim] = _raw_result__angle_loss.mean()
                    pass# for _init__x_axis___dim
            
                print(f"# dim {dim}")
                print(f"# noise_stre={str_the_list(noise_strength_list, 2, segment=",   ")}")
                print(f"# angle_loss {str_the_list(angle_loss__avg, 4)}")
                
                pass# for outter_iter_count
            
            pass#/ test
        
        if "random_dummy_mat__v2__from_target accuracy test" and False:
            # dim 10    
            # target= [ 0.00,    0.10,    0.20,    0.30,   |  0.40,    0.50,    0.60,    0.70,    0.80,    0.90,    1.00,    1.10,    1.20,    1.30,    1.40,    1.50,    1.60]
            # neg   = [ 0.0000, -0.0236, -0.0458, -0.0608, | -0.1324, -0.1200, -0.1317, -0.1850, -0.2514, -0.2221, -0.2145, -0.2030, -0.4464, -0.2907, -0.2721, -0.3367, -0.3567]
            # pos   = [ 0.0000,  0.0237,  0.0647,  0.0640, |  0.0812,  0.1521,  0.2207,  0.1901,  0.1619,  0.2753,  0.2464,  0.2446,  0.2334,  0.3483,  0.4630,  0.4495,  0.3342]
            # diff  = [ 0.0000,  0.0004, -0.0009, -0.0005, | -0.0097, -0.0089,  0.0143,  0.0128,  0.0086,  0.0123,  0.0150,  0.0012, -0.0240,  0.0040,  0.0022,  0.0176, -0.0613]
            # dim 32    
            # target= [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.80,    0.90,    1.00,    1.10,   |  1.20,    1.30,    1.40,    1.50,    1.60]
            # neg   = [ 0.0000, -0.0068, -0.0124, -0.0184, -0.0218, -0.0299, -0.0341, -0.0469, -0.0476, -0.0753, -0.0986, -0.0815, | -0.0756, -0.1008, -0.0951, -0.1250, -0.1433]
            # pos   = [ 0.0000,  0.0103,  0.0181,  0.0315,  0.0363,  0.0388,  0.0514,  0.0481,  0.0821,  0.0726,  0.0811,  0.0666, |  0.1027,  0.1069,  0.1290,  0.0818,  0.0596]
            # diff  = [ 0.0000,  0.0011,  0.0015,  0.0036,  0.0027,  0.0064,  0.0070,  0.0043,  0.0032,  0.0034,  0.0016, -0.0009, |  0.0017,  0.0036,  0.0069, -0.0112, -0.0483]
            # dim 100    
            # target= [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.80,    0.90,    1.00,    1.10,    1.20,    1.30,    1.40,    1.50,    1.60]
            # neg   = [ 0.0000, -0.0017, -0.0060, -0.0045, -0.0071, -0.0099, -0.0107, -0.0208, -0.0305, -0.0257, -0.0198, -0.0416, -0.0236, -0.0252, -0.0356, -0.0356, -0.0781]
            # pos   = [ 0.0000,  0.0031,  0.0054,  0.0117,  0.0095,  0.0146,  0.0177,  0.0213,  0.0186,  0.0250,  0.0241,  0.0263,  0.0386,  0.0368,  0.0540,  0.0511, -0.0135]
            # diff  = [ 0.0000,  0.0006,  0.0002,  0.0018,  0.0003,  0.0028,  0.0014,  0.0019,  0.0015,  0.0023,  0.0019, -0.0026,  0.0045,  0.0077,  0.0071,  0.0055, -0.0469]
            # dim 316    
            # target= [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.80,    0.90,    1.00,    1.10,    1.20,    1.30,    1.40,    1.50,    1.60]
            # neg   = [ 0.0000,  0.0003,  0.0001,  0.0008, -0.0005, -0.0003, -0.0017,  0.0006,  0.0005, -0.0008, -0.0027, -0.0032,  0.0012, -0.0035, -0.0023, -0.0031, -0.0578]
            # pos   = [ 0.0000,  0.0016,  0.0023,  0.0029,  0.0037,  0.0048,  0.0045,  0.0064,  0.0049,  0.0082,  0.0102,  0.0045,  0.0090,  0.0138,  0.0097,  !!! 0.0138, -0.0398]
            # diff  = [ 0.0000,  0.0009,  0.0009,  0.0021,  0.0018,  0.0016,  0.0015,  0.0035,  0.0021,  0.0032,  0.0028,  0.0006,  0.0047,  0.0065,  0.0037,  0.0059, -0.0479]
            # dim 1000    
            # target= [ 0.00,    0.10,    0.20,    0.30,    0.40,    0.50,    0.60,    0.70,    0.80,    0.90,    1.00,    1.10,    1.20,    1.30,    1.40,    1.50,    1.60]
            # neg   = [ 0.0000,  0.0006, -0.0001,  0.0012,  0.0004,  0.0014, -0.0004,  0.0023, -0.0007,  0.0022,  0.0017, -0.0009,  0.0016,  0.0040,  0.0009,  0.0059, -0.0520]
            # pos   = [ 0.0000,  0.0009,  0.0010,  0.0022,  0.0017,  0.0033,  0.0022,  0.0050,  0.0022,  0.0044,  0.0030,  0.0031,  0.0052,  0.0068,  0.0048,  0.0093, -0.0451]
            # diff  = [ 0.0000,  0.0007,  0.0004,  0.0017,  0.0010,  0.0024,  0.0016,  0.0032,  0.0010,  0.0032,  0.0024,  0.0012,  0.0034,  0.0053,  0.0028,  0.0078, -0.0493]
            
            # conclusion
            # more dim, less error.
            # smaller target, less error.
            
            #---------------------#---------------------#---------------------
            dim_list =          [10,   32,   100,  316, 1000]
            test_time_list =    [100, 100,  100,  10, 10]
            for outter_param_list in range(dim_list.__len__()):
                dim =               dim_list        [outter_param_list]
                test_time =  test_time_list  [outter_param_list]
                device = get_device(dim)
                print(f"dim {dim}   test_time {test_time}  {device}")
            #---------------------#---------------------#---------------------
                
                neg = []
                pos = []
                diff = []#dont modify this.
                
                #---------------------#---------------------#---------------------
                target_list = torch.linspace(0., 1.6, 17)
                for target_angle_loss in target_list:
                #---------------------#---------------------#---------------------
                    
                    _raw_result = torch.empty(size=[test_time])
                    for _test_count in range(test_time):
                        
                        #---------------------#---------------------#---------------------
                        mat = random_dummy_mat__v2__from_target(dim=dim,target_angle_loss=target_angle_loss, 
                                                                                        device=device)
                        _, angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                        #---------------------#---------------------#---------------------
                        
                        _raw_result[_test_count] = angle_loss
                        pass# for _test_count
                    neg .append(_raw_result.min()-target_angle_loss)
                    pos .append(_raw_result.max()-target_angle_loss)
                    diff.append(_raw_result.mean()-target_angle_loss)
                    pass# for target_angle_loss
                print(f"dim {dim}    ")
                print(f"target= {str_the_list(target_list, 2, segment=",   ")}")
                print(f"neg   = {str_the_list(neg, 4)}")
                print(f"pos   = {str_the_list(pos, 4)}")
                print(f"diff  = {str_the_list(diff, 4)}")
                pass# for outter_param_list   dim
            
            pass#/ test
        
        return
        
    ____test____random_dummy_mat__v2()
    pass



if "test   random_dummy_mat" and False:
    
    # basic behavior test of the dummy_mat v1
    def ____test____random_dummy_mat():
        
        if "how to init a noisy mat for test" and False:
            #result 
            
            if "the length is affected too slightly, no need to correct it back." and False:          
                
                # dim 10    init__cap_to 0.1
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9754,  0.9759,  0.9768,  0.9759,  0.9796,  0.9745,  0.9808,  0.9776,  0.9792,  0.9799]
                # length_0_std   = [ 0.2134,  0.2071,  0.2128,  0.2104,  0.2110,  0.2103,  0.2129,  0.2128,  0.2125,  0.2123]
                # length_1_avg   = [ 0.9821,  0.9826,  0.9818,  0.9825,  0.9821,  0.9824,  0.9822,  0.9820,  0.9821,  0.9826]
                # length_1_std   = [ 0.1930,  0.1905,  0.1949,  0.1920,  0.1936,  0.1914,  0.1928,  0.1936,  0.1930,  0.1901]
                # length_2_avg   = [ 0.9980,  0.9981,  0.9980,  0.9980,  0.9980,  0.9980,  0.9980,  0.9980,  0.9979,  0.9980]
                # length_2_std   = [ 0.0636,  0.0627,  0.0632,  0.0637,  0.0634,  0.0642,  0.0638,  0.0629,  0.0646,  0.0644]
                # angle_loss__min= [ 0.9613,  0.9958,  0.9282,  1.0402,  1.0533,  1.1147,  1.1184,  1.1690,  1.2772,  1.2035]
                # angle_loss__max= [ 1.7960,  1.8114,  1.8044,  1.9181,  1.8995,  2.0109,  2.0356,  2.0359,  2.1694,  2.2320]
                # angle_loss__avg= [ 1.3553,  1.3545,  1.3600,  1.3737,  1.4202,  1.4731,  1.5229,  1.5418,  1.6011,  1.5930]
                # --------------                                          
                # dim 10    init__cap_to 0.2
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9718,  0.9732,  0.9776,  0.9808,  0.9706,  0.9761,  0.9692,  0.9737,  0.9773,  0.9735]
                # length_0_std   = [ 0.2085,  0.2128,  0.2097,  0.2100,  0.2114,  0.2108,  0.2103,  0.2116,  0.2118,  0.2102]
                # length_1_avg   = [ 0.9840,  0.9835,  0.9840,  0.9846,  0.9835,  0.9838,  0.9837,  0.9845,  0.9835,  0.9840]
                # length_1_std   = [ 0.1824,  0.1856,  0.1825,  0.1793,  0.1855,  0.1835,  0.1844,  0.1796,  0.1847,  0.1828]
                # length_2_avg   = [ 0.9982,  0.9982,  0.9982,  0.9984,  0.9982,  0.9983,  0.9982,  0.9983,  0.9981,  0.9982]
                # length_2_std   = [ 0.0606,  0.0603,  0.0602,  0.0576,  0.0600,  0.0588,  0.0608,  0.0590,  0.0615,  0.0601]
                # angle_loss__min= [ 0.7895,  0.8701,  0.7809,  0.9154,  1.0082,  1.0538,  1.1477,  1.1934,  1.1174,  1.1375]
                # angle_loss__max= [ 1.6789,  1.6768,  1.7559,  1.6691,  1.8454,  1.7588,  1.9716,  1.9601,  2.1236,  2.2380]
                # angle_loss__avg= [ 1.2129,  1.2330,  1.2464,  1.2636,  1.3318,  1.4145,  1.4858,  1.5262,  1.5991,  1.5905]
                # --------------                                          
                # dim 10    init__cap_to 0.3
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9783,  0.9840,  0.9760,  0.9787,  0.9744,  0.9742,  0.9774,  0.9776,  0.9778,  0.9724]
                # length_0_std   = [ 0.2106,  0.2127,  0.2127,  0.2125,  0.2108,  0.2124,  0.2123,  0.2106,  0.2113,  0.2119]
                # length_1_avg   = [ 0.9853,  0.9857,  0.9854,  0.9853,  0.9854,  0.9855,  0.9850,  0.9852,  0.9850,  0.9850]
                # length_1_std   = [ 0.1755,  0.1724,  0.1743,  0.1750,  0.1749,  0.1737,  0.1767,  0.1755,  0.1766,  0.1766]
                # length_2_avg   = [ 0.9983,  0.9984,  0.9983,  0.9984,  0.9983,  0.9983,  0.9982,  0.9984,  0.9984,  0.9984]
                # length_2_std   = [ 0.0585,  0.0572,  0.0582,  0.0583,  0.0587,  0.0583,  0.0602,  0.0575,  0.0573,  0.0577]
                # angle_loss__min= [ 0.7352,  0.6961,  0.6257,  0.7330,  0.9395,  1.0568,  1.0648,  1.1442,  1.2190,  1.2386]
                # angle_loss__max= [ 1.5433,  1.5080,  1.5331,  1.5242,  1.6165,  1.8457,  1.9148,  2.0419,  2.1348,  2.1178]
                # angle_loss__avg= [ 1.1048,  1.0998,  1.1118,  1.1472,  1.2566,  1.3644,  1.4398,  1.5080,  1.6206,  1.6102]
                # --------------                                          
                # dim 10    init__cap_to 0.4
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9720,  0.9783,  0.9716,  0.9737,  0.9741,  0.9755,  0.9779,  0.9754,  0.9798,  0.9717]
                # length_0_std   = [ 0.2096,  0.2144,  0.2097,  0.2111,  0.2115,  0.2141,  0.2145,  0.2129,  0.2141,  0.2101]
                # length_1_avg   = [ 0.9869,  0.9865,  0.9871,  0.9867,  0.9868,  0.9867,  0.9862,  0.9864,  0.9866,  0.9862]
                # length_1_std   = [ 0.1648,  0.1677,  0.1634,  0.1661,  0.1655,  0.1658,  0.1689,  0.1684,  0.1673,  0.1697]
                # length_2_avg   = [ 0.9984,  0.9983,  0.9984,  0.9984,  0.9984,  0.9984,  0.9984,  0.9983,  0.9984,  0.9983]
                # length_2_std   = [ 0.0577,  0.0587,  0.0571,  0.0579,  0.0575,  0.0581,  0.0581,  0.0586,  0.0579,  0.0591]
                # angle_loss__min= [ 0.5207,  0.6458,  0.6222,  0.6651,  0.8242,  0.9458,  1.0810,  1.0690,  1.2239,  1.1546]
                # angle_loss__max= [ 1.4020,  1.4933,  1.4732,  1.4718,  1.4961,  1.6351,  1.8532,  2.1960,  2.1345,  2.1180]
                # angle_loss__avg= [ 0.9856,  0.9939,  1.0056,  1.0501,  1.1743,  1.3200,  1.4224,  1.4858,  1.5952,  1.5979]
                # --------------                                          
                # dim 10    init__cap_to 0.5
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9769,  0.9738,  0.9801,  0.9745,  0.9783,  0.9752,  0.9754,  0.9748,  0.9755,  0.9700]
                # length_0_std   = [ 0.2104,  0.2114,  0.2125,  0.2096,  0.2133,  0.2103,  0.2119,  0.2131,  0.2111,  0.2093]
                # length_1_avg   = [ 0.9884,  0.9880,  0.9877,  0.9878,  0.9874,  0.9883,  0.9880,  0.9877,  0.9882,  0.9880]
                # length_1_std   = [ 0.1551,  0.1574,  0.1596,  0.1590,  0.1611,  0.1564,  0.1573,  0.1603,  0.1563,  0.1573]
                # length_2_avg   = [ 0.9983,  0.9982,  0.9982,  0.9982,  0.9982,  0.9982,  0.9982,  0.9982,  0.9983,  0.9983]
                # length_2_std   = [ 0.0600,  0.0606,  0.0616,  0.0614,  0.0618,  0.0609,  0.0608,  0.0611,  0.0599,  0.0595]
                # angle_loss__min= [ 0.4467,  0.3069,  0.4991,  0.6172,  0.6195,  0.9831,  1.0288,  1.1059,  1.2220,  1.2384]
                # angle_loss__max= [ 1.3827,  1.3150,  1.3200,  1.4077,  1.5719,  1.7404,  1.7933,  1.9522,  2.3370,  2.2437]
                # angle_loss__avg= [ 0.8885,  0.8941,  0.9153,  0.9576,  1.1149,  1.2771,  1.3871,  1.4559,  1.5971,  1.6093]
                # --------------                                          
                # -------------------------------------                                        
                # dim 100   test_time 300  cpu
                # dim 100    init__cap_to 0.1
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9974,  0.9971,  0.9971,  0.9974,  0.9981,  0.9977,  0.9974,  0.9977,  0.9969,  0.9973]
                # length_0_std   = [ 0.0704,  0.0702,  0.0703,  0.0703,  0.0702,  0.0704,  0.0704,  0.0705,  0.0702,  0.0702]
                # length_1_avg   = [ 0.9978,  0.9978,  0.9978,  0.9978,  0.9978,  0.9978,  0.9978,  0.9978,  0.9978,  0.9978]
                # length_1_std   = [ 0.0660,  0.0661,  0.0661,  0.0661,  0.0659,  0.0662,  0.0662,  0.0662,  0.0659,  0.0660]
                # length_2_avg   = [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]
                # length_2_std   = [ 0.0092,  0.0093,  0.0093,  0.0092,  0.0092,  0.0093,  0.0092,  0.0093,  0.0093,  0.0093]
                # angle_loss__min= [ 1.3726,  1.3693,  1.3793,  1.3832,  1.4226,  1.4645,  1.4874,  1.5185,  1.5493,  1.5522]
                # angle_loss__max= [ 1.4629,  1.4665,  1.4675,  1.4688,  1.5048,  1.5556,  1.5787,  1.6016,  1.6363,  1.6389]
                # angle_loss__avg= [ 1.4147,  1.4178,  1.4182,  1.4275,  1.4631,  1.4988,  1.5305,  1.5535,  1.5937,  1.5966]
                # --------------                                          
                # dim 100    init__cap_to 0.2
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9975,  0.9968,  0.9968,  0.9976,  0.9971,  0.9973,  0.9978,  0.9979,  0.9974,  0.9982]
                # length_0_std   = [ 0.0704,  0.0703,  0.0705,  0.0704,  0.0702,  0.0702,  0.0704,  0.0705,  0.0703,  0.0702]
                # length_1_avg   = [ 0.9981,  0.9980,  0.9980,  0.9981,  0.9981,  0.9981,  0.9981,  0.9981,  0.9981,  0.9981]
                # length_1_std   = [ 0.0620,  0.0626,  0.0626,  0.0624,  0.0622,  0.0625,  0.0624,  0.0624,  0.0620,  0.0621]
                # length_2_avg   = [ 0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999]
                # length_2_std   = [ 0.0104,  0.0104,  0.0105,  0.0105,  0.0104,  0.0104,  0.0104,  0.0104,  0.0105,  0.0103]
                # angle_loss__min= [ 1.2153,  1.2186,  1.2132,  1.2348,  1.3141,  1.3870,  1.4415,  1.4788,  1.5516,  1.5538]
                # angle_loss__max= [ 1.3100,  1.2947,  1.3136,  1.3336,  1.4011,  1.4710,  1.5289,  1.5598,  1.6406,  1.6394]
                # angle_loss__avg= [ 1.2578,  1.2567,  1.2629,  1.2852,  1.3516,  1.4233,  1.4783,  1.5179,  1.5957,  1.5965]
                # --------------                                          
                # dim 100    init__cap_to 0.3
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9971,  0.9972,  0.9985,  0.9980,  0.9974,  0.9974,  0.9972,  0.9974,  0.9974,  0.9976]
                # length_0_std   = [ 0.0703,  0.0704,  0.0703,  0.0705,  0.0705,  0.0705,  0.0704,  0.0702,  0.0707,  0.0702]
                # length_1_avg   = [ 0.9983,  0.9983,  0.9983,  0.9983,  0.9983,  0.9983,  0.9983,  0.9983,  0.9983,  0.9983]
                # length_1_std   = [ 0.0586,  0.0590,  0.0589,  0.0592,  0.0585,  0.0588,  0.0588,  0.0587,  0.0590,  0.0588]
                # length_2_avg   = [ 0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999]
                # length_2_std   = [ 0.0127,  0.0128,  0.0129,  0.0128,  0.0128,  0.0128,  0.0129,  0.0129,  0.0127,  0.0128]
                # angle_loss__min= [ 1.0593,  1.0648,  1.0729,  1.1107,  1.2069,  1.3235,  1.3945,  1.4325,  1.5561,  1.5535]
                # angle_loss__max= [ 1.1661,  1.1576,  1.1731,  1.2121,  1.2889,  1.3929,  1.4847,  1.5334,  1.6306,  1.6312]
                # angle_loss__avg= [ 1.1098,  1.1119,  1.1217,  1.1536,  1.2528,  1.3548,  1.4363,  1.4903,  1.5929,  1.5962]
                # --------------                                          
                # dim 100    init__cap_to 0.4
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9979,  0.9977,  0.9975,  0.9970,  0.9972,  0.9972,  0.9977,  0.9976,  0.9979,  0.9976]
                # length_0_std   = [ 0.0703,  0.0705,  0.0704,  0.0701,  0.0700,  0.0701,  0.0705,  0.0702,  0.0697,  0.0705]
                # length_1_avg   = [ 0.9985,  0.9984,  0.9985,  0.9985,  0.9985,  0.9985,  0.9985,  0.9985,  0.9985,  0.9985]
                # length_1_std   = [ 0.0557,  0.0558,  0.0555,  0.0554,  0.0550,  0.0551,  0.0555,  0.0552,  0.0550,  0.0556]
                # length_2_avg   = [ 0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999]
                # length_2_std   = [ 0.0158,  0.0159,  0.0159,  0.0159,  0.0160,  0.0160,  0.0159,  0.0159,  0.0159,  0.0158]
                # angle_loss__min= [ 0.9372,  0.9294,  0.9490,  0.9828,  1.1343,  1.2686,  1.3659,  1.4292,  1.5533,  1.5506]
                # angle_loss__max= [ 1.0369,  1.0325,  1.0416,  1.0851,  1.2040,  1.3412,  1.4394,  1.5093,  1.6351,  1.6401]
                # angle_loss__avg= [ 0.9808,  0.9822,  0.9962,  1.0393,  1.1695,  1.3011,  1.4012,  1.4676,  1.5943,  1.5961]
                # --------------                                          
                # dim 100    init__cap_to 0.5
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9979,  0.9979,  0.9971,  0.9976,  0.9974,  0.9971,  0.9976,  0.9969,  0.9973,  0.9976]
                # length_0_std   = [ 0.0701,  0.0706,  0.0704,  0.0702,  0.0701,  0.0703,  0.0702,  0.0706,  0.0701,  0.0702]
                # length_1_avg   = [ 0.9987,  0.9986,  0.9986,  0.9986,  0.9986,  0.9986,  0.9986,  0.9986,  0.9987,  0.9987]
                # length_1_std   = [ 0.0520,  0.0526,  0.0526,  0.0521,  0.0524,  0.0523,  0.0525,  0.0529,  0.0520,  0.0520]
                # length_2_avg   = [ 0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998]
                # length_2_std   = [ 0.0197,  0.0197,  0.0197,  0.0198,  0.0196,  0.0196,  0.0196,  0.0196,  0.0198,  0.0196]
                # angle_loss__min= [ 0.8243,  0.8291,  0.8388,  0.8845,  1.0727,  1.2249,  1.3344,  1.4121,  1.5544,  1.5486]
                # angle_loss__max= [ 0.9314,  0.9206,  0.9439,  1.0175,  1.1476,  1.2941,  1.4167,  1.4968,  1.6350,  1.6376]
                # angle_loss__avg= [ 0.8706,  0.8745,  0.8921,  0.9466,  1.1042,  1.2576,  1.3733,  1.4499,  1.5920,  1.5953]
                # --------------                                          
                # -------------------------------------                                        
                # dim 1000   test_time 50  cuda
                # dim 1000    init__cap_to 0.1
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9999,  0.9998,  0.9997,  0.9997,  0.9997,  0.9999,  0.9996,  0.9998,  0.9998,  0.9997]
                # length_0_std   = [ 0.0223,  0.0223,  0.0223,  0.0224,  0.0223,  0.0224,  0.0223,  0.0223,  0.0224,  0.0223]
                # length_1_avg   = [ 0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998]
                # length_1_std   = [ 0.0209,  0.0208,  0.0209,  0.0210,  0.0209,  0.0209,  0.0209,  0.0208,  0.0209,  0.0209]
                # length_2_avg   = [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]
                # length_2_std   = [ 0.0016,  0.0016,  0.0016,  0.0016,  0.0016,  0.0016,  0.0016,  0.0016,  0.0016,  0.0016]
                # angle_loss__min= [ 1.3956,  1.3947,  1.3969,  1.4092,  1.4480,  1.4898,  1.5206,  1.5456,  1.5910,  1.5923]
                # angle_loss__max= [ 1.4046,  1.4043,  1.4072,  1.4195,  1.4550,  1.4965,  1.5298,  1.5515,  1.5994,  1.5988]
                # angle_loss__avg= [ 1.3994,  1.3995,  1.4027,  1.4156,  1.4520,  1.4923,  1.5255,  1.5489,  1.5946,  1.5958]
                # --------------                                          
                # dim 1000    init__cap_to 0.2
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9998,  0.9997,  0.9996,  0.9999,  0.9997,  0.9998,  0.9997,  0.9997,  0.9996,  0.9999]
                # length_0_std   = [ 0.0223,  0.0223,  0.0223,  0.0223,  0.0224,  0.0223,  0.0223,  0.0223,  0.0223,  0.0224]
                # length_1_avg   = [ 0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998]
                # length_1_std   = [ 0.0196,  0.0195,  0.0194,  0.0195,  0.0195,  0.0195,  0.0195,  0.0196,  0.0195,  0.0195]
                # length_2_avg   = [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]
                # length_2_std   = [ 0.0028,  0.0028,  0.0028,  0.0028,  0.0028,  0.0028,  0.0028,  0.0028,  0.0028,  0.0028]
                # angle_loss__min= [ 1.2113,  1.2128,  1.2234,  1.2456,  1.3223,  1.4004,  1.4638,  1.5074,  1.5894,  1.5917]
                # angle_loss__max= [ 1.2247,  1.2264,  1.2330,  1.2569,  1.3319,  1.4080,  1.4723,  1.5146,  1.5967,  1.5990]
                # angle_loss__avg= [ 1.2181,  1.2200,  1.2279,  1.2515,  1.3260,  1.4047,  1.4675,  1.5104,  1.5929,  1.5957]
                # --------------                                          
                # dim 1000    init__cap_to 0.3
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9999,  0.9998,  0.9996,  0.9998,  0.9997,  0.9997,  0.9996,  0.9997,  0.9998,  0.9997]
                # length_0_std   = [ 0.0224,  0.0223,  0.0223,  0.0223,  0.0223,  0.0223,  0.0222,  0.0224,  0.0224,  0.0224]
                # length_1_avg   = [ 0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998,  0.9998]
                # length_1_std   = [ 0.0183,  0.0183,  0.0183,  0.0182,  0.0182,  0.0182,  0.0182,  0.0182,  0.0184,  0.0182]
                # length_2_avg   = [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]
                # length_2_std   = [ 0.0041,  0.0041,  0.0041,  0.0041,  0.0040,  0.0041,  0.0041,  0.0041,  0.0041,  0.0040]
                # angle_loss__min= [ 1.0516,  1.0532,  1.0632,  1.1016,  1.2134,  1.3291,  1.4155,  1.4762,  1.5884,  1.5899]
                # angle_loss__max= [ 1.0666,  1.0683,  1.0831,  1.1151,  1.2245,  1.3356,  1.4233,  1.4835,  1.5957,  1.5981]
                # angle_loss__avg= [ 1.0581,  1.0595,  1.0711,  1.1082,  1.2191,  1.3327,  1.4200,  1.4799,  1.5919,  1.5955]
                # --------------                                          
                # dim 1000    init__cap_to 0.4
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9997,  0.9999,  0.9998,  0.9997,  0.9995,  0.9998,  0.9998,  0.9996,  0.9997,  0.9998]
                # length_0_std   = [ 0.0223,  0.0223,  0.0224,  0.0224,  0.0224,  0.0224,  0.0223,  0.0223,  0.0224,  0.0224]
                # length_1_avg   = [ 0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999]
                # length_1_std   = [ 0.0171,  0.0171,  0.0171,  0.0170,  0.0171,  0.0171,  0.0171,  0.0170,  0.0172,  0.0170]
                # length_2_avg   = [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]
                # length_2_std   = [ 0.0055,  0.0055,  0.0055,  0.0055,  0.0055,  0.0055,  0.0055,  0.0055,  0.0055,  0.0055]
                # angle_loss__min= [ 0.9121,  0.9127,  0.9325,  0.9815,  1.1286,  1.2720,  1.3804,  1.4543,  1.5877,  1.5934]
                # angle_loss__max= [ 0.9384,  0.9372,  0.9464,  0.9967,  1.1393,  1.2794,  1.3876,  1.4611,  1.5951,  1.5990]
                # angle_loss__avg= [ 0.9213,  0.9229,  0.9385,  0.9871,  1.1330,  1.2755,  1.3843,  1.4568,  1.5912,  1.5960]
                # --------------                                          
                # dim 1000    init__cap_to 0.5
                # noise_strength_=[ 0.01,    0.03,    0.10,    0.20,    0.40,    0.60,    0.80,    1.00,    3.16,    10.00]
                # length_0_avg   = [ 0.9998,  0.9998,  0.9999,  0.9998,  0.9997,  0.9997,  0.9998,  0.9998,  0.9995,  0.9997]
                # length_0_std   = [ 0.0223,  0.0224,  0.0224,  0.0223,  0.0224,  0.0223,  0.0224,  0.0224,  0.0223,  0.0224]
                # length_1_avg   = [ 0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999,  0.9999]
                # length_1_std   = [ 0.0161,  0.0160,  0.0161,  0.0162,  0.0160,  0.0160,  0.0161,  0.0160,  0.0160,  0.0161]
                # length_2_avg   = [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]
                # length_2_std   = [ 0.0071,  0.0071,  0.0071,  0.0071,  0.0071,  0.0071,  0.0071,  0.0070,  0.0071,  0.0071]
                # angle_loss__min= [ 0.8031,  0.8049,  0.8244,  0.8872,  1.0645,  1.2314,  1.3544,  1.4380,  1.5874,  1.5920]
                # angle_loss__max= [ 0.8194,  0.8229,  0.8444,  0.8991,  1.0763,  1.2370,  1.3627,  1.4439,  1.5939,  1.5983]
                # angle_loss__avg= [ 0.8102,  0.8122,  0.8333,  0.8938,  1.0686,  1.2348,  1.3583,  1.4403,  1.5910,  1.5956]
                # --------------                                          
                pass
            
            # conclusion
            # init__cap_to 0.2 - 0.3
            # noise_strength_ 0.2 - 0.5
            
            
            expansion_factor = 1.#as a const
            #-------------------#-------------------#-------------------
            dim_list =       [ 10,100,1000]
            test_time_list = [500,300,50]
            for outter_iter_count in range(dim_list.__len__()):
                dim = dim_list[outter_iter_count]
                test_time = test_time_list[outter_iter_count]
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                print(f"dim {dim}   test_time {test_time}  {device}")
            #-------------------#-------------------#-------------------
            
                #-------------------#-------------------#-------------------
                init__cap_to_list = [0.1, 0.2, 0.3, 0.4, 0.5]
                for init__cap_to in init__cap_to_list:
                #-------------------#-------------------#-------------------
                    
                    length_0_avg    = []#dont modify this.
                    length_0_std    = []#dont modify this.
                    length_1_avg    = []#dont modify this.
                    length_1_std    = []#dont modify this.
                    length_2_avg    = []#dont modify this.
                    length_2_std    = []#dont modify this.
                    
                    angle_loss__min = []#dont modify this.
                    angle_loss__max = []#dont modify this.
                    angle_loss__avg = []#dont modify this.
                    
                    #-------------------#-------------------#-------------------
                    noise_strength_list = [0.01, 0.0316, 0.1, 0.2, 0.4, 0.6, 0.8, 1., 3.16, 10]#x axis
                    for noise_strength in noise_strength_list:
                    #-------------------#-------------------#-------------------
                        
                        _raw_result__len_0_avg = torch.empty(size=[test_time])
                        _raw_result__len_0_std = torch.empty(size=[test_time])
                        _raw_result__len_1_avg = torch.empty(size=[test_time])
                        _raw_result__len_1_std = torch.empty(size=[test_time])
                        _raw_result__len_2_avg = torch.empty(size=[test_time])
                        _raw_result__len_2_std = torch.empty(size=[test_time])
                        
                        _raw_result__angle_loss = torch.empty(size=[test_time])
                        
                        for _test_count in range(test_time):
                            
                            #-------------------#-------------------#-------------------
                            mat = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)
                            
                            _temp_len_0__d2 = torch.empty(size=[dim*2])
                            _temp_len_0__d2[:dim] = get_vector_length(mat)
                            _temp_len_0__d2[dim:] = get_vector_length(mat.T)
                            
                            
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                        expansion_factor=expansion_factor, cap_to=init__cap_to/2., iota_of_dim=iota_of_dim)
                            _temp_len_1__d = get_vector_length(mat.T)
                            mat = full_test_version_of_angle_correction__by_row(mat.T,
                                        expansion_factor=expansion_factor, cap_to=init__cap_to/2., iota_of_dim=iota_of_dim).T# .T in and .T out, this is col-wise
                            _temp_len_2__d = get_vector_length(mat)
                            
                            mat += torch.randn_like(mat)/math.sqrt(dim)*noise_strength
                            _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(mat)
                            #-------------------#-------------------#-------------------
                            
                            _raw_result__len_0_avg[_test_count] = _temp_len_0__d2.mean()
                            _raw_result__len_0_std[_test_count] = _temp_len_0__d2.std()
                            _raw_result__len_1_avg[_test_count] = _temp_len_1__d.mean()
                            _raw_result__len_1_std[_test_count] = _temp_len_1__d.std()
                            _raw_result__len_2_avg[_test_count] = _temp_len_2__d.mean()
                            _raw_result__len_2_std[_test_count] = _temp_len_2__d.std()
                            
                            _raw_result__angle_loss[_test_count] = angle_loss__in_the_beginning
                            pass# for _test_count
                        
                        
                        length_0_avg.append(_raw_result__len_0_avg.mean())
                        length_0_std.append(_raw_result__len_0_std.mean())
                        length_1_avg.append(_raw_result__len_1_avg.mean())
                        length_1_std.append(_raw_result__len_1_std.mean())
                        length_2_avg.append(_raw_result__len_2_avg.mean())
                        length_2_std.append(_raw_result__len_2_std.mean())
                        
                        angle_loss__min.append(_raw_result__angle_loss.min ())
                        angle_loss__max.append(_raw_result__angle_loss.max ())
                        angle_loss__avg.append(_raw_result__angle_loss.mean())
                        pass# for noise_strength   #x axis
                    
                    print(f"dim {dim}    init__cap_to {init__cap_to}")
                    print(f"noise_strength_={str_the_list(noise_strength_list, 2, segment=",   ")}")
                    print(f"length_0_avg   = {str_the_list(length_0_avg, 4)}")
                    print(f"length_0_std   = {str_the_list(length_0_std, 4)}")
                    print(f"length_1_avg   = {str_the_list(length_1_avg, 4)}")
                    print(f"length_1_std   = {str_the_list(length_1_std, 4)}")
                    print(f"length_2_avg   = {str_the_list(length_2_avg, 4)}")
                    print(f"length_2_std   = {str_the_list(length_2_std, 4)}")
                    print(f"angle_loss__min= {str_the_list(angle_loss__min, 4)}")
                    print(f"angle_loss__max= {str_the_list(angle_loss__max, 4)}")
                    print(f"angle_loss__avg= {str_the_list(angle_loss__avg, 4)}")
                    print(f"--------------                                          ")
                    
                    pass#for init__cap_to
                print(f"-------------------------------------                                        ")
                
                pass# for outter_iter_count
            pass#/ test
        
        if "more accuracy scan" and False:
            
            if True:
                
                # dim 10      (x axis is noise_strength, y axis is cap_to)
                # n s  =[ 0.00,    0.05,    0.11,    0.16,    0.21,    0.27,    0.32,    0.37,    0.43,    0.48,    0.53,    0.59,    0.64,    0.69,    0.75,    0.80]
                # 0.02 [ 1.4970,  1.4642,  1.4778,  1.4467,  1.4746,  1.4916,  1.4758,  1.4896,  1.5170,  1.5085,  1.5201,  1.5584,  1.5787,  1.5073,  1.5434,  1.5462]
                # 0.05 [ 1.4277,  1.4224,  1.4548,  1.4567,  1.4333,  1.4497,  1.4472,  1.4503,  1.4801,  1.4940,  1.4610,  1.5008,  1.5077,  1.4833,  1.5205,  1.5130]
                # 0.08 [ 1.3874,  1.4130,  1.3972,  1.4026,  1.3816,  1.4070,  1.4007,  1.4449,  1.4691,  1.4694,  1.4736,  1.4797,  1.4827,  1.5227,  1.5118,  1.5008]
                # 0.10 [ 1.3745,  1.3712,  1.3717,  1.3547,  1.3897,  1.4218,  1.4071,  1.4577,  1.4531,  1.4744,  1.4633,  1.4702,  1.4966,  1.4632,  1.5589,  1.5668]
                # 0.13 [ 1.3054,  1.2984,  1.3192,  1.3365,  1.3793,  1.3722,  1.3647,  1.3946,  1.3805,  1.4173,  1.4203,  1.4310,  1.4461,  1.4923,  1.5079,  1.5158]
                # 0.16 [ 1.2494,  1.2623,  1.2603,  1.2706,  1.2781,  1.3140,  1.3172,  1.3572,  1.3836,  1.3910,  1.4270,  1.4531,  1.4791,  1.4755,  1.4735,  1.5317]
                # 0.19 [ 1.2384,  1.2188,  1.2313,  1.2608,  1.2985,  1.2692,  1.3175,  1.3656,  1.3327,  1.3764,  1.3977,  1.3999,  1.4353,  1.4870,  1.4641,  1.5220]
                # 0.22 [ 1.2367,  1.1947,  1.1711,  1.2097,  1.2425,  1.2900,  1.2651,  1.3520,  1.3336,  1.3720,  1.3864,  1.4063,  1.4238,  1.4612,  1.4256,  1.4953]
                # 0.25 [ 1.1703,  1.1769,  1.1697,  1.1963,  1.2078,  1.2452,  1.2860,  1.2976,  1.3124,  1.3207,  1.3879,  1.4058,  1.3855,  1.4138,  1.4380,  1.4193]
                # 0.27 [ 1.1319,  1.1253,  1.1456,  1.1403,  1.1861,  1.1673,  1.2098,  1.2661,  1.2914,  1.2985,  1.3720,  1.3797,  1.3963,  1.4081,  1.4047,  1.4625]
                # 0.30 [ 1.0731,  1.1117,  1.1237,  1.1314,  1.1549,  1.1912,  1.2129,  1.2396,  1.2852,  1.2971,  1.3209,  1.3689,  1.4086,  1.3852,  1.4389,  1.4526]
                # 0.33 [ 1.0854,  1.1006,  1.0604,  1.0912,  1.1393,  1.1695,  1.1758,  1.2116,  1.2466,  1.2602,  1.2948,  1.3307,  1.3684,  1.3874,  1.4327,  1.4448]
                # 0.36 [ 1.0185,  1.0244,  1.0512,  1.0797,  1.1023,  1.1248,  1.2381,  1.1961,  1.2135,  1.2845,  1.3251,  1.3198,  1.3457,  1.3721,  1.3965,  1.4156]
                # 0.39 [ 0.9910,  0.9909,  1.0212,  1.0367,  1.0559,  1.1270,  1.1565,  1.1632,  1.2298,  1.1961,  1.2954,  1.3237,  1.3563,  1.3726,  1.4219,  1.4125]
                # 0.42 [ 0.9732,  0.9566,  1.0083,  0.9943,  1.0537,  1.1020,  1.0958,  1.1372,  1.1940,  1.2321,  1.2875,  1.2842,  1.3103,  1.3543,  1.3737,  1.4104]
                # 0.44 [ 0.9442,  0.9249,  0.9630,  0.9633,  0.9833,  1.0804,  1.1049,  1.1144,  1.1722,  1.2316,  1.2280,  1.3038,  1.3040,  1.3622,  1.3664,  1.4285]
                # 0.47 [ 0.9029,  0.9093,  0.9623,  0.9924,  0.9923,  1.0519,  1.0828,  1.1313,  1.1478,  1.1831,  1.2435,  1.3044,  1.3164,  1.3459,  1.3480,  1.3759]
                # 0.50 [ 0.9057,  0.9146,  0.9235,  0.9228,  0.9516,  1.0040,  1.0317,  1.1028,  1.1583,  1.2034,  1.2372,  1.2649,  1.2698,  1.3266,  1.3730,  1.4230]
                
                
                # dim 100      (x axis is noise_strength, y axis is cap_to)
                # n s  =[ 0.00,    0.05,    0.11,    0.16,    0.21,    0.27,    0.32,    0.37,    0.43,    0.48,    0.53,    0.59,    0.64,    0.69,    0.75,    0.80]
                # 0.02 [ 1.5476,  1.5481,  1.5484,  1.5506,  1.5516,  1.5523,  1.5564,  1.5576,  1.5631,  1.5649,  1.5658,  1.5685,  1.5726,  1.5742,  1.5759,  1.5802]
                # 0.05 [ 1.4990,  1.5021,  1.5018,  1.5066,  1.5087,  1.5128,  1.5169,  1.5229,  1.5310,  1.5335,  1.5399,  1.5438,  1.5497,  1.5548,  1.5572,  1.5608]
                # 0.08 [ 1.4528,  1.4527,  1.4562,  1.4601,  1.4660,  1.4719,  1.4798,  1.4880,  1.4964,  1.5022,  1.5117,  1.5188,  1.5271,  1.5325,  1.5388,  1.5454]
                # 0.10 [ 1.4055,  1.4073,  1.4116,  1.4172,  1.4247,  1.4333,  1.4439,  1.4537,  1.4608,  1.4731,  1.4841,  1.4950,  1.5051,  1.5141,  1.5210,  1.5285]
                # 0.13 [ 1.3609,  1.3630,  1.3663,  1.3724,  1.3823,  1.3930,  1.4034,  1.4194,  1.4324,  1.4443,  1.4581,  1.4705,  1.4825,  1.4945,  1.5052,  1.5141]
                # 0.16 [ 1.3168,  1.3179,  1.3231,  1.3297,  1.3411,  1.3553,  1.3700,  1.3850,  1.4008,  1.4173,  1.4344,  1.4490,  1.4616,  1.4756,  1.4879,  1.4985]
                # 0.19 [ 1.2707,  1.2740,  1.2804,  1.2908,  1.3032,  1.3183,  1.3349,  1.3539,  1.3743,  1.3913,  1.4092,  1.4262,  1.4432,  1.4582,  1.4712,  1.4852]
                # 0.22 [ 1.2257,  1.2308,  1.2389,  1.2505,  1.2634,  1.2825,  1.3021,  1.3226,  1.3455,  1.3645,  1.3854,  1.4059,  1.4249,  1.4412,  1.4563,  1.4695]
                # 0.25 [ 1.1850,  1.1887,  1.1995,  1.2110,  1.2275,  1.2466,  1.2691,  1.2929,  1.3171,  1.3396,  1.3642,  1.3870,  1.4050,  1.4246,  1.4414,  1.4594]
                # 0.27 [ 1.1450,  1.1483,  1.1596,  1.1721,  1.1916,  1.2132,  1.2376,  1.2645,  1.2910,  1.3154,  1.3424,  1.3643,  1.3897,  1.4094,  1.4276,  1.4476]
                # 0.30 [ 1.1059,  1.1085,  1.1201,  1.1348,  1.1567,  1.1808,  1.2085,  1.2374,  1.2653,  1.2938,  1.3216,  1.3463,  1.3719,  1.3959,  1.4166,  1.4343]
                # 0.33 [ 1.0703,  1.0713,  1.0829,  1.0998,  1.1246,  1.1491,  1.1780,  1.2109,  1.2413,  1.2718,  1.3019,  1.3302,  1.3576,  1.3813,  1.4044,  1.4246]
                # 0.36 [ 1.0319,  1.0346,  1.0478,  1.0676,  1.0913,  1.1185,  1.1535,  1.1853,  1.2187,  1.2535,  1.2841,  1.3140,  1.3426,  1.3692,  1.3912,  1.4138]
                # 0.39 [ 0.9962,  1.0016,  1.0133,  1.0328,  1.0609,  1.0918,  1.1252,  1.1621,  1.1970,  1.2348,  1.2678,  1.2995,  1.3287,  1.3566,  1.3825,  1.4046]
                # 0.42 [ 0.9628,  0.9655,  0.9792,  1.0030,  1.0312,  1.0633,  1.1024,  1.1394,  1.1786,  1.2151,  1.2511,  1.2852,  1.3171,  1.3464,  1.3722,  1.3936]
                # 0.44 [ 0.9294,  0.9374,  0.9493,  0.9740,  1.0046,  1.0409,  1.0791,  1.1194,  1.1598,  1.1998,  1.2371,  1.2720,  1.3040,  1.3350,  1.3634,  1.3879]
                # 0.47 [ 0.9002,  0.9066,  0.9221,  0.9460,  0.9793,  1.0159,  1.0575,  1.1005,  1.1416,  1.1839,  1.2234,  1.2604,  1.2947,  1.3249,  1.3550,  1.3803]
                # 0.50 [ 0.8731,  0.8759,  0.8952,  0.9187,  0.9547,  0.9940,  1.0364,  1.0833,  1.1265,  1.1696,  1.2098,  1.2496,  1.2842,  1.3186,  1.3466,  1.3731]
                # 1.55  cap_to = 0.05, noise_strength = 0.7
                # 1.5   cap_to = 0.08, noise_strength = 0.5
                # 1.4   cap_to = 0.16, noise_strength = 0.45
                # 1.3   cap_to = 0.22, noise_strength = 0.32
                # 1.2   cap_to = 0.27, noise_strength = 0.31
                # 1.1   cap_to = 0.33, noise_strength = 0.16
                # 1.0   cap_to = 0.42, noise_strength = 0.16
                # 0.9   cap_to = 0.47, noise_strength = 0.05
                
                
                # dim 1000      (x axis is noise_strength, y axis is cap_to)
                # n s  =[ 0.00,    0.05,    0.11,    0.16,    0.21,    0.27,    0.32,    0.37,    0.43,    0.48,    0.53,    0.59,    0.64,    0.69,    0.75,    0.80]
                # 0.02 [ 1.5541,  1.5545,  1.5552,  1.5563,  1.5577,  1.5596,  1.5616,  1.5642,  1.5662,  1.5682,  1.5706,  1.5731,  1.5750,  1.5768,  1.5783,  1.5804]
                # 0.05 [ 1.4981,  1.4993,  1.5011,  1.5033,  1.5072,  1.5116,  1.5162,  1.5218,  1.5265,  1.5323,  1.5379,  1.5425,  1.5479,  1.5525,  1.5564,  1.5608]
                # 0.08 [ 1.4444,  1.4448,  1.4475,  1.4516,  1.4567,  1.4643,  1.4719,  1.4805,  1.4882,  1.4972,  1.5053,  1.5139,  1.5211,  1.5286,  1.5348,  1.5412]
                # 0.10 [ 1.3900,  1.3920,  1.3952,  1.4010,  1.4092,  1.4181,  1.4287,  1.4400,  1.4519,  1.4633,  1.4747,  1.4856,  1.4960,  1.5057,  1.5147,  1.5228]
                # 0.13 [ 1.3375,  1.3393,  1.3439,  1.3516,  1.3610,  1.3737,  1.3872,  1.4011,  1.4159,  1.4305,  1.4455,  1.4588,  1.4715,  1.4834,  1.4950,  1.5049]
                # 0.16 [ 1.2872,  1.2881,  1.2945,  1.3037,  1.3157,  1.3312,  1.3472,  1.3641,  1.3818,  1.3996,  1.4170,  1.4333,  1.4487,  1.4630,  1.4767,  1.4882]
                # 0.19 [ 1.2369,  1.2393,  1.2464,  1.2580,  1.2720,  1.2886,  1.3076,  1.3278,  1.3495,  1.3696,  1.3895,  1.4090,  1.4269,  1.4433,  1.4587,  1.4729]
                # 0.22 [ 1.1897,  1.1917,  1.1997,  1.2122,  1.2288,  1.2488,  1.2705,  1.2938,  1.3176,  1.3411,  1.3636,  1.3853,  1.4055,  1.4246,  1.4422,  1.4582]
                # 0.25 [ 1.1428,  1.1459,  1.1551,  1.1695,  1.1881,  1.2109,  1.2352,  1.2608,  1.2884,  1.3138,  1.3395,  1.3634,  1.3870,  1.4073,  1.4264,  1.4440]
                # 0.27 [ 1.0972,  1.1003,  1.1131,  1.1266,  1.1489,  1.1736,  1.2020,  1.2304,  1.2596,  1.2890,  1.3163,  1.3429,  1.3682,  1.3909,  1.4121,  1.4312]
                # 0.30 [ 1.0546,  1.0571,  1.0694,  1.0878,  1.1102,  1.1387,  1.1685,  1.2007,  1.2326,  1.2644,  1.2946,  1.3237,  1.3509,  1.3752,  1.3984,  1.4191]
                # 0.33 [ 1.0130,  1.0175,  1.0303,  1.0501,  1.0754,  1.1057,  1.1386,  1.1730,  1.2077,  1.2417,  1.2755,  1.3058,  1.3353,  1.3618,  1.3859,  1.4075]
                # 0.36 [ 0.9742,  0.9786,  0.9924,  1.0136,  1.0414,  1.0739,  1.1106,  1.1468,  1.1844,  1.2208,  1.2566,  1.2894,  1.3200,  1.3487,  1.3741,  1.3974]
                # 0.39 [ 0.9374,  0.9415,  0.9562,  0.9795,  1.0097,  1.0452,  1.0829,  1.1227,  1.1628,  1.2019,  1.2392,  1.2737,  1.3067,  1.3361,  1.3634,  1.3880]
                # 0.42 [ 0.9010,  0.9074,  0.9224,  0.9482,  0.9793,  1.0173,  1.0585,  1.1005,  1.1432,  1.1840,  1.2230,  1.2596,  1.2941,  1.3254,  1.3537,  1.3794]
                # 0.44 [ 0.8683,  0.8737,  0.8916,  0.9179,  0.9524,  0.9926,  1.0352,  1.0795,  1.1238,  1.1672,  1.2088,  1.2470,  1.2824,  1.3150,  1.3444,  1.3714]
                # 0.47 [ 0.8383,  0.8442,  0.8624,  0.8901,  0.9265,  0.9690,  1.0142,  1.0614,  1.1076,  1.1527,  1.1955,  1.2354,  1.2725,  1.3064,  1.3368,  1.3647]
                # 0.50 [ 0.8111,  0.8179,  0.8353,  0.8658,  0.9034,  0.9486,  0.9960,  1.0445,  1.0926,  1.1393,  1.1840,  1.2249,  1.2632,  1.2979,  1.3294,  1.3581]
                # 1.55  cap_to = 0.05, noise_strength = 0.7
                # 1.5   cap_to = 0.08, noise_strength = 0.5
                # 1.4   cap_to = 0.16, noise_strength = 0.48
                # 1.3   cap_to = 0.22, noise_strength = 0.39
                # 1.2   cap_to = 0.27, noise_strength = 0.32
                # 1.1   cap_to = 0.33, noise_strength = 0.27
                # 1.0   cap_to = 0.42, noise_strength = 0.26
                # 0.9   cap_to = 0.47, noise_strength = 0.17
                
                pass
            
            
            
            #result 
            
            expansion_factor = 1.#as a const
            #-------------------#-------------------#-------------------
            dim_list =       [ 10,100,1000]
            test_time_list = [50,300,50]
            for outter_iter_count in range(dim_list.__len__()):
                dim = dim_list[outter_iter_count]
                test_time = test_time_list[outter_iter_count]
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                print(f"dim {dim}   test_time {test_time}  {device}")
            #-------------------#-------------------#-------------------
            
                x_axis___dim = 16
                y_axis___dim = 18
            
                angle_loss__avg = torch.empty(size=[y_axis___dim, x_axis___dim])#dont modify this.
                angle_loss__avg.fill_(torch.nan)
                    
                #-------------------#-------------------#-------------------
                init__cap_to_list = torch.linspace(0.02, 0.5, y_axis___dim)
                for _init__y_axis___dim in range(y_axis___dim):
                    init__cap_to = init__cap_to_list[_init__y_axis___dim]
                #-------------------#-------------------#-------------------
                    
                    #-------------------#-------------------#-------------------
                    noise_strength_list = torch.linspace(0., 0.8, x_axis___dim) #x axis
                    for _init__x_axis___dim in range(x_axis___dim):
                        noise_strength = noise_strength_list[_init__x_axis___dim]
                    #-------------------#-------------------#-------------------
                        
                        _raw_result__angle_loss = torch.empty(size=[test_time])#dont modify this.
                        _raw_result__angle_loss.fill_(torch.nan)
                        
                        for _test_count in range(test_time):
                            
                            #-------------------#-------------------#-------------------
                            mat = random_dummy_mat(dim=dim,init__cap_to=init__cap_to, noise_strength=noise_strength,
                                                    device=device, iota_of_dim=iota_of_dim)
                            _, angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                            #-------------------#-------------------#-------------------
                            
                            _raw_result__angle_loss[_test_count] = angle_loss
                            pass# for _test_count
                        
                        angle_loss__avg[_init__y_axis___dim, _init__x_axis___dim] = _raw_result__angle_loss.mean()
                        pass# for _init__x_axis___dim
                    pass# for _init__y_axis___dim
                
                print(f"# dim {dim}      (x axis is noise_strength, y axis is cap_to)")
                print(f"# n s  ={str_the_list(noise_strength_list, 2, segment=",   ")}")
                for ii in range(y_axis___dim):
                    print(f"# {init__cap_to_list[ii].item():.2f} {str_the_list(angle_loss__avg[ii], 4)}")
                    pass
                print(f"# -------------------------------------                                        ")
                
                pass# for outter_iter_count
            
            pass#/ test
        
        if "validation for the param_lookup_func" and False:
            "assertion only. no print."
            dim_list = [100,1000]
            for dim in dim_list:
                target_list = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55]
                for target in target_list:
                    for _ in range(45):
                        init__cap_to, noise_strength = _param_for__random_dummy_mat(dim=dim, target_angle_score=target)
                        mat = random_dummy_mat(dim=dim,init__cap_to = init__cap_to, noise_strength = noise_strength)
                        _, angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                        assert _tensor_equal(angle_loss, target, epsilon=0.1)
                        pass# for _
                    pass# for target
                pass# for dim
            
            pass#/ test
        
        return 
    
    ____test____random_dummy_mat()
    pass

if "test      full_test_version_of_angle_correction__by_row" and False:
    def ____test____full_test_version_of_angle_correction__by_row______basic():
        
        if "basic behavior" and False:
            for dim in [10,23,89,147,379,1000]:
                for _ in range(11):
                    input = torch.randn(size=[dim,dim])
                    result = full_test_version_of_angle_correction__by_row(input,1.)
                    assert _tensor_equal(get_vector_length(result), torch.ones(size=[dim]))
                    pass
                pass
            
            pass
        
        if "is it the same as before?" and False:
            
            for dim in [10,22,88,100,177,222,1000]:
                for _ in range(11):
                    iota_of_dim = iota(dim)
                    cap_to = random.random()*0.3+0.2 # 0.2 to 0.5
                    expansion_factor = random.random()*2.# 0 to 2
                    #<  init for both
                    ori_mat = torch.randn(size=[dim,dim])
                    ori_mat = vector_length_norm(ori_mat)
                    
                    #<  old method>
                    
                    # #<  measure the init>
                    # before__len_loss, before__angle_loss, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                    # before__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(ori_mat)
                    
                    #<  calc grad>
                    mat = ori_mat.detach().clone()
                    
                    manual__mat_matmul_mat__d_d = mat@(mat.T)
                    manual__mat_matmul_mat__d_d[iota_of_dim, iota_of_dim] = 0.
                    
                    #<  original grad>
                    ori__grad__d_d = manual__mat_matmul_mat__d_d@mat
                    ori__grad_len__dim = get_vector_length(ori__grad__d_d)
                    #<  original grad, but len into 1>
                    _temp_reverse_of__ori__grad_len__dim = ori__grad_len__dim.pow(-1.)
                    len_1__ori_grad__d_d = ori__grad__d_d.mul( \
                            expand_vec_to_matrix(_temp_reverse_of__ori__grad_len__dim, each_element_to="row"))
                    del _temp_reverse_of__ori__grad_len__dim # just in case.
                    
                    #<  scale it a bit, to make the distribution a bit wider>
                    #otherwise, they are a lot 0.8,0.9. Wider means most are 0.3 to 0.8.
                    #<old code/>xx__grad_len_sqr__dim = ori__grad__d_d.mul(ori__grad__d_d).sum(dim=1)#mul and then sum, it's a dot.
                    max_of__ori__grad_len__dim = ori__grad_len__dim.max()
                    ratio_of__grad_len__dim = ori__grad_len__dim/max_of__ori__grad_len__dim
                    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this scaled_to_1__grad_len_sqr__dim is still one 1.0 and a lot 0.xx
                    #assert ratio_of__grad_len__dim.le(1.).all()
                    #assert ratio_of__grad_len__dim.eq(1.).sum() == 1
                    
                    grad_len__after_expansion__dim = ratio_of__grad_len__dim.pow(expansion_factor)
                    # assert grad_len__after_expansion__dim.le(1.).all() when expansion_factor is neg, this is wrong.
                    # assert grad_len__after_expansion__dim.eq(1.).sum() == 1 when expansion_factor is 0., this is wrong.
                    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this grad_len__after_expansion__dim is still one 1.0 and a lot 0.xx
                    
                    #<  target length>
                    grad_length_target__dim = grad_len__after_expansion__dim*cap_to# cap_to is scalar
                    grad_length_target__dim__expand_dim = expand_vec_to_matrix(grad_length_target__dim, each_element_to="row")
                    
                    #<  finally 
                    useful_grad__d_d = len_1__ori_grad__d_d.mul(grad_length_target__dim__expand_dim)#row
                    
                    #<  update mat
                    mat -= useful_grad__d_d
                    
                    #<  protect the length after updating
                    mat = vector_length_norm(mat)
                    
                    #<  measure the protected.>
                    # after__len_loss, after__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat)
                    # after__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
                    
                    #</ old method>
                    
                    #<  new method>
                    
                    mat_from_new_method = full_test_version_of_angle_correction__by_row(ori_mat, expansion_factor=expansion_factor,
                                                                                        cap_to= cap_to, iota_of_dim=iota_of_dim)
                    
                    assert _tensor_equal(mat, mat_from_new_method)
                    pass#for _ 
                pass#for dim
            
            pass#/test
        
        if "does it work for columns-wise?" and False:
            for dim in [10,22,88,100,177,222,1000]:
                for _ in range(11):
                    iota_of_dim = iota(dim)
                    cap_to = random.random()*0.3+0.2 # 0.2 to 0.5
                    expansion_factor = random.random()*2.# 0 to 2
                    #<  init for both
                    ori_mat = torch.randn(size=[dim,dim])
                    ori_mat = vector_length_norm(ori_mat.T).T
                    assert _tensor_equal(get_vector_length(ori_mat.T), torch.ones(size=[dim]))
                    
                    #<  calc
                    mat = full_test_version_of_angle_correction__by_row(ori_mat.T, expansion_factor=expansion_factor,
                                                                                        cap_to= cap_to, iota_of_dim=iota_of_dim).T
                    
                    #<  assertions>
                    assert _tensor_equal(get_vector_length(mat.T), torch.ones(size=[dim]))
                    
                    #so the other direction(row in this test) is not standardized. 
                    # Most of the len of vectors are not 1. Not even close.
                    row_vec_len = get_vector_length(mat)#.sort().values  when you need to read, sort it.
                    non_len_1_count = row_vec_len.lt(0.999).sum()+row_vec_len.gt(1.001).sum()
                    assert non_len_1_count>dim*0.8
                    
                    pass#for _ 
                pass#for dim
            
            pass#/test
        
        return 
    
    #____test____full_test_version_of_angle_correction__by_row______basic()


    def ____test____full_test_version_of_angle_correction__by_row______scan_the_process():
        
        if "style doesn't matter......a rough test. To build up intuition." and False:
            #result
            # the measurement is manual. The fluctuating tail is ignored manually.
            # format
            #   hyper_param / result(incr by default)
            #               /steps_until_fluctuating         min_of_step    
            # expansion     /               first_step
            #        cap_to /                     max_of_accumulate
            # 0.0     0.05  /    23 steps     0.118     1.518   0.006
            # 0.0     0.10  /     9 steps     0.225     1.442   0.026
            # 0.0     0.15  /     6 steps     0.326     1.28    0.018
            # 0.0     0.20  /     5 steps     0.417     1.19    0.035
            # 0.0     0.30  /     5 steps     0.541     1.10    0.011
            # 0.0     0.40  /     3 steps     0.569     0.94    0.040    
            
            # 1.0     0.10  /    14 steps     0.178     1.48    0.01
            # 1.0     0.20  /     9 steps     0.337     1.35
            
            # conclusion.
            # The score_incr for each step doesn't show too much. 
            # sometimes it's negative.
            # Every test ends with a very clear 2-step fluctuation.
            style_list = ["r", "rr", "rc", "rrr", "rrc" ]
            for style in style_list:
                expansion_factor = 0.
                cap_to = 0.3
                steps = 11
                
                dim = 100
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                
                test_time = 100
                print(test_time)
                _raw_result__total__score_incr = torch.empty(size=[test_time, steps])
                _raw_result__step__score_incr  = torch.empty(size=[test_time, steps])
                
                for _test_count in range(test_time):
                    
                    #<  init
                    mat = torch.randn(size=[dim,dim])#, device=device)
                    _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(mat)
                    angle_loss__last_step = angle_loss__in_the_beginning.detach().clone()
                    
                    #<  calc
                    for _step_count in range(steps):
                        #----------------#----------------#----------------
                        #only use one of them.
                        if style == "r":
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to, iota_of_dim=iota_of_dim)
                            pass
                        elif style == "rr":
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to/2., iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to/2., iota_of_dim=iota_of_dim)
                            pass
                        elif style == "rc":
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to/2., iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_angle_correction__by_row(mat.T,
                                    expansion_factor=expansion_factor, cap_to=cap_to/2., iota_of_dim=iota_of_dim).T
                            pass
                        elif style == "rrr":
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim)
                            pass
                        elif style == "rrc":
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_angle_correction__by_row(mat.T,
                                    expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim).T
                            pass
                        else:
                            assert False, "bad param: style"
                        #----------------#----------------#----------------
                        
                        #<  measure
                        _, angle_loss__of_this_step, _ = LOSS__mat_is_standard_orthogonal(mat)
                        
                        #<  save result.
                        _raw_result__total__score_incr[_test_count, _step_count] = \
                                angle_loss__in_the_beginning - angle_loss__of_this_step
                        _raw_result__step__score_incr [_test_count, _step_count] = \
                                angle_loss__last_step        - angle_loss__of_this_step
                        
                        #tail
                        angle_loss__last_step = angle_loss__of_this_step
                        
                        pass#for _step_count
                    
                    pass#for _test_count
                    
                total__score_incr = _raw_result__total__score_incr.mean(dim=0)
                step__score_incr  = _raw_result__step__score_incr .mean(dim=0)
                # print()
                # print()
                # print(f"# {expansion_factor:.1f}     {cap_to:.2f}  /      steps     {step__score_incr[0].item():.3f}     ")
                # print()
                # print()
                assert total__score_incr.shape.__len__() == 1
                assert total__score_incr.shape[0] == steps
                x_axis = torch.linspace(1, steps, steps)
                
                from matplotlib import pyplot as plt
                plt.plot(x_axis, total__score_incr)#, x_axis, step__score_incr)
                plt.title(f"{style}  expansion_factor {expansion_factor:.2f}   cap_to {cap_to:.2f}")
                plt.show()
                
                pass#for style
            
            pass#/ test
        
        if "scan it a bit. all row-wise" and False:
            #result
            
            if False:
                
                # expansion_factor 0.0      dim 100      threshold_factor 0.01
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5650,  1.5158,  1.4277,  1.3429,  1.2584,  1.1763,  1.1024]
                # total__step__avg              = [ 49.96,    20.41,    11.20,    7.97,    6.31,    5.32,    5.79]
                # total__score_incr_per_step__avg = [ 0.0314,  0.0745,  0.1287,  0.1727,  0.2049,  0.2263,  0.1962]
                # ------------                                                                         
                # expansion_factor 0.0      dim 100      threshold_factor 0.02
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5647,  1.5154,  1.4239,  1.3430,  1.2588,  1.1704,  1.1008]
                # total__step__avg              = [ 49.88,    20.36,    11.01,    7.83,    6.17,    5.04,    5.48]
                # total__score_incr_per_step__avg = [ 0.0314,  0.0746,  0.1307,  0.1745,  0.2083,  0.2358,  0.2049]
                # ------------                                                                         
                # expansion_factor 0.0      dim 100      threshold_factor 0.05
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5687,  1.5189,  1.4166,  1.3168,  1.2364,  1.1558,  1.0800]
                # total__step__avg              = [ 49.49,    20.62,    10.65,    6.97,    5.55,    4.72,    4.70]
                # total__score_incr_per_step__avg = [ 0.0317,  0.0738,  0.1341,  0.1922,  0.2290,  0.2523,  0.2333]
                # ------------                                                                         
                # expansion_factor 0.0      dim 100      threshold_factor 0.1
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5577,  1.5126,  1.3987,  1.3008,  1.1840,  1.0835,  1.0372]
                # total__step__avg              = [ 49.19,    20.28,    10.05,    6.43,    4.50,    3.40,    3.62]
                # total__score_incr_per_step__avg = [ 0.0317,  0.0748,  0.1401,  0.2044,  0.2687,  0.3288,  0.2914]
                # ------------                                                                         
                # expansion_factor 0.0      dim 100      threshold_factor 0.15
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5586,  1.5050,  1.3864,  1.2809,  1.1660,  1.0554,  1.0020]
                # total__step__avg              = [ 48.85,    19.95,    9.64,    6.08,    4.05,    3.00,    3.06]
                # total__score_incr_per_step__avg = [ 0.0319,  0.0756,  0.1445,  0.2111,  0.2884,  0.3518,  0.3288]
                # ------------                                                                         
                # expansion_factor 0.0      dim 100      threshold_factor 0.2
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5559,  1.4886,  1.3746,  1.2788,  1.1564,  1.0588,  0.9651]
                # total__step__avg              = [ 48.54,    19.56,    9.30,    5.99,    4.00,    3.00,    2.69]
                # total__score_incr_per_step__avg = [ 0.0321,  0.0763,  0.1481,  0.2136,  0.2891,  0.3529,  0.3673]
                # ------------                                                                         
                # expansion_factor 0.0      dim 100      threshold_factor 0.3
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5411,  1.4621,  1.3547,  1.2488,  1.1599,  1.0533,  0.8827]
                # total__step__avg              = [ 48.18,    18.73,    8.97,    5.66,    4.00,    3.00,    2.00]
                # total__score_incr_per_step__avg = [ 0.0320,  0.0781,  0.1511,  0.2215,  0.2900,  0.3511,  0.4414]
                # ------------                                                                         
                # expansion_factor 0.0      dim 100      threshold_factor 0.5
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.4241,  1.3850,  1.2487,  1.1445,  0.9814,  0.8539,  0.8786]
                # total__step__avg              = [ 43.05,    17.23,    7.86,    4.84,    3.00,    2.09,    2.00]
                # total__score_incr_per_step__avg = [ 0.0331,  0.0804,  0.1590,  0.2369,  0.3271,  0.4108,  0.4393]
                # ------------                                                    
                
                
                # expansion_factor 1.0      dim 100      threshold_factor 0.02
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5834,  1.5755,  1.5120,  1.4165,  1.3427,  1.2857,  1.2409]
                # total__step__avg              = [ 65.62,    27.02,    13.84,    9.21,    7.40,    6.64,    6.42]
                # total__score_incr_per_step__avg = [ 0.0242,  0.0585,  0.1097,  0.1558,  0.1868,  0.2011,  0.2004]
                # ------------                                                                         
                # expansion_factor 1.0      dim 100      threshold_factor 0.05
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5753,  1.5631,  1.5049,  1.3979,  1.3249,  1.2569,  1.2188]
                # total__step__avg              = [ 65.34,    26.44,    13.45,    8.66,    6.65,    5.77,    5.73]
                # total__score_incr_per_step__avg = [ 0.0242,  0.0593,  0.1124,  0.1626,  0.2027,  0.2269,  0.2176]
                # ------------                                                                         
                # expansion_factor 1.0      dim 100      threshold_factor 0.1
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5694,  1.5694,  1.5008,  1.3923,  1.3007,  1.1991,  1.1648]
                # total__step__avg              = [ 64.64,    26.32,    13.26,    8.41,    6.11,    4.58,    4.50]
                # total__score_incr_per_step__avg = [ 0.0243,  0.0598,  0.1136,  0.1666,  0.2144,  0.2681,  0.2710]
                # ------------                                                                         
                # expansion_factor 1.0      dim 100      threshold_factor 0.15
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5551,  1.5468,  1.5001,  1.3894,  1.2902,  1.1780,  1.0882]
                # total__step__avg              = [ 63.05,    25.87,    13.11,    8.27,    5.83,    4.14,    3.45]
                # total__score_incr_per_step__avg = [ 0.0247,  0.0600,  0.1147,  0.1686,  0.2224,  0.2863,  0.3246]
                # ------------                                                                         
                # expansion_factor 1.0      dim 100      threshold_factor 0.2
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5477,  1.5349,  1.4869,  1.3805,  1.2711,  1.1707,  1.0628]
                # total__step__avg              = [ 62.82,    25.45,    12.86,    8.03,    5.49,    4.02,    3.12]
                # total__score_incr_per_step__avg = [ 0.0247,  0.0604,  0.1159,  0.1723,  0.2328,  0.2914,  0.3431]
                # ------------                                                                         
                # expansion_factor 1.0      dim 100      threshold_factor 0.3
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.4799,  1.4977,  1.4426,  1.3511,  1.2418,  1.1679,  1.0565]
                # total__step__avg              = [ 58.13,    24.29,    12.04,    7.57,    5.16,    4.00,    3.00]
                # total__score_incr_per_step__avg = [ 0.0255,  0.0618,  0.1202,  0.1790,  0.2413,  0.2920,  0.3522]
                # ------------                                                                         
                # expansion_factor 1.0      dim 100      threshold_factor 0.5
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.2613,  1.3385,  1.2705,  1.2140,  1.1129,  1.0044,  0.9406]
                # total__step__avg              = [ 46.06,    20.39,    9.73,    6.26,    4.28,    3.05,    2.50]
                # total__score_incr_per_step__avg = [ 0.0275,  0.0658,  0.1309,  0.1945,  0.2610,  0.3299,  0.3824]
                # ------------                            
                
                
                # expansion_factor 2.0      dim 100      threshold_factor 0.02
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5770,  1.5657,  1.5491,  1.5357,  1.5331,  1.5219,  1.5010]
                # total__step__avg              = [ 78.15,    32.66,    16.89,    13.41,    11.85,    11.24,    11.13]
                # total__score_incr_per_step__avg = [ 0.0203,  0.0483,  0.0927,  0.1180,  0.1329,  0.1396,  0.1413]
                # ------------                                                                         
                # expansion_factor 2.0      dim 100      threshold_factor 0.05
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5691,  1.5543,  1.5440,  1.5202,  1.5101,  1.5097,  1.4730]
                # total__step__avg              = [ 77.30,    31.66,    16.68,    12.27,    10.73,    10.06,    8.76]
                # total__score_incr_per_step__avg = [ 0.0204,  0.0493,  0.0931,  0.1265,  0.1436,  0.1531,  0.1716]
                # ------------                                                                         
                # expansion_factor 2.0      dim 100      threshold_factor 0.1
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5624,  1.5435,  1.5331,  1.4976,  1.4893,  1.4551,  1.4254]
                # total__step__avg              = [ 75.97,    30.81,    16.28,    11.21,    9.49,    8.03,    7.16]
                # total__score_incr_per_step__avg = [ 0.0206,  0.0503,  0.0947,  0.1352,  0.1594,  0.1833,  0.2016]
                # ------------                                                                         
                # expansion_factor 2.0      dim 100      threshold_factor 0.15
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5504,  1.5213,  1.5043,  1.4821,  1.4544,  1.3998,  1.3780]
                # total__step__avg              = [ 74.44,    29.64,    15.37,    10.65,    8.62,    6.99,    6.29]
                # total__score_incr_per_step__avg = [ 0.0209,  0.0515,  0.0983,  0.1401,  0.1711,  0.2039,  0.2221]
                # ------------                                                                         
                # expansion_factor 2.0      dim 100      threshold_factor 0.2
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.5151,  1.5083,  1.4995,  1.4613,  1.4082,  1.3523,  1.2994]
                # total__step__avg              = [ 71.69,    29.47,    15.06,    10.23,    7.67,    6.25,    5.29]
                # total__score_incr_per_step__avg = [ 0.0212,  0.0513,  0.1000,  0.1438,  0.1852,  0.2202,  0.2511]
                # ------------                                                                         
                # expansion_factor 2.0      dim 100      threshold_factor 0.3
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.4672,  1.4687,  1.4497,  1.4204,  1.3329,  1.2598,  1.1907]
                # total__step__avg              = [ 65.75,    27.45,    14.13,    9.44,    6.60,    5.09,    4.21]
                # total__score_incr_per_step__avg = [ 0.0224,  0.0537,  0.1032,  0.1511,  0.2027,  0.2480,  0.2849]
                # ------------                                                                         
                # expansion_factor 2.0      dim 100      threshold_factor 0.5
                # cap_to_list                    = [ 0.02,    0.05,    0.10,    0.15,    0.20,    0.25,    0.30]
                # total__score_incr__avg      = [ 1.2410,  1.3066,  1.3038,  1.2678,  1.1943,  1.1255,  1.0306]
                # total__step__avg              = [ 51.74,    22.57,    11.70,    7.65,    5.43,    4.10,    3.19]
                # total__score_incr_per_step__avg = [ 0.0241,  0.0580,  0.1119,  0.1665,  0.2207,  0.2755,  0.3251]
                # ------------   
                
                pass
            
            # conclusion:
            # threshold_factor can be 0.0001 to 0.2, and 0.1 works well. It actually can be any number.
            # I'll use 0.1 for a while and see what happens.
            # and from the data, it looks possible to use a big cap_to to reduce the calling time, which saves the calc.
            
            
            dim = 100
            iota_of_dim = iota(dim)
            #<  device
            if dim>100:
                device = 'cuda'
                pass
            else:
                device = 'cpu'
                pass
            #</ device
            
            test_time = 100
            print(test_time)
            
            
            #-------------------#-------------------#-------------------
            #threshold_factor_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
            threshold_factor_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
            for threshold_factor in threshold_factor_list:
            #-------------------#-------------------#-------------------
                
                total__score_incr__min          = []#dont modify this.
                total__score_incr__max          = []#dont modify this.
                total__score_incr__avg          = []#dont modify this.
                total__step__min                = []#dont modify this.
                total__step__max                = []#dont modify this.
                total__step__avg                = []#dont modify this.
                total__score_incr_per_step__min = []#dont modify this.
                total__score_incr_per_step__max = []#dont modify this.
                total__score_incr_per_step__avg = []#dont modify this.
                
                #-------------------#-------------------#-------------------
                max_steps = 100
                expansion_factor = 1.
                cap_to_list = [0.02,0.05,0.1,0.15,0.2,0.25,0.3]
                for param_set_count in range(cap_to_list.__len__()):
                    cap_to = cap_to_list[param_set_count]
                #-------------------#-------------------#-------------------
                    
                    _raw_result__total__score_incr = torch.empty(size=[test_time])
                    _raw_result__total__step = torch.empty(size=[test_time])
                    _raw_result__total__score_incr_per_step = torch.empty(size=[test_time])
                    
                    for _test_count in range(test_time):
                        
                        #-------------------#-------------------#-------------------
                        #<  init
                        mat = torch.randn(size=[dim,dim])#, device=device)
                        _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(mat)
                        
                        mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to, iota_of_dim=iota_of_dim)
                        _, angle_loss__of_step_1, _ = LOSS__mat_is_standard_orthogonal(mat)
                        init_incr_speed = angle_loss__in_the_beginning - angle_loss__of_step_1
                        assert init_incr_speed>0.
                        
                        angle_loss__last_step = angle_loss__of_step_1.detach().clone()
                        del angle_loss__of_step_1
                        
                        #<  calc
                        for _step_count_minus_2 in range(max_steps):
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    expansion_factor=expansion_factor, cap_to=cap_to, iota_of_dim=iota_of_dim)
                            
                            #<  measure
                            _, angle_loss__of_this_step, _ = LOSS__mat_is_standard_orthogonal(mat)
                            score_incr__of_this_step = angle_loss__last_step - angle_loss__of_this_step
                            
                            #<  break contidion.
                            if score_incr__of_this_step< init_incr_speed*threshold_factor:
                                _total_steps = _step_count_minus_2+1.
                                _total_incr = angle_loss__in_the_beginning - angle_loss__last_step
                                
                                _raw_result__total__score_incr[_test_count] = _total_incr
                                _raw_result__total__step[_test_count] = _total_steps
                                _raw_result__total__score_incr_per_step[_test_count] = _total_incr/_total_steps
                                break
                                pass
                            
                            #tail
                            angle_loss__last_step = angle_loss__of_this_step
                            
                            assert _step_count_minus_2 < max_steps -2#other wise the test doesn't stop in given steps.
                            
                            pass#for _step_count
                        #-------------------#-------------------#-------------------
                        
                        pass#for _test_count
                    
                    # total__score_incr__min         .append(_raw_result__total__score_incr         .min ())
                    # total__score_incr__max         .append(_raw_result__total__score_incr         .max ())
                    total__score_incr__avg         .append(_raw_result__total__score_incr         .mean())
                    # total__step__min               .append(_raw_result__total__step               .min ())
                    # total__step__max               .append(_raw_result__total__step               .max ())
                    total__step__avg               .append(_raw_result__total__step               .mean())
                    # total__score_incr_per_step__min.append(_raw_result__total__score_incr_per_step.min ())
                    # total__score_incr_per_step__max.append(_raw_result__total__score_incr_per_step.max ())
                    total__score_incr_per_step__avg.append(_raw_result__total__score_incr_per_step.mean())
                    
                    pass#for param_set_count 
                
                print(f"# expansion_factor {expansion_factor}      dim {dim}      threshold_factor {threshold_factor}")
                print(f"# cap_to_list                    = {str_the_list(cap_to_list         , 2, segment=",   ")}")
                #print(f"# total__score_incr__min         = {str_the_list(total__score_incr__min         , 4)}")
                #print(f"# total__score_incr__max         = {str_the_list(total__score_incr__max         , 4)}")
                print(f"# total__score_incr__avg      = {str_the_list(total__score_incr__avg         , 4)}")
                #print(f"# total__step__min               = {str_the_list(total__step__min               , 0, segment=",      ")}")
                #print(f"# total__step__max               = {str_the_list(total__step__max               , 0, segment=",      ")}")
                print(f"# total__step__avg              = {str_the_list(total__step__avg               , 2, segment=",   ")}")
                #print(f"# total__score_incr_per_step__min= {str_the_list(total__score_incr_per_step__min, 4)}")
                #print(f"# total__score_incr_per_step__max= {str_the_list(total__score_incr_per_step__max, 4)}")
                print(f"# total__score_incr_per_step__avg = {str_the_list(total__score_incr_per_step__avg, 4)}")
                print(f"# ------------                                                                         ")
                pass# for threshold_factor
            
            pass#/ test
        
        #好像用不到了。
        #暂时没跑。我在第一个里面试过了，style完全不影响。调用函数的次数会有影响。
        if "different -wise comparison" and False:
            #---------------#---------------#---------------
            dim_list =       [ 10, 100,1000]
            test_time_list = [200, 100, 30]
            dim_list =        [100]
            test_time_list = [ 100]
            for outter_iter_count in range(dim_list.__len__()):
                dim = dim_list[outter_iter_count]
                test_time = test_time_list[outter_iter_count]
                print(test_time)
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
            #---------------#---------------#---------------
                
                #---------------#---------------#---------------
                expansion_factor_list = [0., 1., 2.]
                for expansion_factor in expansion_factor_list:
                #---------------#---------------#---------------
                    
                    
                    assert False
                    #_raw_result__??? = torch.empty(size=[test_time])
                    
                    
                    #---------------#---------------#---------------
                    cap_to_list = [0.1, 0.2, 0.3, 0.4]
                    for cap_to in cap_to_list:
                    #---------------#---------------#---------------
                
                        #---------------#---------------#---------------
                        
                        #<  init
                        ori_mat = torch.randn(size=[dim,dim])
                        half__cap_to = cap_to/2.
                        
                        #<  row
                        #<  col
                        #<  r r
                        #<  r c
                        #<  c r
                        #<  r r
                        
                        
                        
                            # mat = full_test_version_of_angle_correction__by_row(mat,
                            #             expansion_factor=expansion_factor, cap_to=cap_to, iota_of_dim=iota_of_dim)
                            # mat = full_test_version_of_angle_correction__by_row(mat.T,
                            #             expansion_factor=expansion_factor, cap_to=cap_to, iota_of_dim=iota_of_dim).T# .T in and .T out, this is col-wise
        
        if "a rough scan about multiple steps." and False:
            #result
            
            #          expansion_factor 1.0      dim 10
            # cap_to_list = [ 0.70,    0.80,    0.90,    1.00,    1.10,    1.20]
            # rc____score = [ 0.6693,  0.6600,  0.6277,  0.5540,  0.5341,  0.4566]
            # rcr___score = [ 0.7431,  0.7828,  0.8147,  0.7881,  0.7845,  0.7093]
            # rcrc__score = [ 0.7630,  0.8183,  0.8744,  0.8747,  0.9063,  0.8484]
            # rCr___score = [ 0.7336,  0.7786,  0.8337,  0.8358,  0.8883,  0.8402]
            # ------------                                                                         
            #          expansion_factor 1.0      dim 100
            # cap_to_list = [ 0.70,    0.80,    0.90,    1.00,    1.10,    1.20]
            # rc____score = [ 0.6782,  0.6346,  0.5804,  0.5312,  0.4771,  0.4017]
            # rcr___score = [ 0.8084,  0.8298,  0.8024,  0.7543,  0.7178,  0.7019]
            # rcrc__score = [ 0.8441,  0.9021,  0.9246,  0.8945,  0.8540,  0.8422]
            # rCr___score = [ 0.7834,  0.8159,  0.8424,  0.8559,  0.8441,  0.7860]
            # ------------                                                                         
            #          expansion_factor 1.0      dim 1000
            # cap_to_list = [ 0.70,    0.80,    0.90,    1.00,    1.10,    1.20]
            # rc____score = [ 0.6146,  0.5482,  0.4905,  0.4204,  0.3292,  0.2062]
            # rcr___score = [ 0.8068,  0.7698,  0.7080,  0.6703,  0.6474,  0.6226]
            # rcrc__score = [ 0.8788,  0.8918,  0.8472,  0.8132,  0.8059,  0.8034]
            # rCr___score = [ 0.7921,  0.8210,  0.8385,  0.8183,  0.7413,  0.6088]
            # ------------    
            
            # conclusion
            # I need to scan by calling-the-function-2-times or 3 times.
            
            
            #-------------------#-------------------#-------------------
            expansion_factor = 1. # as a const
            
            dim_list =       [ 10,100,1000]
            test_time_list = [500,500,100]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                print(f"dim {dim}   test_time {test_time}  {device}")
            #-------------------#-------------------#-------------------
                
                rc____score = []#dont modify this.
                rcr___score = []#dont modify this.
                rcrc__score = []#dont modify this.
                rCr___score = []#dont modify this.
                #bad_1_score = []#dont modify this.
                #bad_2_score = []#dont modify this.
                
                #-------------------#-------------------#-------------------
                #cap_to_list = [0.02,0.05,0.1,0.15,0.2,0.25,0.3]#x axis
                #cap_to_list = [0.1,0.3, 0.5, 0.7, 1., 1.2, 1.5]#x axis
                cap_to_list = [0.7, 0.8, 0.9, 1., 1.1, 1.2,]#x axis
                for cap_to in cap_to_list:
                #-------------------#-------------------#-------------------
                    
                    _raw_result__rc____score = torch.empty(size=[test_time])
                    _raw_result__rcr___score = torch.empty(size=[test_time])
                    _raw_result__rcrc__score = torch.empty(size=[test_time])
                    _raw_result__rCr___score = torch.empty(size=[test_time])
                    #_raw_result__bad_1_score = torch.empty(size=[test_time])
                    #_raw_result__bad_2_score = torch.empty(size=[test_time])
                    
                    for _test_count in range(test_time):
                        
                        #-------------------#-------------------#-------------------
                        #<  init
                        ori_mat = random_dummy_mat(dim, device=device, iota_of_dim=iota_of_dim)#or maybe the noise_0.5
                        _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                        #</ init
                        
                        
                        # rc
                        mat = ori_mat.detach().clone()
                        mat = full_test_version_of_angle_correction__by_row(mat,
                                expansion_factor=expansion_factor, cap_to=cap_to/2., iota_of_dim=iota_of_dim)
                        mat = full_test_version_of_angle_correction__by_row(mat.T,
                                expansion_factor=expansion_factor, cap_to=cap_to/2., iota_of_dim=iota_of_dim).T
                        _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                        _raw_result__rc____score[_test_count] = angle_loss__in_the_beginning-angle_loss__after
                        
                        # rcr
                        mat = ori_mat.detach().clone()
                        mat = full_test_version_of_angle_correction__by_row(mat,
                                expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim)
                        mat = full_test_version_of_angle_correction__by_row(mat.T,
                                expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim).T
                        mat = full_test_version_of_angle_correction__by_row(mat,
                                expansion_factor=expansion_factor, cap_to=cap_to/3., iota_of_dim=iota_of_dim)
                        _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                        _raw_result__rcr___score[_test_count] = angle_loss__in_the_beginning-angle_loss__after
                        
                        # rcrc
                        mat = ori_mat.detach().clone()
                        mat = full_test_version_of_angle_correction__by_row(mat,
                                expansion_factor=expansion_factor, cap_to=cap_to/4., iota_of_dim=iota_of_dim)
                        mat = full_test_version_of_angle_correction__by_row(mat.T,
                                expansion_factor=expansion_factor, cap_to=cap_to/4., iota_of_dim=iota_of_dim).T
                        mat = full_test_version_of_angle_correction__by_row(mat,
                                expansion_factor=expansion_factor, cap_to=cap_to/4., iota_of_dim=iota_of_dim)
                        mat = full_test_version_of_angle_correction__by_row(mat.T,
                                expansion_factor=expansion_factor, cap_to=cap_to/4., iota_of_dim=iota_of_dim).T
                        _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                        _raw_result__rcrc__score[_test_count] = angle_loss__in_the_beginning-angle_loss__after
                        
                        # rCr
                        mat = ori_mat.detach().clone()
                        mat = full_test_version_of_angle_correction__by_row(mat,
                                expansion_factor=expansion_factor, cap_to=cap_to/4., iota_of_dim=iota_of_dim)
                        mat = full_test_version_of_angle_correction__by_row(mat.T,
                                expansion_factor=expansion_factor, cap_to=cap_to/2., iota_of_dim=iota_of_dim).T
                        mat = full_test_version_of_angle_correction__by_row(mat,
                                expansion_factor=expansion_factor, cap_to=cap_to/4., iota_of_dim=iota_of_dim)
                        _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                        _raw_result__rCr___score[_test_count] = angle_loss__in_the_beginning-angle_loss__after
                        
                        # # bad 1
                        # mat = ori_mat.detach().clone()
                        # mat = full_test_version_of_angle_correction__by_row(mat,
                        #         expansion_factor=expansion_factor, cap_to=cap_to*0.9, iota_of_dim=iota_of_dim)
                        # mat = full_test_version_of_angle_correction__by_row(mat.T,
                        #         expansion_factor=expansion_factor, cap_to=cap_to*0.1, iota_of_dim=iota_of_dim).T
                        # _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                        # _raw_result__bad_1_score[_test_count] = angle_loss__in_the_beginning-angle_loss__after
                        
                        # #bad 2":
                        # mat = ori_mat.detach().clone()
                        # mat = full_test_version_of_angle_correction__by_row(mat,
                        #         expansion_factor=expansion_factor, cap_to=cap_to*0.1, iota_of_dim=iota_of_dim)
                        # mat = full_test_version_of_angle_correction__by_row(mat.T,
                        #         expansion_factor=expansion_factor, cap_to=cap_to*0.9, iota_of_dim=iota_of_dim).T
                        # _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                        # _raw_result__bad_2_score[_test_count] = angle_loss__in_the_beginning-angle_loss__after
                        
                        
                        pass#for _test_count
                    
                    rc____score.append(_raw_result__rc____score.mean())
                    rcr___score.append(_raw_result__rcr___score.mean())
                    rcrc__score.append(_raw_result__rcrc__score.mean())
                    rCr___score.append(_raw_result__rCr___score.mean())
                    #bad_1_score.append(_raw_result__bad_1_score.mean())
                    #bad_2_score.append(_raw_result__bad_2_score.mean())
                    
                    pass#for cap_to  #x axis
                
                print(f"#          expansion_factor {expansion_factor}      dim {dim}")
                print(f"# cap_to_list = {str_the_list(cap_to_list         , 2, segment=",   ")}")
                # print(f"# angle_score_incr__min = {str_the_list(angle_score_incr__min, 4)}")
                # print(f"# angle_score_incr__max = {str_the_list(angle_score_incr__max, 4)}")
                print(f"# rc____score = {str_the_list(rc____score, 4)}")
                print(f"# rcr___score = {str_the_list(rcr___score, 4)}")
                print(f"# rcrc__score = {str_the_list(rcrc__score, 4)}")
                print(f"# rCr___score = {str_the_list(rCr___score, 4)}")
                #print(f"# bad_1_score = {str_the_list(bad_1_score, 4)}")
                #print(f"# bad_2_score = {str_the_list(bad_2_score, 4)}")
                print(f"# ------------                                                                         ")
                
                pass# for style
            
            pass#/ test
        
        if "rc, 2 steps." and False:
            #result   
            
            if "original result" and False:
                #          dim 10              0.53, 0.30
                # step_1 =    [ 0.20,    0.23,    0.26,    0.29,    0.32,    0.35,    0.38,    0.42,    0.45,    0.48,    0.51,    0.54,    0.57,    0.60]
                # vvvv cap_to__step_2 
                # 0.10     = [ 0.3801,  0.4042,  0.4390,  0.4646,  0.4914,  0.5086,  0.5375,  0.5423,  0.5572,  0.5613,  0.5533,  0.5538,  0.5233,  0.5025]
                # 0.13     = [ 0.4158,  0.4382,  0.4724,  0.4973,  0.5229,  0.5400,  0.5688,  0.5733,  0.5910,  0.5959,  0.5900,  0.5924,  0.5628,  0.5428]
                # 0.16     = [ 0.4506,  0.4715,  0.5048,  0.5290,  0.5532,  0.5701,  0.5984,  0.6025,  0.6226,  0.6278,  0.6242,  0.6287,  0.6002,  0.5813]
                # 0.19     = [ 0.4838,  0.5032,  0.5356,  0.5591,  0.5818,  0.5985,  0.6254,  0.6290,  0.6508,  0.6561,  0.6546,  0.6611,  0.6345,  0.6167]
                # 0.22     = [ 0.5146,  0.5324,  0.5643,  0.5867,  0.6078,  0.6240,  0.6489,  0.6516,  0.6743,  0.6792,  0.6799,  0.6886,  0.6642,  0.6480]
                # 0.26     = [ 0.5422,  0.5587,  0.5895,  0.6107,  0.6297,  0.6452,  0.6678,  0.6692,  0.6918,  0.6958,  0.6981,  0.7097,  0.6877,  0.6739]
                # 0.29     = [ 0.5653,  0.5804,  0.6101,  0.6298,  0.6466,  0.6612,  0.6809,  0.6805,  0.7025,  0.7046,  0.7082,  0.7227,  0.7035,  0.6931]
                # 0.32     = [ 0.5822,  0.5966,  0.6254,  0.6427,  0.6568,  0.6704,  0.6869,  0.6850,  0.7050,  0.7049,  0.7097,  0.7265,  0.7104,  0.7040]
                # 0.35     = [ 0.5923,  0.6063,  0.6339,  0.6485,  0.6597,  0.6724,  0.6855,  0.6818,  0.6993,  0.6966,  0.7024,  0.7212,  0.7079,  0.7059]
                # 0.38     = [ 0.5948,  0.6086,  0.6343,  0.6465,  0.6548,  0.6661,  0.6765,  0.6708,  0.6854,  0.6803,  0.6865,  0.7068,  0.6961,  0.6984]
                # 0.41     = [ 0.5895,  0.6028,  0.6264,  0.6366,  0.6420,  0.6517,  0.6600,  0.6524,  0.6640,  0.6561,  0.6624,  0.6839,  0.6753,  0.6819]
                # 0.44     = [ 0.5760,  0.5886,  0.6103,  0.6185,  0.6215,  0.6293,  0.6358,  0.6269,  0.6353,  0.6244,  0.6305,  0.6533,  0.6467,  0.6570]
                # 0.48     = [ 0.5546,  0.5665,  0.5862,  0.5921,  0.5934,  0.5994,  0.6046,  0.5946,  0.6001,  0.5862,  0.5923,  0.6155,  0.6112,  0.6249]
                # 0.51     = [ 0.5256,  0.5371,  0.5542,  0.5583,  0.5581,  0.5629,  0.5676,  0.5565,  0.5596,  0.5428,  0.5488,  0.5723,  0.5703,  0.5869]
                # 0.54     = [ 0.4900,  0.5008,  0.5156,  0.5179,  0.5164,  0.5203,  0.5252,  0.5133,  0.5146,  0.4955,  0.5014,  0.5251,  0.5253,  0.5442]
                # 0.57     = [ 0.4486,  0.4584,  0.4713,  0.4720,  0.4691,  0.4727,  0.4779,  0.4661,  0.4662,  0.4454,  0.4512,  0.4753,  0.4774,  0.4977]
                # 0.60     = [ 0.4025,  0.4117,  0.4225,  0.4216,  0.4171,  0.4210,  0.4267,  0.4158,  0.4154,  0.3933,  0.3994,  0.4238,  0.4278,  0.4491]
                # ------------       
                #          dim 100            0.50, 0.30
                # step_1 =    [ 0.20,    0.23,    0.26,    0.29,    0.32,    0.35,    0.38,    0.42,    0.45,    0.48,    0.51,    0.54,    0.57,    0.60]
                # vvvv cap_to__step_2 
                # 0.10     = [ 0.4176,  0.4505,  0.4816,  0.5087,  0.5304,  0.5484,  0.5609,  0.5665,  0.5651,  0.5550,  0.5362,  0.5119,  0.4840,  0.4471]
                # 0.13     = [ 0.4566,  0.4883,  0.5184,  0.5446,  0.5656,  0.5836,  0.5968,  0.6036,  0.6041,  0.5956,  0.5783,  0.5549,  0.5274,  0.4902]
                # 0.16     = [ 0.4941,  0.5247,  0.5537,  0.5787,  0.5988,  0.6164,  0.6297,  0.6374,  0.6396,  0.6332,  0.6176,  0.5956,  0.5687,  0.5315]
                # 0.19     = [ 0.5292,  0.5587,  0.5864,  0.6101,  0.6288,  0.6452,  0.6582,  0.6664,  0.6703,  0.6662,  0.6528,  0.6326,  0.6068,  0.5700]
                # 0.22     = [ 0.5606,  0.5890,  0.6153,  0.6373,  0.6541,  0.6688,  0.6808,  0.6889,  0.6945,  0.6930,  0.6824,  0.6645,  0.6403,  0.6045]
                # 0.26     = [ 0.5868,  0.6139,  0.6385,  0.6587,  0.6730,  0.6852,  0.6958,  0.7033,  0.7104,  0.7119,  0.7045,  0.6897,  0.6676,  0.6335]
                # 0.29     = [ 0.6063,  0.6318,  0.6544,  0.6724,  0.6838,  0.6930,  0.7016,  0.7080,  0.7165,  0.7213,  0.7176,  0.7065,  0.6873,  0.6555]
                # 0.32     = [ 0.6172,  0.6409,  0.6611,  0.6768,  0.6850,  0.6908,  0.6972,  0.7020,  0.7118,  0.7200,  0.7202,  0.7135,  0.6978,  0.6691]
                # 0.35     = [ 0.6182,  0.6395,  0.6571,  0.6704,  0.6754,  0.6778,  0.6818,  0.6850,  0.6958,  0.7074,  0.7116,  0.7096,  0.6980,  0.6731]
                # 0.38     = [ 0.6084,  0.6268,  0.6415,  0.6524,  0.6544,  0.6537,  0.6557,  0.6574,  0.6690,  0.6838,  0.6920,  0.6948,  0.6876,  0.6669]
                # 0.41     = [ 0.5873,  0.6024,  0.6140,  0.6229,  0.6223,  0.6191,  0.6197,  0.6200,  0.6325,  0.6503,  0.6622,  0.6695,  0.6668,  0.6505]
                # 0.44     = [ 0.5553,  0.5667,  0.5753,  0.5825,  0.5800,  0.5752,  0.5749,  0.5744,  0.5878,  0.6084,  0.6235,  0.6351,  0.6366,  0.6246]
                # 0.48     = [ 0.5134,  0.5209,  0.5265,  0.5324,  0.5288,  0.5233,  0.5230,  0.5222,  0.5366,  0.5597,  0.5776,  0.5931,  0.5984,  0.5903]
                # 0.51     = [ 0.4630,  0.4664,  0.4690,  0.4741,  0.4701,  0.4649,  0.4654,  0.4650,  0.4804,  0.5060,  0.5264,  0.5451,  0.5537,  0.5491]
                # 0.54     = [ 0.4058,  0.4048,  0.4045,  0.4092,  0.4056,  0.4015,  0.4035,  0.4041,  0.4209,  0.4487,  0.4713,  0.4927,  0.5043,  0.5027]
                # 0.57     = [ 0.3433,  0.3379,  0.3347,  0.3392,  0.3366,  0.3346,  0.3389,  0.3410,  0.3593,  0.3893,  0.4138,  0.4377,  0.4516,  0.4526]
                # 0.60     = [ 0.2772,  0.2671,  0.2611,  0.2657,  0.2647,  0.2655,  0.2728,  0.2768,  0.2970,  0.3291,  0.3553,  0.3811,  0.3972,  0.4002]
                # ------------                                                                         
                #          dim 1000                       0.44, 0.27
                # step_1 =    [ 0.20,    0.23,    0.26,    0.29,    0.32,    0.35,    0.38,    0.42,    0.45,    0.48,    0.51,    0.54,    0.57,    0.60]
                # vvvv cap_to__step_2 
                # 0.10     = [ 0.4503,  0.4839,  0.5131,  0.5367,  0.5546,  0.5655,  0.5679,  0.5608,  0.5423,  0.5167,  0.4801,  0.4395,  0.3927,  0.3492]
                # 0.13     = [ 0.4914,  0.5237,  0.5518,  0.5747,  0.5925,  0.6045,  0.6090,  0.6046,  0.5889,  0.5652,  0.5296,  0.4888,  0.4412,  0.3965]
                # 0.16     = [ 0.5301,  0.5610,  0.5876,  0.6090,  0.6260,  0.6382,  0.6445,  0.6430,  0.6306,  0.6096,  0.5756,  0.5353,  0.4871,  0.4414]
                # 0.19     = [ 0.5647,  0.5940,  0.6185,  0.6377,  0.6527,  0.6644,  0.6721,  0.6737,  0.6653,  0.6477,  0.6163,  0.5772,  0.5290,  0.4827]
                # 0.22     = [ 0.5932,  0.6205,  0.6423,  0.6585,  0.6705,  0.6806,  0.6891,  0.6939,  0.6902,  0.6772,  0.6494,  0.6123,  0.5650,  0.5186]
                # 0.26     = [ 0.6130,  0.6380,  0.6564,  0.6687,  0.6769,  0.6847,  0.6935,  0.7014,  0.7028,  0.6952,  0.6724,  0.6385,  0.5930,  0.5473]
                # 0.29     = [ 0.6217,  0.6438,  0.6583,  0.6663,  0.6702,  0.6750,  0.6836,  0.6944,  0.7012,  0.6998,  0.6829,  0.6533,  0.6109,  0.5667]
                # 0.32     = [ 0.6171,  0.6359,  0.6462,  0.6499,  0.6494,  0.6513,  0.6595,  0.6728,  0.6847,  0.6897,  0.6793,  0.6551,  0.6170,  0.5751]
                # 0.35     = [ 0.5980,  0.6131,  0.6192,  0.6192,  0.6150,  0.6143,  0.6220,  0.6375,  0.6540,  0.6650,  0.6614,  0.6429,  0.6100,  0.5712]
                # 0.38     = [ 0.5642,  0.5756,  0.5779,  0.5750,  0.5681,  0.5657,  0.5731,  0.5905,  0.6109,  0.6273,  0.6300,  0.6174,  0.5900,  0.5548]
                # 0.41     = [ 0.5169,  0.5244,  0.5235,  0.5191,  0.5108,  0.5076,  0.5152,  0.5343,  0.5579,  0.5788,  0.5870,  0.5798,  0.5579,  0.5263]
                # 0.44     = [ 0.4578,  0.4615,  0.4581,  0.4533,  0.4451,  0.4423,  0.4504,  0.4712,  0.4975,  0.5222,  0.5350,  0.5325,  0.5155,  0.4873]
                # 0.48     = [ 0.3891,  0.3891,  0.3839,  0.3799,  0.3731,  0.3717,  0.3811,  0.4035,  0.4322,  0.4599,  0.4764,  0.4777,  0.4649,  0.4397]
                # 0.51     = [ 0.3130,  0.3094,  0.3030,  0.3008,  0.2967,  0.2979,  0.3090,  0.3331,  0.3639,  0.3941,  0.4136,  0.4179,  0.4084,  0.3855]
                # 0.54     = [ 0.2318,  0.2245,  0.2174,  0.2180,  0.2178,  0.2224,  0.2359,  0.2618,  0.2945,  0.3268,  0.3485,  0.3551,  0.3482,  0.3270]
                # 0.57     = [ 0.1475,  0.1363,  0.1290,  0.1332,  0.1378,  0.1468,  0.1632,  0.1911,  0.2255,  0.2595,  0.2831,  0.2914,  0.2862,  0.2661]
                # 0.60     = [ 0.0620,  0.0468,  0.0394,  0.0478,  0.0582,  0.0723,  0.0921,  0.1221,  0.1582,  0.1936,  0.2186,  0.2282,  0.2243,  0.2048]
                # ------------   
                pass
            
            # conclusion
            # dim 10    0.53, 0.30       0.7265
            # dim 100   0.50, 0.30       0.7213
            # dim 1000  0.44, 0.27       0.7028
            
            #-------------------#-------------------#-------------------
            dim_list =       [ 10,100,1000]
            test_time_list = [500,500,100]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                print(f"dim {dim}   test_time {test_time}  {device}")
            #-------------------#-------------------#-------------------
                y_axis__dim = 17
                
                score:list[list[float]] = []    #dont modify this.
                for _ in range(y_axis__dim):    #dont modify this.
                    score.append([])            #dont modify this.
                    pass                        #dont modify this.
                
                #-------------------#-------------------#-------------------
                #cap_to_list = [0.02,0.05,0.1,0.15,0.2,0.25,0.3]#x axis
                #cap_to_list = [0.1,0.3, 0.5, 0.7, 1., 1.2, 1.5]#x axis
                cap_to__step_1_list = torch.linspace(0.2, 0.6, 14) # x axis
                for cap_to__step_1 in cap_to__step_1_list:
                #-------------------#-------------------#-------------------
                    
                    cap_to__step_2_list = torch.linspace(0.1, 0.6, y_axis__dim) # y axis
                    _raw_result__score = torch.empty(size=[y_axis__dim, test_time])
                    
                    for _test_count in range(test_time):
                        
                        #-------------------#-------------------#-------------------
                        #<  init
                        ori_mat = random_dummy_mat(dim, device=device, iota_of_dim=iota_of_dim)#or maybe the noise_0.5
                        _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                        #</ init
                        
                        #<  calc
                        for _y_axis_count in range(y_axis__dim):
                            mat = ori_mat.detach().clone()
                            mat = full_test_version_of_angle_correction__by_row(mat,
                                    cap_to=cap_to__step_1,                 iota_of_dim=iota_of_dim)
                            mat = full_test_version_of_angle_correction__by_row(mat.T,
                                    cap_to=cap_to__step_2_list[_y_axis_count],  iota_of_dim=iota_of_dim).T
                            _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                            _raw_result__score[_y_axis_count, _test_count] = angle_loss__in_the_beginning-angle_loss__after
                            pass# for _y_axis_count
                        
                        pass#for _test_count
                    
                    for ii in range(y_axis__dim):
                        score[ii].append(_raw_result__score[ii].mean())
                        pass
                    
                    pass#   for cap_to__step_1  #x axis
                
                print(f"#          dim {dim}")
                
                print(f"# step_1 = {str_the_list(cap_to__step_1_list         , 2, segment=",   ")}")
                
                print(f"# vvvv cap_to__step_2 ")
                for ii in range(y_axis__dim):
                    print(f"# {cap_to__step_2_list[ii].item():.2f}     = {str_the_list(score[ii], 4)}")
                    pass
                print(f"# ------------                                                                         ")
                
                pass# for outter_param_count
            
            pass#/ test
        
        #the problem is, if I scan the param combination for n-dim, it's a search in n-dim space.
        # My idea is to only scan the last 3 dim. The earlier dims should be basically the same as lower dim case.
        
        # 2 step result:
        # dim 100     0.50, 0.30         max_score 0.7213
        # dim 1000    0.44, 0.27         max_score 0.7028
        # 3 step result:
        # dim 100     0.51  0.45  0.25   max_score 0.9159
        # dim 1000    0.54  0.42  0.22   max_score 0.8915  
        
        # if I need 4 dim param, I probably use the step_1 of 3 step result and scan the rest 3 dim.
        
        if "rcr, 3 steps." and False:
            #result   
            if "original result" and False:
            
                # dim 100      cap_to__step_3 0.20       max_score 0.9015
                # step_1   = [ 0.25,    0.29,    0.34,    0.38,    0.43,    0.47,    0.51,    0.56,    0.60,    0.65,    0.69,   
                # vvvv cap_to__step_2                                                                                            
                # 0.10     = [ 0.6888,  0.7216,  0.7473,  0.7645,  0.7736,  0.7713,  0.7685,  0.7426,  0.7015,  0.6388,  0.5746, 
                # 0.17     = [ 0.7534,  0.7832,  0.8086,  0.8230,  0.8317,  0.8270,  0.8333,  0.8169,  0.7813,  0.7167,  0.6484, 
                # 0.24     = [ 0.8032,  0.8294,  0.8487,  0.8593,  0.8687,  0.8636,  0.8750,  0.8691,  0.8421,  0.7786,  0.7085, 
                # 0.31     = [ 0.8260,  0.8466,  0.8620,  0.8692,  0.8783,  0.8738,  0.8888,  0.8908,  0.8732,  0.8142,  0.7480, 
                # 0.38     = [ 0.8346,  0.8562,  0.8737,  0.8804,  0.8874,  0.8830,  0.8994,  0.9015,  0.8864,  0.8311,  0.7675, 
                # 0.45     = [ 0.8421,  0.8582,  0.8674,  0.8696,  0.8692,  0.8693,  0.8905,  0.9008,  0.8873,  0.8348,  0.7732, 
                # 0.52     = [ 0.7991,  0.8001,  0.7970,  0.7947,  0.7878,  0.7929,  0.8173,  0.8478,  0.8391,  0.7944,  0.7449, 
                # 0.59     = [ 0.6781,  0.6676,  0.6602,  0.6609,  0.6541,  0.6648,  0.6867,  0.7376,  0.7335,  0.6972,  0.6643, 
                # 0.66     = [ 0.5002,  0.4875,  0.4853,  0.4985,  0.4977,  0.5137,  0.5327,  0.5981,  0.5955,  0.5640,  0.5455, 
                # 0.73     = [ 0.2999,  0.2913,  0.3014,  0.3330,  0.3433,  0.3659,  0.3842,  0.4574,  0.4553,  0.4255,  0.4157, 
                # 0.80     = [ 0.1054,  0.1039,  0.1315,  0.1837,  0.2083,  0.2376,  0.2591,  0.3345,  0.3355,  0.3058,  0.3000, 
                # ------------                                                                                                   
                # dim 100      cap_to__step_3 0.21       max_score 0.9031                                                        
                # step_1   = [ 0.25,    0.29,    0.34,    0.38,    0.43,    0.47,    0.51,    0.56,    0.60,    0.65,    0.69,   
                # vvvv cap_to__step_2                                                                                            
                # 0.10     = [ 0.6991,  0.7347,  0.7542,  0.7757,  0.7831,  0.7867,  0.7813,  0.7574,  0.7199,  0.6425,  0.5882, 
                # 0.17     = [ 0.7642,  0.7944,  0.8122,  0.8333,  0.8387,  0.8422,  0.8402,  0.8246,  0.7937,  0.7166,  0.6598, 
                # 0.24     = [ 0.8148,  0.8380,  0.8487,  0.8662,  0.8716,  0.8765,  0.8772,  0.8707,  0.8501,  0.7766,  0.7195, 
                # 0.31     = [ 0.8357,  0.8508,  0.8559,  0.8708,  0.8758,  0.8828,  0.8861,  0.8860,  0.8780,  0.8137,  0.7586, 
                # 0.38     = [ 0.8421,  0.8580,  0.8669,  0.8848,  0.8886,  0.8947,  0.8981,  0.8954,  0.8895,  0.8299,  0.7766, 
                # 0.45     = [ 0.8546,  0.8692,  0.8729,  0.8775,  0.8786,  0.8935,  0.8956,  0.9031,  0.8985,  0.8389,  0.7857, 
                # 0.52     = [ 0.8274,  0.8291,  0.8196,  0.8002,  0.8030,  0.8292,  0.8286,  0.8612,  0.8655,  0.8158,  0.7657, 
                # 0.59     = [ 0.7215,  0.7086,  0.6899,  0.6605,  0.6702,  0.7032,  0.7018,  0.7566,  0.7710,  0.7358,  0.6926, 
                # 0.66     = [ 0.5538,  0.5329,  0.5154,  0.4922,  0.5122,  0.5496,  0.5499,  0.6160,  0.6368,  0.6137,  0.5765, 
                # 0.73     = [ 0.3587,  0.3361,  0.3305,  0.3241,  0.3562,  0.3966,  0.4028,  0.4721,  0.4954,  0.4798,  0.4457, 
                # 0.80     = [ 0.1664,  0.1459,  0.1590,  0.1763,  0.2218,  0.2638,  0.2788,  0.3469,  0.3717,  0.3604,  0.3266, 
                # ------------                                                                                                   
                # dim 100      cap_to__step_3 0.23       max_score 0.9081                                                        
                # step_1   = [ 0.25,    0.29,    0.34,    0.38,    0.43,    0.47,    0.51,    0.56,    0.60,    0.65,    0.69,   
                # vvvv cap_to__step_2                                                                                            
                # 0.10     = [ 0.7120,  0.7404,  0.7661,  0.7804,  0.7890,  0.7918,  0.7865,  0.7707,  0.7239,  0.6704,  0.6324, 
                # 0.17     = [ 0.7741,  0.7988,  0.8219,  0.8359,  0.8437,  0.8426,  0.8377,  0.8345,  0.7936,  0.7430,  0.7024, 
                # 0.24     = [ 0.8214,  0.8394,  0.8554,  0.8651,  0.8742,  0.8742,  0.8704,  0.8756,  0.8462,  0.7997,  0.7577, 
                # 0.31     = [ 0.8357,  0.8456,  0.8580,  0.8649,  0.8746,  0.8778,  0.8776,  0.8880,  0.8693,  0.8292,  0.7891, 
                # 0.38     = [ 0.8386,  0.8507,  0.8680,  0.8802,  0.8909,  0.8915,  0.8888,  0.8972,  0.8784,  0.8402,  0.8014, 
                # 0.45     = [ 0.8567,  0.8693,  0.8842,  0.8862,  0.8876,  0.8957,  0.9032,  0.9081,  0.8920,  0.8526,  0.8116, 
                # 0.52     = [ 0.8365,  0.8371,  0.8469,  0.8250,  0.8134,  0.8398,  0.8619,  0.8702,  0.8690,  0.8356,  0.7943, 
                # 0.59     = [ 0.7335,  0.7186,  0.7338,  0.6930,  0.6762,  0.7218,  0.7539,  0.7679,  0.7832,  0.7574,  0.7216, 
                # 0.66     = [ 0.5623,  0.5394,  0.5694,  0.5251,  0.5118,  0.5727,  0.6106,  0.6296,  0.6536,  0.6325,  0.6042, 
                # 0.73     = [ 0.3612,  0.3386,  0.3883,  0.3553,  0.3520,  0.4219,  0.4638,  0.4877,  0.5133,  0.4942,  0.4715, 
                # 0.80     = [ 0.1633,  0.1446,  0.2156,  0.2050,  0.2164,  0.2899,  0.3352,  0.3636,  0.3881,  0.3706,  0.3528, 
                # ------------                                                                                                   
                # dim 100      cap_to__step_3 0.25       max_score 0.9159                                                        
                # step_1   = [ 0.25,    0.29,    0.34,    0.38,    0.43,    0.47,    0.51,    0.56,    0.60,    0.65,    0.69,   
                # vvvv cap_to__step_2                                                                                            
                # 0.10     = [ 0.7218,  0.7446,  0.7666,  0.7828,  0.7846,  0.7939,  0.8010,  0.7803,  0.7385,  0.6831,  0.6291, 
                # 0.17     = [ 0.7791,  0.7999,  0.8181,  0.8361,  0.8338,  0.8394,  0.8511,  0.8386,  0.8052,  0.7524,  0.6982, 
                # 0.24     = [ 0.8208,  0.8372,  0.8457,  0.8615,  0.8605,  0.8676,  0.8809,  0.8752,  0.8527,  0.8073,  0.7543, 
                # 0.31     = [ 0.8257,  0.8387,  0.8422,  0.8570,  0.8545,  0.8656,  0.8833,  0.8817,  0.8699,  0.8343,  0.7864, 
                # 0.38     = [ 0.8232,  0.8421,  0.8532,  0.8726,  0.8726,  0.8815,  0.8937,  0.8891,  0.8759,  0.8425,  0.7975, 
                # 0.45     = [ 0.8498,  0.8665,  0.8762,  0.8901,  0.8881,  0.9009,  0.9159,  0.9099,  0.8949,  0.8570,  0.8107, 
                # 0.52     = [ 0.8454,  0.8500,  0.8479,  0.8466,  0.8356,  0.8553,  0.8843,  0.8887,  0.8841,  0.8479,  0.8077, 
                # 0.59     = [ 0.7531,  0.7474,  0.7377,  0.7277,  0.7117,  0.7376,  0.7817,  0.7954,  0.8077,  0.7771,  0.7510, 
                # 0.66     = [ 0.5830,  0.5770,  0.5716,  0.5673,  0.5536,  0.5850,  0.6391,  0.6570,  0.6819,  0.6561,  0.6422, 
                # 0.73     = [ 0.3767,  0.3780,  0.3871,  0.3973,  0.3942,  0.4314,  0.4901,  0.5107,  0.5420,  0.5194,  0.5118, 
                # 0.80     = [ 0.1731,  0.1829,  0.2128,  0.2424,  0.2557,  0.2981,  0.3582,  0.3823,  0.4168,  0.3967,  0.3907, 
                # ------------                                                                                                   
                # dim 100      cap_to__step_3 0.26       max_score 0.9025                                                        
                # step_1   = [ 0.25,    0.29,    0.34,    0.38,    0.43,    0.47,    0.51,    0.56,    0.60,    0.65,    0.69,   
                # vvvv cap_to__step_2                                                                                            
                # 0.10     = [ 0.7277,  0.7480,  0.7758,  0.7853,  0.7840,  0.7955,  0.7918,  0.7746,  0.7514,  0.7065,  0.6370, 
                # 0.17     = [ 0.7821,  0.7993,  0.8242,  0.8356,  0.8286,  0.8381,  0.8318,  0.8298,  0.8139,  0.7719,  0.7011, 
                # 0.24     = [ 0.8214,  0.8298,  0.8469,  0.8565,  0.8515,  0.8635,  0.8550,  0.8623,  0.8557,  0.8188,  0.7515, 
                # 0.31     = [ 0.8213,  0.8226,  0.8373,  0.8494,  0.8430,  0.8553,  0.8536,  0.8651,  0.8630,  0.8356,  0.7774, 
                # 0.38     = [ 0.8160,  0.8254,  0.8498,  0.8694,  0.8664,  0.8738,  0.8673,  0.8726,  0.8689,  0.8408,  0.7847, 
                # 0.45     = [ 0.8455,  0.8565,  0.8797,  0.8895,  0.8898,  0.9025,  0.8966,  0.8983,  0.8941,  0.8603,  0.7986, 
                # 0.52     = [ 0.8521,  0.8498,  0.8597,  0.8435,  0.8419,  0.8664,  0.8740,  0.8876,  0.8863,  0.8565,  0.8011, 
                # 0.59     = [ 0.7705,  0.7542,  0.7522,  0.7208,  0.7173,  0.7523,  0.7743,  0.8053,  0.8037,  0.7879,  0.7497, 
                # 0.66     = [ 0.6053,  0.5882,  0.5845,  0.5559,  0.5561,  0.5979,  0.6304,  0.6730,  0.6701,  0.6670,  0.6436, 
                # 0.73     = [ 0.4001,  0.3916,  0.3976,  0.3839,  0.3942,  0.4425,  0.4807,  0.5290,  0.5274,  0.5300,  0.5137, 
                # 0.80     = [ 0.1962,  0.1982,  0.2225,  0.2296,  0.2547,  0.3103,  0.3498,  0.4015,  0.4047,  0.4080,  0.3929, 
                # ------------                                                                                                   
                # dim 100      cap_to__step_3 0.28       max_score 0.8966                                                        
                # step_1   = [ 0.25,    0.29,    0.34,    0.38,    0.43,    0.47,    0.51,    0.56,    0.60,    0.65,    0.69,   
                # vvvv cap_to__step_2                                                                                            
                # 0.10     = [ 0.7292,  0.7547,  0.7708,  0.7774,  0.7831,  0.7908,  0.7922,  0.7842,  0.7450,  0.6968,  0.6402, 
                # 0.17     = [ 0.7812,  0.7987,  0.8168,  0.8249,  0.8227,  0.8272,  0.8265,  0.8304,  0.8024,  0.7578,  0.7004, 
                # 0.24     = [ 0.8174,  0.8252,  0.8400,  0.8438,  0.8421,  0.8497,  0.8454,  0.8527,  0.8396,  0.8037,  0.7475, 
                # 0.31     = [ 0.8132,  0.8163,  0.8272,  0.8280,  0.8301,  0.8437,  0.8400,  0.8491,  0.8480,  0.8219,  0.7701, 
                # 0.38     = [ 0.8050,  0.8154,  0.8322,  0.8468,  0.8496,  0.8588,  0.8528,  0.8551,  0.8514,  0.8241,  0.7722, 
                # 0.45     = [ 0.8389,  0.8508,  0.8683,  0.8832,  0.8863,  0.8938,  0.8927,  0.8902,  0.8773,  0.8430,  0.7886, 
                # 0.52     = [ 0.8581,  0.8618,  0.8674,  0.8612,  0.8613,  0.8833,  0.8857,  0.8966,  0.8812,  0.8548,  0.8003, 
                # 0.59     = [ 0.7892,  0.7831,  0.7784,  0.7523,  0.7532,  0.7934,  0.7969,  0.8249,  0.8159,  0.8090,  0.7622, 
                # 0.66     = [ 0.6315,  0.6231,  0.6207,  0.5908,  0.5986,  0.6505,  0.6555,  0.6949,  0.6952,  0.7008,  0.6646, 
                # 0.73     = [ 0.4288,  0.4262,  0.4378,  0.4187,  0.4370,  0.4959,  0.5050,  0.5492,  0.5570,  0.5649,  0.5379, 
                # 0.80     = [ 0.2226,  0.2303,  0.2627,  0.2627,  0.2940,  0.3577,  0.3731,  0.4190,  0.4322,  0.4369,  0.4178, 
                # ------------                                                                                                   
                # dim 100      cap_to__step_3 0.30       max_score 0.8907                                                        
                # step_1   = [ 0.25,    0.29,    0.34,    0.38,    0.43,    0.47,    0.51,    0.56,    0.60,    0.65,    0.69,   
                # vvvv cap_to__step_2                                                                                            
                # 0.10     = [ 0.7341,  0.7543,  0.7652,  0.7719,  0.7784,  0.7674,  0.7888,  0.7848,  0.7589,  0.7118,  0.6610, 
                # 0.17     = [ 0.7809,  0.7967,  0.8074,  0.8151,  0.8182,  0.7963,  0.8129,  0.8204,  0.8112,  0.7694,  0.7200, 
                # 0.24     = [ 0.8097,  0.8207,  0.8255,  0.8304,  0.8340,  0.8187,  0.8223,  0.8331,  0.8423,  0.8092,  0.7645, 
                # 0.31     = [ 0.7962,  0.8061,  0.8085,  0.8152,  0.8175,  0.8087,  0.8176,  0.8231,  0.8430,  0.8192,  0.7816, 
                # 0.38     = [ 0.7865,  0.7990,  0.8161,  0.8342,  0.8485,  0.8224,  0.8306,  0.8308,  0.8448,  0.8174,  0.7817, 
                # 0.45     = [ 0.8252,  0.8355,  0.8576,  0.8753,  0.8899,  0.8681,  0.8783,  0.8747,  0.8760,  0.8459,  0.7998, 
                # 0.52     = [ 0.8577,  0.8643,  0.8655,  0.8625,  0.8653,  0.8695,  0.8871,  0.8849,  0.8907,  0.8616,  0.8157, 
                # 0.59     = [ 0.8003,  0.8084,  0.7843,  0.7616,  0.7556,  0.7864,  0.8063,  0.8131,  0.8311,  0.8119,  0.7786, 
                # 0.66     = [ 0.6441,  0.6632,  0.6281,  0.6028,  0.5994,  0.6453,  0.6657,  0.6824,  0.7083,  0.6991,  0.6792, 
                # 0.73     = [ 0.4382,  0.4728,  0.4432,  0.4303,  0.4385,  0.4909,  0.5129,  0.5374,  0.5666,  0.5629,  0.5507, 
                # 0.80     = [ 0.2293,  0.2779,  0.2661,  0.2754,  0.2996,  0.3531,  0.3794,  0.4106,  0.4411,  0.4409,  0.4300, 
                # ------------                                                                                                   
                # dim 100      cap_to__step_3 0.31       max_score 0.8946                                                        
                # step_1   = [ 0.25,    0.29,    0.34,    0.38,    0.43,    0.47,    0.51,    0.56,    0.60,    0.65,    0.69,   
                # vvvv cap_to__step_2                                                                                            
                # 0.10     = [ 0.7258,  0.7487,  0.7650,  0.7655,  0.7625,  0.7643,  0.7787,  0.7842,  0.7597,  0.7165,  0.6569, 
                # 0.17     = [ 0.7689,  0.7863,  0.8007,  0.8051,  0.7961,  0.7888,  0.7973,  0.8148,  0.8064,  0.7700,  0.7100, 
                # 0.24     = [ 0.7955,  0.8035,  0.8117,  0.8154,  0.8102,  0.8049,  0.8047,  0.8256,  0.8313,  0.8057,  0.7497, 
                # 0.31     = [ 0.7813,  0.7811,  0.7910,  0.7946,  0.7947,  0.7918,  0.8022,  0.8165,  0.8271,  0.8129,  0.7641, 
                # 0.38     = [ 0.7670,  0.7816,  0.7958,  0.8144,  0.8187,  0.8202,  0.8277,  0.8232,  0.8239,  0.8123,  0.7613, 
                # 0.45     = [ 0.8084,  0.8289,  0.8452,  0.8626,  0.8679,  0.8725,  0.8787,  0.8697,  0.8596,  0.8410,  0.7827, 
                # 0.52     = [ 0.8519,  0.8584,  0.8688,  0.8675,  0.8718,  0.8662,  0.8821,  0.8946,  0.8846,  0.8606,  0.8041, 
                # 0.59     = [ 0.8104,  0.7995,  0.8039,  0.7844,  0.7856,  0.7680,  0.7961,  0.8352,  0.8319,  0.8158,  0.7705, 
                # 0.66     = [ 0.6623,  0.6506,  0.6574,  0.6342,  0.6393,  0.6162,  0.6527,  0.7075,  0.7094,  0.7073,  0.6718, 
                # 0.73     = [ 0.4566,  0.4552,  0.4758,  0.4629,  0.4787,  0.4576,  0.5012,  0.5605,  0.5670,  0.5730,  0.5435, 
                # 0.80     = [ 0.2444,  0.2583,  0.2981,  0.3041,  0.3358,  0.3223,  0.3728,  0.4294,  0.4424,  0.4516,  0.4242, 
                
                
                
                # dim 1000      cap_to__step_3 0.20       max_score 0.8873                                                       
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                        
                # 0.30     = [ 0.8429,  0.8460,  0.8503,  0.8525,  0.8513,  0.8541,  0.8567,  0.8592,  0.8630,  0.8608,  0.8475,  0.8247,  0.7966,  0.7576,  0.7252,  0.6960,  0.6752,  0.6591]
                # 0.33     = [ 0.8541,  0.8573,  0.8613,  0.8634,  0.8613,  0.8634,  0.8647,  0.8656,  0.8677,  0.8649,  0.8517,  0.8295,  0.8020,  0.7634,  0.7318,  0.7038,  0.6840,  0.6692]
                # 0.36     = [ 0.8652,  0.8681,  0.8720,  0.8746,  0.8738,  0.8763,  0.8773,  0.8771,  0.8775,  0.8735,  0.8594,  0.8365,  0.8087,  0.7698,  0.7382,  0.7105,  0.6913,  0.6772]
                # 0.39     = [ 0.8671,  0.8692,  0.8734,  0.8765,  0.8797,  0.8841,  0.8863,  0.8868,  0.8869,  0.8827,  0.8680,  0.8445,  0.8161,  0.7767,  0.7449,  0.7170,  0.6981,  0.6844]
                # 0.42     = [ 0.8525,  0.8534,  0.8580,  0.8616,  0.8705,  0.8779,  0.8828,  0.8857,  0.8873,  0.8847,  0.8709,  0.8480,  0.8194,  0.7801,  0.7484,  0.7209,  0.7024,  0.6894]
                # 0.45     = [ 0.8182,  0.8180,  0.8229,  0.8267,  0.8418,  0.8523,  0.8606,  0.8667,  0.8715,  0.8716,  0.8607,  0.8402,  0.8121,  0.7744,  0.7436,  0.7176,  0.6999,  0.6880]
                # 0.48     = [ 0.7654,  0.7646,  0.7697,  0.7737,  0.7943,  0.8073,  0.8187,  0.8280,  0.8364,  0.8399,  0.8328,  0.8162,  0.7897,  0.7548,  0.7260,  0.7025,  0.6862,  0.6760]
                # 0.51     = [ 0.6977,  0.6971,  0.7022,  0.7067,  0.7315,  0.7463,  0.7601,  0.7721,  0.7837,  0.7904,  0.7869,  0.7749,  0.7507,  0.7194,  0.6933,  0.6731,  0.6584,  0.6502]
                # 0.54     = [ 0.6200,  0.6200,  0.6254,  0.6304,  0.6584,  0.6744,  0.6899,  0.7037,  0.7177,  0.7270,  0.7264,  0.7188,  0.6969,  0.6693,  0.6461,  0.6289,  0.6161,  0.6099]
                # 0.57     = [ 0.5360,  0.5372,  0.5432,  0.5490,  0.5792,  0.5961,  0.6126,  0.6277,  0.6432,  0.6544,  0.6559,  0.6519,  0.6320,  0.6076,  0.5866,  0.5722,  0.5612,  0.5568]
                # 0.60     = [ 0.4492,  0.4523,  0.4592,  0.4657,  0.4977,  0.5152,  0.5322,  0.5480,  0.5646,  0.5770,  0.5798,  0.5785,  0.5601,  0.5380,  0.5190,  0.5065,  0.4970,  0.4939]
                # ------------                                                                           
                # dim 1000      cap_to__step_3 0.21       max_score 0.8908                                                       
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                        
                # 0.30     = [ 0.8365,  0.8394,  0.8436,  0.8441,  0.8437,  0.8467,  0.8481,  0.8529,  0.8566,  0.8563,  0.8484,  0.8234,  0.7925,  0.7630,  0.7310,  0.7008,  0.6786,  0.6686]
                # 0.33     = [ 0.8491,  0.8519,  0.8562,  0.8564,  0.8550,  0.8570,  0.8567,  0.8590,  0.8611,  0.8597,  0.8512,  0.8268,  0.7964,  0.7675,  0.7361,  0.7070,  0.6862,  0.6771]
                # 0.36     = [ 0.8630,  0.8659,  0.8701,  0.8712,  0.8706,  0.8723,  0.8715,  0.8725,  0.8727,  0.8694,  0.8594,  0.8339,  0.8026,  0.7734,  0.7418,  0.7131,  0.6929,  0.6842]
                # 0.39     = [ 0.8693,  0.8720,  0.8760,  0.8786,  0.8810,  0.8839,  0.8847,  0.8864,  0.8855,  0.8812,  0.8704,  0.8436,  0.8112,  0.7813,  0.7491,  0.7202,  0.7002,  0.6916]
                # 0.42     = [ 0.8599,  0.8620,  0.8657,  0.8701,  0.8770,  0.8819,  0.8866,  0.8908,  0.8906,  0.8870,  0.8771,  0.8501,  0.8177,  0.7874,  0.7548,  0.7260,  0.7061,  0.6978]
                # 0.45     = [ 0.8304,  0.8319,  0.8350,  0.8412,  0.8527,  0.8600,  0.8700,  0.8776,  0.8795,  0.8786,  0.8711,  0.8459,  0.8153,  0.7857,  0.7537,  0.7252,  0.7063,  0.6986]
                # 0.48     = [ 0.7812,  0.7824,  0.7849,  0.7925,  0.8083,  0.8173,  0.8326,  0.8437,  0.8482,  0.8511,  0.8470,  0.8254,  0.7985,  0.7709,  0.7404,  0.7129,  0.6955,  0.6893]
                # 0.51     = [ 0.7161,  0.7173,  0.7195,  0.7283,  0.7470,  0.7574,  0.7770,  0.7907,  0.7977,  0.8046,  0.8040,  0.7867,  0.7647,  0.7400,  0.7117,  0.6856,  0.6705,  0.6662]
                # 0.54     = [ 0.6395,  0.6415,  0.6436,  0.6534,  0.6741,  0.6856,  0.7082,  0.7237,  0.7328,  0.7428,  0.7451,  0.7318,  0.7146,  0.6933,  0.6675,  0.6432,  0.6303,  0.6282]
                # 0.57     = [ 0.5558,  0.5593,  0.5617,  0.5723,  0.5945,  0.6065,  0.6314,  0.6478,  0.6585,  0.6705,  0.6753,  0.6651,  0.6521,  0.6335,  0.6101,  0.5876,  0.5766,  0.5764]
                # 0.60     = [ 0.4689,  0.4742,  0.4775,  0.4887,  0.5120,  0.5246,  0.5508,  0.5677,  0.5795,  0.5927,  0.5992,  0.5911,  0.5814,  0.5649,  0.5435,  0.5224,  0.5128,  0.5140]
                # ------------                                                                           
                # dim 1000      cap_to__step_3 0.22       max_score 0.8915                                                       
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                        
                # 0.30     = [ 0.8274,  0.8311,  0.8328,  0.8341,  0.8345,  0.8358,  0.8372,  0.8414,  0.8505,  0.8507,  0.8440,  0.8240,  0.7979,  0.7642,  0.7339,  0.7067,  0.6824,  0.6715]
                # 0.33     = [ 0.8410,  0.8454,  0.8471,  0.8473,  0.8467,  0.8471,  0.8463,  0.8483,  0.8540,  0.8526,  0.8458,  0.8258,  0.8003,  0.7672,  0.7376,  0.7115,  0.6885,  0.6786]
                # 0.36     = [ 0.8580,  0.8622,  0.8645,  0.8653,  0.8650,  0.8649,  0.8632,  0.8639,  0.8660,  0.8621,  0.8539,  0.8328,  0.8067,  0.7726,  0.7430,  0.7167,  0.6943,  0.6848]
                # 0.39     = [ 0.8690,  0.8717,  0.8747,  0.8782,  0.8797,  0.8806,  0.8804,  0.8809,  0.8815,  0.8762,  0.8664,  0.8441,  0.8169,  0.7812,  0.7510,  0.7242,  0.7018,  0.6922]
                # 0.42     = [ 0.8650,  0.8654,  0.8685,  0.8760,  0.8807,  0.8840,  0.8878,  0.8895,  0.8915,  0.8869,  0.8766,  0.8539,  0.8256,  0.7894,  0.7584,  0.7317,  0.7091,  0.6996]
                # 0.45     = [ 0.8405,  0.8382,  0.8410,  0.8527,  0.8611,  0.8678,  0.8769,  0.8810,  0.8867,  0.8855,  0.8760,  0.8544,  0.8257,  0.7908,  0.7595,  0.7343,  0.7118,  0.7027]
                # 0.48     = [ 0.7950,  0.7905,  0.7926,  0.8080,  0.8198,  0.8295,  0.8443,  0.8512,  0.8614,  0.8653,  0.8579,  0.8387,  0.8108,  0.7790,  0.7484,  0.7264,  0.7044,  0.6961]
                # 0.51     = [ 0.7320,  0.7259,  0.7276,  0.7458,  0.7602,  0.7720,  0.7920,  0.8013,  0.8156,  0.8247,  0.8200,  0.8041,  0.7778,  0.7507,  0.7215,  0.7035,  0.6826,  0.6757]
                # 0.54     = [ 0.6563,  0.6498,  0.6513,  0.6716,  0.6878,  0.7011,  0.7248,  0.7360,  0.7535,  0.7669,  0.7647,  0.7520,  0.7279,  0.7056,  0.6783,  0.6648,  0.6449,  0.6397]
                # 0.57     = [ 0.5729,  0.5668,  0.5687,  0.5904,  0.6079,  0.6221,  0.6485,  0.6606,  0.6805,  0.6970,  0.6967,  0.6869,  0.6647,  0.6469,  0.6213,  0.6116,  0.5928,  0.5894]
                # 0.60     = [ 0.4856,  0.4808,  0.4835,  0.5064,  0.5248,  0.5397,  0.5676,  0.5803,  0.6019,  0.6204,  0.6213,  0.6134,  0.5928,  0.5785,  0.5543,  0.5476,  0.5297,  0.5276]
                # ------------                                                                           
                # dim 1000      cap_to__step_3 0.23       max_score 0.8871                                                       
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                        
                # 0.30     = [ 0.8182,  0.8199,  0.8226,  0.8234,  0.8229,  0.8220,  0.8247,  0.8288,  0.8376,  0.8414,  0.8369,  0.8192,  0.7945,  0.7607,  0.7308,  0.7081,  0.6889,  0.6754]
                # 0.33     = [ 0.8323,  0.8350,  0.8374,  0.8383,  0.8356,  0.8336,  0.8336,  0.8353,  0.8415,  0.8431,  0.8373,  0.8201,  0.7956,  0.7623,  0.7331,  0.7112,  0.6934,  0.6810]
                # 0.36     = [ 0.8514,  0.8552,  0.8577,  0.8588,  0.8562,  0.8543,  0.8521,  0.8519,  0.8550,  0.8537,  0.8455,  0.8271,  0.8013,  0.7672,  0.7375,  0.7158,  0.6981,  0.6862]
                # 0.39     = [ 0.8665,  0.8703,  0.8733,  0.8753,  0.8753,  0.8749,  0.8730,  0.8719,  0.8731,  0.8700,  0.8601,  0.8396,  0.8123,  0.7767,  0.7460,  0.7239,  0.7056,  0.6936]
                # 0.42     = [ 0.8680,  0.8707,  0.8745,  0.8776,  0.8821,  0.8852,  0.8858,  0.8857,  0.8869,  0.8842,  0.8736,  0.8517,  0.8237,  0.7871,  0.7556,  0.7330,  0.7142,  0.7021]
                # 0.45     = [ 0.8491,  0.8502,  0.8544,  0.8588,  0.8685,  0.8763,  0.8812,  0.8838,  0.8871,  0.8869,  0.8772,  0.8553,  0.8278,  0.7916,  0.7602,  0.7376,  0.7193,  0.7075]
                # 0.48     = [ 0.8080,  0.8076,  0.8121,  0.8175,  0.8320,  0.8443,  0.8543,  0.8601,  0.8669,  0.8707,  0.8631,  0.8431,  0.8176,  0.7833,  0.7535,  0.7315,  0.7149,  0.7041]
                # 0.51     = [ 0.7479,  0.7466,  0.7515,  0.7574,  0.7755,  0.7915,  0.8059,  0.8145,  0.8248,  0.8333,  0.8284,  0.8112,  0.7891,  0.7578,  0.7311,  0.7100,  0.6964,  0.6870]
                # 0.54     = [ 0.6740,  0.6724,  0.6777,  0.6842,  0.7047,  0.7233,  0.7412,  0.7515,  0.7648,  0.7776,  0.7752,  0.7608,  0.7424,  0.7144,  0.6916,  0.6717,  0.6617,  0.6540]
                # 0.57     = [ 0.5913,  0.5900,  0.5962,  0.6033,  0.6253,  0.6456,  0.6659,  0.6772,  0.6926,  0.7086,  0.7084,  0.6962,  0.6813,  0.6560,  0.6370,  0.6182,  0.6118,  0.6057]
                # 0.60     = [ 0.5042,  0.5039,  0.5112,  0.5191,  0.5420,  0.5633,  0.5852,  0.5970,  0.6138,  0.6318,  0.6332,  0.6227,  0.6104,  0.5872,  0.5710,  0.5532,  0.5500,  0.5452]
                # ------------                                                                           
                # dim 1000      cap_to__step_3 0.24       max_score 0.8873                                                       
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                        
                # 0.30     = [ 0.8028,  0.8067,  0.8091,  0.8089,  0.8087,  0.8064,  0.8105,  0.8167,  0.8242,  0.8304,  0.8285,  0.8134,  0.7868,  0.7566,  0.7303,  0.7048,  0.6902,  0.6796]
                # 0.33     = [ 0.8180,  0.8223,  0.8250,  0.8245,  0.8229,  0.8183,  0.8199,  0.8225,  0.8278,  0.8309,  0.8285,  0.8131,  0.7869,  0.7567,  0.7311,  0.7065,  0.6929,  0.6835]
                # 0.36     = [ 0.8400,  0.8447,  0.8478,  0.8477,  0.8461,  0.8403,  0.8399,  0.8393,  0.8420,  0.8417,  0.8370,  0.8195,  0.7927,  0.7608,  0.7350,  0.7102,  0.6967,  0.6876]
                # 0.39     = [ 0.8597,  0.8641,  0.8677,  0.8691,  0.8685,  0.8644,  0.8635,  0.8622,  0.8626,  0.8603,  0.8528,  0.8328,  0.8047,  0.7708,  0.7440,  0.7184,  0.7043,  0.6949]
                # 0.42     = [ 0.8674,  0.8708,  0.8743,  0.8780,  0.8797,  0.8803,  0.8805,  0.8812,  0.8811,  0.8784,  0.8691,  0.8479,  0.8178,  0.7829,  0.7552,  0.7290,  0.7145,  0.7046]
                # 0.45     = [ 0.8551,  0.8573,  0.8599,  0.8661,  0.8706,  0.8783,  0.8807,  0.8857,  0.8873,  0.8858,  0.8766,  0.8563,  0.8247,  0.7906,  0.7626,  0.7363,  0.7220,  0.7121]
                # 0.48     = [ 0.8197,  0.8211,  0.8221,  0.8306,  0.8378,  0.8530,  0.8579,  0.8686,  0.8732,  0.8742,  0.8668,  0.8499,  0.8175,  0.7865,  0.7591,  0.7338,  0.7204,  0.7113]
                # 0.51     = [ 0.7636,  0.7651,  0.7643,  0.7747,  0.7840,  0.8054,  0.8122,  0.8285,  0.8361,  0.8400,  0.8356,  0.8238,  0.7916,  0.7654,  0.7397,  0.7161,  0.7042,  0.6970]
                # 0.54     = [ 0.6918,  0.6942,  0.6920,  0.7039,  0.7148,  0.7406,  0.7487,  0.7693,  0.7795,  0.7858,  0.7844,  0.7781,  0.7468,  0.7259,  0.7022,  0.6809,  0.6711,  0.6662]
                # 0.57     = [ 0.6102,  0.6138,  0.6110,  0.6241,  0.6361,  0.6645,  0.6735,  0.6973,  0.7091,  0.7172,  0.7182,  0.7165,  0.6864,  0.6702,  0.6486,  0.6296,  0.6217,  0.6196]
                # 0.60     = [ 0.5231,  0.5284,  0.5258,  0.5398,  0.5529,  0.5828,  0.5924,  0.6183,  0.6310,  0.6402,  0.6428,  0.6444,  0.6155,  0.6027,  0.5830,  0.5660,  0.5597,  0.5598]
                # ------------                                                                           
                # dim 1000      cap_to__step_3 0.26       max_score 0.8842                                                       
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                        
                # 0.30     = [ 0.7894,  0.7920,  0.7956,  0.7950,  0.7950,  0.7911,  0.7956,  0.8005,  0.8087,  0.8185,  0.8183,  0.8071,  0.7809,  0.7535,  0.7273,  0.7042,  0.6878,  0.6781]
                # 0.33     = [ 0.8059,  0.8083,  0.8121,  0.8110,  0.8088,  0.8030,  0.8053,  0.8075,  0.8122,  0.8186,  0.8168,  0.8050,  0.7787,  0.7519,  0.7263,  0.7043,  0.6891,  0.6805]
                # 0.36     = [ 0.8302,  0.8325,  0.8366,  0.8364,  0.8337,  0.8270,  0.8268,  0.8270,  0.8269,  0.8299,  0.8248,  0.8118,  0.7826,  0.7555,  0.7290,  0.7067,  0.6917,  0.6833]
                # 0.39     = [ 0.8536,  0.8560,  0.8601,  0.8613,  0.8600,  0.8544,  0.8531,  0.8524,  0.8499,  0.8502,  0.8418,  0.8274,  0.7947,  0.7664,  0.7380,  0.7148,  0.6993,  0.6902]
                # 0.42     = [ 0.8661,  0.8683,  0.8725,  0.8750,  0.8769,  0.8750,  0.8746,  0.8746,  0.8723,  0.8717,  0.8613,  0.8455,  0.8106,  0.7809,  0.7510,  0.7268,  0.7106,  0.7008]
                # 0.45     = [ 0.8590,  0.8610,  0.8652,  0.8687,  0.8744,  0.8786,  0.8809,  0.8829,  0.8839,  0.8842,  0.8739,  0.8576,  0.8230,  0.7917,  0.7618,  0.7371,  0.7205,  0.7107]
                # 0.48     = [ 0.8284,  0.8302,  0.8342,  0.8381,  0.8475,  0.8585,  0.8646,  0.8692,  0.8759,  0.8784,  0.8705,  0.8547,  0.8234,  0.7910,  0.7630,  0.7387,  0.7221,  0.7135]
                # 0.51     = [ 0.7758,  0.7777,  0.7816,  0.7853,  0.7976,  0.8146,  0.8244,  0.8312,  0.8442,  0.8498,  0.8453,  0.8311,  0.8053,  0.7728,  0.7483,  0.7257,  0.7095,  0.7030]
                # 0.54     = [ 0.7063,  0.7088,  0.7128,  0.7160,  0.7304,  0.7519,  0.7647,  0.7729,  0.7913,  0.7997,  0.7989,  0.7865,  0.7670,  0.7352,  0.7151,  0.6949,  0.6795,  0.6756]
                # 0.57     = [ 0.6257,  0.6293,  0.6338,  0.6366,  0.6523,  0.6767,  0.6917,  0.7007,  0.7228,  0.7336,  0.7357,  0.7250,  0.7111,  0.6803,  0.6646,  0.6469,  0.6323,  0.6315]
                # 0.60     = [ 0.5389,  0.5441,  0.5495,  0.5524,  0.5690,  0.5951,  0.6116,  0.6209,  0.6453,  0.6577,  0.6619,  0.6524,  0.6428,  0.6130,  0.6007,  0.5853,  0.5713,  0.5733]
                # ------------                                                            
                pass
            
            # conclusion
            # dim 100     0.51  0.45  0.25   max_score 0.9159
            # dim 1000    0.54  0.42  0.22   max_score 0.8915  
            
            
            
            #-------------------#-------------------#-------------------
            dim_list =       [ 10,100,1000]
            test_time_list = [100,100,100]
            dim_list =       [1000]
            test_time_list = [20]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                print(f"dim {dim}   test_time {test_time}  {device}")
            #-------------------#-------------------#-------------------
                
                #-------------------#-------------------#-------------------
                z_axis__dim = 10
                cap_to__step_3_list = torch.linspace(0.2, 0.3, z_axis__dim) # z axis
                for cap_to__step_3 in cap_to__step_3_list:
                #-------------------#-------------------#-------------------
                
                #-------------------#-------------------#-------------------
                
                    x_axis__dim = 18
                    y_axis__dim = 11
                    
                    score = torch.empty(size=[y_axis__dim, x_axis__dim])    #dont modify this.
                    
                    #-------------------#-------------------#-------------------
                    cap_to__step_1_list = torch.linspace(0.35, 0.7, x_axis__dim) # x axis
                    for _iter__cap_to__step_1 in range(cap_to__step_1_list.shape[0]):
                        cap_to__step_1 = cap_to__step_1_list[_iter__cap_to__step_1]
                    #-------------------#-------------------#-------------------
                        
                        cap_to__step_2_list = torch.linspace(0.3, 0.6, y_axis__dim) # y axis
                        _raw_result__score = torch.empty(size=[y_axis__dim, test_time])
                        
                        for _test_count in range(test_time):
                            
                            #-------------------#-------------------#-------------------
                            #<  init
                            ori_mat = random_dummy_mat(dim, device=device, iota_of_dim=iota_of_dim)#or maybe the noise_0.5
                            _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                            #</ init
                            
                            #<  calc
                            for _y_axis_count in range(y_axis__dim):
                                mat = ori_mat.detach().clone()
                                mat = full_test_version_of_angle_correction__by_row(mat,
                                        cap_to=cap_to__step_1,                 iota_of_dim=iota_of_dim)
                                mat = full_test_version_of_angle_correction__by_row(mat.T,
                                        cap_to=cap_to__step_2_list[_y_axis_count],  iota_of_dim=iota_of_dim).T
                                mat = full_test_version_of_angle_correction__by_row(mat,
                                        cap_to=cap_to__step_3,                 iota_of_dim=iota_of_dim)
                                _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                                _raw_result__score[_y_axis_count, _test_count] = angle_loss__in_the_beginning-angle_loss__after
                                pass# for _y_axis_count
                            
                            pass#for _test_count
                        
                        score[:, _iter__cap_to__step_1] = _raw_result__score.mean(dim=1)
                        
                        pass#   for cap_to__step_1  #x axis
                    
                    print(f"# dim {dim}      cap_to__step_3 {cap_to__step_3.item():.2f}       max_score {score.max().item():.4f}"+" "*55)
                    
                    print(f"# step_1   = {str_the_list(cap_to__step_1_list         , 2, segment=",   ")}")
                    
                    print(f"# vvvv cap_to__step_2 "+" "*55)
                    for ii in range(y_axis__dim):
                        print(f"# {cap_to__step_2_list[ii].item():.2f}     = {str_the_list(score[ii], 4)}")
                        pass
                    print(f"# ------------"+" "*75)
                    
                    pass# for cap_to__step_3
                    
                pass# for outter_param_count
            
            pass#/ test
        
        return 
        
    #____test____full_test_version_of_angle_correction__by_row______scan_the_process()

if "scan for the adaptive angle correct" and True:
    def ____test____scan_for_the_adaptive_angle_correct():
    
        if "test 1, the dummy mat is in between 0.9 to 1.55, after a rc correct, it's 0.41 to 0.64. ." and False:
            #result  
            
            if True:
                
                # this score is a difference.
                # this score is a difference.
                # this score is a difference.
                
                # dim 100      actual_init_angle_score 0.907       max_score 0.4869                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.4090,  0.4109,  0.4052,  0.3978,  0.3810,  0.3615,  0.3379,  0.3010,  0.2655,  0.2219,  0.1763,  0.1279,  0.0860,  0.0327, -0.0243, -0.0885, -0.1436, -0.1997]
                # 0.13     = [ 0.4359,  0.4396,  0.4371,  0.4332,  0.4213,  0.4062,  0.3863,  0.3534,  0.3212,  0.2799,  0.2352,  0.1876,  0.1458,  0.0928,  0.0354, -0.0300, -0.0857, -0.1438]
                # 0.16     = [ 0.4548,  0.4596,  0.4597,  0.4595,  0.4528,  0.4429,  0.4277,  0.4000,  0.3721,  0.3340,  0.2914,  0.2452,  0.2039,  0.1516,  0.0942,  0.0280, -0.0282, -0.0881]
                # 0.19     = [ 0.4639,  0.4691,  0.4713,  0.4747,  0.4733,  0.4692,  0.4598,  0.4387,  0.4162,  0.3826,  0.3432,  0.2992,  0.2590,  0.2081,  0.1511,  0.0846,  0.0282, -0.0332]
                # 0.22     = [ 0.4620,  0.4672,  0.4706,  0.4774,  0.4812,  0.4830,  0.4801,  0.4672,  0.4513,  0.4236,  0.3889,  0.3482,  0.3100,  0.2612,  0.2052,  0.1390,  0.0827,  0.0202]
                # 0.25     = [ 0.4489,  0.4538,  0.4576,  0.4673,  0.4756,  0.4830,  0.4869,  0.4832,  0.4749,  0.4546,  0.4262,  0.3903,  0.3549,  0.3093,  0.2551,  0.1900,  0.1344,  0.0714]
                # 0.28     = [ 0.4252,  0.4296,  0.4335,  0.4454,  0.4574,  0.4695,  0.4797,  0.4855,  0.4852,  0.4734,  0.4531,  0.4233,  0.3920,  0.3508,  0.2994,  0.2364,  0.1820,  0.1192]
                # 0.31     = [ 0.3923,  0.3964,  0.4002,  0.4137,  0.4283,  0.4440,  0.4594,  0.4739,  0.4814,  0.4786,  0.4675,  0.4453,  0.4193,  0.3838,  0.3363,  0.2768,  0.2243,  0.1625]
                # 0.34     = [ 0.3518,  0.3561,  0.3597,  0.3744,  0.3907,  0.4089,  0.4284,  0.4501,  0.4643,  0.4700,  0.4684,  0.4548,  0.4352,  0.4066,  0.3641,  0.3094,  0.2596,  0.2000]
                # 0.37     = [ 0.3056,  0.3105,  0.3141,  0.3299,  0.3471,  0.3669,  0.3894,  0.4165,  0.4358,  0.4488,  0.4563,  0.4512,  0.4387,  0.4178,  0.3813,  0.3328,  0.2866,  0.2304]
                # 0.40     = [ 0.2551,  0.2612,  0.2653,  0.2819,  0.2997,  0.3204,  0.3448,  0.3758,  0.3988,  0.4175,  0.4327,  0.4353,  0.4300,  0.4169,  0.3871,  0.3459,  0.3042,  0.2524]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.997       max_score 0.5409                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.4421,  0.4450,  0.4413,  0.4333,  0.4218,  0.4031,  0.3859,  0.3614,  0.3267,  0.2999,  0.2613,  0.2212,  0.1869,  0.1449,  0.1038,  0.0676,  0.0373, -0.0115]
                # 0.13     = [ 0.4727,  0.4766,  0.4754,  0.4695,  0.4603,  0.4443,  0.4294,  0.4063,  0.3733,  0.3471,  0.3094,  0.2695,  0.2349,  0.1922,  0.1507,  0.1138,  0.0829,  0.0335]
                # 0.16     = [ 0.4978,  0.5025,  0.5034,  0.4997,  0.4932,  0.4804,  0.4682,  0.4471,  0.4166,  0.3913,  0.3549,  0.3155,  0.2809,  0.2377,  0.1960,  0.1587,  0.1272,  0.0772]
                # 0.19     = [ 0.5157,  0.5209,  0.5235,  0.5221,  0.5186,  0.5096,  0.5007,  0.4822,  0.4548,  0.4311,  0.3966,  0.3581,  0.3239,  0.2806,  0.2389,  0.2012,  0.1695,  0.1190]
                # 0.22     = [ 0.5245,  0.5302,  0.5337,  0.5346,  0.5345,  0.5296,  0.5248,  0.5096,  0.4864,  0.4650,  0.4331,  0.3961,  0.3627,  0.3197,  0.2783,  0.2406,  0.2087,  0.1581]
                # 0.25     = [ 0.5229,  0.5287,  0.5326,  0.5359,  0.5393,  0.5388,  0.5387,  0.5274,  0.5093,  0.4911,  0.4626,  0.4278,  0.3959,  0.3539,  0.3132,  0.2758,  0.2440,  0.1935]
                # 0.28     = [ 0.5102,  0.5161,  0.5197,  0.5251,  0.5322,  0.5362,  0.5409,  0.5342,  0.5218,  0.5076,  0.4835,  0.4516,  0.4220,  0.3816,  0.3421,  0.3054,  0.2742,  0.2241]
                # 0.31     = [ 0.4865,  0.4926,  0.4954,  0.5029,  0.5133,  0.5214,  0.5308,  0.5289,  0.5228,  0.5134,  0.4944,  0.4661,  0.4397,  0.4016,  0.3639,  0.3285,  0.2983,  0.2489]
                # 0.34     = [ 0.4527,  0.4591,  0.4610,  0.4704,  0.4837,  0.4954,  0.5092,  0.5120,  0.5121,  0.5078,  0.4942,  0.4703,  0.4478,  0.4129,  0.3774,  0.3437,  0.3151,  0.2668]
                # 0.37     = [ 0.4103,  0.4172,  0.4182,  0.4294,  0.4452,  0.4601,  0.4774,  0.4846,  0.4904,  0.4912,  0.4831,  0.4639,  0.4458,  0.4147,  0.3820,  0.3504,  0.3237,  0.2770]
                # 0.40     = [ 0.3607,  0.3685,  0.3689,  0.3817,  0.3997,  0.4172,  0.4375,  0.4485,  0.4593,  0.4648,  0.4620,  0.4473,  0.4338,  0.4069,  0.3773,  0.3480,  0.3239,  0.2790]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.100       max_score 0.6049                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.4822,  0.4873,  0.4891,  0.4856,  0.4804,  0.4689,  0.4520,  0.4328,  0.4118,  0.3840,  0.3586,  0.3234,  0.2930,  0.2533,  0.2150,  0.1814,  0.1421,  0.1065]
                # 0.13     = [ 0.5135,  0.5196,  0.5228,  0.5208,  0.5177,  0.5082,  0.4931,  0.4759,  0.4564,  0.4294,  0.4047,  0.3698,  0.3393,  0.2990,  0.2600,  0.2250,  0.1855,  0.1489]
                # 0.16     = [ 0.5408,  0.5474,  0.5517,  0.5511,  0.5502,  0.5431,  0.5301,  0.5154,  0.4977,  0.4720,  0.4482,  0.4140,  0.3838,  0.3430,  0.3034,  0.2675,  0.2276,  0.1903]
                # 0.19     = [ 0.5627,  0.5693,  0.5742,  0.5750,  0.5763,  0.5720,  0.5614,  0.5496,  0.5345,  0.5106,  0.4882,  0.4551,  0.4254,  0.3846,  0.3446,  0.3079,  0.2679,  0.2299]
                # 0.22     = [ 0.5775,  0.5835,  0.5885,  0.5907,  0.5943,  0.5930,  0.5853,  0.5769,  0.5650,  0.5436,  0.5230,  0.4916,  0.4630,  0.4226,  0.3825,  0.3455,  0.3054,  0.2670]
                # 0.25     = [ 0.5835,  0.5885,  0.5931,  0.5967,  0.6024,  0.6044,  0.6001,  0.5954,  0.5875,  0.5693,  0.5511,  0.5222,  0.4951,  0.4557,  0.4159,  0.3792,  0.3392,  0.3006]
                # 0.28     = [ 0.5796,  0.5832,  0.5871,  0.5919,  0.5997,  0.6049,  0.6044,  0.6037,  0.6002,  0.5861,  0.5709,  0.5451,  0.5203,  0.4825,  0.4436,  0.4078,  0.3681,  0.3296]
                # 0.31     = [ 0.5651,  0.5672,  0.5703,  0.5761,  0.5856,  0.5941,  0.5973,  0.6006,  0.6019,  0.5925,  0.5808,  0.5588,  0.5369,  0.5018,  0.4642,  0.4301,  0.3910,  0.3531]
                # 0.34     = [ 0.5400,  0.5409,  0.5431,  0.5498,  0.5609,  0.5723,  0.5790,  0.5861,  0.5922,  0.5877,  0.5801,  0.5624,  0.5440,  0.5121,  0.4767,  0.4451,  0.4069,  0.3699]
                # 0.37     = [ 0.5052,  0.5052,  0.5068,  0.5143,  0.5268,  0.5406,  0.5505,  0.5607,  0.5716,  0.5721,  0.5684,  0.5552,  0.5409,  0.5129,  0.4803,  0.4518,  0.4149,  0.3791]
                # 0.40     = [ 0.4617,  0.4614,  0.4626,  0.4711,  0.4848,  0.5007,  0.5134,  0.5261,  0.5414,  0.5464,  0.5467,  0.5379,  0.5276,  0.5037,  0.4745,  0.4498,  0.4144,  0.3801]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.233       max_score 0.6905                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.5324,  0.5381,  0.5442,  0.5449,  0.5433,  0.5376,  0.5289,  0.5180,  0.5025,  0.4837,  0.4591,  0.4401,  0.4170,  0.3884,  0.3648,  0.3390,  0.3222,  0.2960]
                # 0.13     = [ 0.5656,  0.5717,  0.5786,  0.5804,  0.5799,  0.5752,  0.5680,  0.5577,  0.5429,  0.5247,  0.5004,  0.4813,  0.4579,  0.4288,  0.4043,  0.3784,  0.3612,  0.3346]
                # 0.16     = [ 0.5962,  0.6025,  0.6099,  0.6127,  0.6132,  0.6097,  0.6041,  0.5947,  0.5808,  0.5634,  0.5398,  0.5206,  0.4970,  0.4676,  0.4422,  0.4164,  0.3988,  0.3718]
                # 0.19     = [ 0.6230,  0.6293,  0.6368,  0.6405,  0.6420,  0.6396,  0.6359,  0.6278,  0.6150,  0.5988,  0.5760,  0.5571,  0.5334,  0.5039,  0.4779,  0.4522,  0.4342,  0.4071]
                # 0.22     = [ 0.6447,  0.6508,  0.6579,  0.6624,  0.6647,  0.6635,  0.6620,  0.6555,  0.6442,  0.6294,  0.6080,  0.5893,  0.5660,  0.5367,  0.5103,  0.4848,  0.4665,  0.4396]
                # 0.25     = [ 0.6596,  0.6654,  0.6715,  0.6767,  0.6798,  0.6797,  0.6807,  0.6762,  0.6667,  0.6538,  0.6342,  0.6161,  0.5933,  0.5648,  0.5384,  0.5132,  0.4948,  0.4681]
                # 0.28     = [ 0.6664,  0.6715,  0.6763,  0.6820,  0.6859,  0.6869,  0.6903,  0.6882,  0.6810,  0.6705,  0.6532,  0.6358,  0.6142,  0.5868,  0.5611,  0.5362,  0.5178,  0.4918]
                # 0.31     = [ 0.6639,  0.6680,  0.6713,  0.6772,  0.6818,  0.6839,  0.6898,  0.6905,  0.6859,  0.6781,  0.6636,  0.6472,  0.6273,  0.6016,  0.5770,  0.5527,  0.5346,  0.5095]
                # 0.34     = [ 0.6513,  0.6544,  0.6561,  0.6620,  0.6672,  0.6704,  0.6785,  0.6823,  0.6805,  0.6757,  0.6644,  0.6491,  0.6315,  0.6080,  0.5853,  0.5617,  0.5440,  0.5203]
                # 0.37     = [ 0.6287,  0.6307,  0.6308,  0.6367,  0.6424,  0.6466,  0.6568,  0.6636,  0.6648,  0.6629,  0.6551,  0.6410,  0.6262,  0.6054,  0.5850,  0.5624,  0.5452,  0.5233]
                # 0.40     = [ 0.5964,  0.5975,  0.5962,  0.6021,  0.6083,  0.6134,  0.6255,  0.6352,  0.6392,  0.6401,  0.6360,  0.6231,  0.6113,  0.5936,  0.5760,  0.5543,  0.5378,  0.5180]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.301       max_score 0.7305                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.5524,  0.5602,  0.5650,  0.5703,  0.5722,  0.5702,  0.5611,  0.5535,  0.5409,  0.5263,  0.5084,  0.4919,  0.4739,  0.4410,  0.4251,  0.4008,  0.3785,  0.3601]
                # 0.13     = [ 0.5861,  0.5943,  0.5993,  0.6057,  0.6088,  0.6075,  0.5997,  0.5928,  0.5811,  0.5670,  0.5489,  0.5328,  0.5142,  0.4809,  0.4643,  0.4398,  0.4171,  0.3984]
                # 0.16     = [ 0.6178,  0.6260,  0.6311,  0.6383,  0.6425,  0.6421,  0.6355,  0.6296,  0.6189,  0.6054,  0.5876,  0.5719,  0.5529,  0.5193,  0.5021,  0.4774,  0.4544,  0.4355]
                # 0.19     = [ 0.6464,  0.6542,  0.6592,  0.6670,  0.6721,  0.6726,  0.6675,  0.6627,  0.6532,  0.6406,  0.6233,  0.6083,  0.5889,  0.5553,  0.5376,  0.5128,  0.4897,  0.4705]
                # 0.22     = [ 0.6707,  0.6775,  0.6824,  0.6905,  0.6961,  0.6978,  0.6942,  0.6909,  0.6827,  0.6713,  0.6549,  0.6408,  0.6213,  0.5880,  0.5700,  0.5452,  0.5220,  0.5028]
                # 0.25     = [ 0.6894,  0.6946,  0.6993,  0.7071,  0.7131,  0.7162,  0.7141,  0.7125,  0.7059,  0.6961,  0.6812,  0.6680,  0.6488,  0.6161,  0.5981,  0.5734,  0.5504,  0.5312]
                # 0.28     = [ 0.7009,  0.7039,  0.7086,  0.7157,  0.7216,  0.7262,  0.7257,  0.7261,  0.7212,  0.7134,  0.7004,  0.6885,  0.6702,  0.6385,  0.6208,  0.5963,  0.5738,  0.5546]
                # 0.31     = [ 0.7041,  0.7043,  0.7091,  0.7150,  0.7204,  0.7267,  0.7279,  0.7305,  0.7273,  0.7218,  0.7115,  0.7011,  0.6840,  0.6537,  0.6370,  0.6129,  0.5910,  0.5721]
                # 0.34     = [ 0.6980,  0.6948,  0.7001,  0.7044,  0.7091,  0.7172,  0.7199,  0.7247,  0.7233,  0.7205,  0.7133,  0.7046,  0.6894,  0.6608,  0.6455,  0.6219,  0.6010,  0.5825]
                # 0.37     = [ 0.6821,  0.6754,  0.6814,  0.6838,  0.6875,  0.6975,  0.7016,  0.7088,  0.7090,  0.7090,  0.7052,  0.6984,  0.6854,  0.6591,  0.6457,  0.6226,  0.6030,  0.5851]
                # 0.40     = [ 0.6565,  0.6464,  0.6532,  0.6537,  0.6565,  0.6684,  0.6738,  0.6832,  0.6848,  0.6875,  0.6874,  0.6825,  0.6722,  0.6483,  0.6372,  0.6146,  0.5966,  0.5794]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.408       max_score 0.8003                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.5853,  0.5983,  0.6066,  0.6153,  0.6174,  0.6181,  0.6168,  0.6129,  0.6065,  0.5956,  0.5825,  0.5665,  0.5491,  0.5368,  0.5228,  0.5039,  0.4889,  0.4714]
                # 0.13     = [ 0.6202,  0.6336,  0.6422,  0.6514,  0.6540,  0.6553,  0.6546,  0.6515,  0.6456,  0.6353,  0.6221,  0.6059,  0.5883,  0.5760,  0.5616,  0.5419,  0.5273,  0.5092]
                # 0.16     = [ 0.6536,  0.6672,  0.6760,  0.6856,  0.6885,  0.6903,  0.6903,  0.6879,  0.6827,  0.6730,  0.6601,  0.6437,  0.6261,  0.6138,  0.5988,  0.5786,  0.5643,  0.5457]
                # 0.19     = [ 0.6845,  0.6982,  0.7068,  0.7167,  0.7199,  0.7223,  0.7229,  0.7213,  0.7170,  0.7079,  0.6954,  0.6791,  0.6614,  0.6491,  0.6338,  0.6131,  0.5992,  0.5803]
                # 0.22     = [ 0.7120,  0.7255,  0.7337,  0.7437,  0.7469,  0.7500,  0.7511,  0.7506,  0.7472,  0.7388,  0.7270,  0.7109,  0.6934,  0.6812,  0.6655,  0.6446,  0.6311,  0.6120]
                # 0.25     = [ 0.7348,  0.7478,  0.7553,  0.7651,  0.7683,  0.7722,  0.7738,  0.7742,  0.7719,  0.7644,  0.7535,  0.7381,  0.7207,  0.7088,  0.6930,  0.6722,  0.6590,  0.6399]
                # 0.28     = [ 0.7518,  0.7638,  0.7703,  0.7795,  0.7828,  0.7873,  0.7895,  0.7907,  0.7897,  0.7834,  0.7736,  0.7592,  0.7422,  0.7308,  0.7149,  0.6947,  0.6819,  0.6630]
                # 0.31     = [ 0.7613,  0.7723,  0.7774,  0.7858,  0.7890,  0.7942,  0.7969,  0.7991,  0.7996,  0.7943,  0.7861,  0.7731,  0.7565,  0.7459,  0.7302,  0.7110,  0.6987,  0.6802]
                # 0.34     = [ 0.7625,  0.7720,  0.7756,  0.7831,  0.7861,  0.7919,  0.7950,  0.7981,  0.8003,  0.7961,  0.7899,  0.7786,  0.7628,  0.7530,  0.7378,  0.7200,  0.7082,  0.6904]
                # 0.37     = [ 0.7545,  0.7622,  0.7643,  0.7707,  0.7737,  0.7800,  0.7836,  0.7874,  0.7914,  0.7882,  0.7843,  0.7751,  0.7601,  0.7514,  0.7370,  0.7210,  0.7096,  0.6928]
                # 0.40     = [ 0.7370,  0.7429,  0.7435,  0.7489,  0.7517,  0.7587,  0.7626,  0.7672,  0.7729,  0.7707,  0.7692,  0.7622,  0.7482,  0.7407,  0.7273,  0.7134,  0.7024,  0.6871]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.501       max_score 0.8561                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.6125,  0.6251,  0.6368,  0.6459,  0.6560,  0.6567,  0.6573,  0.6555,  0.6499,  0.6448,  0.6371,  0.6309,  0.6139,  0.5990,  0.5910,  0.5754,  0.5663,  0.5489]
                # 0.13     = [ 0.6487,  0.6612,  0.6730,  0.6824,  0.6934,  0.6943,  0.6956,  0.6940,  0.6888,  0.6840,  0.6765,  0.6703,  0.6531,  0.6379,  0.6298,  0.6140,  0.6043,  0.5870]
                # 0.16     = [ 0.6837,  0.6959,  0.7077,  0.7173,  0.7290,  0.7301,  0.7320,  0.7308,  0.7261,  0.7216,  0.7144,  0.7081,  0.6908,  0.6755,  0.6673,  0.6512,  0.6411,  0.6238]
                # 0.19     = [ 0.7168,  0.7284,  0.7400,  0.7497,  0.7618,  0.7633,  0.7657,  0.7650,  0.7608,  0.7567,  0.7498,  0.7436,  0.7263,  0.7108,  0.7025,  0.6864,  0.6759,  0.6586]
                # 0.22     = [ 0.7468,  0.7578,  0.7691,  0.7787,  0.7909,  0.7927,  0.7956,  0.7954,  0.7919,  0.7884,  0.7818,  0.7758,  0.7585,  0.7431,  0.7347,  0.7187,  0.7079,  0.6908]
                # 0.25     = [ 0.7729,  0.7830,  0.7937,  0.8030,  0.8149,  0.8172,  0.8204,  0.8209,  0.8183,  0.8153,  0.8092,  0.8035,  0.7864,  0.7714,  0.7627,  0.7469,  0.7361,  0.7191]
                # 0.28     = [ 0.7937,  0.8027,  0.8128,  0.8216,  0.8327,  0.8354,  0.8389,  0.8401,  0.8385,  0.8364,  0.8308,  0.8256,  0.8087,  0.7943,  0.7854,  0.7701,  0.7594,  0.7428]
                # 0.31     = [ 0.8080,  0.8157,  0.8250,  0.8331,  0.8430,  0.8462,  0.8498,  0.8518,  0.8515,  0.8503,  0.8454,  0.8409,  0.8242,  0.8108,  0.8017,  0.7871,  0.7768,  0.7605]
                # 0.34     = [ 0.8147,  0.8211,  0.8292,  0.8364,  0.8448,  0.8484,  0.8519,  0.8548,  0.8561,  0.8558,  0.8518,  0.8482,  0.8320,  0.8197,  0.8105,  0.7968,  0.7873,  0.7714]
                # 0.37     = [ 0.8126,  0.8177,  0.8246,  0.8309,  0.8374,  0.8414,  0.8448,  0.8486,  0.8515,  0.8524,  0.8493,  0.8466,  0.8313,  0.8202,  0.8109,  0.7986,  0.7900,  0.7746]
                # 0.40     = [ 0.8014,  0.8052,  0.8111,  0.8161,  0.8207,  0.8251,  0.8284,  0.8330,  0.8377,  0.8397,  0.8377,  0.8358,  0.8215,  0.8120,  0.8027,  0.7917,  0.7845,  0.7693]
                # ------------                                                                                                               
                
                
                # dim 1000      actual_init_angle_score 0.899       max_score 0.4809                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.4016,  0.3884,  0.3689,  0.3415,  0.3111,  0.2784,  0.2412,  0.2009,  0.1561,  0.1123,  0.0669,  0.0222, -0.0224, -0.0642, -0.1091, -0.1461, -0.1832, -0.2210]
                # 0.13     = [ 0.4387,  0.4294,  0.4133,  0.3894,  0.3612,  0.3303,  0.2942,  0.2543,  0.2096,  0.1655,  0.1193,  0.0741,  0.0288, -0.0136, -0.0591, -0.0966, -0.1339, -0.1719]
                # 0.16     = [ 0.4655,  0.4606,  0.4490,  0.4297,  0.4048,  0.3765,  0.3421,  0.3033,  0.2592,  0.2150,  0.1685,  0.1231,  0.0772,  0.0345, -0.0115, -0.0494, -0.0868, -0.1249]
                # 0.19     = [ 0.4787,  0.4786,  0.4725,  0.4591,  0.4390,  0.4143,  0.3826,  0.3457,  0.3029,  0.2593,  0.2130,  0.1676,  0.1215,  0.0786,  0.0325, -0.0055, -0.0428, -0.0808]
                # 0.22     = [ 0.4757,  0.4807,  0.4809,  0.4743,  0.4606,  0.4407,  0.4130,  0.3792,  0.3385,  0.2963,  0.2510,  0.2060,  0.1603,  0.1175,  0.0717,  0.0338, -0.0032, -0.0408]
                # 0.25     = [ 0.4561,  0.4659,  0.4723,  0.4731,  0.4669,  0.4529,  0.4306,  0.4011,  0.3638,  0.3239,  0.2806,  0.2366,  0.1918,  0.1496,  0.1045,  0.0672,  0.0308, -0.0061]
                # 0.28     = [ 0.4214,  0.4352,  0.4473,  0.4552,  0.4569,  0.4495,  0.4335,  0.4095,  0.3767,  0.3403,  0.3000,  0.2577,  0.2146,  0.1734,  0.1297,  0.0932,  0.0577,  0.0219]
                # 0.31     = [ 0.3743,  0.3914,  0.4083,  0.4223,  0.4315,  0.4304,  0.4212,  0.4033,  0.3760,  0.3440,  0.3078,  0.2678,  0.2272,  0.1876,  0.1458,  0.1106,  0.0763,  0.0420]
                # 0.34     = [ 0.3178,  0.3375,  0.3585,  0.3775,  0.3932,  0.3978,  0.3949,  0.3833,  0.3618,  0.3348,  0.3035,  0.2664,  0.2288,  0.1912,  0.1519,  0.1185,  0.0857,  0.0531]
                # 0.37     = [ 0.2546,  0.2766,  0.3009,  0.3239,  0.3450,  0.3543,  0.3570,  0.3513,  0.3354,  0.3135,  0.2874,  0.2535,  0.2195,  0.1842,  0.1478,  0.1163,  0.0853,  0.0547]
                # 0.40     = [ 0.1871,  0.2111,  0.2381,  0.2645,  0.2900,  0.3032,  0.3105,  0.3098,  0.2990,  0.2820,  0.2609,  0.2302,  0.1999,  0.1669,  0.1336,  0.1043,  0.0751,  0.0467]
                # ------------                                                                                                               
                # dim 1000      actual_init_angle_score 1.007       max_score 0.5540                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.4602,  0.4520,  0.4380,  0.4196,  0.3960,  0.3682,  0.3396,  0.3067,  0.2715,  0.2349,  0.2009,  0.1620,  0.1297,  0.0982,  0.0680,  0.0410,  0.0124, -0.0120]
                # 0.13     = [ 0.4973,  0.4915,  0.4803,  0.4641,  0.4422,  0.4156,  0.3876,  0.3549,  0.3193,  0.2822,  0.2475,  0.2077,  0.1749,  0.1429,  0.1122,  0.0850,  0.0561,  0.0319]
                # 0.16     = [ 0.5265,  0.5237,  0.5157,  0.5024,  0.4830,  0.4583,  0.4314,  0.3993,  0.3638,  0.3264,  0.2913,  0.2508,  0.2176,  0.1852,  0.1542,  0.1269,  0.0980,  0.0739]
                # 0.19     = [ 0.5450,  0.5454,  0.5413,  0.5316,  0.5158,  0.4938,  0.4688,  0.4380,  0.4030,  0.3658,  0.3306,  0.2898,  0.2565,  0.2238,  0.1928,  0.1655,  0.1368,  0.1131]
                # 0.22     = [ 0.5501,  0.5540,  0.5540,  0.5487,  0.5376,  0.5194,  0.4971,  0.4685,  0.4349,  0.3985,  0.3637,  0.3232,  0.2899,  0.2575,  0.2266,  0.1996,  0.1715,  0.1483]
                # 0.25     = [ 0.5398,  0.5473,  0.5516,  0.5512,  0.5457,  0.5323,  0.5138,  0.4885,  0.4571,  0.4225,  0.3885,  0.3492,  0.3163,  0.2844,  0.2542,  0.2278,  0.2006,  0.1782]
                # 0.28     = [ 0.5143,  0.5250,  0.5333,  0.5379,  0.5384,  0.5305,  0.5166,  0.4958,  0.4675,  0.4356,  0.4033,  0.3660,  0.3340,  0.3031,  0.2739,  0.2486,  0.2226,  0.2014]
                # 0.31     = [ 0.4748,  0.4884,  0.5001,  0.5093,  0.5158,  0.5135,  0.5047,  0.4890,  0.4647,  0.4365,  0.4064,  0.3722,  0.3414,  0.3121,  0.2845,  0.2605,  0.2362,  0.2165]
                # 0.34     = [ 0.4237,  0.4397,  0.4544,  0.4674,  0.4793,  0.4823,  0.4786,  0.4683,  0.4484,  0.4245,  0.3972,  0.3667,  0.3376,  0.3104,  0.2847,  0.2625,  0.2404,  0.2224]
                # 0.37     = [ 0.3636,  0.3817,  0.3987,  0.4150,  0.4314,  0.4392,  0.4401,  0.4349,  0.4194,  0.4001,  0.3758,  0.3497,  0.3224,  0.2976,  0.2743,  0.2541,  0.2344,  0.2182]
                # 0.40     = [ 0.2967,  0.3169,  0.3358,  0.3548,  0.3751,  0.3867,  0.3916,  0.3909,  0.3795,  0.3646,  0.3434,  0.3217,  0.2964,  0.2742,  0.2533,  0.2353,  0.2181,  0.2040]
                # ------------                                                                                                               
                # dim 1000      actual_init_angle_score 1.108       max_score 0.6149                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.5073,  0.5035,  0.4963,  0.4833,  0.4654,  0.4442,  0.4210,  0.3923,  0.3589,  0.3271,  0.2946,  0.2623,  0.2291,  0.1969,  0.1678,  0.1391,  0.1145,  0.0907]
                # 0.13     = [ 0.5439,  0.5420,  0.5370,  0.5262,  0.5101,  0.4905,  0.4681,  0.4399,  0.4064,  0.3742,  0.3413,  0.3082,  0.2743,  0.2414,  0.2118,  0.1828,  0.1580,  0.1341]
                # 0.16     = [ 0.5742,  0.5742,  0.5716,  0.5633,  0.5497,  0.5322,  0.5111,  0.4837,  0.4507,  0.4183,  0.3853,  0.3516,  0.3171,  0.2837,  0.2537,  0.2245,  0.1996,  0.1758]
                # 0.19     = [ 0.5956,  0.5975,  0.5975,  0.5922,  0.5818,  0.5669,  0.5478,  0.5220,  0.4900,  0.4579,  0.4249,  0.3911,  0.3563,  0.3226,  0.2925,  0.2632,  0.2384,  0.2149]
                # 0.22     = [ 0.6056,  0.6094,  0.6120,  0.6102,  0.6038,  0.5922,  0.5758,  0.5526,  0.5221,  0.4910,  0.4586,  0.4250,  0.3902,  0.3566,  0.3266,  0.2976,  0.2731,  0.2501]
                # 0.25     = [ 0.6022,  0.6079,  0.6130,  0.6149,  0.6132,  0.6055,  0.5926,  0.5728,  0.5449,  0.5158,  0.4841,  0.4515,  0.4173,  0.3842,  0.3547,  0.3262,  0.3024,  0.2802]
                # 0.28     = [ 0.5846,  0.5921,  0.5993,  0.6049,  0.6082,  0.6047,  0.5961,  0.5807,  0.5562,  0.5300,  0.4997,  0.4688,  0.4357,  0.4037,  0.3751,  0.3475,  0.3247,  0.3036]
                # 0.31     = [ 0.5532,  0.5623,  0.5714,  0.5803,  0.5886,  0.5893,  0.5852,  0.5749,  0.5545,  0.5321,  0.5038,  0.4752,  0.4439,  0.4136,  0.3863,  0.3600,  0.3386,  0.3190]
                # 0.34     = [ 0.5095,  0.5200,  0.5307,  0.5426,  0.5555,  0.5600,  0.5603,  0.5554,  0.5393,  0.5215,  0.4954,  0.4699,  0.4410,  0.4129,  0.3872,  0.3626,  0.3430,  0.3252]
                # 0.37     = [ 0.4556,  0.4676,  0.4795,  0.4939,  0.5108,  0.5186,  0.5230,  0.5231,  0.5115,  0.4984,  0.4747,  0.4526,  0.4265,  0.4009,  0.3773,  0.3546,  0.3370,  0.3213]
                # 0.40     = [ 0.3938,  0.4072,  0.4202,  0.4367,  0.4571,  0.4677,  0.4756,  0.4802,  0.4726,  0.4641,  0.4428,  0.4242,  0.4010,  0.3781,  0.3566,  0.3359,  0.3206,  0.3071]
                # ------------                                                                                                               
                # dim 1000      actual_init_angle_score 1.206       max_score 0.6761                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.5484,  0.5492,  0.5462,  0.5376,  0.5259,  0.5093,  0.4901,  0.4662,  0.4407,  0.4118,  0.3855,  0.3564,  0.3265,  0.3010,  0.2758,  0.2510,  0.2322,  0.2154]
                # 0.13     = [ 0.5857,  0.5879,  0.5865,  0.5796,  0.5694,  0.5540,  0.5357,  0.5122,  0.4868,  0.4575,  0.4307,  0.4009,  0.3704,  0.3444,  0.3186,  0.2936,  0.2746,  0.2581]
                # 0.16     = [ 0.6179,  0.6214,  0.6217,  0.6167,  0.6084,  0.5947,  0.5776,  0.5548,  0.5297,  0.5005,  0.4733,  0.4430,  0.4120,  0.3856,  0.3594,  0.3344,  0.3154,  0.2992]
                # 0.19     = [ 0.6430,  0.6475,  0.6495,  0.6468,  0.6408,  0.6292,  0.6139,  0.5923,  0.5680,  0.5390,  0.5118,  0.4813,  0.4501,  0.4235,  0.3971,  0.3721,  0.3533,  0.3375]
                # 0.22     = [ 0.6586,  0.6640,  0.6675,  0.6673,  0.6642,  0.6553,  0.6422,  0.6226,  0.5996,  0.5714,  0.5446,  0.5143,  0.4830,  0.4565,  0.4303,  0.4056,  0.3872,  0.3718]
                # 0.25     = [ 0.6626,  0.6685,  0.6734,  0.6760,  0.6761,  0.6706,  0.6602,  0.6433,  0.6223,  0.5956,  0.5696,  0.5400,  0.5092,  0.4831,  0.4575,  0.4334,  0.4156,  0.4009]
                # 0.28     = [ 0.6536,  0.6598,  0.6659,  0.6713,  0.6747,  0.6731,  0.6658,  0.6523,  0.6340,  0.6095,  0.5850,  0.5568,  0.5269,  0.5016,  0.4770,  0.4538,  0.4371,  0.4231]
                # 0.31     = [ 0.6311,  0.6375,  0.6444,  0.6525,  0.6594,  0.6618,  0.6578,  0.6483,  0.6333,  0.6116,  0.5892,  0.5630,  0.5345,  0.5105,  0.4874,  0.4655,  0.4500,  0.4370]
                # 0.34     = [ 0.5959,  0.6023,  0.6099,  0.6205,  0.6305,  0.6368,  0.6361,  0.6307,  0.6194,  0.6009,  0.5811,  0.5575,  0.5309,  0.5085,  0.4875,  0.4672,  0.4533,  0.4412]
                # 0.37     = [ 0.5493,  0.5558,  0.5641,  0.5769,  0.5896,  0.5995,  0.6017,  0.6005,  0.5928,  0.5778,  0.5608,  0.5401,  0.5157,  0.4953,  0.4767,  0.4582,  0.4461,  0.4351]
                # 0.40     = [ 0.4934,  0.5002,  0.5090,  0.5239,  0.5389,  0.5519,  0.5567,  0.5592,  0.5550,  0.5433,  0.5292,  0.5114,  0.4894,  0.4709,  0.4550,  0.4384,  0.4282,  0.4183]
                # ------------                                                                                                               
                # dim 1000      actual_init_angle_score 1.299       max_score 0.7353                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.5851,  0.5888,  0.5887,  0.5847,  0.5765,  0.5643,  0.5489,  0.5309,  0.5094,  0.4859,  0.4634,  0.4402,  0.4157,  0.3943,  0.3724,  0.3558,  0.3380,  0.3259]
                # 0.13     = [ 0.6233,  0.6280,  0.6290,  0.6264,  0.6194,  0.6083,  0.5934,  0.5757,  0.5543,  0.5305,  0.5075,  0.4838,  0.4587,  0.4370,  0.4146,  0.3979,  0.3802,  0.3682]
                # 0.16     = [ 0.6575,  0.6630,  0.6652,  0.6640,  0.6585,  0.6486,  0.6346,  0.6176,  0.5964,  0.5724,  0.5491,  0.5250,  0.4995,  0.4775,  0.4549,  0.4382,  0.4206,  0.4089]
                # 0.19     = [ 0.6859,  0.6919,  0.6952,  0.6956,  0.6918,  0.6835,  0.6708,  0.6547,  0.6341,  0.6102,  0.5868,  0.5625,  0.5368,  0.5147,  0.4921,  0.4755,  0.4582,  0.4469]
                # 0.22     = [ 0.7064,  0.7126,  0.7169,  0.7191,  0.7171,  0.7108,  0.6999,  0.6851,  0.6655,  0.6422,  0.6191,  0.5948,  0.5692,  0.5471,  0.5247,  0.5085,  0.4917,  0.4809]
                # 0.25     = [ 0.7169,  0.7229,  0.7282,  0.7320,  0.7323,  0.7283,  0.7196,  0.7068,  0.6887,  0.6664,  0.6439,  0.6201,  0.5949,  0.5733,  0.5514,  0.5358,  0.5197,  0.5095]
                # 0.28     = [ 0.7157,  0.7211,  0.7273,  0.7328,  0.7353,  0.7340,  0.7281,  0.7176,  0.7015,  0.6809,  0.6595,  0.6366,  0.6123,  0.5913,  0.5704,  0.5556,  0.5405,  0.5312]
                # 0.31     = [ 0.7017,  0.7063,  0.7133,  0.7202,  0.7251,  0.7266,  0.7237,  0.7160,  0.7024,  0.6840,  0.6642,  0.6426,  0.6197,  0.5998,  0.5802,  0.5666,  0.5527,  0.5445]
                # 0.34     = [ 0.6748,  0.6786,  0.6862,  0.6945,  0.7016,  0.7058,  0.7061,  0.7014,  0.6906,  0.6748,  0.6571,  0.6372,  0.6161,  0.5975,  0.5796,  0.5674,  0.5549,  0.5480]
                # 0.37     = [ 0.6361,  0.6390,  0.6473,  0.6567,  0.6658,  0.6725,  0.6759,  0.6741,  0.6663,  0.6533,  0.6378,  0.6199,  0.6009,  0.5839,  0.5680,  0.5574,  0.5465,  0.5410]
                # 0.40     = [ 0.5870,  0.5893,  0.5982,  0.6086,  0.6195,  0.6284,  0.6346,  0.6356,  0.6305,  0.6204,  0.6072,  0.5913,  0.5745,  0.5592,  0.5454,  0.5365,  0.5272,  0.5231]
                # ------------                                                                                                               
                # dim 1000      actual_init_angle_score 1.401       max_score 0.7977                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.6207,  0.6282,  0.6314,  0.6314,  0.6269,  0.6191,  0.6076,  0.5939,  0.5777,  0.5593,  0.5410,  0.5217,  0.5009,  0.4850,  0.4696,  0.4550,  0.4430,  0.4327]
                # 0.13     = [ 0.6598,  0.6681,  0.6722,  0.6731,  0.6695,  0.6627,  0.6515,  0.6381,  0.6218,  0.6033,  0.5845,  0.5648,  0.5434,  0.5273,  0.5117,  0.4971,  0.4853,  0.4752]
                # 0.16     = [ 0.6959,  0.7046,  0.7096,  0.7115,  0.7089,  0.7032,  0.6925,  0.6796,  0.6633,  0.6448,  0.6257,  0.6057,  0.5838,  0.5676,  0.5518,  0.5374,  0.5258,  0.5159]
                # 0.19     = [ 0.7273,  0.7362,  0.7419,  0.7447,  0.7434,  0.7389,  0.7291,  0.7169,  0.7009,  0.6825,  0.6633,  0.6431,  0.6210,  0.6047,  0.5890,  0.5748,  0.5635,  0.5540]
                # 0.22     = [ 0.7522,  0.7610,  0.7673,  0.7710,  0.7710,  0.7679,  0.7594,  0.7481,  0.7328,  0.7147,  0.6955,  0.6755,  0.6534,  0.6372,  0.6218,  0.6079,  0.5971,  0.5882]
                # 0.25     = [ 0.7688,  0.7771,  0.7837,  0.7883,  0.7897,  0.7881,  0.7815,  0.7714,  0.7571,  0.7396,  0.7208,  0.7011,  0.6794,  0.6636,  0.6487,  0.6354,  0.6252,  0.6170]
                # 0.28     = [ 0.7752,  0.7825,  0.7894,  0.7946,  0.7977,  0.7976,  0.7933,  0.7846,  0.7718,  0.7553,  0.7372,  0.7182,  0.6975,  0.6822,  0.6681,  0.6556,  0.6461,  0.6388]
                # 0.31     = [ 0.7699,  0.7761,  0.7829,  0.7887,  0.7934,  0.7947,  0.7933,  0.7863,  0.7754,  0.7601,  0.7431,  0.7253,  0.7060,  0.6914,  0.6784,  0.6670,  0.6583,  0.6522]
                # 0.34     = [ 0.7524,  0.7572,  0.7639,  0.7701,  0.7763,  0.7790,  0.7807,  0.7755,  0.7669,  0.7530,  0.7374,  0.7211,  0.7037,  0.6900,  0.6784,  0.6682,  0.6606,  0.6558]
                # 0.37     = [ 0.7228,  0.7262,  0.7327,  0.7392,  0.7470,  0.7509,  0.7557,  0.7522,  0.7461,  0.7337,  0.7199,  0.7052,  0.6901,  0.6775,  0.6675,  0.6587,  0.6521,  0.6487]
                # 0.40     = [ 0.6822,  0.6844,  0.6907,  0.6974,  0.7066,  0.7114,  0.7193,  0.7175,  0.7138,  0.7030,  0.6909,  0.6779,  0.6653,  0.6538,  0.6455,  0.6382,  0.6327,  0.6308]
                # ------------                                                                                                               
                # dim 1000      actual_init_angle_score 1.496       max_score 0.8559                                 
                # step_1   = [ 0.35,    0.37,    0.39,    0.41,    0.43,    0.45,    0.47,    0.49,    0.51,    0.54,    0.56,    0.58,    0.60,    0.62,    0.64,    0.66,    0.68,    0.70]
                # vvvv cap_to__step_2                                                                                                                
                # 0.10     = [ 0.6529,  0.6623,  0.6682,  0.6705,  0.6697,  0.6646,  0.6574,  0.6467,  0.6346,  0.6219,  0.6050,  0.5904,  0.5765,  0.5635,  0.5503,  0.5394,  0.5312,  0.5240]
                # 0.13     = [ 0.6931,  0.7030,  0.7095,  0.7125,  0.7124,  0.7079,  0.7012,  0.6906,  0.6785,  0.6657,  0.6483,  0.6334,  0.6192,  0.6060,  0.5927,  0.5818,  0.5738,  0.5668]
                # 0.16     = [ 0.7308,  0.7409,  0.7481,  0.7517,  0.7523,  0.7486,  0.7424,  0.7322,  0.7201,  0.7073,  0.6894,  0.6743,  0.6598,  0.6465,  0.6333,  0.6224,  0.6148,  0.6081]
                # 0.19     = [ 0.7646,  0.7748,  0.7824,  0.7867,  0.7881,  0.7852,  0.7797,  0.7699,  0.7581,  0.7452,  0.7273,  0.7119,  0.6973,  0.6840,  0.6710,  0.6603,  0.6530,  0.6466]
                # 0.22     = [ 0.7931,  0.8030,  0.8108,  0.8157,  0.8179,  0.8161,  0.8114,  0.8023,  0.7908,  0.7780,  0.7602,  0.7449,  0.7303,  0.7171,  0.7044,  0.6941,  0.6871,  0.6812]
                # 0.25     = [ 0.8144,  0.8236,  0.8314,  0.8368,  0.8400,  0.8394,  0.8356,  0.8274,  0.8165,  0.8040,  0.7867,  0.7715,  0.7572,  0.7442,  0.7319,  0.7223,  0.7158,  0.7104]
                # 0.28     = [ 0.8267,  0.8349,  0.8425,  0.8482,  0.8524,  0.8532,  0.8505,  0.8433,  0.8334,  0.8212,  0.8049,  0.7902,  0.7762,  0.7638,  0.7521,  0.7433,  0.7373,  0.7326]
                # 0.31     = [ 0.8285,  0.8353,  0.8425,  0.8484,  0.8536,  0.8559,  0.8544,  0.8485,  0.8397,  0.8281,  0.8132,  0.7992,  0.7859,  0.7741,  0.7633,  0.7557,  0.7502,  0.7463]
                # 0.34     = [ 0.8188,  0.8240,  0.8307,  0.8365,  0.8427,  0.8466,  0.8464,  0.8418,  0.8344,  0.8234,  0.8104,  0.7975,  0.7851,  0.7741,  0.7642,  0.7581,  0.7531,  0.7501]
                # 0.37     = [ 0.7972,  0.8008,  0.8068,  0.8126,  0.8198,  0.8252,  0.8262,  0.8231,  0.8171,  0.8069,  0.7961,  0.7843,  0.7730,  0.7630,  0.7541,  0.7498,  0.7452,  0.7432]
                # 0.40     = [ 0.7644,  0.7664,  0.7718,  0.7775,  0.7855,  0.7925,  0.7946,  0.7929,  0.7883,  0.7789,  0.7705,  0.7600,  0.7497,  0.7409,  0.7330,  0.7305,  0.7263,  0.7254]
                # ------------       
                
                pass
            
            
            
            
            
            
            #-------------------#-------------------#-------------------
            dim_list =       [100,1000]
            test_time_list = [100,50]
            #dim_list =       [1000]
            #test_time_list = [20]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                print(f"dim {dim}   test_time {test_time}  {device}")
            #-------------------#-------------------#-------------------
                
                #-------------------#-------------------#-------------------
                init_angle_loss_list = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
                for target_init_angle_loss in init_angle_loss_list:
                    init__cap_to, noise_strength = _param_for__random_dummy_mat(dim=dim, target_angle_score=target_init_angle_loss)
                
                # random_init__cap_to__list = torch.tensor([0.1, 0.5])            # z axis
                # random_init__noise_strength__list = torch.tensor([0. , 0.5])    # z axis
                # for _iter__z_axis in range(z_axis__dim):
                #     random_init__cap_to = random_init__cap_to__list[_iter__z_axis]
                #     random_init__noise_strength = random_init__noise_strength__list[_iter__z_axis]
                #-------------------#-------------------#-------------------

                    x_axis__dim = 18
                    y_axis__dim = 11
                    
                    score = torch.empty(size=[y_axis__dim, x_axis__dim])    #dont modify this.
                    score.fill_(torch.nan)
                    
                    #-------------------#-------------------#-------------------
                    cap_to__step_1_list = torch.linspace(0.35, 0.7, x_axis__dim) # x axis
                    cap_to__step_2_list = torch.linspace(0.1, 0.4, y_axis__dim) # y axis
                    for _iter__cap_to__step_1 in range(cap_to__step_1_list.shape[0]): # x axis
                        cap_to__step_1 = cap_to__step_1_list[_iter__cap_to__step_1]
                    #-------------------#-------------------#-------------------
                        
                        _raw_result__score = torch.empty(size=[y_axis__dim, test_time])
                        _raw_result__score.fill_(torch.nan)
                        _raw_result__actual_init_angle_score = torch.empty(size=[test_time])
                        _raw_result__actual_init_angle_score.fill_(torch.nan)
                        
                        for _test_count in range(test_time):
                            
                            #-------------------#-------------------#-------------------
                            #<  init
                            while True:
                                ori_mat = random_dummy_mat(dim=dim, init__cap_to = init__cap_to, noise_strength = noise_strength,
                                                        device=device, iota_of_dim=iota_of_dim)#or maybe the noise_0.5
                                _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                                if _tensor_equal(angle_loss__in_the_beginning, target_init_angle_loss, epsilon= 0.05):
                                    break
                                #no tail.
                                pass
                            _raw_result__actual_init_angle_score[_test_count] = angle_loss__in_the_beginning
                            #</ init
                            
                            #<  calc
                            for _y_axis_count in range(y_axis__dim):
                                mat = ori_mat.detach().clone()
                                mat = full_test_version_of_angle_correction__by_row(mat,
                                        cap_to=cap_to__step_1,                 iota_of_dim=iota_of_dim)
                                mat = full_test_version_of_angle_correction__by_row(mat.T,
                                        cap_to=cap_to__step_2_list[_y_axis_count],  iota_of_dim=iota_of_dim).T
                                _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                                _raw_result__score[_y_axis_count, _test_count] = angle_loss__in_the_beginning-angle_loss__after
                                pass# for _y_axis_count
                            
                            pass#for _test_count
                        
                        score[:, _iter__cap_to__step_1] = _raw_result__score.mean(dim=1)
                        actual_init_angle_score = _raw_result__actual_init_angle_score.mean()
                        
                        pass#   for cap_to__step_1  #x axis
                    
                    print(f"# dim {dim}      actual_init_angle_score {actual_init_angle_score.item():.3f}       max_score {score.max().item():.4f}"+" "*33)
                    #print(f"# dim {dim}      actual_init_angle_score {actual_init_angle_score.item():.3f}     (y axis is the step_2 cap_to)")
                    print(f"# step_1   = {str_the_list(cap_to__step_1_list         , 2, segment=",   ")}")
                    
                    print(f"# vvvv cap_to__step_2 "+" "*111)
                    for ii in range(y_axis__dim):
                        print(f"# {cap_to__step_2_list[ii].item():.2f}     = {str_the_list(score[ii], 4)}")
                        pass
                    print(f"# ------------"+" "*111)
                    
                    pass# for cap_to__step_3
                    
                pass# for outter_param_count
            
            pass#/ test
        
        # target_init_angle_loss = torch.tensor(0.2)
        # while True:
        #     ori_mat = random_dummy_mat__v2__from_target(dim=100, target_angle_loss= target_init_angle_loss)
                                    
        #     _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
        #     if _tensor_equal(angle_loss__in_the_beginning, target_init_angle_loss, epsilon= 0.05):
        #         break
        #     #no tail.
        #     pass
        # #</ init
        
        # #<  calc
        # mat = ori_mat.detach().clone()
        # mat = full_test_version_of_angle_correction__by_row(mat,    cap_to=0.1,)
        # mat = full_test_version_of_angle_correction__by_row(mat.T,  cap_to=0.1,).T
        # _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
        
        # fds=432
        
        if "test 2, the dummy mat is in between 0. to 1.5, after a rc correct, it's ?????????????? ." and True:
            #result  
            
            if True:
                # score is angle_loss__after, not a difference. 
                
                
                
                
                
                # dim 100      actual_init_angle_score 0.101       best_score 0.0014                                 
                # step_1   = [ 0.000,   0.006,   0.012,   0.018,   0.024,   0.029,   0.035,   0.041,   0.047,   0.053,   0.059,   0.065,   0.071,   0.076,   0.082,   0.088,   0.094,   0.100]
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [ 0.0999,  0.0844,  0.0687,  0.0526,  0.0374,  0.0213,  0.0063,  0.0118,  0.0271,  0.0439,  0.0586,  0.0724,  0.0909,  0.1074,  0.1230,  0.1361,  0.1528,  0.1665]
                # 0.002     = [ 0.0944,  0.0789,  0.0634,  0.0471,  0.0321,  0.0161,  0.0014,  0.0068,  0.0219,  0.0386,  0.0532,  0.0670,  0.0855,  0.1021,  0.1177,  0.1307,  0.1475,  0.1612]
                # 0.004     = [ 0.0890,  0.0734,  0.0580,  0.0416,  0.0267,  0.0108,  0.0039,  0.0028,  0.0166,  0.0332,  0.0478,  0.0617,  0.0802,  0.0967,  0.1124,  0.1254,  0.1421,  0.1558]
                # 0.006     = [ 0.0835,  0.0679,  0.0526,  0.0361,  0.0213,  0.0055,  0.0088,  0.0034,  0.0114,  0.0278,  0.0424,  0.0563,  0.0748,  0.0914,  0.1071,  0.1200,  0.1368,  0.1505]
                # 0.008     = [ 0.0781,  0.0625,  0.0472,  0.0307,  0.0159,  0.0021,  0.0138,  0.0083,  0.0061,  0.0224,  0.0371,  0.0509,  0.0694,  0.0861,  0.1018,  0.1146,  0.1315,  0.1451]
                # 0.010     = [ 0.0727,  0.0570,  0.0418,  0.0252,  0.0106,  0.0050,  0.0188,  0.0133,  0.0025,  0.0170,  0.0317,  0.0456,  0.0640,  0.0807,  0.0965,  0.1092,  0.1262,  0.1398]
                # 0.012     = [ 0.0672,  0.0515,  0.0364,  0.0197,  0.0052,  0.0103,  0.0239,  0.0184,  0.0045,  0.0117,  0.0263,  0.0402,  0.0587,  0.0754,  0.0912,  0.1039,  0.1209,  0.1344]
                # 0.014     = [ 0.0618,  0.0461,  0.0310,  0.0143,  0.0017,  0.0156,  0.0289,  0.0234,  0.0097,  0.0064,  0.0209,  0.0348,  0.0533,  0.0700,  0.0859,  0.0985,  0.1155,  0.1290]
                # 0.016     = [ 0.0564,  0.0406,  0.0257,  0.0088,  0.0057,  0.0209,  0.0340,  0.0285,  0.0150,  0.0035,  0.0155,  0.0295,  0.0479,  0.0647,  0.0806,  0.0931,  0.1102,  0.1237]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.151       best_score 0.0018                                 
                # step_1   = [ 0.000,   0.006,   0.012,   0.018,   0.024,   0.029,   0.035,   0.041,   0.047,   0.053,   0.059,   0.065,   0.071,   0.076,   0.082,   0.088,   0.094,   0.100]
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [ 0.1496,  0.1338,  0.1178,  0.1025,  0.0860,  0.0708,  0.0551,  0.0390,  0.0251,  0.0124,  0.0140,  0.0273,  0.0416,  0.0573,  0.0748,  0.0874,  0.1038,  0.1181]
                # 0.002     = [ 0.1442,  0.1284,  0.1125,  0.0971,  0.0806,  0.0653,  0.0497,  0.0336,  0.0199,  0.0075,  0.0091,  0.0223,  0.0365,  0.0520,  0.0696,  0.0821,  0.0985,  0.1129]
                # 0.004     = [ 0.1387,  0.1230,  0.1071,  0.0916,  0.0752,  0.0599,  0.0443,  0.0282,  0.0147,  0.0025,  0.0043,  0.0173,  0.0313,  0.0468,  0.0643,  0.0769,  0.0933,  0.1076]
                # 0.006     = [ 0.1333,  0.1176,  0.1017,  0.0862,  0.0698,  0.0545,  0.0389,  0.0229,  0.0095,  0.0029,  0.0018,  0.0123,  0.0262,  0.0415,  0.0591,  0.0716,  0.0880,  0.1024]
                # 0.008     = [ 0.1279,  0.1122,  0.0963,  0.0807,  0.0644,  0.0491,  0.0335,  0.0175,  0.0044,  0.0075,  0.0054,  0.0072,  0.0210,  0.0363,  0.0538,  0.0663,  0.0827,  0.0972]
                # 0.010     = [ 0.1225,  0.1068,  0.0909,  0.0753,  0.0590,  0.0437,  0.0281,  0.0122,  0.0028,  0.0125,  0.0103,  0.0034,  0.0158,  0.0310,  0.0486,  0.0611,  0.0775,  0.0919]
                # 0.012     = [ 0.1170,  0.1014,  0.0855,  0.0698,  0.0536,  0.0383,  0.0227,  0.0068,  0.0061,  0.0175,  0.0152,  0.0036,  0.0107,  0.0257,  0.0433,  0.0558,  0.0722,  0.0867]
                # 0.014     = [ 0.1116,  0.0960,  0.0802,  0.0644,  0.0482,  0.0329,  0.0173,  0.0025,  0.0113,  0.0225,  0.0201,  0.0080,  0.0057,  0.0204,  0.0380,  0.0505,  0.0669,  0.0814]
                # 0.016     = [ 0.1062,  0.0906,  0.0748,  0.0590,  0.0429,  0.0275,  0.0119,  0.0042,  0.0165,  0.0275,  0.0250,  0.0130,  0.0030,  0.0152,  0.0327,  0.0452,  0.0617,  0.0761]
                # 0.018     = [ 0.1008,  0.0852,  0.0694,  0.0535,  0.0375,  0.0221,  0.0066,  0.0094,  0.0218,  0.0325,  0.0299,  0.0181,  0.0052,  0.0100,  0.0275,  0.0400,  0.0564,  0.0709]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.200       best_score 0.0018                                 
                # step_1   = [ 0.000,   0.006,   0.012,   0.018,   0.024,   0.029,   0.035,   0.041,   0.047,   0.053,   0.059,   0.065,   0.071,   0.076,   0.082,   0.088,   0.094,   0.100]
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [ 0.1991,  0.1834,  0.1670,  0.1527,  0.1360,  0.1205,  0.1052,  0.0896,  0.0726,  0.0584,  0.0449,  0.0310,  0.0209,  0.0203,  0.0300,  0.0428,  0.0581,  0.0722]
                # 0.002     = [ 0.1937,  0.1780,  0.1616,  0.1473,  0.1306,  0.1151,  0.0999,  0.0844,  0.0673,  0.0531,  0.0398,  0.0261,  0.0160,  0.0154,  0.0252,  0.0379,  0.0532,  0.0672]
                # 0.004     = [ 0.1884,  0.1726,  0.1562,  0.1419,  0.1253,  0.1098,  0.0945,  0.0791,  0.0620,  0.0479,  0.0346,  0.0211,  0.0112,  0.0105,  0.0203,  0.0330,  0.0482,  0.0623]
                # 0.006     = [ 0.1830,  0.1672,  0.1508,  0.1365,  0.1199,  0.1044,  0.0891,  0.0738,  0.0568,  0.0426,  0.0295,  0.0161,  0.0063,  0.0056,  0.0154,  0.0280,  0.0433,  0.0573]
                # 0.008     = [ 0.1777,  0.1618,  0.1455,  0.1311,  0.1145,  0.0991,  0.0838,  0.0685,  0.0515,  0.0373,  0.0243,  0.0111,  0.0025,  0.0018,  0.0105,  0.0231,  0.0383,  0.0523]
                # 0.010     = [ 0.1723,  0.1564,  0.1401,  0.1258,  0.1091,  0.0937,  0.0784,  0.0632,  0.0462,  0.0321,  0.0191,  0.0064,  0.0038,  0.0043,  0.0063,  0.0182,  0.0333,  0.0473]
                # 0.012     = [ 0.1669,  0.1510,  0.1347,  0.1204,  0.1037,  0.0884,  0.0730,  0.0580,  0.0409,  0.0268,  0.0140,  0.0037,  0.0084,  0.0092,  0.0035,  0.0132,  0.0283,  0.0423]
                # 0.014     = [ 0.1616,  0.1456,  0.1293,  0.1150,  0.0983,  0.0830,  0.0677,  0.0527,  0.0356,  0.0215,  0.0088,  0.0051,  0.0133,  0.0141,  0.0051,  0.0085,  0.0233,  0.0373]
                # 0.016     = [ 0.1563,  0.1402,  0.1239,  0.1096,  0.0929,  0.0776,  0.0623,  0.0474,  0.0303,  0.0162,  0.0042,  0.0090,  0.0182,  0.0191,  0.0093,  0.0054,  0.0183,  0.0323]
                # 0.018     = [ 0.1509,  0.1348,  0.1185,  0.1042,  0.0875,  0.0723,  0.0569,  0.0421,  0.0250,  0.0110,  0.0039,  0.0140,  0.0231,  0.0240,  0.0142,  0.0042,  0.0134,  0.0273]
                # 0.020     = [ 0.1456,  0.1294,  0.1131,  0.0989,  0.0821,  0.0669,  0.0516,  0.0368,  0.0198,  0.0059,  0.0071,  0.0191,  0.0280,  0.0290,  0.0192,  0.0071,  0.0089,  0.0223]
                # 0.022     = [ 0.1402,  0.1240,  0.1077,  0.0935,  0.0767,  0.0616,  0.0462,  0.0315,  0.0145,  0.0030,  0.0122,  0.0241,  0.0329,  0.0340,  0.0242,  0.0118,  0.0059,  0.0173]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.250       best_score 0.0018                                 
                # step_1   = [ 0.050,   0.059,   0.068,   0.076,   0.085,   0.094,   0.103,   0.112,   0.121,   0.129,   0.138,   0.147,   0.156,   0.165,   0.174,   0.182,   0.191,   0.200]
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [ 0.1162,  0.0948,  0.0724,  0.0507,  0.0364,  0.0299,  0.0414,  0.0570,  0.0798,  0.0992,  0.1218,  0.1471,  0.1717,  0.1940,  0.2136,  0.2354,  0.2571,  0.2860]
                # 0.002     = [ 0.1108,  0.0896,  0.0672,  0.0458,  0.0316,  0.0249,  0.0366,  0.0523,  0.0750,  0.0943,  0.1169,  0.1421,  0.1666,  0.1889,  0.2085,  0.2304,  0.2521,  0.2810]
                # 0.004     = [ 0.1055,  0.0843,  0.0621,  0.0408,  0.0268,  0.0200,  0.0317,  0.0475,  0.0702,  0.0894,  0.1119,  0.1371,  0.1616,  0.1839,  0.2035,  0.2254,  0.2472,  0.2760]
                # 0.006     = [ 0.1002,  0.0791,  0.0569,  0.0358,  0.0219,  0.0150,  0.0269,  0.0427,  0.0654,  0.0845,  0.1070,  0.1321,  0.1565,  0.1788,  0.1984,  0.2203,  0.2422,  0.2711]
                # 0.008     = [ 0.0949,  0.0739,  0.0517,  0.0309,  0.0171,  0.0100,  0.0221,  0.0379,  0.0606,  0.0796,  0.1020,  0.1271,  0.1515,  0.1738,  0.1934,  0.2153,  0.2372,  0.2661]
                # 0.010     = [ 0.0896,  0.0687,  0.0465,  0.0259,  0.0122,  0.0050,  0.0172,  0.0331,  0.0557,  0.0746,  0.0970,  0.1220,  0.1464,  0.1687,  0.1883,  0.2103,  0.2322,  0.2611]
                # 0.012     = [ 0.0842,  0.0635,  0.0413,  0.0209,  0.0075,  0.0016,  0.0124,  0.0283,  0.0509,  0.0697,  0.0920,  0.1170,  0.1413,  0.1636,  0.1832,  0.2053,  0.2272,  0.2561]
                # 0.014     = [ 0.0789,  0.0583,  0.0362,  0.0159,  0.0039,  0.0051,  0.0078,  0.0235,  0.0460,  0.0647,  0.0871,  0.1120,  0.1363,  0.1585,  0.1782,  0.2002,  0.2222,  0.2511]
                # 0.016     = [ 0.0736,  0.0530,  0.0310,  0.0109,  0.0041,  0.0101,  0.0043,  0.0187,  0.0412,  0.0598,  0.0821,  0.1069,  0.1312,  0.1535,  0.1731,  0.1952,  0.2172,  0.2461]
                # 0.018     = [ 0.0682,  0.0478,  0.0258,  0.0062,  0.0074,  0.0152,  0.0040,  0.0141,  0.0363,  0.0548,  0.0771,  0.1019,  0.1261,  0.1484,  0.1680,  0.1901,  0.2122,  0.2411]
                # 0.020     = [ 0.0629,  0.0426,  0.0206,  0.0039,  0.0122,  0.0202,  0.0073,  0.0100,  0.0314,  0.0498,  0.0721,  0.0968,  0.1210,  0.1433,  0.1629,  0.1851,  0.2072,  0.2361]
                # 0.022     = [ 0.0576,  0.0373,  0.0154,  0.0057,  0.0171,  0.0252,  0.0121,  0.0066,  0.0266,  0.0448,  0.0670,  0.0918,  0.1159,  0.1382,  0.1578,  0.1800,  0.2022,  0.2311]
                # 0.024     = [ 0.0522,  0.0321,  0.0103,  0.0097,  0.0220,  0.0303,  0.0170,  0.0056,  0.0220,  0.0399,  0.0620,  0.0867,  0.1108,  0.1331,  0.1527,  0.1750,  0.1972,  0.2261]
                # 0.026     = [ 0.0469,  0.0269,  0.0058,  0.0145,  0.0269,  0.0354,  0.0219,  0.0073,  0.0176,  0.0349,  0.0570,  0.0816,  0.1057,  0.1280,  0.1476,  0.1699,  0.1922,  0.2211]
                # 0.028     = [ 0.0416,  0.0217,  0.0042,  0.0196,  0.0318,  0.0405,  0.0269,  0.0110,  0.0137,  0.0299,  0.0520,  0.0766,  0.1006,  0.1229,  0.1425,  0.1648,  0.1872,  0.2161]
                # 0.030     = [ 0.0363,  0.0166,  0.0067,  0.0247,  0.0368,  0.0455,  0.0318,  0.0156,  0.0107,  0.0250,  0.0470,  0.0715,  0.0954,  0.1178,  0.1374,  0.1598,  0.1821,  0.2111]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.303       best_score 0.0028                                 
                # step_1   = [ 0.050,   0.059,   0.068,   0.076,   0.085,   0.094,   0.103,   0.112,   0.121,   0.129,   0.138,   0.147,   0.156,   0.165,   0.174,   0.182,   0.191,   0.200]
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [ 0.1701,  0.1496,  0.1239,  0.1007,  0.0821,  0.0643,  0.0489,  0.0427,  0.0489,  0.0640,  0.0840,  0.1062,  0.1255,  0.1440,  0.1652,  0.1873,  0.2091,  0.2298]
                # 0.002     = [ 0.1648,  0.1443,  0.1187,  0.0955,  0.0770,  0.0594,  0.0441,  0.0377,  0.0441,  0.0593,  0.0793,  0.1014,  0.1207,  0.1392,  0.1605,  0.1825,  0.2043,  0.2249]
                # 0.004     = [ 0.1595,  0.1391,  0.1135,  0.0904,  0.0720,  0.0545,  0.0393,  0.0327,  0.0393,  0.0546,  0.0746,  0.0966,  0.1159,  0.1344,  0.1558,  0.1776,  0.1994,  0.2201]
                # 0.006     = [ 0.1542,  0.1338,  0.1084,  0.0852,  0.0669,  0.0496,  0.0344,  0.0277,  0.0345,  0.0499,  0.0698,  0.0918,  0.1111,  0.1296,  0.1510,  0.1728,  0.1945,  0.2152]
                # 0.008     = [ 0.1489,  0.1285,  0.1032,  0.0801,  0.0618,  0.0446,  0.0296,  0.0227,  0.0297,  0.0452,  0.0650,  0.0870,  0.1062,  0.1247,  0.1463,  0.1680,  0.1896,  0.2103]
                # 0.010     = [ 0.1436,  0.1232,  0.0980,  0.0749,  0.0567,  0.0397,  0.0248,  0.0177,  0.0249,  0.0405,  0.0603,  0.0822,  0.1013,  0.1199,  0.1415,  0.1631,  0.1847,  0.2054]
                # 0.012     = [ 0.1383,  0.1179,  0.0928,  0.0697,  0.0516,  0.0348,  0.0199,  0.0127,  0.0200,  0.0358,  0.0555,  0.0773,  0.0965,  0.1150,  0.1367,  0.1583,  0.1798,  0.2006]
                # 0.014     = [ 0.1330,  0.1126,  0.0876,  0.0646,  0.0465,  0.0298,  0.0151,  0.0077,  0.0152,  0.0311,  0.0507,  0.0725,  0.0916,  0.1102,  0.1320,  0.1534,  0.1749,  0.1957]
                # 0.016     = [ 0.1277,  0.1073,  0.0824,  0.0594,  0.0413,  0.0248,  0.0103,  0.0028,  0.0104,  0.0264,  0.0459,  0.0676,  0.0867,  0.1053,  0.1272,  0.1485,  0.1700,  0.1908]
                # 0.018     = [ 0.1224,  0.1020,  0.0772,  0.0542,  0.0362,  0.0199,  0.0057,  0.0029,  0.0064,  0.0216,  0.0411,  0.0627,  0.0818,  0.1004,  0.1224,  0.1437,  0.1650,  0.1859]
                # 0.020     = [ 0.1171,  0.0966,  0.0720,  0.0490,  0.0311,  0.0149,  0.0050,  0.0075,  0.0043,  0.0169,  0.0363,  0.0579,  0.0769,  0.0956,  0.1176,  0.1388,  0.1601,  0.1809]
                # 0.022     = [ 0.1118,  0.0913,  0.0668,  0.0438,  0.0260,  0.0101,  0.0069,  0.0125,  0.0054,  0.0123,  0.0315,  0.0530,  0.0720,  0.0907,  0.1128,  0.1339,  0.1552,  0.1760]
                # 0.024     = [ 0.1065,  0.0860,  0.0616,  0.0387,  0.0208,  0.0064,  0.0104,  0.0175,  0.0094,  0.0085,  0.0267,  0.0481,  0.0671,  0.0858,  0.1080,  0.1290,  0.1502,  0.1711]
                # 0.026     = [ 0.1012,  0.0807,  0.0564,  0.0335,  0.0158,  0.0048,  0.0146,  0.0226,  0.0141,  0.0068,  0.0219,  0.0432,  0.0622,  0.0809,  0.1032,  0.1241,  0.1453,  0.1662]
                # 0.028     = [ 0.0959,  0.0754,  0.0512,  0.0283,  0.0108,  0.0065,  0.0193,  0.0277,  0.0190,  0.0070,  0.0173,  0.0383,  0.0573,  0.0760,  0.0984,  0.1192,  0.1403,  0.1612]
                # 0.030     = [ 0.0906,  0.0701,  0.0460,  0.0232,  0.0069,  0.0106,  0.0242,  0.0328,  0.0239,  0.0094,  0.0132,  0.0335,  0.0524,  0.0711,  0.0936,  0.1143,  0.1354,  0.1563]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.351       best_score 0.0028                                 
                # step_1   = [ 0.050,   0.059,   0.068,   0.076,   0.085,   0.094,   0.103,   0.112,   0.121,   0.129,   0.138,   0.147,   0.156,   0.165,   0.174,   0.182,   0.191,   0.200]
                # vvvv cap_to__step_2                                                                                                                
                # 0.010     = [ 0.1943,  0.1690,  0.1488,  0.1267,  0.1034,  0.0862,  0.0687,  0.0512,  0.0387,  0.0336,  0.0383,  0.0511,  0.0653,  0.0839,  0.1016,  0.1300,  0.1458,  0.1669]
                # 0.012     = [ 0.1891,  0.1637,  0.1437,  0.1216,  0.0983,  0.0812,  0.0639,  0.0464,  0.0339,  0.0287,  0.0335,  0.0464,  0.0605,  0.0792,  0.0970,  0.1253,  0.1411,  0.1622]
                # 0.014     = [ 0.1839,  0.1584,  0.1385,  0.1164,  0.0931,  0.0761,  0.0591,  0.0416,  0.0291,  0.0237,  0.0286,  0.0417,  0.0557,  0.0746,  0.0923,  0.1206,  0.1363,  0.1575]
                # 0.016     = [ 0.1787,  0.1532,  0.1334,  0.1113,  0.0880,  0.0711,  0.0542,  0.0368,  0.0243,  0.0187,  0.0237,  0.0369,  0.0509,  0.0699,  0.0877,  0.1159,  0.1315,  0.1528]
                # 0.018     = [ 0.1735,  0.1479,  0.1282,  0.1061,  0.0828,  0.0661,  0.0494,  0.0320,  0.0195,  0.0137,  0.0188,  0.0322,  0.0461,  0.0652,  0.0830,  0.1112,  0.1268,  0.1481]
                # 0.020     = [ 0.1683,  0.1426,  0.1230,  0.1010,  0.0777,  0.0610,  0.0445,  0.0272,  0.0147,  0.0088,  0.0140,  0.0274,  0.0413,  0.0605,  0.0784,  0.1064,  0.1220,  0.1433]
                # 0.022     = [ 0.1631,  0.1374,  0.1179,  0.0958,  0.0725,  0.0560,  0.0396,  0.0224,  0.0099,  0.0044,  0.0091,  0.0227,  0.0364,  0.0558,  0.0737,  0.1017,  0.1172,  0.1386]
                # 0.024     = [ 0.1579,  0.1321,  0.1127,  0.0907,  0.0673,  0.0510,  0.0348,  0.0176,  0.0058,  0.0038,  0.0050,  0.0183,  0.0316,  0.0511,  0.0690,  0.0970,  0.1124,  0.1338]
                # 0.026     = [ 0.1527,  0.1268,  0.1075,  0.0855,  0.0622,  0.0459,  0.0299,  0.0130,  0.0043,  0.0069,  0.0049,  0.0142,  0.0268,  0.0464,  0.0643,  0.0922,  0.1076,  0.1291]
                # 0.028     = [ 0.1475,  0.1215,  0.1024,  0.0804,  0.0570,  0.0409,  0.0251,  0.0091,  0.0065,  0.0117,  0.0076,  0.0113,  0.0220,  0.0417,  0.0597,  0.0875,  0.1028,  0.1243]
                # 0.030     = [ 0.1423,  0.1163,  0.0972,  0.0752,  0.0518,  0.0358,  0.0203,  0.0072,  0.0104,  0.0167,  0.0114,  0.0098,  0.0173,  0.0371,  0.0550,  0.0827,  0.0980,  0.1195]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.402       best_score 0.0090                                 
                # step_1     = [ 0.050,   0.059,   0.068,   0.076,   0.085,   0.094,   0.103,   0.112,   0.121,   0.129,   0.138,   0.147,   0.156,   0.165,   0.174,   0.182,   0.191,   0.200]
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [ 0.2692,  0.2471,  0.2255,  0.2041,  0.1834,  0.1618,  0.1404,  0.1208,  0.1070,  0.0908,  0.0796,  0.0770,  0.0780,  0.0847,  0.0959,  0.1168,  0.1293,  0.1507]
                # 0.013     = [ 0.2343,  0.2122,  0.1912,  0.1701,  0.1494,  0.1281,  0.1072,  0.0882,  0.0753,  0.0593,  0.0468,  0.0441,  0.0446,  0.0527,  0.0648,  0.0858,  0.0986,  0.1202]
                # 0.027     = [ 0.1993,  0.1773,  0.1568,  0.1359,  0.1151,  0.0941,  0.0736,  0.0553,  0.0431,  0.0274,  0.0140,  0.0111,  0.0112,  0.0205,  0.0334,  0.0545,  0.0676,  0.0894]
                # 0.040     = [ 0.1644,  0.1423,  0.1224,  0.1017,  0.0809,  0.0602,  0.0401,  0.0227,  0.0123,  0.0090,  0.0205,  0.0234,  0.0235,  0.0150,  0.0094,  0.0236,  0.0365,  0.0584]
                # 0.053     = [ 0.1296,  0.1076,  0.0883,  0.0678,  0.0472,  0.0273,  0.0117,  0.0155,  0.0240,  0.0386,  0.0542,  0.0573,  0.0578,  0.0462,  0.0320,  0.0156,  0.0147,  0.0287]
                # 0.067     = [ 0.0956,  0.0737,  0.0554,  0.0360,  0.0199,  0.0184,  0.0319,  0.0472,  0.0568,  0.0718,  0.0885,  0.0916,  0.0925,  0.0797,  0.0644,  0.0444,  0.0317,  0.0182]
                # 0.080     = [ 0.0634,  0.0433,  0.0299,  0.0224,  0.0315,  0.0482,  0.0661,  0.0815,  0.0904,  0.1053,  0.1232,  0.1262,  0.1276,  0.1135,  0.0973,  0.0773,  0.0637,  0.0427]
                # 0.093     = [ 0.0390,  0.0306,  0.0345,  0.0460,  0.0643,  0.0829,  0.1012,  0.1162,  0.1245,  0.1392,  0.1581,  0.1611,  0.1628,  0.1476,  0.1307,  0.1109,  0.0969,  0.0748]
                # 0.107     = [ 0.0402,  0.0504,  0.0626,  0.0792,  0.0993,  0.1184,  0.1367,  0.1513,  0.1589,  0.1735,  0.1934,  0.1962,  0.1984,  0.1821,  0.1644,  0.1449,  0.1306,  0.1080]
                # 0.120     = [ 0.0655,  0.0822,  0.0962,  0.1140,  0.1348,  0.1542,  0.1724,  0.1867,  0.1936,  0.2080,  0.2289,  0.2316,  0.2341,  0.2167,  0.1984,  0.1793,  0.1647,  0.1418]
                # dim 100      actual_init_angle_score 0.450       best_score 0.0075                                 
                # step_1     = [  0.094,   0.105,   0.116,   0.127,   0.138,   0.148,   0.159,   0.170,   0.181,   0.192,   0.203,   0.214,   0.225,   0.236,  
                # vvvv cap_to__step_2                                                                                                                                                             
                # 0.000     = [   0.2141,  0.1847,  0.1628,  0.1404,  0.1231,  0.1071,  0.0975,  0.0955,  0.1026,  0.1142,  0.1316,  0.1576,  0.1763,  0.1958,  
                # 0.013     = [   0.1804,  0.1513,  0.1301,  0.1078,  0.0910,  0.0749,  0.0650,  0.0625,  0.0699,  0.0827,  0.1010,  0.1275,  0.1467,  0.1661,  
                # 0.027     = [   0.1465,  0.1176,  0.0970,  0.0748,  0.0585,  0.0424,  0.0324,  0.0292,  0.0370,  0.0510,  0.0699,  0.0971,  0.1168,  0.1359,  
                # 0.040     = [   0.1125,  0.0838,  0.0638,  0.0418,  0.0261,  0.0116,  0.0084,  0.0075,  0.0086,  0.0198,  0.0389,  0.0665,  0.0867,  0.1055,  
                # 0.053     = [   0.0788,  0.0504,  0.0313,  0.0137,  0.0144,  0.0257,  0.0353,  0.0393,  0.0313,  0.0179,  0.0150,  0.0368,  0.0567,  0.0751,  
                # 0.067     = [   0.0464,  0.0215,  0.0154,  0.0298,  0.0437,  0.0590,  0.0690,  0.0736,  0.0652,  0.0486,  0.0301,  0.0202,  0.0300,  0.0463,  
                # 0.080     = [   0.0258,  0.0286,  0.0429,  0.0634,  0.0776,  0.0931,  0.1032,  0.1082,  0.0995,  0.0818,  0.0613,  0.0363,  0.0257,  0.0292,  
                # 0.093     = [   0.0388,  0.0603,  0.0769,  0.0982,  0.1122,  0.1277,  0.1377,  0.1431,  0.1342,  0.1155,  0.0942,  0.0665,  0.0477,  0.0379,  
                # 0.107     = [   0.0699,  0.0950,  0.1119,  0.1335,  0.1472,  0.1625,  0.1725,  0.1783,  0.1692,  0.1495,  0.1277,  0.0990,  0.0782,  0.0644,  
                # 0.120     = [   0.1042,  0.1305,  0.1472,  0.1691,  0.1826,  0.1977,  0.2075,  0.2136,  0.2044,  0.1838,  0.1616,  0.1323,  0.1106,  0.0960,  
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.501       best_score 0.0126                                 
                # step_1     = [  0.116,   0.127,   0.138,   0.148,   0.159,   0.170,   0.181,   0.192,   0.203,   0.214,   0.225,   0.236,   0.247,   0.258,   0.269,   0.280, 
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [  0.2175,  0.1993,  0.1729,  0.1510,  0.1388,  0.1267,  0.1188,  0.1197,  0.1272,  0.1392,  0.1565,  0.1720,  0.1926,  0.2207,  0.2326,  0.2534, 
                # 0.013     = [  0.1851,  0.1667,  0.1416,  0.1195,  0.1073,  0.0950,  0.0866,  0.0869,  0.0947,  0.1079,  0.1262,  0.1419,  0.1633,  0.1919,  0.2040,  0.2246, 
                # 0.027     = [  0.1524,  0.1338,  0.1099,  0.0877,  0.0755,  0.0629,  0.0541,  0.0538,  0.0619,  0.0764,  0.0955,  0.1114,  0.1337,  0.1627,  0.1750,  0.1955, 
                # 0.040     = [  0.1195,  0.1007,  0.0780,  0.0557,  0.0436,  0.0310,  0.0219,  0.0210,  0.0292,  0.0448,  0.0648,  0.0808,  0.1038,  0.1333,  0.1457,  0.1661, 
                # 0.053     = [  0.0867,  0.0678,  0.0464,  0.0248,  0.0155,  0.0132,  0.0157,  0.0163,  0.0126,  0.0181,  0.0354,  0.0506,  0.0741,  0.1038,  0.1164,  0.1366, 
                # 0.067     = [  0.0548,  0.0366,  0.0203,  0.0187,  0.0269,  0.0381,  0.0472,  0.0492,  0.0410,  0.0262,  0.0206,  0.0269,  0.0459,  0.0750,  0.0875,  0.1073, 
                # 0.080     = [  0.0287,  0.0228,  0.0288,  0.0474,  0.0587,  0.0711,  0.0809,  0.0835,  0.0748,  0.0569,  0.0381,  0.0293,  0.0284,  0.0493,  0.0609,  0.0793, 
                # 0.093     = [  0.0317,  0.0447,  0.0588,  0.0807,  0.0923,  0.1050,  0.1151,  0.1183,  0.1093,  0.0900,  0.0687,  0.0549,  0.0376,  0.0370,  0.0441,  0.0567, 
                # 0.107     = [  0.0593,  0.0775,  0.0921,  0.1148,  0.1265,  0.1392,  0.1496,  0.1533,  0.1441,  0.1237,  0.1012,  0.0864,  0.0636,  0.0463,  0.0458,  0.0475, 
                # 0.120     = [  0.0923,  0.1121,  0.1261,  0.1494,  0.1611,  0.1738,  0.1843,  0.1886,  0.1792,  0.1577,  0.1344,  0.1193,  0.0944,  0.0699,  0.0633,  0.0562, 
                # 0.133     = [  0.1266,  0.1475,  0.1606,  0.1843,  0.1960,  0.2086,  0.2193,  0.2241,  0.2146,  0.1921,  0.1681,  0.1529,  0.1267,  0.0995,  0.0898,  0.0784, 
                # ------------                                                   
                # dim 100      actual_init_angle_score 0.553       best_score 0.0165                                 
                # step_1     = [  0.138,   0.148,   0.159,   0.170,   0.181,   0.192,   0.203,   0.214,   0.225,   0.236,   0.247,   0.258,   0.269,   0.280,   0.291,  
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [  0.2209,  0.2034,  0.1897,  0.1661,  0.1585,  0.1495,  0.1442,  0.1450,  0.1506,  0.1606,  0.1745,  0.2015,  0.2155,  0.2343,  0.2609,  
                # 0.013     = [  0.1892,  0.1726,  0.1586,  0.1356,  0.1270,  0.1180,  0.1120,  0.1124,  0.1185,  0.1294,  0.1437,  0.1727,  0.1864,  0.2060,  0.2327,  
                # 0.027     = [  0.1572,  0.1415,  0.1272,  0.1048,  0.0950,  0.0863,  0.0794,  0.0795,  0.0861,  0.0980,  0.1125,  0.1435,  0.1570,  0.1775,  0.2042,  
                # 0.040     = [  0.1249,  0.1101,  0.0955,  0.0738,  0.0630,  0.0544,  0.0468,  0.0466,  0.0536,  0.0664,  0.0813,  0.1142,  0.1274,  0.1487,  0.1754,  
                # 0.053     = [  0.0926,  0.0788,  0.0639,  0.0432,  0.0319,  0.0241,  0.0165,  0.0165,  0.0226,  0.0357,  0.0505,  0.0849,  0.0979,  0.1198,  0.1466,  
                # 0.067     = [  0.0610,  0.0485,  0.0344,  0.0189,  0.0180,  0.0205,  0.0248,  0.0255,  0.0202,  0.0190,  0.0261,  0.0567,  0.0693,  0.0915,  0.1179,  
                # 0.080     = [  0.0335,  0.0276,  0.0235,  0.0303,  0.0418,  0.0483,  0.0572,  0.0581,  0.0507,  0.0381,  0.0298,  0.0348,  0.0458,  0.0652,  0.0903,  
                # 0.093     = [  0.0291,  0.0357,  0.0446,  0.0600,  0.0746,  0.0812,  0.0913,  0.0924,  0.0843,  0.0693,  0.0564,  0.0338,  0.0383,  0.0471,  0.0663,  
                # 0.107     = [  0.0537,  0.0619,  0.0761,  0.0925,  0.1086,  0.1149,  0.1258,  0.1271,  0.1186,  0.1023,  0.0884,  0.0541,  0.0512,  0.0456,  0.0524,  
                # 0.120     = [  0.0859,  0.0937,  0.1096,  0.1258,  0.1432,  0.1491,  0.1607,  0.1622,  0.1533,  0.1360,  0.1217,  0.0826,  0.0764,  0.0606,  0.0547,  
                # 0.133     = [  0.1200,  0.1271,  0.1439,  0.1596,  0.1781,  0.1835,  0.1959,  0.1975,  0.1883,  0.1701,  0.1555,  0.1135,  0.1064,  0.0856,  0.0718,  
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.598       best_score 0.0172                                 
                # step_1     = [ 0.148,   0.159,   0.170,   0.181,   0.192,   0.203,   0.214,   0.225,   0.236,   0.247,   0.258,   0.269,   0.280,   0.291,  
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [  0.2574,  0.2374,  0.2135,  0.2008,  0.1852,  0.1783,  0.1708,  0.1701,  0.1736,  0.1802,  0.1934,  0.2070,  0.2169,  0.2334, 
                # 0.013     = [  0.2260,  0.2064,  0.1830,  0.1706,  0.1550,  0.1475,  0.1390,  0.1374,  0.1417,  0.1484,  0.1623,  0.1771,  0.1876,  0.2040, 
                # 0.027     = [  0.1943,  0.1749,  0.1522,  0.1400,  0.1245,  0.1163,  0.1069,  0.1045,  0.1095,  0.1164,  0.1310,  0.1468,  0.1581,  0.1743, 
                # 0.040     = [  0.1623,  0.1432,  0.1211,  0.1092,  0.0938,  0.0850,  0.0747,  0.0714,  0.0771,  0.0842,  0.0995,  0.1165,  0.1283,  0.1444, 
                # 0.053     = [  0.1302,  0.1114,  0.0900,  0.0784,  0.0632,  0.0540,  0.0430,  0.0390,  0.0452,  0.0524,  0.0682,  0.0863,  0.0987,  0.1146, 
                # 0.067     = [  0.0984,  0.0800,  0.0595,  0.0488,  0.0347,  0.0267,  0.0190,  0.0172,  0.0186,  0.0249,  0.0389,  0.0572,  0.0699,  0.0855, 
                # 0.080     = [  0.0680,  0.0507,  0.0336,  0.0285,  0.0230,  0.0257,  0.0327,  0.0368,  0.0300,  0.0278,  0.0257,  0.0352,  0.0451,  0.0598, 
                # 0.093     = [  0.0437,  0.0322,  0.0307,  0.0365,  0.0430,  0.0521,  0.0640,  0.0697,  0.0614,  0.0553,  0.0435,  0.0346,  0.0364,  0.0459, 
                # 0.107     = [  0.0394,  0.0420,  0.0532,  0.0623,  0.0734,  0.0841,  0.0975,  0.1042,  0.0950,  0.0881,  0.0731,  0.0561,  0.0502,  0.0493, 
                # 0.120     = [  0.0586,  0.0695,  0.0839,  0.0931,  0.1058,  0.1173,  0.1318,  0.1394,  0.1293,  0.1220,  0.1058,  0.0857,  0.0755,  0.0684, 
                # 0.133     = [  0.0882,  0.1018,  0.1169,  0.1259,  0.1390,  0.1511,  0.1665,  0.1748,  0.1639,  0.1564,  0.1396,  0.1176,  0.1054,  0.0960, 
                # 0.147     = [  0.1212,  0.1357,  0.1508,  0.1595,  0.1727,  0.1853,  0.2014,  0.2106,  0.1989,  0.1912,  0.1738,  0.1504,  0.1370,  0.1269, 
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.650       best_score 0.0240                                 
                # step_1     = [   0.159,   0.170,   0.181,   0.192,   0.203,   0.214,   0.225,   0.236,   0.247,   0.258,   0.269,   0.280,   0.291,   0.302,   0.312,   0.323,  
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [   0.2887,  0.2651,  0.2499,  0.2323,  0.2203,  0.2132,  0.2052,  0.2013,  0.2018,  0.2036,  0.2111,  0.2203,  0.2329,  0.2460,  0.2680,  0.2805,  
                # 0.013     = [   0.2580,  0.2349,  0.2196,  0.2021,  0.1899,  0.1825,  0.1742,  0.1700,  0.1701,  0.1720,  0.1803,  0.1895,  0.2022,  0.2163,  0.2393,  0.2523,  
                # 0.027     = [   0.2270,  0.2044,  0.1890,  0.1716,  0.1593,  0.1514,  0.1429,  0.1384,  0.1380,  0.1401,  0.1492,  0.1584,  0.1712,  0.1863,  0.2104,  0.2237,  
                # 0.040     = [   0.1957,  0.1737,  0.1580,  0.1408,  0.1284,  0.1202,  0.1113,  0.1066,  0.1058,  0.1080,  0.1179,  0.1271,  0.1400,  0.1562,  0.1813,  0.1951,  
                # 0.053     = [   0.1643,  0.1428,  0.1270,  0.1100,  0.0976,  0.0890,  0.0800,  0.0750,  0.0737,  0.0760,  0.0867,  0.0959,  0.1089,  0.1261,  0.1521,  0.1664,  
                # 0.067     = [   0.1331,  0.1121,  0.0962,  0.0796,  0.0674,  0.0588,  0.0499,  0.0447,  0.0430,  0.0453,  0.0564,  0.0656,  0.0785,  0.0965,  0.1233,  0.1380,  
                # 0.080     = [   0.1024,  0.0823,  0.0667,  0.0516,  0.0409,  0.0346,  0.0280,  0.0255,  0.0240,  0.0244,  0.0314,  0.0410,  0.0513,  0.0687,  0.0956,  0.1105,  
                # 0.093     = [   0.0738,  0.0559,  0.0429,  0.0347,  0.0319,  0.0339,  0.0359,  0.0376,  0.0389,  0.0368,  0.0319,  0.0359,  0.0373,  0.0484,  0.0708,  0.0855,  
                # 0.107     = [   0.0524,  0.0419,  0.0382,  0.0426,  0.0495,  0.0566,  0.0634,  0.0667,  0.0694,  0.0666,  0.0558,  0.0535,  0.0476,  0.0458,  0.0544,  0.0670,  
                # 0.120     = [   0.0485,  0.0500,  0.0572,  0.0679,  0.0784,  0.0874,  0.0956,  0.0994,  0.1027,  0.0997,  0.0866,  0.0820,  0.0730,  0.0614,  0.0548,  0.0613,  
                # 0.133     = [   0.0647,  0.0736,  0.0863,  0.0989,  0.1105,  0.1204,  0.1291,  0.1332,  0.1369,  0.1337,  0.1193,  0.1139,  0.1038,  0.0872,  0.0713,  0.0706,  
                # 0.147     = [   0.0917,  0.1036,  0.1186,  0.1318,  0.1439,  0.1544,  0.1633,  0.1675,  0.1717,  0.1682,  0.1528,  0.1472,  0.1365,  0.1174,  0.0966,  0.0906,  
                # 0.160     = [   0.1231,  0.1361,  0.1522,  0.1655,  0.1780,  0.1889,  0.1980,  0.2021,  0.2068,  0.2031,  0.1867,  0.1812,  0.1703,  0.1494,  0.1258,  0.1167,  
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.704       best_score 0.0310                                 
                # step_1     = [    0.192,   0.203,   0.214,   0.225,   0.236,   0.247,   0.258,   0.269,   0.280,   0.291,   0.302,   0.312,   0.323,   0.334,   0.345,  
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [    0.2876,  0.2748,  0.2583,  0.2473,  0.2417,  0.2333,  0.2334,  0.2317,  0.2374,  0.2463,  0.2501,  0.2650,  0.2823,  0.2921,  0.3072,  
                # 0.013     = [    0.2578,  0.2453,  0.2287,  0.2174,  0.2116,  0.2026,  0.2024,  0.2004,  0.2059,  0.2154,  0.2192,  0.2349,  0.2527,  0.2633,  0.2787,  
                # 0.027     = [    0.2276,  0.2154,  0.1987,  0.1873,  0.1812,  0.1715,  0.1711,  0.1689,  0.1740,  0.1843,  0.1880,  0.2045,  0.2230,  0.2341,  0.2500,  
                # 0.040     = [    0.1972,  0.1853,  0.1686,  0.1570,  0.1506,  0.1403,  0.1397,  0.1372,  0.1420,  0.1529,  0.1566,  0.1739,  0.1930,  0.2049,  0.2211,  
                # 0.053     = [    0.1667,  0.1551,  0.1383,  0.1266,  0.1199,  0.1092,  0.1083,  0.1055,  0.1100,  0.1216,  0.1252,  0.1434,  0.1631,  0.1756,  0.1923,  
                # 0.067     = [    0.1362,  0.1250,  0.1084,  0.0966,  0.0897,  0.0786,  0.0774,  0.0743,  0.0785,  0.0906,  0.0943,  0.1131,  0.1335,  0.1466,  0.1636,  
                # 0.080     = [    0.1065,  0.0957,  0.0795,  0.0682,  0.0613,  0.0504,  0.0490,  0.0459,  0.0497,  0.0614,  0.0651,  0.0841,  0.1048,  0.1184,  0.1358,  
                # 0.093     = [    0.0787,  0.0691,  0.0548,  0.0455,  0.0415,  0.0340,  0.0325,  0.0310,  0.0336,  0.0396,  0.0429,  0.0588,  0.0786,  0.0922,  0.1096,  
                # 0.107     = [    0.0575,  0.0510,  0.0435,  0.0406,  0.0420,  0.0441,  0.0438,  0.0451,  0.0451,  0.0403,  0.0414,  0.0457,  0.0594,  0.0714,  0.0874,  
                # 0.120     = [    0.0516,  0.0504,  0.0534,  0.0575,  0.0620,  0.0705,  0.0712,  0.0738,  0.0725,  0.0614,  0.0615,  0.0533,  0.0556,  0.0624,  0.0741,  
                # 0.133     = [    0.0645,  0.0676,  0.0772,  0.0851,  0.0908,  0.1018,  0.1030,  0.1062,  0.1047,  0.0909,  0.0906,  0.0758,  0.0690,  0.0688,  0.0737,  
                # 0.147     = [    0.0892,  0.0944,  0.1068,  0.1163,  0.1225,  0.1348,  0.1363,  0.1398,  0.1385,  0.1232,  0.1228,  0.1048,  0.0928,  0.0874,  0.0860,  
                # 0.160     = [    0.1192,  0.1252,  0.1387,  0.1489,  0.1556,  0.1687,  0.1703,  0.1741,  0.1731,  0.1567,  0.1563,  0.1364,  0.1216,  0.1128,  0.1071,  
                # 0.173     = [    0.1517,  0.1578,  0.1719,  0.1824,  0.1893,  0.2030,  0.2048,  0.2088,  0.2081,  0.1907,  0.1904,  0.1692,  0.1528,  0.1418,  0.1334,  
                # 0.187     = [    0.1853,  0.1914,  0.2057,  0.2165,  0.2235,  0.2377,  0.2395,  0.2438,  0.2433,  0.2252,  0.2251,  0.2028,  0.1852,  0.1728,  0.1627,  
                # 0.200     = [    0.2197,  0.2256,  0.2400,  0.2508,  0.2580,  0.2727,  0.2746,  0.2790,  0.2789,  0.2599,  0.2600,  0.2368,  0.2185,  0.2049,  0.1938,  
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.755       best_score 0.0420                                 
                # step_1     = [    0.192,   0.203,   0.214,   0.225,   0.236,   0.247,   0.258,   0.269,   0.280,   0.291,   0.302,   0.312,   0.323,   0.334,   0.345,   0.356,   0.367,   0.378,   0.389,   0.400]
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [    0.3373,  0.3284,  0.3079,  0.3031,  0.2868,  0.2790,  0.2735,  0.2703,  0.2688,  0.2682,  0.2746,  0.2796,  0.2877,  0.3058,  0.3126,  0.3305,  0.3384,  0.3623,  0.3714,  0.3885]
                # 0.013     = [    0.3085,  0.2993,  0.2786,  0.2742,  0.2580,  0.2491,  0.2438,  0.2401,  0.2386,  0.2372,  0.2440,  0.2486,  0.2576,  0.2762,  0.2836,  0.3021,  0.3105,  0.3352,  0.3445,  0.3622]
                # 0.027     = [    0.2795,  0.2699,  0.2489,  0.2451,  0.2288,  0.2190,  0.2139,  0.2096,  0.2080,  0.2059,  0.2130,  0.2174,  0.2272,  0.2464,  0.2544,  0.2735,  0.2824,  0.3079,  0.3175,  0.3358]
                # 0.040     = [    0.2501,  0.2402,  0.2190,  0.2156,  0.1995,  0.1886,  0.1837,  0.1789,  0.1773,  0.1744,  0.1819,  0.1859,  0.1966,  0.2165,  0.2250,  0.2448,  0.2541,  0.2805,  0.2903,  0.3094]
                # 0.053     = [    0.2206,  0.2103,  0.1889,  0.1861,  0.1700,  0.1582,  0.1535,  0.1482,  0.1465,  0.1428,  0.1507,  0.1544,  0.1659,  0.1865,  0.1956,  0.2160,  0.2259,  0.2530,  0.2632,  0.2829]
                # 0.067     = [    0.1911,  0.1805,  0.1589,  0.1566,  0.1407,  0.1281,  0.1236,  0.1177,  0.1160,  0.1115,  0.1198,  0.1232,  0.1355,  0.1566,  0.1664,  0.1874,  0.1978,  0.2257,  0.2362,  0.2566]
                # 0.080     = [    0.1619,  0.1509,  0.1294,  0.1276,  0.1120,  0.0988,  0.0947,  0.0883,  0.0865,  0.0814,  0.0897,  0.0928,  0.1057,  0.1274,  0.1378,  0.1594,  0.1703,  0.1988,  0.2096,  0.2308]
                # 0.093     = [    0.1333,  0.1223,  0.1012,  0.0999,  0.0851,  0.0722,  0.0686,  0.0621,  0.0600,  0.0550,  0.0627,  0.0653,  0.0780,  0.0998,  0.1107,  0.1324,  0.1440,  0.1728,  0.1839,  0.2057]
                # 0.107     = [    0.1066,  0.0959,  0.0765,  0.0757,  0.0632,  0.0536,  0.0516,  0.0465,  0.0440,  0.0420,  0.0458,  0.0471,  0.0566,  0.0761,  0.0871,  0.1080,  0.1199,  0.1483,  0.1597,  0.1821]
                # 0.120     = [    0.0838,  0.0749,  0.0608,  0.0605,  0.0535,  0.0523,  0.0526,  0.0512,  0.0496,  0.0526,  0.0496,  0.0498,  0.0508,  0.0621,  0.0718,  0.0889,  0.1003,  0.1267,  0.1382,  0.1606]
                # 0.133     = [    0.0699,  0.0653,  0.0615,  0.0609,  0.0614,  0.0690,  0.0702,  0.0728,  0.0720,  0.0778,  0.0712,  0.0711,  0.0640,  0.0642,  0.0696,  0.0796,  0.0891,  0.1103,  0.1212,  0.1427]
                # 0.147     = [    0.0704,  0.0717,  0.0781,  0.0766,  0.0824,  0.0954,  0.0966,  0.1017,  0.1012,  0.1085,  0.1003,  0.1002,  0.0883,  0.0809,  0.0808,  0.0828,  0.0891,  0.1019,  0.1115,  0.1304]
                # 0.160     = [    0.0846,  0.0911,  0.1037,  0.1013,  0.1099,  0.1259,  0.1269,  0.1333,  0.1329,  0.1414,  0.1323,  0.1324,  0.1177,  0.1060,  0.1014,  0.0973,  0.1001,  0.1033,  0.1106,  0.1253]
                # 0.173     = [    0.1078,  0.1177,  0.1336,  0.1305,  0.1403,  0.1582,  0.1590,  0.1662,  0.1659,  0.1753,  0.1656,  0.1659,  0.1494,  0.1351,  0.1277,  0.1193,  0.1193,  0.1141,  0.1187,  0.1282]
                # 0.187     = [    0.1358,  0.1480,  0.1657,  0.1619,  0.1723,  0.1916,  0.1920,  0.2000,  0.1995,  0.2097,  0.1996,  0.2002,  0.1822,  0.1664,  0.1570,  0.1458,  0.1438,  0.1320,  0.1342,  0.1387]
                # 0.200     = [    0.1665,  0.1802,  0.1991,  0.1945,  0.2051,  0.2257,  0.2257,  0.2342,  0.2336,  0.2446,  0.2341,  0.2350,  0.2158,  0.1988,  0.1882,  0.1751,  0.1715,  0.1549,  0.1552,  0.1554]
                # ------------                                 
                # dim 100      actual_init_angle_score 0.753       best_score 0.0414                                 
                # step_1     = [   0.203,   0.214,   0.225,   0.236,   0.247,   0.258,   0.269,   0.280,   0.291,   0.302,   0.312,   0.323,   0.334,   0.345,   0.356,   0.367,   0.378,   0.389,   0.400]
                # vvvv cap_to__step_2                                                                                                                
                # 0.000     = [   0.3292,  0.3122,  0.2962,  0.2863,  0.2739,  0.2726,  0.2701,  0.2682,  0.2695,  0.2740,  0.2795,  0.2905,  0.2987,  0.3177,  0.3255,  0.3378,  0.3508,  0.3763,  0.3876]
                # 0.013     = [   0.3001,  0.2836,  0.2673,  0.2571,  0.2443,  0.2422,  0.2398,  0.2371,  0.2383,  0.2431,  0.2489,  0.2601,  0.2690,  0.2890,  0.2970,  0.3096,  0.3228,  0.3497,  0.3615]
                # 0.027     = [   0.2707,  0.2547,  0.2382,  0.2277,  0.2144,  0.2115,  0.2093,  0.2057,  0.2069,  0.2120,  0.2181,  0.2295,  0.2391,  0.2600,  0.2682,  0.2812,  0.2945,  0.3229,  0.3352]
                # 0.040     = [   0.2409,  0.2256,  0.2088,  0.1980,  0.1843,  0.1807,  0.1785,  0.1741,  0.1752,  0.1807,  0.1870,  0.1986,  0.2089,  0.2308,  0.2394,  0.2526,  0.2660,  0.2960,  0.3089]
                # 0.053     = [   0.2111,  0.1963,  0.1793,  0.1682,  0.1541,  0.1498,  0.1477,  0.1425,  0.1435,  0.1494,  0.1559,  0.1677,  0.1788,  0.2016,  0.2105,  0.2239,  0.2376,  0.2691,  0.2825]
                # 0.067     = [   0.1812,  0.1671,  0.1499,  0.1386,  0.1242,  0.1192,  0.1171,  0.1111,  0.1120,  0.1183,  0.1250,  0.1370,  0.1488,  0.1726,  0.1818,  0.1955,  0.2094,  0.2423,  0.2563]
                # 0.080     = [   0.1516,  0.1382,  0.1210,  0.1097,  0.0952,  0.0897,  0.0875,  0.0810,  0.0817,  0.0880,  0.0949,  0.1070,  0.1196,  0.1441,  0.1537,  0.1675,  0.1818,  0.2159,  0.2306]
                # 0.093     = [   0.1231,  0.1104,  0.0936,  0.0828,  0.0688,  0.0635,  0.0611,  0.0546,  0.0551,  0.0608,  0.0674,  0.0791,  0.0921,  0.1170,  0.1270,  0.1407,  0.1552,  0.1904,  0.2056]
                # 0.107     = [   0.0967,  0.0852,  0.0704,  0.0615,  0.0508,  0.0477,  0.0453,  0.0418,  0.0414,  0.0436,  0.0484,  0.0574,  0.0694,  0.0928,  0.1031,  0.1162,  0.1308,  0.1662,  0.1819]
                # 0.120     = [   0.0759,  0.0667,  0.0573,  0.0537,  0.0510,  0.0527,  0.0505,  0.0530,  0.0520,  0.0489,  0.0488,  0.0513,  0.0584,  0.0751,  0.0852,  0.0965,  0.1103,  0.1445,  0.1605]
                # 0.133     = [   0.0664,  0.0614,  0.0607,  0.0637,  0.0690,  0.0743,  0.0726,  0.0790,  0.0780,  0.0721,  0.0680,  0.0646,  0.0647,  0.0696,  0.0780,  0.0857,  0.0971,  0.1269,  0.1428]
                # 0.147     = [   0.0723,  0.0719,  0.0785,  0.0861,  0.0959,  0.1032,  0.1016,  0.1101,  0.1092,  0.1019,  0.0961,  0.0895,  0.0844,  0.0783,  0.0836,  0.0865,  0.0943,  0.1158,  0.1304]
                # 0.160     = [   0.0910,  0.0934,  0.1044,  0.1143,  0.1264,  0.1351,  0.1334,  0.1433,  0.1424,  0.1343,  0.1276,  0.1193,  0.1112,  0.0977,  0.0998,  0.0982,  0.1027,  0.1132,  0.1254]
                # 0.173     = [   0.1171,  0.1208,  0.1340,  0.1453,  0.1587,  0.1684,  0.1665,  0.1774,  0.1766,  0.1679,  0.1605,  0.1514,  0.1414,  0.1232,  0.1229,  0.1180,  0.1199,  0.1196,  0.1286]
                # 0.187     = [   0.1470,  0.1511,  0.1656,  0.1778,  0.1919,  0.2025,  0.2003,  0.2122,  0.2114,  0.2021,  0.1943,  0.1846,  0.1733,  0.1521,  0.1501,  0.1429,  0.1431,  0.1336,  0.1394]
                # 0.200     = [   0.1789,  0.1830,  0.1983,  0.2111,  0.2258,  0.2371,  0.2346,  0.2473,  0.2465,  0.2368,  0.2285,  0.2185,  0.2062,  0.1828,  0.1798,  0.1712,  0.1700,  0.1534,  0.1563]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.804       best_score 0.0536                                 
                # step_1     = [ 0.225,   0.236,   0.247,   0.258,   0.269,   0.280,   0.291,   0.302,   0.312,   0.323,   0.334,   0.345,   0.356,   0.367,   0.378,   0.389,   0.400]
                # vvvv cap_to__step_2                                                                                                                
                # 0.040     = [ 0.2638,  0.2552,  0.2435,  0.2250,  0.2231,  0.2161,  0.2139,  0.2134,  0.2123,  0.2212,  0.2284,  0.2374,  0.2485,  0.2594,  0.2742,  0.2975,  0.3082]
                # 0.053     = [ 0.2351,  0.2265,  0.2147,  0.1953,  0.1934,  0.1861,  0.1835,  0.1827,  0.1813,  0.1905,  0.1982,  0.2076,  0.2186,  0.2300,  0.2459,  0.2697,  0.2809]
                # 0.067     = [ 0.2065,  0.1978,  0.1859,  0.1656,  0.1638,  0.1563,  0.1532,  0.1521,  0.1504,  0.1600,  0.1681,  0.1779,  0.1888,  0.2007,  0.2177,  0.2422,  0.2538]
                # 0.080     = [ 0.1781,  0.1695,  0.1575,  0.1365,  0.1348,  0.1271,  0.1235,  0.1222,  0.1201,  0.1299,  0.1385,  0.1487,  0.1595,  0.1719,  0.1900,  0.2149,  0.2271]
                # 0.093     = [ 0.1504,  0.1418,  0.1300,  0.1086,  0.1071,  0.0994,  0.0954,  0.0938,  0.0913,  0.1012,  0.1100,  0.1206,  0.1312,  0.1441,  0.1631,  0.1885,  0.2011]
                # 0.107     = [ 0.1242,  0.1158,  0.1046,  0.0840,  0.0826,  0.0755,  0.0714,  0.0697,  0.0668,  0.0758,  0.0843,  0.0948,  0.1052,  0.1183,  0.1378,  0.1635,  0.1765]
                # 0.120     = [ 0.1010,  0.0933,  0.0840,  0.0670,  0.0660,  0.0611,  0.0578,  0.0565,  0.0536,  0.0592,  0.0656,  0.0745,  0.0844,  0.0967,  0.1156,  0.1408,  0.1542]
                # 0.133     = [ 0.0842,  0.0782,  0.0730,  0.0643,  0.0640,  0.0629,  0.0616,  0.0613,  0.0599,  0.0594,  0.0608,  0.0656,  0.0739,  0.0834,  0.0992,  0.1223,  0.1356]
                # 0.147     = [ 0.0782,  0.0750,  0.0754,  0.0769,  0.0776,  0.0795,  0.0803,  0.0811,  0.0813,  0.0761,  0.0723,  0.0723,  0.0777,  0.0824,  0.0922,  0.1105,  0.1230]
                # 0.160     = [ 0.0851,  0.0852,  0.0899,  0.0996,  0.1009,  0.1045,  0.1070,  0.1085,  0.1097,  0.1021,  0.0948,  0.0911,  0.0938,  0.0935,  0.0960,  0.1077,  0.1187]
                # 0.173     = [ 0.1024,  0.1051,  0.1126,  0.1276,  0.1292,  0.1338,  0.1373,  0.1393,  0.1412,  0.1322,  0.1229,  0.1168,  0.1175,  0.1136,  0.1096,  0.1145,  0.1233]
                # 0.187     = [ 0.1265,  0.1309,  0.1398,  0.1584,  0.1600,  0.1651,  0.1693,  0.1718,  0.1742,  0.1642,  0.1537,  0.1461,  0.1456,  0.1391,  0.1302,  0.1294,  0.1360]
                # 0.200     = [ 0.1544,  0.1599,  0.1697,  0.1906,  0.1922,  0.1976,  0.2025,  0.2052,  0.2081,  0.1974,  0.1858,  0.1772,  0.1762,  0.1679,  0.1554,  0.1502,  0.1548]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.853       best_score 0.0725                                 
                # step_1     = [   0.236,   0.247,   0.258,   0.269,   0.280,   0.291,   0.302,   0.312,   0.323,   0.334,   0.345,   0.356,   0.367,   0.378,   0.389,   0.400]
                # vvvv cap_to__step_2                                                                                                                
                # 0.067     = [   0.2575,  0.2381,  0.2294,  0.2235,  0.2085,  0.2035,  0.1993,  0.1931,  0.1949,  0.1984,  0.2026,  0.2128,  0.2201,  0.2312,  0.2512,  0.2610]
                # 0.080     = [   0.2296,  0.2101,  0.2013,  0.1954,  0.1798,  0.1749,  0.1700,  0.1636,  0.1653,  0.1687,  0.1729,  0.1834,  0.1908,  0.2021,  0.2232,  0.2330]
                # 0.093     = [   0.2021,  0.1827,  0.1738,  0.1680,  0.1520,  0.1472,  0.1417,  0.1349,  0.1366,  0.1398,  0.1439,  0.1546,  0.1621,  0.1735,  0.1958,  0.2057]
                # 0.107     = [   0.1754,  0.1563,  0.1475,  0.1419,  0.1259,  0.1213,  0.1154,  0.1082,  0.1100,  0.1127,  0.1166,  0.1273,  0.1350,  0.1464,  0.1695,  0.1795]
                # 0.120     = [   0.1502,  0.1320,  0.1236,  0.1185,  0.1034,  0.0990,  0.0932,  0.0860,  0.0877,  0.0894,  0.0928,  0.1031,  0.1107,  0.1219,  0.1453,  0.1551]
                # 0.133     = [   0.1279,  0.1116,  0.1041,  0.0999,  0.0880,  0.0838,  0.0794,  0.0732,  0.0745,  0.0741,  0.0765,  0.0852,  0.0921,  0.1024,  0.1247,  0.1339]
                # 0.147     = [   0.1106,  0.0980,  0.0923,  0.0897,  0.0837,  0.0803,  0.0787,  0.0744,  0.0755,  0.0725,  0.0732,  0.0782,  0.0835,  0.0917,  0.1101,  0.1182]
                # 0.160     = [   0.1013,  0.0943,  0.0914,  0.0908,  0.0919,  0.0896,  0.0912,  0.0889,  0.0901,  0.0853,  0.0840,  0.0849,  0.0877,  0.0929,  0.1045,  0.1106]
                # 0.173     = [   0.1021,  0.1015,  0.1018,  0.1028,  0.1102,  0.1086,  0.1128,  0.1120,  0.1135,  0.1076,  0.1049,  0.1026,  0.1034,  0.1054,  0.1090,  0.1127]
                # 0.187     = [   0.1129,  0.1179,  0.1209,  0.1228,  0.1349,  0.1337,  0.1396,  0.1399,  0.1416,  0.1352,  0.1314,  0.1272,  0.1264,  0.1261,  0.1224,  0.1241]
                # 0.200     = [   0.1315,  0.1406,  0.1456,  0.1479,  0.1633,  0.1621,  0.1694,  0.1704,  0.1723,  0.1655,  0.1612,  0.1556,  0.1539,  0.1518,  0.1423,  0.1425]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 0.902       best_score 0.0921                                 
                # step_1     = [  0.236,   0.247,   0.258,   0.269,   0.280,   0.291,   0.302,   0.312,   0.323,   0.334,   0.345,   0.356,   0.367,   0.378,   0.389,   0.400]
                # vvvv cap_to__step_2                                                                                                                
                # 0.067     = [  0.3180,  0.3016,  0.2978,  0.2753,  0.2655,  0.2599,  0.2445,  0.2506,  0.2402,  0.2450,  0.2391,  0.2439,  0.2524,  0.2530,  0.2655,  0.2837]
                # 0.080     = [  0.2912,  0.2742,  0.2710,  0.2477,  0.2378,  0.2326,  0.2163,  0.2228,  0.2117,  0.2163,  0.2099,  0.2146,  0.2232,  0.2243,  0.2368,  0.2558]
                # 0.093     = [  0.2646,  0.2471,  0.2444,  0.2206,  0.2106,  0.2059,  0.1887,  0.1955,  0.1838,  0.1883,  0.1812,  0.1859,  0.1945,  0.1960,  0.2086,  0.2283]
                # 0.107     = [  0.2385,  0.2206,  0.2185,  0.1943,  0.1843,  0.1801,  0.1623,  0.1692,  0.1571,  0.1613,  0.1538,  0.1583,  0.1668,  0.1687,  0.1811,  0.2016]
                # 0.120     = [  0.2133,  0.1952,  0.1935,  0.1695,  0.1596,  0.1559,  0.1381,  0.1448,  0.1326,  0.1364,  0.1286,  0.1328,  0.1410,  0.1432,  0.1553,  0.1761]
                # 0.133     = [  0.1895,  0.1717,  0.1702,  0.1471,  0.1376,  0.1346,  0.1179,  0.1238,  0.1122,  0.1153,  0.1076,  0.1112,  0.1186,  0.1211,  0.1322,  0.1530]
                # 0.147     = [  0.1680,  0.1514,  0.1497,  0.1290,  0.1203,  0.1181,  0.1043,  0.1086,  0.0987,  0.1007,  0.0943,  0.0965,  0.1023,  0.1048,  0.1140,  0.1335]
                # 0.160     = [  0.1501,  0.1358,  0.1335,  0.1174,  0.1101,  0.1089,  0.1002,  0.1019,  0.0955,  0.0962,  0.0921,  0.0925,  0.0957,  0.0976,  0.1038,  0.1198]
                # 0.173     = [  0.1375,  0.1270,  0.1238,  0.1145,  0.1093,  0.1088,  0.1068,  0.1053,  0.1035,  0.1030,  0.1015,  0.1004,  0.1007,  0.1014,  0.1037,  0.1143]
                # 0.187     = [  0.1319,  0.1264,  0.1221,  0.1208,  0.1181,  0.1180,  0.1223,  0.1179,  0.1206,  0.1193,  0.1200,  0.1180,  0.1159,  0.1150,  0.1138,  0.1179]
                # 0.200     = [  0.1343,  0.1344,  0.1286,  0.1350,  0.1348,  0.1346,  0.1441,  0.1375,  0.1437,  0.1421,  0.1443,  0.1419,  0.1382,  0.1358,  0.1321,  0.1299]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.056       best_score 0.1786                                 
                # step_1     = [   0.338,   0.344,   0.350,   0.356,   0.363,   0.369,   0.375,   0.381,   0.387,   0.394,   0.400,   0.406,   0.413,   0.419,   0.425,   0.431,   0.438,   0.444,   0.450,  
                # vvvv cap_to__step_2                                                                                                                
                # 0.113     = [   0.3162,  0.3156,  0.3104,  0.3029,  0.3022,  0.3050,  0.3061,  0.2980,  0.2954,  0.3077,  0.2993,  0.2986,  0.3075,  0.3089,  0.3114,  0.3135,  0.3197,  0.3211,  0.3273,  
                # 0.127     = [   0.2933,  0.2925,  0.2874,  0.2796,  0.2785,  0.2814,  0.2824,  0.2738,  0.2710,  0.2836,  0.2746,  0.2741,  0.2832,  0.2844,  0.2867,  0.2887,  0.2954,  0.2965,  0.3030,  
                # 0.140     = [   0.2716,  0.2707,  0.2657,  0.2575,  0.2561,  0.2591,  0.2600,  0.2511,  0.2480,  0.2607,  0.2512,  0.2509,  0.2600,  0.2612,  0.2632,  0.2650,  0.2722,  0.2730,  0.2798,  
                # 0.153     = [   0.2516,  0.2505,  0.2457,  0.2372,  0.2356,  0.2386,  0.2392,  0.2303,  0.2270,  0.2394,  0.2297,  0.2294,  0.2385,  0.2397,  0.2413,  0.2430,  0.2503,  0.2512,  0.2581,  
                # 0.167     = [   0.2340,  0.2325,  0.2280,  0.2194,  0.2176,  0.2205,  0.2207,  0.2122,  0.2088,  0.2206,  0.2106,  0.2105,  0.2192,  0.2206,  0.2217,  0.2234,  0.2304,  0.2317,  0.2384,  
                # 0.180     = [   0.2194,  0.2176,  0.2134,  0.2049,  0.2031,  0.2058,  0.2054,  0.1978,  0.1943,  0.2048,  0.1950,  0.1952,  0.2032,  0.2048,  0.2053,  0.2069,  0.2131,  0.2152,  0.2215,  
                # 0.193     = [   0.2087,  0.2066,  0.2028,  0.1945,  0.1932,  0.1952,  0.1942,  0.1882,  0.1847,  0.1931,  0.1840,  0.1843,  0.1912,  0.1932,  0.1930,  0.1947,  0.1994,  0.2027,  0.2082,  
                # 0.207     = [   0.2025,  0.2002,  0.1969,  0.1892,  0.1885,  0.1897,  0.1879,  0.1843,  0.1809,  0.1865,  0.1786,  0.1789,  0.1844,  0.1868,  0.1858,  0.1876,  0.1901,  0.1950,  0.1994,  
                # 0.220     = [   0.2014,  0.1990,  0.1962,  0.1895,  0.1896,  0.1897,  0.1871,  0.1864,  0.1833,  0.1856,  0.1793,  0.1797,  0.1832,  0.1862,  0.1846,  0.1864,  0.1860,  0.1929,  0.1958,  
                # 0.233     = [   0.2056,  0.2032,  0.2008,  0.1952,  0.1964,  0.1953,  0.1918,  0.1945,  0.1918,  0.1904,  0.1862,  0.1865,  0.1879,  0.1916,  0.1894,  0.1911,  0.1874,  0.1966,  0.1977,  
                # 0.247     = [   0.2149,  0.2124,  0.2103,  0.2061,  0.2084,  0.2061,  0.2019,  0.2078,  0.2057,  0.2005,  0.1987,  0.1988,  0.1979,  0.2024,  0.1998,  0.2015,  0.1944,  0.2056,  0.2052,  
                # 0.260     = [   0.2287,  0.2261,  0.2243,  0.2214,  0.2248,  0.2214,  0.2165,  0.2255,  0.2241,  0.2153,  0.2160,  0.2156,  0.2127,  0.2180,  0.2151,  0.2168,  0.2065,  0.2195,  0.2177,  
                # 0.273     = [   0.2462,  0.2435,  0.2421,  0.2403,  0.2448,  0.2404,  0.2350,  0.2467,  0.2458,  0.2339,  0.2369,  0.2361,  0.2314,  0.2375,  0.2342,  0.2360,  0.2227,  0.2374,  0.2344,  
                # 0.287     = [   0.2667,  0.2640,  0.2629,  0.2622,  0.2677,  0.2624,  0.2565,  0.2707,  0.2703,  0.2555,  0.2608,  0.2594,  0.2532,  0.2601,  0.2564,  0.2584,  0.2424,  0.2587,  0.2546,  
                # 0.300     = [   0.2898,  0.2870,  0.2861,  0.2864,  0.2929,  0.2867,  0.2804,  0.2968,  0.2969,  0.2797,  0.2868,  0.2850,  0.2775,  0.2850,  0.2810,  0.2834,  0.2649,  0.2826,  0.2775,  
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.096       best_score 0.2220                                 
                # step_1     = [  0.350,   0.356,   0.363,   0.369,   0.375,   0.381,   0.387,   0.394,   0.400,   0.406,   0.413,   0.419,   0.425,   0.431,   0.438,   0.444,   0.450,   0.456,   0.463, 
                # vvvv cap_to__step_2                                                                                                                
                # 0.140     = [   0.3274,  0.3310,  0.3260,  0.3254,  0.3185,  0.3126,  0.3142,  0.3106,  0.3130,  0.3077,  0.3093,  0.3197,  0.3199,  0.3219,  0.3267,  0.3192,  0.3276,  0.3311,  0.3355,
                # 0.153     = [   0.3073,  0.3110,  0.3061,  0.3054,  0.2984,  0.2922,  0.2938,  0.2896,  0.2921,  0.2866,  0.2882,  0.2986,  0.2985,  0.3005,  0.3050,  0.2976,  0.3057,  0.3095,  0.3136,
                # 0.167     = [   0.2888,  0.2926,  0.2879,  0.2871,  0.2801,  0.2737,  0.2751,  0.2705,  0.2729,  0.2674,  0.2690,  0.2791,  0.2787,  0.2808,  0.2850,  0.2776,  0.2855,  0.2895,  0.2933,
                # 0.180     = [   0.2726,  0.2763,  0.2718,  0.2708,  0.2640,  0.2575,  0.2589,  0.2539,  0.2560,  0.2507,  0.2523,  0.2616,  0.2611,  0.2632,  0.2671,  0.2599,  0.2673,  0.2715,  0.2748,
                # 0.193     = [   0.2591,  0.2626,  0.2584,  0.2572,  0.2508,  0.2444,  0.2456,  0.2404,  0.2419,  0.2372,  0.2389,  0.2469,  0.2462,  0.2483,  0.2517,  0.2451,  0.2518,  0.2561,  0.2588,
                # 0.207     = [   0.2489,  0.2522,  0.2482,  0.2467,  0.2409,  0.2349,  0.2358,  0.2307,  0.2314,  0.2275,  0.2293,  0.2355,  0.2347,  0.2368,  0.2398,  0.2338,  0.2397,  0.2438,  0.2459,
                # 0.220     = [   0.2426,  0.2455,  0.2418,  0.2399,  0.2350,  0.2295,  0.2301,  0.2254,  0.2250,  0.2223,  0.2244,  0.2281,  0.2273,  0.2294,  0.2317,  0.2268,  0.2315,  0.2354,  0.2368,
                # 0.233     = [   0.2406,  0.2430,  0.2397,  0.2372,  0.2335,  0.2288,  0.2287,  0.2249,  0.2233,  0.2220,  0.2244,  0.2252,  0.2244,  0.2265,  0.2283,  0.2246,  0.2279,  0.2313,  0.2319,
                # 0.247     = [   0.2430,  0.2449,  0.2419,  0.2388,  0.2365,  0.2328,  0.2319,  0.2294,  0.2264,  0.2267,  0.2295,  0.2271,  0.2264,  0.2285,  0.2297,  0.2272,  0.2291,  0.2317,  0.2314,
                # 0.260     = [   0.2498,  0.2512,  0.2484,  0.2447,  0.2439,  0.2412,  0.2396,  0.2385,  0.2343,  0.2362,  0.2394,  0.2337,  0.2332,  0.2353,  0.2359,  0.2346,  0.2351,  0.2369,  0.2356,
                # 0.273     = [   0.2608,  0.2616,  0.2588,  0.2546,  0.2554,  0.2538,  0.2514,  0.2518,  0.2464,  0.2500,  0.2535,  0.2445,  0.2444,  0.2464,  0.2466,  0.2463,  0.2455,  0.2466,  0.2443,
                # 0.287     = [   0.2754,  0.2756,  0.2728,  0.2681,  0.2705,  0.2698,  0.2668,  0.2688,  0.2622,  0.2674,  0.2714,  0.2592,  0.2595,  0.2613,  0.2613,  0.2620,  0.2600,  0.2602,  0.2570,
                # 0.300     = [   0.2932,  0.2928,  0.2900,  0.2847,  0.2887,  0.2889,  0.2853,  0.2887,  0.2811,  0.2878,  0.2922,  0.2771,  0.2778,  0.2796,  0.2793,  0.2809,  0.2778,  0.2772,  0.2732,
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.151       best_score 0.2716                                 
                # step_1     = [   0.363,   0.369,   0.375,   0.381,   0.387,   0.394,   0.400,   0.406,   0.413,   0.419,   0.425,   0.431,   0.438,   0.444,   0.450,   0.456,   0.463,   0.469,   0.475,  
                # vvvv cap_to__step_2                                                                                                                
                # 0.167     = [   0.3569,  0.3523,  0.3454,  0.3425,  0.3354,  0.3418,  0.3397,  0.3454,  0.3397,  0.3355,  0.3459,  0.3417,  0.3422,  0.3488,  0.3445,  0.3471,  0.3527,  0.3511,  0.3533,  
                # 0.180     = [   0.3404,  0.3357,  0.3285,  0.3260,  0.3187,  0.3250,  0.3225,  0.3282,  0.3224,  0.3181,  0.3282,  0.3239,  0.3244,  0.3306,  0.3267,  0.3289,  0.3342,  0.3326,  0.3350,  
                # 0.193     = [   0.3259,  0.3212,  0.3136,  0.3118,  0.3041,  0.3104,  0.3073,  0.3129,  0.3073,  0.3028,  0.3123,  0.3080,  0.3085,  0.3144,  0.3108,  0.3126,  0.3174,  0.3158,  0.3187,  
                # 0.207     = [   0.3137,  0.3090,  0.3011,  0.3001,  0.2922,  0.2982,  0.2947,  0.3000,  0.2947,  0.2900,  0.2988,  0.2946,  0.2949,  0.3006,  0.2974,  0.2986,  0.3029,  0.3014,  0.3048,  
                # 0.220     = [   0.3042,  0.2997,  0.2913,  0.2915,  0.2833,  0.2889,  0.2849,  0.2898,  0.2852,  0.2804,  0.2881,  0.2840,  0.2842,  0.2895,  0.2869,  0.2874,  0.2909,  0.2897,  0.2936,  
                # 0.233     = [   0.2979,  0.2936,  0.2847,  0.2862,  0.2779,  0.2829,  0.2784,  0.2828,  0.2790,  0.2741,  0.2804,  0.2767,  0.2768,  0.2816,  0.2796,  0.2793,  0.2820,  0.2811,  0.2858,  
                # 0.247     = [   0.2950,  0.2909,  0.2815,  0.2847,  0.2763,  0.2807,  0.2755,  0.2793,  0.2765,  0.2716,  0.2764,  0.2731,  0.2731,  0.2773,  0.2759,  0.2749,  0.2764,  0.2761,  0.2816,  
                # 0.260     = [   0.2955,  0.2919,  0.2820,  0.2871,  0.2787,  0.2823,  0.2763,  0.2794,  0.2780,  0.2732,  0.2761,  0.2733,  0.2732,  0.2769,  0.2759,  0.2743,  0.2747,  0.2750,  0.2813,  
                # 0.273     = [   0.2996,  0.2967,  0.2862,  0.2933,  0.2851,  0.2876,  0.2810,  0.2833,  0.2835,  0.2787,  0.2796,  0.2774,  0.2772,  0.2803,  0.2799,  0.2776,  0.2767,  0.2778,  0.2849,  
                # 0.287     = [   0.3072,  0.3050,  0.2941,  0.3031,  0.2950,  0.2966,  0.2893,  0.2907,  0.2926,  0.2881,  0.2868,  0.2853,  0.2850,  0.2876,  0.2876,  0.2848,  0.2826,  0.2844,  0.2924,  
                # 0.300     = [   0.3180,  0.3167,  0.3052,  0.3162,  0.3084,  0.3089,  0.3009,  0.3016,  0.3051,  0.3008,  0.2977,  0.2968,  0.2963,  0.2985,  0.2989,  0.2956,  0.2921,  0.2946,  0.3035,  
                # ------------           
                # dim 100      actual_init_angle_score 1.205       best_score 0.3232                                 
                # step_1     = [ 0.350,   0.358,   0.366,   0.373,   0.381,   0.389,   0.397,   0.405,   0.412,   0.420,   0.428,   0.436,   0.444,   0.452,   0.459,   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,  
                # vvvv cap_to__step_2                                                                                                                
                # 0.190     = [ 0.4088,  0.4039,  0.3935,  0.3950,  0.3888,  0.3770,  0.3865,  0.3772,  0.3806,  0.3806,  0.3840,  0.3693,  0.3733,  0.3833,  0.3817,  0.3778,  0.3856,  0.3897,  0.3952,  0.3964,  0.4048,  0.4017,  
                # 0.211     = [ 0.3876,  0.3831,  0.3733,  0.3746,  0.3683,  0.3573,  0.3658,  0.3574,  0.3597,  0.3596,  0.3623,  0.3482,  0.3514,  0.3611,  0.3590,  0.3554,  0.3622,  0.3670,  0.3720,  0.3728,  0.3816,  0.3782,  
                # 0.231     = [ 0.3708,  0.3671,  0.3583,  0.3591,  0.3530,  0.3431,  0.3504,  0.3431,  0.3440,  0.3440,  0.3457,  0.3331,  0.3351,  0.3440,  0.3416,  0.3385,  0.3439,  0.3493,  0.3536,  0.3542,  0.3631,  0.3595,  
                # 0.252     = [ 0.3594,  0.3570,  0.3494,  0.3495,  0.3439,  0.3357,  0.3412,  0.3355,  0.3348,  0.3349,  0.3353,  0.3250,  0.3253,  0.3331,  0.3307,  0.3282,  0.3318,  0.3377,  0.3413,  0.3416,  0.3508,  0.3468,  
                # 0.273     = [ 0.3541,  0.3533,  0.3474,  0.3468,  0.3417,  0.3359,  0.3392,  0.3353,  0.3327,  0.3332,  0.3321,  0.3248,  0.3232,  0.3295,  0.3271,  0.3255,  0.3268,  0.3332,  0.3359,  0.3360,  0.3455,  0.3412,  
                # 0.293     = [ 0.3554,  0.3566,  0.3527,  0.3512,  0.3471,  0.3439,  0.3444,  0.3428,  0.3381,  0.3392,  0.3365,  0.3328,  0.3291,  0.3335,  0.3314,  0.3309,  0.3296,  0.3363,  0.3380,  0.3381,  0.3477,  0.3432,  
                # 0.314     = [ 0.3636,  0.3671,  0.3652,  0.3627,  0.3597,  0.3593,  0.3569,  0.3575,  0.3509,  0.3528,  0.3482,  0.3484,  0.3427,  0.3452,  0.3435,  0.3441,  0.3400,  0.3469,  0.3475,  0.3478,  0.3575,  0.3525,  
                # 0.335     = [ 0.3782,  0.3842,  0.3843,  0.3807,  0.3789,  0.3813,  0.3759,  0.3786,  0.3705,  0.3730,  0.3668,  0.3710,  0.3631,  0.3638,  0.3626,  0.3643,  0.3577,  0.3644,  0.3640,  0.3648,  0.3746,  0.3690,  
                # 0.355     = [ 0.3987,  0.4073,  0.4093,  0.4045,  0.4038,  0.4088,  0.4005,  0.4054,  0.3958,  0.3989,  0.3914,  0.3994,  0.3895,  0.3884,  0.3879,  0.3904,  0.3815,  0.3881,  0.3868,  0.3881,  0.3979,  0.3918,  
                # 0.376     = [ 0.4246,  0.4354,  0.4390,  0.4333,  0.4335,  0.4408,  0.4299,  0.4368,  0.4261,  0.4296,  0.4209,  0.4326,  0.4206,  0.4180,  0.4181,  0.4214,  0.4106,  0.4168,  0.4149,  0.4168,  0.4265,  0.4199,  
                # 0.397     = [ 0.4549,  0.4676,  0.4727,  0.4660,  0.4670,  0.4766,  0.4632,  0.4719,  0.4602,  0.4640,  0.4544,  0.4693,  0.4555,  0.4515,  0.4524,  0.4561,  0.4438,  0.4496,  0.4472,  0.4497,  0.4594,  0.4522,  
                # 0.417     = [ 0.4886,  0.5032,  0.5095,  0.5020,  0.5035,  0.5152,  0.4996,  0.5099,  0.4972,  0.5015,  0.4910,  0.5089,  0.4932,  0.4882,  0.4898,  0.4937,  0.4802,  0.4857,  0.4829,  0.4860,  0.4957,  0.4878,  
                # 0.438     = [ 0.5252,  0.5415,  0.5487,  0.5404,  0.5425,  0.5559,  0.5385,  0.5501,  0.5366,  0.5411,  0.5299,  0.5506,  0.5333,  0.5272,  0.5295,  0.5336,  0.5191,  0.5242,  0.5212,  0.5248,  0.5346,  0.5260,  
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.260       best_score 0.3841                                 
                # step_1     = [   0.405,   0.412,   0.420,   0.428,   0.436,   0.444,   0.452,   0.459,   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,   0.522,   0.530,   0.538,   0.545,   0.553, 
                # vvvv cap_to__step_2                                                                                                                
                # 0.190     = [   0.4570,  0.4519,  0.4520,  0.4567,  0.4494,  0.4475,  0.4539,  0.4448,  0.4526,  0.4442,  0.4568,  0.4578,  0.4613,  0.4686,  0.4643,  0.4769,  0.4852,  0.4873,  0.4933,  0.5044, 
                # 0.211     = [   0.4363,  0.4307,  0.4311,  0.4352,  0.4279,  0.4261,  0.4317,  0.4227,  0.4303,  0.4217,  0.4336,  0.4351,  0.4383,  0.4452,  0.4410,  0.4533,  0.4616,  0.4634,  0.4694,  0.4807, 
                # 0.231     = [   0.4198,  0.4137,  0.4144,  0.4177,  0.4106,  0.4090,  0.4136,  0.4049,  0.4124,  0.4036,  0.4143,  0.4165,  0.4194,  0.4254,  0.4217,  0.4335,  0.4416,  0.4429,  0.4489,  0.4603, 
                # 0.252     = [   0.4082,  0.4018,  0.4028,  0.4051,  0.3983,  0.3970,  0.4005,  0.3923,  0.3998,  0.3908,  0.4000,  0.4030,  0.4054,  0.4102,  0.4075,  0.4183,  0.4259,  0.4268,  0.4327,  0.4439, 
                # 0.273     = [   0.4021,  0.3958,  0.3969,  0.3982,  0.3918,  0.3908,  0.3930,  0.3856,  0.3933,  0.3842,  0.3913,  0.3954,  0.3972,  0.4002,  0.3991,  0.4086,  0.4152,  0.4156,  0.4214,  0.4321, 
                # 0.293     = [   0.4020,  0.3962,  0.3974,  0.3974,  0.3916,  0.3911,  0.3917,  0.3854,  0.3934,  0.3841,  0.3889,  0.3941,  0.3951,  0.3962,  0.3971,  0.4050,  0.4103,  0.4099,  0.4157,  0.4257, 
                # 0.314     = [   0.4081,  0.4031,  0.4043,  0.4031,  0.3978,  0.3978,  0.3968,  0.3919,  0.4002,  0.3908,  0.3932,  0.3995,  0.3998,  0.3987,  0.4019,  0.4079,  0.4114,  0.4103,  0.4161,  0.4249, 
                # 0.335     = [   0.4203,  0.4166,  0.4175,  0.4150,  0.4104,  0.4110,  0.4084,  0.4051,  0.4134,  0.4042,  0.4040,  0.4115,  0.4110,  0.4076,  0.4135,  0.4173,  0.4188,  0.4168,  0.4227,  0.4301, 
                # 0.355     = [   0.4383,  0.4360,  0.4365,  0.4328,  0.4290,  0.4301,  0.4259,  0.4242,  0.4328,  0.4238,  0.4209,  0.4296,  0.4282,  0.4225,  0.4313,  0.4329,  0.4322,  0.4293,  0.4353,  0.4412, 
                # 0.376     = [   0.4615,  0.4606,  0.4607,  0.4558,  0.4528,  0.4545,  0.4487,  0.4486,  0.4575,  0.4486,  0.4433,  0.4530,  0.4509,  0.4429,  0.4546,  0.4541,  0.4511,  0.4474,  0.4536,  0.4578, 
                # 0.397     = [   0.4891,  0.4897,  0.4892,  0.4833,  0.4811,  0.4833,  0.4762,  0.4775,  0.4867,  0.4779,  0.4706,  0.4811,  0.4783,  0.4681,  0.4825,  0.4800,  0.4747,  0.4704,  0.4768,  0.4794, 
                # 0.417     = [   0.5204,  0.5225,  0.5214,  0.5146,  0.5132,  0.5158,  0.5076,  0.5102,  0.5195,  0.5108,  0.5018,  0.5130,  0.5096,  0.4975,  0.5144,  0.5099,  0.5025,  0.4978,  0.5042,  0.5052, 
                # 0.438     = [   0.5548,  0.5583,  0.5567,  0.5490,  0.5483,  0.5513,  0.5420,  0.5459,  0.5553,  0.5466,  0.5362,  0.5481,  0.5441,  0.5302,  0.5494,  0.5431,  0.5338,  0.5286,  0.5353,  0.5347, 
                # 0.459     = [   0.5916,  0.5964,  0.5943,  0.5859,  0.5858,  0.5890,  0.5790,  0.5840,  0.5934,  0.5847,  0.5731,  0.5855,  0.5809,  0.5657,  0.5868,  0.5790,  0.5678,  0.5624,  0.5693,  0.5672, 
                # 0.479     = [   0.6303,  0.6363,  0.6337,  0.6247,  0.6251,  0.6285,  0.6178,  0.6238,  0.6333,  0.6246,  0.6118,  0.6248,  0.6197,  0.6031,  0.6261,  0.6170,  0.6041,  0.5984,  0.6057,  0.6021, 
                # 0.500     = [   0.6705,  0.6775,  0.6745,  0.6649,  0.6657,  0.6693,  0.6580,  0.6648,  0.6746,  0.6657,  0.6520,  0.6654,  0.6599,  0.6422,  0.6667,  0.6565,  0.6421,  0.6361,  0.6438,  0.6389, 
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.305       best_score 0.4389                                 
                # step_1     = [  0.412,   0.420,   0.428,   0.436,   0.444,   0.452,   0.459,   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,   0.522,   0.530,   0.538,   0.545,   0.553,   0.561, 
                # vvvv cap_to__step_2                                                                                                                
                # 0.190     = [   0.5181,  0.5188,  0.5109,  0.5141,  0.5125,  0.5081,  0.5050,  0.5131,  0.5136,  0.5095,  0.5146,  0.5213,  0.5190,  0.5203,  0.5280,  0.5367,  0.5383,  0.5475,  0.5463,  0.5570, 
                # 0.211     = [   0.4972,  0.4981,  0.4900,  0.4932,  0.4908,  0.4867,  0.4835,  0.4908,  0.4909,  0.4873,  0.4917,  0.4985,  0.4963,  0.4978,  0.5050,  0.5134,  0.5154,  0.5243,  0.5231,  0.5341, 
                # 0.231     = [   0.4799,  0.4808,  0.4729,  0.4757,  0.4725,  0.4688,  0.4657,  0.4719,  0.4715,  0.4685,  0.4719,  0.4790,  0.4770,  0.4785,  0.4851,  0.4934,  0.4957,  0.5043,  0.5030,  0.5142, 
                # 0.252     = [   0.4667,  0.4677,  0.4602,  0.4624,  0.4585,  0.4552,  0.4523,  0.4572,  0.4560,  0.4539,  0.4561,  0.4635,  0.4617,  0.4633,  0.4692,  0.4770,  0.4798,  0.4881,  0.4865,  0.4979, 
                # 0.273     = [   0.4583,  0.4592,  0.4526,  0.4540,  0.4492,  0.4465,  0.4439,  0.4473,  0.4452,  0.4444,  0.4448,  0.4528,  0.4510,  0.4528,  0.4577,  0.4651,  0.4682,  0.4763,  0.4743,  0.4858, 
                # 0.293     = [   0.4552,  0.4561,  0.4506,  0.4508,  0.4453,  0.4433,  0.4410,  0.4426,  0.4396,  0.4403,  0.4390,  0.4472,  0.4456,  0.4477,  0.4513,  0.4582,  0.4616,  0.4696,  0.4670,  0.4785, 
                # 0.314     = [   0.4576,  0.4585,  0.4545,  0.4533,  0.4470,  0.4460,  0.4440,  0.4436,  0.4398,  0.4419,  0.4389,  0.4473,  0.4458,  0.4484,  0.4504,  0.4568,  0.4604,  0.4682,  0.4650,  0.4763, 
                # 0.335     = [   0.4658,  0.4665,  0.4643,  0.4614,  0.4546,  0.4546,  0.4530,  0.4505,  0.4457,  0.4493,  0.4446,  0.4531,  0.4516,  0.4548,  0.4551,  0.4611,  0.4647,  0.4726,  0.4686,  0.4797, 
                # 0.355     = [   0.4794,  0.4801,  0.4796,  0.4752,  0.4679,  0.4688,  0.4676,  0.4629,  0.4572,  0.4626,  0.4561,  0.4646,  0.4631,  0.4669,  0.4654,  0.4711,  0.4746,  0.4825,  0.4778,  0.4886, 
                # 0.376     = [   0.4982,  0.4988,  0.5003,  0.4941,  0.4866,  0.4882,  0.4875,  0.4807,  0.4742,  0.4812,  0.4728,  0.4814,  0.4799,  0.4843,  0.4810,  0.4865,  0.4897,  0.4979,  0.4922,  0.5029, 
                # 0.397     = [   0.5216,  0.5223,  0.5256,  0.5178,  0.5100,  0.5124,  0.5120,  0.5032,  0.4960,  0.5046,  0.4943,  0.5032,  0.5014,  0.5064,  0.5015,  0.5067,  0.5096,  0.5184,  0.5116,  0.5220, 
                # 0.417     = [   0.5491,  0.5499,  0.5549,  0.5456,  0.5375,  0.5407,  0.5405,  0.5298,  0.5220,  0.5323,  0.5202,  0.5292,  0.5272,  0.5327,  0.5262,  0.5313,  0.5339,  0.5433,  0.5355,  0.5456, 
                # 0.438     = [   0.5801,  0.5810,  0.5875,  0.5768,  0.5687,  0.5725,  0.5724,  0.5600,  0.5517,  0.5634,  0.5497,  0.5590,  0.5568,  0.5625,  0.5547,  0.5597,  0.5620,  0.5721,  0.5631,  0.5730, 
                # 0.459     = [   0.6140,  0.6149,  0.6229,  0.6109,  0.6027,  0.6072,  0.6072,  0.5932,  0.5843,  0.5975,  0.5823,  0.5917,  0.5894,  0.5951,  0.5863,  0.5913,  0.5933,  0.6040,  0.5940,  0.6036, 
                # 0.479     = [   0.6503,  0.6512,  0.6606,  0.6474,  0.6390,  0.6441,  0.6442,  0.6288,  0.6193,  0.6339,  0.6173,  0.6269,  0.6245,  0.6302,  0.6204,  0.6255,  0.6271,  0.6387,  0.6276,  0.6369, 
                # 0.500     = [   0.6883,  0.6894,  0.7000,  0.6858,  0.6772,  0.6828,  0.6829,  0.6662,  0.6562,  0.6722,  0.6543,  0.6639,  0.6614,  0.6672,  0.6565,  0.6617,  0.6630,  0.6754,  0.6633,  0.6723, 
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.358       best_score 0.4917                                 
                # step_1     = [  0.420,   0.428,   0.436,   0.444,   0.452,   0.459,   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,   0.522,   0.530,   0.538,   0.545,   0.553,   0.561,   0.569,   0.577,   0.584,   0.592,   0.600]
                # vvvv cap_to__step_2                                                                                                                
                # 0.190     = [  0.5735,  0.5674,  0.5657,  0.5746,  0.5706,  0.5641,  0.5711,  0.5706,  0.5740,  0.5734,  0.5716,  0.5800,  0.5802,  0.5878,  0.5834,  0.5950,  0.6006,  0.5954,  0.6127,  0.6099,  0.6093,  0.6176,  0.6224,  0.6294]
                # 0.211     = [  0.5529,  0.5468,  0.5447,  0.5536,  0.5497,  0.5428,  0.5502,  0.5493,  0.5522,  0.5514,  0.5491,  0.5576,  0.5576,  0.5650,  0.5611,  0.5723,  0.5782,  0.5726,  0.5903,  0.5872,  0.5866,  0.5952,  0.5995,  0.6065]
                # 0.231     = [  0.5351,  0.5292,  0.5269,  0.5356,  0.5319,  0.5247,  0.5324,  0.5311,  0.5334,  0.5323,  0.5295,  0.5382,  0.5378,  0.5450,  0.5416,  0.5524,  0.5586,  0.5525,  0.5705,  0.5671,  0.5666,  0.5752,  0.5792,  0.5861]
                # 0.252     = [  0.5208,  0.5154,  0.5127,  0.5211,  0.5180,  0.5103,  0.5183,  0.5165,  0.5182,  0.5167,  0.5133,  0.5224,  0.5213,  0.5281,  0.5255,  0.5357,  0.5423,  0.5360,  0.5537,  0.5501,  0.5498,  0.5584,  0.5619,  0.5686]
                # 0.273     = [  0.5104,  0.5058,  0.5029,  0.5107,  0.5083,  0.5003,  0.5084,  0.5062,  0.5070,  0.5052,  0.5013,  0.5107,  0.5090,  0.5151,  0.5133,  0.5229,  0.5298,  0.5237,  0.5406,  0.5367,  0.5366,  0.5451,  0.5484,  0.5547]
                # 0.293     = [  0.5045,  0.5009,  0.4978,  0.5048,  0.5035,  0.4953,  0.5033,  0.5007,  0.5004,  0.4985,  0.4940,  0.5036,  0.5010,  0.5065,  0.5055,  0.5144,  0.5218,  0.5158,  0.5316,  0.5274,  0.5277,  0.5359,  0.5390,  0.5447]
                # 0.314     = [  0.5035,  0.5011,  0.4980,  0.5039,  0.5037,  0.4957,  0.5034,  0.5003,  0.4988,  0.4967,  0.4917,  0.5013,  0.4980,  0.5025,  0.5026,  0.5105,  0.5186,  0.5129,  0.5271,  0.5225,  0.5234,  0.5312,  0.5342,  0.5394]
                # 0.335     = [  0.5075,  0.5066,  0.5035,  0.5082,  0.5092,  0.5016,  0.5088,  0.5053,  0.5024,  0.5002,  0.4947,  0.5041,  0.5003,  0.5034,  0.5050,  0.5115,  0.5204,  0.5151,  0.5274,  0.5226,  0.5239,  0.5313,  0.5341,  0.5388]
                # 0.355     = [  0.5166,  0.5175,  0.5144,  0.5176,  0.5200,  0.5129,  0.5195,  0.5156,  0.5111,  0.5090,  0.5031,  0.5121,  0.5079,  0.5096,  0.5126,  0.5177,  0.5273,  0.5224,  0.5327,  0.5278,  0.5297,  0.5363,  0.5391,  0.5431]
                # 0.376     = [  0.5307,  0.5335,  0.5304,  0.5320,  0.5359,  0.5294,  0.5352,  0.5310,  0.5250,  0.5227,  0.5168,  0.5252,  0.5207,  0.5208,  0.5254,  0.5291,  0.5392,  0.5349,  0.5429,  0.5378,  0.5405,  0.5462,  0.5490,  0.5523]
                # 0.397     = [  0.5494,  0.5540,  0.5513,  0.5511,  0.5564,  0.5507,  0.5557,  0.5511,  0.5438,  0.5411,  0.5352,  0.5430,  0.5384,  0.5368,  0.5431,  0.5453,  0.5558,  0.5523,  0.5577,  0.5526,  0.5563,  0.5608,  0.5639,  0.5664]
                # 0.417     = [  0.5722,  0.5787,  0.5763,  0.5744,  0.5812,  0.5761,  0.5804,  0.5754,  0.5670,  0.5638,  0.5582,  0.5651,  0.5604,  0.5573,  0.5651,  0.5659,  0.5769,  0.5742,  0.5770,  0.5717,  0.5765,  0.5800,  0.5833,  0.5850]
                # 0.438     = [  0.5988,  0.6070,  0.6051,  0.6015,  0.6097,  0.6053,  0.6087,  0.6035,  0.5939,  0.5904,  0.5850,  0.5912,  0.5865,  0.5818,  0.5911,  0.5904,  0.6019,  0.6000,  0.6002,  0.5948,  0.6007,  0.6032,  0.6068,  0.6076]
                # 0.459     = [  0.6286,  0.6384,  0.6370,  0.6319,  0.6414,  0.6376,  0.6402,  0.6347,  0.6241,  0.6203,  0.6151,  0.6207,  0.6160,  0.6096,  0.6204,  0.6184,  0.6303,  0.6291,  0.6270,  0.6213,  0.6285,  0.6300,  0.6338,  0.6338]
                # 0.479     = [  0.6611,  0.6724,  0.6715,  0.6651,  0.6757,  0.6724,  0.6743,  0.6686,  0.6570,  0.6530,  0.6479,  0.6530,  0.6482,  0.6404,  0.6525,  0.6495,  0.6616,  0.6610,  0.6568,  0.6510,  0.6592,  0.6598,  0.6639,  0.6631]
                # 0.500     = [  0.6959,  0.7085,  0.7082,  0.7004,  0.7120,  0.7093,  0.7106,  0.7046,  0.6920,  0.6878,  0.6830,  0.6874,  0.6827,  0.6735,  0.6869,  0.6829,  0.6953,  0.6954,  0.6890,  0.6832,  0.6924,  0.6922,  0.6965,  0.6949]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.409       best_score 0.5423                                 
                # step_1     = [   0.428,   0.436,   0.444,   0.452,   0.459,   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,   0.522,   0.530,   0.538,   0.545,   0.553,   0.561,   0.569,   0.577,   0.584,   0.592,   0.600]
                # vvvv cap_to__step_2                                                                                                                
                # 0.190     = [   0.6357,  0.6368,  0.6356,  0.6335,  0.6241,  0.6219,  0.6332,  0.6305,  0.6312,  0.6211,  0.6314,  0.6404,  0.6410,  0.6420,  0.6375,  0.6545,  0.6556,  0.6530,  0.6627,  0.6696,  0.6670,  0.6777,  0.6759]
                # 0.211     = [   0.6151,  0.6164,  0.6151,  0.6129,  0.6033,  0.6009,  0.6122,  0.6089,  0.6096,  0.5994,  0.6095,  0.6184,  0.6189,  0.6201,  0.6150,  0.6321,  0.6332,  0.6305,  0.6405,  0.6473,  0.6445,  0.6554,  0.6536]
                # 0.231     = [   0.5970,  0.5984,  0.5972,  0.5948,  0.5851,  0.5826,  0.5937,  0.5898,  0.5906,  0.5805,  0.5900,  0.5989,  0.5994,  0.6004,  0.5951,  0.6119,  0.6131,  0.6105,  0.6205,  0.6272,  0.6245,  0.6352,  0.6336]
                # 0.252     = [   0.5819,  0.5834,  0.5822,  0.5797,  0.5701,  0.5675,  0.5782,  0.5738,  0.5746,  0.5648,  0.5736,  0.5822,  0.5828,  0.5836,  0.5783,  0.5945,  0.5955,  0.5935,  0.6032,  0.6098,  0.6074,  0.6175,  0.6162]
                # 0.273     = [   0.5701,  0.5717,  0.5708,  0.5681,  0.5589,  0.5562,  0.5661,  0.5614,  0.5623,  0.5529,  0.5605,  0.5688,  0.5697,  0.5702,  0.5651,  0.5803,  0.5812,  0.5799,  0.5891,  0.5955,  0.5937,  0.6028,  0.6019]
                # 0.293     = [   0.5622,  0.5641,  0.5632,  0.5605,  0.5517,  0.5491,  0.5580,  0.5529,  0.5540,  0.5452,  0.5514,  0.5593,  0.5605,  0.5606,  0.5560,  0.5698,  0.5705,  0.5702,  0.5786,  0.5849,  0.5840,  0.5916,  0.5912]
                # 0.314     = [   0.5587,  0.5607,  0.5598,  0.5570,  0.5490,  0.5466,  0.5543,  0.5488,  0.5500,  0.5423,  0.5465,  0.5540,  0.5557,  0.5554,  0.5517,  0.5635,  0.5637,  0.5648,  0.5722,  0.5782,  0.5787,  0.5843,  0.5845]
                # 0.335     = [   0.5597,  0.5618,  0.5609,  0.5581,  0.5511,  0.5490,  0.5551,  0.5494,  0.5507,  0.5443,  0.5462,  0.5531,  0.5555,  0.5547,  0.5520,  0.5615,  0.5613,  0.5640,  0.5702,  0.5758,  0.5780,  0.5812,  0.5822]
                # 0.355     = [   0.5654,  0.5675,  0.5667,  0.5638,  0.5582,  0.5563,  0.5605,  0.5549,  0.5563,  0.5515,  0.5507,  0.5569,  0.5601,  0.5586,  0.5571,  0.5640,  0.5634,  0.5680,  0.5727,  0.5779,  0.5822,  0.5825,  0.5844]
                # 0.376     = [   0.5759,  0.5778,  0.5772,  0.5742,  0.5703,  0.5685,  0.5705,  0.5651,  0.5668,  0.5637,  0.5600,  0.5654,  0.5695,  0.5673,  0.5671,  0.5711,  0.5703,  0.5768,  0.5799,  0.5846,  0.5912,  0.5883,  0.5912]
                # 0.397     = [   0.5909,  0.5926,  0.5922,  0.5892,  0.5870,  0.5854,  0.5852,  0.5799,  0.5820,  0.5806,  0.5739,  0.5786,  0.5834,  0.5807,  0.5819,  0.5829,  0.5817,  0.5906,  0.5917,  0.5959,  0.6051,  0.5987,  0.6025]
                # 0.417     = [   0.6102,  0.6117,  0.6115,  0.6084,  0.6081,  0.6065,  0.6041,  0.5991,  0.6016,  0.6019,  0.5922,  0.5961,  0.6017,  0.5984,  0.6013,  0.5990,  0.5976,  0.6088,  0.6078,  0.6115,  0.6232,  0.6134,  0.6183]
                # 0.438     = [   0.6335,  0.6347,  0.6346,  0.6315,  0.6331,  0.6314,  0.6271,  0.6222,  0.6253,  0.6271,  0.6145,  0.6176,  0.6240,  0.6203,  0.6247,  0.6191,  0.6175,  0.6313,  0.6280,  0.6313,  0.6455,  0.6321,  0.6382]
                # 0.459     = [   0.6602,  0.6612,  0.6613,  0.6579,  0.6614,  0.6598,  0.6535,  0.6489,  0.6525,  0.6559,  0.6404,  0.6427,  0.6499,  0.6457,  0.6518,  0.6430,  0.6410,  0.6574,  0.6518,  0.6547,  0.6713,  0.6544,  0.6617]
                # 0.479     = [   0.6900,  0.6908,  0.6910,  0.6874,  0.6928,  0.6911,  0.6829,  0.6787,  0.6827,  0.6876,  0.6694,  0.6708,  0.6788,  0.6743,  0.6818,  0.6701,  0.6678,  0.6867,  0.6788,  0.6813,  0.7003,  0.6800,  0.6884]
                # 0.500     = [   0.7224,  0.7231,  0.7233,  0.7196,  0.7267,  0.7250,  0.7148,  0.7108,  0.7153,  0.7217,  0.7010,  0.7016,  0.7103,  0.7056,  0.7145,  0.7002,  0.6974,  0.7186,  0.7085,  0.7108,  0.7318,  0.7084,  0.7179]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.449       best_score 0.5879                                 
                # step_1     = [  0.412,   0.420,   0.428,   0.436,   0.444,   0.452,   0.459,   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,   0.522,   0.530,   0.538,   0.545,   0.553,   0.561,   0.569,   0.577,   0.584,   0.592,   0.600]
                # vvvv cap_to__step_2                                                                                                                
                # 0.190     = [  0.6955,  0.6896,  0.6860,  0.6895,  0.6794,  0.6772,  0.6795,  0.6760,  0.6801,  0.6782,  0.6725,  0.6803,  0.6846,  0.6848,  0.6813,  0.6808,  0.6798,  0.6927,  0.6928,  0.7005,  0.7013,  0.7113,  0.7131,  0.7142,  0.7238]
                # 0.211     = [  0.6750,  0.6690,  0.6657,  0.6690,  0.6590,  0.6564,  0.6587,  0.6551,  0.6590,  0.6567,  0.6511,  0.6590,  0.6631,  0.6630,  0.6591,  0.6587,  0.6575,  0.6708,  0.6707,  0.6779,  0.6790,  0.6890,  0.6910,  0.6922,  0.7020]
                # 0.231     = [  0.6567,  0.6506,  0.6476,  0.6508,  0.6408,  0.6378,  0.6403,  0.6364,  0.6402,  0.6375,  0.6321,  0.6401,  0.6439,  0.6436,  0.6392,  0.6390,  0.6376,  0.6510,  0.6507,  0.6575,  0.6589,  0.6688,  0.6710,  0.6720,  0.6821]
                # 0.252     = [  0.6408,  0.6348,  0.6322,  0.6350,  0.6253,  0.6221,  0.6248,  0.6204,  0.6240,  0.6210,  0.6159,  0.6240,  0.6273,  0.6269,  0.6222,  0.6221,  0.6206,  0.6337,  0.6333,  0.6396,  0.6413,  0.6510,  0.6534,  0.6542,  0.6647]
                # 0.273     = [  0.6279,  0.6221,  0.6198,  0.6222,  0.6130,  0.6095,  0.6126,  0.6076,  0.6109,  0.6077,  0.6030,  0.6112,  0.6138,  0.6134,  0.6083,  0.6086,  0.6069,  0.6194,  0.6189,  0.6246,  0.6267,  0.6361,  0.6387,  0.6393,  0.6499]
                # 0.293     = [  0.6182,  0.6129,  0.6109,  0.6128,  0.6044,  0.6005,  0.6041,  0.5984,  0.6013,  0.5980,  0.5938,  0.6021,  0.6040,  0.6034,  0.5981,  0.5989,  0.5970,  0.6086,  0.6080,  0.6130,  0.6156,  0.6244,  0.6273,  0.6276,  0.6384]
                # 0.314     = [  0.6122,  0.6077,  0.6056,  0.6070,  0.5997,  0.5955,  0.5997,  0.5931,  0.5957,  0.5921,  0.5886,  0.5971,  0.5981,  0.5974,  0.5919,  0.5933,  0.5912,  0.6015,  0.6011,  0.6054,  0.6083,  0.6163,  0.6195,  0.6195,  0.6305]
                # 0.335     = [  0.6103,  0.6067,  0.6045,  0.6052,  0.5991,  0.5948,  0.5996,  0.5922,  0.5944,  0.5907,  0.5879,  0.5963,  0.5966,  0.5957,  0.5902,  0.5923,  0.5899,  0.5985,  0.5984,  0.6018,  0.6054,  0.6122,  0.6159,  0.6153,  0.6265]
                # 0.355     = [  0.6126,  0.6102,  0.6076,  0.6076,  0.6028,  0.5985,  0.6039,  0.5957,  0.5975,  0.5937,  0.5917,  0.6000,  0.5994,  0.5985,  0.5930,  0.5961,  0.5933,  0.5999,  0.6001,  0.6027,  0.6069,  0.6122,  0.6166,  0.6153,  0.6266]
                # 0.376     = [  0.6190,  0.6181,  0.6150,  0.6143,  0.6110,  0.6067,  0.6129,  0.6038,  0.6051,  0.6013,  0.6001,  0.6084,  0.6067,  0.6060,  0.6004,  0.6046,  0.6014,  0.6057,  0.6065,  0.6081,  0.6129,  0.6167,  0.6218,  0.6196,  0.6310]
                # 0.397     = [  0.6298,  0.6305,  0.6268,  0.6254,  0.6235,  0.6193,  0.6263,  0.6163,  0.6172,  0.6133,  0.6129,  0.6212,  0.6184,  0.6181,  0.6125,  0.6177,  0.6143,  0.6159,  0.6174,  0.6180,  0.6235,  0.6257,  0.6314,  0.6282,  0.6396]
                # 0.417     = [  0.6449,  0.6471,  0.6428,  0.6406,  0.6403,  0.6362,  0.6441,  0.6330,  0.6335,  0.6297,  0.6301,  0.6385,  0.6342,  0.6346,  0.6289,  0.6351,  0.6317,  0.6305,  0.6326,  0.6324,  0.6384,  0.6390,  0.6453,  0.6410,  0.6524]
                # 0.438     = [  0.6638,  0.6678,  0.6627,  0.6599,  0.6610,  0.6571,  0.6658,  0.6537,  0.6538,  0.6501,  0.6512,  0.6598,  0.6541,  0.6551,  0.6495,  0.6566,  0.6533,  0.6491,  0.6520,  0.6508,  0.6573,  0.6564,  0.6632,  0.6579,  0.6691]
                # 0.459     = [  0.6864,  0.6921,  0.6862,  0.6829,  0.6854,  0.6816,  0.6910,  0.6780,  0.6778,  0.6742,  0.6758,  0.6849,  0.6775,  0.6793,  0.6737,  0.6818,  0.6785,  0.6714,  0.6750,  0.6730,  0.6799,  0.6775,  0.6849,  0.6785,  0.6896]
                # 0.479     = [  0.7123,  0.7197,  0.7130,  0.7091,  0.7130,  0.7093,  0.7194,  0.7056,  0.7050,  0.7015,  0.7036,  0.7133,  0.7041,  0.7067,  0.7012,  0.7102,  0.7069,  0.6971,  0.7012,  0.6986,  0.7059,  0.7019,  0.7098,  0.7026,  0.7133]
                # 0.500     = [  0.7412,  0.7500,  0.7427,  0.7382,  0.7434,  0.7398,  0.7505,  0.7360,  0.7350,  0.7318,  0.7340,  0.7444,  0.7336,  0.7368,  0.7316,  0.7412,  0.7381,  0.7256,  0.7303,  0.7272,  0.7348,  0.7293,  0.7378,  0.7298,  0.7400]
                # ------------                                                                                                               
                # dim 100      actual_init_angle_score 1.508       best_score 0.6279                                 
                # step_1     = [  0.412,   0.420,   0.428,   0.436,   0.444,   0.452,   0.459,   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,   0.522,   0.530,   0.538,   0.545,   0.553,   0.561,   0.569,   0.577,   0.584,   0.592,   0.600]
                # vvvv cap_to__step_2                                                                                                                
                # 0.190     = [  0.7462,  0.7407,  0.7375,  0.7290,  0.7309,  0.7241,  0.7260,  0.7333,  0.7223,  0.7250,  0.7240,  0.7235,  0.7254,  0.7292,  0.7299,  0.7346,  0.7315,  0.7384,  0.7419,  0.7445,  0.7454,  0.7490,  0.7618,  0.7558,  0.7651]
                # 0.211     = [  0.7255,  0.7203,  0.7169,  0.7083,  0.7102,  0.7032,  0.7052,  0.7121,  0.7011,  0.7035,  0.7025,  0.7018,  0.7037,  0.7075,  0.7077,  0.7129,  0.7092,  0.7160,  0.7196,  0.7223,  0.7233,  0.7268,  0.7397,  0.7332,  0.7428]
                # 0.231     = [  0.7068,  0.7017,  0.6982,  0.6895,  0.6915,  0.6844,  0.6863,  0.6929,  0.6819,  0.6841,  0.6830,  0.6821,  0.6839,  0.6876,  0.6875,  0.6930,  0.6888,  0.6956,  0.6993,  0.7021,  0.7032,  0.7064,  0.7194,  0.7125,  0.7222]
                # 0.252     = [  0.6903,  0.6854,  0.6819,  0.6732,  0.6751,  0.6681,  0.6698,  0.6760,  0.6653,  0.6671,  0.6660,  0.6649,  0.6666,  0.6701,  0.6698,  0.6753,  0.6708,  0.6775,  0.6812,  0.6842,  0.6854,  0.6883,  0.7013,  0.6941,  0.7039]
                # 0.273     = [  0.6766,  0.6716,  0.6681,  0.6596,  0.6614,  0.6546,  0.6561,  0.6619,  0.6516,  0.6531,  0.6518,  0.6504,  0.6520,  0.6553,  0.6550,  0.6602,  0.6555,  0.6621,  0.6659,  0.6691,  0.6703,  0.6729,  0.6856,  0.6783,  0.6881]
                # 0.293     = [  0.6660,  0.6608,  0.6574,  0.6492,  0.6508,  0.6444,  0.6457,  0.6510,  0.6412,  0.6424,  0.6410,  0.6392,  0.6405,  0.6437,  0.6434,  0.6482,  0.6435,  0.6496,  0.6536,  0.6571,  0.6585,  0.6604,  0.6728,  0.6657,  0.6753]
                # 0.314     = [  0.6588,  0.6533,  0.6502,  0.6423,  0.6435,  0.6379,  0.6389,  0.6436,  0.6344,  0.6354,  0.6336,  0.6316,  0.6326,  0.6356,  0.6356,  0.6395,  0.6350,  0.6405,  0.6448,  0.6486,  0.6502,  0.6514,  0.6633,  0.6565,  0.6660]
                # 0.335     = [  0.6555,  0.6494,  0.6468,  0.6392,  0.6401,  0.6354,  0.6361,  0.6402,  0.6317,  0.6324,  0.6302,  0.6279,  0.6286,  0.6313,  0.6319,  0.6346,  0.6305,  0.6352,  0.6399,  0.6440,  0.6458,  0.6460,  0.6572,  0.6511,  0.6602]
                # 0.355     = [  0.6562,  0.6493,  0.6474,  0.6402,  0.6407,  0.6369,  0.6375,  0.6408,  0.6334,  0.6337,  0.6310,  0.6283,  0.6286,  0.6311,  0.6324,  0.6337,  0.6304,  0.6339,  0.6389,  0.6436,  0.6456,  0.6447,  0.6550,  0.6498,  0.6585]
                # 0.376     = [  0.6610,  0.6534,  0.6521,  0.6455,  0.6454,  0.6428,  0.6433,  0.6457,  0.6397,  0.6394,  0.6361,  0.6329,  0.6329,  0.6351,  0.6373,  0.6368,  0.6345,  0.6366,  0.6421,  0.6475,  0.6496,  0.6476,  0.6568,  0.6526,  0.6611]
                # 0.397     = [  0.6700,  0.6615,  0.6612,  0.6551,  0.6543,  0.6530,  0.6534,  0.6548,  0.6504,  0.6494,  0.6457,  0.6419,  0.6417,  0.6434,  0.6466,  0.6443,  0.6430,  0.6435,  0.6498,  0.6558,  0.6580,  0.6546,  0.6627,  0.6597,  0.6679]
                # 0.417     = [  0.6831,  0.6735,  0.6743,  0.6689,  0.6674,  0.6674,  0.6677,  0.6681,  0.6654,  0.6637,  0.6594,  0.6550,  0.6545,  0.6559,  0.6602,  0.6558,  0.6557,  0.6545,  0.6615,  0.6683,  0.6705,  0.6659,  0.6728,  0.6711,  0.6789]
                # 0.438     = [  0.7002,  0.6897,  0.6914,  0.6867,  0.6845,  0.6859,  0.6859,  0.6854,  0.6845,  0.6821,  0.6771,  0.6722,  0.6713,  0.6724,  0.6778,  0.6713,  0.6724,  0.6695,  0.6771,  0.6849,  0.6871,  0.6811,  0.6867,  0.6866,  0.6940]
                # 0.459     = [  0.7212,  0.7095,  0.7123,  0.7081,  0.7053,  0.7082,  0.7079,  0.7064,  0.7072,  0.7044,  0.6984,  0.6931,  0.6919,  0.6927,  0.6993,  0.6905,  0.6928,  0.6883,  0.6965,  0.7052,  0.7075,  0.7002,  0.7044,  0.7061,  0.7129]
                # 0.479     = [  0.7456,  0.7328,  0.7366,  0.7330,  0.7295,  0.7337,  0.7331,  0.7308,  0.7331,  0.7299,  0.7231,  0.7175,  0.7159,  0.7162,  0.7243,  0.7132,  0.7166,  0.7106,  0.7192,  0.7289,  0.7314,  0.7227,  0.7254,  0.7290,  0.7352]
                # 0.500     = [  0.7731,  0.7592,  0.7639,  0.7609,  0.7566,  0.7622,  0.7613,  0.7582,  0.7620,  0.7584,  0.7508,  0.7448,  0.7429,  0.7427,  0.7522,  0.7388,  0.7434,  0.7360,  0.7450,  0.7556,  0.7583,  0.7484,  0.7495,  0.7549,  0.7608]
                # ------------       
                # dim 100      actual_init_angle_score 1.547       best_score 0.6573                                 
# step_1     = [   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,   0.522,   0.530,   0.538,   0.545,   0.553,   0.561,   0.569,   0.577,   0.584,   0.592,   0.600]
# vvvv cap_to__step_2                                                                                                                
# 0.211     = [   0.7434,  0.7436,  0.7429,  0.7398,  0.7372,  0.7489,  0.7415,  0.7417,  0.7407,  0.7429,  0.7566,  0.7483,  0.7512,  0.7570,  0.7589,  0.7723,  0.7603,  0.7648]
# 0.231     = [   0.7238,  0.7241,  0.7232,  0.7196,  0.7174,  0.7291,  0.7211,  0.7216,  0.7201,  0.7221,  0.7359,  0.7279,  0.7305,  0.7362,  0.7381,  0.7513,  0.7396,  0.7440]
# 0.252     = [   0.7064,  0.7067,  0.7056,  0.7017,  0.6999,  0.7113,  0.7029,  0.7037,  0.7019,  0.7035,  0.7174,  0.7096,  0.7121,  0.7175,  0.7196,  0.7322,  0.7212,  0.7254]
# 0.273     = [   0.6917,  0.6917,  0.6907,  0.6864,  0.6851,  0.6959,  0.6872,  0.6886,  0.6864,  0.6876,  0.7013,  0.6941,  0.6964,  0.7015,  0.7035,  0.7154,  0.7056,  0.7094]
# 0.293     = [   0.6801,  0.6796,  0.6788,  0.6741,  0.6734,  0.6832,  0.6745,  0.6766,  0.6740,  0.6747,  0.6881,  0.6816,  0.6837,  0.6883,  0.6903,  0.7013,  0.6931,  0.6965]
# 0.314     = [   0.6720,  0.6708,  0.6703,  0.6651,  0.6653,  0.6737,  0.6652,  0.6681,  0.6650,  0.6651,  0.6782,  0.6727,  0.6746,  0.6784,  0.6803,  0.6904,  0.6839,  0.6870]
# 0.335     = [   0.6678,  0.6657,  0.6655,  0.6599,  0.6609,  0.6675,  0.6596,  0.6634,  0.6599,  0.6592,  0.6718,  0.6675,  0.6694,  0.6722,  0.6739,  0.6828,  0.6784,  0.6812]
# 0.355     = [   0.6675,  0.6645,  0.6647,  0.6586,  0.6605,  0.6651,  0.6579,  0.6630,  0.6590,  0.6573,  0.6694,  0.6663,  0.6683,  0.6699,  0.6714,  0.6789,  0.6770,  0.6793]
# 0.376     = [   0.6714,  0.6674,  0.6679,  0.6616,  0.6642,  0.6666,  0.6603,  0.6667,  0.6623,  0.6595,  0.6708,  0.6694,  0.6715,  0.6716,  0.6729,  0.6790,  0.6799,  0.6815]
# 0.397     = [   0.6795,  0.6744,  0.6752,  0.6687,  0.6723,  0.6722,  0.6669,  0.6748,  0.6699,  0.6660,  0.6764,  0.6768,  0.6789,  0.6773,  0.6785,  0.6831,  0.6869,  0.6879]
# 0.417     = [   0.6917,  0.6855,  0.6867,  0.6800,  0.6846,  0.6818,  0.6776,  0.6871,  0.6818,  0.6767,  0.6860,  0.6884,  0.6905,  0.6872,  0.6882,  0.6912,  0.6983,  0.6985]
# 0.438     = [   0.7080,  0.7005,  0.7022,  0.6954,  0.7009,  0.6954,  0.6923,  0.7034,  0.6978,  0.6914,  0.6997,  0.7040,  0.7063,  0.7011,  0.7020,  0.7036,  0.7138,  0.7131]
# 0.459     = [   0.7280,  0.7192,  0.7215,  0.7145,  0.7211,  0.7127,  0.7108,  0.7235,  0.7176,  0.7099,  0.7171,  0.7235,  0.7258,  0.7188,  0.7195,  0.7198,  0.7330,  0.7317]
# 0.479     = [   0.7516,  0.7414,  0.7441,  0.7372,  0.7447,  0.7335,  0.7326,  0.7471,  0.7409,  0.7317,  0.7381,  0.7464,  0.7489,  0.7400,  0.7407,  0.7396,  0.7557,  0.7537]
# 0.500     = [   0.7784,  0.7666,  0.7699,  0.7632,  0.7714,  0.7573,  0.7577,  0.7737,  0.7672,  0.7566,  0.7625,  0.7724,  0.7752,  0.7645,  0.7649,  0.7626,  0.7815,  0.7789]
# ------------                                                                                                               
# dim 100      actual_init_angle_score 1.565       best_score 0.6600                                 
# step_1     = [   0.444,   0.452,   0.459,   0.467,   0.475,   0.483,   0.491,   0.498,   0.506,   0.514,   0.522,   0.530,   0.538,   0.545,   0.553,   0.561,   0.569,   0.577,   0.584,   0.592,   0.600]
# vvvv cap_to__step_2                                                                                                                
# 0.190     = [   0.7778,  0.7752,  0.7753,  0.7654,  0.7732,  0.7690,  0.7751,  0.7664,  0.7666,  0.7710,  0.7690,  0.7658,  0.7710,  0.7748,  0.7760,  0.7803,  0.7865,  0.7840,  0.7885,  0.7924,  0.7930]
# 0.211     = [   0.7565,  0.7540,  0.7540,  0.7435,  0.7516,  0.7474,  0.7534,  0.7444,  0.7444,  0.7490,  0.7468,  0.7432,  0.7485,  0.7523,  0.7536,  0.7575,  0.7639,  0.7615,  0.7659,  0.7700,  0.7703]
# 0.231     = [   0.7371,  0.7344,  0.7345,  0.7234,  0.7317,  0.7277,  0.7335,  0.7243,  0.7240,  0.7287,  0.7265,  0.7224,  0.7279,  0.7317,  0.7330,  0.7365,  0.7429,  0.7408,  0.7451,  0.7493,  0.7493]
# 0.252     = [   0.7199,  0.7169,  0.7172,  0.7055,  0.7140,  0.7103,  0.7156,  0.7065,  0.7060,  0.7106,  0.7084,  0.7039,  0.7095,  0.7133,  0.7147,  0.7177,  0.7239,  0.7222,  0.7264,  0.7307,  0.7305]
# 0.273     = [   0.7052,  0.7017,  0.7024,  0.6902,  0.6989,  0.6956,  0.7002,  0.6914,  0.6907,  0.6950,  0.6929,  0.6881,  0.6937,  0.6975,  0.6990,  0.7015,  0.7073,  0.7061,  0.7102,  0.7145,  0.7143]
# 0.293     = [   0.6934,  0.6893,  0.6905,  0.6779,  0.6866,  0.6841,  0.6876,  0.6794,  0.6785,  0.6824,  0.6806,  0.6755,  0.6809,  0.6846,  0.6863,  0.6882,  0.6935,  0.6928,  0.6969,  0.7010,  0.7010]
# 0.314     = [   0.6850,  0.6800,  0.6819,  0.6689,  0.6774,  0.6759,  0.6782,  0.6708,  0.6698,  0.6730,  0.6718,  0.6663,  0.6715,  0.6750,  0.6769,  0.6782,  0.6829,  0.6828,  0.6869,  0.6908,  0.6910]
# 0.335     = [   0.6802,  0.6740,  0.6770,  0.6636,  0.6718,  0.6716,  0.6723,  0.6661,  0.6649,  0.6672,  0.6667,  0.6611,  0.6658,  0.6691,  0.6713,  0.6719,  0.6758,  0.6763,  0.6805,  0.6841,  0.6846]
# 0.355     = [   0.6792,  0.6716,  0.6759,  0.6622,  0.6702,  0.6712,  0.6701,  0.6654,  0.6639,  0.6653,  0.6656,  0.6600,  0.6641,  0.6671,  0.6697,  0.6696,  0.6724,  0.6736,  0.6780,  0.6812,  0.6822]
# 0.376     = [   0.6824,  0.6731,  0.6789,  0.6650,  0.6726,  0.6750,  0.6717,  0.6689,  0.6672,  0.6675,  0.6688,  0.6632,  0.6667,  0.6693,  0.6723,  0.6715,  0.6731,  0.6748,  0.6796,  0.6821,  0.6838]
# 0.397     = [   0.6898,  0.6786,  0.6858,  0.6720,  0.6791,  0.6831,  0.6774,  0.6766,  0.6747,  0.6738,  0.6760,  0.6707,  0.6736,  0.6757,  0.6793,  0.6776,  0.6779,  0.6801,  0.6853,  0.6871,  0.6897]
# 0.417     = [   0.7014,  0.6880,  0.6968,  0.6832,  0.6898,  0.6954,  0.6872,  0.6884,  0.6864,  0.6844,  0.6875,  0.6824,  0.6848,  0.6862,  0.6905,  0.6880,  0.6869,  0.6894,  0.6952,  0.6963,  0.6995]
# 0.438     = [   0.7169,  0.7012,  0.7116,  0.6984,  0.7045,  0.7118,  0.7008,  0.7044,  0.7023,  0.6989,  0.7030,  0.6981,  0.7000,  0.7007,  0.7057,  0.7026,  0.6997,  0.7027,  0.7092,  0.7095,  0.7133]
# 0.459     = [   0.7361,  0.7182,  0.7303,  0.7175,  0.7231,  0.7320,  0.7182,  0.7242,  0.7220,  0.7171,  0.7222,  0.7175,  0.7192,  0.7192,  0.7246,  0.7210,  0.7164,  0.7197,  0.7269,  0.7264,  0.7309]
# 0.479     = [   0.7588,  0.7386,  0.7525,  0.7401,  0.7451,  0.7556,  0.7391,  0.7476,  0.7453,  0.7388,  0.7449,  0.7405,  0.7419,  0.7412,  0.7472,  0.7431,  0.7367,  0.7404,  0.7481,  0.7469,  0.7520]
# 0.500     = [   0.7847,  0.7622,  0.7778,  0.7658,  0.7702,  0.7823,  0.7631,  0.7740,  0.7718,  0.7637,  0.7708,  0.7665,  0.7677,  0.7663,  0.7729,  0.7683,  0.7601,  0.7642,  0.7725,  0.7705,  0.7761]
# ------------                                                      
                
                
                
                
                
                1w1w1w
                1w1w1w
                1w1w1w
                1w1w1w继续。  d 1000
                0.1 以下
                pass
            
            
            #                                                 step 1   step 2
            # dim 100
            # 0.101     result in 0.0014    correct_strength   0.035    0.002      
            # 0.151     result in 0.0018    correct_strength   0.059    0.006
            # 0.200     result in 0.0018    correct_strength   0.076    0.008
            # 0.250     result in 0.0018    correct_strength   0.094    0.012
            # 0.303     result in 0.0028    correct_strength   0.112    0.016
            # 0.351     result in 0.0028    correct_strength   0.129    0.024
            # 0.402     result in 0.0090    correct_strength   0.129    0.040
            # 0.450     result in 0.0075    correct_strength   0.170    0.040
            # 0.501     result in 0.0126    correct_strength   0.203    0.053
            # 0.553     result in 0.0165    correct_strength   0.209    0.053
            # 0.598     result in 0.0172    correct_strength   0.225    0.067
            # 0.650     result in 0.0240    correct_strength   0.247    0.080
            # 0.704     result in 0.0310    correct_strength   0.269    0.093 
            # 0.755     result in 0.0420    correct_strength   0.291    0.107 
            # 0.753     result in 0.0414    correct_strength   0.334    0.093 
            # 0.804     result in 0.0536    correct_strength   0.312    0.120   
            # 0.853     result in 0.0725    correct_strength   0.334    0.147   
            # 0.902     result in 0.0921    correct_strength   0.345    0.160  
            # 1.056     result in 0.1786    correct_strength   0.400    0.207    
            # 1.096     result in 0.2220    correct_strength   0.406    0.233   
            # 1.151     result in 0.2716    correct_strength   0.419    0.247  
            # 1.205     result in 0.3232    correct_strength   0.444    0.273   
            # 1.260     result in 0.3841    correct_strength   0.475    0.293    
            # 1.305     result in 0.4389    correct_strength   0.491    0.314    
            # 1.358     result in 0.4917    correct_strength   0.498    0.314    
            # 1.409     result in 0.5423    correct_strength   0.498    0.314  
            # 1.449     result in 0.5879    correct_strength   0.491    0.335    
            # 1.508     result in 0.6279    correct_strength   0.498    0.335   
            # 1.547     result in 0.6573    correct_strength   0.538    0.355    
            # 1.565     result in 0.6600    correct_strength   0.530    0.355     
            
            
            
            
            
            
            #-------------------#-------------------#-------------------
            dim_list =       [100,1000]
            test_time_list = [100,50]
            dim_list =       [100]
            test_time_list = [20]
            #dim_list =       [1000]
            #test_time_list = [20]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
                iota_of_dim = iota(dim)
                #<  device
                if dim>100:
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
                #</ device
                print(f"dim {dim}   test_time {test_time}  {device}")
            #-------------------#-------------------#-------------------
                
                #-------------------#-------------------#-------------------
                init_angle_loss_list = torch.linspace(1.55,1.65, 3) ################################################################################
                for target_init_angle_loss in init_angle_loss_list:
                
                # random_init__cap_to__list = torch.tensor([0.1, 0.5])            # z axis
                # random_init__noise_strength__list = torch.tensor([0. , 0.5])    # z axis
                # for _iter__z_axis in range(z_axis__dim):
                #     random_init__cap_to = random_init__cap_to__list[_iter__z_axis]
                #     random_init__noise_strength = random_init__noise_strength__list[_iter__z_axis]
                #-------------------#-------------------#-------------------

                    x_axis__dim = 33
                    y_axis__dim = 16
                    
                    score = torch.empty(size=[y_axis__dim, x_axis__dim])    #dont modify this.
                    score.fill_(torch.nan)
                    
                    #-------------------#-------------------#-------------------
                    cap_to__step_1_list = torch.linspace(0.35, 0.6, x_axis__dim) # x axis ####################################################
                    cap_to__step_2_list = torch.linspace(0.19, 0.5, y_axis__dim) # y axis
                    for _iter__cap_to__step_1 in range(cap_to__step_1_list.shape[0]): # x axis
                        cap_to__step_1 = cap_to__step_1_list[_iter__cap_to__step_1]
                    #-------------------#-------------------#-------------------
                        
                        _raw_result__score = torch.empty(size=[y_axis__dim, test_time])
                        _raw_result__score.fill_(torch.nan)
                        _raw_result__actual_init_angle_score = torch.empty(size=[test_time])
                        _raw_result__actual_init_angle_score.fill_(torch.nan)
                        
                        for _test_count in range(test_time):
                            
                            #-------------------#-------------------#-------------------
                            #<  init
                            while True:
                                ori_mat = random_dummy_mat__v2__from_target(dim=dim, target_angle_loss= target_init_angle_loss,
                                                        device=device, iota_of_dim=iota_of_dim)#or maybe the noise_0.5
                                _, angle_loss__in_the_beginning, _ = LOSS__mat_is_standard_orthogonal(ori_mat)
                                if _tensor_equal(angle_loss__in_the_beginning, target_init_angle_loss, epsilon= 0.05):
                                    break
                                #no tail.
                                pass
                            _raw_result__actual_init_angle_score[_test_count] = angle_loss__in_the_beginning
                            #</ init
                            
                            #<  calc
                            for _y_axis_count in range(y_axis__dim):
                                mat = ori_mat.detach().clone()
                                mat = full_test_version_of_angle_correction__by_row(mat,
                                        cap_to=cap_to__step_1,                 iota_of_dim=iota_of_dim)
                                mat = full_test_version_of_angle_correction__by_row(mat.T,
                                        cap_to=cap_to__step_2_list[_y_axis_count],  iota_of_dim=iota_of_dim).T
                                _, angle_loss__after, _ = LOSS__mat_is_standard_orthogonal(mat)
                                _raw_result__score[_y_axis_count, _test_count] = angle_loss__after
                                
                                #assert angle_loss__after<angle_loss__in_the_beginning
                                
                                pass# for _y_axis_count
                            
                            pass#for _test_count
                        
                        score[:, _iter__cap_to__step_1] = _raw_result__score.mean(dim=1)
                        actual_init_angle_score = _raw_result__actual_init_angle_score.mean()
                        
                        pass#   for cap_to__step_1  #x axis
                    
                    print(f"# dim {dim}      actual_init_angle_score {actual_init_angle_score.item():.3f}       best_score {score.min().item():.4f}"+" "*33)
                    #print(f"# dim {dim}      actual_init_angle_score {actual_init_angle_score.item():.3f}     (y axis is the step_2 cap_to)")
                    print(f"# step_1     = {str_the_list(cap_to__step_1_list         , 3, segment=",  ")}")
                    
                    print(f"# vvvv cap_to__step_2 "+" "*111)
                    for ii in range(y_axis__dim):
                        print(f"# {cap_to__step_2_list[ii].item():.3f}     = {str_the_list(score[ii], 4)}")
                        pass
                    print(f"# ------------"+" "*111)
                    
                    pass# for cap_to__step_3
                    
                pass# for outter_param_count
            
            pass#/ test
        
        
        
        
        
        return
    
    ____test____scan_for_the_adaptive_angle_correct()
    pass


# 不同的测量值所需要的最佳参数，rc上。之后是不是就可以根据测量值来决定参数了？？？
# 自适应。


# "bc the length test should be short, let me also write it here vvvvvv"

# 写函数

# r vs rr
# rr rc,rcr,rCr
# 长度和角度的相互影响。









#还有一个修正长度的时候会破坏角度。。。可能也要测。。。。。。。。
#还有一个修正长度的时候会破坏角度。。。可能也要测。。。。。。。。
#还有一个修正长度的时候会破坏角度。。。可能也要测。。。。。。。。

#最后就是两个合并起来测。

