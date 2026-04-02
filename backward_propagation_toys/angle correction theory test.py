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


def ____init_mat_with_row_len_is_1(dim:int, device = 'cpu', _debug__assert = False)->torch.Tensor:
    mat = torch.randn(size=[dim,dim], device=device)
    _temp_len_sqr = mat.mul(mat).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
    mul_me___to_get_unit_length = (_temp_len_sqr).pow(-0.5)
    mat = mat * (mul_me___to_get_unit_length.reshape([-1,1]).expand([-1,dim]))
    mat.requires_grad_()
    if _debug__assert:
        temp_vec_len = get_vector_length(mat[random.randint(0,dim-1)])
        assert _tensor_equal(temp_vec_len, [1.])
        pass
    return mat

if "test" and True:
    for _ in range(4312):
        ____init_mat_with_row_len_is_1(random.randint(2,1000), _debug__assert = True)
        pass
    pass
    



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
                mat = ____init_mat_with_row_len_is_1(dim)
                
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
                mat = ____init_mat_with_row_len_is_1(dim)
                
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
            mat = ____init_mat_with_row_len_is_1(dim)
            
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
            mat = ____init_mat_with_row_len_is_1(dim)
            
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
    
    if "scan 2 hyperparams." and False:
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
                        ori_mat = ____init_mat_with_row_len_is_1(dim, device=device)
                        
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
                        ori_mat = ____init_mat_with_row_len_is_1(dim, device=device)
                        
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
    
    
    
    
    
    
    1w下一个测试就是尝试找到对每一个情况来说的最佳参数组合，然后扫那个参数组合的附近。
    expansion 如果是1，就是最简单的，如果是0就全部变成相同的长度。都算简单。
    
    return 

____test____correction_method_test()



