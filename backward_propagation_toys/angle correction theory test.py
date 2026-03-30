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
    if "标准正交阵里面，随机1到n个维度对指标的影响" and True:
        
        1w
        1w
        1w继续。
        #------------------#------------------#------------------
        # dim_list =        [ 2, 10,100,1000]
        # test_time_list = [100,100,100, 20]
        # test_time_list = [10,10,10, 2]
        
        dim_list =       [ 10, 100,1000]
        test_time_list = [ 50, 30, 10]
        test_time_list = [ 5, 3, 1]
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
            # expansion_factor_list = [-2.,-1.,0.,1.,2.,3.,4.]
            # for expansion_factor in expansion_factor_list:
            #------------------#------------------#------------------
        
            len_loss__opt_value              = []  # dont modity this
            len_loss__opt_prob               = []  # dont modity this
            angle_loss__opt_value            = []  # dont modity this
            angle_loss__opt_prob             = []  # dont modity this
            length_retention_loss__opt_value = []  # dont modity this
            length_retention_loss__opt_prob  = []  # dont modity this
            #neg_behavior_similar_loss            = []  # dont modity this
                
            #------------------#------------------#------------------
            #cap_to_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1., 1.2, 1.5, 2.]
            random_vec_count_list = [1,2,3,5,10,20,50]
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
                    
                    #<  update mat
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
                avg_of_abs_of_final_grad__min   .append(_raw_result__avg_of_abs_of_final_grad.min ())
                avg_of_abs_of_final_grad__max   .append(_raw_result__avg_of_abs_of_final_grad.max ())
                avg_of_abs_of_final_grad__avg   .append(_raw_result__avg_of_abs_of_final_grad.mean())
                #neg_behavior_similar_loss       .append(_raw_result__neg_behavior_similar_loss.mean().item())
                
                pass# for expansion_factor(y axis)
            
            print(f"dim = {dim}")
            # print(f"expansion_factor_list   = {str_the_list(expansion_factor_list, 2, segment=",    ")}")
            # print(f"cap_to = {cap_to}")
            print(f"random_vec_count_list               = {random_vec_count_list}")
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
    
    
    
    
    
    
    if "the useful one" and True:
        # greater is better
        1w
        1w
        1w
        1w
        cap_to_list往两边扩宽。。
        
        另外查一下，对一个标准正交阵，如果一个向量被随机一下，指标的变化。
        
        if False:
            
            # dim = 10
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   -2.0
            # len_loss__opt_value     = [ 0.00542,  0.00272,  0.00098,  0.00210, -0.00031,  0.00025, -0.00220,  0.00276, -0.00784, -0.02533]
            # len_loss__opt_prob      = [   0.160,    0.080, XX 0.000,    0.040,    0.080,    0.040, XX-0.120,    0.080, XX-0.120, XX-0.800]
            # angle_loss__opt_value   = [ 0.14453,  0.16235,  0.01892, -0.01153, -0.04827, -0.04588, -0.04867,  0.04934, -0.19384, -0.54515]
            # angle_loss__opt_prob    = [   0.440,    0.400,    0.080,    0.080, XX-0.200, XX-0.040, XX-0.160,    0.040, XX-0.240, XX-0.800]
            # length_retention_loss__v= [ 0.00607,  0.00725,  0.00099,  0.00005, -0.00236, -0.00393, -0.00304,  0.00101, -0.00732, -0.01964]
            # length_retention_loss__p= [   0.240,    0.320,    0.160,    0.160, XX-0.240, XX-0.320, XX-0.080,    0.040, XX-0.200, XX-0.640]
            # avg_abs_final_grad__min = [ 0.10210,  0.12891,  0.13847,  0.15354,  0.16701,  0.20724,  0.26777,  0.28701,  0.37906,  0.42452]
            # avg_abs_final_grad__max = [ 0.31021,  0.50967,  0.52137,  0.67323,  0.70474,  0.90151,  0.88937,  1.12690,  1.53158,  1.95482]
            # avg_abs_final_grad__avg = [ 0.18883,  0.22155,  0.26815,  0.28589,  0.33910,  0.39315,  0.45537,  0.48617,  0.60663,  0.76317]
            # ------------->>>>>>>>>>>>>
            # dim = 10
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   -1.0
            # len_loss__opt_value     = [ 0.01788,  0.01995,  0.01573,  0.00905,  0.00584,  0.00306, -0.00146,  0.00259, -0.00141, -0.00550]
            # len_loss__opt_prob      = [   0.720,    0.640,    0.480,    0.440,    0.200,    0.120, XX 0.000,    0.080,    0.040, XX-0.320]
            # angle_loss__opt_value   = [ 0.41741,  0.37804,  0.28461,  0.18456,  0.18111,  0.12727,  0.08924,  0.07280, -0.00203, -0.26601]
            # angle_loss__opt_prob    = [ v      ,    0.920,    0.920,    0.640,    0.720,    0.520,    0.160,    0.280, XX 0.000, XX-0.480]
            # length_retention_loss__v= [ 0.01924,  0.01767,  0.01453,  0.00931,  0.00929,  0.00423,  0.00707,  0.00442, -0.00053, -0.00592]
            # length_retention_loss__p= [   0.840,    0.920,    0.720,    0.480,    0.560,    0.200,    0.400,    0.360, XX-0.040, XX-0.240]
            # avg_abs_final_grad__min = [ 0.08931,  0.10773,  0.12396,  0.13913,  0.15107,  0.17487,  0.20925,  0.24358,  0.31817,  0.35328]
            # avg_abs_final_grad__max = [ 0.16860,  0.17279,  0.19468,  0.23154,  0.25686,  0.30803,  0.33819,  0.41276,  0.52123,  0.57629]
            # avg_abs_final_grad__avg = [ 0.11538,  0.13502,  0.15372,  0.17691,  0.19251,  0.23960,  0.26464,  0.31030,  0.39486,  0.46976]
            # ------------->>>>>>>>>>>>>
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
            # dim = 10
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   3.0
            # len_loss__opt_value     = [ 0.01233,  0.01434,  0.01705,  0.01722,  0.01862,  0.02153,  0.01821,  0.01770,  0.01456,  0.01938]
            # len_loss__opt_prob      = [   0.960,    0.880,    0.960,    0.960,  v      ,    0.920,    0.880,    0.600,    0.400,    0.520]
            # angle_loss__opt_value   = [ 0.28434,  0.32312,  0.37548,  0.37197,  0.42318,  0.45994,  0.46847,  0.41111,  0.43336,  0.38395]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,    0.960]
            # length_retention_loss__v= [ 0.01056,  0.01551,  0.01626,  0.01622,  0.01808,  0.01930,  0.02304,  0.01534,  0.01776,  0.01772]
            # length_retention_loss__p= [   0.720,    0.880,    0.880,    0.880,    0.840,    0.800,  v      ,    0.840,    0.800,    0.720]
            # avg_abs_final_grad__min = [ 0.02440,  0.02093,  0.02965,  0.03438,  0.04407,  0.05242,  0.05872,  0.06638,  0.07835,  0.08935]
            # avg_abs_final_grad__max = [ 0.05471,  0.06389,  0.06831,  0.08594,  0.08796,  0.10300,  0.12312,  0.15412,  0.18489,  0.20352]
            # avg_abs_final_grad__avg = [ 0.03741,  0.04230,  0.04974,  0.05209,  0.06303,  0.07336,  0.08803,  0.09671,  0.12301,  0.15047]
            # ------------->>>>>>>>>>>>>
            # dim = 10
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.01145,  0.01356,  0.01389,  0.01747,  0.01827,  0.01520,  0.01922,  0.01678,  0.01686,  0.01906]
            # len_loss__opt_prob      = [   0.960,    0.920,    0.920,    0.920,    0.960,    0.840,    0.760,    0.720,    0.640,    0.560]
            # angle_loss__opt_value   = [ 0.24222,  0.27379,  0.31780,  0.35590,  0.37058,  0.43918,  0.42407,  0.42253,  0.37894,  0.37992]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.01185,  0.01402,  0.01522,  0.01780,  0.01605,  0.02074,  0.02168,  0.02096,  0.01777,  0.01921]
            # length_retention_loss__p= [   0.560,    0.880,    0.880,    0.840,    0.800,    0.840,  v      ,    0.960,    0.840,    0.840]
            # avg_abs_final_grad__min = [ 0.01821,  0.02290,  0.02295,  0.03144,  0.03540,  0.03417,  0.04147,  0.04794,  0.07347,  0.07649]
            # avg_abs_final_grad__max = [ 0.05185,  0.05422,  0.06633,  0.07770,  0.07736,  0.08745,  0.12078,  0.12949,  0.14854,  0.18304]
            # avg_abs_final_grad__avg = [ 0.03138,  0.03583,  0.04151,  0.04771,  0.05049,  0.06022,  0.07256,  0.08248,  0.10514,  0.12267]
            # ------------->>>>>>>>>>>>>



            # dim = 100
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   -2.0
            # len_loss__opt_value     = [ 0.00829,  0.00651,  0.00554,  0.00576,  0.00476,  0.00217, -0.00219, -0.00638, -0.01799, -0.02422]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,    0.933,    0.933,    0.400, XX-0.267, XX-0.933, XX      , XX      ]
            # angle_loss__opt_value   = [ 0.49331,  0.44718,  0.40190,  0.37503,  0.33960,  0.14956, -0.17626, -0.46910, -1.20822, -1.70278]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,    0.733, XX-0.533, XX-0.933, XX      , XX      ]
            # length_retention_loss__v= [ 0.00757,  0.00738,  0.00634,  0.00545,  0.00522,  0.00301, -0.00230, -0.00634, -0.01683, -0.02546]
            # length_retention_loss__p= [ v      ,  v      ,  v      ,    0.933,    0.933,    0.600, XX-0.467, XX-0.733, XX      , XX      ]
            # avg_abs_final_grad__min = [ 0.03351,  0.04183,  0.04980,  0.05120,  0.05772,  0.07602,  0.08610,  0.09229,  0.12047,  0.14155]
            # avg_abs_final_grad__max = [ 0.05518,  0.05520,  0.06317,  0.07143,  0.07773,  0.09748,  0.11088,  0.13616,  0.15683,  0.20519]
            # avg_abs_final_grad__avg = [ 0.04080,  0.04753,  0.05530,  0.06123,  0.06778,  0.08336,  0.09752,  0.10897,  0.13764,  0.16960]
            # ------------->>>>>>>>>>>>>
            # dim = 100
            # cap_to_list      = [左边还有 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
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
            # ------------->>>>>>>>>>>>>
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
            # dim = 100
            # cap_to_list               = [ 0.30,     0.35,     0.40,     0.45,     0.50,     0.60,     0.70,     0.80,     1.00,     1.20]
            # expansion_factor =   4.0
            # len_loss__opt_value     = [ 0.00379,  0.00438,  0.00513,  0.00565,  0.00615,  0.00700,  0.00728,  0.00762,  0.00755,  0.00777]
            # len_loss__opt_prob      = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # angle_loss__opt_value   = [ 0.25093,  0.29792,  0.33364,  0.36780,  0.38633,  0.44650,  0.47198,  0.49476,  0.51399,  0.50412]
            # angle_loss__opt_prob    = [ v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # length_retention_loss__v= [ 0.00385,  0.00423,  0.00558,  0.00594,  0.00627,  0.00716,  0.00725,  0.00724,  0.00811,  0.00795]
            # length_retention_loss__p= [   0.867,    0.933,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ,  v      ]
            # avg_abs_final_grad__min = [ 0.00592,  0.00831,  0.00756,  0.01029,  0.01018,  0.01278,  0.01192,  0.01752,  0.02207,  0.02246]
            # avg_abs_final_grad__max = [ 0.01155,  0.01448,  0.01581,  0.01676,  0.01912,  0.02408,  0.03206,  0.03134,  0.04010,  0.04915]
            # avg_abs_final_grad__avg = [ 0.00919,  0.01113,  0.01268,  0.01423,  0.01527,  0.01905,  0.02188,  0.02486,  0.03075,  0.03664]
            # ------------->>>>>>>>>>>>>
                    
                    
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
            # ------------->>>>>>>>>>>>>
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
            # ------------->>>>>>>>>>>>>
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
            # ------------->>>>>>>>>>>>>
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
            # ------------->>>>>>>>>>>>>
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
            # ------------->>>>>>>>>>>>>
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
            # ------------->>>>>>>>>>>>>
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
            # ------------->>>>>>>>>>>>>
                    
                    
                    
                    
                    
                    
                    
            
            
            
            
            
            
            
            
            
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
        
        
        
        
        
        
        
        
        
        #------------------#------------------#------------------
        # dim_list =        [ 2, 10,100,1000]
        # test_time_list = [100,100,100, 20]
        # test_time_list = [10,10,10, 2]
        
        dim_list =       [ 10, 100,1000]
        test_time_list = [ 50, 30, 10]
        #test_time_list = [ 10, 10,  2]
        #dim_list =        [10]
        #test_time_list = [ 100]
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
            expansion_factor_list = [-2.,-1.,0.,1.,2.,3.,4.]
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
                cap_to_list = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1., 1.2]
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
    
    return 

____test____correction_method_test()



