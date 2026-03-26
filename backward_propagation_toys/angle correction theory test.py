import datetime
from pathlib import Path
import math, random
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal, \
    iota, is_square_matrix, \
    vector_length_norm, get_vector_length,\
    log10_avg_safe, log10_avg__how_similar, get_mask_of_top_element__rough,\
    str_the_list
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



def ____test____basic_order_of_magnitude_test():
    
    "0.58 to 0.82"    
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
                #<  init
                mat = torch.randn(size=[dim,dim])
                _temp_len_sqr = mat.mul(mat).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
                mul_me___to_get_unit_length = (_temp_len_sqr).pow(-0.5)
                mat = mat * (mul_me___to_get_unit_length.reshape([-1,1]).expand([-1,dim]))
                mat.requires_grad_()
                assert mat.abs().flatten().max()<=1.00001
                
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
    
    if "visualization" and True:
        dim = 1000
        _div_me = 4*(dim-1)/(dim*dim)    *    (dim/1.42)
        #    part 1111111111111111111    part 2222222222
        #part 1 is from formula, part 2 is from tested distribution.
        # so the result should be around 1.
        iota_of_dim = iota(dim)
        1w
        1w
        1w包括自适应缩放一起测试了。后面就直接用了。
        考虑一下长度按最大的作为1，其他的小于1，直接n次方，扫这个n
        for test_count in range(123123123):
            
            #--------------------#--------------------#--------------------
            #<  init
            mat = torch.randn(size=[dim,dim])
            _temp_len_sqr = mat.mul(mat).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
            mul_me___to_get_unit_length = (_temp_len_sqr).pow(-0.5)
            mat = mat * (mul_me___to_get_unit_length.reshape([-1,1]).expand([-1,dim]))
            mat.requires_grad_()
            assert mat.abs().flatten().max()<=1.00001
            
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
    
    
____test____basic_order_of_magnitude_test()





