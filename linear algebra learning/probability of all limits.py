import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, iota
#from pytorch_yagaodirac_v2.Random import random_rotation_matrix,angle_to_rotation_matrix_2d
from pytorch_yagaodirac_v2.Random import random_symmetric_matrix
import random, time
import matplotlib.pyplot as plt

def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######


if "dtype test" and False:
    #mat = torch.randn(size=[5], dtype=torch.float8_e4m3fn)
    #mat = torch.randn(size=[5], dtype=torch.float8_e4m3fnuz)
    #mat = torch.randn(size=[5], dtype=torch.float8_e5m2)
    #mat = torch.randn(size=[5], dtype=torch.float8_e5m2fnuz)
    #mat = torch.randn(size=[5], dtype=torch.float8_e8m0fnu)
    #mat = torch.randn(size=[5], dtype=torch.float4_e2m1fn_x2)
    mat = torch.randn(size=[5], dtype=torch.float16)
    mat = torch.randn(size=[5], dtype=torch.bfloat16)
    mat = torch.randn(size=[5], dtype=torch.float32)
    mat = torch.randn(size=[5], dtype=torch.float64)

    #gtx 1660
    #mat = torch.randn(size=[5], device='cuda', dtype=torch.float8_e4m3fn)
    #mat = torch.randn(size=[5], device='cuda', dtype=torch.float8_e4m3fnuz)
    #mat = torch.randn(size=[5], device='cuda', dtype=torch.float8_e5m2)
    #mat = torch.randn(size=[5], device='cuda', dtype=torch.float8_e5m2fnuz)
    #mat = torch.randn(size=[5], device='cuda', dtype=torch.float8_e8m0fnu)
    #mat = torch.randn(size=[5], device='cuda', dtype=torch.float4_e2m1fn_x2)
    mat = torch.randn(size=[5], device='cuda', dtype=torch.float16)
    mat = torch.randn(size=[5], device='cuda', dtype=torch.bfloat16)
    mat = torch.randn(size=[5], device='cuda', dtype=torch.float32)
    mat = torch.randn(size=[5], device='cuda', dtype=torch.float64)
    pass



if "prob of a randn mat is not inversable." and False:

    # dim 2      test 10000   fail 0   >5 0.2081   >10 0.1017  >30 0.0333  >100 0.0107
    # dim 10     test 10000   fail 0   >5 0.2308   >10 0.1115  >30 0.0373  >100 0.0114
    # dim 100    test 10000   fail 0   >5 0.1499   >10 0.0767  >30 0.0268  >100 0.0070
    # dim 1000   test 10000   fail 0   >5 0.0686   >10 0.0329  >30 0.0117  >100 0.0033
    # dim 3000   test  1000   fail 0   >5 0.0510   >10 0.0260  >30 0.0050  >100 0.0020

    print(f"__LINE__ {_line_()}")

    dtype= torch.float
    dim_list =                          [  2,    10,  100,   1000, 3000]
    number_of_tests_list = torch.tensor([10000,10000,10000, 10000, 1000])
    number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
    for outter_param_set in range(dim_list.__len__()):
        dim = dim_list[outter_param_set]
        # iota_of_dim = iota(dim)
        number_of_tests = number_of_tests_list[outter_param_set]
        device = 'cpu'
        #device = 'cuda'
        if dim>100:
            device = 'cuda'
            pass
        the_eye = torch.eye(n=dim, dtype=dtype, device=device)
        total_fail_times = 0
        top_element_gt_5 = 0
        top_element_gt_10 = 0
        top_element_gt_30 = 0
        top_element_gt_100 = 0
        
        _when_start = time.perf_counter()
        
        for _ in range(number_of_tests):
                
            #mat = torch.tensor([[1.,2],[1.,2]])
            mat = torch.randn(size=[dim,dim], dtype=dtype, device=device)
            it_works = True
            
            try:
                inv_mat = torch.linalg.solve(mat, the_eye)
                pass
            except RuntimeError as e:
                total_fail_times +=1
                it_works = False
                #print(f"Caught a RuntimeError: {e}")
                pass
            
            if it_works:
                the_max = inv_mat.abs().max()
                if the_max>5.:
                    top_element_gt_5 +=1
                    pass
                if the_max>10.:
                    top_element_gt_10 +=1
                    pass
                if the_max>30.:
                    top_element_gt_30 +=1
                    pass
                if the_max>100.:
                    top_element_gt_100 +=1
                    pass
                pass
            pass
        _when_end = time.perf_counter()
        print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
        
        print(f"dim {dim}   test {number_of_tests}   fail {total_fail_times}   >5 { \
                top_element_gt_5/number_of_tests:.4f}   >10 {top_element_gt_10/number_of_tests:.4f}  >30 { \
                top_element_gt_30/number_of_tests:.4f}  >100 {top_element_gt_100/number_of_tests:.4f}")
        pass#for dim
    pass#/test

if "prob of for a randn mat to have complex eigen values." and False:
    # dim 2      test 10000   complex ratio 0.29
    # dim 10     test 10000   complex ratio 0.71
    # dim 100    test 300     complex ratio 0.92
    # dim 1000   test 15      complex ratio 0.97
    # dim 3000   test 2       complex ratio 0.98
    
    print(f"__LINE__ {_line_()}")

    dtype= torch.float
    dim_list =                          [  2,    10,   100, 1000, 3000]
    number_of_tests_list = torch.tensor([10000,10000,   300,  15,  2])
    number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
    for outter_param_set in range(dim_list.__len__()):
        dim = dim_list[outter_param_set]
        # iota_of_dim = iota(dim)
        number_of_tests = number_of_tests_list[outter_param_set]
        device = 'cpu'
        #device = 'cuda'
        # if dim>100:
        #     device = 'cuda'
        #     pass
        complex_eig_val_count = torch.tensor(0, device=device)
        
        _when_start = time.perf_counter()
        
        for _ in range(number_of_tests):
                
            #mat = torch.tensor([[1.,2],[1.,2]])
            mat = torch.randn(size=[dim,dim], dtype=dtype, device=device)
            
            eig_vals_complex:torch.Tensor
            eig_vals_complex, _ = torch.linalg.eig(mat)
            
            complex_eig_val_count +=eig_vals_complex.imag.abs().gt(1e-3).sum()
            pass
        _when_end = time.perf_counter()
        #print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
        
        print(f"dim {dim:5}   test {number_of_tests:6}   complex ratio {complex_eig_val_count/dim/number_of_tests:.2f}")
        pass#for dim
    pass#/test

if "prob of for a symmetric randn mat to have complex eigen values." and False:
    # but notice the init algo is M+M.T.
    #eig_vals.abs().max()  /sqrt(dim) == 1.41   when dim >=100
    #eig_vals.abs().top 90%/sqrt(dim) == 1.14   when dim >=100
    #eig_vals.abs().mean() /sqrt(dim) == 0.60   when dim >=100
    
    print(f"__LINE__ {_line_()}")
    
    dtype= torch.float
    dim_list =                          [  10,   100, 1000, 3000]
    number_of_tests_list = torch.tensor([10000,   300,  15,  2])
    number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
    for outter_param_set in range(dim_list.__len__()):
        dim = dim_list[outter_param_set]
        # iota_of_dim = iota(dim)
        number_of_tests = int(number_of_tests_list[outter_param_set].item())
        device = 'cpu'
        #device = 'cuda'
        # if dim>100:
        #     device = 'cuda'
        #     pass
        total__complex_eig_val_count = torch.tensor(0, device=device)
        
        raw_result__max = torch.empty(size=[number_of_tests])
        raw_result__max.fill_(torch.nan)
        raw_result__top90 = torch.empty(size=[number_of_tests])
        raw_result__top90.fill_(torch.nan)
        raw_result__avg = torch.empty(size=[number_of_tests])
        raw_result__avg.fill_(torch.nan)
        
        _when_start = time.perf_counter()
        
        for ii in range(number_of_tests):
            #<  init
            mat = random_symmetric_matrix(dim, dtype=dtype, device=device)
            
            #<  calc eigen
            eig_vals_complex, _ = torch.linalg.eig(mat)
            assert isinstance(eig_vals_complex, torch.Tensor)
            
            #<  measure    
            #< has complex eigen?
            _complex_eig_val_count = eig_vals_complex.imag.abs().gt(1e-3).sum()
            total__complex_eig_val_count +=_complex_eig_val_count
            del _complex_eig_val_count
            
            eig_vals__abs = eig_vals_complex.real.abs()
            #< max and top90
            _how_many = int(dim*0.1)
            _top_element = eig_vals__abs.topk(_how_many).values
            raw_result__max[ii] = _top_element[0]
            raw_result__top90[ii] = _top_element[_how_many-1]
            del _how_many, _top_element
            #< avg
            raw_result__avg[ii] = eig_vals__abs.mean()
            
            pass
        
        plt.hist(raw_result__max/torch.tensor(dim, dtype=torch.float32).sqrt(), bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('the max')
        plt.ylabel('hist')
        plt.title(f'dim {dim}')
        plt.show()
        
        plt.hist(raw_result__top90/torch.tensor(dim, dtype=torch.float32).sqrt(), bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('the top90')
        plt.ylabel('hist')
        plt.title(f'dim {dim}')
        plt.show()
        
        plt.hist(raw_result__avg/torch.tensor(dim, dtype=torch.float32).sqrt(), bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('the avg')
        plt.ylabel('hist')
        plt.title(f'dim {dim}')
        plt.show()
        
        _when_end = time.perf_counter()
        #print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
        pass#for dim
    pass#/test







