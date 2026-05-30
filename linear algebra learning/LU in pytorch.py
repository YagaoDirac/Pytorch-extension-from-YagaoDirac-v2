import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, iota
#from pytorch_yagaodirac_v2.Random import random_rotation_matrix,angle_to_rotation_matrix_2d
# from pytorch_yagaodirac_v2.Random import random_symmetric_matrix, angle_to_rotation_matrix_2d
import random, time, math
# import matplotlib.pyplot as plt
from pytorch_yagaodirac_v2.Util import str_the_list






if "torch feature test" and False:
    def ____torch_linalg_lu_func_test()->None:
        P:torch.Tensor
        L:torch.Tensor
        U:torch.Tensor
        
        '''
        pivot = True is DEFAULT, it's PLU decomposation, or permutation+LU.
        When processing the ith column, the also changes lines to make the a_ii element
        the greatest among a_ii through a_ni, so the factor is always inside (-1,1). 
        This design is for the numeric stability.
        In pytorch, this also is only implemented in gpu.
        
        pivot = False and it's LU
        In this case, the also never changes lines to make sure the numeric stability.
        The result may be useless, like it can have nan and inf in it.
        '''
        
        if "pivot=False is permutation+LU" and True:
            ''' 1 . .
                . 1 .
                . . 1'''
            mat = torch.tensor([[1., 0,  0],
                                [0., 1,  0],
                                [0., 0,  1],])
            P,L,U = torch.linalg.lu(mat)
            assert _tensor_equal(P, torch.eye(n=3))
            assert _tensor_equal(L, torch.eye(n=3))
            assert _tensor_equal(U, torch.eye(n=3))
            
            ''' 1 . .
                . . 1
                . 1 .'''
            mat = torch.tensor([[1., 0,  0],
                                [0., 0,  1],
                                [0., 1,  0],])
            P,L,U = torch.linalg.lu(mat)
            assert _tensor_equal(P, mat)
            assert _tensor_equal(L, torch.eye(n=3))
            assert _tensor_equal(U, torch.eye(n=3))
            
            # the U is always triu(upper triangle matrix), non zero elements are ignored.
            mat = torch.tensor([[1., 0,  0],
                                [0., 0,  1],
                                [0., 1,  0],], device= 'cuda')
            P,L,U = torch.linalg.lu(mat, pivot = False)
            assert P.nelement() == 0
            L = L.cpu()
            U = U.cpu()
            assert _tensor_equal(L, torch.eye(n=3))
            assert _tensor_equal(U, torch.tensor([[  1., 0., 0.],
                                                    [0., 0., 1.],
                                                    [0., 0., 0.]]))
            
            ''' 1 . . .
                . . 4 .
                . 1 1 .
                . . 2 1'''
            mat = torch.tensor([[1., 0,  0,  0],
                                [0., 0,  4,  0],
                                [0., 1,  1,  0],
                                [0., 0,  2,  1],], device= 'cuda')
            #1 0 0 0 P(2,3) 1 0 0 0 E   1 0 0 0
            #0 0 4 0 >>>    0 1 1 0 >>> 0 1 1 0
            #0 1 1 0        0 0 4 0     0 0 4 0
            #0 0 2 1        0 0 2 1     0 0 0 1
            P,L,U = torch.linalg.lu(mat, pivot = True)
            P = P.cpu()
            L = L.cpu()
            U = U.cpu()
            assert _tensor_equal(P, torch.tensor([[  1., 0,  0,  0],
                                                    [0., 0,  1,  0],
                                                    [0., 1,  0,  0],
                                                    [0., 0,  0,  1],]))
            assert _tensor_equal(L, torch.tensor([[  1., 0,  0,  0],
                                                    [0., 1,  0,  0],
                                                    [0., 0,  1,  0],
                                                    [0., 0,  0.5,  1],]))
            #                                       the only ^^^ 0.5
            assert _tensor_equal(U, torch.tensor([[  1., 0,  0,  0],
                                                    [0., 1,  1,  0],
                                                    [0., 0,  4,  0],
                                                    [0., 0,  0,  1],]))
            
            ''' when pivot is False, if any aii is 0, then all the 
            elements below are ignored.
                1 . .
                . . 1
                1 1 2'''
            mat = torch.tensor([[1., 0,  0],
                                [0., 0,  1],
                                [1., 1,  2],], device= 'cuda')
            #1 0 0 E    1 0 0 Ignore    1 0 0
            #0 0 1 >>>  0 0 1 >>>       0 0 1
            #1 1 2      0 1 2           0 0 2
            P,L,U = torch.linalg.lu(mat, pivot = False)
            assert P.nelement() == 0
            L = L.cpu()
            U = U.cpu()
            assert _tensor_equal(L, torch.tensor([[  1., 0., 0.],
                                                    [0., 1., 0.],
                                                    [1., 0., 1.]]))
            assert _tensor_equal(U, torch.tensor([[  1., 0., 0.],
                                                    [0., 0., 1.],
                                                    [0., 0., 2.]]))
            
            ''' when pivot is False, if any aii is 0, then all the 
            elements below are ignored.
                1 . . .
                2 0 . .
                3 5 1 .
                4 6 7 1'''
            mat = torch.tensor([[1., 0,  0,  0],
                                [2., 0,  0,  0],
                                [3., 5,  1,  0],
                                [4., 6,  7,  1],], device= 'cuda')
            #1 . . . E   1 . . . Ignore 1 . . . E   1 . . . 
            #2 0 . . >>> . 0 . . >>>    . 0 . . >>> . 0 . . 
            #3 5 1 .     . 5 1 .        . . 1 .     . . 1 . 
            #4 6 7 1     . 6 7 1        . . 7 1     . . . 1 
            P,L,U = torch.linalg.lu(mat, pivot = False)
            assert P.nelement() == 0
            L = L.cpu()
            U = U.cpu()
            assert _tensor_equal(L, torch.tensor([[  1., 0., 0., 0.],
                                                    [2., 1., 0., 0.],
                                                    [3., 0., 1., 0.],
                                                    [4., 0., 7., 1.]]))
            assert _tensor_equal(U, torch.tensor([[  1., 0., 0., 0.],
                                                    [0., 0., 0., 0.],
                                                    [0., 0., 1., 0.],
                                                    [0., 0., 0., 1.]]))
            pass
        
        if "the numeric stability design. It only works on gpu.":
            ''' For every column, the greatest element is permutated to the aii position.
                1 1 .
                2 3 .
                4 4 .'''
            mat = torch.tensor([[1., 1,  0],
                                [2., 3,  0],
                                [4., 4,  0],], device= 'cuda')
            # 1 1 0  P(1,3) 4 4 0 E     4 4 0
            # 2 3 0  >>>    2 3 0 >>>   . 1 0
            # 4 4 0         1 1 0       . 0 0    
            P,L,U = torch.linalg.lu(mat, pivot = True)
            P = P.cpu()
            assert _tensor_equal(P, torch.tensor([[  0., 0., 1.],
                                                    [0., 1., 0.],
                                                    [1., 0., 0.]]))
            L = L.cpu()
            U = U.cpu()
            assert _tensor_equal(L, torch.tensor([[  1.,  0., 0.],
                                                    [0.5, 1., 0.],
                                                    [0.25,0., 1.]]))
            assert _tensor_equal(U, torch.tensor([[  4., 4., 0.],
                                                    [0., 1., 0.],
                                                    [0., 0., 0.]]))
            
            ''' this one permutates twice
            Notice the second permutation also affects the L in result.
                1 1 .
                2 3 .
                4 4 .'''
            mat = torch.tensor([[1., 3,  0],
                                #    ^ here it's 3.
                                [2., 3,  0],
                                [4., 4,  0],], device= 'cuda')
            # 1 3 .  P(1,3) 4 4 . E   4 4 . P(2,3) 4 4 . E   4 4 .
            # 2 3 .  >>>    2 3 . >>> . 1 . >>>    . 2 . >>> . 2 .
            # 4 4 .         1 3 .     . 2 .        . 1 .     . . .
            #                           2 it greater than 1, so...
            P,L,U = torch.linalg.lu(mat, pivot = True)
            P = P.cpu()
            assert _tensor_equal(P, torch.tensor([[  0., 1., 0.],
                                                    [0., 0., 1.],
                                                    [1., 0., 0.]]))
            L = L.cpu()
            U = U.cpu()
            assert _tensor_equal(L, torch.tensor([[  1.,  0.,  0.],
                                                    [0.25, 1.,  0.],
                                                    [0.5,0.5, 1.]]))
            assert _tensor_equal(U, torch.tensor([[  4., 4., 0.],
                                                    [0., 2., 0.],
                                                    [0., 0., 0.]]))
            
            
            pass
        
        return 
    #____torch_linalg_lu_func_test()
    
    def ____other_funcs_test()->None:
        P:torch.Tensor
        L:torch.Tensor
        U:torch.Tensor
        
        if "torch.linalg.lu":
            ''' 1 . .
                . . 1
                . 1 .'''
            mat = torch.tensor([[1., 0,  0],
                                [0., 0,  1],
                                [0., 1,  0],])
            P,L,U = torch.linalg.lu(mat)
            assert _tensor_equal(P, mat)
            assert _tensor_equal(L, torch.eye(n=3))
            assert _tensor_equal(U, torch.eye(n=3))
            
            mat_gpu = torch.tensor([[1., 0,  0],
                                    [0., 0,  1],
                                    [0., 1,  0],], device= 'cuda')
            P,L,U = torch.linalg.lu(mat_gpu, pivot = False)
            P = P.cpu()
            L = L.cpu()
            U = U.cpu()
            assert P.nelement() == 0
            assert _tensor_equal(L, torch.eye(n=3))
            assert _tensor_equal(U, torch.tensor([[  1., 0,  0],
                                                    [0., 0,  1],
                                                    [0., 0,  0],]))
            pass
        
        LU:torch.Tensor
        pivots:torch.Tensor
        iota_3 = iota(3)
        
        if "torch.linalg.lu_factor.":
            ''' 1 . .    with pivot
                . . 2
                . 3 .'''
            mat = torch.tensor([[1., 0,  0],
                                [0., 0,  2],
                                [0., 3,  0],])
            LU, pivots = torch.linalg.lu_factor(mat)
            assert _tensor_equal(LU, torch.tensor([[ 1., 0,  0],
                                                    [0., 3,  0],
                                                    [0., 0,  2],]))
            # comparing to lu???
            P,L,U = torch.linalg.lu(mat)
            P = P.cpu()
            L = L.cpu()
            U = U.cpu()
            assert _tensor_equal(LU.triu(), U)
            temp_L = LU.tril()
            temp_L[iota_3,iota_3] = 1.
            assert _tensor_equal(temp_L, L)
            
            
            ''' 1 . .    without pivot
                . 1 2
                . 3 1'''
            mat_gpu = torch.tensor([[1., 0,  0],
                                    [0., 1,  2],
                                    [0., 3,  1],], device='cuda')
            LU, pivots = torch.linalg.lu_factor(mat_gpu, pivot = False)
            LU = LU.cpu()
            pivots = pivots.cpu()
            assert _tensor_equal(LU, torch.tensor([[ 1., 0,  0],
                                                    [0., 1,  2],
                                                    [0., 3, -5],]))
            # comparing to lu???
            P,L,U = torch.linalg.lu(mat_gpu, pivot = False)
            P = P.cpu()
            L = L.cpu()
            U = U.cpu()
            assert _tensor_equal(LU.triu(), U)
            temp_L = LU.tril()
            temp_L[iota_3,iota_3] = 1.
            assert _tensor_equal(temp_L, L)
            pass
        
        '''torch.linalg.lu_factor_ex is “experimental” and it may change in a future PyTorch release.
        So, no test for it.'''
        
        '''torch.linalg.lu_solve doesn't do the LU decomposation. No test for it'''
        
        return 
    ____other_funcs_test()
    
    pass


if "LU of chunked matrix" and True:
    def ____LU_of_chunked_matrix()->None:
        #------------------#------------------#------------------
        dim_list =                          [ 3,   10,  100, 1000, 3000   ]
        number_of_tests_list = torch.tensor([2000,2000,2000, 1000,  100   ])
        number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
        for ii_outter_param_set in range(dim_list.__len__()):
            dim_big = dim_list[ii_outter_param_set]
            dim_small = dim_big-1
            assert dim_small>1
            # iota_of_dim = iota(dim)
            number_of_tests = number_of_tests_list[ii_outter_param_set]
            print(f"dim {dim_big}   test_time {number_of_tests}   ")
        #------------------#------------------#------------------
            
            _when_start = time.perf_counter()
            total_try_times = 0
        
            for ii__test in range(number_of_tests):
                
                #------------------#------------------#------------------
                #<  generate the big matrix
                while True:
                    total_try_times += 1
                    
                    mat_big = torch.randn(size=[dim_big,dim_big], device='cuda')
                    P:torch.Tensor
                    big_L:torch.Tensor
                    big_U:torch.Tensor
                    P, big_L, big_U = torch.linalg.lu(mat_big, pivot = False)
                    big_L = big_L.cpu()
                    big_U = big_U.cpu()
                    assert P.nelement() == 0
                    if (not big_L.isnan().any()) and (big_U.isnan().any() == False):
                        break
                    pass#while true 

                #<  small matrix
                mat_small = mat_big.detach().clone()[:dim_small,:dim_small]#clone, otherwise they share memory

                small_L:torch.Tensor
                small_U:torch.Tensor
                P, small_L, small_U = torch.linalg.lu(mat_small, pivot = False)
                assert P.nelement() == 0
                small_L = small_L.cpu()
                small_U = small_U.cpu()
                assert small_L.isnan().any() == False
                assert small_U.isnan().any() == False

                #<  assertion
                assert _tensor_equal(big_L[:dim_small,:dim_small], small_L)
                assert _tensor_equal(big_U[:dim_small,:dim_small], small_U)
                #------------------#------------------#------------------
                
                pass#for ii__test
            _when_end = time.perf_counter()
            
            print(f"   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
            print(f"dim {dim_big}/{dim_small}         total bad init times {total_try_times-number_of_tests}    number of test{number_of_tests}")#########################
            print()
            pass#for ii_outter_param_set
        #pass#/ test


        return
    
    ____LU_of_chunked_matrix()









    












