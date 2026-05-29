import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, iota
#from pytorch_yagaodirac_v2.Random import random_rotation_matrix,angle_to_rotation_matrix_2d
# from pytorch_yagaodirac_v2.Random import random_symmetric_matrix, angle_to_rotation_matrix_2d
# import random, time, math
# import matplotlib.pyplot as plt






if "torch feature test":
    def ____torch_lu_decomp_feature_test()->None:
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
    ____torch_lu_decomp_feature_test()


fds=432






1w 继续

dim_big = 3
dim_small = dim_big-1

mat_big = torch.randn(size=[dim_big,dim_big], device='cuda')
P:torch.Tensor
big_L:torch.Tensor
big_U:torch.Tensor
P, big_L, big_U = torch.linalg.lu(mat_big, pivot = False)
assert P.nelement() == 0
assert big_L.isnan().any() == False
assert big_U.isnan().any() == False

mat_small = mat_big.detach()[:dim_small,:dim_small]
检查一下内存是不是断开的。just in case

small_L:torch.Tensor
small_U:torch.Tensor
P, small_L, small_U = torch.linalg.lu(mat_small, pivot = False)
assert P.nelement() == 0
assert small_L.isnan().any() == False
assert small_U.isnan().any() == False

assert _tensor_equal(big_L[:dim_small,:dim_small], small_L)
assert _tensor_equal(big_U[:dim_small,:dim_small], small_U)


检查下面的函数
torch.linalg.lu_solve
torch.linalg.lu_factor
torch.linalg.lu_factor_ex








