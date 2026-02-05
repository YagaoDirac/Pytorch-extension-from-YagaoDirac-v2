from typing import Any, Optional, Literal
import random
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import is_square_matrix, _tensor_equal, have_same_elements

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"


def _get_2_diff_rand_int(dim:int, device)->tuple[torch.Tensor,torch.Tensor]:
    rand_index_1 = torch.randint(0, dim,   size = [], device = device)
    rand_index_2 = torch.randint(0, dim-1, size = [], device = device)
    if rand_index_2>=rand_index_1:
        rand_index_2+=1
        pass
    return rand_index_1, rand_index_2
if "test" and __DEBUG_ME__() and True:
    def ____test_____get_2_diff_rand_int():
        for _ in range(155):
            dim = random.randint(2,300)
            #-------------------#-------------------#-------------------
            a,b = _get_2_diff_rand_int(dim, 'cpu')
            #-------------------#-------------------#-------------------
            assert a!=b
            assert a.ge(0)
            assert b.ge(0)
            assert a.lt(dim)
            assert b.lt(dim)
            pass
        return 
    ____test_____get_2_diff_rand_int()




def random_rotate(input:torch.Tensor, times:int|None = None)->torch.Tensor:
    assert is_square_matrix(input)
    dim = input.shape[0]
    the_device = input.device
    the_dtype = input.dtype
    
    if times is None:
        times = dim*3
        pass
    
    with torch.no_grad():
        times_by_row = torch.randint(0,times+1,size=[])
        times_by_column = times-times_by_row
        for _ in range(times_by_row):# each ROW vec into a new rotated ROW vec
            rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=the_device)
            angle = torch.rand(size=[], dtype = the_dtype,device=the_device)*torch.pi*2.
            _cos = angle.cos()
            _sin = angle.sin()
            input[:,[rand_index_1, rand_index_2]] = \
                input[:,[rand_index_1, rand_index_2]]@ \
                        torch.tensor(  [[ _cos.item(),_sin],
                                        [-_sin, _cos]], dtype=the_dtype, device=the_device)
            pass
        
        for _ in range(times_by_column):# each COLUMN vec into a new rotated COLUMN vec
            rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=the_device)
            angle = torch.rand(size=[], dtype = the_dtype,device=the_device)*torch.pi*2.
            _cos = angle.cos()
            _sin = angle.sin()
            input[[rand_index_1, rand_index_2]] = \
                        torch.tensor(  [[ _cos.item(),_sin],
                                        [-_sin, _cos]], dtype=the_dtype, device=the_device)@\
                input[[rand_index_1, rand_index_2]]
            pass
        return input
    pass#/function
if "test" and __DEBUG_ME__() and True:
    def ____test____random_rotate():
        for _ in range(155):
            dim = random.randint(2,6)
            #-------------#-------------#-------------
            init_eye = torch.eye(n = dim)
            random_rotate(init_eye)
            #-------------#-------------#-------------
            assert ???
            长度1的向量，乘，看看结果的长度。
            pass
        return
    ____test____random_rotate()

def random_rotation_matrix__by_once(dim:int, dtype = torch.float32, device = 'cpu')->torch.Tensor:
    result = torch.eye(n=dim, dtype=dtype, device=device)
    rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=device)
    angle = torch.rand(size=[], dtype = dtype,device=device)*torch.pi*2.
    _cos = angle.cos()
    _sin = angle.sin()
    result = torch.tensor( [[ _cos.item(),_sin],
                            [-_sin, _cos]], dtype=dtype, device=device)
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____random_rotation_matrix__by_once():
        for _ in range(155):
            dim = random.randint(2,6)
            #-------------#-------------#-------------
            mat = random_rotation_matrix__by_once(dim)
            #-------------#-------------#-------------
            assert mat.reshape([-1]).sort().values 的前dim-2个应该是1，
            
            
            然后有4个，不是1不是0的，用flag来扣。这4个应该是刚好那啥
            
            pass
        return 
    ____test____random_rotation_matrix__by_once()

def random_rotation_matrix(dim:int, times:int|None = None, dtype = torch.float32, device = 'cpu')->torch.Tensor:
    init_eye = torch.eye(n=dim, dtype=dtype, device=device)
    result = random_rotate(init_eye,times=times)
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____random_rotation_matrix():
        "bc the inner function is validated, this func doesn't need to get validated again."
        return 
    ____test____random_rotation_matrix()
    pass








def random_permutate(input:torch.Tensor, times_by_row:int|None = None, times_by_column:int|None = None)->torch.Tensor:
    assert is_square_matrix(input)
    dim = input.shape[0]
    the_device = input.device
    
    if times_by_row is None:
        times_by_row = dim*3
        pass
    if times_by_column is None:
        times_by_column = dim*3
        pass
    
    with torch.no_grad():
        for _ in range(times_by_row):
            rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=the_device)
            _temp = input[rand_index_1].clone()
            input[rand_index_1] = input[rand_index_2]
            input[rand_index_2] = _temp
            pass
        for _ in range(times_by_column):
            rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=the_device)
            _temp = input[:,rand_index_1].clone()
            input[:,rand_index_1] = input[:,rand_index_2]
            input[:,rand_index_2] = _temp
        return input
    pass#/function
if "test" and __DEBUG_ME__() and True:
    def ____test____random_permutate():
        "only row"
        for _ in range(5):
            mat = torch.tensor([[0.,1,2],[0,1,2],[0,1,2]])
            mat_after = random_permutate(mat.detach().clone(), times_by_row = random.randint(2,5),
                                            times_by_column = 0)
            #-------------------------------------------------------------------------
            assert mat.eq(mat_after).all()
            pass
        
        "only column"
        for _ in range(5):
            mat = torch.tensor([[0.,0,0],[1,1,1],[2,2,2]])
            mat_after = random_permutate(mat.detach().clone(), times_by_row = 0,
                                            times_by_column = random.randint(2,5))
            #-------------------------------------------------------------------------
            assert mat.eq(mat_after).all()
            pass
            
        "only column"
        for _ in range(115):
            dim = random.randint(2,300)
            #-------------------------------------------------------------------------
            mat = torch.randn(size=[dim,dim])
            mat_after = random_permutate(mat.detach().clone(), 
                                        times_by_row    = random.randint(dim+10,dim*3+50),
                                        times_by_column = random.randint(dim+10,dim*3+50))
            #-------------------------------------------------------------------------
            _ori_sum = mat.sum()
            _after_sum = mat_after.sum()
            assert _tensor_equal(_ori_sum, _after_sum, epsilon=_ori_sum.abs()*0.0002)
            assert _tensor_equal(mat.std(), mat_after.std())
            
            assert have_same_elements(mat, mat_after)
            pass
        return 
    ____test____random_permutate()
    pass

def random_permutation_matrix__by_once(dim:int, dtype = torch.float32, device = 'cpu')->torch.Tensor:
    result = torch.eye(n=dim, dtype=dtype, device=device)
    rand_index_1,rand_index_2 = _get_2_diff_rand_int(dim, device=device)
    result[rand_index_1,rand_index_1] = 0.
    result[rand_index_2,rand_index_2] = 0.
    result[rand_index_1,rand_index_2] = 1.
    result[rand_index_2,rand_index_1] = 1.
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____random_permutation_matrix__by_once():
        for _ in range(166):
            dim = random.randint(2,300)
            #---------------------------
            mat = random_permutation_matrix__by_once(dim)
            #---------------------------
            assert mat.sum() == dim
            
            _ref_mat = torch.zeros_like(mat)
            _ref_mat[0] = 1.
            assert have_same_elements(mat, _ref_mat)
            
            assert mat.det().abs() == 1
            pass
        
        "if multiple this thing multiply together"
        for _ in range(15):
            dim = random.randint(2,88)
            amount = random.randint(2,10)
            #---------------------------
            mat = random_permutation_matrix__by_once(dim)
            for _the_number in range(amount-1):
                new_mat = random_permutation_matrix__by_once(dim)
                mat = mat@new_mat
                pass
            #---------------------------
            assert mat.sum() == dim
            
            _ref_mat = torch.zeros_like(mat)
            _ref_mat[0] = 1.
            assert have_same_elements(mat, _ref_mat)
            
            assert mat.det().abs() == 1
            pass
        
        
        return 
    ____test____random_permutation_matrix__by_once()
    pass

def random_permutation_matrix(dim:int, times:int|None = None, dtype = torch.float32, device = 'cpu')->torch.Tensor:
    init_eye = torch.eye(n=dim, dtype=dtype, device=device)
    if times is None:
        times = dim*3
        pass
    row_times = times//2
    column_times = times-row_times
    result = random_permutate(init_eye,times_by_row=row_times,times_by_column=column_times)
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____random_permutation_matrix():
        for _ in range(166):
            dim = random.randint(2,300)
            #---------------------------
            mat = random_permutation_matrix(dim,dim*5)
            #---------------------------
            assert mat.sum() == dim
            
            _ref_mat = torch.zeros_like(mat)
            _ref_mat[0] = 1.
            assert have_same_elements(mat, _ref_mat)
            
            assert mat.det().abs() == 1
            pass
        return 
    ____test____random_permutation_matrix()
    pass
