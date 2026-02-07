from typing import Any, Optional, Literal
import random
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import is_square_matrix, _tensor_equal, have_same_elements, \
    get_vector_length, iota

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"


"random vector"

def random_standard_vector(dim:int, dtype = torch.float32, device='cpu')->torch.Tensor:
    result = torch.randn(size=[dim], dtype=dtype, device=device).\
                                div(torch.tensor(dim, dtype=torch.float64).sqrt().to(dtype))
    length_sqr = result.dot(result).sum()
    while True:
        if length_sqr>0.001 and length_sqr<1000.:
            break
        #tail
        result = torch.randn(size=[dim], dtype=dtype, device=device).\
                                div(torch.tensor(dim, dtype=torch.float64).sqrt().to(dtype))
        length_sqr = result.dot(result).sum()
        pass
    result = result/(length_sqr.sqrt())
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test_____get_2_diff_rand_int():
        for _ in range(16):
            dim = random.randint(2,300)
            vec = random_standard_vector(dim)
            assert vec.dtype == torch.float32
            assert _tensor_equal(get_vector_length(vec), [1.]) 
            pass
        
        vec = random_standard_vector(5, torch.float64, 'cuda')
        assert vec.dtype == torch.float64
        assert vec.device.type == 'cuda'
        return
    ____test_____get_2_diff_rand_int()
    pass
if "log10 test    copy pasted from Util.py" and __DEBUG_ME__() and True:
    #result: basically the log10 of a standard vec is -0.5*log10(dim)-0.21
    def ____test____log10_avg_safe____standard_vec():
        import math, random
        from pytorch_yagaodirac_v2.Util import log10_avg_safe
        for _ in range(0):
            vec = random_standard_vector(1)
            assert _tensor_equal(get_vector_length(vec), [1.])
            log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
            assert _tensor_equal(log_10_of_vec, [0.])
            
            vec = random_standard_vector(100)
            assert _tensor_equal(get_vector_length(vec), [1.])
            log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
            #assert _tensor_equal(log_10_of_vec, [-1.16], epsilon=0.12)#maybe unstable
            assert _tensor_equal(log_10_of_vec, [-1.16], epsilon=0.16)
            
            vec = random_standard_vector(10000)
            assert _tensor_equal(get_vector_length(vec), [1.])
            log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
            #assert _tensor_equal(log_10_of_vec, [-2.21], epsilon=0.08)#maybe unstable
            assert _tensor_equal(log_10_of_vec, [-2.21], epsilon=0.11)
            pass
        
        for _ in range(0):
            _ref_log10 = random.random()*0.+1.
            dim = int(math.pow(10,_ref_log10))
            assert dim == 10 
            vec = random_standard_vector(dim)
            assert _tensor_equal(get_vector_length(vec), [1.])
            log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
            assert log_10_of_vec>-1.16 and log_10_of_vec<-0.45# 0.8
            pass
        
        for _ in range(0):
            _ref_log10 = random.random()*0.+2.
            dim = int(math.pow(10,_ref_log10))
            assert dim == 100 
            vec = random_standard_vector(dim)
            assert _tensor_equal(get_vector_length(vec), [1.])
            log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
            assert log_10_of_vec>-1.29 and log_10_of_vec<-1.05# 1.17
            pass
        
        for _ in range(0):
            _ref_log10 = random.random()*0.+3.
            dim = int(math.pow(10,_ref_log10))
            assert dim == 1000 
            vec = random_standard_vector(dim)
            assert _tensor_equal(get_vector_length(vec), [1.])
            log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
            assert log_10_of_vec>-1.79 and log_10_of_vec<-1.61# 1.7
            pass
        
        for _ in range(0):
            _ref_log10 = random.random()*0.+4.
            dim = int(math.pow(10,_ref_log10))
            assert dim == 10000 
            #assert dim >= 10 and dim <= 100000
            vec = random_standard_vector(dim, device='cpu')
            assert _tensor_equal(get_vector_length(vec), torch.tensor([1.], device='cpu'))
            log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
            #_ref = (_ref_log10*0.5* 1.05 + 0.05)*-1.
            #assert _tensor_equal(log_10_of_vec, [-0.69], epsilon=0.22)
            assert log_10_of_vec>-2.29 and log_10_of_vec<-2.13# 2.21
            #print(log_10_of_vec, [_ref])
            pass
        
        for _ in range(0):
            _ref_log10 = random.random()*0.+5.
            dim = int(math.pow(10,_ref_log10))
            assert dim == 100000
            #assert dim >= 10 and dim <= 100000
            vec = random_standard_vector(dim, device='cpu')
            assert _tensor_equal(get_vector_length(vec), torch.tensor([1.], device='cpu'))
            log_10_of_vec = log10_avg_safe(vec.reshape([1,-1]))
            #_ref = (_ref_log10*0.5* 1.05 + 0.05)*-1.
            #assert _tensor_equal(log_10_of_vec, [-0.69], epsilon=0.22)
            assert log_10_of_vec>-2.78 and log_10_of_vec<-2.64# 2.71
            #print(log_10_of_vec, [_ref])
            pass
        
        return 
    ____test____log10_avg_safe____standard_vec()
    pass

def _get_2_diff_rand_int(dim:int, device)->tuple[torch.Tensor,torch.Tensor]:
    '''return index_tensor.
    
    shape:torch.Size([2])
    
    The 2 elements are different.'''
    assert dim>=2
    rand_index_1 = torch.randint(0, dim,   size = [], device = device)
    rand_index_2 = torch.randint(0, dim-1, size = [], device = device)
    if rand_index_2>=rand_index_1:
        rand_index_2+=1
        pass
    stacked = torch.stack([rand_index_1,rand_index_2])
    return stacked#, rand_index_1, rand_index_2
if "test" and __DEBUG_ME__() and True:
    def ____test_____get_2_diff_rand_int():
        for _ in range(155):
            dim = random.randint(2,300)
            #-------------------#-------------------#-------------------
            index_tensor = _get_2_diff_rand_int(dim, 'cpu')
            #-------------------#-------------------#-------------------
            assert index_tensor.dtype == torch.int64
            assert index_tensor.shape == torch.Size([2])
            assert index_tensor[0]!=index_tensor[1]
            assert index_tensor.ge(0).all()
            assert index_tensor.lt(dim).all()
            pass
        return 
    ____test_____get_2_diff_rand_int()



"random sqr matrix"

def angle_to_rotation_matrix_2d(angle:float|torch.Tensor)->torch.Tensor:
    if isinstance(angle, float):
        angle = torch.tensor(angle)
        pass
    _cos = angle.cos()
    _sin = angle.sin()
    result = torch.tensor( [[ _cos, _sin],
                            [-_sin, _cos]])
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test_____angle_to_rotation_matrix_2d():
        import math
        mat = angle_to_rotation_matrix_2d(0.1)
        assert _tensor_equal(mat,  [[ math.cos(0.1),math.sin(0.1)],
                                    [-math.sin(0.1),math.cos(0.1)]])
        mat = angle_to_rotation_matrix_2d(torch.pi/6.)
        assert _tensor_equal(mat,  [[ math.sqrt(3.)/2, 0.5],
                                    [-0.5,             math.sqrt(3.)/2]])
        return
    ____test_____angle_to_rotation_matrix_2d()
    pass



"   random rotation"
if "random_rotate algo test" and __DEBUG_ME__() and True:
    "column"
    dim = 3
    input = torch.linspace(0,dim*dim-1,dim*dim).reshape([dim,dim])
    #index_tensor ,_,_ = _get_2_diff_rand_int(dim,'cpu')
    index_tensor = torch.tensor([0,1])
    #angle = torch.rand(size=[])*torch.pi*2.
    angle = torch.tensor(torch.pi/6.)

    #写一个专用的旋转矩阵的函数。rotate2d一类的。

    the_only_modified_part = input.index_select(dim=1,index=index_tensor)#new tensor.
    assert the_only_modified_part.eq(torch.tensor([[0,1],[3,4],[6,7]])).all()
    the_only_modified_part = the_only_modified_part@angle_to_rotation_matrix_2d(angle)
    assert _tensor_equal(the_only_modified_part[0], torch.tensor([0.,1])@angle_to_rotation_matrix_2d(angle))
    input[:,index_tensor[0]] = the_only_modified_part[:,0]
    input[:,index_tensor[1]] = the_only_modified_part[:,1]
    
    "row"
    dim = 3
    input = torch.linspace(0,dim*dim-1,dim*dim).reshape([dim,dim])
    #index_tensor ,_,_ = _get_2_diff_rand_int(dim,'cpu')
    index_tensor = torch.tensor([0,1])
    #angle = torch.rand(size=[])*torch.pi*2.
    angle = torch.tensor(torch.pi/6.)

    #写一个专用的旋转矩阵的函数。rotate2d一类的。

    the_only_modified_part = input.index_select(dim=0,index=index_tensor)#new tensor.
    assert the_only_modified_part.eq(torch.tensor([[0,1,2],[3,4,5]])).all()
    the_only_modified_part = angle_to_rotation_matrix_2d(angle)@the_only_modified_part
    assert _tensor_equal(the_only_modified_part[:,0], (angle_to_rotation_matrix_2d(angle)@torch.tensor([[0.],[3]])).reshape([-1]))
    input[index_tensor[0]] = the_only_modified_part[0]
    input[index_tensor[1]] = the_only_modified_part[1]
    pass
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
            index_tensor = _get_2_diff_rand_int(dim, device=the_device)
            angle = torch.rand(size=[], dtype = the_dtype,device=the_device)*torch.pi*2.
            the_only_modified_part = input.index_select(dim=1,index=index_tensor)#new tensor.
            the_only_modified_part = the_only_modified_part@angle_to_rotation_matrix_2d(angle)
            input[:,index_tensor[0]] = the_only_modified_part[:,0]
            input[:,index_tensor[1]] = the_only_modified_part[:,1]
            pass
        
        for _ in range(times_by_column):# each COLUMN vec into a new rotated COLUMN vec
            index_tensor  = _get_2_diff_rand_int(dim, device=the_device)
            angle = torch.rand(size=[], dtype = the_dtype,device=the_device)*torch.pi*2.
            the_only_modified_part = input.index_select(dim=0,index=index_tensor)#new tensor.
            the_only_modified_part = angle_to_rotation_matrix_2d(angle)@the_only_modified_part
            input[index_tensor[0]] = the_only_modified_part[0]
            input[index_tensor[1]] = the_only_modified_part[1]
            pass
        return input
    pass#/function
if "test" and __DEBUG_ME__() and True:
    def ____test____random_rotate():
        for _ in range(1):
            dim = random.randint(2,300)
            #-------------#-------------#-------------
            _init_eye = torch.eye(n = dim)
            mat = random_rotate(_init_eye)
            #-------------#-------------#-------------
            vec = random_standard_vector(dim)
            vec = vec@mat
            assert _tensor_equal(get_vector_length(vec), [1.])
            pass
        return
    ____test____random_rotate()

def random_rotation_matrix__by_once(dim:int, dtype = torch.float32, device = 'cpu')->torch.Tensor:
    '''Basically it's a eye_tensor, but with some rotation. This rotation only affects 2 dimentions.'''
    index_tensor = _get_2_diff_rand_int(dim, device=device)
    _angle = torch.rand(size=[], dtype = dtype,device=device)*torch.pi*2.
    _cos = _angle.cos()
    _sin = _angle.sin()
    result = torch.eye(n=dim, dtype=dtype, device=device)
    result[index_tensor[0],index_tensor[0]] = _cos
    result[index_tensor[0],index_tensor[1]] = _sin
    result[index_tensor[1],index_tensor[0]] = -_sin
    result[index_tensor[1],index_tensor[1]] = _cos
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____random_rotation_matrix__by_once():
        for _ in range(155):
            dim = random.randint(2,300)
            #-------------#-------------#-------------
            mat = random_rotation_matrix__by_once(dim)
            #-------------#-------------#-------------
            iota_of_dim = iota(dim)
            _flag_is_1 = mat.eq(1.)
            _flag_is_0 = mat.eq(0.)
            _flag_non1_and_non0 = _flag_is_1.logical_not().logical_and(_flag_is_0.logical_not())
            if _flag_non1_and_non0.sum()>0:#non special case.
                #number of non1 non0 elements.
                assert _tensor_equal(_flag_is_1.sum(), [dim-2])#2 ones are modified.
                assert _tensor_equal(_flag_is_0.sum(), [dim*dim-dim-2])#2 ones are modified.
                other_elements = mat[_flag_non1_and_non0]
                assert other_elements[0] == other_elements[3]#cos
                assert other_elements[1] == -other_elements[2]#sin
                #2 cosines equal, 2 sines equal.
                other_elements_abs = other_elements.abs()
                other_elements_abs = other_elements_abs.sort().values
                assert other_elements_abs[0] == other_elements_abs[1]
                assert other_elements_abs[0]<1.
                assert other_elements_abs[2] == other_elements_abs[3]
                assert other_elements_abs[2]<1.
                #number of sign, 3pos 1neg or 3neg 1pos
                other_elements_gt_0 = other_elements.gt(0.)
                other_elements_gt_0__sum = other_elements_gt_0.sum()
                assert other_elements_gt_0__sum == 1 or other_elements_gt_0__sum == 3
                #no 1 outside main diagonal
                _flag_is_1[iota_of_dim,iota_of_dim] = False
                assert _flag_is_1.sum() == 0
                pass
            else:#special case
                #number of 1 0 elements.
                assert _flag_is_1.sum() == dim
                assert _flag_is_0.sum() == dim*dim-dim
                #no 1 outside main diagonal
                _flag_is_1[iota_of_dim,iota_of_dim] = False
                _flag_is_1__sum = _flag_is_1.sum()
                assert _flag_is_1__sum == 0 or _flag_is_1__sum == 2
                pass
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




"   random permutate"
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
            assert _tensor_equal(_ori_sum, _after_sum, epsilon=_ori_sum.abs()*0.0003)
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
        for _ in range(116):
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
        for _ in range(116):
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




