from typing import Any, Optional, Literal
import random
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import is_square_matrix, _tensor_equal, have_same_elements, \
    get_vector_length, vector_length_norm, iota,\
        str_the_list \

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"




"rand sign"

def rand_sign(size:list[int], dtype = torch.float32, device = 'cpu')->torch.Tensor:
    '''
    >>> for bool: true/false
    >>> for signed int: -1/1
    >>> for unsigned int: 0/1
    >>> for fp : -1./1.
    '''
    if dtype == torch.bool:
        result = torch.randint(low=0, high=2, dtype=torch.int8, size=size, device=device).eq(0)
        return result
    elif dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:#uint, 0 or 1
        result = torch.randint(low=0, high=2, dtype=dtype, size=size, device=device)
        return result
    elif dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:#int, -1 or 1
        result = torch.randint(low=0, high=2, dtype=torch.int8, size=size, device=device)*2-1
        return result.to(dtype)
    else:#float, -1. or 1.
        result = torch.randint(low=0, high=2, dtype=torch.int8, size=size, device=device)*2-1
        return result.to(dtype)
    pass#/function
if "test" and __DEBUG_ME__() and False:
    def ____test____rand_sign():
        size = torch.Size([1,2,3,4,5])
        result = rand_sign(size=size)
        assert result.shape == size
        assert result.dtype == torch.float32
        assert result.eq(1.).sum() + result.eq(-1.).sum() == 120
        
        result = rand_sign(size=[], dtype=torch.bool)
        assert result.dtype == torch.bool
        assert result.shape == torch.Size([])
        
        result = rand_sign(size=size, dtype=torch.uint16)
        assert result.dtype == torch.uint16
        assert result.eq(1).sum() + result.eq(0).sum() == 120
        
        result = rand_sign(size=size, dtype=torch.int64)
        assert result.dtype == torch.int64
        assert result.eq(1).sum() + result.eq(-1).sum() == 120
        
        return
    ____test____rand_sign()
    pass

"randn safe"

def randn_safe(size:list[int], dtype = torch.float32, device = 'cpu')->torch.Tensor:
    '''step1 provides 68% as a randn but only the [-1,1] elements.
    step2 minus the abs of elements with abs in (1,2] with 1 and push them back into [-1,1], 13%.
    step3 re random the last 18% elements with a uniform distribution in [-1,1].'''
    result = torch.randn(size=size, dtype=dtype, device=device)
    flag__abs_gt_1 = result.abs().gt(1.)
    #aaa = flag__abs_gt_1.sum()
    result[flag__abs_gt_1] = result[flag__abs_gt_1]-1.
    flag__abs_gt_1 = result.abs().gt(1.)
    #bbb = flag__abs_gt_1.sum()
    _this_shape = result[flag__abs_gt_1].shape
    result[flag__abs_gt_1] = torch.rand(size=_this_shape, dtype=dtype)*2.-1
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____randn_safe():
        size = torch.Size([2,3,4])
        result = randn_safe(size=size)
        assert result.shape == size
        assert result.le(1.).logical_or(result.ge(-1.)).all()
        
        result = randn_safe(size=[100000])
        assert result.le(1.).logical_or(result.ge(-1.)).all()
        
        return
    ____test____randn_safe()
    pass



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
if "basic test" and __DEBUG_ME__() and False:
    def ____test_____random_standard_vector__basic_test():
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
    ____test_____random_standard_vector__basic_test()
    pass
if "log10 test    copy pasted from Util.py" and __DEBUG_ME__() and False:
    #result: basically the log10 of a standard vec is -0.5*log10(dim)-0.21
    def ____test____log10_avg_safe____standard_vec():
        import math, random
        from pytorch_yagaodirac_v2.Util import log10_avg_safe
            
        if "measure the log10" and True:
            # output:
            # the_min_gt_this_list =[-1.470, -1.821, -2.285]
            # the_max_lt_this_list =[-1.170, -1.740, -2.266]
            # the_mean_eq_this_list=[-1.275, -1.775, -2.275]
            # epsilon_list       =[ 0.194,  0.046,  0.010]
            # -0.5*log10(dim)-0.275, basically from randn.
            print("the log10")
            device = 'cuda'
            #--------------------#--------------------#--------------------
            dim_list = [100,1000,10000]
            test_time_list = [3000,1000,300]
            #--------------------#--------------------#--------------------
            the_min_gt_this_list =  []#don't modify here.
            the_max_lt_this_list =  []
            the_mean_eq_this_list = []
            epsilon_list =          []
            for inner_iter_count in range(dim_list.__len__()):
                dim = dim_list[inner_iter_count]
                test_time = test_time_list[inner_iter_count]
                print(test_time)
            
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #--------------------#--------------------#--------------------
                    vec = random_standard_vector(dim=dim)
                    assert _tensor_equal(get_vector_length(vec), [1.])
                    _this_result = log10_avg_safe(vec)
                    #--------------------#--------------------#--------------------
                    _raw_result[test_count] = _this_result
                    pass
                the_min = _raw_result.min()
                the_max = _raw_result.max()
                the_mean = _raw_result.mean()
                the_min_gt_this_list.append(the_min.item())
                the_max_lt_this_list.append(the_max.item())
                the_mean_eq_this_list.append(the_mean.item())
                _delta_1 = the_mean - the_min 
                _delta_2 = the_max  - the_mean
                epsilon = max(_delta_1, _delta_2)
                epsilon_list.append(epsilon.item())    
                print(f"dim:{dim}  ///  {the_min:.3f}   {the_max:.3f}   {the_mean:.3f}   ")
                pass# for macro_iter_count
            print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
            print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
            print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
            print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
            pass#/test
        
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
if "test" and __DEBUG_ME__() and False:
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
    # result = torch.tensor( [[ _cos, _sin],
    #                         [-_sin, _cos]], device=angle.device)
    
    temp_row0 = torch.stack([_cos, _sin])
    temp_row1 = torch.stack([-_sin, _cos])
    result = torch.stack([temp_row0, temp_row1])
    return result
if "basic behavior and device adaption." and __DEBUG_ME__() and False:
    def ____test_____angle_to_rotation_matrix_2d():
        import math
        mat = angle_to_rotation_matrix_2d(0.1)
        assert _tensor_equal(mat,  [[ math.cos(0.1),math.sin(0.1)],
                                    [-math.sin(0.1),math.cos(0.1)]])
        mat = angle_to_rotation_matrix_2d(torch.tensor(torch.pi/6., device='cuda'))
        assert _tensor_equal(mat,  torch.tensor([[ math.sqrt(3.)/2,            0.5],
                                                        [   -0.5,     math.sqrt(3.)/2]], device='cuda'))
        assert mat.device.type == 'cuda'
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
def random_rotate_this_matrix(input:torch.Tensor, times:int|None = None)->torch.Tensor:
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
if "test" and __DEBUG_ME__() and False:
    def ____test____random_rotate():
        for _ in range(1):
            dim = random.randint(2,300)
            #-------------#-------------#-------------
            _init_eye = torch.eye(n = dim)
            mat = random_rotate_this_matrix(_init_eye)
            #-------------#-------------#-------------
            vec = random_standard_vector(dim)
            vec = vec@mat
            assert _tensor_equal(get_vector_length(vec), [1.])
            pass
        return
    ____test____random_rotate()

def random_rotate_this_vector(input:torch.Tensor, times:int|None = None)->torch.Tensor:
    assert input.shape.__len__() == 1, "Batch is not implemented. Maybe later."
    dim = input.shape[0]
    the_device = input.device
    the_dtype = input.dtype
    
    if times is None:
        times = dim*3
        pass
    
    with torch.no_grad():
        for _ in range(times):# each ROW vec into a new rotated ROW vec
            index_tensor = _get_2_diff_rand_int(dim, device=the_device)
            angle = torch.rand(size=[], dtype = the_dtype,device=the_device)*torch.pi*2.
            the_only_modified_part = input.index_select(dim=0,index=index_tensor)#new tensor.
            the_only_modified_part = the_only_modified_part@angle_to_rotation_matrix_2d(angle)
            input[index_tensor[0]] = the_only_modified_part[0]
            input[index_tensor[1]] = the_only_modified_part[1]
            pass
        
        return input
    pass#/function
if "test" and __DEBUG_ME__() and False:
    def ____test____random_rotate_this_vector():
        device = 'cpu'#cpu is faster here.
        dim_list =           [2,    3,   5,  10,100,1000]
        test_time_list = [10000,5000,1000,1000,300,30]
        for inner_param_iter in range(dim_list.__len__()):
            dim = dim_list[inner_param_iter]
            test_time = test_time_list[inner_param_iter]
            print(f"{dim}", end=" ")
            for test_count in range(test_time):
                #-------------#-------------#-------------
                _init_vec = vector_length_norm(torch.randn(size=[1,dim],device=device)).reshape([-1])
                vec = random_rotate_this_vector(_init_vec)
                #-------------#-------------#-------------
                assert _tensor_equal(get_vector_length(vec), torch.tensor([1.],device=device))
            pass
        return
    ____test____random_rotate_this_vector()
    pass

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
if "test" and __DEBUG_ME__() and False:
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
    result = random_rotate_this_matrix(init_eye,times=times)
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____random_rotation_matrix():
        "bc the inner function is validated, this func doesn't need to get validated again."
        return 
    ____test____random_rotation_matrix()
    pass


def random_standard_vector__pre_rotated(dim:int, rotate_count:int|None = None, 
                                        _protect_length_every = 30, 
                                        dtype = torch.float32, device='cpu',)->torch.Tensor:
    if rotate_count is None:
        rotate_count = int((dim+10)*1.2)
        pass
    assert rotate_count>=0
    vec = random_standard_vector(dim=dim, dtype = dtype, device = device)
    
    # new vvvvvvvvvv
    iter_count = 0
    for _ in range(rotate_count):
        vec = random_rotate_this_vector(vec)
        
        #<  length protection>
        iter_count +=1
        if iter_count>=_protect_length_every:
            iter_count = 0
            vec = vector_length_norm(vec.reshape([1,-1])).reshape([-1])
            pass
        #</ length protection>
        pass
    
    #<  before return/>
    vec = vector_length_norm(vec.reshape([1,-1])).reshape([-1])
    # new ^^^^^^^^^^^
    return vec
if "basic test" and True:
    def ____test____random_standard_vector__pre_rotated(): 
        
        if "length accuracy test" and True: 1w
            # output:
            print("length accuracy test")
            device = 'cpu'
            #--------------------#--------------------#--------------------
            dim_list =          [2,  3,  5, 10,100,1000]
            test_time_list = [1000,500,100,100,30, 3]
            for outter_iter_count in range(dim_list.__len__()):
                dim = dim_list[outter_iter_count]
                test_time = test_time_list[outter_iter_count]
                print(test_time)
            #--------------------#--------------------#--------------------
                the_min_gt_this_list =  []#don't modify here.
                the_max_lt_this_list =  []#don't modify here.
                the_mean_eq_this_list = []#don't modify here.
                epsilon_list =          []#don't modify here.
                #--------------------#--------------------#--------------------
                rc_list = [0,1,2,3,5,10,100]
                for inner_iter_count in range(rc_list.__len__()):
                    rc = rc_list[inner_iter_count]
                #--------------------#--------------------#--------------------
                
                    _raw_result = torch.empty(size=[test_time])
                    for test_count in range(test_time):
                        #--------------------#--------------------#--------------------
                        vec = random_standard_vector__pre_rotated(dim=dim,rotate_count=rc)
                        _this_result = get_vector_length(vec)
                        #--------------------#--------------------#--------------------
                        _raw_result[test_count] = _this_result
                        pass
                    the_min = _raw_result.min()
                    the_max = _raw_result.max()
                    the_mean = _raw_result.mean()
                    the_min_gt_this_list.append(the_min.item())
                    the_max_lt_this_list.append(the_max.item())
                    the_mean_eq_this_list.append(the_mean.item())
                    _delta_1 = the_mean - the_min 
                    _delta_2 = the_max  - the_mean
                    epsilon = max(_delta_1, _delta_2)
                    epsilon_list.append(epsilon.item())    
                    print(f"dim:{dim}  ///  {the_min:.3f}   {the_max:.3f}   {the_mean:.3f}   ")
                    pass# for macro_iter_count
                print(f"the_min_gt_this_list ={str_the_list(the_min_gt_this_list, 3)}")    
                print(f"the_max_lt_this_list ={str_the_list(the_max_lt_this_list, 3)}")    
                print(f"the_mean_eq_this_list={str_the_list(the_mean_eq_this_list,3)}")    
                print(f"epsilon_list       ={    str_the_list(epsilon_list,         3)}")    
                pass#/test
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        dim_list =          [2,  3,  5, 10,100,1000]
        test_time_list = [1000,500,100,100,30, 3]
        for outter_param_iter in range(dim_list.__len__()):
            dim = dim_list[outter_param_iter]
            test_time = test_time_list[outter_param_iter]
            print(f"{dim}", end=" ")
            
            rc_list = [0,1,2,3,5,10,100]
            for inner_param_iter in range(rc_list.__len__()):
                rc = rc_list[inner_param_iter]
                for test_count in range(test_time):
                    #-------------#-------------#-------------
                    vec = random_standard_vector__pre_rotated(dim=dim,rotate_count=rc)
                    #-------------#-------------#-------------
                    assert _tensor_equal(get_vector_length(vec), torch.tensor([1.],device=device))
                    pass#for test_count
                pass#for inner_param_iter
            pass#for outter_param_iter
        
        
        device = 'cpu'#cpu is faster here.
        dim_list =          [2,  3,  5, 10,100,1000]
        test_time_list = [1000,500,100,100,30, 3]
        for outter_param_iter in range(dim_list.__len__()):
            dim = dim_list[outter_param_iter]
            test_time = test_time_list[outter_param_iter]
            print(f"{dim}", end=" ")
            
            for test_count in range(test_time):
                #-------------#-------------#-------------
                vec = random_standard_vector__pre_rotated(dim=dim,rotate_count=None)
                #-------------#-------------#-------------
                assert _tensor_equal(get_vector_length(vec), torch.tensor([1.],device=device))
                pass#for test_count
            pass#for outter_param_iter
        
        assert False,"每多少次保护一下长度。"
        
        return 
    
    ____test____random_standard_vector__pre_rotated()
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




