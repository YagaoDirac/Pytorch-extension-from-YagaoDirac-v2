# so, this file is a test to check the idea if it's possible to correct a random square matrix
# back to a rotation matrix.
# This rotation matrix may help in forward prapagation, and against the information vanish
# problem.

# idea:
# for a rotation matrix M, the element is Mij.
# Then the matrix of (Mij * Mij) should be a doubly stockestic matrix. It should be easy 
# to correct it back to a doubly stockestic matrix after the matrix receives some tweak.
# This method is much easier than what I tried previously.

# 旋转矩阵是不是还可以用，非对角线上的元素是反对称的，然后加上上面说的，为1，好像也可以操作。
# 突然发现，标准正交矩阵的乘法也是保长度的（和它乘的向量长度不会变），所以其实不用旋转矩阵那么强的限制，
# 不存在一个右上是-sin，左下必须对应sin的限制。

# 开平方以后的符号怎么确定？

# 然后又想到。正交矩阵的定义里面只有一个方向的正交，而没有另外一个方向。也就是说，是不是保护一个方向的正交性就够了？？

# 但是正交本身的保护就很麻烦，尤其矩阵很大的时候。如果保护得不是很好，对效果的影响有多大？


import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal, avg_log10_safe, \
    get_mask_of_top_element__rough, iota, is_square_matrix, vector_length_norm
from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2_mean_abs_to_1

import torch







def measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(matrix:torch.Tensor,
                                    test_time = 10, cap = 2.)->tuple[torch.Tensor,torch.Tensor]:
    '''    return all_score.mean(),all_score
    The result is always >=0. The smaller the better.'''
    assert is_square_matrix(matrix)
    assert test_time>=1
    assert cap>1.#at least >0
    
    the_device = matrix.device
    all_score = torch.empty([test_time], device = the_device)
    for epoch in range(test_time):
        vec = torch.randn(size=[matrix.shape[0]], device = the_device)
        while True:
            ori_len_sqr = vec.dot(vec)
            #too small or too large, reroll.
            if ori_len_sqr<0.001 or ori_len_sqr>10000.:
                vec = torch.randn(size=[matrix.shape[0]])
                continue
            break
            pass#/while
        __mul_me = ori_len_sqr.pow(-0.5)
        vec.mul_(__mul_me)
        assert _tensor_equal(vec.dot(vec), torch.tensor([1.], device=vec.device))
        
        #vec = vec.reshape(shape=[1,-1])
        after_matmul = vec@matrix
        new_len_sqr = after_matmul.dot(after_matmul)
        
        score_of_this_epoch = new_len_sqr.log10().abs()/2.
        all_score[epoch] = score_of_this_epoch
        pass#/ for
    all_score = all_score.clamp_max_(cap)
    all_score = all_score.clamp_min_(-cap)
    all_score = all_score.cpu()
    return all_score.mean(),all_score
if "____test____measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10" and False:
    def ____test____measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10():
        #eye should be perfect.
        if False:
            for size in range(3, 15):
                for _ in range(5):
                    _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(torch.eye(n=size),test_time=100)
                    assert _tensor_equal(_result_tuple[1], torch.zeros(size=[100]), epsilon=1e-6)
                    pass
                pass
            for size in range(3, 15):
                for _ in range(5):
                    _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(torch.eye(n=size)*10. ,test_time=100)
                    assert _tensor_equal(_result_tuple[1], torch.ones(size=[100]))
                    pass
                pass
            for size in range(3, 15):
                for _ in range(5):
                    _to_the_power = torch.rand(size=[1])*3.
                    assert _to_the_power>=0.
                    _factor = torch.pow(10, _to_the_power)
                    _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(torch.eye(n=size)*_factor, 
                                                                                                test_time=100, cap=5.)
                    assert _tensor_equal(_result_tuple[1], torch.ones(size=[100])*_to_the_power)
                    pass
                pass
            for size in range(3, 15):
                for _ in range(5):
                    _to_the_power = torch.rand(size=[1])*-3.
                    assert _to_the_power<=0.
                    _factor = torch.pow(10, _to_the_power)
                    _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(torch.eye(n=size)*_factor, 
                                                                                                test_time=100, cap=5.)
                    assert _tensor_equal(_result_tuple[1], torch.ones(size=[100])*-_to_the_power)
                    pass
                pass
        
        
        #rotation should also be perfect. This test is done in the rand_basic_ratation_matrix's test
        
        
        #some affine matrix.
        for _ in range(55):
            mat = torch.eye(n=100)
            mat[0,1] = 1.
            _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat,test_time=100)
            result_for_1 = _result_tuple[0]
            mat = torch.eye(n=100)
            mat[0,1] = 10.
            _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat,test_time=100)
            result_for_2 = _result_tuple[0]
            assert result_for_2.gt(result_for_1) #theoretically unstable. But mostly cases it holds.
            pass
        return 
    ____test____measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10()
    pass






# What's the test:
# When correct the length in one direction(say row), when the orthogonality is guarunteed(or not, idk),
# the correction also affects the length measured by the other direction.
# This test is to measure how much it affects.
# Result:
# when the correction_factor is in [0.25, 0.3], the correction of one direction 
# doesn't affect the other direction much.
# but basicall, when the factor is 0.28 to 0.3 or 0.36 to 0.38, it's slightly a bit better.
# To what I know, I decide to use 0.28 for now.


def length_protection_test(DIM:int, correction_factor = 0.25, correction_factor_for_pre_protect = 0.2)->tuple[torch.Tensor,torch.Tensor]:
    '''
    return better__value, better_amount
    '''
    with torch.no_grad():
        #<hyper param>
        # moved into in param.
        #</ hyper param>
        
        #<init>
        mat:torch.Tensor = torch.randn(size=[DIM,DIM],requires_grad=True)
        #pre correct
        for epoch in range(2):
            if "add noise" and True:
                mat+=torch.randn_like(mat) * torch.tensor(DIM, dtype=torch.float32).pow(-0.5)*0.01
                pass
            
            
            # row?
            _temp_len_sqr = mat.mul(mat).mean(dim=1)#mul and then sum, it's a dot.
            if "check it a lil bit":
                _temp_len_sqr__ref = mat[0].dot(mat[0]).div(DIM)
                assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                pass
            #0.25 to 0.5 power
            mul_me = _temp_len_sqr.pow(-correction_factor_for_pre_protect)
            mat.mul_(mul_me.reshape([-1,1]).expand([-1,DIM]))
            if "check it after modifying":
                _temp_len_sqr__after_modifying = mat.mul(mat).mean(dim=1)#mul and then sum, it's a dot.
                abs_log10__of_ori = _temp_len_sqr.log10().abs()
                abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                better = abs_log10__of_ori.ge(abs_log10__of_after)
                assert better.to(torch.float32).mean()>0.9
                pass
            
            
            # column?
            _temp_len_sqr = mat.mul(mat).mean(dim=0)#mul and then sum, it's a dot.
            if "check it a lil bit":
                _temp_len_sqr__ref = mat[:,0].dot(mat[:,0]).div(DIM)
                assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                pass
            #0.25 to 0.5 power
            mul_me = _temp_len_sqr.pow(-correction_factor_for_pre_protect)
            mat.mul_(mul_me.reshape([1,-1]).expand([DIM,-1]))
            if "check it after modifying":
                _temp_len_sqr__after_modifying = mat.mul(mat).mean(dim=0)#mul and then sum, it's a dot.
                abs_log10__of_ori = _temp_len_sqr.log10().abs()
                abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                better = abs_log10__of_ori.ge(abs_log10__of_after)
                assert better.to(torch.float32).mean()>0.9
                pass
            pass#for "pre protect"
        #</ init>
        
        
        
        #< test>
        _temp_len_sqr__for_the_other_direction = mat.mul(mat).mean(dim=0)
        
        # row?
        _temp_len_sqr = mat.mul(mat).mean(dim=1)#mul and then sum, it's a dot.
        if "check it a lil bit":
            _temp_len_sqr__ref = mat[0].dot(mat[0]).div(DIM)
            assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
            pass
        #0.25 to 0.5 power
        mul_me = _temp_len_sqr.pow(-correction_factor)
        mat.mul_(mul_me.reshape([-1,1]).expand([-1,DIM]))
        if "check it after modifying":
            _temp_len_sqr__after_modifying = mat.mul(mat).mean(dim=1)#mul and then sum, it's a dot.
            abs_log10__of_ori = _temp_len_sqr.log10().abs()
            abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
            better = abs_log10__of_ori.ge(abs_log10__of_after)
            assert better.to(torch.float32).mean()>0.9
            pass
        
        _temp_len_sqr__for_the_other_direction_after = mat.mul(mat).mean(dim=0)
        if DIM>=10:
            _flag_1:torch.Tensor = get_mask_of_top_element__rough(_temp_len_sqr__for_the_other_direction.log10().abs().      reshape([1,-1]))[0].reshape([-1])
            _flag_2:torch.Tensor = get_mask_of_top_element__rough(_temp_len_sqr__for_the_other_direction_after.log10().abs().reshape([1,-1]))[0].reshape([-1])
            _flag_both = _flag_1.logical_and(_flag_2)
            abs_log10__temp_len_sqr__for_the_other_direction =       _temp_len_sqr__for_the_other_direction      [_flag_both].log10().abs()
            abs_log10__temp_len_sqr__for_the_other_direction_after = _temp_len_sqr__for_the_other_direction_after[_flag_both].log10().abs()
            pass
        else:
            abs_log10__temp_len_sqr__for_the_other_direction =       _temp_len_sqr__for_the_other_direction.log10().abs()
            abs_log10__temp_len_sqr__for_the_other_direction_after = _temp_len_sqr__for_the_other_direction_after.log10().abs()
            pass
        _better__value__before_mean =   abs_log10__temp_len_sqr__for_the_other_direction /     \
                                        abs_log10__temp_len_sqr__for_the_other_direction_after
        better__value = _better__value__before_mean.mean()
            
        better = abs_log10__temp_len_sqr__for_the_other_direction.ge(abs_log10__temp_len_sqr__for_the_other_direction_after)
        better_amount = better.to(torch.float32).mean()
        #</ test>
        pass#no grad
    return better__value, better_amount
if "____protection_hyper_param_scan" and False:
    def ____protection_hyper_param_scan():
        #file name
        _time = datetime.datetime.now()
        _time_str = _time.isoformat(sep=" ")
        _time_str = _time_str[0:19]
        _time_str = _time_str.replace(":","-")
        _file_name = f"{Path(__file__).parent/"test result"/"protection_hyper_param_scan"} {_time_str}.txt"
        with open(_file_name, mode = "a", encoding="utf-8") as file:#maybe the mode should be "a"
            file.write("method: protection_hyper_param_scan\n\n")
            file.write(f"{_time_str}\n\n")
            pass#open
        
        #<param>
        for DIM in [10,100,1000]:#,10000]:#dont try more.
            assert DIM <= 20000
        #</ param>
            if DIM<1000:
                _test_time = 50
                pass
            elif DIM<10000:
                _test_time = 10
                pass
            else:
                _test_time = 5
                pass
            with open(_file_name, mode = "a", encoding="utf-8") as file:
                DIM_DIM_DIM_str = f"{DIM}  "*18
                file.write(f"DIM   {DIM_DIM_DIM_str}\n")
                file.write(f"test time {_test_time}\n\n")
                pass
            
            #<param>
            #for correction_factor in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
            for _correction_factor in range(20,41):#[0.25,0.4]:
                correction_factor = _correction_factor/100.
            #</ param>
                #<init>
                better__value_list = torch.empty(size=[_test_time], dtype=torch.float32)
                better_amount_list = torch.empty(size=[_test_time], dtype=torch.float32)
                #</ init>
                
                #<test
                for _iter_count in range(_test_time):
                    better__value, better_amount = length_protection_test(DIM=DIM, correction_factor=correction_factor)
                    better__value_list[_iter_count] = better__value
                    better_amount_list[_iter_count] = better_amount
                    pass
                    
                #log out.
                with open(_file_name, mode = "a", encoding="utf-8") as file:
                    file.write(f"correction_factor: {correction_factor}\n")
                    file.write(f"test time: {_test_time}\n")
                    better__value_to_save = better__value.mean().item()
                    if better__value_to_save>1:
                        file.write(f"better__value: {better__value_to_save:.4f}\n")
                        pass
                    else:#bad
                        file.write(f"better__value: {better__value_to_save:.4f}    BAD   BAD   BAD\n")
                        pass
                    
                    better_amount_to_save = better_amount.mean().item()
                    if better_amount_to_save>0.5:
                        file.write(f"       better_amount: {better_amount_to_save:.3f}\n")
                        pass
                    else:
                        file.write(f"       better_amount: {better_amount_to_save:.3f}    BAD   BAD   BAD\n")
                        pass
                    file.write("\n\n")
                    pass# open
                pass# for correction_factor in range
            pass#for DIM
        return
    ____protection_hyper_param_scan()
    pass







def rand_basic_ratation_matrix(dim:int)->torch.Tensor:
    mat = torch.eye(n=dim)
    rand_deg = (torch.rand(size=[1])-0.5)*321.
    cos_of_rand = rand_deg.cos()
    sin_of_rand = rand_deg.sin()
    dim1 = torch.randint(0,dim,size=[1])
    dim2 = torch.randint(0,dim-1,size=[1])
    if dim2.ge(dim1):
        dim2+=1
        pass
    assert dim1.ne(dim2)
    mat[dim1,dim1] =  cos_of_rand
    mat[dim1,dim2] =  sin_of_rand
    mat[dim2,dim1] = -sin_of_rand
    mat[dim2,dim2] =  cos_of_rand
    return mat
if False:
    def ____test____rand_basic_ratation_matrix():
        for dim in range(3,14):
            for _ in range(11):
                mat = rand_basic_ratation_matrix(dim)
                _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat, test_time=100)
                assert _tensor_equal(_result_tuple[1], torch.zeros(size=[100]))
                pass#for _
            pass# for dim
        
        for dim in range(3,14):
            for _test_time in range(11):
                mat = rand_basic_ratation_matrix(dim)
                for _rotating_time in range(dim*5):
                    new_mat = rand_basic_ratation_matrix(dim)
                    mat = mat.matmul(new_mat)
                    pass
                    
                _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat, test_time=100)
                assert _tensor_equal(_result_tuple[1], torch.zeros(size=[100]))
                pass#for _
            pass# for dim
        return 
    ____test____rand_basic_ratation_matrix()
    pass









# if "some old code " and False:
#     # row?
#     _temp_len_sqr = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
#     if "check it a lil bit":
#         _temp_len_sqr__ref = matrix[0].dot(matrix[0]).div(DIM)
#         assert _temp_len_sqr[0].eq(_temp_len_sqr__ref)
#         pass
#     #0.25 to 0.5 power
#     mul_me = _temp_len_sqr.pow(-0.25)
#     matrix.mul_(mul_me.reshape([-1,1]).expand([-1,DIM]))
#     if "check it after modifying":
#         _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
#         abs_log10__of_ori = _temp_len_sqr.log10().abs()
#         abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
#         better = abs_log10__of_ori.ge(abs_log10__of_after)
#         assert better.to(torch.float32).mean()>0.9
#         pass
    
#     # column?
#     _temp_len_sqr = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
#     if "check it a lil bit":
#         _temp_len_sqr__ref = matrix[:,0].dot(matrix[:,0]).div(DIM)
#         assert _temp_len_sqr[0].eq(_temp_len_sqr__ref)
#         pass
#     #0.25 to 0.5 power
#     mul_me = _temp_len_sqr.pow(-0.25)
#     matrix.mul_(mul_me.reshape([1,-1]).expand([DIM,-1]))
#     if "check it after modifying":
#         _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
#         abs_log10__of_ori = _temp_len_sqr.log10().abs()
#         abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
#         better = abs_log10__of_ori.ge(abs_log10__of_after)
#         assert better.to(torch.float32).mean()>0.9
#         pass





#方阵的可以直接计算出来的长度指标，和用一个dummy vec乘上去观察长度变化，得出的指标，之间的关系是什么。是否有可能推测正交性。
#1w 继续。
# 一些特殊的mat可能不行。
# 主对角线可能无法被优化。可能需要额外的保护。

def ____old_code____correct_the_matrix___version_1(matrix:torch.Tensor, lr = 0.3,correction_factor = 0.15, 
                                        iter_count = 1, dont_correct_length_with_error_prapagation = False, 
                                                __debug__need_log = False)->tuple[torch.Tensor, list|None]:
    '''this function removes the grad stored on the param:matrix'''
    assert is_square_matrix(matrix)
    dim = matrix.shape[0]
    does_matrix_require_grad = matrix.requires_grad
    matrix.requires_grad_()
    matrix.grad = None
    
    the_device = matrix.device
    correction_factor_tensor = torch.tensor(-correction_factor, device=the_device)#0.25 to 0.4. safe range is 0 to 0.5
    #matrix:torch.Tensor = torch.randn(size=[DIM,DIM],requires_grad=True, device = device)
    # DIM = 2
    # mat:torch.Tensor = torch.tensor([[16.,16],[1,1]], requires_grad=True)
    gramo = GradientModification_v2_mean_abs_to_1().to(the_device)
    train_them = [matrix]

    optim = torch.optim.SGD(params=train_them, lr=lr)
    loss_func = torch.nn.MSELoss()
    #<  the main protection >
    if __debug__need_log:
        _log:list|None = []
        pass
    else:
        _log = None
        pass
    
    for epoch in range(iter_count):
        with torch.no_grad():
            #<  add some noise >
            if "add noise" and False:
                matrix+=torch.randn_like(matrix)*lr*0.1
                if __debug__need_log:
                    _log.append("noise added !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    pass
                pass
            #</ add some noise >
            
            #<  correct the length>
            if "correct the length" and True:
                # row?
                _temp_len_sqr = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
                if "check it a lil bit" and True:
                    _temp_len_sqr__ref = matrix[0].dot(matrix[0]).div(dim)
                    assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                    pass
                #0.25 to 0.5 power
                mul_me = _temp_len_sqr.pow(correction_factor_tensor)
                matrix.mul_(mul_me.reshape([-1,1]).expand([-1,dim]))
                if "check it after modifying":
                    _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
                    abs_log10__of_ori = _temp_len_sqr.log10().abs()
                    abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                    better = abs_log10__of_ori.ge(abs_log10__of_after)
                    assert better.to(torch.float32).mean()>0.9
                    pass
                
                if _log is not None:
                    _log.append(("Length corrected by row", matrix.detach().clone()))
                    pass
                
                # column?
                _temp_len_sqr = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
                if "check it a lil bit" and True:
                    _temp_len_sqr__ref = matrix[:,0].dot(matrix[:,0]).div(dim)
                    assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                    pass
                #0.25 to 0.5 power
                mul_me = _temp_len_sqr.pow(correction_factor_tensor)
                matrix.mul_(mul_me.reshape([1,-1]).expand([dim,-1]))
                if "check it after modifying":
                    _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
                    abs_log10__of_ori = _temp_len_sqr.log10().abs()
                    abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                    better = abs_log10__of_ori.ge(abs_log10__of_after)
                    assert better.to(torch.float32).mean()>0.9
                    pass
                
                if _log is not None:
                    _log.append(("Length corrected by column", matrix.detach().clone()))
                    pass
                
                pass#for "pre protect"
            #</ correct the length>
            
            pass
        assert matrix.requires_grad
        
        #<  correct the direction>
        mat_after_gramo:torch.Tensor = gramo(matrix.reshape([1,-1])).reshape([dim,dim])
        assert is_square_matrix(mat_after_gramo)
        #should_be_eye = mat@(mat.T)
        should_be_eye = mat_after_gramo@(mat_after_gramo.T)
        
        if _log is not None:
            _log.append(("should_be_eye", should_be_eye.detach().clone()))
            pass
        
        if dont_correct_length_with_error_prapagation:
            _iota_of_dim = iota(dim)
            should_be_eye[_iota_of_dim, _iota_of_dim] = 0.#this op also cuts the grad chain.
            loss = loss_func(should_be_eye, torch.zeros_like(matrix))
            if _log is not None:
                _log.append(("should_be_eye after diagonal elements into 0.", should_be_eye.detach().clone()))
                pass
            
            #   ^^^^^   optimizable   ^^^^^
            pass
        else:
            loss = loss_func(should_be_eye, torch.eye(n=dim, device=the_device))
            pass
        
        
        
        optim.zero_grad()
        loss.backward(inputs = train_them)
        if "print in training" and False:
            print(f"loss={loss.item():.3f},        should be 1:{matrix[0].dot(matrix[0]).item():.3f},{matrix[1].dot(matrix[1]).item():.3f},{matrix[2].dot(matrix[2]).item():.3f},{\
                        matrix[:,0].dot(matrix[:,0]).item():.3f},{matrix[:,1].dot(matrix[:,1]).item():.3f},{matrix[:,2].dot(matrix[:,2]).item():.3f}")
            print(f"                  should be 0:{matrix[0].dot(matrix[1]).item():.3f},{matrix[1].dot(matrix[2]).item():.3f},{matrix[2].dot(matrix[0]).item():.3f},{\
                        matrix[:,0].dot(matrix[:,1]).item():.3f},{matrix[:,1].dot(matrix[:,2]).item():.3f},{matrix[:,2].dot(matrix[:,0]).item():.3f}")
            print(f"             vec vec vec : {matrix}")
            assert matrix.grad is not None
            print(f"                               grad grad : {matrix.grad[0]}")
            pass
        
        if _log is not None:
            _log.append(("loss", loss.item()))
            assert matrix.grad
            _log.append(("grad", matrix.grad.detach().clone()))
            pass
        
        optim.step()
        #</ correct the direction>
        
        # if "temp print":
        #     _result_tuple = check_how_much_the_matmul_keeps_the_length_of_vec(mat)
        #     print(f"epoch {epoch}  score {_result_tuple[0].item():.3f}")
        #     pass
        
        
        pass# for epoch in range
    #</ the main protection >
    
    #<  print after finish>
    # print(f"should be 1:{mat[0].dot(mat[0]).item():.3f},{mat[1].dot(mat[1]).item():.3f},{mat[2].dot(mat[2]).item():.3f},{\
    #             mat[:,0].dot(mat[:,0]).item():.3f},{mat[:,1].dot(mat[:,1]).item():.3f},{mat[:,2].dot(mat[:,2]).item():.3f}")
    # print(f"should be 0:{mat[0].dot(mat[1]).item():.3f},{mat[1].dot(mat[2]).item():.3f},{mat[2].dot(mat[0]).item():.3f},{\
    #             mat[:,0].dot(mat[:,1]).item():.3f},{mat[:,1].dot(mat[:,2]).item():.3f},{mat[:,2].dot(mat[:,0]).item():.3f}")
    # print(f"             mat mat mat : {mat}")
    # assert mat.grad is not None
    # print(f"                               grad grad : {mat.grad[0]}")
    #</ print after finish>
    matrix.requires_grad_(does_matrix_require_grad)
    return matrix, _log
if "test" and False:
    def ____test________old_code____correct_the_matrix___version_1():
        import math
        #if False:
        mat = torch.tensor([[16.,16],[1,1]])
        mat = ____old_code____correct_the_matrix___version_1(mat,lr = 0., correction_factor = 0.25, iter_count=1, 
                                    dont_correct_length_with_error_prapagation = True)[0]
        _div_me = math.sqrt(16.)*math.pow(17/2., 0.25)
        _ref_tensor = torch.empty(size=[2,2])
        _ref_tensor[0].fill_(16/math.pow(16*16, 0.25)/math.pow((4*4+1*1)/2., 0.25))
        _ref_tensor[1].fill_(1/math.pow((4*4+1*1)/2., 0.25))
        assert _tensor_equal(mat, _ref_tensor)
        
        # correction_factor = 0.5, this is a bit overshoot.
        mat = torch.tensor([[16.,16],[1,1]])
        mat = ____old_code____correct_the_matrix___version_1(mat,lr = 0., correction_factor = 0.5, iter_count=1, 
                                    dont_correct_length_with_error_prapagation = True)[0]
        assert _tensor_equal(mat, torch.ones(size=[2,2]))
        for _ in range(11):
            _rand_1 = (torch.rand(size=[1])+0.3).item()
            _rand_2 = (torch.rand(size=[1])+0.3).item()
            mat = torch.tensor([[_rand_1, _rand_1],[_rand_2, _rand_2]])
            mat = correct_the_matrix(mat,lr = 0., correction_factor = 0.5, iter_count=1, 
                                        dont_correct_length_with_error_prapagation = True)[0]
            assert _tensor_equal(mat, torch.ones(size=[2,2]))
            pass
        for _ in range(11):
                mat = torch.empty(size=[5,5])
                for ii in range(5):
                    mat[ii].fill_(torch.rand(size=[])+0.3)
                    pass
                mat = correct_the_matrix(mat,lr = 0., correction_factor = 0.5, iter_count=1, 
                                            dont_correct_length_with_error_prapagation = True)[0]
                assert _tensor_equal(mat, torch.ones(size=[5,5]))
                pass
        
        
        
        
        
        mat = torch.tensor([[1.,0.1],[1.,-0.1]])
        _result_tuple_tl = ____old_code____correct_the_matrix___version_1(mat,lr = 0.01, correction_factor = 0., iter_count=1, 
                                                dont_correct_length_with_error_prapagation = True)
        mat = _result_tuple_tl[0]
        _add_them_up = mat[0]+mat[1]
        assert _add_them_up.shape.__len__() == 1
        assert _add_them_up.shape == torch.Size([2])
        assert _tensor_equal(_add_them_up[1], [0.])
        
        
        
        
        mat = torch.tensor([[1.,0.1],[1.,-0.1]])
        _result_tuple_tt = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat)
        _score_before = _result_tuple_tt[0]
        _result_tuple_tl = ____old_code____correct_the_matrix___version_1(mat,lr = 0.01, correction_factor = 0., iter_count=1, dont_correct_length_with_error_prapagation = True)
        mat = _result_tuple_tl[0]
        _result_tuple_tt = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat)
        _score_after = _result_tuple_tt[0]
        
        
        
        
        
        
        
        
        
        
        
        
        DIM = 5
        mat = torch.randn(size=[DIM,DIM], device='cuda')
        mat = ____old_code____correct_the_matrix___version_1(mat,lr = 1., correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#1.3
            
        fds=432
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        DIM = 100
        mat = torch.randn(size=[DIM,DIM], device='cuda')
        for _ in range(100):
            _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat)
            score_before = _result_tuple[0]
            #1w 前几个测试好像是错的。重新跑。
            #进去读一遍所有过程。
            #mat = correct_the_matrix(mat,lr = 0.03, correction_factor = 0.01, iter_count=1)
            #mat = correct_the_matrix(mat,lr = 0., correction_factor = 0.25, iter_count=1)#goes to 1.
            #mat = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=1)#goes to 1.99
            #mat = correct_the_matrix(mat,lr = 0.1, correction_factor = 0., iter_count=10)#lt 1., unstable
            #mat = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#0.78
            #mat = correct_the_matrix(mat,lr = 0.1, correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#0.15
            mat = correct_the_matrix(mat,lr = 1., correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#1.3
            _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat)
            score_after = _result_tuple[0]
            pass
        return 
    ____test________old_code____correct_the_matrix___version_1()
    pass




#matrix:torch.Tensor, lr = 0.3,correction_factor = 0.15, iter_count = 1,
#dont_correct_length_with_error_prapagation = False, __debug__need_log = False)->tuple[torch.Tensor, list]:

def correct_the_matrix___version_2(matrix:torch.Tensor, lr = 0.3,correction_factor = 0.15, 
                            iter_count = 1, dont_correct_length_with_error_prapagation = False, 
                            __debug__need_log = False, 
                            the_length_multiplies_at_most:float|torch.Tensor = 100.)->tuple[torch.Tensor, list|None]:
    '''this function removes the grad stored on the param:matrix'''
    assert is_square_matrix(matrix)
    dim = matrix.shape[0]
    does_matrix_require_grad = matrix.requires_grad
    matrix.requires_grad_()
    matrix.grad = None
    
    the_device = matrix.device
    correction_factor_tensor = torch.tensor(-correction_factor, device=the_device)#0.25 to 0.4. safe range is 0 to 0.5
    
    #dtype adaption
    if isinstance(the_length_multiplies_at_most, float):
        the_length_multiplies_at_most = torch.tensor(the_length_multiplies_at_most, device=the_device)
        pass
    
    #init the iota (if needed)
    if dont_correct_length_with_error_prapagation:
        _iota_of_dim:torch.Tensor|None = iota(dim)
        pass
    else:
        _iota_of_dim = None
        pass
    
    #log?
    if __debug__need_log:
        _log:list|None = []
        pass
    else:
        _log = None
        pass
    
    #<  the main protection >
    for _epoch in range(iter_count):
        #<  correct the length for row>
        with torch.no_grad():
            _temp_len_sqr = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
            if "check it a lil bit" and True:
                _temp_len_sqr__ref = matrix[0].dot(matrix[0]).div(dim)
                assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                pass
            #0.25 to 0.5 power
            mul_me___to_get_unit_length = (_temp_len_sqr*dim).pow(-0.5)
            mul_me___to_get_unit_length = mul_me___to_get_unit_length.minimum(the_length_multiplies_at_most)
            matrix_of_unit_vec_as_rows = matrix * (mul_me___to_get_unit_length.reshape([-1,1]).expand([-1,dim]))
            #^^^^^ optimizable ^^^^^
            if "check it a lil bit" and True:
                assert _tensor_equal((matrix_of_unit_vec_as_rows*matrix_of_unit_vec_as_rows).sum(dim=1), [1.]*dim)
                pass
            
            mul_me_to_correct_length = _temp_len_sqr.pow(correction_factor_tensor)
            mul_me_to_correct_length = mul_me_to_correct_length.minimum(the_length_multiplies_at_most)
            matrix.mul_(mul_me_to_correct_length.reshape([-1,1]).expand([-1,dim]))
            if "check it after modifying":
                _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
                abs_log10__of_ori = _temp_len_sqr.log10().abs()
                abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                better = abs_log10__of_ori.ge(abs_log10__of_after)
                assert better.to(torch.float32).mean()>0.9
                pass
            
            if _log is not None:
                _log.append(("Length corrected by row", matrix.detach().clone()))
                pass
            pass#no grad
        #</ correct the length for row>
        
        #<  correct the direction for row>
        if "only to fold the lines" and True:
            assert matrix.requires_grad
            #<  neural net infra>
            gramo = GradientModification_v2_mean_abs_to_1().to(the_device)

            train_them = [matrix]
            optim = torch.optim.SGD(params=train_them, lr=lr)
            loss_func = torch.nn.MSELoss()
            #</ neural net infra>
            
            mat_after_gramo:torch.Tensor = gramo(matrix.reshape([1,-1])).reshape([dim,dim])
            assert is_square_matrix(mat_after_gramo)
            #should_be_eye = mat@(mat.T)
            should_be_eye = mat_after_gramo@(matrix_of_unit_vec_as_rows.T)#.T is on the right
            
            if _log is not None:
                _log.append(("should_be_eye", should_be_eye.detach().clone()))
                pass
            
            if dont_correct_length_with_error_prapagation:
                should_be_eye[_iota_of_dim, _iota_of_dim] = 0.#this op also cuts the grad chain.
                loss = loss_func(should_be_eye, torch.zeros_like(matrix))
                if _log is not None:
                    _log.append(("should_be_eye after diagonal elements into 0.", should_be_eye.detach().clone()))
                    pass
                #   ^^^^^   optimizable   ^^^^^
                pass
            else:
                loss = loss_func(should_be_eye, torch.eye(n=dim, device=the_device))
                pass
            
            optim.zero_grad()
            loss.backward(inputs = train_them)
            
            if _log is not None:
                _log.append(("loss", loss.item()))
                assert matrix.grad is not None
                _log.append(("grad", matrix.grad.detach().clone()))
                pass
            
            optim.step()
            matrix.grad = None
        #</ correct the direction for row>
        
        
        
        #<  correct the length for column>
        with torch.no_grad():
            # column?
            _temp_len_sqr = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
            if "check it a lil bit" and True:
                _temp_len_sqr__ref = matrix[:,0].dot(matrix[:,0]).div(dim)
                assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                pass
            #0.25 to 0.5 power
            mul_me___to_get_unit_length = (_temp_len_sqr*dim).pow(-0.5)
            mul_me___to_get_unit_length = mul_me___to_get_unit_length.minimum(the_length_multiplies_at_most)
            matrix_of_unit_vec_as_columns = matrix * (mul_me___to_get_unit_length.reshape([1,-1]).expand([dim,-1]))
            #^^^^^ optimizable ^^^^^
            if "check it a lil bit" and True:
                assert _tensor_equal((matrix_of_unit_vec_as_columns*matrix_of_unit_vec_as_columns).sum(dim=0), [1.]*dim)
                pass
            
            mul_me_to_correct_length = _temp_len_sqr.pow(correction_factor_tensor)
            mul_me_to_correct_length = mul_me_to_correct_length.minimum(the_length_multiplies_at_most)
            matrix.mul_(mul_me_to_correct_length.reshape([1,-1]).expand([dim,-1]))
            if "check it after modifying":
                _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
                abs_log10__of_ori = _temp_len_sqr.log10().abs()
                abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                better = abs_log10__of_ori.ge(abs_log10__of_after)
                assert better.to(torch.float32).mean()>0.9
                pass
            
            if _log is not None:
                _log.append(("Length corrected by column", matrix.detach().clone()))
                pass
            
            pass#no grad
        #</ correct the length for column>
        
        
        #<  correct the direction for column>
        if "only to fold the lines" and True:
            assert matrix.requires_grad
            #<  neural net infra>
            gramo = GradientModification_v2_mean_abs_to_1().to(the_device)

            train_them = [matrix]
            optim = torch.optim.SGD(params=train_them, lr=lr)
            loss_func = torch.nn.MSELoss()
            #</ neural net infra>
            
            mat_after_gramo:torch.Tensor = gramo(matrix.reshape([1,-1])).reshape([dim,dim])
            assert is_square_matrix(mat_after_gramo)
            #should_be_eye = mat@(mat.T)
            should_be_eye = matrix_of_unit_vec_as_columns.T@mat_after_gramo#.T is on the left
            
            if _log is not None:
                _log.append(("should_be_eye", should_be_eye.detach().clone()))
                pass
            
            if dont_correct_length_with_error_prapagation:
                should_be_eye[_iota_of_dim, _iota_of_dim] = 0.#this op also cuts the grad chain.
                loss = loss_func(should_be_eye, torch.zeros_like(matrix))
                if _log is not None:
                    _log.append(("should_be_eye after diagonal elements into 0.", should_be_eye.detach().clone()))
                    pass
                #   ^^^^^   optimizable   ^^^^^
                pass
            else:
                loss = loss_func(should_be_eye, torch.eye(n=dim, device=the_device))
                pass
            
            optim.zero_grad()
            loss.backward(inputs = train_them)
            
            if _log is not None:
                _log.append(("loss", loss.item()))
                assert matrix.grad is not None
                _log.append(("grad", matrix.grad.detach().clone()))
                pass
            
            optim.step()
            matrix.grad = None
        #</ correct the direction for column>
        
        pass# for epoch in range
    #</ the main protection >
    
    #set the param back before return.
    matrix.requires_grad_(does_matrix_require_grad)
    return matrix, _log
if "test" and True:
    def ____test____correct_the_matrix___version_2():
        import math
        #if False:
        mat = torch.tensor([[16.,16],[1,1]])
        _result_tuple_tl = correct_the_matrix___version_2(mat,lr = 0., correction_factor = 0.25, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
        mat = _result_tuple_tl[0]
        _div_me = math.sqrt(16.)*math.pow(17/2., 0.25)
        _ref_tensor = torch.empty(size=[2,2])
        _ref_tensor[0].fill_(16/math.pow(16*16, 0.25)/math.pow((4*4+1*1)/2., 0.25))
        _ref_tensor[1].fill_(1/math.pow((4*4+1*1)/2., 0.25))
        assert _tensor_equal(mat, _ref_tensor)
        
        1w 继续
        
        # correction_factor = 0.5, this is a bit overshoot.
        mat = torch.tensor([[16.,16],[1,1]])
        mat = correct_the_matrix___version_2(mat,lr = 0., correction_factor = 0.5, iter_count=1, 
                                    dont_correct_length_with_error_prapagation = True)[0]
        assert _tensor_equal(mat, torch.ones(size=[2,2]))
        for _ in range(11):
            _rand_1 = (torch.rand(size=[1])+0.3).item()
            _rand_2 = (torch.rand(size=[1])+0.3).item()
            mat = torch.tensor([[_rand_1, _rand_1],[_rand_2, _rand_2]])
            mat = correct_the_matrix___version_2(mat,lr = 0., correction_factor = 0.5, iter_count=1, 
                                        dont_correct_length_with_error_prapagation = True)[0]
            assert _tensor_equal(mat, torch.ones(size=[2,2]))
            pass
        for _ in range(11):
            mat = torch.empty(size=[5,5])
            for ii in range(5):
                mat[ii].fill_(torch.rand(size=[])+0.3)
                pass
            mat = correct_the_matrix___version_2(mat,lr = 0., correction_factor = 0.5, iter_count=1, 
                                        dont_correct_length_with_error_prapagation = True)[0]
            assert _tensor_equal(mat, torch.ones(size=[5,5]))
            pass
        
        assert False,'''
        1w 看看这个角度修正的实际行为。进去读。
        加一个不gramo的选项。不一定有用。
        
        应该是，先修长度，修出2个版本，一个是稍微修了一点点的本身要用的版本，另外一个是纯粹只为了得到方向的版本。
        用只有方向的版本来辅助修方向。
        row和column分开做。
        也就是，row长度，row方向，col长度，col方向。循环。
        重新写一个函数。分开。
        
        '''
        
        
        mat = torch.tensor([[1.,0.1],[1.,-0.1]])
        _result_tuple_tl = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=1, 
                                                dont_correct_length_with_error_prapagation = True)
        mat = _result_tuple_tl[0]
        _add_them_up = mat[0]+mat[1]
        assert _add_them_up.shape.__len__() == 1
        assert _add_them_up.shape == torch.Size([2])
        assert _tensor_equal(_add_them_up[1], [0.])
        
        
        
        
        mat = torch.tensor([[1.,0.1],[1.,-0.1]])
        _result_tuple_tt = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat)
        _score_before = _result_tuple_tt[0]
        _result_tuple_tl = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=1, dont_correct_length_with_error_prapagation = True)
        mat = _result_tuple_tl[0]
        _result_tuple_tt = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat)
        _score_after = _result_tuple_tt[0]
        
        
        
        
        
        
        
        
        
        
        
        
        DIM = 5
        mat = torch.randn(size=[DIM,DIM], device='cuda')
        mat = correct_the_matrix(mat,lr = 1., correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#1.3
            
        fds=432
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        DIM = 100
        mat = torch.randn(size=[DIM,DIM], device='cuda')
        for _ in range(100):
            _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat)
            score_before = _result_tuple[0]
            #1w 前几个测试好像是错的。重新跑。
            #进去读一遍所有过程。
            #mat = correct_the_matrix(mat,lr = 0.03, correction_factor = 0.01, iter_count=1)
            #mat = correct_the_matrix(mat,lr = 0., correction_factor = 0.25, iter_count=1)#goes to 1.
            #mat = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=1)#goes to 1.99
            #mat = correct_the_matrix(mat,lr = 0.1, correction_factor = 0., iter_count=10)#lt 1., unstable
            #mat = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#0.78
            #mat = correct_the_matrix(mat,lr = 0.1, correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#0.15
            mat = correct_the_matrix(mat,lr = 1., correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#1.3
            _result_tuple = measure_how_much_the_matmul_keeps_the_length_of_vec__output_abs_log10(mat)
            score_after = _result_tuple[0]
            pass
        return 
    ____test____correct_the_matrix___version_2()
    pass






#     assert False,''' 在这个保护下，训练本身的有效性，就是参数往一个方向推到底能不能真的推出去，还是会被保护回来。 
#     '''
#     pass


#works well.
def does_back_prapagation_work_for__a_vector_dot_itself():
    lr = 0.1
    DIM = 5
    vec_main = torch.randn(size=[DIM],requires_grad=True)
    gramo_main = GradientModification_v2_mean_abs_to_1()
    train_them = [vec_main]

    optim = torch.optim.SGD(params=train_them, lr=lr)
    loss_func = torch.nn.MSELoss()

    for epoch in range(116):
        with torch.no_grad():
            vec_main+=torch.rand_like(vec_main)*lr*2.
            pass
        assert vec_main.requires_grad
        
        vec_main_gramo = gramo_main(vec_main.reshape([1,-1])).reshape([-1])
        
        _length = vec_main_gramo.dot(vec_main_gramo)
        loss_main = loss_func(_length, torch.tensor(1.))
        
        optim.zero_grad()
        loss_main.backward(inputs = train_them)
        
        print(f"_length should be 1: {_length.item():.4f}")
        #if epoch%10 == 10-1:
        print(f"             vec vec vec : {vec_main}")
        print(f"               grad grad : {vec_main.grad}")
        
        optim.step()
        pass
    return

#works well.
def does_back_prapagation_work_for__2__vector_dot_themselves():
    lr = 0.1
    DIM = 5
    vec_1 = torch.randn(size=[DIM],requires_grad=True)
    gramo_1 = GradientModification_v2_mean_abs_to_1()
    vec_2 = torch.randn(size=[DIM],requires_grad=True)
    gramo2 = GradientModification_v2_mean_abs_to_1()
    train_them = [vec_1, vec_2]

    optim = torch.optim.SGD(params=train_them, lr=lr)
    loss_func = torch.nn.MSELoss()

    for epoch in range(116):
        with torch.no_grad():
            vec_1+=torch.rand_like(vec_1)*lr*2.
            vec_2+=torch.rand_like(vec_2)*lr*2.
            pass
        assert vec_1.requires_grad
        assert vec_2.requires_grad
        
        vec_1_gramo = gramo_1(vec_1.reshape([1,-1])).reshape([-1])
        vec_2_gramo = gramo2(vec_2.reshape([1,-1])).reshape([-1])
        
        vec1_dot_vec2 = vec_1_gramo.mul(vec_2_gramo)
        should_be_zero = vec1_dot_vec2.sum()
        loss = loss_func(should_be_zero, torch.tensor(0.))
        
        optim.zero_grad()
        loss.backward(inputs = train_them)
        
        print(f"vec1_dot_vec2: {vec1_dot_vec2}, should_be_zero = {should_be_zero.item():.4f}")
        print(f"             vec vec vec : {vec_1}")
        print(f"               grad grad : {vec_1.grad}")
        print(f"                                 vec vec vec : {vec_2}")
        print(f"                                   grad grad : {vec_2.grad}")
        
        optim.step()
        pass
    return

#works well.
def does_1_self_dot_with_1_cross_dot_work():
    lr = 0.1
    DIM = 5
    vec_main = torch.randn(size=[DIM],requires_grad=True)
    gramo_main = GradientModification_v2_mean_abs_to_1()
    vec_1 = torch.randn(size=[DIM],requires_grad=True)
    gramo_1 = GradientModification_v2_mean_abs_to_1()
    train_them = [vec_main, vec_1]

    optim = torch.optim.SGD(params=train_them, lr=lr)
    loss_func_1 = torch.nn.MSELoss()
    loss_func_2 = torch.nn.MSELoss()

    for epoch in range(116):
        with torch.no_grad():
            vec_main+=torch.rand_like(vec_main)*lr*0.1
            vec_1+=torch.rand_like(vec_1)*lr*      0.1
            pass
        assert vec_main.requires_grad
        assert vec_1.requires_grad
        
        vec_main_gramo = gramo_main(vec_main.reshape([1,-1])).reshape([-1])
        vec_1_gramo = gramo_1(vec_1.reshape([1,-1])).reshape([-1])
        
        _length = vec_main_gramo.dot(vec_main_gramo)
        loss_main = loss_func_1(_length, torch.tensor(1.))
        should_be_zero__from_vec_1 = vec_main_gramo.dot(vec_1_gramo)
        loss_main_from_vec_1 = loss_func_2(should_be_zero__from_vec_1, torch.tensor(0.))
        #loss_total = loss_main*0.9+loss_main_1*0.1
        loss_total = loss_main+loss_main_from_vec_1
        
        optim.zero_grad()
        loss_total.backward(inputs = train_them)
        
        print(f"_length should be 1: {_length.item():.4f}, should_be_zero = {should_be_zero__from_vec_1.item():.4f}")
        #if epoch%10 == 10-1:
        print(f"             vec vec vec : {vec_main}")
        print(f"               grad grad : {vec_main.grad}")
        print(f"                                 vec vec vec : {vec_1}")
        print(f"                                   grad grad : {vec_1.grad}")
        #   pass
        optim.step()
        pass
    return
        