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
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal, avg_log10_safe, get_mask_of_top_element__rough
from pytorch_yagaodirac_v2.Util import iota
from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2_mean_abs_to_1

import torch




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
if False:
    ____protection_hyper_param_scan()
    pass







def check_how_much_the_matmul_keeps_the_length_of_vec(matrix:torch.Tensor,
                                    test_time = 10)->tuple[torch.Tensor,torch.Tensor]:
    assert matrix.shape.__len__() == 2
    assert matrix.shape[0] == matrix.shape[1]
    the_device = matrix.device
    all_score = torch.empty([test_time], device = the_device)
    for epoch in range(test_time):
        vec = torch.randn(size=[matrix.shape[0]], device = the_device)
        while True:
            #check and maybe break
            ori_len_sqr = vec.dot(vec)
            if ori_len_sqr>2.:
                break
            
            #too small, reroll.
            if ori_len_sqr<0.01:
                vec = torch.randn(size=[matrix.shape[0]])
                continue
            
            #still ok, but let's make it larger.
            vec.mul_(torch.rand(size=[])*0.4+1.2)
            pass
        
        #vec = vec.reshape(shape=[1,-1])
        after_matmul = vec@matrix
        new_len_sqr = after_matmul.dot(after_matmul)
        
        score_of_this_epoch = ori_len_sqr.log10().abs() - new_len_sqr.log10().abs()
        all_score[epoch] = score_of_this_epoch
        pass#/ for
    return all_score.mean(),all_score
def ____test____check_how_much_the_matmul_keeps_the_length_of_vec():
    for size in range(3, 15):
        _result_tuple = check_how_much_the_matmul_keeps_the_length_of_vec(torch.eye(n=size),test_time=100)
        assert _tensor_equal(_result_tuple[1], torch.zeros(size=[100]), epsilon=1e-10)
        pass
    #another test is done in the rand_basic_ratation_matrix's test
    return 
if False:
    ____test____check_how_much_the_matmul_keeps_the_length_of_vec()
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
def ____test____rand_basic_ratation_matrix():
    for dim in range(3,14):
        for _ in range(11):
            mat = rand_basic_ratation_matrix(dim)
            _result_tuple = check_how_much_the_matmul_keeps_the_length_of_vec(mat, test_time=100)
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
                
            _result_tuple = check_how_much_the_matmul_keeps_the_length_of_vec(mat, test_time=100)
            assert _tensor_equal(_result_tuple[1], torch.zeros(size=[100]))
            pass#for _
        pass# for dim
    
    return 
if False:
    ____test____rand_basic_ratation_matrix()
    pass














if "some old code " and False:
    # row?
    _temp_len_sqr = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
    if "check it a lil bit":
        _temp_len_sqr__ref = matrix[0].dot(matrix[0]).div(DIM)
        assert _temp_len_sqr[0].eq(_temp_len_sqr__ref)
        pass
    #0.25 to 0.5 power
    mul_me = _temp_len_sqr.pow(-0.25)
    matrix.mul_(mul_me.reshape([-1,1]).expand([-1,DIM]))
    if "check it after modifying":
        _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
        abs_log10__of_ori = _temp_len_sqr.log10().abs()
        abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
        better = abs_log10__of_ori.ge(abs_log10__of_after)
        assert better.to(torch.float32).mean()>0.9
        pass
    
    # column?
    _temp_len_sqr = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
    if "check it a lil bit":
        _temp_len_sqr__ref = matrix[:,0].dot(matrix[:,0]).div(DIM)
        assert _temp_len_sqr[0].eq(_temp_len_sqr__ref)
        pass
    #0.25 to 0.5 power
    mul_me = _temp_len_sqr.pow(-0.25)
    matrix.mul_(mul_me.reshape([1,-1]).expand([DIM,-1]))
    if "check it after modifying":
        _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
        abs_log10__of_ori = _temp_len_sqr.log10().abs()
        abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
        better = abs_log10__of_ori.ge(abs_log10__of_after)
        assert better.to(torch.float32).mean()>0.9
        pass





方阵的可以直接计算出来的长度指标，和用一个dummy vec乘上去观察长度变化，得出的指标，之间的关系是什么。是否有可能推测正交性。
1w 继续。
# 一些特殊的mat可能不行。
# 主对角线可能无法被优化。可能需要额外的保护。
def correct_the_matrix(matrix:torch.Tensor, lr = 0.3,correction_factor = 0.15, iter_count = 1,
                        dont_correct_length_with_error_prapagation = False)->torch.Tensor:
    assert matrix.shape.__len__() == 2
    assert matrix.shape[0] == matrix.shape[1]
    does_matrix_require_grad = matrix.requires_grad
    matrix.requires_grad_()
    
    if dont_correct_length_with_error_prapagation:
        iota_of_dim = iota(matrix.shape[0])
        pass
    else:
        iota_of_dim = None
        pass
    
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
    for epoch in range(iter_count):
        with torch.no_grad():
            #<  add some noise >
            if "add noise" and False:
                matrix+=torch.randn_like(matrix)*lr*0.1
                pass
            #</ add some noise >
            
            #<  correct the length>
            if "correct the length" and True:
                # row?
                _temp_len_sqr = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
                if "check it a lil bit" and False:
                    _temp_len_sqr__ref = matrix[0].dot(matrix[0]).div(DIM)
                    assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                    pass
                #0.25 to 0.5 power
                mul_me = _temp_len_sqr.pow(correction_factor_tensor)
                matrix.mul_(mul_me.reshape([-1,1]).expand([-1,matrix.shape[0]]))
                if "check it after modifying":
                    _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=1)#mul and then sum, it's a dot.
                    abs_log10__of_ori = _temp_len_sqr.log10().abs()
                    abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                    better = abs_log10__of_ori.ge(abs_log10__of_after)
                    assert better.to(torch.float32).mean()>0.9
                    pass
                
                
                # column?
                _temp_len_sqr = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
                if "check it a lil bit" and False:
                    _temp_len_sqr__ref = matrix[:,0].dot(matrix[:,0]).div(DIM)
                    assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                    pass
                #0.25 to 0.5 power
                mul_me = _temp_len_sqr.pow(correction_factor_tensor)
                matrix.mul_(mul_me.reshape([1,-1]).expand([matrix.shape[0],-1]))
                if "check it after modifying":
                    _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
                    abs_log10__of_ori = _temp_len_sqr.log10().abs()
                    abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                    better = abs_log10__of_ori.ge(abs_log10__of_after)
                    assert better.to(torch.float32).mean()>0.9
                    pass
                pass#for "pre protect"
                
                pass
            #</ correct the length>
            
            pass
        assert matrix.requires_grad
        
        #<  correct the direction>
        mat_after_gramo = gramo(matrix.reshape([1,-1])).reshape([-1])
        
        #should_be_eye = mat@(mat.T)
        should_be_eye = mat_after_gramo@(mat_after_gramo.T)
        if dont_correct_length_with_error_prapagation:
            should_be_eye[iota, iota] = 0.#this op also cuts the grad chain.
            loss = loss_func(should_be_eye, torch.zeros_like(matrix))
            #   ^^^^^   optimizable   ^^^^^
            pass
        else:
            loss = loss_func(should_be_eye, torch.eye(n=matrix.shape[0], device=the_device))
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
    return matrix
def ____test____correct_the_matrix():
    DIM = 100
    mat = torch.randn(size=[DIM,DIM])
    for _ in range(100):
        _result_tuple = check_how_much_the_matmul_keeps_the_length_of_vec(mat)
        score_before = _result_tuple[0]
        mat = correct_the_matrix(mat,lr = 0.1, correction_factor = 0.01, iter_count=1)
        _result_tuple = check_how_much_the_matmul_keeps_the_length_of_vec(mat)
        score_after = _result_tuple[0]
        pass
    return 
____test____correct_the_matrix()
if False:
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
        