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

# 1w 

if "开刀前留的笔记，应该是3月2号" and False:
    '''
    做个笔记。

    现在在1440行的那个地方。

    这个工具就是保护向量的二阶长度用的。
    之后要做的事情是看看这个能不能辅助gramo强化的某种mlp

    再后面是数字神经网络，用拆分的方式分治法。
    再后面是逻辑学习，把xor的逻辑优化一下。
    两个工具现在都很难处理xor，现在想到的办法是单独设置一个xor标记，就是彻底的分成普通输出和xor专用输出。
    可能是普通输出上加一个xor标记，如果这个标记点亮了，那就跑xor专用的一个版本，输出也按照专用的方式来解释，最终输出解释出来的内容。

    再后面就是安排意识和意识转移的实验了。

    再后面就是尝试一下破解数学，或者其他的什么应用了。

    中间可以考虑的一些支线任务，比如确认conv的数量级，确认tfm相关的数量级和训练动力学，尤其是softmax相关的那些，到底怎么办。
    '''
    pass



import datetime
from pathlib import Path
import math
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal, \
    iota, is_square_matrix, \
    vector_length_norm, get_vector_length,\
    log10_avg_safe, get_mask_of_top_element__rough,\
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




# What's the test:
# When correct the length in one direction(say row), when the orthogonality is guarunteed(or not, idk),
# the correction also affects the length measured by the other direction.
# This test is to measure how much it affects.
# Result:
# when the correction_factor is in [0.25, 0.3], the correction of one direction 
# doesn't affect the other direction much.
# but basicall, when the factor is 0.28 to 0.3 or 0.36 to 0.38, it's slightly a bit better.
# To what I know, I decide to use 0.28 for now.


if "old code. ____protection_hyper_param_scan" and False:
    def ____protection_hyper_param_scan():
        
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
                    assert False, "is it mean    vvvv     or sum here ????"
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
                    assert False, "is it mean    vvvv     or sum here ????"
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
    assert False, "1w the gramo api updated"
    gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False).to(the_device)
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
                assert False, "is it mean or sum   vvvv  here ????"
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
                assert False, "is it mean or sum   vvvv  here ????"
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
        _result_tuple_tt = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
        _score_before = _result_tuple_tt[0]
        _result_tuple_tl = ____old_code____correct_the_matrix___version_1(mat,lr = 0.01, correction_factor = 0., iter_count=1, dont_correct_length_with_error_prapagation = True)
        mat = _result_tuple_tl[0]
        _result_tuple_tt = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
        _score_after = _result_tuple_tt[0]
        
        
        
        
        
        
        
        
        
        
        
        
        DIM = 5
        mat = torch.randn(size=[DIM,DIM], device='cuda')
        mat = ____old_code____correct_the_matrix___version_1(mat,lr = 1., correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#1.3
            
        fds=432
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        DIM = 100
        mat = torch.randn(size=[DIM,DIM], device='cuda')
        for _ in range(100):
            _result_tuple = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
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
            _result_tuple = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
            score_after = _result_tuple[0]
            pass
        return 
    ____test________old_code____correct_the_matrix___version_1()
    pass

#matrix:torch.Tensor, lr = 0.3,correction_factor = 0.15, iter_count = 1,
#dont_correct_length_with_error_prapagation = False, __debug__need_log = False)->tuple[torch.Tensor, list]:






















def correct_the_matrix___version_2(matrix:torch.Tensor, length_factor = 0.15, angle_factor = 0.3,
                            iter_count = 1, dont_correct_length_with_error_prapagation = False, 
                            __debug__need_log = False, __debug__ckeck_alone_the_way = False, 
                            the_length_multiplies_at_most:float|torch.Tensor = 100.)->tuple[torch.Tensor, list|None]:
    '''this function removes the grad stored on the param:matrix'''
    assert is_square_matrix(matrix)
    
    #<  param name translation>
    lr = angle_factor
    correction_factor = length_factor
    #</ param name translation>
    
    dim = matrix.shape[0]
    does_matrix_require_grad = matrix.requires_grad
    matrix.requires_grad_()
    matrix.grad = None
    
    the_device = matrix.device
    the_dtype = matrix.dtype
    correction_factor_tensor = torch.tensor(-correction_factor, device=the_device, dtype=the_dtype)#0.25 to 0.4. safe range is 0 to 0.5
    
    #dtype adaption
    if isinstance(the_length_multiplies_at_most, float):
        the_length_multiplies_at_most = torch.tensor(the_length_multiplies_at_most, device=the_device, dtype=the_dtype)
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
        if _log is not None:
            _log.append((_line_(),"-------------------------"))
            _log.append(("MATRIX   ready to correct by row", matrix.detach().clone(), _line_()))
            pass
        with torch.no_grad():
            #assert False, "is it mean or sum   vvvv  here ????"
            _temp_len_sqr = matrix.mul(matrix).sum(dim=1)#mean(dim=1)#mul and then sum, it's a dot.
            if "check it a lil bit" and __debug__ckeck_alone_the_way:
                _temp_len_sqr__ref = matrix[0].dot(matrix[0])
                assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                pass
            #0.25 to 0.5 power
            mul_me___to_get_unit_length = (_temp_len_sqr).pow(-0.5)
            mul_me___to_get_unit_length = mul_me___to_get_unit_length.minimum(the_length_multiplies_at_most)
            matrix_of_unit_vec_as_rows = matrix * (mul_me___to_get_unit_length.reshape([-1,1]).expand([-1,dim]))
            #^^^^^ optimizable ^^^^^
            if "check it a lil bit" and __debug__ckeck_alone_the_way:
                assert _tensor_equal((matrix_of_unit_vec_as_rows*matrix_of_unit_vec_as_rows).sum(dim=1), [1.]*dim)
                pass
            
            mul_me_to_correct_length = _temp_len_sqr.pow(correction_factor_tensor)
            mul_me_to_correct_length = mul_me_to_correct_length.minimum(the_length_multiplies_at_most)
            matrix.mul_(mul_me_to_correct_length.reshape([-1,1]).expand([-1,dim]))
            if "check it after modifying" and __debug__ckeck_alone_the_way:
                _temp_len_sqr__after_modifying = matrix.mul(matrix).sum(dim=1)#mul and then sum, it's a dot.
                abs_log10_of_element__of_ori = (_temp_len_sqr*dim).log10().abs()
                abs_log10_of_element__of_after = (_temp_len_sqr__after_modifying*dim).log10().abs()
                better = abs_log10_of_element__of_ori.ge(abs_log10_of_element__of_after)
                assert better.to(torch.float32).mean()>0.9
                pass
            
            if _log is not None:
                _temp_len_sqr
                _log.append(("_temp_len_sqr", _temp_len_sqr.detach().clone(), _line_()))
                _log.append(("mul_me_to_correct_length", mul_me_to_correct_length.detach().clone(), _line_()))
                _log.append(("MATRIX   Length corrected by row", matrix.detach().clone(), _line_()))
                _log.append((_line_(), "-------------"))
                pass
            pass#no grad
        #</ correct the length for row>
        
        #<  correct the direction for row>
        if "only to fold the lines" and True:
            assert matrix.requires_grad
            #<  neural net infra>
            gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False,
                                                per_what="vector", device=the_device, dtype=the_dtype)
            
            #1w sum?
            train_them = [matrix]
            optim = torch.optim.SGD(params=train_them, lr=lr)
            loss_func = torch.nn.MSELoss()
            #</ neural net infra>
            
            #old mat_after_gramo:torch.Tensor = gramo(matrix.reshape([1,-1])).reshape([dim,dim])
            #1w
            mat_after_gramo:torch.Tensor = gramo(matrix)#no .T, this is row.
            assert is_square_matrix(mat_after_gramo)
            #should_be_eye = mat@(mat.T)
            should_be_eye = mat_after_gramo@(matrix_of_unit_vec_as_rows.T)#.T is on the right
            
            if _log is not None:
                _log.append(("should_be_eye", should_be_eye.detach().clone(), _line_()))
                pass
            
            if dont_correct_length_with_error_prapagation:
                should_be_eye[_iota_of_dim, _iota_of_dim] = 0.#this op also cuts the grad chain.
                loss = loss_func(should_be_eye, torch.zeros_like(matrix))
                if _log is not None:
                    _log.append(("should_be_eye after diagonal elements into 0.", should_be_eye.detach().clone(), _line_()))
                    pass
                #   ^^^^^   optimizable   ^^^^^
                pass
            else:
                loss = loss_func(should_be_eye, torch.eye(n=dim, device=the_device, dtype=the_dtype))
                pass
            
            optim.zero_grad()
            loss.backward(inputs = train_them)
            
            if _log is not None:
                _log.append(("loss", loss.item(), _line_()))
                assert matrix.grad is not None
                _log.append(("ori grad", matrix.grad.detach().clone(), _line_()))
                pass
            
            #1w modify the grad
            vector_length_in_matrix = get_vector_length(matrix, result_dtype = matrix.dtype)#row
            vector_length_in_matrix = vector_length_in_matrix.reshape([-1,1]).expand([-1,dim])
            if "some check" and __debug__ckeck_alone_the_way:
                assert is_square_matrix(vector_length_in_matrix)
                pass
                
            assert matrix.grad is not None
            matrix.grad = matrix.grad * vector_length_in_matrix
            if "some check" and __debug__ckeck_alone_the_way:
                _grad_length = get_vector_length(matrix.grad)
                for ii in range(dim):
                    assert _tensor_equal(_grad_length[ii], vector_length_in_matrix[ii]) or \
                        _grad_length[ii]<gramo.epsilon*gramo.mul_me_when_g_too_small*1.01
                    pass
                pass
            
            if _log is not None:
                _log.append(("vector_length_in_matrix", vector_length_in_matrix.detach().clone(), _line_()))
                _log.append(("scaled grad", matrix.grad.detach().clone(), _line_()))
                pass
            
            
            optim.step()
            matrix.grad = None
        #</ correct the direction for row>
        
        
        
        
        
        
        
        #<  correct the length for column>
        if _log is not None:
            _log.append((_line_(),"-------------------------"))
            _log.append(("MATRIX   ready to correct by column", matrix.detach().clone(), _line_()))
            pass
        with torch.no_grad():
            # column?
            #assert False, "is it mean or sum   vvvv  here ????"
            _temp_len_sqr = matrix.mul(matrix).sum(dim=0)#mul and then sum, it's a dot.
            if "check it a lil bit" and __debug__ckeck_alone_the_way:
                _temp_len_sqr__ref = matrix[:,0].dot(matrix[:,0])
                assert _tensor_equal(_temp_len_sqr[0], _temp_len_sqr__ref)
                pass
            #0.25 to 0.5 power
            mul_me___to_get_unit_length = (_temp_len_sqr).pow(-0.5)
            mul_me___to_get_unit_length = mul_me___to_get_unit_length.minimum(the_length_multiplies_at_most)
            matrix_of_unit_vec_as_columns = matrix * (mul_me___to_get_unit_length.reshape([1,-1]).expand([dim,-1]))
            #^^^^^ optimizable ^^^^^
            if "check it a lil bit" and __debug__ckeck_alone_the_way:
                assert _tensor_equal((matrix_of_unit_vec_as_columns*matrix_of_unit_vec_as_columns).sum(dim=0), [1.]*dim)
                pass
            
            mul_me_to_correct_length = _temp_len_sqr.pow(correction_factor_tensor)
            mul_me_to_correct_length = mul_me_to_correct_length.minimum(the_length_multiplies_at_most)
            matrix.mul_(mul_me_to_correct_length.reshape([1,-1]).expand([dim,-1]))
            if "check it after modifying" and __debug__ckeck_alone_the_way:
                _temp_len_sqr__after_modifying = matrix.mul(matrix).mean(dim=0)#mul and then sum, it's a dot.
                abs_log10_of_element__of_ori = (_temp_len_sqr*dim).log10().abs()
                abs_log10_of_element__of_after = (_temp_len_sqr__after_modifying*dim).log10().abs()
                better = abs_log10_of_element__of_ori.ge(abs_log10_of_element__of_after)
                assert better.to(torch.float32).mean()>0.9
                pass
            
            if _log is not None:
                _log.append(("_temp_len_sqr", _temp_len_sqr.detach().clone(), _line_()))
                _log.append(("mul_me_to_correct_length", mul_me_to_correct_length.detach().clone(), _line_()))
                _log.append(("MATRIX   Length corrected by column", matrix.detach().clone(), _line_()))
                _log.append((_line_(), "-------------"))
                pass
            
            pass#no grad
        #</ correct the length for column>
        
        #<  correct the direction for column>
        if "only to fold the lines" and True:
            assert matrix.requires_grad
            #<  neural net infra>
            #1w 
            gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False,
                                                    per_what="vector", device=the_device, dtype=the_dtype)
            
            
            train_them = [matrix]
            optim = torch.optim.SGD(params=train_them, lr=lr)
            loss_func = torch.nn.MSELoss()
            #</ neural net infra>
            
            #old mat_after_gramo:torch.Tensor = gramo(matrix.reshape([1,-1])).reshape([dim,dim])
            #1w
            mat_after_gramo = gramo(matrix.T).T#column.
            assert is_square_matrix(mat_after_gramo)
            #should_be_eye = mat@(mat.T)
            should_be_eye = matrix_of_unit_vec_as_columns.T@mat_after_gramo#.T is on the left
            
            if _log is not None:
                _log.append(("should_be_eye", should_be_eye.detach().clone(), _line_()))
                pass
            
            if dont_correct_length_with_error_prapagation:
                should_be_eye[_iota_of_dim, _iota_of_dim] = 0.#this op also cuts the grad chain.
                loss = loss_func(should_be_eye, torch.zeros_like(matrix))
                if _log is not None:
                    _log.append(("should_be_eye after diagonal elements into 0.", should_be_eye.detach().clone(), _line_()))
                    pass
                #   ^^^^^   optimizable   ^^^^^
                pass
            else:
                loss = loss_func(should_be_eye, torch.eye(n=dim, device=the_device, dtype=the_dtype))
                pass
            
            optim.zero_grad()
            loss.backward(inputs = train_them)
            
            if _log is not None:
                _log.append(("loss", loss.item(), _line_()))
                assert matrix.grad is not None
                _log.append(("ori grad", matrix.grad.detach().clone(), _line_()))
                pass
            
            #1w modify the grad
            vector_length_in_matrix = get_vector_length(matrix.T, result_dtype = matrix.dtype)#column
            vector_length_in_matrix = vector_length_in_matrix.reshape([1,-1]).expand([dim,-1])
            if "some check" and __debug__ckeck_alone_the_way:
                assert is_square_matrix(vector_length_in_matrix)
                pass
            
            assert matrix.grad is not None
            matrix.grad = matrix.grad * vector_length_in_matrix
            if "some check" and __debug__ckeck_alone_the_way:
                _grad_length = get_vector_length(matrix.grad)
                for ii in range(dim):
                    assert _tensor_equal(_grad_length[ii], vector_length_in_matrix[ii]) or \
                        _grad_length[ii]<gramo.epsilon*gramo.mul_me_when_g_too_small*1.01
                    pass
                pass
            
            if _log is not None:
                _log.append(("vector_length_in_matrix", vector_length_in_matrix.detach().clone(), _line_()))
                _log.append(("scaled grad", matrix.grad.detach().clone(), _line_()))
                pass
            
            optim.step()
            matrix.grad = None
        #</ correct the direction for column>
        
        pass# for epoch in range
    #</ the main protection >
    
    #set the param back before return.
    matrix.requires_grad_(does_matrix_require_grad)
    return matrix, _log


if "length correction only" and __DEBUG_ME__() and True:
    import random, math
    def ____test____correct_the_matrix___version_2____length_correction():
        if "the step 1 len correction   for 0.25" and False:
            #<  preparation>
            _one_over_sqrt_2 = 1/math.sqrt(2.)
            mat = torch.empty(size=[2,2],dtype=torch.float64)
            mat.fill_(_one_over_sqrt_2)
            mat[0] *= 4.
            assert _tensor_equal(mat, [[4*0.7071,  4*0.7071],
                                        [ 0.7071,    0.7071]])
            #<  calc>
            _result_tuple_tl = correct_the_matrix___version_2(mat,length_factor = 0.25, angle_factor=0., iter_count=1, 
                                dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            _log = _result_tuple_tl[1]
            #<  assertions>
            assert _log[2][0] == "_temp_len_sqr"
            assert _tensor_equal(_log[2][1], torch.tensor([16., 1]))
            
            assert _log[4][0] == 'MATRIX   Length corrected by row'
            assert _tensor_equal(_log[4][1], torch.tensor([[2*0.7071, 2*0.7071],
                                                            [ 0.7071,   0.7071]]))
            pass
        
        if "the step 2 len correction   for 0.25" and False:
            #<  preparation>
            _one_over_sqrt_2 = 1/math.sqrt(2.)
            mat = torch.empty(size=[2,2],dtype=torch.float64)
            mat.fill_(_one_over_sqrt_2)
            mat[0] *= 16.*25*2
            mat[1] *= 9.*25*2
            assert _tensor_equal(mat, [[16*25*2*0.7071, 16*25*2*0.7071],
                                        [9*25*2*0.7071,  9*25*2*0.7071]], epsilon=0.01)
            #<  calc>
            _result_tuple_tl = correct_the_matrix___version_2(mat,length_factor = 0.25, angle_factor=0., iter_count=1, 
                                dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            result_mat = _result_tuple_tl[0]
            _log = _result_tuple_tl[1]
            #<  assertions>
            assert _log[2][0] == "_temp_len_sqr"
            assert _tensor_equal(_log[2][1], torch.tensor([16*16*25*25.*4, 9*9*25*25 *4]))
            
            assert _log[4][0] == 'MATRIX   Length corrected by row'
            assert _tensor_equal(_log[4][1], torch.tensor([[ 4*5, 4*5],
                                                            [3*5, 3*5]]), epsilon=0.001)
            
            assert _log[14][0] == "_temp_len_sqr"
            assert _tensor_equal(_log[14][1], torch.tensor([25*25, 25*25]), epsilon=0.001)
            
            
            assert _tensor_equal(result_mat, torch.tensor([[ 4, 4],
                                                            [3, 3]]), epsilon=0.001)
            pass
        
        
        if "0.3333333333333333333333333333" and True:
            #<  preparation>
            _one_over_sqrt_2 = 1/math.sqrt(2.)
            mat = torch.empty(size=[2,2],dtype=torch.float64)
            mat.fill_(_one_over_sqrt_2)
            mat[0] *= 1000
            #mat[1] *= z
            assert _tensor_equal(mat, [[1000*0.7071, 1000*0.7071],
                                        [    0.7071,      0.7071]], epsilon=0.01)
            #<  calc>
            _result_tuple_tl = correct_the_matrix___version_2(mat,length_factor = 0.333333333333333333333333333333, 
                                                    angle_factor=0., iter_count=1, 
                                dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            result_mat = _result_tuple_tl[0]
            _log = _result_tuple_tl[1]
            #<  assertions>
            assert _log[2][0] == "_temp_len_sqr"
            assert _tensor_equal(_log[2][1], torch.tensor([1000*1000, 1]))
            
            assert _log[4][0] == 'MATRIX   Length corrected by row'
            assert _tensor_equal(_log[4][1], torch.tensor([[ 10*0.7071, 10*0.7071],
                                                            [   0.7071,    0.7071]]), epsilon=0.001)
            1w
            1w
            1w
            1w继续
            assert _log[14][0] == "_temp_len_sqr"
            assert _tensor_equal(_log[14][1], torch.tensor([25*25, 25*25]), epsilon=0.001)
            
            
            assert _tensor_equal(result_mat, torch.tensor([[ 4, 4],
                                                            [3, 3]]), epsilon=0.001)
            pass
        
        
        fds=432
        
        
        # old code below.
        ########################################################
        ########################################################
        ########################################################
        ########################################################
        ########################################################
        
        
        
        
        
        
        #special case 1        
        _one_over_sqrt_2 = 1/math.sqrt(2.)
        mat = torch.empty(size=[2,2],dtype=torch.float64)
        mat.fill_(_one_over_sqrt_2)
        mat[0] *= 4.
        assert _tensor_equal(mat, [[4*0.7071,  4*0.7071],
                                    [ 0.7071,    0.7071]])
        _result_tuple_tl = correct_the_matrix___version_2(mat,length_factor = 0.25, angle_factor=0., iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
        mat = _result_tuple_tl[0]
        _log = _result_tuple_tl[1]
        
        assert _tensor_equal(mat, torch.tensor([[1.1247, 1.1247],
                                                [0.5623, 0.5623]]))
        assert _log[4][0] == 'MATRIX   Length corrected by row'
        assert _tensor_equal(_log[4][1], torch.tensor([[ 1.4142, 1.4142],
                                                        [0.7071, 0.7071]]))
        
        
        #_a1 = 1.4142*0.7953
        assert _float_equal(1.4142*0.7953, 1.1247)
        #_a2 = 0.7071*0.7953
        assert _float_equal(0.7071*0.7953, 0.5623)
        #_a3 = math.pow(2.5, -0.25)
        assert _float_equal(math.pow(2.5, -0.25), 0.7953)
        
        _temp_length_sqr = 2*2*0.5+1*1*0.5
        assert _temp_length_sqr == 2.5
        _div_me = math.pow(_temp_length_sqr, 0.25)
        __some_ref = math.pow(2.5, -0.25)
        assert _float_equal(1/_div_me, __some_ref)
        _element_for_row_0 = 0.7071 *2. /_div_me
        assert _float_equal(_element_for_row_0, 1.1247, epsilon=0.05)
        _element_for_row_1 = 0.7071     /_div_me
        assert _float_equal(_element_for_row_1, 0.5623)
        _ref_tensor = torch.empty(size=[2,2])
        _ref_tensor[0].fill_(_element_for_row_0)
        _ref_tensor[1].fill_(_element_for_row_1)
        assert _tensor_equal(mat, _ref_tensor, epsilon=1e-2)
        
        #special case 2
        mat = torch.tensor([[16.,16],[1,1]])
        mat = correct_the_matrix___version_2(mat, length_factor = 0.5, angle_factor=0., iter_count=1, 
                                                    dont_correct_length_with_error_prapagation = True)[0]
        assert _tensor_equal(mat, torch.ones(size=[2,2])*math.sqrt(0.5))
        
        # random test.
        for _ in range(6):
            #print(iiiii, end=", ")
            dim = random.randint(2, 300)#2 to 10000 works.
            #dim = random.randint(3000,10000)this is slow. But passed.
            #correction_factor = 0.25#先看看再说。
            correction_factor = random.random()*0.3+0.1
            assert correction_factor>0.09 and correction_factor<0.41
            
            # normal path
            mat_input = torch.randn(size=[dim,dim], device='cuda')/math.sqrt(dim)*1.3
            _result_tuple_tl = correct_the_matrix___version_2(mat_input, length_factor = correction_factor, 
                                                            angle_factor=0., iter_count=1, 
                                dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            mat_output = _result_tuple_tl[0]
            _log = _result_tuple_tl[1]
            
            # reverse path.
            _ref_mat_answer = mat_output.detach().clone()
            
            _column_length = get_vector_length(_ref_mat_answer.T)#column
            _column_length_target = _column_length.pow(0.5/(0.5-correction_factor))
            ___column_length_target_sqr = _column_length_target*_column_length_target
            assert _tensor_equal(___column_length_target_sqr, _log[14][1])
            _mul_me__to_get_column_length = _column_length_target/_column_length
            assert _tensor_equal(1./_mul_me__to_get_column_length, _log[15][1]) #[0.7953, 0.7953], 1.2574
            #or oneshot? correction_factor/(0.5-correction_factor)
            _mul_me__to_get_column_length__expanded = _mul_me__to_get_column_length.reshape([1,-1]).expand([dim,-1])
            _ref_mat_answer__halfway = _ref_mat_answer*_mul_me__to_get_column_length__expanded
            assert _tensor_equal(_ref_mat_answer__halfway, _log[13][1])
            
            _row_length = get_vector_length(_ref_mat_answer__halfway)#row
            _row_length_target = _row_length.pow(0.5/(0.5-correction_factor))
            ___row_length_target_sqr = _row_length_target*_row_length_target
            assert _tensor_equal(___row_length_target_sqr, _log[2][1])
            _mul_me__to_get_row_length = _row_length_target/_row_length
            assert _tensor_equal(1./_mul_me__to_get_row_length, _log[3][1])
            _mul_me__to_get_row_length__expanded = _mul_me__to_get_row_length.reshape([-1,1]).expand([-1,dim])
            _ref_mat_input = _ref_mat_answer__halfway*_mul_me__to_get_row_length__expanded
            assert _tensor_equal(_ref_mat_input, _log[1][1])
            pass
        
        for _ in range(11):
            _rand_1 = (torch.rand(size=[1])+0.3).item()
            _rand_2 = (torch.rand(size=[1])+0.3).item()
            mat = torch.tensor([[_rand_1, _rand_1],[_rand_2, _rand_2]])
            mat = correct_the_matrix___version_2(mat,length_factor = 0.5, angle_factor=0., iter_count=1, 
                            dont_correct_length_with_error_prapagation = True)[0]
            assert _tensor_equal(mat, torch.ones(size=[2,2])*math.sqrt(1/2.))
            pass
        for _ in range(11):
            mat = torch.empty(size=[5,5])
            for ii in range(5):
                mat[ii].fill_(torch.rand(size=[])+0.3)
                pass
            mat = correct_the_matrix___version_2(mat,length_factor = 0.5, angle_factor=0., iter_count=1, 
                            dont_correct_length_with_error_prapagation = True)[0]
            assert _tensor_equal(mat, torch.ones(size=[5,5])*math.sqrt(1/5.))
            pass
        
        return 
    
    ____test____correct_the_matrix___version_2____length_correction()
    pass
if "angle correction only" and __DEBUG_ME__() and True:
    def ____test____correct_the_matrix___version_2____angle_correction():
        import random, math
        if True:
            "orthogonal, nothing is touched."
            mat = torch.tensor([[1.,0],[0,1]])
            _result_tuple_tl = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., angle_factor = 0.1, iter_count=1, 
                        dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            mat_after = _result_tuple_tl[0]
            assert _tensor_equal(mat_after, mat)
            
            "2d 45deg, slightly correct it."
            for _ in range(5):
                #<  rand angle>
                _angle = (torch.rand([]))*0.1+0.05
                if random.random()<0.5:
                    _angle *= -1.
                    pass
                _angle += torch.pi/4.
                #</ rand angle>
                dont_correct_length_with_error_prapagation = random.choice([True, False])
                
                #<  calc>
                mat = torch.tensor([[_angle.sin().item(),_angle.cos().item()],
                                    [1./math.sqrt(2.), -1./math.sqrt(2.)]])#trigonometric func in rad
                _result_tuple_tl = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., angle_factor = 0.01, iter_count=1, 
                        dont_correct_length_with_error_prapagation = dont_correct_length_with_error_prapagation, __debug__need_log = True)
                mat_after = _result_tuple_tl[0]
                #</ calc>
                #<  ref>
                _ref = torch.tensor([(1./math.sqrt(2.)), 1./math.sqrt(2.)])
                assert _tensor_equal(mat[0].dot(mat[0]), [1.], epsilon=1e-4)
                before_score = mat[0].dot(_ref)
                assert before_score.lt(0.9988)
                #</ ref>
                #<  assertions>
                mat_after___index_0 = mat_after[0]
                _len_of__mat_after___index_1 = mat_after___index_0.dot(mat_after___index_0).sqrt()
                assert _tensor_equal(_len_of__mat_after___index_1, [1.], epsilon=0.02)
                mat_after___index_0___with_len_of_1 = mat_after___index_0/_len_of__mat_after___index_1
                assert _tensor_equal(mat_after___index_0___with_len_of_1.dot(mat_after___index_0___with_len_of_1), [1.], epsilon=1e-4)
                after_score = mat_after___index_0___with_len_of_1.dot(_ref)
                assert after_score.ge(before_score)#new is better than old.
                #</ assertions>
                pass# for _
            
            
            "the same direction, the length optimizes, if the flag is set to false."
            lr = 0.1
            mat = torch.tensor([[1.,0],[1,0]])
            _result_tuple_tl = correct_the_matrix___version_2(mat, length_factor=0., angle_factor = lr, iter_count=1, 
                        dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this flag.
            mat = _result_tuple_tl[0]
            assert _tensor_equal(mat, [[1-lr, 0],[1-lr, 0]])
            #the grad is 0.7071, but the grad is scaled up by sqrt(2), so the grad is 1 eventually.
            # ori - lr  * grad * vec_length( only the by-row works in this case.)
            #  1. - 0.1 * 1    * 1
            if "old code, the old ref" and False:
                _ref_mat__no_gramo = torch.tensor( [[1.,0],
                                                    [1 ,0]], requires_grad=True)
                _temp = torch.zeros(size=[2,2])
                _temp[0,1] = _ref_mat__no_gramo[0].dot(_ref_mat__no_gramo[1])
                _temp[1,0] = _ref_mat__no_gramo[1].dot(_ref_mat__no_gramo[0])
                _temp.backward(inputs=[_ref_mat__no_gramo], gradient=_temp.detach().clone()*2.)
                assert _ref_mat__no_gramo.grad is not None
                assert _tensor_equal(_ref_mat__no_gramo.grad,  [[4., 0.],
                                                                [4., 0.]])
                
                _ref_mat = torch.tensor(   [[1.,0],
                                            [1 ,0]], requires_grad=True)
                gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False)
                _ref_mat__after_gramo = gramo(_ref_mat)
                _temp = torch.zeros(size=[2,2])
                _temp[0,1] = _ref_mat__after_gramo[0].dot(_ref_mat__after_gramo[1])
                _temp[1,0] = _ref_mat__after_gramo[1].dot(_ref_mat__after_gramo[0])
                _temp.backward(inputs=[_ref_mat], gradient=_temp.detach().clone()*2.)
                assert _ref_mat.grad is not None
                assert _tensor_equal(_ref_mat.grad,    [[1.4142, 0.0000],
                                                        [1.4142, 0.0000]])
                pass
            
            
            lr = 0.1
            mat = torch.tensor([[1.1111111111,0],[1.1111111111,0]]) #1w 改成了1.1看看。
            _result_tuple_tl = correct_the_matrix___version_2(mat, length_factor = 0., angle_factor = lr, iter_count=1, 
                        dont_correct_length_with_error_prapagation = False, __debug__need_log = True)
            #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this flag.
            mat, _log= _result_tuple_tl
            assert _tensor_equal(_log[12][1] ,[[1.,0],[1.,0]])
            assert _tensor_equal(mat, [[1-lr, 0],[1-lr, 0]])
            # in the "by-row angle", the formula is the same as uppon, 
            # ori    - lr  * grad * vec_length
            # 1.1111 - 0.1 * 1    * 1.1111
            #this results in a [[1.,0],[1.,0]], so the rest is similar to the last test uppon.
            # ori    - lr  * grad * vec_length
            #  1.    - 0.1 * 1    * 1
            #the second column is 0, so it doesn't do anything. The formula also shows the same conclusion.
            if "old ref, a bit ref" and False:
                x = torch.tensor(0.7071, requires_grad=True)
                pred = 2.*x*x
                Loss = (pred-1)*(pred-1)
                Loss.backward(inputs=x)
                assert x.grad is not None
                assert _tensor_equal(x.grad, [0], epsilon=1e-3)
                
                x = torch.tensor(-0.7071, requires_grad=True)
                pred = 2.*x*x
                Loss = (pred-1)*(pred-1)
                Loss.backward(inputs=x)
                assert x.grad is not None
                assert _tensor_equal(x.grad, [0], epsilon=1e-3)
                
                x = torch.tensor(0., requires_grad=True)
                pred = 2.*x*x
                Loss = (pred-1)*(pred-1)
                Loss.backward(inputs=x)
                assert x.grad is not None
                assert _tensor_equal(x.grad, [0], epsilon=1e-3)
                pass
            
        return 
    ____test____correct_the_matrix___version_2____angle_correction()
        
if "param scan         the old test" and __DEBUG_ME__() and True:
    def ____test____correct_the_matrix___version_2____param_scan():
        "let's scan param a bit."
        #   dim    lr
        #assert False, "1000   200  检查一下  1w "
        #   300    1
        #   100    0.01
        
        
        
        the_device = 'cpu'
        
        dim = 300
        
        dont_correct_length_with_error_prapagation = True
        
        
        #for iter_count in [1,2,3,4,5,6,7,8,10,20,30]:#4 or 5 is enough.
        iter_count = 1
        #for lr in [0.1, 1.,10,15,22,33,47,56,68,100,150,220,330,470,560,680,1000,10000]:#0.02,0.03,0.035, 0.04,0.045,0.05,0.055,0.06]:
        #for lr in [0, 0.0001, 0.001,0.003, 0.004, 0.005, 0.006, 0.0065, 0.007, 0.0075,0.008,0.0085,0.009,0.0095,0.01,0.1, 0.3, 1.,10,100,10000]:#0.02,0.03,0.035, 0.04,0.045,0.05,0.055,0.06]:
        for lr in [1.,2.2,4.7,10,22,47,100]:#0.02,0.03,0.035, 0.04,0.045,0.05,0.055,0.06]:
        #lr = 5.
        #lr = 0.008#???
            #lr = 200.# this is the best??????
        #for correction_factor in [0.1,0.15, 0.2, 0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.75,0.8]:#这个真的很抽象。。。。。
        #for correction_factor in [0.4,0.42, 0.44,0.45,0.46,0.48,0.5,0.52,0.55,0.57,0.6,0.62]:#这个真的很抽象。。。。。
        #for correction_factor in [0.48,0.49,0.5,0.51,0.52,0.53,0.54]:#这个真的很抽象。。。。。
            #correction_factor = 0.7#best?
            correction_factor = 0.52#iter_count = 1
        #if True:
        
            #<  init test group>
            test_time = 10
            angle_score_tensor = torch.empty(size=[test_time], device=the_device)
            length_score_tensor = torch.empty(size=[test_time], device=the_device)
            measure_score_tensor = torch.empty(size=[test_time], device=the_device)
            #angle_similarity_tensor = torch.empty(size=[test_time], device=the_device)
            #</ init test group>
            for test_count in range(test_time):
                #<  init  />
                mat_ori = torch.randn(size=[dim, dim], device=the_device)
                #<  calc>
                _result_tuple_tl = correct_the_matrix___version_2(mat_ori.detach().clone(),
                                length_factor = correction_factor, angle_factor =  lr, iter_count=iter_count, 
                                dont_correct_length_with_error_prapagation = dont_correct_length_with_error_prapagation)
                mat_after = _result_tuple_tl[0]
                #</ calc>
                #<  measure>
                measure_score = LOSS__vec_len_retention__of_a_mat_in_matmul(mat_after,test_time=50)[0]
                measure_score_tensor[test_count] = measure_score
                
                #angle_similarity_tensor[test_count] = ____xxxx____LOSS__angle_similarity(mat_after, mat_ori)
                
                length_score, angle_score = LOSS__mat_is_standard_orthogonal(mat_after)
                length_score_tensor[test_count] = length_score
                angle_score_tensor[test_count] = angle_score
                #</ measure>
                pass
            
            print(f"{dim:4}   {iter_count:2}  {lr:.4f}  {correction_factor:.2f}  {dont_correct_length_with_error_prapagation
                    }   meas {measure_score_tensor.mean().item():.5f
                    }   ang:{angle_score_tensor.mean().item():.5f}   len:{length_score_tensor.mean().item():.6f
                    }   (all smaller better)")
            pass#for param
        
        print(f"dim  iter_count  lr  correction_factor  flag")
        
        assert False, "继续扫。"
        
        fds=432
        
        
        
        
        
        
        # 后面要测试的。0长度。
        # 确认小角度，且高维度。
        # 是否和某些训练冲突？？
        
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
                            __debug__ckeck_alone_the_way = True,                    dont_correct_length_with_error_prapagation = True)
        mat = _result_tuple_tl[0]
        _add_them_up = mat[0]+mat[1]
        assert _add_them_up.shape.__len__() == 1
        assert _add_them_up.shape == torch.Size([2])
        assert _tensor_equal(_add_them_up[1], [0.])
        
        
        
        
        mat = torch.tensor([[1.,0.1],[1.,-0.1]])
        _result_tuple_tt = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
        _score_before = _result_tuple_tt[0]
        _result_tuple_tl = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=1, __debug__ckeck_alone_the_way = True,dont_correct_length_with_error_prapagation = True)
        mat = _result_tuple_tl[0]
        _result_tuple_tt = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
        _score_after = _result_tuple_tt[0]
        
        
        
        
        
        
        
        
        
        
        
        
        DIM = 5
        mat = torch.randn(size=[DIM,DIM], device='cuda')
        mat = correct_the_matrix(mat,lr = 1., correction_factor = 0., iter_count=10, __debug__ckeck_alone_the_way = True,dont_correct_length_with_error_prapagation = True)#1.3
            
        fds=432
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        DIM = 100
        mat = torch.randn(size=[DIM,DIM], device='cuda')
        for _ in range(100):
            _result_tuple = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
            score_before = _result_tuple[0]
            #1w 前几个测试好像是错的。重新跑。
            #进去读一遍所有过程。
            #mat = correct_the_matrix(mat,lr = 0.03, correction_factor = 0.01, iter_count=1)
            #mat = correct_the_matrix(mat,lr = 0., correction_factor = 0.25, iter_count=1)#goes to 1.
            #mat = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=1)#goes to 1.99
            #mat = correct_the_matrix(mat,lr = 0.1, correction_factor = 0., iter_count=10)#lt 1., unstable
            #mat = correct_the_matrix(mat,lr = 0.01, correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#0.78
            #mat = correct_the_matrix(mat,lr = 0.1, correction_factor = 0., iter_count=10, dont_correct_length_with_error_prapagation = True)#0.15
            mat = correct_the_matrix(mat,lr = 1., correction_factor = 0., iter_count=10, __debug__ckeck_alone_the_way = True,dont_correct_length_with_error_prapagation = True)#1.3
            _result_tuple = LOSS__vec_len_retention__of_a_mat_in_matmul(mat)
            score_after = _result_tuple[0]
            pass
        return 
    ____test____correct_the_matrix___version_2____param_scan()
    
    
    if "also scan this in angle test.":
        assert False
    # "high dim case."
    #     dim = 2
    #     lr = 0.1
    #     mat = torch.eye(n=dim)
    #     mat[0,1] = 0.1
    #     _result_tuple_tl = correct_the_matrix___version_2(mat,lr = lr, correction_factor = 0., iter_count=1, 
    #                 dont_correct_length_with_error_prapagation = False, __debug__need_log = True)
    #     #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this flag.
    #     mat, _log= _result_tuple_tl
        
    #     fds=432
        pass
    
    
    
    pass


if "param scan" and __DEBUG_ME__() and True:
    def ____test____correct_the_matrix___version_2():
        #result:when only corrects once, the larger dim, the more tolerant.
        # If the length_factor == 0.3 is the standard,
        # dim == 3 <==> angle_factor <= 0.3
        # dim == 10 <==> angle_factor <= 1
        # dim == 100 <==> angle_factor <= 3
        # dim == 1000 <==> angle_factor don't care...
        # correction time doesn't change the relationship of result. Better is always better no matter how many times it corrects.
        
        if "dim == 3   once" and False:
            # angle_factor 0.0001      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 80 70  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # 90 80  90 90  80 v   70 v   80 70  80 90  90 90  v  90  90 70  90 90  90 v   70 70  70 80  50 70
            # angle_factor 0.0003      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 90 90  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  80  70 80  v  70  v  80  90 80  90 90  90 90  90 v   90 v   70 90  60 80  90 80  v  70  20 70
            # angle_factor 0.001      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  80  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  90  v  v   v  90  60 80  v  90  v  v   v  v   90 90  80 80  90 80  80 90  v  80  70 80  80 60
            # angle_factor 0.003      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 90 90  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  90  90 v   80 v   v  v   90 v   90 v   80 80  90 70  90 80  90 v   70 60  80 40  80 80
            # angle_factor 0.01      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 80 90  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   80 v   90 90  v  90  v  90  80 90  80 v   90 70  v  80  80 90  90 80  80 v   80 90  70 40
            # angle_factor 0.03      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  90  v  v   90 v   90 90  v  90  v  v   80 90  90 v   v  90  90 80  v  70  70 60
            # angle_factor 0.1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   90 v   80 v   v  v   v  v   80 v   v  v   v  v   v  v   v  v   90 v   90 v   90 80  80 80
            # angle_factor 0.3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            #   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #   v  v   v  v   v  v   v  v   90 v   v  v   v  v   90 v   v  v   90 v   v  v   80 80  90 v   v  v
            # angle_factor 1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            #   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  90  v  v   v  v   v  v
            #   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  90  v  v   v  v   v  v
            #   10 60  60 40  40 50  40 60  40 v   30 40  30 20  50 50  30 60  50 40  50 40  50 40  50 30  60 90
            # angle_factor 3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0 20  70 v   70 v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # 0  0   0  0   0 10  60 90  70 v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # 40  0  50  0  30  0  30  0  50  0  30  0  40  0  30  0  30 10  50 20  40 40  30 60  20 70  20 60
            # angle_factor 10      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 20  10 50  60 80  90 90  v  90  80 90  90 v
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0  10 20  40 50  80 80  v  90  80 v   90 v
            # 0  0   0  0   0  0   0  0   0  0   0  0   0 10   0  0   0  0   0  0  10 60   0 80   0 90   0 90
            # angle_factor 30      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0  10  0  50  0  10 20  60 50
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0  30  0  10 10  50 40
            # 10  0   0 10   0  0   0  0   0 10   0  0   0  0   0  0   0 10   0 10   0 80   0 90   0 v    0 v
            # angle_factor 100      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0  20  0
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 v    0 90   0 v    0 80
            # angle_factor 300      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            # 0  0   0  0   0  0   0  0  10  0  10  0   0 10   0  0   0 10   0  0   0 v    0 90   0 v    0 90
            pass
        
        if "dim == 10   once" and False:
            # angle_factor 0.0001      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  90  v  90  v  v   90 v   80 90  80 80
            # angle_factor 0.0003      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   90 v   v  90  v  90  v  v   80 80  70 50
            # angle_factor 0.001      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   90 v   v  v   90 90  80 90  80 v   90 v
            # angle_factor 0.003      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   90 v   v  90  90 v   90 70  80 80
            # angle_factor 0.01      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   90 80  70 90
            # angle_factor 0.03      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  70
            # angle_factor 0.1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 0.3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            #   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #   90 70  80 70  v  90  80 80  v  80  60 90  90 70  70 v   80 50  v  90  60 v   90 80  80 70  80 v
            # angle_factor 3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            #    0  0   0  0   0  0  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #    0  0   0  0   0  0  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 60   0 20   0 20
            # angle_factor 10      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 30  60 70  90 90  v  v   v  v   v  v   v  v
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0  90 70  v  v   v  v   v  v   v  v
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 90   0 v    0 v
            # angle_factor 30      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0  90  0  50 30  v  70
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0  40 10  30 90
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 10   0 v    0 v    0 v
            # angle_factor 100      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 40   0 v    0 v    0 v
            # angle_factor 300      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            # 0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 70   0 v    0 v    0 v
            pass
        
        if "dim == 100   once" and False:
            # for angle_factor < 0.3, all 100%
            # angle_factor 0.3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # 0  0   0  0  80  0   v  70  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # 0  0   0  0   0  0  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # 0  0   0  0   0  0  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # 0  0   0  0   0  0   0  0   0  0  v   0  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 10      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            #    0  0   0  0   0  0   0  0   0  0  v   0  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #    0  0   0  0   0  0   0  0  30  0  v  80  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #    0  0   0  0   0  0   0  0   0  0  v   0  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 30      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0  40  0  v  v   v  v   v  v   v  v   v  v   v  v
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0  90  0  v  v   v  v   v  v   v  v   v  v   v  v
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0  v  v   v  v   v  v   v  v   v   0
            # angle_factor 100      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 v   30 v   80 v   60 v
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 v   30 v   80 v   90 v
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            # angle_factor 300      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 v    0 v    0 v
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0 v    0 v    0 v
            #    0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0   0  0
            pass
        
        if "dim == 1000   once" and False:
            # angle_factor 0.0001      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 0.0003      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 0.001      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 0.003      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 0.01      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 0.03      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 0.1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 0.3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 10      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 30      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 100      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # angle_factor 300      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,  0.50,  0.60,  0.70,  0.75,  0.80]
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            # v   0  v   0  v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            pass
        
        if "dim == 100    1x, 2x, 5x   20x" and False:
            #1x
            # angle_factor 1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  
            #   v  v   v  v   v  v   v  v   v  v   v  v   
            #   v  v   v  v   v  v   v  v   v  v   v  v   
            #    0  0   0  0  90  0  v  90  v  v   v  v   
            # angle_factor 3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  
            #    0  0   0  0   0  0  v  v   v  v   v  v     
            #    0  0   0  0   0  0  v  v   v  v   v  v     
            #    0  0   0  0   0  0   0  0   0  0  v   0    
            # angle_factor 10      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  
            #    0  0   0  0   0  0   0  0   0  0  v   0    
            #    0  0   0  0   0  0   0  0  70  0  v  v     
            #    0  0   0  0   0  0   0  0   0  0  90  0    
            
            #2x
            # angle_factor 1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30, 
            #   v  v   v  v   v  v   v  v   v  v   v  v  
            #   v  v   v  v   v  v   v  v   v  v   v  v  
            #    0  0   0  0  40  0  v  v   v  v   v  v  
            # angle_factor 3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30, 
            #    0  0   0  0   0  0  v  v   v  v   v  v  
            #    0  0   0  0   0  0  v  v   v  v   v  v  
            #    0  0   0  0   0  0   0  0   0  0  v   0 
            # angle_factor 10      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30, 
            #    0  0   0  0   0  0   0  0   0  0  v   0 
            #    0  0   0  0   0  0   0  0   0  0  v   0 
            #    0  0   0  0   0  0   0  0   0  0  v   0 
            
            #5x
            # angle_factor 1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30, 
            #   v  v   v  v   v  v   v  v   v  v   v  v  
            #   v  v   v  v   v  v   v  v   v  v   v  v  
            #    0  0   0  0  40 10  v  v   v  v   v  v  
            # angle_factor 3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30, 
            #    0  0   0  0   0  0  v  v   v  v   v  v  
            #    0  0   0  0   0  0  v  v   v  v   v  v  
            #    0  0   0  0   0  0   0  0   0  0  v   0 
            # angle_factor 10      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30, 
            #    0  0   0  0   0  0   0  0   0  0  90  0 
            #    0  0   0  0   0  0   0  0   0  0  90  0 
            #    0  0   0  0   0  0   0  0   0  0  90  0 
            
            #20x
            # angle_factor 1      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35]
            #   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #   v  v   v  v   v  v   v  v   v  v   v  v   v  v
            #    0  0   0  0  50 20  v  v   v  v   v  v   v  v
            # angle_factor 3      TT.FF.. as %
            # [ 0.01,  0.05,  0.10,  0.20,  0.25,  0.30,  0.35]
            #    0  0   0  0   0  0  v  v   v  v   v  v   v  v
            #    0  0   0  0   0  0  v  v   v  v   v  v   v  v
            #    0  0   0  0   0  0   0  0   0  0  v   0  v  v
            pass
        
        
        
        # please manually set dim and test_time.
        device = 'cuda'
        dim = 100
        if dim>100:
            len_retention_test_time = 100
            pass
        else:
            len_retention_test_time = None
            pass
        test_time = 10
        correct_time = 20
        print(f"dim {dim}  test_time {test_time}  correct_time{correct_time}") 
        #for angle_fct in [0.0001,0.0003, 0.001,0.003, 0.01,0.03, 0.1,0.3, 1,3, 10,30, 100,300]:
        for angle_fct in [1,3, ]:
            length_retention_loss__result = []
            length_loss__result = []
            angle_loss__result = []
            len_fct_list = [0.01,0.05,0.1, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5, 0.6, 0.7,0.75, 0.8]
            len_fct_list = [0.01,0.05,0.1, 0.2,0.25, 0.3,0.35]
            for len_fct in len_fct_list:
                for flag in [True, False]:
                    retention_loss__USEFUL_times = 0
                    retention_loss__better_times = 0
                    length_loss__better_times = 0
                    angle_loss__better_times = 0
                    for test_count in range(test_time):
                        #<  init/>
                        mat_before = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)*10.
                        #<  before/>
                        
                        before__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat_before, test_time=len_retention_test_time)
                        before__length_loss, before__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat_before)
                        #<  correction/>
                        mat_after = mat_before.detach().clone()
                        for correct_count in range(correct_time):
                            mat_after,_ = correct_the_matrix___version_2(mat_after,
                                length_factor= len_fct, angle_factor= angle_fct, dont_correct_length_with_error_prapagation = flag)
                            pass
                        assert mat_after.device.type == device
                        #<  after/>
                        after__length_retention_loss, _ = LOSS__vec_len_retention__of_a_mat_in_matmul(mat_after, test_time=len_retention_test_time)
                        after__length_loss, after__angle_loss, _ = LOSS__mat_is_standard_orthogonal(mat_after)
                        #<  measure>
                        if before__length_retention_loss.isnan() or after__length_retention_loss.isnan():
                            # do nothing here.
                            pass
                        elif after__length_retention_loss<before__length_retention_loss:
                            retention_loss__USEFUL_times +=1
                            retention_loss__better_times +=1
                            pass
                        else:
                            retention_loss__USEFUL_times +=1
                            pass
                        
                        if after__length_loss<before__length_loss:
                            length_loss__better_times +=1
                            pass
                        
                        if after__angle_loss<before__angle_loss:
                            angle_loss__better_times +=1
                            pass
                        #</ measure>
                        pass#for test_count
                    
                    #<  convert result from numbers into readable>
                    if retention_loss__USEFUL_times == 0:
                        length_retention_loss__result.append("---")
                        pass
                    else:
                        if retention_loss__better_times == retention_loss__USEFUL_times:
                            length_retention_loss__result.append(" v ")
                            pass
                        else:
                            length_retention_loss__result.append(f"{int(retention_loss__better_times/retention_loss__USEFUL_times*100):3}")
                            pass
                        pass
                    
                    if length_loss__better_times == test_time:
                        length_loss__result.append(" v ")
                        pass
                    else:
                        length_loss__result.append(f"{int(length_loss__better_times/test_time*100):3}")
                        pass
                        
                    if angle_loss__better_times == test_time:
                        angle_loss__result.append(" v ")
                        pass
                    else:
                        angle_loss__result.append(f"{int(angle_loss__better_times/test_time*100):3}")
                        pass
                    #</ convert result from numbers into readable>
                        
                    pass#/for flag
                length_retention_loss__result.  append(" ")
                length_loss__result.            append(" ")
                angle_loss__result.             append(" ")
                pass#/for len_fct
            print(f"angle_factor {angle_fct}      TT.FF.. as %")
            print(f"{str_the_list(len_fct_list, 2)}")
            print(" "+"".join(length_retention_loss__result))
            print(" "+"".join(length_loss__result))
            print(" "+"".join(angle_loss__result))
            
            pass#/for angle_fct
        #print(better_count, total_count)
        return 
    
    ____test____correct_the_matrix___version_2()
    pass

#参数的影响到底有多大，不能只是是否。
#参数的影响到底有多大，不能只是是否。
#参数的影响到底有多大，不能只是是否。
#参数的影响到底有多大，不能只是是否。










assert False










#     assert False,''' 在这个保护下，训练本身的有效性，就是参数往一个方向推到底能不能真的推出去，还是会被保护回来。 
#     '''
#     pass


#works well.
def does_back_prapagation_work_for__a_vector_dot_itself():
    lr = 0.1
    DIM = 5
    vec_main = torch.randn(size=[DIM],requires_grad=True)
    gramo_main = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False)
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
    gramo_1 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False)
    vec_2 = torch.randn(size=[DIM],requires_grad=True)
    gramo2 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False)
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
    gramo_main = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False)
    vec_1 = torch.randn(size=[DIM],requires_grad=True)
    gramo_1 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False)
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
        