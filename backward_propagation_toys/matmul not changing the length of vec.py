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
            _log.append((_line_(),"-------------------------"))#0
            _log.append(("MATRIX   ready to correct by row", matrix.detach().clone(), _line_()))#1
            pass
        with torch.no_grad():
            #assert False, "is it mean or sum  vvv  here ????"
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
                _log.append(("_temp_len_sqr", _temp_len_sqr.detach().clone(), _line_()))#2
                _log.append(("mul_me_to_correct_length", mul_me_to_correct_length.detach().clone(), _line_()))
                _log.append(("MATRIX   Length corrected by row", matrix.detach().clone(), _line_()))#4
                _log.append((_line_(), "-------------"))#5
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
                _log.append(("should_be_eye", should_be_eye.detach().clone(), _line_()))#6
                pass
            
            if dont_correct_length_with_error_prapagation:
                should_be_eye[_iota_of_dim, _iota_of_dim] = 0.#this op also cuts the grad chain.
                loss = loss_func(should_be_eye, torch.zeros_like(matrix))
                if _log is not None:
                    _log.append(("should_be_eye after diagonal elements into 0.", should_be_eye.detach().clone(), _line_()))#7
                    pass
                #   ^^^^^   optimizable   ^^^^^
                pass
            else:
                loss = loss_func(should_be_eye, torch.eye(n=dim, device=the_device, dtype=the_dtype))
                pass
            
            optim.zero_grad()
            loss.backward(inputs = train_them)
            
            if _log is not None:
                _log.append(("loss", loss.item(), _line_()))#8
                assert matrix.grad is not None
                _log.append(("ori grad", matrix.grad.detach().clone(), _line_()))#9
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
                _log.append(("vector_length_in_matrix", vector_length_in_matrix.detach().clone(), _line_()))#10
                _log.append(("scaled grad", matrix.grad.detach().clone(), _line_()))#11
                pass
            
            
            optim.step()
            matrix.grad = None
        #</ correct the direction for row>
        
        
        
        
        
        
        
        #<  correct the length for column>
        if _log is not None:
            _log.append((_line_(),"-------------------------"))#12
            _log.append(("MATRIX   ready to correct by column", matrix.detach().clone(), _line_()))
            pass
        with torch.no_grad():
            # column?
            #assert False, "is it mean or sum  vvv  here ????"
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
                _temp_len_sqr__after_modifying = matrix.mul(matrix).sum(dim=0)#mul and then sum, it's a dot.
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


if "length correction only, finished, mar 21" and __DEBUG_ME__() and False:
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
        
        if "the step 1 len correction   for 0.3333333333333333333333333333" and False:
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
            pass
        
        if "the step 1 len correction   for 0.3333333333333333333333333333" and False:
            #<  preparation>
            _one_over_sqrt_2 = 1/math.sqrt(2.)
            mat = torch.empty(size=[2,2],dtype=torch.float64)
            mat.fill_(_one_over_sqrt_2)
            mat[0] *= 25*25*25*4*4*4
            mat[1] *= 25*25*25*3*3*3
            assert _tensor_equal(mat, [[ 25*25*25*4*4*4*0.7071, 25*25*25*4*4*4*0.7071],
                                        [25*25*25*3*3*3*0.7071, 25*25*25*3*3*3*0.7071]], epsilon=11)
            #<  calc>
            _result_tuple_tl = correct_the_matrix___version_2(mat,length_factor = 0.333333333333333333333333333333, 
                                                    angle_factor=0., iter_count=1, 
                                dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            result_mat = _result_tuple_tl[0]
            _log = _result_tuple_tl[1]
            #<  assertions>
            assert _log[2][0] == "_temp_len_sqr"
            assert _tensor_equal(_log[2][1], torch.tensor([1e12, #25*25*25*4*4*4* 25*25*25*4*4*4, 
                                                            25*25*25*3*3*3* 25*25*25*3*3*3]), epsilon=1e4)
            
            assert _log[4][0] == 'MATRIX   Length corrected by row'
            assert _tensor_equal(_log[4][1], torch.tensor([[ 25*4*0.7071, 25*4*0.7071],
                                                            [25*3*0.7071, 25*3*0.7071]]), epsilon=0.001)
            assert _log[14][0] == "_temp_len_sqr"
            assert _tensor_equal(_log[14][1], torch.tensor([25*25*25 *0.5, 25*25*25 *0.5]), epsilon=0.01)
            
            ____temp = 25*4*0.7071/(25 *math.pow(0.5,0.3333333))
            assert _tensor_equal(result_mat, torch.tensor([[ 4*0.7071/math.pow(0.5,0.333333), 4*0.7071/math.pow(0.5,0.333333)],
                                                            [3*0.7071/math.pow(0.5,0.333333), 3*0.7071/math.pow(0.5,0.333333)]]), epsilon=0.001)
            pass
        
        if "the step 1 len correction   for 0.75" and False:
            #<  preparation>
            _one_over_sqrt_2 = 1/math.sqrt(2.)
            mat = torch.empty(size=[2,2],dtype=torch.float64)
            mat.fill_(_one_over_sqrt_2)
            mat[0] /= 4.
            assert _tensor_equal(mat, [[ 0.7071/4, 0.7071/4],
                                        [0.7071,   0.7071]])
            #<  calc>
            _result_tuple_tl = correct_the_matrix___version_2(mat,length_factor = 0.75, angle_factor=0., iter_count=1, 
                                dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            _log = _result_tuple_tl[1]
            #<  assertions>
            assert _log[2][0] == "_temp_len_sqr"
            assert _tensor_equal(_log[2][1], torch.tensor([1/16., 1]))
            
            assert _log[4][0] == 'MATRIX   Length corrected by row'
            assert _tensor_equal(_log[4][1], torch.tensor([[2*0.7071, 2*0.7071],
                                                            [ 0.7071,   0.7071]]))
            pass
        
        if "reverse test with special values." and False:
            #dim = ???
            # length_factor = random.random()*0.2+0.2#0.2 to 0.4
            # if random.random()<0.2:
            #     length_factor = random.random()*0.15+0.6#0.6 to 0.75
            #     pass
            # actual_pow = (0.5-length_factor)/0.5# pow(length_sqr, this_nubmer)
            
            # random_result_mat = torch.randn(size=[dim,dim])/math.sqrt(dim)
            # random_result_mat *= random.random()*1.5+0.5# 0.5 to 2.
            
            #<  forward>
            ____sqrt_05 = math.sqrt(0.5)
            mat = torch.tensor([[16*25*2*____sqrt_05, 16*25*2*____sqrt_05],
                                [9*25*2*____sqrt_05,  9*25*2*____sqrt_05]])
            result_mat, _log = correct_the_matrix___version_2(mat.detach().clone(),length_factor = 0.25, angle_factor=0., iter_count=1, 
                                dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            #</ forward>
            #<  backward, 4>
            dim = 2
            length_factor = 0.25
            how_much_is_left = (0.5-length_factor)/0.5# pow(length_sqr, this_nubmer)
            assert how_much_is_left == 0.5
            random_result_mat = torch.tensor([[4., 4],[3, 3]])
            assert _tensor_equal(result_mat, random_result_mat)
            #<  backward, 3>
            _vec_len_in_step_2 = get_vector_length(random_result_mat.T)
            _result_length__log14 = _vec_len_in_step_2.pow(1/how_much_is_left)#1/
            assert _tensor_equal(_result_length__log14*_result_length__log14, _log[14][1], epsilon=0.001)
            assert _tensor_equal(_result_length__log14*_result_length__log14, torch.tensor([25*25, 25*25]), epsilon=0.001)
            #<  backward, 2>
            _step_2_scale_factor = _vec_len_in_step_2.pow(1/how_much_is_left -1)#1/   -1
            halfway_mat = random_result_mat*(_step_2_scale_factor.reshape([1,-1]).expand([dim,-1]))
            assert _tensor_equal(_log[4][1], halfway_mat, epsilon=0.001)
            assert _tensor_equal(halfway_mat, torch.tensor([[ 4*5, 4*5],
                                                            [3*5, 3*5]]), epsilon=0.001)
            #<  backward, 1>
            _vec_len_in_step_1 = get_vector_length(halfway_mat)
            _result_length__log2 = _vec_len_in_step_1.pow(1/how_much_is_left)#1/
            assert _tensor_equal(_result_length__log2*_result_length__log2, _log[2][1], epsilon=0.1)
            assert _tensor_equal(_result_length__log2*_result_length__log2, torch.tensor([16*16*25*25.*4, 9*9*25*25 *4]), epsilon=0.001)
            #<  backward, 0>
            _step_1_scale_factor = _vec_len_in_step_1.pow(1/how_much_is_left -1)#1/   -1
            ori_mat = halfway_mat*(_step_1_scale_factor.reshape([-1, 1]).expand([-1,dim]))
            assert _tensor_equal(mat, ori_mat, epsilon=0.1)
            assert _tensor_equal(ori_mat, [[16*25*2*0.7071, 16*25*2*0.7071],
                                            [9*25*2*0.7071,  9*25*2*0.7071]], epsilon=0.01)
            
            
            pass
        
        if "reverse test, not very good. CHeck out the style 2 below." and False:
            # result
            # a bit explaining. Bc the test is random the output first, and calcs backwardly to find out the input.
            # When the length_factor is 0.2 or 0.8,(or 0.5???), the input is corrected too much in forward pass
            # (not the forward in training). If the random generated output is a bit far from "length of 1", the 
            # calculated input is very far from "length of 1". Bc the floating point number suffers from rounding-up error,
            # the result is very weird.
            # Although from dim at least 10, this barely happens. 
            # 
            
            # if dim == 3,   
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 0.000,  0.000,  8.191,  7.071,  6.508,  4.159,  -0.131,  -37207051271474774016.000]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 15.265, 15.305, 15.630, 15.287, 15.388, 15.312,  15.623, -186035254295789568.000]
            # halfway_mat__vs__log4__min                                   =[ 0.000,  0.000,  6.627,  0.298,  0.000, -0.478,  -0.335,  -0.084]
            # halfway_mat__vs__log4__avg                                   =[ 15.103, 15.249, 15.789, 15.287, 12.457, 4.523,   10.648,  12.795]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 7.399,  7.464,  1.035,  0.173,  0.000,  0.000,   0.000,   0.000]
            # result_length__log14_mul_result_length__log14__vs__log14__avg=[ 7.980,  7.990,  7.921,  7.719,  5.831,  1.761,   4.737,   6.088]
            # result_mat__vs__random_result_mat__min                       =[ 8.119,  8.065,  1.359,  0.615, -5.633, -27.134, -5.103,  -1.774]
            # result_mat__vs__random_result_mat__avg                       =[ 8.658,  8.595,  8.425,  8.131,  5.730, -0.606,   4.644,   6.132]
            # length_factor_list                                           =[ 0.2  ,  0.25 ,  0.3  ,  0.35 ,  0.4  ,  0.6  ,   0.7  ,   0.75 ]
            # pass
            # if dim == 10,  
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 15.275,  15.231,  15.023,  14.748,  14.091,  13.309, 15.040, 14.844]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 15.668,  15.663,  15.638,  15.622,  15.609,  15.528, 15.647, 15.629]
            # halfway_mat__vs__log4__min                                   =[ 15.573,  15.637,  15.683,  15.617,  11.613,  0.000,  9.544,  13.869]
            # halfway_mat__vs__log4__avg                                   =[ 15.837,  15.864,  15.841,  15.830,  15.780,  4.487,  15.539, 15.773]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 7.799,   7.779,   7.836,   7.786,   1.845,   0.000,  0.536,  1.985]
            # result_length__log14_mul_result_length__log14__vs__log14__avg=[ 8.165,   8.181,   8.158,   8.167,   8.084,   0.998,  7.560,  8.121]
            # result_mat__vs__random_result_mat__min                       =[ 8.565,   8.363,   8.381,   8.287,   1.845,  -9.479,  0.314,  1.995]
            # result_mat__vs__random_result_mat__avg                       =[ 8.883,   8.799,   8.699,   8.641,   8.492,   0.104,  7.726,  8.270]
            # length_factor_list                                           =[ 0.2  ,   0.25 ,   0.3  ,   0.35 ,   0.4  ,   0.6  ,  0.7  ,  0.75 ]
            # pass
            # if dim == 100,   
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 15.633,  15.616,  15.627,  15.611,  15.583,  15.577,  15.611,  15.587]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 15.713,  15.719,  15.697,  15.692,  15.675,  15.680,  15.693,  15.698]
            # halfway_mat__vs__log4__min                                   =[ 15.799,  15.805,  15.801,  15.803,  15.794,  15.759,  15.740,  15.740]
            # halfway_mat__vs__log4__avg                                   =[ 15.822,  15.826,  15.820,  15.819,  15.811,  15.786,  15.771,  15.764]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 8.492,   8.525,   8.505,   8.523,   8.541,   8.520,   8.480,   8.523]
            # result_length__log14_mul_result_length__log14__vs__log14__avg=[ 8.636,   8.632,   8.629,   8.631,   8.634,   8.620,   8.631,   8.632]
            # result_mat__vs__random_result_mat__min                       =[ 9.217,   9.151,   9.050,   9.015,   8.953,   8.760,   8.666,   8.677]
            # result_mat__vs__random_result_mat__avg                       =[ 9.359,   9.258,   9.179,   9.116,   9.060,   8.872,   8.813,   8.786]
            # length_factor_list                                           =[ 0.2  ,   0.25 ,   0.3  ,   0.35 ,   0.4  ,   0.6  ,   0.7  ,   0.75 ]
            # pass
            # if dim == 1000,   
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 15.710,  15.723,  15.700,  15.683,  15.650,  15.712,  15.709,  15.688]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 15.734,  15.744,  15.723,  15.709,  15.675,  15.727,  15.724,  15.703]
            # halfway_mat__vs__log4__min                                   =[ 15.820,  15.823,  15.818,  15.814,  15.810,  15.777,  15.767,  15.757]
            # halfway_mat__vs__log4__avg                                   =[ 15.823,  15.824,  15.819,  15.817,  15.813,  15.783,  15.771,  15.764]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 9.107,   9.107,   9.093,   9.100,   9.116,   9.108,   9.099,   9.087]
            # result_length__log14_mul_result_length__log14__vs__log14__avg=[ 9.137,   9.130,   9.129,   9.125,   9.134,   9.128,   9.127,   9.129]
            # result_mat__vs__random_result_mat__min                       =[ 9.835,   9.732,   9.642,   9.585,   9.534,   9.357,   9.280,   9.237]
            # result_mat__vs__random_result_mat__avg                       =[ 9.858,   9.754,   9.675,   9.607,   9.558,   9.377,   9.306,   9.278]
            # length_factor_list                                           =[ 0.2  ,   0.25 ,   0.3  ,   0.35 ,   0.4  ,   0.6  ,   0.7  ,   0.75 ]
            # pass


            # 1w 测量一下到底差距是多少。
            # 现在系数不能超过0.5。
            
            # 是否允许超过0.5，
            # 是否
            
            # scan this extra_factor = 2.
            
            #1w 查一下这两个0是什么地方出来的。
            
            #--------------------#--------------------#--------------------
            dim_list = [3,10,100,1000,]
            test_time_list = [200,200,150,15]
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                test_time = test_time_list[outter_param_set]
                if dim>100:#tested. <100 cpu, >100 cuda.
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
            #--------------------#--------------------#--------------------
                time_start = datetime.datetime.now()
                print(test_time)
                
                result_length__log2_mul_result_length__log2__vs__log2__min   = []#don't modify here.
                result_length__log2_mul_result_length__log2__vs__log2__avg   = []#don't modify here.
                halfway_mat__vs__log4__min                                   = []#don't modify here.
                halfway_mat__vs__log4__avg                                   = []#don't modify here.
                result_length__log14_mul_result_length__log14__vs__log14__min= []#don't modify here.
                result_length__log14_mul_result_length__log14__vs__log14__avg= []#don't modify here.
                result_mat__vs__random_result_mat__min                       = []#don't modify here.
                result_mat__vs__random_result_mat__avg                       = []#don't modify here.
                
                #--------------------#--------------------#--------------------
                length_factor_list = [0.2,0.25,0.3,0.35,0.4, 0.6,0.7,0.75]
                for inner_param_set in range(length_factor_list.__len__()):
                    length_factor = length_factor_list[inner_param_set]
                #--------------------#--------------------#--------------------
                    how_much_is_left = (0.5-length_factor)/0.5# pow(length_sqr, this_nubmer)
                    
                    _raw_result_of__result_length__log2_mul_result_length__log2__vs__log2    = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__halfway_mat__vs__log4                                    = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__result_length__log14_mul_result_length__log14__vs__log14 = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__result_mat__vs__random_result_mat                        = torch.empty(size=[test_time])#don't modify here.
                    
                    for test_count in range(test_time):
                        #--------------------#--------------------#--------------------
                        random_result_mat = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)
                        # random_result_mat *= random.random()*2.3+0.2# 0.2 to 2.5 no, not needed.
                        
                        # random_result_mat *= random.random()*1.5+0.5# 0.5 to 2. to scale it a bit.
                        # random_result_mat *= extra_factor
                        
                        #<  backward, 3>
                        _vec_len_in_step_2 = get_vector_length(random_result_mat.T)
                        _result_length__log14 = _vec_len_in_step_2.pow(1/how_much_is_left)#1/
                        #<  backward, 2>
                        _step_2_scale_factor = _vec_len_in_step_2.pow(1/how_much_is_left -1)#1/   -1
                        halfway_mat = random_result_mat*(_step_2_scale_factor.reshape([1,-1]).expand([dim,-1]))
                        #<  backward, 1>
                        _vec_len_in_step_1 = get_vector_length(halfway_mat)
                        _result_length__log2 = _vec_len_in_step_1.pow(1/how_much_is_left)#1/
                        #<  backward, 0>
                        _step_1_scale_factor = _vec_len_in_step_1.pow(1/how_much_is_left -1)#1/   -1
                        ori_mat = halfway_mat*(_step_1_scale_factor.reshape([-1, 1]).expand([-1,dim]))
                        
                        #<  forward>
                        result_mat, _log = correct_the_matrix___version_2(ori_mat.detach().clone(),
                                                    length_factor = length_factor, angle_factor=0., iter_count=1, 
                                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
                        #</ forward>
                        #--------------------#--------------------#--------------------
                        if "all the prints" and False:
                            #<  assertion, 1>
                            print("---------------------------------------------------------------------")
                            print(_result_length__log2*_result_length__log2)#, _log[2][1])
                            print(_result_length__log2*_result_length__log2 - _log[2][1])
                            #<  assertion, 2>
                            print(halfway_mat)#, _log[4][1])
                            print(halfway_mat - _log[4][1])
                            #<  assertion, 3>
                            print("-------------")
                            print(_result_length__log14*_result_length__log14)#, _log[14][1])
                            print(_result_length__log14*_result_length__log14 - _log[14][1])
                            #<  assertion, 4>
                            print(result_mat)#, random_result_mat)
                            print(result_mat - random_result_mat)
                            pass
                        #<  measurement>  
                        #<  assertion, 1>
                        _result_is_valid, the_difference = log10_avg__how_similar(_result_length__log2*_result_length__log2, _log[2][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__result_length__log2_mul_result_length__log2__vs__log2[test_count] = the_difference
                            pass
                        #<  assertion, 2>
                        _result_is_valid, the_difference = log10_avg__how_similar(halfway_mat, _log[4][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__halfway_mat__vs__log4[test_count] = the_difference
                            pass
                        #<  assertion, 3>
                        _result_is_valid, the_difference = log10_avg__how_similar(_result_length__log14*_result_length__log14, _log[14][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__result_length__log14_mul_result_length__log14__vs__log14[test_count] = the_difference
                            pass
                        #<  assertion, 4>
                        _result_is_valid, the_difference = log10_avg__how_similar(result_mat, random_result_mat)
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__result_mat__vs__random_result_mat[test_count] = the_difference
                            pass
                        #</ measurement>
                        pass#for test_count
                    
                    result_length__log2_mul_result_length__log2__vs__log2__min   .append(_raw_result_of__result_length__log2_mul_result_length__log2__vs__log2.min())
                    result_length__log2_mul_result_length__log2__vs__log2__avg   .append(_raw_result_of__result_length__log2_mul_result_length__log2__vs__log2.mean())
                    halfway_mat__vs__log4__min                                   .append(_raw_result_of__halfway_mat__vs__log4.min())
                    halfway_mat__vs__log4__avg                                   .append(_raw_result_of__halfway_mat__vs__log4.mean())
                    result_length__log14_mul_result_length__log14__vs__log14__min.append(_raw_result_of__result_length__log14_mul_result_length__log14__vs__log14.min())
                    result_length__log14_mul_result_length__log14__vs__log14__avg.append(_raw_result_of__result_length__log14_mul_result_length__log14__vs__log14.mean())
                    result_mat__vs__random_result_mat__min                       .append(_raw_result_of__result_mat__vs__random_result_mat.min())
                    result_mat__vs__random_result_mat__avg                       .append(_raw_result_of__result_mat__vs__random_result_mat.mean())
                    
                    pass#for inner_param_set
                
                time_end = datetime.datetime.now()
                
                print(f"if dim == {dim},   with time {(time_end - time_start)} and {(time_end - time_start)/test_time} per test")
                print(f"result_length__log2_mul_result_length__log2__vs__log2__min   ={str_the_list(result_length__log2_mul_result_length__log2__vs__log2__min, 3)}")
                print(f"result_length__log2_mul_result_length__log2__vs__log2__avg   ={str_the_list(result_length__log2_mul_result_length__log2__vs__log2__avg, 3)}")
                print(f"halfway_mat__vs__log4__min                                   ={str_the_list(halfway_mat__vs__log4__min                                   , 3)}")
                print(f"halfway_mat__vs__log4__avg                                   ={str_the_list(halfway_mat__vs__log4__avg                                   , 3)}")
                print(f"result_length__log14_mul_result_length__log14__vs__log14__min={str_the_list(result_length__log14_mul_result_length__log14__vs__log14__min, 3)}")
                print(f"result_length__log14_mul_result_length__log14__vs__log14__avg={str_the_list(result_length__log14_mul_result_length__log14__vs__log14__avg, 3)}")
                print(f"result_mat__vs__random_result_mat__min                       ={str_the_list(result_mat__vs__random_result_mat__min                       , 3)}")
                print(f"result_mat__vs__random_result_mat__avg                       ={str_the_list(result_mat__vs__random_result_mat__avg                       , 3)}")
                print(f"length_factor_list                                           ={str_the_list(length_factor_list                       , 3)}")
                print("pass")
                
                pass# for outter_param_set
                
            if "the old assertions" and False:
                # maybe still useful.
                # #<  assertion, 1>
                # assert _tensor_equal(_result_length__log2*_result_length__log2, _log[2][1])#, epsilon=0.1)
                # #<  assertion, 2>
                # assert _tensor_equal(halfway_mat, _log[4][1])#, epsilon=0.1)
                # #<  assertion, 3>
                # assert _tensor_equal(_result_length__log14*_result_length__log14, _log[14][1], epsilon=0.005)
                # #<  assertion, 4>
                # assert _tensor_equal(result_mat, random_result_mat)
                pass
            
            pass#/ test
            
            fds=432
            
            pass
        
        if "reverse test    style 2" and False:
            # result
            # the problem in the previous test is solved... In this test, it's forward and then backward...
            # no matter what param, the difference is at least 5 orders of magnitudes smaller. 
            # Some are 7, basically the max precision of fp32.
            
            # if dim == 3,   with time 0:00:11.239046 and 0:00:00.056195 per test
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 6.525,  6.299,  6.029,  5.675,  5.502,  5.387,  5.969,  6.090]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 7.130,  6.911,  6.738,  6.540,  6.212,  6.086,  6.666,  6.752]
            # halfway_mat__vs__log4__min                                   =[ 7.090,  6.975,  6.772,  6.591,  6.534,  6.441,  6.667,  6.762]
            # halfway_mat__vs__log4__avg                                   =[ 7.552,  7.473,  7.350,  7.227,  7.028,  6.969,  7.255,  7.289]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 6.600,  6.518,  6.422,  6.184,  6.193,  6.193,  6.472,  6.581]
            # result_length__log14_mul_result_length__log14__vs__log14__avg=[ 7.225,  7.137,  7.037,  6.899,  6.724,  6.730,  7.064,  7.134]
            # ori_mat__vs__calculated_ori_mat__min                         =[ 6.868,  6.617,  6.423,  6.168,  5.840,  5.629,  6.272,  6.302]
            # ori_mat__vs__calculated_ori_mat__max                         =[ 7.359,  7.224,  7.039,  6.832,  6.510,  6.327,  6.846,  6.938]
            # length_factor_list                                           =[ 0.2  ,  0.25 ,  0.3  ,  0.35 ,  0.4  ,  0.6  ,  0.7  ,  0.75 ]
            # pass
            # 200
            # if dim == 10,   with time 0:00:23.553592 and 0:00:00.117768 per test
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 6.726,  6.646,  6.447,  6.174,  5.882,  5.738,  6.212,  6.541]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 7.070,  6.970,  6.829,  6.626,  6.307,  6.269,  6.754,  6.928]
            # halfway_mat__vs__log4__min                                   =[ 7.322,  7.179,  7.034,  6.950,  6.734,  6.649,  6.964,  6.937]
            # halfway_mat__vs__log4__avg                                   =[ 7.528,  7.444,  7.343,  7.236,  7.027,  6.982,  7.235,  7.317]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 6.901,  6.739,  6.633,  6.562,  6.395,  6.331,  6.716,  6.696]
            # result_length__log14_mul_result_length__log14__vs__log14__avg=[ 7.174,  7.084,  6.986,  6.891,  6.676,  6.700,  7.014,  7.118]
            # ori_mat__vs__calculated_ori_mat__min                         =[ 7.177,  6.975,  6.819,  6.554,  6.259,  6.034,  6.493,  6.772]
            # ori_mat__vs__calculated_ori_mat__max                         =[ 7.367,  7.271,  7.136,  6.949,  6.636,  6.505,  6.912,  7.050]
            # length_factor_list                                           =[ 0.2  ,  0.25 ,  0.3  ,  0.35 ,  0.4  ,  0.6  ,  0.7  ,  0.75 ]
            # pass
            # 150
            # if dim == 100,   with time 0:00:23.769360 and 0:00:00.158462 per test
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 7.044,  6.957,  6.831,  6.677,  6.422,  6.268,  6.827,  6.923]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 7.159,  7.066,  6.962,  6.816,  6.587,  6.557,  6.943,  7.054]
            # halfway_mat__vs__log4__min                                   =[ 7.468,  7.386,  7.268,  7.113,  6.941,  6.822,  7.066,  7.192]
            # halfway_mat__vs__log4__avg                                   =[ 7.535,  7.459,  7.354,  7.222,  7.026,  6.951,  7.206,  7.284]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 7.020,  6.963,  6.869,  6.695,  6.570,  6.532,  6.823,  6.980]
            # result_length__log14_mul_result_length__log14__vs__log14__avg=[ 7.143,  7.069,  6.972,  6.847,  6.664,  6.672,  6.973,  7.077]
            # ori_mat__vs__calculated_ori_mat__min                         =[ 7.354,  7.275,  7.166,  7.005,  6.718,  6.582,  6.987,  7.067]
            # ori_mat__vs__calculated_ori_mat__max                         =[ 7.395,  7.317,  7.209,  7.067,  6.846,  6.733,  7.044,  7.127]
            # length_factor_list                                           =[ 0.2  ,  0.25 ,  0.3  ,  0.35 ,  0.4  ,  0.6  ,  0.7  ,  0.75 ]
            # pass
            # 15
            # if dim == 1000,   with time 0:00:06.247949 and 0:00:00.416530 per test
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 7.151,  7.073,  6.985,  6.859,  6.682,  6.655,  6.979,  7.081]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 7.182,  7.095,  7.004,  6.878,  6.699,  6.687,  7.001,  7.104]
            # halfway_mat__vs__log4__min                                   =[ 7.523,  7.445,  7.346,  7.219,  7.025,  6.948,  7.205,  7.280]
            # halfway_mat__vs__log4__avg                                   =[ 7.538,  7.462,  7.356,  7.232,  7.038,  6.961,  7.223,  7.303]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 7.132,  7.064,  6.957,  6.846,  6.662,  6.662,  6.966,  7.052]
            # result_length__log14_mul_result_length__log14__vs__log14__avg=[ 7.157,  7.080,  6.975,  6.862,  6.679,  6.679,  6.990,  7.085]
            # ori_mat__vs__calculated_ori_mat__min                         =[ 7.392,  7.311,  7.219,  7.089,  6.893,  6.813,  7.087,  7.163]
            # ori_mat__vs__calculated_ori_mat__max                         =[ 7.402,  7.324,  7.225,  7.097,  6.907,  6.832,  7.095,  7.175]
            # length_factor_list                                           =[ 0.2  ,  0.25 ,  0.3  ,  0.35 ,  0.4  ,  0.6  ,  0.7  ,  0.75 ]
            # pass
            
            # scan this extra_factor = 2.
            
            #--------------------#--------------------#--------------------
            dim_list = [3,10,100,1000,]
            test_time_list = [200,200,150,15]
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                test_time = test_time_list[outter_param_set]
                if dim>100:#tested. <100 cpu, >100 cuda.
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
            #--------------------#--------------------#--------------------
                time_start = datetime.datetime.now()
                print(test_time)
                
                result_length__log2_mul_result_length__log2__vs__log2__min   = []#don't modify here.
                result_length__log2_mul_result_length__log2__vs__log2__avg   = []#don't modify here.
                halfway_mat__vs__log4__min                                   = []#don't modify here.
                halfway_mat__vs__log4__avg                                   = []#don't modify here.
                result_length__log14_mul_result_length__log14__vs__log14__min= []#don't modify here.
                result_length__log14_mul_result_length__log14__vs__log14__avg= []#don't modify here.
                ori_mat__vs__calculated_ori_mat__min                         = []#don't modify here.
                ori_mat__vs__calculated_ori_mat__max                         = []#don't modify here.
                
                #--------------------#--------------------#--------------------
                length_factor_list = [0.2,0.25,0.3,0.35,0.4, 0.6,0.7,0.75]
                for inner_param_set in range(length_factor_list.__len__()):
                    length_factor = length_factor_list[inner_param_set]
                #--------------------#--------------------#--------------------
                    how_much_is_left = (0.5-length_factor)/0.5# pow(length_sqr, this_nubmer)
                    
                    _raw_result_of__result_length__log2_mul_result_length__log2__vs__log2    = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__halfway_mat__vs__log4                                    = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__result_length__log14_mul_result_length__log14__vs__log14 = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__ori_mat__vs__calculated_ori_mat                       = torch.empty(size=[test_time])#don't modify here.
                    
                    for test_count in range(test_time):
                        #--------------------#--------------------#--------------------
                        ori_mat = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)
                        # ori_mat *= extra_factor
                        
                        
                        #<  forward>
                        result_mat, _log = correct_the_matrix___version_2(ori_mat.detach().clone(),
                                                    length_factor = length_factor, angle_factor=0., iter_count=1, 
                                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
                        #</ forward>
                        
                        
                        
                        #<  backward, 3>
                        _vec_len_in_step_2 = get_vector_length(result_mat.T)
                        _result_length__log14 = _vec_len_in_step_2.pow(1/how_much_is_left)#1/
                        #<  backward, 2>
                        _step_2_scale_factor = _vec_len_in_step_2.pow(1/how_much_is_left -1)#1/   -1
                        halfway_mat = result_mat*(_step_2_scale_factor.reshape([1,-1]).expand([dim,-1]))
                        #<  backward, 1>
                        _vec_len_in_step_1 = get_vector_length(halfway_mat)
                        _result_length__log2 = _vec_len_in_step_1.pow(1/how_much_is_left)#1/
                        #<  backward, 0>
                        _step_1_scale_factor = _vec_len_in_step_1.pow(1/how_much_is_left -1)#1/   -1
                        calculated_ori_mat = halfway_mat*(_step_1_scale_factor.reshape([-1, 1]).expand([-1,dim]))
                        
                        
                        #--------------------#--------------------#--------------------
                        if "all the prints" and False:
                            #<  assertion, 1>
                            print("---------------------------------------------------------------------")
                            print(_result_length__log2*_result_length__log2)#, _log[2][1])
                            print(_result_length__log2*_result_length__log2 - _log[2][1])
                            #<  assertion, 2>
                            print(halfway_mat)#, _log[4][1])
                            print(halfway_mat - _log[4][1])
                            #<  assertion, 3>
                            print("-------------")
                            print(_result_length__log14*_result_length__log14)#, _log[14][1])
                            print(_result_length__log14*_result_length__log14 - _log[14][1])
                            #<  assertion, 4>
                            print(result_mat)#, random_result_mat)
                            print(result_mat - random_result_mat)
                            pass
                        #<  measurement>  
                        #<  assertion, 1>
                        _result_is_valid, the_difference = log10_avg__how_similar(_result_length__log2*_result_length__log2, _log[2][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__result_length__log2_mul_result_length__log2__vs__log2[test_count] = the_difference
                            pass
                        #<  assertion, 2>
                        _result_is_valid, the_difference = log10_avg__how_similar(halfway_mat, _log[4][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__halfway_mat__vs__log4[test_count] = the_difference
                            pass
                        #<  assertion, 3>
                        _result_is_valid, the_difference = log10_avg__how_similar(_result_length__log14*_result_length__log14, _log[14][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__result_length__log14_mul_result_length__log14__vs__log14[test_count] = the_difference
                            pass
                        #<  assertion, 4>
                        _result_is_valid, the_difference = log10_avg__how_similar(ori_mat, calculated_ori_mat)
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__ori_mat__vs__calculated_ori_mat[test_count] = the_difference
                            pass
                        #</ measurement>
                        pass#for test_count
                    
                    result_length__log2_mul_result_length__log2__vs__log2__min   .append(_raw_result_of__result_length__log2_mul_result_length__log2__vs__log2.min())
                    result_length__log2_mul_result_length__log2__vs__log2__avg   .append(_raw_result_of__result_length__log2_mul_result_length__log2__vs__log2.mean())
                    halfway_mat__vs__log4__min                                   .append(_raw_result_of__halfway_mat__vs__log4.min())
                    halfway_mat__vs__log4__avg                                   .append(_raw_result_of__halfway_mat__vs__log4.mean())
                    result_length__log14_mul_result_length__log14__vs__log14__min.append(_raw_result_of__result_length__log14_mul_result_length__log14__vs__log14.min())
                    result_length__log14_mul_result_length__log14__vs__log14__avg.append(_raw_result_of__result_length__log14_mul_result_length__log14__vs__log14.mean())
                    ori_mat__vs__calculated_ori_mat__min                         .append(_raw_result_of__ori_mat__vs__calculated_ori_mat.min())
                    ori_mat__vs__calculated_ori_mat__max                         .append(_raw_result_of__ori_mat__vs__calculated_ori_mat.mean())
                    
                    pass#for inner_param_set
                
                time_end = datetime.datetime.now()
                
                print(f"if dim == {dim},   with time {(time_end - time_start)} and {(time_end - time_start)/test_time} per test")
                print(f"result_length__log2_mul_result_length__log2__vs__log2__min   ={str_the_list(result_length__log2_mul_result_length__log2__vs__log2__min, 3)}")
                print(f"result_length__log2_mul_result_length__log2__vs__log2__avg   ={str_the_list(result_length__log2_mul_result_length__log2__vs__log2__avg, 3)}")
                print(f"halfway_mat__vs__log4__min                                   ={str_the_list(halfway_mat__vs__log4__min                                   , 3)}")
                print(f"halfway_mat__vs__log4__avg                                   ={str_the_list(halfway_mat__vs__log4__avg                                   , 3)}")
                print(f"result_length__log14_mul_result_length__log14__vs__log14__min={str_the_list(result_length__log14_mul_result_length__log14__vs__log14__min, 3)}")
                print(f"result_length__log14_mul_result_length__log14__vs__log14__avg={str_the_list(result_length__log14_mul_result_length__log14__vs__log14__avg, 3)}")
                print(f"ori_mat__vs__calculated_ori_mat__min                         ={str_the_list(ori_mat__vs__calculated_ori_mat__min                       , 3)}")
                print(f"ori_mat__vs__calculated_ori_mat__max                         ={str_the_list(ori_mat__vs__calculated_ori_mat__max                       , 3)}")
                print(f"length_factor_list                                           ={str_the_list(length_factor_list                       , 3)}")
                print("pass")
                
                pass# for outter_param_set
            
            pass#/ test
        
        if "reverse test    scan the extra scaling_factor" and False:
            # result
            # in this test, only the difference from the previous "style 2" is kept.
            
            # extra_factor = 0.01
            # if dim == 3,   
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 3.425,  0.821,  0.000,  0.000,  0.000]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[ 6.246,  5.688,  0.053,  0.000,  0.000]
            # ori_mat__vs__calculated_ori_mat__min                         =[ 4.597,  2.713, -4.350, -3.330, -3.121]
            # ori_mat__vs__calculated_ori_mat__max                         =[ 6.541,  6.087, -2.235, -2.218, -2.164]
            # length_factor_list                                           =[ 0.350,  0.400,  0.600,  0.700,  0.750]
            # if dim == 10,  
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[  0.000,  0.000,  0.000]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[  0.000,  0.000,  0.000]
            # ori_mat__vs__calculated_ori_mat__min                         =[ -2.553, -2.316, -2.418]
            # ori_mat__vs__calculated_ori_mat__max                         =[ -2.100, -2.058, -2.053]
            # length_factor_list                                           =[  0.600,  0.700,  0.750]
            # if dim == 100,  
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[  0.000,  0.000,  0.000]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[  0.000,  0.000,  0.000]
            # ori_mat__vs__calculated_ori_mat__min                         =[ -2.044, -2.028, -2.030]
            # ori_mat__vs__calculated_ori_mat__max                         =[ -2.003, -2.002, -2.003]
            # length_factor_list                                           =[  0.600,  0.700,  0.750]
            # if dim == 1000,   with time 0:00:04.264423 and 0:00:00.284295 per test
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[   0.000,  0.000,  0.000]
            # result_length__log2_mul_result_length__log2__vs__log2__avg   =[   0.000,  0.000,  0.000]
            # ori_mat__vs__calculated_ori_mat__min                         =[  -2.001, -1.999, -1.999]
            # ori_mat__vs__calculated_ori_mat__max                         =[  -1.997, -1.997, -1.996]
            # length_factor_list                                           =[   0.600,  0.700,  0.750]
            
            
            # extra_factor = 0.1
            # if dim == 3
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[3.711,  3.108,  0.056]
            # ori_mat__vs__calculated_ori_mat__min                         =[3.275,  1.847, -0.723]
            # length_factor_list                                           =[0.600,  0.700,  0.750]
            # pass
            
            
            # extra_factor = 10.
            # if dim == 3
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 1.736, -0.175]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 4.906,  5.536]
            # ori_mat__vs__calculated_ori_mat__min                         =[ 1.159, -0.400]
            # length_factor_list                                           =[ 0.700,  0.750]
            
            
            # extra_factor = 100.
            # if dim == 3
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 0.567, -3.137]
            # halfway_mat__vs__log4__min                                   =[ 5.245,  3.202]
            # result_length__log14_mul_result_length__log14__vs__log14__min=[ 4.958,  3.977]
            # ori_mat__vs__calculated_ori_mat__min                         =[ 0.319, -0.032]
            # length_factor_list                                           =[ 0.700,  0.750]
            # if dim == 10
            # result_length__log2_mul_result_length__log2__vs__log2__min   =[ 1.416]
            # ori_mat__vs__calculated_ori_mat__min                         =[ 1.464]
            # length_factor_list                                           =[ 0.750]
            
            # scan this extra_factor
            extra_factor = 100.#0.01 to 100.
            
            #--------------------#--------------------#--------------------
            dim_list = [3,10,100,1000,]
            test_time_list = [200,200,150,15]
            #test_time_list = [20,20,15,3]
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                test_time = test_time_list[outter_param_set]
                if dim>100:#tested. <100 cpu, >100 cuda.
                    device = 'cuda'
                    pass
                else:
                    device = 'cpu'
                    pass
            #--------------------#--------------------#--------------------
                time_start = datetime.datetime.now()
                print(test_time)
                
                result_length__log2_mul_result_length__log2__vs__log2__min   = []#don't modify here.
                result_length__log2_mul_result_length__log2__vs__log2__avg   = []#don't modify here.
                halfway_mat__vs__log4__min                                   = []#don't modify here.
                halfway_mat__vs__log4__avg                                   = []#don't modify here.
                result_length__log14_mul_result_length__log14__vs__log14__min= []#don't modify here.
                result_length__log14_mul_result_length__log14__vs__log14__avg= []#don't modify here.
                ori_mat__vs__calculated_ori_mat__min                         = []#don't modify here.
                ori_mat__vs__calculated_ori_mat__max                         = []#don't modify here.
                
                #--------------------#--------------------#--------------------
                length_factor_list = [0.2,0.25,0.3,0.35,0.4, 0.6,0.7,0.75]
                for inner_param_set in range(length_factor_list.__len__()):
                    length_factor = length_factor_list[inner_param_set]
                #--------------------#--------------------#--------------------
                    how_much_is_left = (0.5-length_factor)/0.5# pow(length_sqr, this_nubmer)
                    
                    _raw_result_of__result_length__log2_mul_result_length__log2__vs__log2    = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__halfway_mat__vs__log4                                    = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__result_length__log14_mul_result_length__log14__vs__log14 = torch.empty(size=[test_time])#don't modify here.
                    _raw_result_of__ori_mat__vs__calculated_ori_mat                       = torch.empty(size=[test_time])#don't modify here.
                    
                    for test_count in range(test_time):
                        #--------------------#--------------------#--------------------
                        ori_mat = torch.randn(size=[dim,dim], device=device)/math.sqrt(dim)
                        ori_mat *= extra_factor
                        
                        
                        #<  forward>
                        result_mat, _log = correct_the_matrix___version_2(ori_mat.detach().clone(),
                                                    length_factor = length_factor, angle_factor=0., iter_count=1, 
                                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
                        #</ forward>
                        
                        
                        
                        #<  backward, 3>
                        _vec_len_in_step_2 = get_vector_length(result_mat.T)
                        _result_length__log14 = _vec_len_in_step_2.pow(1/how_much_is_left)#1/
                        #<  backward, 2>
                        _step_2_scale_factor = _vec_len_in_step_2.pow(1/how_much_is_left -1)#1/   -1
                        halfway_mat = result_mat*(_step_2_scale_factor.reshape([1,-1]).expand([dim,-1]))
                        #<  backward, 1>
                        _vec_len_in_step_1 = get_vector_length(halfway_mat)
                        _result_length__log2 = _vec_len_in_step_1.pow(1/how_much_is_left)#1/
                        #<  backward, 0>
                        _step_1_scale_factor = _vec_len_in_step_1.pow(1/how_much_is_left -1)#1/   -1
                        calculated_ori_mat = halfway_mat*(_step_1_scale_factor.reshape([-1, 1]).expand([-1,dim]))
                        
                        
                        #--------------------#--------------------#--------------------
                        if "all the prints" and False:
                            #<  assertion, 1>
                            print("---------------------------------------------------------------------")
                            print(_result_length__log2*_result_length__log2)#, _log[2][1])
                            print(_result_length__log2*_result_length__log2 - _log[2][1])
                            #<  assertion, 2>
                            print(halfway_mat)#, _log[4][1])
                            print(halfway_mat - _log[4][1])
                            #<  assertion, 3>
                            print("-------------")
                            print(_result_length__log14*_result_length__log14)#, _log[14][1])
                            print(_result_length__log14*_result_length__log14 - _log[14][1])
                            #<  assertion, 4>
                            print(result_mat)#, random_result_mat)
                            print(result_mat - random_result_mat)
                            pass
                        #<  measurement>  
                        #<  assertion, 1>
                        _result_is_valid, the_difference = log10_avg__how_similar(_result_length__log2*_result_length__log2, _log[2][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__result_length__log2_mul_result_length__log2__vs__log2[test_count] = the_difference
                            pass
                        #<  assertion, 2>
                        _result_is_valid, the_difference = log10_avg__how_similar(halfway_mat, _log[4][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__halfway_mat__vs__log4[test_count] = the_difference
                            pass
                        #<  assertion, 3>
                        _result_is_valid, the_difference = log10_avg__how_similar(_result_length__log14*_result_length__log14, _log[14][1])
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__result_length__log14_mul_result_length__log14__vs__log14[test_count] = the_difference
                            pass
                        #<  assertion, 4>
                        _result_is_valid, the_difference = log10_avg__how_similar(ori_mat, calculated_ori_mat)
                        if _result_is_valid:
                            assert the_difference!=0.
                            _raw_result_of__ori_mat__vs__calculated_ori_mat[test_count] = the_difference
                            pass
                        #</ measurement>
                        pass#for test_count
                    
                    result_length__log2_mul_result_length__log2__vs__log2__min   .append(_raw_result_of__result_length__log2_mul_result_length__log2__vs__log2.min())
                    result_length__log2_mul_result_length__log2__vs__log2__avg   .append(_raw_result_of__result_length__log2_mul_result_length__log2__vs__log2.mean())
                    halfway_mat__vs__log4__min                                   .append(_raw_result_of__halfway_mat__vs__log4.min())
                    halfway_mat__vs__log4__avg                                   .append(_raw_result_of__halfway_mat__vs__log4.mean())
                    result_length__log14_mul_result_length__log14__vs__log14__min.append(_raw_result_of__result_length__log14_mul_result_length__log14__vs__log14.min())
                    result_length__log14_mul_result_length__log14__vs__log14__avg.append(_raw_result_of__result_length__log14_mul_result_length__log14__vs__log14.mean())
                    ori_mat__vs__calculated_ori_mat__min                         .append(_raw_result_of__ori_mat__vs__calculated_ori_mat.min())
                    ori_mat__vs__calculated_ori_mat__max                         .append(_raw_result_of__ori_mat__vs__calculated_ori_mat.mean())
                    
                    pass#for inner_param_set
                
                time_end = datetime.datetime.now()
                
                print(f"if dim == {dim}")#,   with time {(time_end - time_start)} and {(time_end - time_start)/test_time} per test")
                print(f"result_length__log2_mul_result_length__log2__vs__log2__min   ={str_the_list(result_length__log2_mul_result_length__log2__vs__log2__min, 3)}")
                print(f"result_length__log2_mul_result_length__log2__vs__log2__avg   ={str_the_list(result_length__log2_mul_result_length__log2__vs__log2__avg, 3)}")
                print(f"halfway_mat__vs__log4__min                                   ={str_the_list(halfway_mat__vs__log4__min                                   , 3)}")
                print(f"halfway_mat__vs__log4__avg                                   ={str_the_list(halfway_mat__vs__log4__avg                                   , 3)}")
                print(f"result_length__log14_mul_result_length__log14__vs__log14__min={str_the_list(result_length__log14_mul_result_length__log14__vs__log14__min, 3)}")
                print(f"result_length__log14_mul_result_length__log14__vs__log14__avg={str_the_list(result_length__log14_mul_result_length__log14__vs__log14__avg, 3)}")
                print(f"ori_mat__vs__calculated_ori_mat__min                         ={str_the_list(ori_mat__vs__calculated_ori_mat__min                       , 3)}")
                print(f"ori_mat__vs__calculated_ori_mat__max                         ={str_the_list(ori_mat__vs__calculated_ori_mat__max                       , 3)}")
                print(f"length_factor_list                                           ={str_the_list(length_factor_list                       , 3)}")
                print("pass")
                
                pass# for outter_param_set
            
            pass#/ test
        
        
        
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
    
    
    特定角度的2维的会被撇到一个特定的新角度？？会不会不是精确的。（只关注一个方向的矫正）
    高维的有没有办法做上面这一条的相同的操作？？
    
    
    
    def ____test____correct_the_matrix___version_2____angle_correction():
        import random, math
        
        if "zero vector" and True:
            mat = torch.tensor([[1.,0], [0,0]])
            angle_factor = 0.1
            _, _log = correct_the_matrix___version_2(mat_1.detach().clone(), length_factor=0., 
                                                    angle_factor = angle_factor, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            _log[4][1]#only correct by row
            what# is the result?
            
            mat = torch.tensor([[1.,1], [0,0]])
            angle_factor = 0.1
            _, _log = correct_the_matrix___version_2(mat_1.detach().clone(), length_factor=0., 
                                                    angle_factor = angle_factor, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            _log[4][1]#only correct by row
            what# is the result?
            
            mat = torch.tensor([[1.,2], [0,0]])
            angle_factor = 0.1
            _, _log = correct_the_matrix___version_2(mat_1.detach().clone(), length_factor=0., 
                                                    angle_factor = angle_factor, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            _log[4][1]#only correct by row
            what# is the result?
            
            pass#/ test
        
        if "zero vector?  random a bit." and True:
            mat = torch.randn(size=[2,2])
            mat[1].fill_(0.)
            angle_factor = random.random()*0.1
            
            _, _log = correct_the_matrix___version_2(mat_1.detach().clone(), length_factor=0., 
                                                    angle_factor = angle_factor, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            what# is the result?
            _log[4][1]#only correct by row
            what# is the result?
            
            pass#/ test
        
        if "length should not affect the result." and True:
            for _ in range(55):
                dim = random.randint(2,100)
                mat_1 = torch.randn(size=[dim,dim])
                scaling_factor = random.random()*2.+0.5
                mat_2 = mat_1.detach().clone()*scaling_factor
                #<  calc/>            
                angle_factor = random.random()*0.1
                mat_1_after, _ = correct_the_matrix___version_2(mat_1.detach().clone(), length_factor=0., 
                                                    angle_factor = angle_factor, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
                mat_2_after, _ = correct_the_matrix___version_2(mat_2.detach().clone(), length_factor=0., 
                                                    angle_factor = angle_factor, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
                #<  assertion>
                assert _tensor_equal(mat_1_after, mat_2_after/scaling_factor)
                pass
            
            pass#/ test
        
        if "[[1,0],[0,1]] is orthogonal, nothing is touched."and True:
            mat = torch.tensor([[1.,0],[0,1]])
            #<  calc/>
            _result_tuple_tl = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., 
                                                angle_factor = random.random()*0.1, iter_count=1, 
                        dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            mat_after = _result_tuple_tl[0]
            #<  mat is not touched.>
            assert _tensor_equal(mat_after, mat)
            pass
        
        if "2d orthogonal" and True:
            for _ in range(16):
                mat = torch.rand(size=[2,2])
                mat[0,1] = -1.
                mat[1,1] = mat[0,0] * mat[1,0]
                #<  are they orthogonal?/>
                assert _tensor_equal(mat[0]*mat[1].sum(), [0.])
                assert _tensor_equal(mat[:,0]*mat[:,1].sum(), [0.])
                #<  calc/>
                _result_tuple_tl = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., 
                                                angle_factor = random.random()*0.1, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
                mat_after = _result_tuple_tl[0]
                #<  mat is not touched.>
                assert _tensor_equal(mat_after, mat)
                pass
            
            pass#/ test
        
        if "orthogonal" and True:
            for _ in range(36):
                dim = random.randint(2,100)
                mat = torch.eye(n=dim)
                mat = randomly_rotate__matrix(mat)
                mat = randomly_permutate__matrix(mat)
                #<  is orthogonal?>
                len_loss, angle_loss, _=LOSS__mat_is_standard_orthogonal(mat)
                assert _tensor_equal(len_loss, [0.])
                assert _tensor_equal(angle_loss, [0.])
                #<  calc/>
                _result_tuple_tl = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., 
                                                angle_factor = random.random()*0.1, iter_count=1, 
                            dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
                mat_after = _result_tuple_tl[0]
                #<  mat is not touched.>
                assert _tensor_equal(mat_after, mat)
                pass
            
            pass#/ test
            
        if "[[1,0],[sqrt(2)/2, sqrt(2)/2]]" and True:
            mat = torch.tensor([[1.,0],[math.sqrt(0.5),math.sqrt(0.5)]])
            #<  calc/>
            _, _log = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., 
                                                angle_factor = random.random()*0.1, iter_count=1, 
                        dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            halfway_mat = _log[4][1]
            what # is the result?
            pass#/ test
        
        if "[[1,0,0],[sqrt(2)/2, sqrt(2)/2, 0],[sqrt(2)/2, 0, sqrt(2)/2]]" and True:
            mat = torch.tensor([[1,0,0],[math.sqrt(0.5), math.sqrt(0.5), 0],
                                        [math.sqrt(0.5), 0, math.sqrt(0.5)]])
            #<  calc/>
            _, _log = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., 
                                                angle_factor = random.random()*0.1, iter_count=1, 
                        dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            halfway_mat = _log[4][1]
            what # is the result?
            pass#/ test
        
        if "random 2d 45deg" and True:
            mat = torch.randn(size=[2,2])
            _cos = torch.tensor(torch.pi/4.).cos()
            _sin = torch.tensor(torch.pi/4.).sin()
            
            mat[1] = mat[0].detach().clone()@torch.tensor([[  _cos.item(), _sin.item()],
                                                            [-_sin.item(), _cos.item()],])
            _, _log = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., 
                                                angle_factor = random.random()*0.1, iter_count=1, 
                        dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            halfway_mat = _log[4][1]
            what # is the result?
            pass#/ test
        
        if "random 3d 45deg" and True:
            mat = torch.randn(size=[3,3])
            _cos = torch.tensor(torch.pi/4.).cos()
            _sin = torch.tensor(torch.pi/4.).sin()
            
            mat[1] = mat[0].detach().clone()@torch.tensor([[  _cos.item(), _sin.item(), 0],
                                                            [-_sin.item(), _cos.item(), 0],
                                                            [0,            0,           1]])
            
            mat[2] = mat[0].detach().clone()@torch.tensor([[  _cos.item(), 0, _sin.item()],
                                                            [0,            1,           0],
                                                            [-_sin.item(), 0, _cos.item()]])

            _, _log = correct_the_matrix___version_2(mat.detach().clone(), length_factor=0., 
                                                angle_factor = random.random()*0.1, iter_count=1, 
                        dont_correct_length_with_error_prapagation = True, __debug__need_log = True)
            halfway_mat = _log[4][1]
            what # is the result?
            pass#/ test
        
        
        
            
            
            
            
            
            
        1w
        1w这个是怎么写的？？？？
            
            
        if "2d 45deg, slightly correct it." and True:
            for _ in range(5):
                #<  rand angle>
                _angle = (torch.rand([]))*0.1+0.05
                if random.random()<0.5:
                    _angle *= -1.
                    pass
                _angle += torch.pi/4.
                #</ rand angle>
                #dont_correct_length_with_error_prapagation = random.choice([True, False])
                dont_correct_length_with_error_prapagation = True
                
                #<  calc>
                mat = torch.tensor([[_angle.sin().item(),_angle.cos().item()],
                                    [1./math.sqrt(2.),   -1./math.sqrt(2.)]])#trigonometric func in rad
                1w感觉这个mat是错的。不是45度。
                
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
            
            
            
            
            
            dont_correct_length_with_error_prapagation 的测试。
            
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

__debug__ckeck_alone_the_way








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
        