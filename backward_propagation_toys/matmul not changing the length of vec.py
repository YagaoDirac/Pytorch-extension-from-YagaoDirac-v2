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

def length_protection_test(DIM:int, correction_factor = 0.25, 
                        correction_factor_for_pre_protect = 0.2)->tuple[torch.Tensor,torch.Tensor]:
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

def protection_hyper_param_scan():
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
protection_hyper_param_scan()










# 一些特殊的mat可能不行。
# 主对角线可能无法被优化。可能需要额外的保护。
def main_test():
    lr = 0.3
    #DIM = 3
    #mat:torch.Tensor = torch.randn(size=[DIM,DIM],requires_grad=True)
    DIM = 2
    mat:torch.Tensor = torch.tensor([[16.,16],[1,1]])
    gramo = GradientModification_v2_mean_abs_to_1()
    train_them = [mat]

    optim = torch.optim.SGD(params=train_them, lr=lr)
    loss_func = torch.nn.MSELoss()

    for epoch in range(10):
        with torch.no_grad():
            if "add noise" and False:
                mat+=torch.randn_like(mat)*lr*0.1
                pass
            
            if "correct the length manually???" and True:
                # a temp test with only 1 row
                # _temp_len_sqr = mat[0].dot(mat[0])
                # #0.25 to 0.5 power
                # mul_me = _temp_len.pow(-0.25)
                # mat[0].mul_(mul_me)
                # _temp_len_sqr = mat[0].dot(mat[0])
                
                
                
                # row?
                _temp_len_sqr = mat.mul(mat).mean(dim=1)#mul and then sum, it's a dot.
                if "check it a lil bit":
                    _temp_len_sqr__ref = mat[0].dot(mat[0]).div(DIM)
                    assert _temp_len_sqr[0].eq(_temp_len_sqr__ref)
                    pass
                #0.25 to 0.5 power
                mul_me = _temp_len_sqr.pow(-0.25)
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
                    assert _temp_len_sqr[0].eq(_temp_len_sqr__ref)
                    pass
                #0.25 to 0.5 power
                mul_me = _temp_len_sqr.pow(-0.25)
                mat.mul_(mul_me.reshape([1,-1]).expand([DIM,-1]))
                if "check it after modifying":
                    _temp_len_sqr__after_modifying = mat.mul(mat).mean(dim=0)#mul and then sum, it's a dot.
                    abs_log10__of_ori = _temp_len_sqr.log10().abs()
                    abs_log10__of_after = _temp_len_sqr__after_modifying.log10().abs()
                    better = abs_log10__of_ori.ge(abs_log10__of_after)
                    assert better.to(torch.float32).mean()>0.9
                    pass
                
                
                pass
            pass
        assert mat.requires_grad
        
        mat_gramo = gramo(mat.reshape([1,-1])).reshape([-1])
        
        should_be_eye = mat@(mat.T)
        loss = loss_func(should_be_eye, torch.eye(n=DIM))
        
        optim.zero_grad()
        loss.backward(inputs = train_them)
        
        if "print in training" and False:
            print(f"loss={loss.item():.3f},        should be 1:{mat[0].dot(mat[0]).item():.3f},{mat[1].dot(mat[1]).item():.3f},{mat[2].dot(mat[2]).item():.3f},{\
                        mat[:,0].dot(mat[:,0]).item():.3f},{mat[:,1].dot(mat[:,1]).item():.3f},{mat[:,2].dot(mat[:,2]).item():.3f}")
            print(f"                  should be 0:{mat[0].dot(mat[1]).item():.3f},{mat[1].dot(mat[2]).item():.3f},{mat[2].dot(mat[0]).item():.3f},{\
                        mat[:,0].dot(mat[:,1]).item():.3f},{mat[:,1].dot(mat[:,2]).item():.3f},{mat[:,2].dot(mat[:,0]).item():.3f}")
            print(f"             vec vec vec : {mat}")
            assert mat.grad is not None
            print(f"                               grad grad : {mat.grad[0]}")
            pass
        
        optim.step()
        pass
    
    print(f"should be 1:{mat[0].dot(mat[0]).item():.3f},{mat[1].dot(mat[1]).item():.3f},{mat[2].dot(mat[2]).item():.3f},{\
                mat[:,0].dot(mat[:,0]).item():.3f},{mat[:,1].dot(mat[:,1]).item():.3f},{mat[:,2].dot(mat[:,2]).item():.3f}")
    print(f"should be 0:{mat[0].dot(mat[1]).item():.3f},{mat[1].dot(mat[2]).item():.3f},{mat[2].dot(mat[0]).item():.3f},{\
                mat[:,0].dot(mat[:,1]).item():.3f},{mat[:,1].dot(mat[:,2]).item():.3f},{mat[:,2].dot(mat[:,0]).item():.3f}")
    print(f"             vec vec vec : {mat}")
    assert mat.grad is not None
    print(f"                               grad grad : {mat.grad[0]}")
    
    
    "验证是否保长度" 
    for _ in range(4):
        vec = torch.randn(size=[DIM])
        ori_len_sqr = vec.dot(vec)
        while ori_len_sqr<1.:
            vec = torch.randn(size=[DIM])
            ori_len_sqr = vec.dot(vec)
            pass
        
        #vec = vec.reshape(shape=[1,-1])
        after_matmul = vec@mat
        new_len_sqr = after_matmul.dot(after_matmul)
        print(f"len_before:{ori_len_sqr.item():.3f}, len_after{new_len_sqr.item():.3f}")
        #assert _float_equal(ori_len_sqr, new_len_sqr)
        pass
    return 
main_test()

#     assert False,''' 在这个保护下，训练本身的有效性，就是参数往一个方向推到底能不能真的推出去，还是会被保护回来。 
#     '''
#     pass



if "a vec dot itself, works well." and False:
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
    pass

if "2 vecs dot, also works well" and True:
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
    pass

if "1 self, 1 cross, works well" and True:
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
    pass
        