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


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal
from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2_mean_abs_to_1

import torch








# 一些特殊的mat可能不行。
# 主对角线可能无法被优化。可能需要额外的保护。
if "it always collapses into eye" and True:
    lr = 0.3
    DIM = 3
    mat:torch.Tensor = torch.randn(size=[DIM,DIM],requires_grad=True)
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
                # _temp_len_sqr = mat[0].dot(mat[0])
                # #0.25 to 0.5 power
                # mul_me = _temp_len.pow(-0.25)
                # mat[0].mul_(mul_me)
                
                # _temp_len_sqr = mat[0].dot(mat[0])
                
                
                
                
                _temp_len_sqr = mat.mul(mat).sum(dim=1)
                #ref
                _temp_len_sqr__ref = mat[0].dot(mat[0])
                assert _temp_len_sqr[0].eq(_temp_len_sqr__ref)
                #0.25 to 0.5 power
                mul_me = _temp_len_sqr.pow(-0.25)
                mat.mul_(mul_me)
                
                _temp_len_sqr = mat.mul(mat).sum
                
                1w
                
                
                
                
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
    pass


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
        