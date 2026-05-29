import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, iota
#from pytorch_yagaodirac_v2.Random import random_rotation_matrix,angle_to_rotation_matrix_2d
from pytorch_yagaodirac_v2.Random import random_symmetric_matrix, angle_to_rotation_matrix_2d
import random, time, math
import matplotlib.pyplot as plt





if "transpose" and False:
    for _ in range(432432):
        
        #<  the background
        plt.scatter([3.28,-3.28], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([2.,-2.], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([1.,-1.], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [2.5,-2.5],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [2.,-2.],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [1.,-1.],marker="x",color="#6c7308", s=5)
        plt.scatter([0],                       [0],marker="x",color="#1E1E1E")
        
        mat_1 = torch.randn(size=[2,2])
        
        #<  original
        plt.scatter(mat_1[0,0], mat_1[1,0], marker="o", color="#e20000", s=75)
        plt.scatter(mat_1[0,1], mat_1[1,1], marker="^", color="#e20000", s=75)
        
        plt.scatter(mat_1.T[0,0], mat_1.T[1,0], marker="o", label="T", color="#00ce1f", s=45)
        plt.scatter(mat_1.T[0,1], mat_1.T[1,1], marker="^",            color="#00ce1f", s=45)
        
        for ii in range(21):
            ii-=10
            plt.scatter(ii/5., ii/5., marker="o", color="#9dd100ff", s=1)
            pass
        
        plt.legend()
        plt.show()
        pass
    pass#/ test

if "inverse" and True:
    for _ in range(432432):
        
        #<  the background
        plt.scatter([3.28,-3.28], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([2.,-2.], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([1.,-1.], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [2.5,-2.5],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [2.,-2.],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [1.,-1.],marker="x",color="#6c7308", s=5)
        plt.scatter([0],                       [0],marker="x",color="#1E1E1E")
        
        while True:
            mat = torch.randn(size=[2,2])
            _temp_det = mat.det().abs()
            _temp_det_sqrt = _temp_det.sqrt()
            mat /= _temp_det_sqrt
            #del _temp_det, _temp_det_sqrt
            if not mat.isnan().any():
                break
            pass
        inv_of_mat = torch.linalg.solve(mat, torch.eye(n=2))
        assert _tensor_equal(mat@inv_of_mat, torch.eye(n=2))
        
        #this only works in 2d.
        assert _tensor_equal((mat[:,0] - mat[:,1]).dot(inv_of_mat.T[:,0]*-1 - inv_of_mat.T[:,1]), [0.])
        
        #<  original
        plt.scatter(mat[0,0], mat[1,0], marker="o", color="#00000020", s=150)
        plt.scatter(mat[0,0], mat[1,0], marker="o", color="#e20000", s=65)
        plt.scatter(mat[0,1], mat[1,1], marker="o", color="#00000020", s=120)
        plt.scatter(mat[0,1], mat[1,1], marker="^", color="#e20000", s=75)
        
        #<  inv
        plt.scatter(inv_of_mat[0,0], inv_of_mat[1,0], marker="o", label="inv", color="#00ce1f", s=45)
        plt.scatter(inv_of_mat[0,1], inv_of_mat[1,1], marker="^",              color="#00ce1f", s=45)
        
        for ii in range(21):
            ii-=10
            plt.scatter(ii/5., ii/5., marker="o", color="#9dd100ff", s=1)
            pass
        
        #<  inv.T
        plt.scatter(inv_of_mat.T[0,0], inv_of_mat.T[1,0], marker="o", label="inv.T", color="#0089ce", s=45)
        plt.scatter(inv_of_mat.T[0,1], inv_of_mat.T[1,1], marker="o", color="#00000020", s=120)
        plt.scatter(inv_of_mat.T[0,1], inv_of_mat.T[1,1], marker="^",                color="#0089ce", s=45)
        plt.scatter(inv_of_mat.T[0,0]*-1, inv_of_mat.T[1,0]*-1, marker="o", color="#00000020", s=120)
        plt.scatter(inv_of_mat.T[0,0]*-1, inv_of_mat.T[1,0]*-1, marker="x",                color="#0089ce", s=45)
        
        plt.title("gray dots are orthogonal???")
        plt.legend()
        plt.show()
        pass
    pass#/ test




if "mat mul in col vec" and False:
    for _ in range(432432):
        
        #<  the background
        plt.scatter([3.28,-3.28], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([2.,-2.], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([1.,-1.], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [2.5,-2.5],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [2.,-2.],marker="x",color="#6c7308", s=5)
        plt.scatter([0,0], [1.,-1.],marker="x",color="#6c7308", s=5)
        plt.scatter([0],                       [0],marker="x",color="#1E1E1E")
        
        mat_1 = torch.randn(size=[2,2])
        mat_1 /= mat_1.abs().mean()*math.sqrt(2.)
        mat_2 = torch.randn(size=[2,2])
        mat_2 /= mat_2.abs().mean()*math.sqrt(2.)
        
        #<  original
        plt.scatter(mat_1[0,0], mat_1[1,0], marker="o", label="1", color="#e20000", s=55)
        plt.scatter(mat_1[0,1], mat_1[1,1], marker="^",            color="#e20000", s=55)
        
        plt.scatter(mat_2[0,0], mat_2[1,0], marker="o", label="2", color="#00ce1f", s=55)
        plt.scatter(mat_2[0,1], mat_2[1,1], marker="^",            color="#00ce1f", s=55)
        
        prod = mat_1@mat_2
        
        plt.scatter(prod[0,0], prod[1,0], marker="o", label="p", color="#4613ff", s=100)
        plt.scatter(prod[0,1], prod[1,1], marker="^",            color="#4613ff", s=100)
        
        plt.legend()
        plt.show()
        pass
    pass#/ test

if "eigen values" and True:
    for _ in range(432432):

        mat = random_symmetric_matrix(2)
        
        # if mat.abs().mean().lt(1.):
        #     mat *= 1.5
        #     pass
        
        #<  calc eigen
        eig_vals_complex, eig_vectors_complex = torch.linalg.eig(mat)
        assert eig_vals_complex.imag.abs().lt(1e-4).all()
        assert eig_vectors_complex.imag.abs().lt(1e-4).all()
        #<  are eigen real or complex
        eig_vals = eig_vals_complex.real    
        del eig_vals_complex
        eig_vectors = eig_vectors_complex.real    
        del eig_vectors_complex



        #<  the background
        plt.scatter([0,0], [2.5,-2.5],marker="x",color="#6c7308", s=5)
        plt.scatter([3.22,-3.22], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([2.5,-2.5], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([0],                       [0],marker="x",color="#1E1E1E")
        
        #<  original
        plt.scatter(mat[0,0], mat[1,0], marker="o", label="MAT", color="#e20000", s=85)
        plt.scatter(mat[0,1], mat[1,1], marker="^",              color="#e20000", s=85)
        
        #<  eigen
        plt.scatter(eig_vectors[0]*2.,  eig_vectors[1]*2.,   marker="x", label = "eig", color="black", s=100)
        plt.scatter(eig_vectors[0,0]*2, eig_vectors[1,0]*2., marker="o",                color="black", s=50)
        plt.scatter(eig_vectors[0,1]*2, eig_vectors[1,1]*2., marker="^",                color="black", s=50)

        far_est_dist = torch.tensor(0.)
        far_est_point = torch.empty(size=[])
        near_est_dist = torch.tensor(999999999.)
        near_est_point = torch.empty(size=[])
        
        for ii_angle in range(50):
            rotation_mat = angle_to_rotation_matrix_2d(ii_angle*0.02*torch.pi)
            after_rotating = mat@rotation_mat
            plt.scatter(after_rotating[0,0],     after_rotating[1,0],     marker="o", color="#6dbb00", s=1)
            plt.scatter(after_rotating[0,0]*-1., after_rotating[1,0]*-1., marker="o", color="#6dbb00", s=1)
            
            dist_sqr = after_rotating[:,0].dot(after_rotating[:,0])
            if dist_sqr>far_est_dist:
                far_est_dist = dist_sqr
                far_est_point = after_rotating[:,0]
                pass
            if dist_sqr<near_est_dist:
                near_est_dist = dist_sqr
                near_est_point = after_rotating[:,0]
                pass
            
            pass
        far_est_dist = far_est_dist.sqrt()
        near_est_dist = near_est_dist.sqrt()
        
        #<  oval info???
        _target_length_over_dist = 2.2/near_est_dist
        plt.scatter(near_est_point[0]*_target_length_over_dist, near_est_point[1]*_target_length_over_dist, \
                marker="x", label="ref", color="#e20000", s=55)
        _target_length_over_dist = 2.2/far_est_dist
        plt.scatter(far_est_point[0]*_target_length_over_dist,  far_est_point[1]*_target_length_over_dist,  \
                marker="x",              color="#e20000", s=55)
        
        plt.legend()
        plt.show()
        pass
    pass#/ test

if "eigen decomposition" and True:
    for _ in range(432432):

        mat = random_symmetric_matrix(2)
        
        if mat.abs().mean().lt(1.):
            mat *= 1.5
            pass
        

        #<  calc eigen
        eig_vals_complex, eig_vectors_complex = torch.linalg.eig(mat)
        assert eig_vals_complex.imag.abs().lt(1e-4).all()
        assert eig_vectors_complex.imag.abs().lt(1e-4).all()
        #<  are eigen real or complex
        eig_vals = eig_vals_complex.real    
        del eig_vals_complex
        eig_vectors = eig_vectors_complex.real    
        del eig_vectors_complex
        eig_vectors__inv = torch.linalg.solve(eig_vectors, torch.eye(n=2))
        assert _tensor_equal(eig_vectors, eig_vectors__inv.T)#is the inverse calc correct.



        #<  the background
        plt.scatter([0,0], [2.5,-2.5],marker="x",color="#6c7308", s=5)
        plt.scatter([3.22,-3.22], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([2.5,-2.5], [0,0],marker="x",color="#6c7308", s=5)
        plt.scatter([0],                       [0],marker="x",color="#1E1E1E")

        mat_mm_eig = mat@eig_vectors
        inv_eig_mm_mat = eig_vectors__inv@mat
        should_be_eye = eig_vectors__inv@mat@eig_vectors
        
        
        #<  original
        plt.scatter(mat[0,0], mat[1,0], marker="o", label="MAT", color="#e20000", s=85)
        plt.scatter(mat[0,1], mat[1,1], marker="^",              color="#e20000", s=85)
        
        #<  eigen
        plt.scatter(eig_vectors[0]*2.,  eig_vectors[1]*2.,   marker="x", label = "eig", color="black", s=100)
        plt.scatter(eig_vectors[0,0]*2, eig_vectors[1,0]*2., marker="o",                color="black", s=50)
        plt.scatter(eig_vectors[0,1]*2, eig_vectors[1,1]*2., marker="^",                color="black", s=50)
        
        #<  mat@eig
        if True:
            plt.scatter(mat_mm_eig[0,0], mat_mm_eig[1,0], marker="o", label="MAT_mm_EIG", color="#6dbb00", s=55)
            plt.scatter(mat_mm_eig[0,1], mat_mm_eig[1,1], marker="^",                     color="#6dbb00", s=55)
            for ii_angle in range(50):
                rotation_mat = angle_to_rotation_matrix_2d(ii_angle*0.02*torch.pi)
                after_rotating = mat@rotation_mat
                plt.scatter(after_rotating[0,0],     after_rotating[1,0],     marker="o", color="#6dbb00", s=1)
                plt.scatter(after_rotating[0,0]*-1., after_rotating[1,0]*-1., marker="o", color="#6dbb00", s=1)
                pass
            pass
        #mat is symmetric
        
        #<  eig.T@mat
        if True:
            plt.scatter(inv_eig_mm_mat[0,0], inv_eig_mm_mat[1,0], marker="o", label="INV_EIG_mm_MAT", color="#0080d0")
            plt.scatter(inv_eig_mm_mat[0,1], inv_eig_mm_mat[1,1], marker="^",                         color="#0080d0")
            for ii_angle in range(50):
                rotation_mat = angle_to_rotation_matrix_2d(ii_angle*0.02*torch.pi)
                after_rotating = rotation_mat@mat
                plt.scatter(after_rotating[0,0],     after_rotating[1,0],     marker="o", color="#0080d0", s=1)
                plt.scatter(after_rotating[0,0]*-1., after_rotating[1,0]*-1., marker="o", color="#0080d0", s=1)
                plt.scatter(after_rotating[0,1],     after_rotating[1,1],     marker="o", color="#0080d0", s=1)
                plt.scatter(after_rotating[0,1]*-1., after_rotating[1,1]*-1., marker="o", color="#0080d0", s=1)
                pass
            pass
        
        assert _tensor_equal(inv_eig_mm_mat.T, mat_mm_eig)
        # plt.scatter(inv_eig_mm_mat[0,0], inv_eig_mm_mat[0,1], marker="o", label="inv_eig_mm_mat T", color="#8f00d6")
        # plt.scatter(inv_eig_mm_mat[1,0], inv_eig_mm_mat[1,1], marker="^",                           color="#8f00d6")

        #<  eig.T@mat@eig
        #plt.scatter(should_be_eye[0,0], should_be_eye[1,0], marker="+", label="should_be_eye", color="#2f5100", s=55)
        plt.scatter(should_be_eye[0,0], should_be_eye[1,0], marker="+", color="#2f5100", s=55)
        plt.scatter(should_be_eye[0,1], should_be_eye[1,1], marker="+", color="#2f5100", s=55)

        plt.legend()
        plt.show()
        pass
    pass#/ test









print("eig_vectors")
print(eig_vectors)
print("eig_vectors__inv")
print(eig_vectors__inv)
print("mat_mm_eig")
print(mat_mm_eig)
print("inv_eig_mm_mat")
print(inv_eig_mm_mat)


fds=543



