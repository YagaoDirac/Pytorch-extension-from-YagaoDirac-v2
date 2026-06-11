import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, iota
#from pytorch_yagaodirac_v2.Random import random_rotation_matrix,angle_to_rotation_matrix_2d
import random
import matplotlib.pyplot as plt



''' In triangle matrix, if 1 element is modified, how do the inversed matrix change correspondingly???'''



if "dim 3 triu" and False:

    dim = 3
    #B = torch.randn(size=[dim, dim]).triu()
    #B = torch.ones(size=[dim, dim]).triu()
    B = torch.tensor([  [1,0,0],
                        [2,1,0],
                        [6,3,1]], dtype= torch.float32)
    _temp_det_of_B = B.det()
    B /= _temp_det_of_B.abs().pow(1/dim)
    print(B.det())
    assert _tensor_equal(B.det().abs(), [1.])
    print(f"B")
    print(f"{B}")
    B_inv = B.inverse()
    assert _tensor_equal(B@B_inv, torch.eye(n=dim))
    print(f"B_inv")
    print(f"{B_inv}")
    pass



if "dim 4 triu" and False:

    dim = 4
    #B = torch.randn(size=[dim, dim]).tril()
    #B = torch.ones(size=[dim, dim]).tril()
    B = torch.tensor([  [1,0,0,0],
                        [2,1,0,0],
                        [6,3,1,0],
                        [24,12,4,1]], dtype= torch.float32)
    _temp_det_of_B = B.det()
    B /= _temp_det_of_B.abs().pow(1/dim)
    print(B.det())
    assert _tensor_equal(B.det().abs(), [1.])
    print(f"B")
    print(f"{B}")
    B_inv = B.inverse()
    assert _tensor_equal(B@B_inv, torch.eye(n=dim))
    print(f"B_inv")
    print(f"{B_inv}")
    
    
    B = torch.tensor([  [1,0,0,0],
                        [2,1,0,0],
                        [7,3,1,0],
                        [0,13,4,1]], dtype= torch.float32)
    _temp_det_of_B = B.det()
    B /= _temp_det_of_B.abs().pow(1/dim)
    print(B.det())
    assert _tensor_equal(B.det().abs(), [1.])
    print(f"B")
    print(f"{B}")
    B_inv = B.inverse()
    assert _tensor_equal(B@B_inv, torch.eye(n=dim))
    print(f"B_inv")
    print(f"{B_inv}")
    
    pass



for _ in range(1):
    #<  hyper param>
    dim = 4# >=3
    iota_of_dim = iota(dim)
    #<  ori>
    mat_ori = torch.randn(size=[dim,dim]).tril()
    mat_ori[iota_of_dim,iota_of_dim] = 1.
    assert _tensor_equal(mat_ori.det(), [1.])
    inv_of_ori = mat_ori.inverse()
    
    #<  rand the position>
    y_pos = random.randint(1,dim-1)
    x_pos = random.randint(0,dim-1)
    if y_pos<=x_pos:
        y_pos = dim     - y_pos
        x_pos = (dim-1) - x_pos
        pass
    #<  new mat>
    mat = mat_ori.detach().clone()
    mat[y_pos,x_pos] = random.random()
    inv_of_new = mat.inverse()
    
    #<  assertion>
    _flag_of_eq = inv_of_ori.eq(inv_of_new)
    assert _flag_of_eq[:,x_pos+1:].all()#right side.
    #print(y_pos,x_pos)
    #print(_flag_of_eq[:,x_pos+1:])
    #print(_flag_of_eq[0:y_pos,0:x_pos+1])
    assert _flag_of_eq[0:y_pos,0:x_pos+1].all()#upper left corner.
    pass







