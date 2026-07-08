import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, iota
from pytorch_yagaodirac_v2.Random import random_standard_vector, randomly_rotate__vector, randomly_permutate__vector
import random
import matplotlib.pyplot as plt



''' To answer a question, what's the probability for purely random vectors to have an angle in a range.'''

#angle_in_deg_list = torch.tensor([0., 30., 45, 60, 90])
STEPS = 18
angle_in_deg_list = torch.linspace(start= 0., end = 90., steps= 4)
angle_in_rad_list = angle_in_deg_list.div(180.).mul(torch.pi)
sin_list = angle_in_rad_list.sin()
cos_list = angle_in_rad_list.cos()
assert _tensor_equal(sin_list, [0., 0.5, 0.8660, 1.])
assert _tensor_equal(cos_list, [1., 0.8660, 0.5, 0.])
angle_in_deg_list = torch.linspace(start= 0., end = 90., steps= STEPS+1)
angle_in_rad_list = angle_in_deg_list.div(180.).mul(torch.pi)
sin_list = angle_in_rad_list.sin()
cos_list = angle_in_rad_list.cos()
assert _tensor_equal(sin_list[0],  [0.])
assert _tensor_equal(sin_list[6],  [0.5])
assert _tensor_equal(sin_list[12], [0.8660])
assert _tensor_equal(sin_list[STEPS], [1.])
assert _tensor_equal(cos_list[0],  [1.])
assert _tensor_equal(cos_list[6],  [0.8660])
assert _tensor_equal(cos_list[12], [0.5])
assert _tensor_equal(cos_list[STEPS], [0.])
x_axis = torch.linspace(start= 2.5, end = 87.5, steps= 18)
if "????????" and True:
    for dim in [3, 5, 10, 100, 1000]:

        device = 'cpu'
        _total_test_times = 1000

        _total_happen_times = torch.zeros(size=[STEPS], dtype=torch.int32)
            
        for ii in range(_total_test_times):
            vec = random_standard_vector(dim, device=device)
            vec_abs = vec.abs()

            how_many__list = []

            for ii in range(cos_list.nelement()-1):
                flags = vec_abs.le(cos_list[ii]).logical_and(vec_abs.gt(cos_list[ii+1]))
                how_many__list.append(flags.sum().item())
                pass

            _total_happen_times += torch.tensor(how_many__list)
            pass
        
        _total_happen_times = _total_happen_times.to(torch.float32)
        y_axis = _total_happen_times.div(float(_total_test_times)*dim)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_axis.tolist(),y_axis.tolist(), label="", color="blue", linewidth=2)
        plt.title(f"parallel <<<<< dim {dim} >>>>> ortho")
        plt.show()
        pass

1w 继续。

