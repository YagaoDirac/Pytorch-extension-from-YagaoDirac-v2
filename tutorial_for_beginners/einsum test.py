from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _bool_equal___0_as_false, all_False, \
        _tensor_equal, _tensor_shape_check, iota, is_list_int, _float_equal
#         iota, str_the_list


import random
import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######







'''this file is a lil test on 
https://docs.pytorch.org/docs/2.15/generated/torch.einsum.html
last example at the bottom of the page.

conclusion:
on the rhs of -> symbol, all dims are forcely treated as free dimensions.
So, in the "bn,anm,bm->ba", both dim_b and dim_a are made as free dimensions.
Then, 2 dim_n disappear as a matmal. Also for dim_m.
In summary, bn is treated as a series of(with amount of b) vectors of n elements.
Or

'''







#  dim
dim_a = 3
dim_b = 2
dim_m = 4
dim_n = 5

#  data
left___b_n = torch.tensor([  [1., 0,0,0,0],
                    [0., 0,0,0,0],])
assert _tensor_shape_check(left___b_n, 2, 5)
assert _tensor_shape_check(left___b_n, dim_b, dim_n)

mid___a_n_m = torch.tensor([
    [
        [111., 112, 113, 114 ],
        [121., 122, 123, 124 ],
        [131., 132, 133, 134 ],
        [141., 142, 143, 144 ],
        [151., 152, 153, 154 ],],
    [
        [211., 212, 213, 214 ],
        [221., 222, 223, 224 ],
        [231., 232, 233, 234 ],
        [241., 242, 243, 244 ],
        [251., 252, 253, 254 ],],
    [
        [311., 312, 313, 314 ],
        [321., 322, 323, 324 ],
        [331., 332, 333, 334 ],
        [341., 342, 343, 344 ],
        [351., 352, 353, 354 ],],
                        ])
assert _tensor_shape_check(mid___a_n_m, 3, 5, 4)
assert _tensor_shape_check(mid___a_n_m, dim_a, dim_n, dim_m)

right___b_m = torch.tensor([  [1., 0,0,0],
                    [0., 0,0,0],])
assert _tensor_shape_check(right___b_m, 2, 4)
assert _tensor_shape_check(right___b_m, dim_b, dim_m)

#  calc
result___b_a = torch.einsum('bn,anm,bm->ba', left___b_n, mid___a_n_m, right___b_m)

assert _tensor_shape_check(result___b_a, 2, 3)
assert _tensor_shape_check(result___b_a, dim_b, dim_a)

print(result___b_a)

#  for loop ver
for_loop_ver_result___b_a = torch.empty(size=[dim_b, dim_a], dtype=torch.float32)
for_loop_ver_result___b_a.fill_(torch.nan)

for ii_b in range(dim_b):
    for ii_a in range(dim_a):
        left_vector___n = left___b_n[ii_b]
        assert _tensor_shape_check(left_vector___n, dim_n)
        mid_matrix___n_m = mid___a_n_m[ii_a]
        assert _tensor_shape_check(mid_matrix___n_m, dim_n, dim_m)
        right_vector___m = right___b_m[ii_b]
        assert _tensor_shape_check(right_vector___m, dim_m)

        for_loop_ver_result___b_a[ii_b, ii_a] = left_vector___n@mid_matrix___n_m@right_vector___m
        pass#for ii_a
    pass#for ii_b

assert all_False(for_loop_ver_result___b_a.isnan())

assert _tensor_equal(result___b_a, for_loop_ver_result___b_a)

1w继续，普遍情况。




fds=432








