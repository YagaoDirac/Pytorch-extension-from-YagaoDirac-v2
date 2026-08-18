import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
def _tensor_equal(  a:torch.Tensor|list[float]|list[list[float]], \
                    b:torch.Tensor|list[float]|list[list[float]], \
                        epsilon:float = 0.0001)->bool:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        pass
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
        pass
    #check the shape.
    if a.shape == torch.Size([]):
        assert b.shape == torch.Size([]) or b.shape == torch.Size([1])
        pass
    elif b.shape == torch.Size([]):#a is not Size([])
        assert a.shape == torch.Size([1])
        pass
    else:#no Size([]), a normal check.
        assert a.shape == b.shape
        pass
    
    
    with torch.inference_mode():
        diff = a-b
        abs_of_diff = diff.abs()
        less_than = abs_of_diff.lt(epsilon)
        after_all = less_than.all()
        assert after_all.dtype == torch.bool
        the_item = after_all.item()
        assert isinstance(the_item, bool)
        return the_item
    pass#end of function

# not sure.
# assert _tensor_equal(   torch.tensor([   1.,          1,          -1,          -1    ])/
#                         torch.tensor([   0.,         -0,           0,          -0    ]),
#                         torch.tensor([torch.inf,  torch.inf,  -torch.inf,  -torch.inf]))

# the epsilon...
# assert _tensor_equal(   torch.tensor([torch.nan, torch.inf ]).nan_to_num(123.),
#                         torch.tensor([     123 , 3.4028e+38]))



assert _tensor_equal(   torch.tensor([torch.nan, torch.inf, -torch.inf,         0.,   -0.0001,  0.0001 ]).log(),
                        torch.tensor([torch.nan, torch.inf,  torch.nan, -torch.inf, torch.nan, -9.2103]))


log_of_input = input.abs().log10()
