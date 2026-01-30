import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal


assert _tensor_equal(   torch.tensor([   1.,          1,          -1,          -1    ])/
                        torch.tensor([   0.,         -0,           0,          -0    ]),
                        torch.tensor([torch.inf,  torch.inf,  -torch.inf,  -torch.inf]))

assert _tensor_equal(   torch.tensor([torch.nan, torch.inf ]).nan_to_num(123.),
                        torch.tensor([     123 , 3.4028e+38]))