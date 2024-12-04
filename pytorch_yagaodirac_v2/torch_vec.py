#from typing import Union
import torch
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
#print(sys.path[-1])

# ori_path = sys.path[0]
# index = ori_path.rfind("\\")
# upper_folder = ori_path[:index]
# del ori_path
# del index
# del upper_folder
# #sys.path.append(os.getcwd())

class torch_vec(torch.nn.Module):
    def __init__(self, DIM:int, init_cap = 4, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.data = torch.nn.Parameter(torch.empty([init_cap,DIM], **factory_kwargs), requires_grad=False)
        self.len = torch.nn.Parameter(torch.tensor(0), requires_grad=False)
        pass
    def cap(self)->torch.Tensor:
        result = torch.tensor(self.data.shape[0], device=self.data.device)
        return result
        
    def re_cap(self, target_len:int):
        with torch.no_grad():
            new_memory = torch.empty([target_len,self.data.shape[1]],device=self.data.device, dtype= self.data.dtype)
            copy_amount = torch.minimum(torch.tensor(target_len), self.cap()) 
            new_memory[:copy_amount] = self.data[:copy_amount]
            self.data.data = new_memory
            self.len.data = torch.minimum(self.len,copy_amount)
            pass
        pass
        
    def pushback(self, data:torch.Tensor):
        with torch.no_grad():
            if self.cap() == self.len:
                self.re_cap(int(self.cap().item())*2)
                pass
            self.data.data[self.len] = data.detach()
            self.len.data+=1
            pass
        pass
    
    def get_useful(self)->torch.Tensor:
        return self.data.data[:self.len].detach()
    
    def __len__(self)->int:
        return int(self.len.item())
    def extra_repr(self):
        return f"len:{self.__len__()}, data:{self.get_useful()}"
        #return super().extra_repr()
    pass#end of class.
        
if 'basic test' and False:
    vec = torch_vec(2, init_cap=1)
    print(vec)
    vec.pushback(torch.tensor([22,33]))
    print(vec)
    vec.pushback(torch.tensor([22,44]))
    print(vec)
    vec.pushback(torch.tensor([22,55]))
    print(vec)
    vec.pushback(torch.tensor([33,66]))
    print(vec)
    
    
    