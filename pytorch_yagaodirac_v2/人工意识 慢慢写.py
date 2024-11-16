from typing import Any, List, Tuple, Optional
from enum import Enum, auto
import sys
import math
import torch

import sys
ori_path = sys.path[0]
index = ori_path.rfind("\\")
upper_folder = ori_path[:index]
sys.path.append(upper_folder)
del ori_path
del index
del upper_folder
#sys.path.append(os.getcwd())

from pytorch_yagaodirac_v2.Util import Print_Timing
from pytorch_yagaodirac_v2.training_ended_sound import play_noise
from pytorch_yagaodirac_v2.Enhanced_MLP import FCL_from_yagaodirac, MLP_from_yagaodirac
from pytorch_yagaodirac_v2.Util import data_gen_from_random_teacher
from pytorch_yagaodirac_v2.torch_ring_buffer import Torch_Ring_buffer_1D




class AC_index_test__model_exe(torch.nn.Module):
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        pass
    def forward(self, input:torch.Tensor)->torch.Tensor:
        if (input[:,0]!=input[:,1]).any() or (input[:,0]!=input[:,2]).any():
            raise Exception("the design is, input is the index in real case. This dummy model output the index.")
        #input is crit_input, input, memory, for short: c i m
        #they should equal in each batch in this test.
        #say the input is [t,t,t], output is [t,t+1] for output(t) and memory(t+1)
        result = torch.empty([input.shape[0], 2], dtype=input.dtype, device=input.device)
        result[:,0] = input[:,0].detach().clone()
        result[:,1] = input[:,0].detach()+1.
        return result
    def extra_repr(self):
        return "This is a TEST tool ONLY FOR INDEX TEST. If you see this in ANY OTHER PLACES, IT'S WRONG!!!"
    pass# end of class
if "basic test" and False:
    model = AC_index_test__model_exe()
    input = torch.tensor([[0.,0,0],[1,1,1]])
    output = model(input)
    print(output, "should be 0 1, 1 2")
    pass

class AC_index_test__model_ob(torch.nn.Module):
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        pass
    def forward(self, input:torch.Tensor)->torch.Tensor:
        if (input[:,0]!=(input[:,1]-1)).any():
            raise Exception("the design is, input is the index in real case. This dummy model output the index.")
        #input is output, mem(t+1), for short: o mem(t+1)
        #say the input is [t,t+1], output is [t] for crit_input(t)
        result = input[:,1].detach().clone()
        return result
    def extra_repr(self):
        return "This is a TEST tool ONLY FOR INDEX TEST. If you see this in ANY OTHER PLACES, IT'S WRONG!!!"
    pass# end of class
if "basic test" and False:
    model = AC_index_test__model_ob()
    input = torch.tensor([[0.,1],[1,2]])
    output = model(input)
    print(output, "should be 1, 2")
    pass




class AC_Mode(Enum):
    EXECUTING = 0
    UPDATING = 0

class AC(torch.nn.Module):
                 #first_big_number:float = 3., 
    def __init__(self, crit_features :int,in_features :int,\
            out_features :int, mem_features:int, \
                crit_lower_limit:torch.Tensor, crit_upper_limit:torch.Tensor, \
                model_exe:torch.nn.Module, model_ob:torch.nn.Module, \
                history_len = 5, \
                init_mem:Optional[torch.Tensor] = None, \
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.crit_features = crit_features
        self.in_features = in_features
        self.out_features = out_features
        self.mem_features = mem_features
        self.model_exe = model_exe
        self.model_ob = model_ob
        if self.get_shape_left()!=model_exe.in_features:
            raise Exception()
        if self.get_shape_mid() != model_exe.out_features:
            raise Exception()
        if self.get_shape_mid() != model_ob.in_features:
            raise Exception()
        if self.get_shape_right() != model_ob.out_features:
            raise Exception()
        
        if crit_lower_limit>=crit_upper_limit:
            raise Exception()
        if crit_lower_limit.shape.__len__()!=1:
            raise Exception()
        if crit_upper_limit.shape.__len__()!=1:
            raise Exception()
        if crit_features!=crit_lower_limit.shape[0]:
            raise Exception()
        if crit_features!=crit_upper_limit.shape[0]:
            raise Exception()
        self.crit_lower_limit_c = crit_lower_limit
        self.crit_upper_limit_c = crit_upper_limit
        
        self.mode:AC_Mode = AC_Mode.EXECUTING
        self.set_mode_executing()
        
        self.history_len = history_len
        
        self.crit_input_rb = Torch_Ring_buffer_1D(history_len)#+0
        self.input_rb = Torch_Ring_buffer_1D(history_len)#+0
        self.output__1_longer_rb = Torch_Ring_buffer_1D(history_len+1)#+1
        self.mem__1_longer_rb = Torch_Ring_buffer_1D(history_len+1)#+1
        
        if init_mem is None:
            self.mem__1_longer_rb.pushback(torch.zeros([self.mem_features], **factory_kwargs))
        else:
            self.mem__1_longer_rb.pushback(init_mem.detach().clone())
            pass
        
        #     保存历史
        #     训练策略，死线，单独class
        #     M的规范。
            
        pass
    
    def set_mode_executing(self):
        self.mode:AC_Mode = AC_Mode.EXECUTING
        self.model_exe.eval()
        self.model_ob.train()
        pass
    def set_mode_updating(self):
        self.mode:AC_Mode = AC_Mode.UPDATING
        self.model_exe.train()
        self.model_ob.eval()
        pass
    def get_shape_left(self)->int:
        result = self.crit_features + self.in_features + self.mem_features
        return result
    def get_shape_mid(self)->int:
        result = self.out_features + self.mem_features
        return result
    def get_shape_right(self)->int:
        return self.crit_features
    
    def _make_training_data_for_model_exe(self)->torch.Tensor:
        raise Exception()
    
    
    def run(self, crit_input:torch.Tensor, input:torch.Tensor)->torch.Tensor:
        #step1, if crit is out of limit, do the AC_Mode.UPDATING
        __flag_temp_1 = crit_input.ge(self.crit_upper_limit_c)
        __flag_temp_2 = crit_input.le(self.crit_lower_limit_c)
        flag_crit_out_of_limit = __flag_temp_1.logical_or(__flag_temp_2)
        del __flag_temp_1
        del __flag_temp_2
        
        if flag_crit_out_of_limit.any():
            self.set_mode_updating()
            figure out the desired crit
            bwlu to figure out the mid_tensor
            train model_exe
            raise Exception()
        
        
        self.set_mode_executing()
        left_tensor = torch.concat([crit_input, input, ]).reshape???????????
        mid_tensor = self.model_exe(left_tensor)
        
        save history
        
        train the model_ob

        return mid_tensor
        
        
                

        
        
        
        
        pass
    
    pass












