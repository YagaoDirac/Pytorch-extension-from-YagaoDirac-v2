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
        
        self.to_update_ob__crit_rb = Torch_Ring_buffer_1D(history_len,crit_features,**factory_kwargs)#+0
        #self.input_rb = Torch_Ring_buffer_1D(history_len,in_features,**factory_kwargs)#+0
        self.to_update_ob__output_t_minus_1_and_mem_rb = Torch_Ring_buffer_1D(history_len,out_features+mem_features,**factory_kwargs)#O(t-1) and M(t)
        
        self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb = Torch_Ring_buffer_1D(history_len,crit_features+in_features+mem_features,**factory_kwargs)#
        self.to_update_exe__crit_desired_rb = Torch_Ring_buffer_1D(history_len, crit_features, dtype=torch.bool, device=device)#
        self.to_update_exe__crit_desired__flag_useful_element__rb = Torch_Ring_buffer_1D(history_len, crit_features, dtype=torch.bool, device=device)#+0
        
        if init_mem is None:
            self.mem = torch.nn.Parameter(torch.zeros([self.mem_features], **factory_kwargs), requires_grad=False)
        else:
            self.mem = torch.nn.Parameter(init_mem.detach().clone(), requires_grad=False)
            pass
        self.mem_t_plus_1 = torch.nn.Parameter(torch.empty([out_features], **factory_kwargs), requires_grad=False)
        self.output__and_mem_t_plus_1:torch.nn.parameter.Parameter = torch.nn.Parameter(torch.empty([out_features+mem_features], **factory_kwargs), requires_grad=False)
        
        #debug:
        self.mem_t_plus_1.fill_(-99999)
        self.__debug__is_t_t = True
                
        self.step = 0
        
        #     保存历史
        #     训练策略，死线，单独class
        #     M的规范。
            
        pass
    
    def set_mode_executing(self):
        self.mode:AC_Mode = AC_Mode.EXECUTING
        self.model_exe.eval()
        self.model_ob.train()
        pass
    def set_mode_updating_exe_model(self):
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
    
    def get_useful_steps(self)->int:
        if self.step<self.history_len:
            return self.step
        else:
            return self.history_len
        #end of function
        
        
    
    def _make_training_data_for_model_exe(self)->torch.Tensor:
        raise Exception()
    
    def update_to_update_exe_history(self, crit_input_c:torch.Tensor)->bool:
        #T == t
        flag_crit_too_big_c = crit_input_c.ge(self.crit_upper_limit_c)
        flag_crit_too_small_c = crit_input_c.le(self.crit_lower_limit_c)
        flag_crit_out_of_limit_c:torch.Tensor = flag_crit_too_big_c.logical_or(flag_crit_too_small_c)
        
        if flag_crit_out_of_limit_c.any():
            self.to_update_exe__crit_desired__flag_useful_element__rb.pushback(flag_crit_out_of_limit_c)
            #now they are t minus 1
            self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb.pushback(self.crit_and_input_and_mem__t_minus_1)
            
            part1 = flag_crit_out_of_limit_c.logical_not()*crit_input_c
            part2 = flag_crit_too_big_c*(self.crit_lower_limit_c)
            part3 = flag_crit_too_small_c*(self.crit_upper_limit_c)
            __crit_desired = part1 + part2 + part3
            
            self.to_update_exe__crit_desired_rb.pushback(__crit_desired)
            return True
        return False
    #end of function.
    
    def run(self, crit_input_c:torch.Tensor, input_i:torch.Tensor)->torch.Tensor:
        self.__debug__is_t_t = True
        __crit_and_input_and_mem = torch.concat([crit_input_c,input_i, self.mem])
        
        if self.step>0:
            # step 1, 
            self.to_update_ob__crit_rb.pushback(crit_input_c)
            #now M is M(t)
            __output_t_minus_1_and_mem = self.output__and_mem_t_plus_1#from last step
            self.to_update_ob__output_t_minus_1_and_mem_rb.pushback(__output_t_minus_1_and_mem)
            
            # step 2 update ob
            self.set_mode_executing()
            optim_to_update_ob = torch.optim.SGD(params=self.model_ob.parameters(), lr=0.001)
            loss_function_to_update_ob = torch.nn.MSELoss()
            __length = self.to_update_ob__output_t_minus_1_and_mem_rb.length
            for epoch in range(100):
                pred = self.model_ob(self.to_update_ob__output_t_minus_1_and_mem_rb[:__length])
                loss:torch.Tensor = loss_function_to_update_ob(pred, self.to_update_ob__crit_rb[:__length])
                optim_to_update_ob.zero_grad()
                loss.backward()
                optim_to_update_ob.step()
                pass
            
            
            # step 3 update_to_update_exe_history
            if_to_update_exe = self.update_to_update_exe_history()
            
            # step 4 update exe(optional)
            if if_to_update_exe:
                self.set_mode_updating_exe_model()
                optim_to_update_exe = torch.optim.SGD(params=self.model_exe.parameters(), lr=0.001)
                loss_function_to_update_exe = torch.nn.MSELoss()
                __length = self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb.length
                
                for epoch in range(100):
                    raw_pred = self.model_ob(self.model_exe(self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb[:__length]))
                    ####### ob is fixed here. Only exe is updated.
                    pred = raw_pred*self.to_update_exe__crit_desired__flag_useful_element__rb[:__length]
                    # Tensor being multiplied by 0 stops the grad from flowing back.
                    loss:torch.Tensor = loss_function_to_update_exe(pred, self.to_update_exe__crit_desired_rb[:__length])
                    # but the loss is also not very accurate.
                    optim_to_update_exe.zero_grad()
                    loss.backward()
                    optim_to_update_exe.step()
                    pass
                pass
            pass
        
        #step 5 exe.
        with torch.inference_mode():
            self.output__and_mem_t_plus_1 = self.model_exe(__crit_and_input_and_mem)
            pass
            
        self.__debug__is_t_t = False#now it's T == t+1
        self.step+=1
        #now T == t+1
        self.crit_and_input_and_mem__t_minus_1 = __crit_and_input_and_mem#from T == t
        
        return self.output__and_mem_t_plus_1.detach()
    #end of function.
    
    pass


if "basic test." and True:
    









