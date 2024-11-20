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
from pytorch_yagaodirac_v2.torch_ring_buffer import Torch_Ring_buffer_1D_only_pushback


'''
DOCS:
The torch ring buffer is used in a non-standard way. Since data is only pushed into it, 
in the raw memory, the first len elements( the [:len]) are guaranteed to be valid.

笔记。
取消了反向查找，新方案是把两个神经网络作为一个整体，但是只训练前半截。效果应该差不多。
两个训练行为的训练数据分别整理。
小心 T == t 和 T == t+1 的区别。

2个测试model里面的数据是时间步，也就是t
'''



class AC_index_test__model_exe(torch.nn.Module):
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.tensor([111.]))
        pass
    def forward(self, input:torch.Tensor)->torch.Tensor:
        if (input[:,0]!=input[:,1]).any() or (input[:,0]!=input[:,2]).any():
            raise Exception("the design is, input is the index in real case. This dummy model output the index.")
        #input is crit_input, input, memory, for short: c i m
        #they should equal in each batch in this test.
        #say the input is [t,t,t], output is [t,t+1] for output(t) and memory(t+1)
        result = torch.empty([input.shape[0], 2], dtype=input.dtype, device=input.device)
        result[:,0] = input[:,0].detach()
        result[:,1] = input[:,0].detach()+1.
        result.requires_grad_()
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
        self.dummy_param = torch.nn.Parameter(torch.tensor([111.]))
        pass
    def forward(self, input:torch.Tensor)->torch.Tensor:
        if (input[:,0]!=(input[:,1]-1)).any():
            raise Exception("the design is, input is the index in real case. This dummy model output the index.")
        #input is output, mem(t+1), for short: o mem(t+1)
        #say the input is [t,t+1], output is [t] for crit_input(t)
        result = torch.empty([input.shape[0], 1], dtype=input.dtype, device=input.device)
        result[:,0] = input[:,1].detach()
        result.requires_grad_()
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

class Artificial_Consciousness(torch.nn.Module):
                 #first_big_number:float = 3., 
    def __init__(self, crit_features :int,in_features :int,\
            out_features :int, mem_features:int, \
                crit_lower_limit:torch.Tensor, crit_upper_limit:torch.Tensor, \
                model_exe:torch.nn.Module, model_ob:torch.nn.Module, \
                history_len = 5, \
                init_mem:Optional[torch.Tensor] = None, \
                    debug__debugging_step = False, \
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.__DEBUG__debugging_step = debug__debugging_step
        
        self.crit_features = crit_features
        self.in_features = in_features
        self.out_features = out_features
        self.mem_features = mem_features
        self.model_exe = model_exe
        self.model_ob = model_ob
        # if self.get_shape_left()!=model_exe.in_features:
        #     raise Exception()
        # if self.get_shape_mid() != model_exe.out_features:
        #     raise Exception()
        # if self.get_shape_mid() != model_ob.in_features:
        #     raise Exception()
        # if self.get_shape_right() != model_ob.out_features:
        #     raise Exception()
        
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
        
        
        self.history_len = history_len #to do:两组数据的分开。
        
        self.to_update_ob__crit_rb = Torch_Ring_buffer_1D_only_pushback(history_len,crit_features,**factory_kwargs)#+0
        self.to_update_ob__output_t_minus_1_and_mem_rb = Torch_Ring_buffer_1D_only_pushback(history_len,out_features+mem_features,**factory_kwargs)#O(t-1) and M(t)
        #self.input_rb = Torch_Ring_buffer_1D(history_len,in_features,**factory_kwargs)#+0
        
        self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb = Torch_Ring_buffer_1D_only_pushback(history_len,crit_features+in_features+mem_features,**factory_kwargs)#
        self.to_update_exe__desired_crit_rb = Torch_Ring_buffer_1D_only_pushback(history_len, crit_features, **factory_kwargs)#
        self.to_update_exe__desired_crit__flag_useful_element__rb = Torch_Ring_buffer_1D_only_pushback(history_len, crit_features, dtype=torch.bool, device=device)#+0
        
        self.output__and_mem_t_plus_1:torch.nn.parameter.Parameter = torch.nn.Parameter(torch.empty([out_features+mem_features], **factory_kwargs), requires_grad=False)
        #debug:
        #self.mem_t_plus_1.fill_(-99999)
        self.output__and_mem_t_plus_1.fill_(-99999)
        if init_mem is None:
            self.output__and_mem_t_plus_1.data[self.out_features:] = 0.
        else:
            self.output__and_mem_t_plus_1.data[self.out_features:] = init_mem.detach().clone().to(self.output__and_mem_t_plus_1.device).to(self.output__and_mem_t_plus_1.dtype)
            pass
        # 好像也不用了。self.mem_t_plus_1 = torch.nn.Parameter(torch.empty([out_features], **factory_kwargs), requires_grad=False)
                
        self.step = 0
        pass
    
    def debug_print_all_data_members(self,shape=False,limits=False,history=False,temp=False,\
                exe_training_data=False,ob_training_data=False,all=True):
        print("Artificial_Consciousness model debug print: ")
        if shape or all:
            print(f"shape(c/i/o/m):{self.crit_features}/{self.in_features}/{self.out_features}/{self.mem_features}")
            pass
        if limits or all:
            print("crit limits:",self.crit_lower_limit_c)
            print(self.crit_upper_limit_c)
            pass
        if history or all:
            print("history info: ",f"history_len: {self.history_len}, current_step: {self.step}")
            pass
        if temp or all:
            #print("M(t):",self.mem)
            #print("M(t+1):",self.mem_t_plus_1)
            print("Out(t-1) M(t):",self.output__and_mem_t_plus_1)
            pass
        if exe_training_data or all:
            print("to_update_exe: input(c/i/m): ",self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb)
            print("                         c': ", self.to_update_exe__desired_crit_rb)
            print(" flag useful elements of c': ", self.to_update_exe__desired_crit__flag_useful_element__rb)
            pass
        if ob_training_data or all:
            print("to_update_ob: Out(t-1) and M: ",self.to_update_ob__output_t_minus_1_and_mem_rb)
            print("                           c: ",self.to_update_ob__crit_rb)
            pass
        pass#end of function.
    
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
    
    def ______a():
        '''
        this func looks useless now.
    # def get_useful_steps(self)->int:
    #     if self.step<self.history_len:
    #         return self.step
    #     else:
    #         return self.history_len
        #end of function
        '''
    
    def _make_training_data_for_model_exe(self)->torch.Tensor:
        raise Exception()
    
    def update__to_update_exe_history(self, crit_input_c:torch.Tensor)->bool:
        #T == t
        flag_crit_too_big_c = crit_input_c.ge(self.crit_upper_limit_c)
        flag_crit_too_small_c = crit_input_c.le(self.crit_lower_limit_c)
        flag_crit_out_of_limit_c:torch.Tensor = flag_crit_too_big_c.logical_or(flag_crit_too_small_c)
        
        if flag_crit_out_of_limit_c.any():
            self.to_update_exe__desired_crit__flag_useful_element__rb.pushback(flag_crit_out_of_limit_c)
            #now they are t minus 1
            self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb.pushback(self.crit_and_input_and_mem__t_minus_1)
            
            part1 = flag_crit_out_of_limit_c.logical_not()*crit_input_c
            part2 = flag_crit_too_big_c*(self.crit_lower_limit_c)
            part3 = flag_crit_too_small_c*(self.crit_upper_limit_c)
            __crit_desired = part1 + part2 + part3
            
            self.to_update_exe__desired_crit_rb.pushback(__crit_desired)
            return True
        return False
    #end of function.
    
    def run(self, crit_input_c:torch.Tensor, input_i:torch.Tensor)->torch.Tensor:
        _crit_and_input_and_mem__cim = torch.concat([crit_input_c,input_i, self.output__and_mem_t_plus_1[self.out_features:]])
        
        if self.step>0:
            # step 1, 
            self.to_update_ob__crit_rb.pushback(crit_input_c)
            #now M is M(t)
            _output_t_minus_1_and_mem = self.output__and_mem_t_plus_1#from last step
            self.to_update_ob__output_t_minus_1_and_mem_rb.pushback(_output_t_minus_1_and_mem)
            
            # step 2 update ob
            self.set_mode_executing()
            
            if self.__DEBUG__debugging_step:
                _length = self.to_update_ob__output_t_minus_1_and_mem_rb.length
                pred = self.model_ob(self.to_update_ob__output_t_minus_1_and_mem_rb.data[:_length])
                print(f"step == {self.step}, the ob maps", self.to_update_ob__output_t_minus_1_and_mem_rb.data[:_length],"into",pred)
                pass
            else:
                optim_to_update_ob = torch.optim.SGD(params=self.model_ob.parameters(), lr=0.001)
                loss_function_to_update_ob = torch.nn.MSELoss()
                _length = self.to_update_ob__output_t_minus_1_and_mem_rb.length
                for epoch in range(100):
                    pred = self.model_ob(self.to_update_ob__output_t_minus_1_and_mem_rb.data[:_length])
                    loss:torch.Tensor = loss_function_to_update_ob(pred, self.to_update_ob__crit_rb.data[:_length])
                    optim_to_update_ob.zero_grad()
                    loss.backward()
                    optim_to_update_ob.step()
                    pass
                pass
            
            
            if self.__DEBUG__debugging_step:
                self.update__to_update_exe_history(crit_input_c)
                
                #now they are t minus 1
                self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb.pushback(self.crit_and_input_and_mem__t_minus_1)
                self.to_update_exe__desired_crit_rb.pushback(crit_input_c+0.5)#correct for step debug.
                self.to_update_exe__desired_crit__flag_useful_element__rb.pushback(torch.tensor([True]))
                
                _length = self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb.length
                raw_pred = self.model_ob(self.model_exe(self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb.data[:_length]))
                print(f"step == {self.step}, the exe maps", self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb.data[:_length],"into",raw_pred)
                pass
            else:
                # step 3 update__to_update_exe_history
                if_to_update_exe = self.update__to_update_exe_history(crit_input_c)
                
                # step 4 update exe(optional)
                if if_to_update_exe:
                    self.set_mode_updating_exe_model()
                    optim_to_update_exe = torch.optim.SGD(params=self.model_exe.parameters(), lr=0.001)
                    loss_function_to_update_exe = torch.nn.MSELoss()
                    _length = self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb.length
                    
                    for epoch in range(1):
                        raw_pred = self.model_ob(self.model_exe(self.to_update_exe__crit_and_input_and_mem__all_t_minus_1_rb[:_length]))
                        ####### ob is fixed here. Only exe is updated.
                        pred = raw_pred*self.to_update_exe__desired_crit__flag_useful_element__rb[:_length]
                        # Tensor being multiplied by 0 stops the grad from flowing back.
                        loss:torch.Tensor = loss_function_to_update_exe(pred, self.to_update_exe__desired_crit_rb[:_length])
                        # but the loss is also not very accurate.
                        optim_to_update_exe.zero_grad()
                        loss.backward()
                        optim_to_update_exe.step()
                        pass
                    pass
                pass
            pass
        
        #step 5 exe.
        with torch.inference_mode():
            self.output__and_mem_t_plus_1.data = self.model_exe(_crit_and_input_and_mem__cim.reshape([1,-1])).reshape([-1])
            pass
        if self.__DEBUG__debugging_step:
            print(f"step == {self.step}, the exe maps", _crit_and_input_and_mem__cim,"into",self.output__and_mem_t_plus_1.data)
            pass
        #now it's T == t+1
        self.step+=1
        #now T == t+1
        self.crit_and_input_and_mem__t_minus_1 = _crit_and_input_and_mem__cim#from T == t
        #self.mem = output__and_mem_t_plus_1??????????????继续。
        return self.output__and_mem_t_plus_1[:self.out_features].detach()
    #end of function.
    
    pass


if "index test." and True:
    crit_lower_limit = torch.tensor([-99.])
    crit_upper_limit = torch.tensor([111.])
    step = torch.tensor([0.])
    init_mem = step
    model_exe = AC_index_test__model_exe()
    model_ob = AC_index_test__model_ob()
    ac_step_test = Artificial_Consciousness(1,1,1,1,crit_lower_limit=crit_lower_limit,
            crit_upper_limit = crit_upper_limit, model_exe=model_exe, model_ob=model_ob,history_len=3,init_mem=init_mem,
            debug__debugging_step=True)
    
    ac_step_test.debug_print_all_data_members()
    
    print(ac_step_test.run(crit_input_c = step, input_i = step), "should be", step)
    ac_step_test.debug_print_all_data_members(all=False, temp=True, exe_training_data=True, ob_training_data=True)
    print()
    
    step = torch.tensor([1.])
    print(ac_step_test.run(crit_input_c = step, input_i = step), "should be", step)
    ac_step_test.debug_print_all_data_members(all=False, temp=True, exe_training_data=True, ob_training_data=True)
    print()
    
    step = torch.tensor([2.])
    print(ac_step_test.run(crit_input_c = step, input_i = step), "should be", step)
    ac_step_test.debug_print_all_data_members(all=False, temp=True, exe_training_data=True, ob_training_data=True)
    print()
    
    step = torch.tensor([3.])
    print(ac_step_test.run(crit_input_c = step, input_i = step), "should be", step)
    ac_step_test.debug_print_all_data_members(all=False, temp=True, exe_training_data=True, ob_training_data=True)
    print()
    pass







