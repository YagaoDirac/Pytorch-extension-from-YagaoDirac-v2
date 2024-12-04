from typing import Any, List, Tuple, Optional
from enum import Enum, auto
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
from pytorch_yagaodirac_v2.Artificial_Consciousness import Artificial_Consciousness




class Env_Simple(torch.nn.Module):
    def __init__(self,options = 5,*args, **kwargs):
        super().__init__(*args, **kwargs)   
        self.options = torch.nn.Parameter(torch.tensor([options]), requires_grad=False)
        self.answer = torch.nn.Parameter(torch.randint(0,options,[1]), requires_grad=False)
        pass
    def observe(self)->Tuple[torch.tensor,torch.tensor]:
        result = torch.zeros([5],dtype=torch.float32,device=self.answer.device)
        result[self.answer] = torch.rand([1])*0.5+0.5
        return result
        #end of function.
    def act(self, action:torch.Tensor)->torch.tensor:
        argmax = action.argmax()
        result = (argmax == self.answer).to(torch.float32)*2.-1.
        self.answer.data = torch.randint_like(self.answer,0,self.options)
        return result
        #end of function.
    pass


class Player(torch.nn.Module):
    def get_shape_left(self)->int:
        result = self.crit_features + self.in_features + self.mem_features
        return result
    def get_shape_mid(self)->int:
        result = self.out_features + self.mem_features
        return result
    def get_shape_right(self)->int:
        return self.crit_features
    
    def __init__(self, options:int, *args, **kwargs):
        super().__init__(*args, **kwargs)    

        self.crit_features = 1
        self.in_features = options
        self.out_features = options
        self.mem_features = 20
        
        self.score = torch.nn.Parameter(torch.Tensor([50.]))
        
        crit_lower_limit = torch.tensor([0.5])
        crit_upper_limit = torch.tensor([10.])
        
        model_exe = MLP_from_yagaodirac(self.get_shape_left(),self.get_shape_mid(),20,True,num_layers=3)
        model_ob = MLP_from_yagaodirac(self.get_shape_mid(),self.get_shape_right(),10,True,num_layers=3)
        self.ac = Artificial_Consciousness(self.crit_features,self.in_features,self.out_features,
                self.mem_features,crit_lower_limit=crit_lower_limit,
                crit_upper_limit = crit_upper_limit, model_exe=model_exe, model_ob=model_ob,
                model_exe_training_data_len=10,model_ob_training_data_len=10,
                #init_mem=init_mem,
                epochs_to_update_model_ob=1e3,epochs_to_update_model_exe=1e3,
                lr_to_update_model_ob=1e-3,lr_to_update_model_exe=1e-3,
                #training_config_for_model_ob: , training_config_for_model_exe:
                )
        pass
    
    def get_crit(self)->torch.Tensor:
        return self.score*0.01
        
    pass#end of class

options = 5
is_cuda = True
env = Env_Simple(options)
pl = Player(options)
if is_cuda:
    env.cuda()
    pl.cuda()
    pass
pt = Print_Timing(max_gap=10)
ref_score = pl.score.item()
for iter in range(100):
    #observe
    result_of_ob = env.observe()
    #print(result_of_ob, "result_of_ob")
    #think
    _crit = pl.get_crit()
    #print(_crit, "_crit")
    action = pl.ac.run(_crit, result_of_ob)
    #print(action, "action")
    print(pl.ac.output_and__mem_t_plus_1, "out and mem")
    #act
    result_of_act = env.act(action)
    print(result_of_act, "result_of_act")
    pl.score.data+=result_of_act
    ref_score-=0.6
    if pt.check(iter) or True:
        print(pl.score.item()," ac/ref ", f"{ref_score:.1f}")
        pass
    pass
print(pl.score.item())

play_noise()

