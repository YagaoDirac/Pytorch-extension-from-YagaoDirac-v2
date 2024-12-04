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



class Env_Analog(torch.nn.Module):
    def __init__(self,options = 5,*args, **kwargs):
        super().__init__(*args, **kwargs)   
        self.options = options
        self.model = MLP_from_yagaodirac(5,1,100,bias=False, num_layers=5)
        layer:FCL_from_yagaodirac = self.model.layers[0]
        
        for layer in self.model.layers:
            layer.weight_o_i.data*=1.5
            pass
            
        self.model.eval()
        self.input = torch.nn.Parameter(self._gen_random_input(), requires_grad=False)
        self.answer = torch.nn.Parameter(self.model(self.input), requires_grad=False)
        self.baseline = self._calc_baseline()
        pass
    def _gen_random_input(self)->torch.tensor:
        layer:FCL_from_yagaodirac = self.model.layers[0]
        result = torch.randn([1, self.options],device=layer.weight_o_i.device)
        return result
    def _update(self):
        self.input.data = self._gen_random_input()
        self.answer.data = self.model(self.input)
        pass
    
    def _calc_baseline(self)->torch.Tensor:
        accumulated = torch.zeros_like(self.answer).reshape(1)
        batch = 1000
        total_iter_count = 100
        for iter in range(total_iter_count):
            self._update()
            layer:FCL_from_yagaodirac = self.model.layers[0]
            fake_input_b_opt = torch.randn([batch,self.options],device=layer.weight_o_i.device)
            fake_action_b_o = self.model(fake_input_b_opt)
            answer_reshaped_b_o = self.answer.reshape([1,-1]).repeat([batch, 1])
            loss_function = torch.nn.L1Loss()
            loss = loss_function(fake_action_b_o, answer_reshaped_b_o)
            accumulated_loss+=loss
            #print(loss, "loss")
            pass
        
        继续
        
        
        
        
        accumulated_loss = torch.zeros_like(self.answer).reshape(1)
        batch = 1000
        total_iter_count = 100
        for iter in range(total_iter_count):
            self._update()
            layer:FCL_from_yagaodirac = self.model.layers[0]
            fake_input_b_opt = torch.randn([batch,self.options],device=layer.weight_o_i.device)
            fake_action_b_o = self.model(fake_input_b_opt)
            answer_reshaped_b_o = self.answer.reshape([1,-1]).repeat([batch, 1])
            loss_function = torch.nn.L1Loss()
            loss = loss_function(fake_action_b_o, answer_reshaped_b_o)
            accumulated_loss+=loss
            #print(loss, "loss")
            pass
        accumulated_loss/=total_iter_count
        #print(accumulated_loss, "accumulated_loss")
        return accumulated_loss
        # end of function
    
    def observe(self)->torch.tensor:
        return self.input.reshape([-1])
        #end of function.
    def act(self, action:torch.Tensor)->torch.tensor:
        diff = (self.answer.reshape([-1])-action).abs()
        result = self.baseline-diff
        self._update()
        return result
        #end of function.
    pass

if 'basic test' and True:
    options = 5
    env = Env_Analog(options)
    print(env.baseline)
    for _ in range(5):
        action = torch.tensor([0.])
        print(env.act(action))
        pass
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

