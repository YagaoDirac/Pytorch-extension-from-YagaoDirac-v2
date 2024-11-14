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

from pytorch_yagaodirac_v2.Util import bitwise_acc, data_gen_for_directly_stacking_test
from pytorch_yagaodirac_v2.Util import data_gen_for_directly_stacking_test_same_dim_no_duplicated
from pytorch_yagaodirac_v2.Util import debug_strong_grad_ratio, make_grad_noisy
from pytorch_yagaodirac_v2.Util import Print_Timing
from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2
from pytorch_yagaodirac_v2.training_ended_sound import play_noise
from pytorch_yagaodirac_v2.Enhanced_MLP import FCL_from_yagaodirac, MLP_from_yagaodirac
from pytorch_yagaodirac_v2.Util import data_gen_from_random_teacher




#backward_lookup_test
class Backward_Lookup_Test_Model(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        #self.out_features = out_features
        self.main_model = MLP_from_yagaodirac(in_features,out_features,20,num_layers=3)
        pass
    def forward(self, input:torch.Tensor)->torch.Tensor:
        return self.main_model(input)
        
    def fit(self, input:torch.Tensor, target:torch.Tensor):
        loss_function = torch.nn.MSELoss()
        lr=0.0001
        optim_for_fit = torch.optim.SGD(params=self.main_model.parameters(), lr = lr)
        print(lr, "lr in fit")
        self.main_model.train()
        iter = 5
        epoch_per_iter = 5000
        
        pred = self.main_model(input)
        loss:torch.Tensor = loss_function(pred, target)
        print(0, "  epoch/loss in fit ", f"{loss.item():.4e}")
        
        for epoch in range(iter*epoch_per_iter):
            pred = self.main_model(input)
            loss:torch.Tensor = loss_function(pred, target)
            if 0 == epoch and False:
                print(pred.shape, "pred.shape")
                print(target.shape, "target.shape")
                pass
            if epoch%epoch_per_iter == epoch_per_iter-1:
                print(epoch+1, "  epoch/loss in fit ", f"{loss.item():.4e}")
                pass
            optim_for_fit.zero_grad()
            loss.backward()
            optim_for_fit.step()
            pass
        print()
        pass
            
    def backward_lookup(self, input:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        loss_function = torch.nn.MSELoss()
        lr=0.0005
        print(lr, "lr in bwlu")
        self.main_model.eval()
        iter = 5
        epoch_per_iter = 2000
        
        继续。
        
        _layer:FCL_from_yagaodirac = self.main_model.layers[0]
        dummy_input = torch.rand([target.shape[0],self.in_features],dtype = _layer.weight_o_i.dtype, device=_layer.weight_o_i.device, requires_grad=True)
        optim_for_backward_lookup = torch.optim.SGD(params=[dummy_input], lr=lr)
        loss_function = torch.nn.MSELoss()
        loss_function_for_input = torch.nn.MSELoss()
        
        loss_of_input:torch.Tensor = loss_function_for_input(dummy_input, input)
        
        loss:torch.Tensor = loss_function(self.main_model(dummy_input), target)
        print(0, "  epoch/pred loss in bwlu ", f"{loss.item():.4e}", "  /input loss", f"{loss_of_input.item():.4e}")
        
        for epoch_backward_lookup in range(iter*epoch_per_iter):
            pred = self.main_model(dummy_input)
            loss:torch.Tensor = loss_function(pred, target)
            if 0 == epoch_backward_lookup and True:
                print(pred.shape, "pred.shape")
                print(target.shape, "target.shape")
                pass
            if epoch_backward_lookup%epoch_per_iter == epoch_per_iter-1:
                loss_of_input:torch.Tensor = loss_function_for_input(dummy_input, input)
                print(epoch_backward_lookup+1, "  epoch/pred loss in bwlu ", f"{loss.item():.4e}", "  /input loss", f"{loss_of_input.item():.4e}")
                pass
                pass
            optim_for_backward_lookup.zero_grad()
            loss.backward()
            optim_for_backward_lookup.step()
            pass
        print()
        return dummy_input.detach().clone()
    pass

if 'self backward look up test' and True:
    batch = 100000
    out_features = 3
    in_features = 5
    model = Backward_Lookup_Test_Model(in_features,out_features)
    is_gpu = True
    input = torch.randn([batch,in_features])
    if is_gpu:
        model.cuda()
        input = input.cuda()
        pass
    model.eval()
    target = model(input).detach().clone()
    
    bwlu_result = model.backward_lookup(input, target)
    loss_function = torch.nn.MSELoss()
    loss_of_backward_lookup = loss_function(bwlu_result, input)
    
    print(loss_of_backward_lookup, "loss_of_backward_lookup")
    print(bwlu_result[3], "good_result")
    print(input[3], "valid_input")
    pass








if 'backward look up fake teacher test' and True:
    batch = 100000
    out_features = 3
    in_features = 5
    is_gpu = True
    fake_teacher = MLP_from_yagaodirac(in_features,out_features,100,num_layers=2)
    if is_gpu:
        fake_teacher.cuda()
        pass
    fake_teacher.eval()
    input = torch.randn([batch,in_features])
    valid_input = torch.randn([batch,in_features])
    if is_gpu:
        input = input.cuda()
        valid_input = valid_input.cuda()
        pass
    target = fake_teacher(input).detach().clone()
    valid_target = fake_teacher(valid_input).detach().clone()
    
    loss_function = torch.nn.MSELoss()
    student = Backward_Lookup_Test_Model(in_features,out_features)
    if is_gpu:
        student.cuda()
        pass
    
    #bad_result = student.backward_lookup(valid_target)
    #loss_of_backward_lookup_without_training = loss_function(bad_result, valid_input)
    
    student.fit(input, target)
    # good_result = student.backward_lookup(valid_target)
    # loss_of_backward_lookup = loss_function(good_result, valid_input)
    
    # print(loss_of_backward_lookup_without_training, "loss_of_backward_lookup_without_training")
    # print(loss_of_backward_lookup, "loss_of_backward_lookup")
    # print(good_result[3], "good_result")
    # print(valid_input[3], "valid_input")
    #print(bad_result[3], "bad_result")
    pass
    
    
        
            








class AC_Mode(Enum):
    EXECUTING = 0
    UPDATING = 0


class AC(torch.nn.Module):
                 #first_big_number:float = 3., 
    def __init__(self, crit_features :int,in_features :int,\
            out_features :int, mem_features:int, \
                crit_lower_limit:torch.Tensor, crit_upper_limit:torch.Tensor, \
                M1:torch.nn.Module, M2:torch.nn.Module, \
                history_len = 5, \
                init_mem:Optional[torch.Tensor] = None, \
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        
        self.crit_features = crit_features
        self.in_features = in_features
        self.out_features = out_features
        self.mem_features = mem_features
        self.M1 = M1
        self.M2 = M2
        if self.get_shape_left()!=M1.in_features:
            raise Exception()
        if self.get_shape_mid() != M1.out_features:
            raise Exception()
        if self.get_shape_mid() != M2.in_features:
            raise Exception()
        if self.get_shape_right() != M2.out_features:
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
        self.crit_lower_limit = crit_lower_limit
        self.crit_upper_limit = crit_upper_limit
        
        self.mode:AC_Mode = AC_Mode.EXECUTING
        self.set_mode_executing()
        
        if init_mem is None:
            self.mem = torch.nn.Parameter(torch.zeros([self.mem_features], **factory_kwargs))
        else:
            self.mem = torch.nn.Parameter(init_mem.detach().clone())
            pass
        
        self.step = 0
        
        self.history_len = history_len
        # self.history of in and out.
            
        #     制作训练数据，
        #     保存历史
        #     训练策略，死线，单独class
        #     M1重命名。
        #     M的规范。
            
        pass
    
    def set_mode_executing(self):
        self.mode:AC_Mode = AC_Mode.EXECUTING
        self.M1.eval()
        self.M2.train()
        pass
    def set_mode_executing(self):
        self.mode:AC_Mode = AC_Mode.UPDATING
        self.M1.train()
        self.M2.eval()
        pass
    def get_shape_left(self)->int:
        result = self.crit_features + self.in_features + self.mem_features
        return result
    def get_shape_mid(self)->int:
        result = self.out_features + self.mem_features
        return result
    def get_shape_right(self)->int:
        return self.crit_features
    
    
    
    def run():
        pass
    
    pass












