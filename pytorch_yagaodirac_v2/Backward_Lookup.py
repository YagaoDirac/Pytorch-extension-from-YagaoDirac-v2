from typing import Optional
import sys
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

from pytorch_yagaodirac_v2.training_ended_sound import play_noise
from pytorch_yagaodirac_v2.Enhanced_MLP import FCL_from_yagaodirac, MLP_from_yagaodirac
from pytorch_yagaodirac_v2.torch_ring_buffer import Torch_Ring_buffer_1D_only_pushback




#backward lookup able model.
#this class will be moved to somewhere else in the future.
class Backward_Lookup_able(torch.nn.Module):
    def __init__(self, model:torch.nn.Module, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #self.out_features = out_features
        
        self.main_model = model#MLP_from_yagaodirac(in_features,out_features,20,num_layers=3)
        pass
    def get_in_features(self)->int:
        return self.main_model.in_features
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        return self.main_model(input)
        
    def fit(self, input:torch.Tensor, target:torch.Tensor):
        loss_function = torch.nn.MSELoss()
        lr=0.0001
        optim_for_fit = torch.optim.SGD(params=self.main_model.parameters(), lr = lr)
        #print(lr, "lr in fit")
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
            
    def backward_lookup(self, input:torch.Tensor, target:torch.Tensor, \
                start_from:Optional[torch.Tensor] = None)->torch.Tensor:
        loss_function = torch.nn.MSELoss()
        lr=0.0005
        #print(lr, "lr in bwlu")
        self.main_model.eval()
        iter = 5
        epoch_per_iter = 2000
        
        _layer:FCL_from_yagaodirac = self.main_model.layers[0]
        _some_param:torch.nn.parameter.Parameter = _layer.weight_o_i
        if start_from is None:
            dummy_input = torch.rand([target.shape[0],self.get_in_features()],dtype = _some_param.dtype, device=_some_param.device, requires_grad=True)
        else:
            dummy_input = start_from.detach().clone()
            dummy_input.requires_grad_()
            pass
        optim_for_backward_lookup = torch.optim.SGD(params=[dummy_input], lr=lr)
        loss_function = torch.nn.MSELoss()
        loss_function_for_input = torch.nn.MSELoss()
        
        loss_of_input:torch.Tensor = loss_function_for_input(dummy_input, input)
        
        loss:torch.Tensor = loss_function(self.main_model(dummy_input), target)
        print(0, "  epoch/pred loss in bwlu ", f"{loss.item():.4e}", "  /input loss", f"{loss_of_input.item():.4e}")
        
        for epoch_backward_lookup in range(iter*epoch_per_iter):
            pred = self.main_model(dummy_input)
            loss:torch.Tensor = loss_function(pred, target)
            if 0 == epoch_backward_lookup and False:
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
#end of class.

if 'self backward look up test' and False:
    #this tests when the model is perfectly trained, what result of backward lookup(bwlu) would be like.
    # I directly used the same model to do all jobs here. Training data is generated with the same model.
    # So, no training here in this test.

    #the result is represented by how much % of elements in the backward lookup(bwlu) result is similar to the reference input.
    # all results are eyeball stats.
    # If dim < out dim, 85%elements are similar. If in dim >= out dim, 50% are similar.
    # when the start_from is provided, If dim < out dim, ~100%elements are VERY similar. If in dim >= out dim, 95~100% are similar.
    
    #conclusion. The theory of linear degree of freedom applies here.
    
    batch = 100000
    out_features = 5
    in_features = 3
    _inner_model = MLP_from_yagaodirac(in_features, out_features, mid_width=20,bias=True, num_layers=3)
    model = Backward_Lookup_able(_inner_model)
    is_gpu = True
    input = torch.randn([batch,in_features])
    if is_gpu:
        model.cuda()
        input = input.cuda()
        pass
    model.eval()
    target = model(input).detach().clone()
    
    loss_function = torch.nn.MSELoss()
    
    start_from = input+torch.randn_like(input)*0.1
    bwlu_result_with_start_from = model.backward_lookup(input, target, start_from=start_from)
    loss_of_backward_lookup_with_start_from = loss_function(bwlu_result_with_start_from, input)
    print(loss_of_backward_lookup_with_start_from, "loss_of_backward_lookup_with_start_from")
    print(bwlu_result_with_start_from[:3], "bwlu result with start from")
    print(input[:3], "ref input")
    
    bwlu_result = model.backward_lookup(input, target)
    loss_of_backward_lookup = loss_function(bwlu_result, input)
    print(loss_of_backward_lookup, "loss_of_backward_lookup")
    print(bwlu_result[:3], "bwlu result")
    print(input[:3], "ref input")
    pass



if 'backward look up fake teacher test' and False:
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!eyeball stats warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #                                   in dim > out dim      in dim == out dim      in dim < out dim 
    # trained student with start_from         90%                   70~90%                70~90%
    #         trained student                 20%                   30%                    40%
    #        untrained student                <5%                   <10%                   0%?
    batch = 100000
    out_features = 5
    in_features = 3
    print(in_features,"in_features / out_features",out_features)
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
    _inner_model_of_student = MLP_from_yagaodirac(in_features, out_features, mid_width=20,bias=True, num_layers=3)
    student = Backward_Lookup_able(_inner_model_of_student)
    if is_gpu:
        student.cuda()
        pass
    
    bad_result = student.backward_lookup(valid_input, valid_target)
    loss_of_backward_lookup_without_training = loss_function(bad_result, valid_input)
    print(loss_of_backward_lookup_without_training, "loss_of_backward_lookup_without_training")
    for i in range(5):
        print(bad_result[i], "bad_result")
        print(valid_input[i], "ref input")
        pass
    
    student.fit(input, target)
    good_result = student.backward_lookup(valid_input, valid_target)
    loss_of_backward_lookup = loss_function(good_result, valid_input)
    print(loss_of_backward_lookup, "loss_of_backward_lookup")
    for i in range(5):
        print(good_result[i], "good_result")
        print(valid_input[i], "ref input")
        pass
    
    start_from = valid_input+torch.randn_like(valid_input)*0.1
    better_result = student.backward_lookup(valid_input, valid_target, start_from=start_from)
    loss_of_backward_lookup_with_start_from = loss_function(better_result, valid_input)
    print(loss_of_backward_lookup_with_start_from, "loss_of_backward_lookup")
    for i in range(5):
        print(better_result[i], "better_result")
        print(valid_input[i], "ref input")
        pass
    play_noise()
    pass
    
    
      
