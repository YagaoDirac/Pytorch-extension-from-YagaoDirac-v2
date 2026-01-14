from typing import Optional
from datetime import datetime
from pathlib import Path
import statistics

import torch

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2_mean_abs_to_1 as Gramo



class Rotation_layer(torch.nn.Module):
    mat:torch.Tensor
    def __init__(self, dim:int, original:Optional[torch.Tensor] = None, 
                device=None,dtype=None,*args, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if original:
            self.mat:torch.Tensor = torch.nn.Parameter(original.detach().clone(), requires_grad=True, device=device, dtype=dtype)
            pass
        else:
            self.mat:torch.Tensor = torch.nn.Parameter(torch.randn((dim,dim),requires_grad=True, device=device, dtype=dtype))
            pass
        self.gramo_1 = Gramo(device=device)
        self.gramo_2 = Gramo(device=device)
        pass
    
    def forward(self) -> torch.Tensor:
        #model
        mat_1_after_gramo = self.gramo_1(self.mat)
        mat_2_after_gramo = self.gramo_2(self.mat.transpose())
        should_be_identical_mat:torch.Tensor = mat_1_after_gramo@mat_2_after_gramo
        return should_be_identical_mat

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"dim={self.mat.shape[0]}"
    
    def correct(self):
        assert False, "copy the entire training process into here."
    
    pass#end of class
if "no test for the class. Probably correct" and False:
    pass

#torch.manual_seed(123)



below_this_then_incr_lr = 0.8#[0.5, 0.8]:
incr_lr_by = 2.
uppon_this_then_decr_lr = 1.
decr_lr_by = 0.4

for sweep in [0.8]:#[0.5, 0.8]:
    below_this_then_incr_lr = sweep
    
    dim = 5
    gt_shape:tuple[int,int] = (dim,dim)
    start_mat:torch.Tensor = torch.randn(gt_shape)
    _identical_mat = torch.eye(dim)

    finished_count = 0
    epoch_count_list:list[int] = []
    loss_list:list[float] = []
    ori_mean_list:list[float] = []
    ori_std_list:list[float] = []
    ori_abs_mean_list:list[float] = []
    for test_count in range(1,20+1):
        #init test
        model = Rotation_layer(dim)
        model_start_point:torch.Tensor = model.mat.detach().clone()

        start_lr = 0.1
        optimizer = torch.optim.SGD(params=model.parameters(), lr = start_lr)
        old_loss:float = 9999999999999999999999.
        finished = False
        for epoch in range(10000):
            should_be_identical_mat = model()
            diff:torch.Tensor = should_be_identical_mat-_identical_mat
            diff_sqr:torch.Tensor = diff*diff
            loss:torch.Tensor = diff_sqr.mean()#batch?
            
            #tail
            new_loss:float = loss.item()
            old_lr = optimizer.param_groups[0]['lr']
            #update the lr.
            #new loss??
            new_lr = old_lr
            if new_loss<old_loss*below_this_then_incr_lr:
                new_lr = old_lr*incr_lr_by
                pass
            elif new_loss>old_loss*uppon_this_then_decr_lr:
                new_lr = old_lr*decr_lr_by
                pass
            else:
                pass
            #safety for new loss
            if new_lr>start_lr:
                new_lr = start_lr
                pass
            if new_lr<start_lr*0.001:
                new_lr = start_lr*0.001
                pass
            #set new loss
            optimizer.param_groups[0]['lr'] = new_lr
            
            #finished?
            if old_lr < start_lr*0.002 and new_loss < 1e-6:
                print(f"epoch {epoch:5} / loss {loss.item():.10f} / lr {old_lr:.8f} / # end")
                finished = True
                break
            
            #report
            if epoch%500 == 500-1:
                #report every 1000 epoch.
                print(f"epoch {epoch:5} / loss {loss.item():.10f} / lr {old_lr:.8f}")
                pass
            
            #update loss track
            old_loss = new_loss
            
            #optim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass

        if finished:
            print(f"gt_shape {gt_shape} / target_rank {target_rank} / epoch {epoch+1:5} / loss {loss.item():.10f}")
            finished_count +=1
            epoch_count_list.append(epoch+1)
            loss_list.append(new_loss)
            
            1w 误差怎么定义？？？
            #the real error.
            model_start_point@(model_start_point.transpose())
            
            ori_mean_list.append(gt_mat.mean().item())
            ori_abs_mean_list.append(gt_mat.abs().mean().item())
            ori_std_list.append(gt_mat.std().item())
            pass
        
        if finished_count>=5:
            break
        pass

    
    
    
    


    print(f"test {finished_count}/{test_count}")
    print(f"epoch_count(mean,std) {statistics.mean(epoch_count_list)},{statistics.stdev(epoch_count_list):.2f}")
    loss_abs_list = []
    for _loss in loss_list:
        loss_abs_list.append(abs(_loss))
        pass
    loss_abs_mean = statistics.mean(loss_abs_list)
    loss_mean = statistics.mean(loss_list)
    loss_std = statistics.stdev(loss_list)
    print(f"loss(abs mean,mean,std) {loss_abs_mean}, {loss_mean}, {loss_std}")

    # ori_mean = statistics.mean(ori_mean_list)
    # ori_abs_mean = statistics.mean(ori_abs_mean_list)
    # ori_std = statistics.mean(ori_std_list)
    # print(f"ori(abs mean,mean,std) {ori_abs_mean}, {ori_mean}, {ori_std}")

    # print(f"loss/ori {loss_abs_mean/ori_abs_mean}, {loss_mean/ori_mean}, {loss_std/ori_std}")
    
    
    
    #log out.
    with open(_file_name, mode = "a", encoding="utf-8") as file:
        file.write(f"test {finished_count}/{test_count}\n")
        file.write(f"epoch_count(mean,std) {statistics.mean(epoch_count_list)},{statistics.stdev(epoch_count_list):.2f}\n")
        loss_abs_list = []
        for _loss in loss_list:
            loss_abs_list.append(abs(_loss))
            pass
        loss_abs_mean = statistics.mean(loss_abs_list)
        loss_mean = statistics.mean(loss_list)
        loss_std = statistics.stdev(loss_list)
        file.write(f"loss(abs mean,mean,std) {loss_abs_mean}, {loss_mean}, {loss_std}\n")

        ori_mean = statistics.mean(ori_mean_list)
        ori_abs_mean = statistics.mean(ori_abs_mean_list)
        ori_std = statistics.mean(ori_std_list)
        file.write(f"ori(abs mean,mean,std) {ori_abs_mean}, {ori_mean}, {ori_std}\n")

        file.write(f"loss/ori {loss_abs_mean/ori_abs_mean}, {loss_mean/ori_mean}, {loss_std/ori_std}\n\n")
        pass# open
    
    
    
    

    pass#for sweep param.
