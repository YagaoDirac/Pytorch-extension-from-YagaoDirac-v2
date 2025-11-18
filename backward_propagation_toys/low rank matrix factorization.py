import statistics

import torch

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2_mean_abs_to_1 as Gramo



class Low_rand_mat_fac(torch.nn.Module):
    mat_1:torch.Tensor
    mat_2:torch.Tensor
    def __init__(self, gt_shape:tuple[int,int], target_rank:int, 
                device=None,dtype=None,*args, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert target_rank>0
        self.mat_1:torch.Tensor = torch.nn.Parameter(torch.randn((gt_shape[0], target_rank),requires_grad=True))
        self.mat_2:torch.Tensor = torch.nn.Parameter(torch.randn((target_rank, gt_shape[1]),requires_grad=True))
        self.gramo = Gramo()
        pass
    
    def forward(self) -> torch.Tensor:
        #model
        inferenced_mat:torch.Tensor = self.mat_1@self.mat_2
        after_gramo = self.gramo(inferenced_mat)
        return after_gramo

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"gt_shape={self.gt_shape}, target_rank={self.target_rank}"
    pass#end of class
if "no test for the class. Probably correct" and False:
    pass

#torch.manual_seed(123)

gt_dim = 5
gt_shape:tuple[int,int] = (gt_dim,gt_dim)
gt_mat:torch.Tensor = torch.randn(gt_shape)
target_rank = 5

finished_count = 0
epoch_count_list:list[int] = []
loss_list:list[float] = []
for test_count in range(1,20+1):
    #init test
    model = Low_rand_mat_fac(gt_shape, target_rank)

    start_lr = 0.1
    optimizer = torch.optim.SGD(params=model.parameters(), lr = start_lr)
    old_loss:float = 9999999999999999999999.
    finished = False
    for epoch in range(10000):
        inferenced_mat = model()
        diff:torch.Tensor = inferenced_mat-gt_mat
        diff_sqr:torch.Tensor = diff*diff
        loss:torch.Tensor = diff_sqr.mean()#batch?
        #optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #tail
        new_loss:float = loss.item()
        old_lr = optimizer.param_groups[0]['lr']
        #update the lr.
        _100 = 100
        if epoch%_100 == _100-1:
            if new_loss<old_loss-0.001:
                #it changes too fast, 
                optimizer.param_groups[0]['lr'] = 0.1
                pass
            elif new_loss>old_loss+0.1:
                #it getting worse. maybe restart the test.
                break
            elif new_loss>old_loss-0.00000001:
                #it doesn't change
                if old_lr < 0.0001:
                    print(f"epoch {epoch:5} / loss {loss.item():.10f} / lr {old_lr:.8f} / # end")
                    finished = True
                    break
                optimizer.param_groups[0]['lr'] = old_lr*0.6
                pass
            else:
                #don't touch the lr.
                pass
            #not too fast or too slow.
            pass
        #report
        _1000 = 500
        if epoch%_1000 == _1000-1:
            #report every 1000 epoch.
            print(f"epoch {epoch:5} / loss {loss.item():.10f} / lr {old_lr:.8f}")
            pass
        old_loss = new_loss
        pass
    print(f"gt_shape {gt_shape} / target_rank {target_rank} / epoch {epoch+1:5} / loss {loss.item():.10f}")

    if finished:
        epoch_count_list.append(epoch+1)
        loss_list.append(new_loss)
        finished_count +=1
        pass
    
    if finished_count>=5:
        break
    pass


1w 继续。
statistics.mean()
statistics.stdev(loss_list)
print(f"test {finished_count}/{test_count}")
print(f"epoch_count(mean,std) {statistics.mean(epoch_count_list)},{statistics.stdev(epoch_count_list)}")
print(f"loss(mean,std) {statistics.mean(loss_list)},{statistics.stdev(loss_list)}")
        
        
