from datetime import datetime
from pathlib import Path
import statistics

import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))

from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2_mean_abs_to_1 as Gramo

import test_tool.simple_data_analysis as sda #also my tool.



# def print_list_stats(inp:list[float])->str:
#     if inp.__len__() == 0:
#         return "No data"
#     if inp.__len__() == 1:
#         return f'{inp[0]} (1 sample)'
#     return f'(mean,median,std) {statistics.mean(inp)} , {statistics.median(inp)} , {statistics.stdev(inp)} , ({inp.__len__()} samples)'
# if "test" and True:
#     empty_str = print_list_stats([])
#     a_lil_str = print_list_stats([1])
#     a_lot_str = print_list_stats([1,2])
#     a_lot_str_2 = print_list_stats([1,2,3,2,2])
#     pass


# def print_list_stats__int(inp:list[int])->str:
#     if inp.__len__() == 0:
#         return "No data"
#     if inp.__len__() == 1:
#         return f'{inp[0]} (1 sample)'
#     return f'(mean,median,std) {statistics.mean(inp)} , {statistics.median(inp)} , {statistics.stdev(inp)} , ({inp.__len__()} samples)'


# class first_time_epoch_tracker():
#     original_loss:float
#     from_1_over_10_to_the_power_of_n:int
#     to_1_over_10_to_the_power_of_n:int
    
#     _max_possible_pos:int
#     _current_pos:int
#     data:list[list[int]]#[power][index]
#     def __init__(self, original_loss:float, from_1_over_10_to_the_power_of_n:int, 
#                                         to_1_over_10_to_the_power_of_n:int):
#         assert original_loss>0.
#         assert 0<from_1_over_10_to_the_power_of_n
#         assert from_1_over_10_to_the_power_of_n<to_1_over_10_to_the_power_of_n
        
#         self.original_loss = original_loss
#         self.from_1_over_10_to_the_power_of_n = from_1_over_10_to_the_power_of_n
#         self.to_1_over_10_to_the_power_of_n = to_1_over_10_to_the_power_of_n
        
#         self._max_possible_pos = to_1_over_10_to_the_power_of_n-from_1_over_10_to_the_power_of_n
#         self._current_pos = 0
#         self.data = []
#         for _ in range(self._max_possible_pos+1):#+1
#             self.data.append([])
#         pass
#     def next_round(self):
#         self._current_pos = 0
#         pass
#     def clear(self):
#         self._current_pos = 0
#         self.data = []
#         for _ in range(self._max_possible_pos+1):#+1
#             self.data.append([])
#         pass
#     def _get_loss_standard(self, index:int)->float:
#         result = self.original_loss * 0.1**(self.from_1_over_10_to_the_power_of_n+index)
#         return result
#     def try_add(self, epoch:int, loss:float):
#         while True:
#             the_standard = self._get_loss_standard(self._current_pos)
#             if loss<the_standard:
#                 self.data[self._current_pos].append(epoch)
#                 self._current_pos +=1
#                 continue
#             else:
#                 break
#             #no tail
#             pass
#         pass
#     def print(self)->str:
#         result = ""
#         for ii in range(self.data.__len__()):
#             result += f'''reached loss {self._get_loss_standard(ii)}, {print_list_stats__int(self.data[ii]) }'''
#             if self.data[ii].__len__()>0:
#                 result += f''' / {self.data[ii].__len__()} sample(s) in total\n'''
#                 pass
#             else:
#                 result += f'''\n'''
#                 break
#             pass
#         return result
#     #end of class
# if "test":
#     ftet = first_time_epoch_tracker(1.0, 2, 5)
#     assert ftet._get_loss_standard(0) == 1.0*(0.1**2)    
#     assert ftet._get_loss_standard(1) == 1.0*(0.1**3)    
    
    
#     ftet = first_time_epoch_tracker(1.0, 2, 5)
#     assert ftet.data.__len__() == 4
#     _the_str = ftet.print()
#     assert _the_str == f'''reached loss {1.0*(0.1**2)}, {print_list_stats__int([])}\n'''
                
    
#     ftet = first_time_epoch_tracker(1.0, 2, 5)
#     ftet.try_add(3, 0.015)
#     assert ftet.data[0].__len__() == 0
#     assert ftet._current_pos == 0
#     _the_str = ftet.print()
#     assert _the_str == f'''reached loss {1.0*(0.1**2)}, {print_list_stats__int([])}\n'''
    
#     ftet.try_add(5, 0.005)
#     assert ftet.data[0].__len__() == 1
#     assert ftet.data[1].__len__() == 0
#     assert ftet._current_pos == 1
#     _the_str = ftet.print()
#     assert _the_str == f'''reached loss {1.0*(0.1**2)}, {print_list_stats__int([5]) \
#                 } / {1} sample(s) in total\n''' + \
#                     f'''reached loss {1.0*(0.1**3)}, {print_list_stats__int([])}\n'''
    
#     ftet.try_add(15, 0.0005)
#     assert ftet.data == [[5],[15],[],[]]
#     ftet.try_add(25, 0.0005)
#     assert ftet.data == [[5],[15],[],[]]
#     ftet.next_round()
#     ftet.try_add(33, 0.005)
#     assert ftet.data == [[5,33],[15],[],[]]
#     _the_str = ftet.print()
#     assert _the_str == f'''reached loss {1.0*(0.1**2)}, {print_list_stats__int([5,33]) \
#                 } / {2} sample(s) in total\n''' + \
#                     f'''reached loss {1.0*(0.1**3)}, {print_list_stats__int([15]) \
#                 } / {1} sample(s) in total\n''' + \
#                     f'''reached loss {1.0*(0.1**4)}, {print_list_stats__int([])}\n'''
    
#     ftet.clear()
#     assert ftet.data == [[],[],[],[]]
#     pass






sda.first_time_epoch_tracker()












#######################################################################
#test hyper param         test hyper param         test hyper param         
#test hyper param         test hyper param         test hyper param         
#test hyper param         test hyper param         test hyper param         
#test hyper param         test hyper param         test hyper param         
device = torch.device('cuda')
dtype = torch.float32
factory_kwargs = {"device": device, "dtype": dtype}

__NO_PRINT__ = True
def fake_print(*args, **kwargs):
    pass
if __NO_PRINT__:
    print = fake_print
    pass

test_count_target = 20
test_count_max = 50
#######################################################################



class Low_rand_mat_fac(torch.nn.Module):
    mat_1:torch.Tensor
    mat_2:torch.Tensor
    def __init__(self, gt_shape:tuple[int,int], target_rank:int, 
                device=None,dtype=None,*args, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert target_rank>0
        self.mat_1:torch.Tensor = torch.nn.Parameter(torch.randn((gt_shape[0], target_rank), device=device, dtype=dtype))
        self.mat_2:torch.Tensor = torch.nn.Parameter(torch.randn((target_rank, gt_shape[1]), **factory_kwargs))
        self.gramo_1 = Gramo(**factory_kwargs)
        self.gramo_2 = Gramo(**factory_kwargs)
        pass
    
    def forward(self) -> torch.Tensor:
        #model
        mat_1_after_gramo = self.gramo_1(self.mat_1)
        mat_2_after_gramo = self.gramo_2(self.mat_2)
        inferenced_mat:torch.Tensor = mat_1_after_gramo@mat_2_after_gramo
        return inferenced_mat

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"gt_shape={self.gt_shape}, target_rank={self.target_rank}"
    pass#end of class
if "no test for the class. Probably correct" and False:
    pass

#torch.manual_seed(123)

_swept_param = "decr_lr_by"

_time = datetime.now()
_time_str = _time.isoformat(sep=" ")
_time_str = _time_str[0:19]
_time_str_filename = _time_str.replace(":","-")
_file_name = f"{Path(__file__).parent/"test result"}\\sweep {_swept_param} {_time_str_filename}.txt"
with open(_file_name, mode = "a", encoding="utf-8") as file:
    file.write(f"sweep: {_swept_param}\n\n")
    file.write(f"{_time_str}\n\n")
    pass#open



##############################################################################
#param section         param section         param section         param section         
#param section         param section         param section         param section         
#param section         param section         param section         param section         
#param section         param section         param section         param section     
base_lr = 0.1    
below_this_then_incr_lr = 0.6#<0.2 or 0.6
incr_lr_by = 1.5
uppon_this_then_decr_lr = 1.
decr_lr_by = 0.4
#file.write(f"-  -  -  -  -  -  -  -  -  -  -  \n")
gt_dim = 5
gt_shape:tuple[int,int] = (gt_dim,gt_dim)
target_rank = 5
#file.write(f"-  -  -  -  -  -  -  -  -  -  -  \n")


sweep_in:torch.Tensor = torch.linspace(start=0.7, end=0.9,steps=5)
#sweep_in:torch.Tensor = torch.linspace(start=1., end=3.,steps=30)
for sweep in sweep_in:
    decr_lr_by = sweep.item()
    
    #param finished     param finished     param finished     param finished     
    #param finished     param finished     param finished     param finished     
    #param finished     param finished     param finished     param finished     
    #param finished     param finished     param finished     param finished     
    ##############################################################################
    
    
    
    
    
    
    

    ##############################################################################
    #test group begins      test group begins      test group begins      
    #test group begins      test group begins      test group begins      
    #test group begins      test group begins      test group begins      
    #test group begins      test group begins      test group begins      

    finished_count = 0
    epoch_count_list:list[int] = []
    end_loss:list[int] = []
    loss_at_start_list:list[float] = []
    loss_at_50_list:list[float] = []
    loss_at_100_list:list[float] = []
    loss_at_300_list:list[float] = []
    when_0_01_loss:list[float] = []
    when_0_001_loss:list[float] = []
    when_0_0001_loss:list[float] = []
    when_0_00001_loss:list[float] = []
    when_0_000001_loss:list[float] = []
    when_under_0_000001_loss:list[float] = []

#    loss_list:list[float] = []
#    ori_mean_list:list[float] = []
#    ori_std_list:list[float] = []
#    ori_abs_mean_list:list[float] = []
    for test_count in range(1,test_count_max+1):
        gt_mat:torch.Tensor = torch.randn(gt_shape, device=device,dtype=dtype)
        ##############################################################################
        #test begins        test begins        test begins        test begins        
        #test begins        test begins        test begins        test begins        
        #test begins        test begins        test begins        test begins        
        #test begins        test begins        test begins        test begins        
        model = Low_rand_mat_fac(gt_shape, target_rank, **factory_kwargs)
        start_lr = base_lr/incr_lr_by
        optimizer = torch.optim.SGD(params=model.parameters(), lr = start_lr)

        with torch.inference_mode():
            print(f"rand init: {model.mat_1@model.mat_2}")
            
            inferenced_mat = model()
            diff:torch.Tensor = inferenced_mat-gt_mat
            diff_sqr:torch.Tensor = diff*diff
            loss:torch.Tensor = diff_sqr.mean()#batch?
            
            loss_at_start = loss.item()
            loss_at_start_list.append(loss_at_start)
            pass
        
        finished = False
        for epoch in range(3000):
            inferenced_mat = model()
            diff:torch.Tensor = inferenced_mat-gt_mat
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
            if epoch == 49:
                loss_at_50_list.append(loss.item())
                print(f"epoch {epoch:5} / loss {loss.item():.10f} / lr {old_lr:.8f}")
                pass
            if epoch == 99:
                loss_at_100_list.append(loss.item())
                print(f"epoch {epoch:5} / loss {loss.item():.10f} / lr {old_lr:.8f}")
                pass
            if epoch == 299:
                loss_at_300_list.append(loss.item())
                print(f"epoch {epoch:5} / loss {loss.item():.10f} / lr {old_lr:.8f}")
                pass
            
            #if epoch%500 == 500-1:
            
            #update loss track
            old_loss = new_loss
            
            #optim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass

        print(f"test finished: {model.mat_1@model.mat_2}")
        print(f"gt: {gt_mat}")
        print(f"diff: {model.mat_1@model.mat_2-gt_mat}")

        if finished:
            print(f"gt_shape {gt_shape} / target_rank {target_rank} / epoch {epoch+1:5} / loss {loss.item():.10f}")
            finished_count +=1
            epoch_count_list.append(epoch+1)
            
            
            
            # loss_list.append(new_loss)
            # ori_mean_list.append(gt_mat.mean().item())
            # ori_abs_mean_list.append(gt_mat.abs().mean().item())
            # ori_std_list.append(gt_mat.std().item())
            pass
        
        if finished_count>=test_count_target:
            break
        pass

    
    
    
    


    print(f"test {finished_count}/{test_count}")
    if epoch_count_list.__len__()>=5:
        print(f"epoch_count(mean,std) ***{statistics.mean(epoch_count_list)}***,{statistics.stdev(epoch_count_list):.2f}")
    elif epoch_count_list.__len__()>=1:
        print(f"epoch_count(mean) ***{statistics.mean(epoch_count_list)}***")
    else:
        #nothing to print
        pass
    
        
    
    
        
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
        if epoch_count_list.__len__()>=5:
            file.write(f"epoch_count(mean,std) ***{statistics.mean(epoch_count_list)}***,{statistics.stdev(epoch_count_list):.2f}")
        elif epoch_count_list.__len__()>=1:
            file.write(f"epoch_count(mean) ***{statistics.mean(epoch_count_list)}***")
        else:
            #nothing to file.write
            pass
        
    
    
    with open(_file_name, mode = "a", encoding="utf-8") as file:
        file.write(f"base_lr {base_lr}\n")
        file.write(f"below_this_then_incr_lr {below_this_then_incr_lr}\n")
        file.write(f"incr_lr_by {incr_lr_by}\n")
        file.write(f"uppon_this_then_decr_lr {uppon_this_then_decr_lr}\n")
        file.write(f"decr_lr_by {decr_lr_by}\n")
        file.write(f"-  -  -  -  -  -  -  -  -  -  -  \n")
        file.write(f"gt_dim {gt_dim}\n")
        file.write(f"gt_shape {gt_shape}\n")
        file.write(f"target_rank {target_rank}\n")
        file.write(f"-  -  -  -  -  -  -  -  -  -  -  \n")
        pass
    
    with open(_file_name, mode = "a", encoding="utf-8") as file:
        file.write(f"test {finished_count}/{test_count}\n")
        if epoch_count_list.__len__()>=5:
            file.write(f"epoch_count(mean,std) ***{statistics.mean(epoch_count_list)}***,{statistics.stdev(epoch_count_list):.2f}")
        elif epoch_count_list.__len__()>=1:
            file.write(f"epoch_count(mean) ***{statistics.mean(epoch_count_list)}***")
        else:
            #nothing to file.write
            pass
        
        loss_abs_list = []
        for _loss in loss_list:
            loss_abs_list.append(abs(_loss))
            pass
        loss_abs_mean = statistics.mean(loss_abs_list)
        loss_mean = statistics.mean(loss_list)
        loss_std = statistics.stdev(loss_list)
        file.write(f"loss(abs mean,mean,std) {loss_abs_mean},    {loss_mean},    {loss_std}\n")

        ori_mean = statistics.mean(ori_mean_list)
        ori_abs_mean = statistics.mean(ori_abs_mean_list)
        ori_std = statistics.mean(ori_std_list)
        file.write(f"ori(abs mean,mean,std) {ori_abs_mean},    {ori_mean},    {ori_std}\n")

        file.write(f"loss/ori {loss_abs_mean/ori_abs_mean},    {loss_mean/ori_mean},    {loss_std/ori_std}\n")
        file.write(f"\n\n\n\n")
        pass# open
    
    
    
    

    pass#for sweep param.
