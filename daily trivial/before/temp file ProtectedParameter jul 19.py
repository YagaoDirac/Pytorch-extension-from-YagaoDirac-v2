from typing import Any, List, Optional, Self
import torch
import math





'''style 1, if anything exceeds 7, anything exceeds 3 are compressed back a bit.'''

epi = 0.01
threshold1 = 3.
threshold2 = 7.
margin = 1.
test_weight = torch.tensor([[1., 2., 3.],[1., 3., 6.],[3., 6., 11.],[5., 10., 15.],])
gt_t2 = test_weight.gt(threshold2)
print(gt_t2, "gt_t2")
at_least_smt_gt_t2 = gt_t2.any(dim=1)
print(at_least_smt_gt_t2, "at_least_smt_gt_t2")
if at_least_smt_gt_t2.any().item():
    the_max_value = test_weight.max(dim=1, keepdim=True).values
    print(the_max_value, "the_max_value")
    gt_t1 = test_weight.gt(threshold1)
    print(gt_t1, "gt_t1")
    
    at_least_smt_gt_t2_expanded = at_least_smt_gt_t2[:, None].expand(-1, test_weight.shape[1])
    print(at_least_smt_gt_t2_expanded.shape, "at_least_smt_gt_t2_expanded.shape")
    
    modify_these = gt_t1.logical_and(at_least_smt_gt_t2_expanded)
    print(modify_these, "modify_these")
    
    exceed = the_max_value-threshold1
    exceed = exceed.abs()+epi#or exceed.mul(at_least_smt_gt_t2)+epi
    print(exceed, "exceed")
    
    mul_me = (threshold2-threshold1-margin)/exceed
    print(mul_me, "mul_me")
    test_weight_1 = modify_these*((test_weight-threshold1)*mul_me+threshold1)+(modify_these.logical_not())*test_weight
    print(test_weight_1, "test_weight_1")
pass
    
test_weight2 = test_weight.detach().clone()*-1
print(test_weight2, "test_weight2")
lt_nt2 = test_weight2.lt(-1.*threshold2)
print(lt_nt2, "lt_nt2")
at_least_smt_lt_nt2 = lt_nt2.any(dim=1)
print(at_least_smt_lt_nt2, "at_least_smt_lt_nt2")
if at_least_smt_lt_nt2.any().item():
    the_min_value = test_weight2.min(dim=1, keepdim=True).values
    print(the_min_value, "the_min_value")
    lt_nt1 = test_weight2.lt(-threshold1)
    print(lt_nt1, "lt_nt1")
    
    at_least_smt_lt_nt2_expanded = at_least_smt_lt_nt2[:, None].expand(-1, test_weight.shape[1])
    print(at_least_smt_lt_nt2_expanded, "at_least_smt_lt_nt2_expanded")
    print(at_least_smt_lt_nt2_expanded.shape, "at_least_smt_lt_nt2_expanded.shape")
    
    modify_these_negative = lt_nt1.logical_and(at_least_smt_lt_nt2_expanded)
    print(modify_these_negative, "modify_these_negative")
    
    exceed = the_min_value+threshold1
    exceed = exceed.abs()+epi#or exceed.mul(at_least_smt_gt_t2)+epi
    print(exceed, "exceed")
    
    mul_me_negative = (threshold2-threshold1-margin)/exceed
    print(mul_me_negative, "mul_me_negative")
    test_weight_3 = modify_these_negative*((test_weight2+threshold1)*mul_me_negative-threshold1) \
        +(modify_these_negative.logical_not())*test_weight2
    print(test_weight_3, "test_weight_3")
pass
    
fds=432






'''style 2, the mean is always 0, while extreme values are clamped back a bit.'''

if self.update_count>=self.auto_merge_duration:
    self.update_count = 0
    with torch.no_grad():
        #this automatic modification may mess sometime.
        boundary = self.raw_weight_boundary_for_f32
        if self.raw_weight.dtype == torch.float64:
            boundary *= 2.
            pass
        if self.raw_weight.dtype == torch.float16:
            boundary *= 0.5
            pass
        
        flag = self.raw_weight.gt(boundary)
        temp = flag*boundary
        temp = temp.to(self.raw_weight.dtype)
        temp = temp + self.raw_weight.data*(flag.logical_not())
        self.raw_weight.data = temp

        boundary*=-1.
        flag = self.raw_weight.lt(boundary)
        temp = flag*boundary
        temp = temp.to(self.raw_weight.dtype)
        temp = temp + self.raw_weight.data*(flag.logical_not())
        self.raw_weight.data = temp
        
        mean = self.raw_weight.mean(dim=1,keepdim=True)
        self.raw_weight.data = self.raw_weight.data-mean
        pass
    pass
else:
    self.update_count+=1
    pass