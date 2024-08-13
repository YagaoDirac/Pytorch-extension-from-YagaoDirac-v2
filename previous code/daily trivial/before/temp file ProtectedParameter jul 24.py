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








    def set_param_protection_config(self, threshold1:float = -1., \
            threshold2:float = -1., margin:float = -1., epi:float = -1.):
        r'''Use any number <= 0. not to touch this param.
        
        example:
        >>> set_param_protection_config(-1., -1., 0.5, -1.)
        to set only margin.
        '''
        if threshold1<=0.:
            threshold1 = self.threshold1.item()
            pass
        if threshold2<=0.:
            threshold2 = self.threshold2.item()
            pass
        if margin<=0.:
            margin = self.margin.item()
            pass
        if epi<=0.:
            epi = self.epi.item()
            pass
        if threshold1+margin>=threshold2:
            raise Exception("threshold1+margin must be less than threshold2.")
        if epi>=threshold1 or epi>=threshold2 or epi>=margin:
            raise Exception("epi seems too big. If you know what you are doing, comment this line out.")
        
        self.threshold1 = torch.nn.Parameter(torch.tensor([threshold1]), requires_grad=False)
        self.threshold2 = torch.nn.Parameter(torch.tensor([threshold2]), requires_grad=False)
        self.margin = torch.nn.Parameter(torch.tensor([margin]), requires_grad=False)
        self.epi = torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
        raise Exception("untested.")







self.protect_param_every____training = protect_param_every____training 
        self.training_count = 0 
        self.threshold1 = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        self.threshold2 = torch.nn.Parameter(torch.tensor([7.]), requires_grad=False)
        self.margin = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.epi = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=False)








#param protection
        if self.training_count<self.protect_param_every____training:
            self.training_count +=1
        else:
            self.training_count = 0
            #param protection
            # the positive part
            target_exceed = self.threshold2-self.threshold1-self.margin
            gt_t2 = self.raw_weight.gt(self.threshold2)
            at_least_smt_gt_t2 = gt_t2.any(dim=1)
            if at_least_smt_gt_t2.any().item():
                the_max_value = self.raw_weight.max(dim=1, keepdim=True).values
                gt_t1 = self.raw_weight.gt(self.threshold1)
                at_least_smt_gt_t2_expanded = at_least_smt_gt_t2[:, None].expand(-1, self.raw_weight.shape[1])
                modify_these = gt_t1.logical_and(at_least_smt_gt_t2_expanded)
                exceed = the_max_value-self.threshold1
                exceed = exceed.abs()+self.epi#or exceed.mul(at_least_smt_gt_t2)+epi
                mul_me = target_exceed/exceed
                self.raw_weight.data = modify_these*((self.raw_weight-self.threshold1)*mul_me+self.threshold1)+(modify_these.logical_not())*self.raw_weight
            pass
            # the negative part
            lt_nt2 = self.raw_weight.lt(-1.*self.threshold2)
            at_least_smt_lt_nt2 = lt_nt2.any(dim=1)
            if at_least_smt_lt_nt2.any().item():
                the_min_value = self.raw_weight.min(dim=1, keepdim=True).values
                lt_nt1 = self.raw_weight.lt(-1.*self.threshold1)
                at_least_smt_lt_nt2_expanded = at_least_smt_lt_nt2[:, None].expand(-1, self.raw_weight.shape[1])
                modify_these_negative = lt_nt1.logical_and(at_least_smt_lt_nt2_expanded)
                exceed = the_min_value+self.threshold1
                exceed = exceed.abs()+self.epi#or exceed.mul(at_least_smt_gt_t2)+epi
                mul_me_negative = target_exceed/exceed
                self.raw_weight.data = modify_these_negative*((self.raw_weight+self.threshold1)*mul_me_negative-self.threshold1) \
                    +(modify_these_negative.logical_not())*self.raw_weight
            pass
        # end of param protection.



























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