from typing import Any, List, Optional, Self
import torch
import math


class GradientModificationFunction(torch.autograd.Function):
    r'''input param:
    >>> x:torch.Tensor (must be set as require_grad = True)
    >>> scaling_ratio = torch.tensor([1.])
    >>> epi = torch.tensor([1e-5])
    >>> div_me_when_g_too_small = torch.tensor([1e-3])
    
    retur type: torch.Tensor
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, scaling_ratio:float = torch.tensor([1.]), \
        #               epi=torch.tensor([1e-5]), \
        #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
        #but python grammar punched me.
        x:torch.Tensor = args[0]
        scaling_ratio = args[1]
        epi = args[2]
        div_me_when_g_too_small = args[3]
        # the default values:
        # scaling_ratio = torch.tensor([1.])
        # epi = torch.tensor([0.00001]) 
        # div_me_when_g_too_small = torch.tensor([0.001]) 
        # the definition of the 3 param are different from the previous version
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")

        ctx.save_for_backward(scaling_ratio, epi, div_me_when_g_too_small)
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        scaling_ratio:torch.Tensor
        scaling_ratio, epi, div_me_when_g_too_small = ctx.saved_tensors

        #the shape should only be rank2 with[batch, something]
        # original_shape = g.shape
        # if len(g.shape) == 1:
        #     g = g.unsqueeze(1)
        # protection against div 0    
        length:torch.Tensor = g.mul(g).sum(dim=1,).sqrt()
        too_small:torch.Tensor = length.le(epi)
        div_me = too_small.logical_not()*length + too_small*div_me_when_g_too_small
        div_me = div_me.unsqueeze(dim=1)
        div_me = div_me.to(g.dtype)
        g_out:torch.Tensor = g/div_me
        
        scaling_ratio = scaling_ratio.to(g.dtype)
        if 1.!=scaling_ratio.item():
            g_out *= scaling_ratio
            pass

        return g_out, None, None, None

    pass  # class


class GradientModification(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """
    scaling_ratio:torch.Tensor
    def __init__(self, scaling_ratio:float = 1., \
                       epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio]))
        self.scaling_ratio.requires_grad_(False)
        self.epi=torch.nn.Parameter(torch.tensor([epi]))
        self.epi.requires_grad_(False)
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small]))
        self.div_me_when_g_too_small.requires_grad_(False)
        #raise Exception("untested")
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if not x.requires_grad:
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction.apply(x, self.scaling_ratio, self.epi, \
                                                   self.div_me_when_g_too_small)
    def set_scaling_ratio(self, scaling_ratio:float)->None:
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio], requires_grad=False))
        self.scaling_ratio.requires_grad_(False)
    def set_epi(self, epi:float)->None:
        self.epi = torch.nn.Parameter(torch.tensor([epi], requires_grad=False))
        self.epi.requires_grad_(False)
    def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small], requires_grad=False))
        self.div_me_when_g_too_small.requires_grad_(False)
        pass



#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111
#111111111111111111111111111111111111111111111111111111111111










class DigitalMapperFunction_v2(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x = args[0]# shape must be [batch, in_features]
        raw_weight:torch.Tensor = args[1]# shape must be [out_features, in_features]
                
        #print(out_features_iota, "out_features_iota")
        index_of_max_o = raw_weight.max(dim=1).indices
        #print(index_of_max_o, "index_of_max_o")
        output = x[:, index_of_max_o]
        #print(output, "output")

        out_features_s = torch.tensor(raw_weight.shape[0])
        raw_weight_shape = torch.tensor(raw_weight.shape)
        ctx.save_for_backward(x, index_of_max_o, out_features_s, raw_weight_shape)
        return output

    @staticmethod
    def backward(ctx, g_in):
        #shape of g_in must be [batch, out_features]
        
        x:torch.Tensor
        index_of_max_o:torch.Tensor
        out_features_s:torch.Tensor
        raw_weight_shape:torch.Tensor
        x, index_of_max_o, out_features_s, raw_weight_shape = ctx.saved_tensors
                
        in_features_s = x.shape[1]
        out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)

        input_reshaped_b_1_i = x[:,None,:]
        # print(input_reshaped_b_1_i, "input_reshaped_b_1_i")
        # print(input_reshaped_b_1_i.shape, "input_reshaped_b_1_i.shape")
        g_in_reshaped_b_o_1:torch.Tensor = g_in[:,:,None]
        # print(g_in_reshaped_b_o_1, "g_in_reshaped_b_o_1")
        # print(g_in_reshaped_b_o_1.shape, "g_in_reshaped_b_o_1.shape")
        grad_for_raw_weight_before_sum_b_o_i = g_in_reshaped_b_o_1.matmul(input_reshaped_b_1_i)
        # print(grad_for_raw_weight_before_sum_b_o_i, "grad_for_raw_weight_before_sum_b_o_i before sum")
        # print(grad_for_raw_weight_before_sum_b_o_i.shape, "grad_for_raw_weight_before_sum_b_o_i.shape before sum")
        grad_for_raw_weight_o_i = grad_for_raw_weight_before_sum_b_o_i.sum(dim=0, keepdim=False)
        # print(grad_for_raw_weight_o_i, "grad_for_raw_weight_o_i after sum, the final result.")
        # print(grad_for_raw_weight_o_i.shape, "grad_for_raw_weight_o_i.shape. Should be ", raw_weight.shape)

        the_temp_o_i = torch.zeros_like(raw_weight_shape)
        the_temp_o_i[out_features_iota_o, index_of_max_o] = 1.
        the_temp_expanded_fake_b_o_i = the_temp_o_i.expand(g_in.shape[0], -1, -1)
        # print(the_temp_expanded_fake_b_o_i.shape, "the_temp_fake_b_o_i.shape")
        # print(the_temp_expanded_fake_b_o_i, "the_temp_fake_b_o_i")

        g_in_reshaped_expanded_b_o_fake_i = g_in_reshaped_b_o_1.expand(-1, -1, in_features_s)
        # print(g_in_reshaped_expanded_b_o_fake_i, "g_in_reshaped_expanded_b_o_fake_i")
        # print(g_in_reshaped_expanded_b_o_fake_i.shape, "g_in_reshaped_expanded_b_o_fake_i.shape")

        the_temp_after_mul_b_o_i = the_temp_expanded_fake_b_o_i.mul(g_in_reshaped_expanded_b_o_fake_i)
        # print(the_temp_after_mul_b_o_i, "the_temp_after_mul_b_o_i")

        grad_for_x_b_i = the_temp_after_mul_b_o_i.sum(dim=1, keepdim=False)
        # print(grad_for_x_b_i, "grad_for_x_b_i")
        # print(grad_for_x_b_i.shape, "grad_for_x_b_i.shape")
                
        return grad_for_x_b_i, grad_for_raw_weight_o_i

    pass  # class


# '''This basic test is a bit long. It consists of 3 parts. 
# A raw computation, using x, raw_weight, and g_in as input, calcs the output and both grads.
# Then, the equivalent calc, which use the proxy weight instead of raw_weight,
# which allows directly using autograd.backward function to calc the grads.
# At last, the customized backward version.
# They all should yield the same results.'''

# x = torch.tensor([[5., 6., 7.], [8., 9., 13]])
# in_features_s = x.shape[1]

# raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
# out_features_s = raw_weight.shape[0]
# out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
# #print(out_features_iota, "out_features_iota")
# index_of_max_o = raw_weight.max(dim=1).indices
# print(index_of_max_o, "index_of_max_o")
# output = x[:, index_of_max_o]
# print(output, "output")

# #fake_ctx = (x, index_of_max_o, out_features_s)

# g_in = torch.tensor([[0.30013, 0.30103, 0.30113, 0.31003, 0.31013], [0.40013, 0.40103, 0.40113, 0.41003, 0.41013], ])
# print(output.shape, "output.shape, should be ", g_in.shape)
# #grad_for_raw_weight = torch.zeros_like(raw_weight)

# #sum_of_input = x.sum(dim=0, keepdim=True)
# input_reshaped_b_1_i = x[:,None,:]#x.unsqueeze(dim=1)
# print(input_reshaped_b_1_i, "input_reshaped_b_1_i")
# print(input_reshaped_b_1_i.shape, "input_reshaped_b_1_i.shape")
# g_in_reshaped_b_o_1 = g_in[:,:,None]#.unsqueeze(dim=-1)
# print(g_in_reshaped_b_o_1, "g_in_reshaped_b_o_1")
# print(g_in_reshaped_b_o_1.shape, "g_in_reshaped_b_o_1.shape")
# grad_for_raw_weight_before_sum_b_o_i = g_in_reshaped_b_o_1.matmul(input_reshaped_b_1_i)
# print(grad_for_raw_weight_before_sum_b_o_i, "grad_for_raw_weight_before_sum_b_o_i before sum")
# print(grad_for_raw_weight_before_sum_b_o_i.shape, "grad_for_raw_weight_before_sum_b_o_i.shape before sum")
# grad_for_raw_weight_o_i = grad_for_raw_weight_before_sum_b_o_i.sum(dim=0, keepdim=False)
# #grad_for_raw_weight_before_sum_b_o_i = grad_for_raw_weight_before_sum_b_o_i.squeeze(dim=0)
# print(grad_for_raw_weight_o_i, "grad_for_raw_weight_o_i after sum, the final result.")
# print(grad_for_raw_weight_o_i.shape, "grad_for_raw_weight_o_i.shape. Should be ", raw_weight.shape)

# the_temp_o_i = torch.zeros_like(raw_weight)
# the_temp_o_i[out_features_iota_o, index_of_max_o] = 1.
# the_temp_expanded_fake_b_o_i = the_temp_o_i.expand(g_in.shape[0], -1, -1)
# print(the_temp_expanded_fake_b_o_i.shape, "the_temp_expanded_fake_b_o_i.shape")
# print(the_temp_expanded_fake_b_o_i, "the_temp_expanded_fake_b_o_i")

# g_in_reshaped_expanded_b_o_fake_i = g_in_reshaped_b_o_1.expand(-1, -1, in_features_s)
# print(g_in_reshaped_expanded_b_o_fake_i, "g_in_reshaped_expanded_b_o_fake_i")
# print(g_in_reshaped_expanded_b_o_fake_i.shape, "g_in_reshaped_expanded_b_o_fake_i.shape")

# the_temp_after_mul_b_o_i = the_temp_expanded_fake_b_o_i.mul(g_in_reshaped_expanded_b_o_fake_i)
# print(the_temp_after_mul_b_o_i, "the_temp_after_mul_b_o_i")

# grad_for_x_b_i = the_temp_after_mul_b_o_i.sum(dim=1, keepdim=False)
# print(grad_for_x_b_i, "grad_for_x_b_i")
# print(grad_for_x_b_i.shape, "grad_for_x_b_i.shape")

# print("--------------SUMMARIZE-------------")
# print(output, "output")
# print(grad_for_x_b_i, "grad_for_x")
# print(grad_for_raw_weight_o_i, "grad_for_raw_weight")

# print("--------------VALIDATION-------------")

# __proxy_weight = torch.zeros_like(raw_weight)
# __proxy_weight.requires_grad_(True)
# __proxy_weight.data[out_features_iota_o, index_of_max_o] = 1.
# #print(__proxy_weight, "__proxy_weight")
# #print(__proxy_weight.shape, "__proxy_weight.shape")

# __valid_input_reshaped = x.unsqueeze(dim=-1)
# __valid_input_reshaped.requires_grad_(True)
# #print(__valid_input_reshaped, "__valid_input_reshaped")
# #print(__valid_input_reshaped.shape, "__valid_input_reshaped.shape")

# __valid_output = __proxy_weight.matmul(__valid_input_reshaped)
# __valid_output = __valid_output.squeeze(dim=-1)

# print(__valid_output, "__valid_output")
# #print(__valid_output.shape, "__valid_output.shape")

# #print(__valid_input_reshaped.grad, "__valid_input_reshaped.grad before")
# #print(__proxy_weight.grad, "__proxy_weight.grad before")
# torch.autograd.backward(__valid_output, g_in, inputs=[__valid_input_reshaped, __proxy_weight])
# print(__valid_input_reshaped.grad, "__valid_input_reshaped.grad after")
# print(__proxy_weight.grad, "__proxy_weight.grad after")

# print("-----------now test for the autograd function-----------")
# function_test__input = x.detach().clone().requires_grad_(True)
# function_test__input.grad = None
# function_test__raw_weight = raw_weight.detach().clone().requires_grad_(True)
# function_test__raw_weight.grad = None
# function_test__output = DigitalMapperFunction_v2.apply(function_test__input, function_test__raw_weight)
# print(function_test__output, "function_test__output")
# print(function_test__output.shape, "function_test__output.shape")
# torch.autograd.backward(function_test__output, g_in, inputs = \
#     [function_test__input, function_test__raw_weight])
# print(function_test__input.grad, "function_test__input.grad")
# print(function_test__raw_weight.grad, "function_test__raw_weight.grad")
# fds=432














class DigitalMapper_V2(torch.nn.Module):
    r'''This layer is NOT designed to be used directly.
    Use the wrapper class DigitalMapper.
    '''
    #__constants__ = []
    
    protect_param_every____training:int
    training_count:int
    
    def __init__(self, in_features: int, out_features: int, \
                        scaling_ratio:float = 200., \
                        protect_param_every____training:int = 20, \
                    device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        self.gramo = GradientModification(scaling_ratio=scaling_ratio)
        
        self.protect_param_every____training = protect_param_every____training 
        self.training_count = 0 
        
        self.threshold1 = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        self.threshold2 = torch.nn.Parameter(torch.tensor([7.]), requires_grad=False)
        self.margin = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.epi = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=False)
        
        pass

    def reset_parameters(self) -> None:
        '''copied from torch.nn.Linear'''
        torch.nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        pass
        
    def accepts_non_standard_range(self)->str:
        return "although this layer accepts non standard input, I recommend you only feed standard +-1(np) as input."
    def outputs_standard_range(self)->str:
        return "It depends on what you feed in. If the input is standard +-1(np), the output is also standard +-1(np)."    
    # def outputs_non_standard_range(self)->bool:
    #     return not self.outputs_standard_range()
    
    def set_scaling_ratio(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
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
        pass
    
    # def __update_fake_weight(self):
    #     temp = self.raw_weight.data.max(dim=1, keepdim=False)
        
    #     self.fake_weight.data = torch.zeros_like(self.fake_weight.data)
    #     self.fake_weight[:][temp[:]] =1.
        
    #     self.fake_weight = self.fake_weight.requires_grad_()
            
    def get_one_hot_format(self)->torch.Tensor:
        index_of_max_o = self.raw_weight.max(dim=1).indices
        return index_of_max_o
    def get_index_format(self)->torch.Tensor:
        raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
        out_features_s = raw_weight.shape[0]
        out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
        #print(out_features_iota, "out_features_iota")
        index_of_max_o = raw_weight.max(dim=1).indices
        #print(index_of_max_o, "index_of_max_o")
        
        the_temp_o_i = torch.zeros_like(raw_weight)
        the_temp_o_i[out_features_iota_o, index_of_max_o] = 1.
        return the_temp_o_i
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not input.requires_grad):
            raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
        if len(input.shape)!=2:
            raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")
        
        if self.training_count<self.protect_param_every____training:
            self.training_count +=1
        else:
            self.training_count = 0

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
        
        w = self.gramo(self.raw_weight)
        x = DigitalMapperFunction_v2.apply(input, w)
        
        return x
   
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
    pass

# '''Param protection test. Real case vs equivalent individual code.'''
# layer = DigitalMapper_V2(3,8,protect_param_every____training=0)
# layer.raw_weight.data = torch.tensor([
#     [1., 2., 3.],[1., 3., 6.],[3., 6., 11.],[5., 10., 15.],
#     [-10., 2., 10.],[-1., -3., -6.],[-3., -6., -11.],[-5., -10., -15.],
#     ])
# layer(torch.tensor([[1., 2., 3.]], requires_grad=True))
# print(layer.raw_weight)
# fds=432
# '''Then ,the individual code.'''
# epi = 0.01
# threshold1 = 3.
# threshold2 = 7.
# margin = 1.

# target_exceed = threshold2-threshold1-margin
# test_weight = torch.tensor([
#     [1., 2., 3.],[1., 3., 6.],[3., 6., 11.],[5., 10., 15.],
#     [-10., 2., 10.],[-1., -3., -6.],[-3., -6., -11.],[-5., -10., -15.],
#     ])
# gt_t2 = test_weight.gt(threshold2)
# #print(gt_t2, "gt_t2")
# at_least_smt_gt_t2 = gt_t2.any(dim=1)
# #print(at_least_smt_gt_t2, "at_least_smt_gt_t2")
# if at_least_smt_gt_t2.any().item():
#     the_max_value = test_weight.max(dim=1, keepdim=True).values
#     #print(the_max_value, "the_max_value")
#     gt_t1 = test_weight.gt(threshold1)
#     #print(gt_t1, "gt_t1")
    
#     at_least_smt_gt_t2_expanded = at_least_smt_gt_t2[:, None].expand(-1, test_weight.shape[1])
#     #print(at_least_smt_gt_t2_expanded.shape, "at_least_smt_gt_t2_expanded.shape")
    
#     modify_these = gt_t1.logical_and(at_least_smt_gt_t2_expanded)
#     #print(modify_these, "modify_these")
    
#     exceed = the_max_value-threshold1
#     exceed = exceed.abs()+epi#or exceed.mul(at_least_smt_gt_t2)+epi
#     #print(exceed, "exceed")
    
#     mul_me = target_exceed/exceed
#     print(mul_me, "mul_me")
#     test_weight = modify_these*((test_weight-threshold1)*mul_me+threshold1)+(modify_these.logical_not())*test_weight
#     print(test_weight, "test_weight")
# pass
    
# #test_weight2 = test_weight.detach().clone()*-1
# #print(test_weight2, "test_weight2")
# lt_nt2 = test_weight.lt(-1.*threshold2)
# #print(lt_nt2, "lt_nt2")
# at_least_smt_lt_nt2 = lt_nt2.any(dim=1)
# #print(at_least_smt_lt_nt2, "at_least_smt_lt_nt2")
# if at_least_smt_lt_nt2.any().item():
#     the_min_value = test_weight.min(dim=1, keepdim=True).values
#     #print(the_min_value, "the_min_value")
#     lt_nt1 = test_weight.lt(-threshold1)
#     #print(lt_nt1, "lt_nt1")
    
#     at_least_smt_lt_nt2_expanded = at_least_smt_lt_nt2[:, None].expand(-1, test_weight.shape[1])
#     #print(at_least_smt_lt_nt2_expanded, "at_least_smt_lt_nt2_expanded")
#     #print(at_least_smt_lt_nt2_expanded.shape, "at_least_smt_lt_nt2_expanded.shape")
    
#     modify_these_negative = lt_nt1.logical_and(at_least_smt_lt_nt2_expanded)
#     #print(modify_these_negative, "modify_these_negative")
    
#     exceed = the_min_value+threshold1
#     exceed = exceed.abs()+epi#or exceed.mul(at_least_smt_gt_t2)+epi
#     #print(exceed, "exceed")
    
#     mul_me_negative = target_exceed/exceed
#     print(mul_me_negative, "mul_me_negative")
#     test_weight = modify_these_negative*((test_weight+threshold1)*mul_me_negative-threshold1) \
#         +(modify_these_negative.logical_not())*test_weight
#     print(test_weight, "test_weight")
# pass
# print("--------------SUMMARIZATION-----------")
# print(layer.raw_weight, "real code.")
# print(test_weight, "test code.")
    
# fds=432

class DigitalMapper_eval_only_v2(torch.nn.Module):
    def __init__(self):
        raise Exception("to do")

raise Exception("to do")



# class DigitalMapper_eval_only(torch.nn.Module):
#     r'''This class is not designed to be created by user directly. 
#     Use DigitalMapper.can_convert_into_eval_only_mode 
#     and DigitalMapper.convert_into_eval_only_mode to create this layer.
    
#     And, if I only provide result in this form, it's possible to make puzzles.
#     To solve the puzzle, figure the source of this layer.
#     '''
#     def __init__(self, in_features: int, out_features: int, indexes:torch.Tensor, \
#                     device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         if len(indexes.shape)!=1:
#             raise Exception("Param:indexes must be a rank-1 tensor. This class is not designed to be created by user directly. Use DigitalMapper.can_convert_into_eval_only_mode and DigitalMapper.convert_into_eval_only_mode to create this layer.")
#         self.indexes = torch.nn.Parameter(indexes, requires_grad=False)
#         self.indexes.requires_grad_(False)
#     pass
#     def accepts_non_standard_range(self)->bool:
#         return False
#     def outputs_standard_range(self)->bool:
#         return True    
#     def outputs_non_standard_range(self)->bool:
#         return not self.outputs_standard_range()
#     def forward(self, input:torch.Tensor):
#         x = input[:, self.indexes]
#         #print(x)
#         return x

