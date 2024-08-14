from typing import Any, List, Tuple, Optional, Self
import sys
import math
import torch
from util import debug_Rank_1_parameter_to_List_float, int_into_floats, floats_into_int
from util import data_gen_half_adder_1bit, data_gen_full_adder, bitwise_acc, data_gen_for_directly_stacking_test
from util import debug_strong_grad_ratio
from ParamMo import GradientModification
from Binarize import Binarize





class DigitalMapperFunction_v2_1(torch.autograd.Function):
    r'''
    forward input list:
    >>> x = args[0]# shape must be [batch, in_features]
    >>> raw_weight:torch.Tensor = args[1]# shape must be [out_features, in_features], must requires grad.
    backward input list:
    >>> g_in #shape of g_in must be [batch, out_features]
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x = args[0]# shape must be [batch, in_features]
        raw_weight:torch.Tensor = args[1]# shape must be [out_features, in_features]
                
        #print(out_features_iota, "out_features_iota")
        index_of_max_o = raw_weight.max(dim=1).indices
        #print(index_of_max_o, "index_of_max_o")
        output:torch.Tensor
        output = x[:, index_of_max_o]
        
        '''Because raw_weight always requires grad, but the output is 
        calculated with x[:, index_of_max_o], which is unlike other multiplication or
        addition, if any of the input tensor/parameter requires grad, the result requires grad.
        In this case, the output should always require grad, but the program doesn't infer 
        this from the "x[:, index_of_max_o]", it only inherits from the x.
        So, I have to manually modify it.
        '''
        output.requires_grad_()

        out_features_s = torch.tensor(raw_weight.shape[0], device=raw_weight.device)
        raw_weight_shape = torch.tensor(raw_weight.shape, device=raw_weight.device)
        x_needs_grad = torch.tensor([x.requires_grad], device=x.device)
        ctx.save_for_backward(x, index_of_max_o, out_features_s, raw_weight_shape, x_needs_grad)
        return output

    @staticmethod
    def backward(ctx, g_in_b_o):
        #shape of g_in must be [batch, out_features]
        
        
        x:torch.Tensor
        index_of_max_o:torch.Tensor
        out_features_s:torch.Tensor
        raw_weight_shape:torch.Tensor
        x_needs_grad:torch.Tensor
        x, index_of_max_o, out_features_s, raw_weight_shape, x_needs_grad = ctx.saved_tensors
            
        # print(g_in, "g_in", __line__str())
            
        input_reshaped_b_1_i = x[:,None,:]
        # print(input_reshaped_b_1_i, "input_reshaped_b_1_i")
        #print(input_reshaped_b_1_i.shape, "input_reshaped_b_1_i.shape")
        g_in_reshaped_b_o_1:torch.Tensor = g_in_b_o[:,:,None]
        #print(g_in_reshaped_b_o_1, "g_in_reshaped_b_o_1")
        #print(g_in_reshaped_b_o_1.shape, "g_in_reshaped_b_o_1.shape")
        grad_for_raw_weight_before_sum_b_o_i = g_in_reshaped_b_o_1.matmul(input_reshaped_b_1_i)
        # print(grad_for_raw_weight_before_sum_b_o_i, "grad_for_raw_weight_before_sum_b_o_i before sum")
        #print(grad_for_raw_weight_before_sum_b_o_i.shape, "grad_for_raw_weight_before_sum_b_o_i.shape before sum")
        grad_for_raw_weight__raw__o_i = grad_for_raw_weight_before_sum_b_o_i.sum(dim=0, keepdim=False)
        # print(grad_for_raw_weight__raw__o_i, "grad_for_raw_weight__raw__o_i after sum, the final result.")
        #print(grad_for_raw_weight__raw__o_i.shape, "grad_for_raw_weight__raw__o_i.shape. Should be ", raw_weight.shape)
        
        flag_neg_of_grad_for_raw_weight__raw__o_i = grad_for_raw_weight__raw__o_i.lt(0.)
        grad_for_raw_weight_o_i = flag_neg_of_grad_for_raw_weight__raw__o_i*grad_for_raw_weight__raw__o_i*3.+grad_for_raw_weight__raw__o_i

        if x_needs_grad.logical_not():
            return None, grad_for_raw_weight_o_i

        in_features_s = x.shape[1]
        out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)

        one_hot_o_i = torch.zeros(raw_weight_shape[0], raw_weight_shape[1], device=raw_weight_shape.device)
        one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
        # print(g_in, "g_in")
        # print(one_hot_o_i, "one_hot_o_i")
        one_hot_expanded_fake_b_o_i = one_hot_o_i.expand(g_in_b_o.shape[0], -1, -1)
        #print(one_hot_expanded_fake_b_o_i, "one_hot_expanded_fake_b_o_i")
        #print(one_hot_expanded_fake_b_o_i.shape, "one_hot_expanded_fake_b_o_i.shape")

        g_in_reshaped_expanded_b_o_fake_i = g_in_reshaped_b_o_1.expand(-1, -1, in_features_s)
        #print(g_in_reshaped_expanded_b_o_fake_i, "g_in_reshaped_expanded_b_o_fake_i")
        #print(g_in_reshaped_expanded_b_o_fake_i.shape, "g_in_reshaped_expanded_b_o_fake_i.shape")

        one_hot_mul_g_in_b_o_i = one_hot_expanded_fake_b_o_i.mul(g_in_reshaped_expanded_b_o_fake_i)
        # print(one_hot_mul_g_in_b_o_i, "one_hot_mul_g_in_b_o_i")

        grad_for_x_b_i = one_hot_mul_g_in_b_o_i.sum(dim=1, keepdim=False)
        # print(grad_for_x_b_i, "grad_for_x_b_i")
        #print(grad_for_x_b_i.shape, "grad_for_x_b_i.shape")
                
        return grad_for_x_b_i, grad_for_raw_weight_o_i

    pass  # class

# '''single layer grad backward test.'''
# x = torch.tensor([[1.11]])
# w = torch.tensor([[1.21], [1.22], ], requires_grad=True)
# pred:torch.Tensor = DigitalMapperFunction_v2_1.apply(x, w)
# print(pred, "pred")
# pred.backward(torch.tensor([[1.31, -1.32]]))
# print(w.grad, "w.grad")
# fds=432


# '''multi layer grad backward test.'''
# x = torch.tensor([[1.11]])
# w1 = torch.tensor([[1.21], [1.22], ], requires_grad=True)
# w2 = torch.tensor([[1.31, 1.32]], requires_grad=True)
# pred1 = DigitalMapperFunction_v2.apply(x, w1)
# print(pred1, "pred1")
# pred_final:torch.Tensor = DigitalMapperFunction_v2.apply(pred1, w2)
# print(pred_final, "pred_final")
# g_in = torch.ones_like(pred_final.detach())
# #torch.autograd.backward(pred_final, g_in, inputs=[w1, w2])
# pred_final.backward(g_in)
# print(w1.grad, "w1.grad")
# print(w2.grad, "w2.grad")
# fds=423


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

# one_hot_o_i = torch.zeros_like(raw_weight)
# one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
# one_hot_expanded_fake_b_o_i = one_hot_o_i.expand(g_in.shape[0], -1, -1)
# print(one_hot_expanded_fake_b_o_i.shape, "one_hot_expanded_fake_b_o_i.shape")
# print(one_hot_expanded_fake_b_o_i, "one_hot_expanded_fake_b_o_i")

# g_in_reshaped_expanded_b_o_fake_i = g_in_reshaped_b_o_1.expand(-1, -1, in_features_s)
# print(g_in_reshaped_expanded_b_o_fake_i, "g_in_reshaped_expanded_b_o_fake_i")
# print(g_in_reshaped_expanded_b_o_fake_i.shape, "g_in_reshaped_expanded_b_o_fake_i.shape")

# one_hot_mul_g_in_b_o_i = one_hot_expanded_fake_b_o_i.mul(g_in_reshaped_expanded_b_o_fake_i)
# print(one_hot_mul_g_in_b_o_i, "one_hot_mul_g_in_b_o_i")

# grad_for_x_b_i = one_hot_mul_g_in_b_o_i.sum(dim=1, keepdim=False)
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




class DigitalMapper_eval_only_v2(torch.nn.Module):
    r'''This class is not designed to be created by user directly. 
    Use DigitalMapper.can_convert_into_eval_only_mode 
    and DigitalMapper.convert_into_eval_only_mode to create this layer.
   
    And, if I only provide result in this form, it's possible to make puzzles.
    To solve the puzzle, figure the source of this layer.
    '''
    def __init__(self, in_features: int, indexes:torch.Tensor, \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if len(indexes.shape)!=1:
            raise Exception("Param:indexes must be a rank-1 tensor. This class is not designed to be created by user directly. Use DigitalMapper.can_convert_into_eval_only_mode and DigitalMapper.convert_into_eval_only_mode to create this layer.")
        if indexes.amax()>=in_features:
            raise Exception("it's definitely gonna have out of range bug.")
        self.indexes = torch.nn.Parameter(indexes, requires_grad=False)
        self.indexes.requires_grad_(False)
        self.in_features = in_features
    pass
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True    
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    def forward(self, input:torch.Tensor):
        x = input[:, self.indexes]
        #print(x)
        return x
    def extra_repr(self) -> str:
        return f'in_features:{self.in_features}, in_features:{self.indexes.shape[0]}.'
    
    pass #end of class.
        
    
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试
#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试#补测试







#scaling_ratio_for_gramo的默认值




class DigitalMapper_V2_1(torch.nn.Module):
    r'''This layer is designed to be used between digital layers.
    The input should be in STANDARD range so to provide meaningful output
    in STANDARD range. It works for both 01 and np styles.

    Notice: unlike most layers in this project, this layer is stateful.
    In other words, it has inner param in neural network path.

    Remember to concat a constant 0. and 1. to the input before sending into this layer.
    In other words, provide Vdd and Vss as signal to the chip.
    '''
    #__constants__ = []

    def __init__(self, in_features: int, out_features: int, needs_out_gramo = True, \
                    auto_print_difference:bool = False, \
                    scale_the_scaling_ratio_for_learning_gramo = 1., \
                    #protect_param_every____training:int = 5, \
                    device=None, dtype=None) -> None:   #, debug_info = ""

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if in_features<2:
            raise Exception("emmmm")

        self.in_features = in_features
        #self.log_of_in_features = torch.nn.Parameter(torch.log(torch.tensor([in_features])), requires_grad=False)
        self.out_features = out_features
        self.sqrt_of_out_features = torch.nn.Parameter(torch.sqrt(torch.tensor([out_features])), requires_grad=False)
        #self.out_iota = torch.nn.Parameter(torch.linspace(0,out_features-1, out_features, dtype=torch.int32), requires_grad=False)
        self.needs_out_gramo = needs_out_gramo

        self.raw_weight_o_i = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.__reset_parameters__the_plain_rand01_style()
     
        self.raw_weight_max = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.raw_weight_min = torch.nn.Parameter(torch.tensor([-1./100]), requires_grad=False)

        self.gramo_for_raw_weight = GradientModification()
        self.reset_scaling_ratio_for_raw_weight()
        self.scale_the_scaling_ratio_for_raw_weight(scale_the_scaling_ratio_for_learning_gramo)
        
        self.set_auto_print_difference_between_epochs(auto_print_difference)
        #self.out_binarize_does_NOT_need_gramo = Binarize.create_analog_to_np(needs_gramo=False)
        self.out_gramo = GradientModification()

        #to keep track of the training.
        # self.protect_param_every____training = protect_param_every____training
        # self.protect_param__training_count = 0
        pass

    def __reset_parameters__the_plain_rand01_style(self) -> None:
        '''copied from torch.nn.Linear'''
        self.raw_weight_o_i.data = torch.rand_like(self.raw_weight_o_i)# they should be <0.
        pass

    def accepts_non_standard_range(self)->str:
        return "although this layer accepts non standard input, I recommend you only feed standard +-1(np) as input."
    def outputs_standard_range(self)->str:
        return "It depends on what you feed in. If the input is standard +-1(np), the output is also standard +-1(np)."    
    # def outputs_non_standard_range(self)->bool:
    
    def reset_scaling_ratio_for_raw_weight(self):
        '''simply sets the inner'''
        #the *10 is conventional. If lr is 0.001, planned epoch is 100, the overall lr is 0.001*100.
        self.gramo_for_raw_weight.set_scaling_ratio((self.sqrt_of_out_features).item()*10.)
        pass
    def scale_the_scaling_ratio_for_raw_weight(self, by:float):
        '''simply sets the inner'''
        self.gramo_for_raw_weight.set_scaling_ratio((self.gramo_for_raw_weight.scaling_ratio*by).item())
        pass
    def set_scaling_ratio_for_raw_weight(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.gramo_for_raw_weight.set_scaling_ratio(scaling_ratio)
        pass


    def set_auto_print_difference_between_epochs(self, set_to:bool = True):
        with torch.no_grad():
            if not set_to:
                self.raw_weight_before = torch.nn.Parameter(torch.empty([0,], requires_grad=False))
                self.raw_weight_before.requires_grad_(False)
                # use self.raw_weight_before.nelement() == 0 to test it.
                pass
            if set_to:
                if self.raw_weight_o_i is None:
                    raise Exception("This needs self.raw_weight first. Report this bug to the author, thanks.")
                if self.raw_weight_o_i.nelement() == 0:
                    raise Exception("Unreachable code. self.raw_weight contains 0 element. It's so wrong.")
                #if not hasattr(self, "raw_weight_before") or self.raw_weight_before is None:
                if self.raw_weight_before is None:
                    self.raw_weight_before = torch.nn.Parameter(torch.empty_like(self.raw_weight_o_i), requires_grad=False)
                    self.raw_weight_before.requires_grad_(False)
                    pass
                self.raw_weight_before.data = self.raw_weight_o_i.detach().clone()
                pass
            pass



    def get_max_index(self)->torch.Tensor:
        with torch.no_grad():
            the_max_index = self.raw_weight_o_i.max(dim=1,keepdim=False).indices
            return the_max_index
    def get_one_hot_format(self)->torch.Tensor:
        with torch.no_grad():
            #raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
            out_features_s = self.raw_weight_o_i.shape[0]
            out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, device=self.raw_weight_o_i.device, dtype=torch.int32)
            #or arange?
            
            #print(out_features_iota, "out_features_iota")
            index_of_max_o = self.raw_weight_o_i.max(dim=1).indices
            #print(index_of_max_o, "index_of_max_o")

            one_hot_o_i = torch.zeros_like(self.raw_weight_o_i, device=self.raw_weight_o_i.device, dtype=self.raw_weight_o_i.dtype)
            one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
            return one_hot_o_i

    def debug_get_zero_grad_ratio(self, directly_print_out:float = False)->float:
        with torch.no_grad():
            result = 0.
            if not self.raw_weight_o_i.grad is None:
                flags = self.raw_weight_o_i.grad.eq(0.)
                total_amount = flags.sum().item()
                result = float(total_amount)/self.raw_weight_o_i.nelement()
            if directly_print_out:
                print("get_zero_grad_ratio:", result)
            return result



    def debug_print_param_overlap_ratio(self):
        with torch.no_grad():
            the_max_index = self.get_index_format()
            the_dtype = torch.int32
            if self.out_features<=1:
                print("Too few output, The overlapping ratio doesn't mean anything. __line__903")
            else:
                total_overlap_count = 0
                total_possible_count = self.in_features*(self.in_features-1)//2
                for i in range(self.in_features-1):
                    host_index = torch.tensor([i], dtype=the_dtype)
                    guest_index = torch.linspace(i+1, self.in_features-1,
                                            self.in_features-i-1, dtype=the_dtype)
                    flag_overlapped = the_max_index[guest_index].eq(the_max_index[host_index])
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("overlap_ratio:",
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.SIG_gate_count>0:
            pass
    #end of function.


    def forward(self, input:torch.Tensor)->torch.Tensor:
        if len(input.shape)!=2:
            raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")

        # raw weight changes???
        #use self.raw_weight_before.nelement() == 0 to test it.
        if self.raw_weight_before.nelement() != 0:
            ne_flag = self.raw_weight_before.data.ne(self.raw_weight_o_i)
            nan_inf_flag = self.raw_weight_before.data.isnan().logical_and(self.raw_weight_before.data.isinf())
            report_these_flag = ne_flag.logical_and(nan_inf_flag.logical_not())
            if report_these_flag.any()>0:
                to_report_from = self.raw_weight_before[report_these_flag]
                to_report_from = to_report_from[:16]
                to_report_to = self.raw_weight_o_i[report_these_flag]
                to_report_to = to_report_to[:16]
                line_number_info = "    Line number: "+str(sys._getframe(1).f_lineno)
                print("Raw weight changed, from:\n", to_report_from, ">>>to>>>\n",
                        to_report_to, line_number_info)
            else:
                print("Raw weight was not changed in the last stepping")
                pass
            self.raw_weight_before.data = self.raw_weight_o_i.detach().clone()
            pass


        if self.training:
            with torch.no_grad():
                self.protect_raw_weight()
                pass

            x = input
            w_after_gramo = self.gramo_for_raw_weight(self.raw_weight_o_i)
            x = DigitalMapperFunction_v2_1.apply(x, w_after_gramo)

#            x = self.out_binarize_does_NOT_need_gramo(x)
            if self.needs_out_gramo:
                x = self.out_gramo(x)
                pass
            
            if torch.isnan(x).any():
                fds = 432
                pass
            return x
        else:#eval mode.
            the_max_indexes = self.get_max_index()
            x = input[:, the_max_indexes]
            return x            
            
    #end of function.


    def extra_repr(self) -> str:
        #return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'
        return f'In_features={self.in_features}, out_features={self.out_features}'



    def get_eval_only(self)->DigitalMapper_eval_only_v2:
        result = DigitalMapper_eval_only_v2(self.in_features, self.get_max_index())
        #raise Exception("untested")
        return result
    
    # def after_step(self, print_out_level = 0):
    #     '''the algorithm:
    #     the top raw weight element is a, the exp(a) is A
    #     the exp(...) or all other sums up to B
    #     so:
    #     A == k*(A+B), where k is the softmax result of the top element.
    #     (I'm not going to figure k out.)

    #     Then, I need a new A' to make the k back to 0.52, it's like:
    #     A' == 0.52*(A'+B)
    #     B is known,
    #     0.48A' == 0.52B
    #     A' == 1.08B
    #     then, log(A') is what I need.

    #     The code is like, i manually calc ed the softmax, so I can access the intermidiate result.
    #     The max index of the raw weight is also of the cooked weight.
    #     By cooking, I mean the softmax operation.
    #     '''
    #     with torch.no_grad():
    #         #self.anti_nan_for_raw_weight()
    #         self.protect_raw_weight()
    #         pass
    #     pass
    # #end of function

    def protect_raw_weight(self, report_nan_and_inf = False):
        '''
        Call this before evaluate tha accuracy.
        Moves everything between 1 and -0.01
        '''
        with torch.no_grad():
            the_device = self.raw_weight_o_i.device
            the_dtype = self.raw_weight_o_i.dtype
            
            debug_self_raw_weight = self.raw_weight_o_i
            
            #step 1, get the flags, then nan to num.
            flag_pos_inf = self.raw_weight_o_i.isposinf()
            flag_nan_and_too_small = self.raw_weight_o_i.isnan().logical_or(self.raw_weight_o_i.lt(0.))
            if report_nan_and_inf:
                nan_count = self.raw_weight_o_i.isnan().sum()
                pos_inf_count = self.raw_weight_o_i.isposinf().sum()
                neg_inf_count = self.raw_weight_o_i.isneginf().sum()
                if nan_count>0 or pos_inf_count>0 or neg_inf_count>0 :
                    fds=432
                    pass
                pass
            self.raw_weight_o_i.data.nan_to_num_(0.)
            
            #step 2, too big back into the max boundary.
            self.raw_weight_o_i.data = flag_pos_inf*self.raw_weight_max+flag_pos_inf.logical_not()*self.raw_weight_o_i
            
            #step 3, too small and nan, nan are treated as too small. 
            if flag_nan_and_too_small.any():
                print(flag_nan_and_too_small.sum().item(), "  <- elements of raw_weight became nan or neg inf.  Probably the scaling_ratio of gramo is too big.  __line 1113")
                pass
            #'''self.raw_weight_min == -30.-self.log_of_in_features'''
            self.raw_weight_o_i.data = flag_nan_and_too_small*(self.raw_weight_min*torch.rand_like(self.raw_weight_o_i, dtype=the_dtype, device=the_device))+flag_nan_and_too_small.logical_not()*self.raw_weight_o_i

            #step 4, offset to prevent too big element.
            #a = self.raw_weight.max(dim=1,keepdim=True).values
            offset = self.raw_weight_max - self.raw_weight_o_i.max(dim=1,keepdim=True).values
            #offset = offset.maximum(torch.tensor([0.], device=the_device, dtype=the_dtype))# don't do this.
            offset = offset.to(self.raw_weight_o_i.device).to(self.raw_weight_o_i.dtype)
            self.raw_weight_o_i.data = self.raw_weight_o_i + offset
            pass
        pass
    # end of function.

    def get_strong_grad_ratio(self, log10_diff = 0., \
            epi_for_w = 0.01, epi_for_g = 0.01)->float:
        result = debug_strong_grad_ratio(self.raw_weight_o_i, log10_diff, epi_for_w, epi_for_g)
        return result

    pass
fast_traval____end_of_digital_mapper_layer_class = 432

# '''Param protection test.'''
# layer = DigitalMapper_V2_1(3,9,needs_out_gramo=False)
# #print(layer.raw_weight_o_i.shape)
# layer.raw_weight_o_i.data = torch.tensor([
#     [1.,0.,torch.inf],[1.,0.,torch.inf*-1.],[1.,0.,torch.nan],
#     [1.,0.,99],[1.,0.,-99.],[1.,0.,0.],
#     [3.,4.,5],[-3,-4,-5,],[11,0,-11]])
# #print(layer.raw_weight_o_i.shape)
# print(layer.raw_weight_o_i)
# layer.protect_raw_weight()
# print(layer.raw_weight_o_i)
# layer.protect_raw_weight()
# print(layer.raw_weight_o_i)
# fds=432


# '''Test for all the modes, and eval only layer.'''
# print("All 5 prints below should be equal.")
# x = torch.tensor([[5., 6., 7.], [8., 11., 12], ], requires_grad=True)
# layer = DigitalMapper_V2_1(3,4, needs_out_gramo=False)
# print(layer(x))
# layer.eval()
# print(layer(x))
# print(x[:,layer.get_max_index()])
# print(layer.get_one_hot_format().matmul(x[:,:,None]).squeeze(dim=-1))
# fds=432
# eval_only_layer = layer.get_eval_only()
# print(eval_only_layer(x))
# print("All 4 prints above should be equal.")
# fds=432


# '''basic test. Also, the eval mode.'''
# layer = DigitalMapper_V2_1(2,3, needs_out_gramo=False)
# layer.scale_the_scaling_ratio_for_raw_weight(1.)
# # print(layer.raw_weight.data.shape)
# layer.raw_weight_o_i.data=torch.Tensor([[1., 0.5],[ 0.1, 0.],[ 0.5, 1]])
# layer.raw_weight_o_i.requires_grad_(True)
# print(layer.raw_weight_o_i.requires_grad)
# # print(layer.raw_weight.data)
# input = torch.tensor([[1., -1.]], requires_grad=True)
# pred:torch.Tensor = layer(input)
# print(pred, "should be 1 1 -1")
# pred.backward(torch.tensor([[1.1, -1.2, 1.3]])*1.)
# print(input.grad)
# print(layer.raw_weight_o_i.grad)
# fds=432


fast_traval____training_test_for_digital_mapper = 432
# '''some real training'''
# input = torch.Tensor([[1., 1.],[1., -1.],[-1., 1.],[-1., -1.],])
# target = torch.Tensor([[1.],[1.],[-1.],[-1.],])
# # print(input)
# # print(target)

# model = DigitalMapper_V2_1(2,1)
# #print(model.raw_weight_o_i.shape)
# model.raw_weight_o_i.data = torch.tensor([[0.1, 0.9]])
# #model.scale_the_scaling_ratio_for_raw_weight(10000.)
# # for p in model.parameters():
# #     print(p)
# loss_function = torch.nn.MSELoss()
# #loss_function = torch.nn.L1Loss()???
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# iter_per_print = 1#1111
# print_count = 555555
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     if False and "shape":
#         print(pred.shape, "pred.shape")
#         print(target.shape, "target.shape")
#         fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred[:5], "pred")
#             print(target[:5], "target")
#             pass
#         pass
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.raw_weight_o_i.grad, "   grad")
#             pass
#         pass
#     if True and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             layer = model
#             print(layer.raw_weight_o_i, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight_o_i, "first_layer.in_mapper   after update")
#             pass    
#         pass    
    
#     if True and "print strong grad ratio":
#         if epoch%iter_per_print == iter_per_print-1:
#             result = model.get_strong_grad_ratio()
#             print("print strong grad ratio: ", result)
#             pass
#         pass
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
   
#     with torch.inference_mode():
#         model.eval()
#         pred = model(input)
#         #print(pred, "pred", __line__str())
#         #print(target, "target")
#         acc = bitwise_acc(pred, target)
#         if 1. != acc:
#             print(epoch+1, "    ep/acc    ", acc)
#         else:
#             print("FINISHED, ep:", epoch+1)
#             print("FINISHED, ep:", epoch+1)
#             print("FINISHED, ep:", epoch+1)
#             print(pred[:5].T, "pred", "    __line 788")
#             print(target[:5].T, "target")
#             break
#         pass
    
#     pass

# fds=432





'''Dry stack test. OK, the dry is actually a Chinese word, which means, only, or directly.'''
class test_directly_stacking_multiple_digital_mappers(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_width: int, layer_count:int, \
            auto_print_difference:bool = False, \
            device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if layer_count<2:
            raise Exception("emmmm..")
        
        self.in_features = in_features
        self.out_features = out_features
        self.mid_width = mid_width
        
        self.digital_mappers = torch.nn.ParameterList([])
        self.digital_mappers.append(DigitalMapper_V2_1(in_features,mid_width, 
                                        auto_print_difference = auto_print_difference))
        for _ in range(num_layers-2):# I know what you are thinking. I know it.
            self.digital_mappers.append(DigitalMapper_V2_1(mid_width,mid_width, 
                                        auto_print_difference = auto_print_difference))
        self.digital_mappers.append(DigitalMapper_V2_1(mid_width,out_features, 
                                        auto_print_difference = auto_print_difference))
        pass

    def reset_scaling_ratio_for_raw_weight(self):
        layer:DigitalMapper_V2_1
        for layer in self.digital_mappers:
            layer.reset_scaling_ratio_for_raw_weight()
            pass
        pass
    def scale_the_scaling_ratio_for_raw_weight(self, by:float):
        layer:DigitalMapper_V2_1
        for layer in self.digital_mappers:
            layer.scale_the_scaling_ratio_for_raw_weight(by)
            pass
        pass
    def print_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01):
        temp_list:List[float] = []
        layer:DigitalMapper_V2_1
        for layer in self.digital_mappers:
            temp_list.append(layer.get_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g))
            pass
        pass
        print("debug_strong_grad_ratio: ", end="")
        for item in temp_list:
            print(f'{item:.3f}', end=", ")
            pass
        print()
        pass    
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        x = input
        for layer in self.digital_mappers:
            x = layer(x)
        return x
    pass


fast_traval____dry_stack_test = 432
batch = 50
in_features = 30
out_features = 20
mid_width = 70
num_layers = 8
lr = 0.001

iter_per_print = 1#1111
print_count = 2000011

(input, target) = data_gen_for_directly_stacking_test(batch, in_features, out_features)

model = test_directly_stacking_multiple_digital_mappers(in_features, out_features, mid_width, num_layers)
model.scale_the_scaling_ratio_for_raw_weight(3.)
if False and "print parameters":
    if True and "only the training params":
        for p in model.parameters():
            if p.requires_grad:
                print(p)
                pass
            pass
        pass
    else:# prints all the params.
        for p in model.parameters():
            print(p)
            pass
        pass
    
loss_function = torch.nn.MSELoss()
#loss_function = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model.cuda()
input = input.cuda()
target = target.cuda()
for epoch in range(iter_per_print*print_count):
    model.train()
    pred = model(input)
    #print(pred, "pred", __line__str())
    if False and "shape":
        print(pred.shape, "pred.shape")
        print(target.shape, "target.shape")
        fds=423
    if False and "print pred":
        if epoch%iter_per_print == iter_per_print-1:
            print(pred[:5], "pred")
            print(target[:5], "target")
            pass
        pass
    loss = loss_function(pred, target)
    optimizer.zero_grad()
    loss.backward()
    if False and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            print(model.digital_mappers[0].raw_weight_o_i.grad, "0th layer    grad")
            print(model.digital_mappers[1].raw_weight_o_i.grad, "1 layer    grad")
            print(model.digital_mappers[-1].raw_weight_o_i.grad, "-1 layer    grad")
            pass
        pass
    if True and "print the weight":
        layer:DigitalMapper_V2_1 = model.digital_mappers[0]
        print(layer.raw_weight_o_i[:2,:7], "first_layer   before update")
        optimizer.step()
        print(layer.raw_weight_o_i[:2,:7], "first_layer   after update")
        
        layer = model.digital_mappers[1]
        print(layer.raw_weight_o_i[:2,:7], "second   before update")
        optimizer.step()
        print(layer.raw_weight_o_i[:2,:7], "second   after update")
        
        layer = model.digital_mappers[-1]
        print(layer.raw_weight_o_i[:2,:7], "last   before update")
        optimizer.step()
        print(layer.raw_weight_o_i[:2,:7], "last   after update")
        pass    
    if True and "print strong grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            result = model.print_strong_grad_ratio()
            print("print strong grad ratio: ", result)
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    optimizer.step()
    # if True and "print param overlap":
    #     every = 100
    #     if epoch%every == every-1:
    #         model.print_param_overlap_ratio()
    #         pass
    #     pass
    if True and "print acc":
        with torch.inference_mode():
            model.eval()
            pred = model(input)
            acc = bitwise_acc(pred, target)
            if 1. != acc:
                print(epoch+1, "    ep/acc    ", acc)
            else:
                print("FINISHED, ep:", epoch+1)
                print("FINISHED, ep:", epoch+1)
                print("FINISHED, ep:", epoch+1)
                print(pred[:2,:7].T, "pred", "    __line 888")
                print(target[:2,:7].T, "target")
                break
            pass
        pass

fds=432




























