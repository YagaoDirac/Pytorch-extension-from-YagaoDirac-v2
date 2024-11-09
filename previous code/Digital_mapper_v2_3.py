from typing import Any, List, Tuple, Optional, Self
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


#from util import debug_Rank_1_parameter_to_List_float, int_into_floats, floats_into_int
from pytorch_yagaodirac_v2.Util import bitwise_acc, bitwise_acc_with_str, data_gen_for_directly_stacking_test
from pytorch_yagaodirac_v2.Util import data_gen_for_directly_stacking_test_same_dim_no_duplicated
from pytorch_yagaodirac_v2.Util import debug_strong_grad_ratio
from pytorch_yagaodirac_v2.Util import Print_Timing
from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2
#from Binarize import Binarize



# 回传的东西，如果是0，那么表示没有被后面的层选用。
# 如果不是0，符号表示后面需要的正确答案，绝对值表示强度。
# target是答案乘以强度，直接用autograd.backward给进去，而不用loss.backward()。
# 自定义回传的时候，顺着主梯度链传的其实是答案和对应强度，而不是梯度。
# 而w得到的是梯度，那个地方用了乘法，其实是，当x和答案（g）一致的时候，是对的，要强化。不一致表示错误，要弱化。
# 乘以-1是因为，pytorch的更新是减去梯度。神经网络都这么干的。

# 改进空间。
# 因为整个体系的数字是很普通的，不需要真的用gramo去保护，可以另外单独写一个东西，效率还有可能更高。


class DigitalMapperFunction_v2_3(torch.autograd.Function):
    r'''
    forward input list:
    >>> x = args[0]# shape must be [batch, in_features]
    >>> raw_weight:torch.Tensor = args[1]# shape must be [out_features, in_features], must requires grad.
    backward input list:
    >>> g_in #shape of g_in must be [batch, out_features]
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        input_b_i:torch.Tensor = args[0]# shape must be [batch, in_features]
        raw_weight:torch.Tensor = args[1]# shape must be [out_features, in_features]
                
        index_of_max_o = raw_weight.max(dim=1).indices
        output:torch.Tensor = input_b_i[:, index_of_max_o]
        '''Because raw_weight always requires grad, but the output is 
        calculated with x[:, index_of_max_o], which is unlike other multiplication or
        addition, if any of the input tensor/parameter requires grad, the result requires grad.
        In this case, the output should always require grad, but the program doesn't infer 
        this from the "x[:, index_of_max_o]", it only inherits from the x.
        So, I have to manually modify it.
        '''
        output.requires_grad_(input_b_i.requires_grad and raw_weight.requires_grad)
        w_requires_grad = torch.tensor([raw_weight.requires_grad])
        ctx.save_for_backward(input_b_i, index_of_max_o, w_requires_grad)
        return output

    @staticmethod
    def backward(ctx, g_in_b_o):
        #shape of g_in must be [batch, out_features]
        input_b_i:torch.Tensor
        index_of_max_o:torch.Tensor
        w_requires_grad:torch.Tensor
        
        (input_b_i, index_of_max_o, w_requires_grad) = ctx.saved_tensors
            
        grad_for_x_b_i:Tuple[torch.tensor|None] = None
        grad_for_raw_weight_o_i:Tuple[torch.tensor|None] = None
        
        input_requires_grad = torch.tensor([input_b_i.requires_grad], device=input_b_i.device)
        
        if w_requires_grad:
            g_in_reshaped_b_o_1:torch.Tensor = g_in_b_o[:,:,None]
            input_reshaped_b_1_i:torch.Tensor = input_b_i[:,None,:]
            
            grad_for_raw_weight__before_sum__b_o_i:torch.Tensor = g_in_reshaped_b_o_1*(input_reshaped_b_1_i*-1)
            #print(grad_for_raw_weight__before_sum__b_o_i)
            grad_for_raw_weight_o_i = grad_for_raw_weight__before_sum__b_o_i.sum(dim=0, keepdim=False)
            #print(grad_for_raw_weight_o_i)
            pass
        
        if input_requires_grad:
            batch_size_s = g_in_b_o.shape[0]
            out_features_s = g_in_b_o.shape[1]
            in_features_s = input_b_i.shape[1]
            
            batch_features_iota_o = torch.linspace(0, batch_size_s-1, batch_size_s, dtype=torch.int32).reshape([-1,1])
            batch_features_iota_expanded_o_mul_fake_i = batch_features_iota_o.expand([-1, out_features_s])
            batch_features_iota_expanded_o_mul_fake_i = batch_features_iota_expanded_o_mul_fake_i.reshape([-1])
            
            out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32).reshape([1,-1])
            out_features_iota_expanded_fake_o_mul_i = out_features_iota_o.expand([batch_size_s, -1])
            out_features_iota_expanded_fake_o_mul_i = out_features_iota_expanded_fake_o_mul_i.reshape([-1])
            
            index_of_max_expanded_fake_o_mul_i = index_of_max_o.reshape([1,-1]).expand([batch_size_s, -1])
            index_of_max_expanded_fake_o_mul_i = index_of_max_expanded_fake_o_mul_i.reshape([-1])
            
            grad_for_x__before_sum__b_o_i = torch.zeros([batch_size_s, out_features_s, in_features_s], dtype=input_b_i.dtype, device=input_b_i.device)
            grad_for_x__before_sum__b_o_i[batch_features_iota_expanded_o_mul_fake_i,out_features_iota_expanded_fake_o_mul_i,\
                index_of_max_expanded_fake_o_mul_i] = g_in_b_o[batch_features_iota_expanded_o_mul_fake_i,out_features_iota_expanded_fake_o_mul_i]
            #print(grad_for_x__before_sum__b_o_i)
            
            grad_for_x_b_i = grad_for_x__before_sum__b_o_i.sum(dim=1, keepdim=False)
            #print(grad_for_x_b_i)
            pass
            
        return grad_for_x_b_i, grad_for_raw_weight_o_i

    pass  # class

if '''main check.''' and True:
    b=2
    o=3
    i=5
    #x = torch.rand([b,i], requires_grad=True)
    #x = torch.tensor([[11.,12,13,14,15],[21,22,23,24,25]], requires_grad=True)
    x = torch.tensor([[1.,-1,1,1,1,],[1.,1,1,1,1,],], requires_grad=True)
    #w = torch.rand([o,i], requires_grad=True)
    w = torch.tensor([[0.,0,0,0,0.1],[0.,0,0.1,0,0],[0.,0.1,0,0,0],], requires_grad=True)
    #w = torch.tensor([[0.,0.1,0,0,0],[0.,0.1,0,0,0],[0.,0.1,0,0,0],], requires_grad=True)
    pred:torch.Tensor = DigitalMapperFunction_v2_3.apply(x, w)
    #g = torch.tensor([[111.,112,113],[121,122,123]])
    g = torch.tensor([[1001.,1002,1003],[1004,1005,1006]])
    pred.backward(g)#torch.tensor([[1.31, -1.32]]))
    
    # print(x.shape == x.grad.shape)
    # print(w.shape == w.grad.shape)
    print(x.grad, "x.grad is the correct answer with strength")
    print(w.grad, "w.grad, neg for correct mapping")
    pass

if '''individual element check.''' and False:
    b=2
    o=3
    i=5
    x = torch.tensor([[1.,-1.,1.,1.,1.],[1.,1.,1.,1.,1.],], requires_grad=True)
    w = torch.tensor([[0.,0.,0.,0.,0.1],[0.,0.,0.1,0.,0.],[0.,0.1,0.,0.,0.],], requires_grad=True)
    g = torch.tensor([[11.,-12.,13.],[21.,22.,23.]])
    pred:torch.Tensor = DigitalMapperFunction_v2_3.apply(x, w)
    pred.backward(g)
    print(x.grad, "x.grad is the correct answer with strength")
    print(w.grad, "w.grad, neg for correct mapping")
    pass
    


class DigitalMapper_eval_only_v2(torch.nn.Module):
    r'''This class is not designed to be created by user directly. 
    Use DigitalMapper.can_convert_into_eval_only_mode 
    and DigitalMapper.convert_into_eval_only_mode to create this layer.
   
    And, if I only provide result in this form, it's possible to make puzzles.
    To solve the puzzle, figure the source of this layer.
    '''
    def __init__(self, in_features: int, indexes:torch.Tensor, \
                    device=None, dtype=None) -> None:
        raise Exception ("not tested.")
        
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
        


class DigitalMapper_V2_3(torch.nn.Module):
    r'''This layer is designed to be used between digital layers.
    The input should be in STANDARD range so to provide meaningful output
    in STANDARD range. It works for both 01 and np styles.

    Notice: unlike most layers in this project, this layer is stateful.
    In other words, it has inner param in neural network path.

    Remember to concat a constant 0. and 1. to the input before sending into this layer.
    In other words, provide Vdd and Vss as signal to the chip.
    '''
    #__constants__ = []

    def __init__(self, in_features: int, out_features: int, 
                    #is_out_mapper_in_DSPU:bool, \
                 #needs_out_gramo = True, \
                    # auto_print_difference:bool = False, \
                    # out_gramo_scale_factor = 1., \
                    # gramo_for_raw_weight_scale_factor = 1., \
                    #protect_param_every____training:int = 5, \
                    debug_allow_any_shape = False, \
                    number_in_model = -1, \
                    device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if in_features<2 and not debug_allow_any_shape:
            raise Exception('If this is intentional, search "if statement in python" on google.com.')

        self.number_in_model = number_in_model

        self.in_features = in_features
        #self.log_of_in_features = torch.nn.Parameter(torch.log(torch.tensor([in_features])), requires_grad=False)
        self.out_features = out_features
        #self.sqrt_of_out_features = torch.nn.Parameter(torch.sqrt(torch.tensor([out_features])), requires_grad=False)
        #self.out_iota = torch.nn.Parameter(torch.linspace(0,out_features-1, out_features, dtype=torch.int32), requires_grad=False)
        #self.is_out_mapper_in_DSPU = torch.nn.Parameter(torch.tensor([is_out_mapper_in_DSPU]), requires_grad=False)
        #self.set_epoch_factor(epoch_factor)
        if out_features>in_features and not debug_allow_any_shape:
            raise Exception("out dim must be <= in dim. For debug purpose, set debug_allow_any_shape=True.")

        if in_features == out_features:
            self.raw_weight_o_i = torch.nn.Parameter(torch.empty((0), **factory_kwargs))
        else:
            self.raw_weight_o_i = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            self.__reset_parameters__the_plain_rand01_style()
            pass
     
        self.raw_weight_max = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.raw_weight_min = torch.nn.Parameter(torch.tensor([-1./100]), requires_grad=False)

        self.gramo_for_raw_weight = GradientModification_v2()
        #self.reset_scaling_ratio_for_raw_weight()
        #self.scale_the_scaling_ratio_for_raw_weight(scale_the_scaling_ratio_for_learning_gramo)
        
        #self.raw_weight_before:Tuple[torch.nn.parameter.Parameter|None] = None
        #self.set_auto_print_difference_between_epochs(auto_print_difference)
        #self.out_binarize_does_NOT_need_gramo = Binarize.create_analog_to_np(needs_gramo=False)
        self.out_gramo = GradientModification_v2()

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
    
    def ____a() :
        '''
    # def outputs_non_standard_range(self)->bool:
    
    # def reset_scaling_ratio_for_raw_weight(self):
    #     ''simply sets the inner''
    #     #the *10 is conventional. If lr is 0.001, planned epoch is 100, the overall lr is 0.001*100.
    #     self.gramo_for_raw_weight.set_scaling_ratio(self.sqrt_of_out_features*10.)
    #     pass
    # def scale_the_scaling_ratio_for_raw_weight(self, by:float):
    #     ''simply sets the inner''
    #     self.gramo_for_raw_weight.set_scaling_ratio((self.gramo_for_raw_weight.scaling_ratio*by).item())
    #     pass
    # def set_scaling_ratio_for_raw_weight(self, scaling_ratio:float):
    #     ''simply sets the inner''
    #     self.gramo_for_raw_weight.set_scaling_ratio(scaling_ratio)
    #     pass

    # def set_epoch_factor(self, epoch_factor:float):
    #     self.epoch_factor = torch.nn.Parameter(torch.tensor([epoch_factor]), requires_grad=False)
    #     pass
    

    # def __________set_auto_print_difference_between_epochs(self, set_to:bool = True):
    #     with torch.no_grad():
    #         if not set_to:
    #             self.raw_weight_before = torch.nn.Parameter(torch.empty([0,], requires_grad=False))
    #             self.raw_weight_before.requires_grad_(False)
    #             # use self.raw_weight_before.nelement() == 0 to test it.
    #             pass
    #         if set_to:
    #             if self.raw_weight_o_i is None:
    #                 raise Exception("This needs self.raw_weight first. Report this bug to the author, thanks.")
    #             if self.raw_weight_o_i.nelement() == 0:
    #                 raise Exception("Unreachable code. self.raw_weight contains 0 element. It's so wrong.")
    #             #if not hasattr(self, "raw_weight_before") or self.raw_weight_before is None:
    #             if self.raw_weight_before is None:
    #                 self.raw_weight_before = torch.nn.Parameter(torch.empty_like(self.raw_weight_o_i), requires_grad=False)
    #                 self.raw_weight_before.requires_grad_(False)
    #                 pass
    #             self.raw_weight_before.data = self.raw_weight_o_i.detach().clone()
    #             pass
    #         pass
        '''

    def get_eval_only(self)->Tuple[DigitalMapper_eval_only_v2|None]:
        if self.out_features == self.in_features:
            raise Exception("untested")
            return None
        result = DigitalMapper_eval_only_v2(self.in_features, self.get_max_index())
        #raise Exception("untested")
        return result
    def get_max_index(self)->torch.Tensor:
        if self.out_features == self.in_features:
            raise Exception("untested")
            return torch.linspace(0,self.in_features-1,self.in_features,dtype=torch.int64)
            
        with torch.no_grad():
            
            the_max_index = self.raw_weight_o_i.max(dim=1,keepdim=False).indices
            return the_max_index
    def _________get_one_hot_format(self)->torch.Tensor:
        if self.out_features == self.in_features:
            raise Exception("untested")
        with torch.no_grad():
            raise Exception("untested")
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
            raise Exception("untested")
            result = 0.
            if not self.raw_weight_o_i.grad is None:
                flags = self.raw_weight_o_i.grad.eq(0.)
                total_amount = flags.sum().item()
                result = float(total_amount)/self.raw_weight_o_i.nelement()
            if directly_print_out:
                print("get_zero_grad_ratio:", result)
            return result

    def debug_print_param_overlap_ratio(self):
        if self.out_features == self.in_features:
            raise Exception("untested")
            print("overlap_ratio: 0, 0/{self.in_features}")
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
        '''
        # raw weight changes???
        #use self.raw_weight_before.nelement() == 0 to test it.
        # if self.raw_weight_before.nelement() != 0:
        #     ne_flag = self.raw_weight_before.data.ne(self.raw_weight_o_i)
        #     nan_inf_flag = self.raw_weight_before.data.isnan().logical_and(self.raw_weight_before.data.isinf())
        #     report_these_flag = ne_flag.logical_and(nan_inf_flag.logical_not())
        #     if report_these_flag.any()>0:
        #         to_report_from = self.raw_weight_before[report_these_flag]
        #         to_report_from = to_report_from[:16]
        #         to_report_to = self.raw_weight_o_i[report_these_flag]
        #         to_report_to = to_report_to[:16]
        #         line_number_info = "    Line number: "+str(sys._getframe(1).f_lineno)
        #         print("Raw weight changed, from:\n", to_report_from, ">>>to>>>\n",
        #                 to_report_to, line_number_info)
        #     else:
        #         print("Raw weight was not changed in the last stepping")
        #         pass
        #     self.raw_weight_before.data = self.raw_weight_o_i.detach().clone()
        #     pass
        '''

        if self.out_features == self.in_features:
            return input

        if self.training:
            with torch.no_grad():
                self.protect_raw_weight()
                pass
            
            
            x = input
            
            #w_after_gramo = self.gramo_for_raw_weight(self.raw_weight_o_i.view([1, -1])).view([self.out_features,self.in_features])
            
            #this shape is intentional.
            w_after_gramo = self.gramo_for_raw_weight(self.raw_weight_o_i)
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            #测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下#测一下
            
            x = DigitalMapperFunction_v2_3.apply(x, w_after_gramo)

            #x = self.out_binarize_does_NOT_need_gramo(x)
            x = self.out_gramo(x)
            
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

    def deduplicate(self, chosen_index:Tuple[torch.Tensor|None] = None)->torch.Tensor:
        # if chosen_index is None, this is the last layer, or all layers after this layer are all square.
        # square digital mapper doesn't do anything, and simply pass the chosen_index to previous layer.
        if self.out_features == self.in_features:
            return chosen_index
        if chosen_index is None:
            self.__deduplicate_all()
            result = self.get_max_index().unique()
            return result
        if chosen_index.shape[0] == 0:
            return chosen_index
        with torch.no_grad():
            max_index = self.get_max_index()
            buff = torch.empty([chosen_index.shape[0]], device=self.raw_weight_o_i.device, dtype=max_index.dtype)
            buff.fill_(-1)
            buff[0] = max_index[chosen_index[0]]
            len_s = torch.tensor(1, device=self.raw_weight_o_i.device)
            for i in range(1,chosen_index.shape[0]):
                chosen_index_now_s:torch.Tensor = chosen_index[i]
                max_index_now_s = max_index[chosen_index_now_s]
                flag = buff[:i].eq(max_index_now_s)
                if flag.any():
                    self.raw_weight_o_i.data[chosen_index_now_s,max_index_now_s] = -1.
                else:
                    buff[len_s] = max_index_now_s
                    len_s+=1
                    pass
                pass#for 
            new_max = self.get_max_index()
            new_max_chosen = new_max[chosen_index]
            result = new_max_chosen.unique()
            return result
        #end of function
        
    def __deduplicate_all(self):
        if self.out_features == self.in_features:
            raise Exception("untested")
            return chosen_index
        with torch.no_grad():
            max_index = self.get_max_index()
            buff = torch.empty([max_index.shape[0]], device=self.raw_weight_o_i.device, dtype=max_index.dtype)
            buff.fill_(-1)
            buff[0] = max_index[0]
            len = torch.tensor([1])
            for i in range(1,max_index.shape[0]):
                flag = buff[:i].eq(max_index[i])
                if flag.any():
                    self.raw_weight_o_i.data[i,max_index[i]] = -1.
                else:
                    buff[len.item()] = max_index[i]
                    len+=1
                    pass
                pass
            return
        #end of function

    def protect_raw_weight(self, report_nan_and_inf = False):
        '''
        Call this before evaluate tha accuracy.
        Moves everything between 1 and -0.01
        '''
        with torch.no_grad():
            the_device = self.raw_weight_o_i.device
            the_dtype = self.raw_weight_o_i.dtype
            
            #step 1, get the flags, then nan to num.
            #flag_inf = self.raw_weight_o_i.isinf()
            #flag_nan = self.raw_weight_o_i.isnan()
            # flag_nan_and_TOO_MUCH_small = self.raw_weight_o_i.isnan().logical_or(self.raw_weight_o_i.lt(-10.))
            # if flag_nan_and_TOO_MUCH_small.any():
            #     print(flag_nan_and_TOO_MUCH_small.sum().item(), "  <- elements of raw_weight became nan or neg inf.  Probably the scaling_ratio of gramo is too big.  __line 1113")
            #     print(self.raw_weight_o_i[flag_nan_and_TOO_MUCH_small], " <- nan_and_too_small")
            #     pass
            if report_nan_and_inf:
                nan_count = self.raw_weight_o_i.isnan().sum()
                pos_inf_count = self.raw_weight_o_i.isposinf().sum()
                neg_inf_count = self.raw_weight_o_i.isneginf().sum()
                flat_too_much_big = self.raw_weight_o_i.abs().gt(10.)
                if nan_count>0 or pos_inf_count>0 or neg_inf_count>0 or flat_too_much_big.sum()>0:
                    fds=432
                    pass
                pass
            
            nan_to_this = self.raw_weight_min.item()
            self.raw_weight_o_i.data.nan_to_num_(nan=nan_to_this, posinf=nan_to_this, neginf=nan_to_this)
            
            #if max of each is less than 1, offset it to have a max of 1.
            max_element = self.raw_weight_o_i.max(dim = 1,keepdim=True).values
            dist_to_1_raw = 1-max_element
            dist_to_1 = torch.max(dist_to_1_raw,torch.zeros([1], device= the_device))
            #print(self.raw_weight_o_i.data)
            self.raw_weight_o_i += dist_to_1
            
            #compress anything greater than 1.
            row_gt_1:torch.Tensor = max_element.gt(1.)
            element_gt_0:torch.Tensor = self.raw_weight_o_i.gt(0.)
            needs_shrink_for_pos = row_gt_1.expand([-1,element_gt_0.shape[1]]).logical_and(element_gt_0)
            div_this_for_pos = needs_shrink_for_pos*max_element+needs_shrink_for_pos.logical_not()
            #print(self.raw_weight_o_i.data)
            self.raw_weight_o_i.data = self.raw_weight_o_i/div_this_for_pos
            
            #compress anything less than -1
            min_element = self.raw_weight_o_i.min(dim = 1,keepdim=True).values
            row_lt_neg_1:torch.Tensor = min_element.lt(-1.)
            element_lt_0:torch.Tensor = self.raw_weight_o_i.lt(0.)
            needs_shrink_for_neg = row_lt_neg_1.expand([-1,element_gt_0.shape[1]]).logical_and(element_lt_0)
            div_this_for_neg = needs_shrink_for_neg*min_element*-1.+needs_shrink_for_neg.logical_not()
            #print(self.raw_weight_o_i.data)
            self.raw_weight_o_i.data = self.raw_weight_o_i/div_this_for_neg
            #print(self.raw_weight_o_i.data)
            
            pass
        pass
    # end of function.

    def get_strong_grad_ratio(self, log10_diff = 0., \
            epi_for_w = 0.01, epi_for_g = 0.01)->float:
        result = debug_strong_grad_ratio(self.raw_weight_o_i, log10_diff, epi_for_w, epi_for_g)
        return result

    pass
fast_traval____end_of_digital_mapper_layer_class = 432

if 'deduplicate' and False:
    #case 1
    layer = DigitalMapper_V2_3(8,6)
    print(layer.deduplicate(torch.tensor([])))
    #case 2
    layer = DigitalMapper_V2_3(8,6)
    layer.raw_weight_o_i.data = torch.zeros_like(layer.raw_weight_o_i.data)
    layer.raw_weight_o_i.data[0,5] = 1.
    layer.raw_weight_o_i.data[1,6] = 1.
    layer.raw_weight_o_i.data[2,5] = 1.
    layer.raw_weight_o_i.data[3,6] = 1.
    layer.raw_weight_o_i.data[4,5] = 1.
    layer.raw_weight_o_i.data[5,5] = 1.
    print(layer.get_max_index())
    print(layer.deduplicate(torch.tensor([1,2,4,5])))
    #layer.deduplicate()
    print(layer.raw_weight_o_i.data)
    pass

if 'get_max_index test' and False:
    layer = DigitalMapper_V2_3(3,2)
    layer.raw_weight_o_i.data = torch.tensor([[0.,0.2,0.],[0.4,0.,0.]])
    input = torch.tensor([[1., -1, -1,],[-1, 1, -1,],[-1, -1, 1,]])
    print(layer(input))
    layer.eval()
    print(layer(input))
    print(layer.get_max_index())
    pass

if 'param protection test' and False:
    layer = DigitalMapper_V2_3(4,3)
    layer.raw_weight_o_i.data = torch.tensor([[-2.,3,1,-1,],[-4.,5,1,-1,],
                                                 [-3.,-2,0,0,],])
    layer.protect_raw_weight()
    print(layer.raw_weight_o_i.data)
    pass
    
if '''basic single layer test''' and False:
    layer = DigitalMapper_V2_3(3,1)
    #print(layer.raw_weight_o_i.shape)
    layer.raw_weight_o_i.data = torch.tensor([[0., 0., 1.]])
    #print(layer.raw_weight_o_i.shape)
    #print(layer.raw_weight_o_i.requires_grad)
    input = torch.tensor([[-1., 1., 1.]], requires_grad=True)
    target = torch.tensor([[-1.]]) 
    optim = torch.optim.SGD(layer.parameters(), lr=0.01)
    pred:torch.Tensor = layer(input)
    print(pred, "pred")
    pred.backward(target)
    print(layer.raw_weight_o_i.grad, "weight grad")
    print(input.grad, "input grad")
    print(layer.raw_weight_o_i.data, "weight before step")
    optim.step()
    print(layer.raw_weight_o_i.data, "weight after step")
    layer.protect_raw_weight()
    print(layer.raw_weight_o_i.data, "weight after protection")
    pass

if '''basic 2 layer test.''' and False:
    layer1 = DigitalMapper_V2_3(3,2)
    layer1.raw_weight_o_i.data = torch.tensor([[0., 0., 1.],[0., 1., 0.]])
    layer2 = DigitalMapper_V2_3(2,1)
    layer2.raw_weight_o_i.data = torch.tensor([[0., 1.]])
    input = torch.tensor([[-1., -1., 1.]], requires_grad=True)
    target = torch.tensor([[1.]]) 
    optim_them = []
    optim_them.extend(layer1.parameters())
    optim_them.extend(layer2.parameters())
    optim = torch.optim.SGD(optim_them, lr=0.01)

    pred:torch.Tensor = layer2(layer1(input))
    print(pred, "pred")
    pred.backward(target)
    print(layer1.raw_weight_o_i.grad, "weight grad")
    print(layer2.raw_weight_o_i.grad, "weight grad")
    print(input.grad, "input grad")
    pass

if '''basic 2 layer test 2. Answer with weight.''' and False:
    layer1 = DigitalMapper_V2_3(2,2)
    layer1.raw_weight_o_i.data = torch.tensor([[1., 0.],[0., 1.]])
    layer2 = DigitalMapper_V2_3(2,2)
    layer2.raw_weight_o_i.data = torch.tensor([[0., 1.],[0., 1.]])
    input = torch.tensor([[-1., 1.]], requires_grad=True)
    target = torch.tensor([[1.,-0.2]]) 
    optim_them = []
    optim_them.extend(layer1.parameters())
    optim_them.extend(layer2.parameters())
    optim = torch.optim.SGD(optim_them, lr=0.01)

    pred:torch.Tensor = layer2(layer1(input))
    print(pred, "pred")
    pred.backward(target)
    print(layer1.raw_weight_o_i.grad, "weight grad")
    print(layer2.raw_weight_o_i.grad, "weight grad")
    print(input.grad, "input grad")
    pass

if '1 layer real training.' and False:
    layer = DigitalMapper_V2_3(2,1)
    layer.raw_weight_o_i.data = torch.tensor([[1., 0.5]])
    input = torch.tensor([[-1., 1.]])#, requires_grad=True)
    target = torch.tensor([[1.]]) 
    optim = torch.optim.SGD(layer.parameters(), lr=0.1)
    print("11111111", layer(input))
    for _ in range(5):
        print(layer.raw_weight_o_i.data)
        pred:torch.Tensor = layer(input)
        optim.zero_grad()
        pred.backward(target)
        optim.step()
        pass
    print(layer.raw_weight_o_i.data)
    print(layer(input), "   <-both should the same as the strongest answer")
    pass

if 'does the weight affect the training.' and False:
    layer1 = DigitalMapper_V2_3(2,1)
    layer2 = DigitalMapper_V2_3(1,2)
    input = torch.tensor([[-1., 1.]])#, requires_grad=True)
    target = torch.tensor([[1.,-0.2]]) 
    optim_them = []
    optim_them.extend(layer1.parameters())
    optim_them.extend(layer2.parameters())
    optim = torch.optim.SGD(optim_them, lr=0.1)
    for _ in range(100):
        print(layer1.raw_weight_o_i.data)
        pred:torch.Tensor = layer2(layer1(input))
        optim.zero_grad()
        pred.backward(target)
        optim.step()
        pass
    print(layer1.raw_weight_o_i.data)
    print(layer2(layer1(input)), "   <-both should the same as the strongest answer")
    print("now uncomment the raise instruction.")
    pass

if '1 layer real training' and False:
    batch = 10
    n_in = 10
    n_out = 5
    input, target = data_gen_for_directly_stacking_test(batch,n_in, n_out)
    # print(input)
    # print(target)
    layer = DigitalMapper_V2_3(n_in, n_out)
    optim = torch.optim.SGD(layer.parameters(), lr = 0.1)
    for _ in range(10):
        pred = layer(input)  
        acc = bitwise_acc_with_str(pred, target,print_out=True)
        if 1. == acc:
            break
        optim.zero_grad()      
        pred.backward(target)
        optim.step()
        pass        
    pass        







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
# print("All 5 prints above should be equal.")
# fds=432




'''Dry stack test. OK, the dry is actually a Chinese word, which means, only, or directly.'''
class test_directly_stacking_multiple_digital_mappers(torch.nn.Module):
    @staticmethod
    def check_shape_config(input:List[int], allows_square_layer = False):
        if allows_square_layer:
            for i in range(input.__len__()-1):
                if input[i]<input[i+1]:
                    raise Exception("in this test, in dim must <= out dim.")
                pass
            pass
        else:
            for i in range(input.__len__()-1):
                if input[i]<=input[i+1]:
                    raise Exception("in this test, in dim must < out dim.")
                pass
            pass
        return
        #   end of function.
        
    @staticmethod
    def gen_shape_config(in_features:int, out_features:int, layer_count:int, allows_square_layer = False)->List[int]:
        if allows_square_layer:
            if out_features>in_features:
                raise Exception("in this test, in dim must <= out dim.")
            pass
        else:
            if out_features+layer_count>in_features:
                raise Exception("in this test, in dim must < out dim. If == is allowed, set allows_square_layer = True .")
            pass
        
        diff_every_layer = diff = float(in_features-out_features)/float(layer_count)
        result:List[int] = [in_features]
        current_width = in_features
        for i in range(1, layer_count):
            current_width -= diff_every_layer
            result.append(int(current_width))
            pass
        result.append(out_features)
        return result
    
    def __init__(self, shape_config:List[int], \
            #auto_print_difference:bool = False, \
            device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.digital_mappers = torch.nn.ParameterList([])
        self.shape_config = shape_config
        for i in range(shape_config.__len__()-1):
            in_features = shape_config[i]
            out_features = shape_config[i+1]
            self.digital_mappers.append(DigitalMapper_V2_3(in_features,out_features, debug_allow_any_shape=True,number_in_model=i))
            pass
        
        self.index_of_last_update = torch.nn.Parameter(torch.empty([shape_config.__len__()-1, shape_config[0]], dtype=torch.int64), requires_grad=False)
        self.index_of_last_init_ed = False
        pass

    def ______a():
        '''
    # def reset_scaling_ratio_for_raw_weight(self):
    #     layer:DigitalMapper_V2_3
    #     for layer in self.digital_mappers:
    #         layer.reset_scaling_ratio_for_raw_weight()
    #         pass
    #     pass
    # def scale_the_scaling_ratio_for_raw_weight(self, by:float):
    #     layer:DigitalMapper_V2_3
    #     for layer in self.digital_mappers:
    #         layer.scale_the_scaling_ratio_for_raw_weight(by)
    #         pass
    #     pass
    # def print_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01):
    #     temp_list:List[float] = []
    #     layer:DigitalMapper_V2_3
    #     for layer in self.digital_mappers:
    #         temp_list.append(layer.get_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g))
    #         pass
    #     pass
    #     print("debug_strong_grad_ratio: ", end="")
    #     for item in temp_list:
    #         print(f'{item:.3f}', end=", ")
    #         pass
    #     print()
    #     pass    
        '''
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        x = input
        for layer in self.digital_mappers:
            x = layer(x)
        return x
    
    #def print_strong_grad_ratio(self):
    #    for layer in self.digital_mappers:
    def _________print_debug_info(self):
        with torch.no_grad():
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.max())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.min())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.isnan().sum())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.isinf().sum())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
        
    def print_non_zero_grad_ratio(self):
        with torch.no_grad():
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                temp = layer.raw_weight_o_i.grad.eq(torch.zeros([1], device=layer.raw_weight_o_i.device))
                numerator = temp.sum().to(torch.float64)
                denominator = layer.raw_weight_o_i.nelement()
                to_print.append(1.-numerator/denominator)
                pass
            pass
            print(f"print_zero_grad_ratio: ", end="")            
            for number in to_print:
                print(f"{number:.2f} ", end="")            
                pass
            print()
            pass
        pass
    #end of function
    def __xxxx__set_update_str_Decrease(self):
        factor = torch.logspace(0,self.layer_count-1,self.layer_count ,base=0.9)
        layer:DigitalMapper_V2_3
        for i, layer in enumerate(self.digital_mappers):
            layer.gramo_for_raw_weight.set_scaling_factor(factor[i])
            pass
        fds=432

    def __save_max_index(self):
        r'''this function is only used in xxxx__print_diff. DO NOT directly call this.'''
        layer:DigitalMapper_V2_3
        with torch.no_grad():
            for i, layer in enumerate(self.digital_mappers):
                temp = layer.get_max_index()
                self.index_of_last_update[i,:temp.shape[0]] = temp
                pass
            pass
        pass#end of function.

    def xxxx__print_diff(self, epoch:int):
        # self.index_of_last_update = torch.nn.Parameter(torch.empty([shape_config.__len__()-1, shape_config[0]], dtype=torch.int64), requires_grad=False)
        # self.index_of_last_init_ed = False
        with torch.no_grad():
            layer:DigitalMapper_V2_3
            printed_something = False
            if self.index_of_last_init_ed:
                already_printed_epoch = False
                for i, layer in enumerate(self.digital_mappers):
                    new_result = layer.get_max_index()
                    old_result = self.index_of_last_update[i,:new_result.shape[0]]
                    flag_same = new_result.ne(old_result)
                    if flag_same.any():
                        printed_something = True
                        if not already_printed_epoch:
                            already_printed_epoch = True
                            print(epoch, "   epoch:  ")
                            pass
                        print(f"layer:{i}, ", old_result[flag_same], " >>> ", new_result[flag_same])
                        pass
                    self.index_of_last_update[i,:new_result.shape[0]] = new_result
                    pass
                return printed_something
            else:#self.index_of_last_init_ed: is false
                self.index_of_last_init_ed = True
                self.__save_max_index()
                return False#end if self.index_of_last_init_ed:
            pass#with torch.no_grad():
        pass#end of function
    
    def _print_max_index_count(self):
        r'''only used in besides_stepping. Do not call this function directly.'''
        with torch.no_grad():
            layer:DigitalMapper_V2_3 = self.digital_mappers[-1]
            temp1 = layer.get_max_index()
            previous_index = temp1.unique()
            layer_number = self.digital_mappers.__len__()-1
            if layer_number%10==0:
                print(f"(layer {layer_number}):", end="")
                pass
            print(f"{previous_index.shape[0]}, ", end="")

            for i in range(self.digital_mappers.__len__()-2,-1,-1):
                layer = self.digital_mappers[i]
                temp1 = layer.get_max_index()
                current_index = temp1[previous_index].unique()
                if i % 10 == 0:
                    print(f"(layer {i}):", end="")
                    pass
                print(f"{current_index.shape[0]}, ", end="")
                previous_index = current_index
                pass
            print()
            return
    #end of function.
    
    def _deduplicate(self, print_final_max_index_count = False):
        r'''only used in besides_stepping. Do not call this function directly.'''
    #I assume the input dim is always bigger than the output dim, because I believe this is the real case in the integrated situation.
    #If the in and out dims are the same, simply paste the input to the output, it should work perfectly.
    #if the input dim is smaller than output, why do I need this shape. This is the case in some test.
        layer:DigitalMapper_V2_3
        chosen_index:Tuple[torch.Tensor|None] = None
        for i in range(self.digital_mappers.__len__()-1,-1,-1):
            layer = self.digital_mappers[i]
            chosen_index = layer.deduplicate(chosen_index)
            pass
        if print_final_max_index_count:
            print("print_final_max_index_count", chosen_index.shape[0])
            pass
        pass#end of function.
    
    def besides_stepping(self, deduplicate = True, print_final_max_index_count = False, print_all_max_index_count = False):
        with torch.no_grad():
            if deduplicate:
                self._deduplicate(print_final_max_index_count)
                pass
            if print_all_max_index_count:
                self._print_max_index_count()
                pass
            return
    #end of function.
    pass

if 'besides_stepping' and False:
    model = test_directly_stacking_multiple_digital_mappers([10,8,6,4])
    layer:DigitalMapper_V2_3 = model.digital_mappers[-1]
    uniqued = layer.get_max_index().unique()
    print(uniqued)
    print()
    layer = model.digital_mappers[-2]
    pure_max = layer.get_max_index()
    print(pure_max)
    chosen_max = pure_max[uniqued]
    print(chosen_max)
    uniqued = chosen_max.unique()
    print(uniqued)
    print()
    layer = model.digital_mappers[-3]
    pure_max = layer.get_max_index()
    print(pure_max)
    chosen_max = pure_max[uniqued]
    print(chosen_max)
    uniqued = chosen_max.unique()
    print(uniqued)
    
    #now set breakpoint inside this function. The result may be a bit different, since the function deduplicates.
    model._deduplicate(print_final_max_index_count=True)
    pass

if '2 static methods test' and False:
    allows_square_layer = True
    the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(11,11,3,allows_square_layer=allows_square_layer)
    test_directly_stacking_multiple_digital_mappers.check_shape_config(the_config,allows_square_layer=allows_square_layer)
    the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(11,10,3,allows_square_layer=allows_square_layer)
    test_directly_stacking_multiple_digital_mappers.check_shape_config(the_config,allows_square_layer=allows_square_layer)
    # the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(11,12,3,allows_square_layer=allows_square_layer)
    # test_directly_stacking_multiple_digital_mappers.check_shape_config(the_config,allows_square_layer=allows_square_layer)
    
    allows_square_layer = False
    the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(11,5,3,allows_square_layer=allows_square_layer)
    test_directly_stacking_multiple_digital_mappers.check_shape_config(the_config,allows_square_layer=allows_square_layer)
    the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(11,8,3,allows_square_layer=allows_square_layer)
    test_directly_stacking_multiple_digital_mappers.check_shape_config(the_config,allows_square_layer=allows_square_layer)
    # the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(11,9,3,allows_square_layer=allows_square_layer)
    # test_directly_stacking_multiple_digital_mappers.check_shape_config(the_config,allows_square_layer=allows_square_layer)
    pass

if 'print_max_index test'and False:
    the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(1111,111,21)
    model = test_directly_stacking_multiple_digital_mappers(the_config)
    model.print_max_index_count()
    pass

if 'print_diff test(useless now)' and False:
    model = test_directly_stacking_multiple_digital_mappers([3,2,1])
    layer:DigitalMapper_V2_3 = model.digital_mappers[0]
    layer.raw_weight_o_i.data = torch.tensor([[0.,0.,1.,],[0.,1,0,]])
    layer = model.digital_mappers[1]
    layer.raw_weight_o_i.data = torch.tensor([[0.,1.]])
    model.xxxx__print_diff(0)
    layer = model.digital_mappers[0]
    layer.raw_weight_o_i.data = torch.tensor([[0.,1.,0.,],[0.,0,1,]])
    layer = model.digital_mappers[1]
    layer.raw_weight_o_i.data = torch.tensor([[1.,0.]])
    model.xxxx__print_diff(1)
    pass

if 'direct stack test' and False:
    batch = 10000
    in_features = 130
    out_features = 10
    num_layers = 40
    lr = 0.1#1.(11,29,300+,300+,300+),0.3(13,30,300+,300+)
    #0.2(7,7,300+,300+),0.1(9,17,27,36,98),0.03(135,300+,300+,300+), 0.01(27,300+,300+,300+)
    is_cuda = True
    is_half
    re_rand_target_weight_every = 1
    deduplicate_every = 1
    #in/out/layers:result in epoch.
    #100 10 30:    69,180,180
    #100 10 40:    500+
    #130 10 40:    43,340
    
    def print_config_after_finish():
        print("in_features:", in_features, "/out_features:", out_features, "/num_layers:", num_layers)
        return
    
    print(deduplicate_every, "   deduplicate_every")
    pt = Print_Timing(first=1, max_gap=20, density=1)
    
    iter_per_print = 1#1111
    print_count = 2000011

    (input, target_ori) = data_gen_for_directly_stacking_test(batch, in_features, out_features, no_duplicated = True)
    target = target_ori
    #print(target)

    the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(in_features, out_features, num_layers)
    model = test_directly_stacking_multiple_digital_mappers(the_config)
    #model.set_update_str_Decrease()
    #model.scale_the_scaling_ratio_for_raw_weight(3.)
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
        
    #model.print_debug_info()
        
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if is_cuda:
        model.cuda()
        input = input.cuda()
        target_ori = target_ori.cuda()
        target = target.cuda()
        pass
    if is_half:
        model.to(torch.float16)
        input = input.half()
        target_ori = target_ori.half()
        target = target.half()
    for _ in range(5):
        model.besides_stepping(deduplicate = True,print_final_max_index_count=True)
        pass
    
    
    #################################################################training loop
    for epoch in range(iter_per_print*print_count):
        if epoch%deduplicate_every == deduplicate_every-1:
            model.besides_stepping(deduplicate = True,print_all_max_index_count=True)
            #print("deduplicate")
            pass
        # re rand the temp weight for target
        if epoch%re_rand_target_weight_every == re_rand_target_weight_every-1:
            random_number = torch.rand([1,out_features], device=model.digital_mappers[0].raw_weight_o_i.device)
            random_number = torch.pow(random_number,3)
            target = target_ori*random_number
            pass
        
        # print diff of index
        #if epoch>100:
        # if model.print_diff(epoch):
        #     fds=432
        #     pass
        # if epoch == 222:
        #     model.print_max_index()
        #     fds=432
        #     pass
        
        model.train()
        pred = model(input)
        if "shape" and False:
            print(pred.shape, "pred.shape")
            print(target.shape, "target.shape")
            fds=423
        if "print pred" and False:
            if epoch%iter_per_print == iter_per_print-1:
                print(pred[:5], "pred")
                print(target[:5], "target")
                pass
            pass
        optimizer.zero_grad()
        
        if True and "print acc":
            with torch.inference_mode():
                (acc, perfect) = bitwise_acc_with_str(pred, target)
                if perfect:
                    print("FINISHED, ep:", epoch+1)
                    print("FINISHED, ep:", epoch+1)
                    print("FINISHED, ep:", epoch+1)
                    print_config_after_finish()
                    print(pred[:2,:5], "pred", "    __line 888")
                    print(target[:2,:5], "target")
                    break
                if pt.check(epoch):
                    print(epoch+1, f"    ep/acc    {acc:.3f}")
                    model._print_max_index_count()
                    pass
                pass
            pass#if True and "print acc":
        
        pred.backward(target)#intentional.
        # if epoch >1000:
        #     for layer in model.digital_mappers:
        #         print(layer.raw_weight_o_i.grad)
        #         pass        
        #     pass        
        
        if "print the grad" and False:
            if epoch%iter_per_print == iter_per_print-1:
                print(model.digital_mappers[0].raw_weight_o_i.grad, "0th layer    grad")
                print(model.digital_mappers[1].raw_weight_o_i.grad, "1 layer    grad")
                print(model.digital_mappers[-1].raw_weight_o_i.grad, "-1 layer    grad")
                pass
            pass
        if "print the weight" and False:
            layer:DigitalMapper_V2_3 = model.digital_mappers[0]
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
        if "print_zero_grad_ratio" and False:
            if pt.check(epoch):
                model.print_non_zero_grad_ratio()
                pass
            pass
       
        # if "print strong grad ratio" and True:
        #     if pt.check(epoch):
        #         result = model.print_strong_grad_ratio()
        #         print("print strong grad ratio: ", result)
        #         pass
        #     pass
        
        # if 1 == epoch:
        #     fds=432
        #optimizer.param_groups[0]["lr"] = 0.01
        optimizer.step()
        # if True and "print param overlap":
        #     every = 100
        #     if epoch%every == every-1:
        #         model.print_param_overlap_ratio()
        #         pass
        #     pass
        

    pass

#result below~
'''

########################
# with new better deduplicate which only handles chosen ones.
#batch = 10000   lr = 0.1
#re_rand_target_weight_every = 1    #deduplicate_every = 1
#in_features/out_features/num_layers:result in epoch.
#100 10 30:    69,180,180
#100 10 40:    500+(did not finish in 500 epochs)
#130 10 40:    43,340


########################
# with new better deduplicate which only handles chosen ones.
# batch = 10000  in_features = 100  out_features = 10
# num_layers = 25  lr = 0.1
#re_rand_target_weight_every:deduplicate_every(result epoch)
#1:    1(47,71,110)3(40,66,69)5(37,75,110)
#inf:  1(39,59,62)            5(62,100,130)


########################
# with new better deduplicate which only handles chosen ones.
# batch = 10000  in_features = 50  out_features = 10
# num_layers = 25  lr = 0.1
#re_rand_target_weight_every:deduplicate_every(result epoch)
#1:       1(320,580)          3(140,1k+)    5(260)
#3:       1(410,640)2(140,210)3(150,200,)   5(1k+)
#10:      1(290,470)          3(160,220,350)5(230,1k)
#inf:     1(180,260,450)                    5(145,900+)


########################
# with new better deduplicate which only handles chosen ones.
# batch = 10000  in_features = 50  out_features = 10
# num_layers = 20  lr = 0.1
#re_rand_target_weight_every:deduplicate_every(result epoch)
#1:       1(89,120,130)3(46,62,72,83,120)5(76,92,160)
#5:        1(85,95,170)3(65,140,310)5(94,150,240)
#50:       1(81,97,250)3(100,110,230)5(120,220,220)
#500:                  3(58,97,110)5(62,91,160)
#50000:1(39,96,120,120)3(43,52,150)5(120,180,600+)


####################
#old test for deduplicate_every vs re_rand_target_weight_every
# this test was done when deduplicate function manipulates everything.
#result: epoch to train. xxx+ means within xxx epochs, the training does not finish.

#batch = 10000  in_features = 50  out_features = 10
#num_layers = 20  lr = 0.1

#re_rand_weight(1): 1(500+,400+)3(130,230,280)5(64,160,250)7(150,190,210)10(230,290,350)
#re_rand_weight(3): 1(700+,400+)3(200,250,300)5(88,130,200)7(110,120,170)10(150,150,250)
#re_rand_weight(5): 1(400+,500+)3(80,210,220) 5(67,87,180) 7(58,110,150) 10(220,240,280)
#re_rand_weight(7): 1(400+)    3(180,220,260,)5(110,140,140)7(95,100,170)10(310,350)
#re_rand_weight(10):1(900+)     3(100,230,230)5(82,140,270)7(160,200,210)10(140,180,290)
#re_rand_weight(15):            3(74,74,110,280)5(100,120,150)7(66,93,120)10(640)
#re_rand_weight(20):            3(120,151,180)5(94,150,160,)7(38,74,130,130)
#re_rand_weight(30):1(800+)     3(65,200,300)5(64,92,130,290)7(130,180,200)

#15 and 15(230,250,260),20 and 20(400+,720...)

#re_rand_weight(3): 3(200,250,300)   10(110,160,200)60(380,490,610)
#re_rand_weight(10):3(50,70,130,220) 10(120,220,330)60(1k+,1k+,)
#re_rand_weight(30):3(90,100,120,470)10(190,290,780)
    
'''





class test_directly_stacking_multiple_digital_mappers_with_halfway_widen(torch.nn.Module):
    def __init__(self, width:int, extra_width:int, layer_count:int, \
            device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.digital_mappers = torch.nn.ParameterList([])
        self.width = torch.nn.Parameter(torch.tensor([width],dtype=torch.int64), requires_grad=False)
        self.extra_width = torch.nn.Parameter(torch.tensor([extra_width],dtype=torch.int64), requires_grad=False)
        #self.total_width = torch.nn.Parameter(torch.tensor([width+extra_width],dtype=torch.int64), requires_grad=False)
        total_width = width+extra_width
        
        for i in range(layer_count):
            self.digital_mappers.append(DigitalMapper_V2_3(total_width,width,number_in_model=i))
            pass
        
        #self.index_of_last_update = torch.nn.Parameter(torch.empty([shape_config.__len__()-1, shape_config[0]], dtype=torch.int64), requires_grad=False)
        #self.index_of_last_init_ed = False
        pass
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        x = input
        for layer in self.digital_mappers:
            halfway_noise = torch.randint(0,2,[batch,self.extra_width], device = x.device)*2-1
            x_with_noise = torch.concat((x,halfway_noise),dim=-1)
            x = layer(x_with_noise)
        return x
    
    #def print_strong_grad_ratio(self):
    #    for layer in self.digital_mappers:
    def _________print_debug_info(self):
        with torch.no_grad():
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.max())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.min())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.isnan().sum())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.isinf().sum())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
        
        '''
    def print_non_zero_grad_ratio(self):
        with torch.no_grad():
            to_print = []
            layer:DigitalMapper_V2_3
            for layer in self.digital_mappers:
                temp = layer.raw_weight_o_i.grad.eq(torch.zeros([1], device=layer.raw_weight_o_i.device))
                numerator = temp.sum().to(torch.float64)
                denominator = layer.raw_weight_o_i.nelement()
                to_print.append(1.-numerator/denominator)
                pass
            pass
            print(f"print_zero_grad_ratio: ", end="")            
            for number in to_print:
                print(f"{number:.2f} ", end="")            
                pass
            print()
            pass
        pass
    #end of function
    def __xxxx__set_update_str_Decrease(self):
        factor = torch.logspace(0,self.layer_count-1,self.layer_count ,base=0.9)
        layer:DigitalMapper_V2_3
        for i, layer in enumerate(self.digital_mappers):
            layer.gramo_for_raw_weight.set_scaling_factor(factor[i])
            pass
        fds=432

    def __save_max_index(self):
        r''this function is only used in xxxx__print_diff. DO NOT directly call this.''
        layer:DigitalMapper_V2_3
        with torch.no_grad():
            for i, layer in enumerate(self.digital_mappers):
                temp = layer.get_max_index()
                self.index_of_last_update[i,:temp.shape[0]] = temp
                pass
            pass
        pass#end of function.

    def xxxx__print_diff(self, epoch:int):
        # self.index_of_last_update = torch.nn.Parameter(torch.empty([shape_config.__len__()-1, shape_config[0]], dtype=torch.int64), requires_grad=False)
        # self.index_of_last_init_ed = False
        with torch.no_grad():
            layer:DigitalMapper_V2_3
            printed_something = False
            if self.index_of_last_init_ed:
                already_printed_epoch = False
                for i, layer in enumerate(self.digital_mappers):
                    new_result = layer.get_max_index()
                    old_result = self.index_of_last_update[i,:new_result.shape[0]]
                    flag_same = new_result.ne(old_result)
                    if flag_same.any():
                        printed_something = True
                        if not already_printed_epoch:
                            already_printed_epoch = True
                            print(epoch, "   epoch:  ")
                            pass
                        print(f"layer:{i}, ", old_result[flag_same], " >>> ", new_result[flag_same])
                        pass
                    self.index_of_last_update[i,:new_result.shape[0]] = new_result
                    pass
                return printed_something
            else:#self.index_of_last_init_ed: is false
                self.index_of_last_init_ed = True
                self.__save_max_index()
                return False#end if self.index_of_last_init_ed:
            pass#with torch.no_grad():
        pass#end of function
        '''
    
    def _print_max_index_count(self):
        r'''only used in besides_stepping. Do not call this function directly.'''
        with torch.no_grad():
            layer:DigitalMapper_V2_3 = self.digital_mappers[-1]
            temp1 = layer.get_max_index()
            previous_index:torch.Tensor = temp1.unique()
            flag_useful = previous_index.lt(self.width)
            useful_previous_index = previous_index[flag_useful]
            layer_number = self.digital_mappers.__len__()-1
            if layer_number%10==0:
                print(f"(layer {layer_number}):", end="")
                pass
            print(f"{useful_previous_index.shape[0]}, ", end="")

            for i in range(self.digital_mappers.__len__()-2,-1,-1):
                layer = self.digital_mappers[i]
                temp1 = layer.get_max_index()
                current_index = temp1[useful_previous_index].unique()
                flag_useful = previous_index.lt(self.width)
                useful_previous_index = previous_index[flag_useful]
                if i % 10 == 0:
                    print(f"(layer {i}):", end="")
                    pass
                print(f"{current_index.shape[0]}, ", end="")
                previous_index = current_index
                pass
            print()
            return
    #end of function.
    
    def _deduplicate(self, print_final_max_index_count = False):
        r'''only used in besides_stepping. Do not call this function directly.'''
    #I assume the input dim is always bigger than the output dim, because I believe this is the real case in the integrated situation.
    #If the in and out dims are the same, simply paste the input to the output, it should work perfectly.
    #if the input dim is smaller than output, why do I need this shape. This is the case in some test.
        layer:DigitalMapper_V2_3
        useful_chosen:Tuple[torch.Tensor|None] = None
        for i in range(self.digital_mappers.__len__()-1,-1,-1):
            layer = self.digital_mappers[i]
            chosen_index:torch.Tensor = layer.deduplicate(useful_chosen)
            flag_useful = chosen_index.lt(self.width)
            useful_chosen = chosen_index[flag_useful]
            pass
        if print_final_max_index_count:
            print("print_final_max_index_count", chosen_index.shape[0])
            pass
        pass#end of function.
    
    def besides_stepping(self, deduplicate = True, print_final_max_index_count = False, print_all_max_index_count = False):
        with torch.no_grad():
            if deduplicate:
                self._deduplicate(print_final_max_index_count)
                pass
            if print_all_max_index_count:
                self._print_max_index_count()
                pass
            return
    #end of function.
    pass

if 'basic test' and False:
    batch = 11
    width = 2
    extra_width = 3
    (input, target) = data_gen_for_directly_stacking_test(batch, width, width, no_duplicated = True)
    model = test_directly_stacking_multiple_digital_mappers_with_halfway_widen(width,extra_width,4)
    print(model(input))
    pass

if 'besides_stepping' and False:
    width = 5
    model = test_directly_stacking_multiple_digital_mappers_with_halfway_widen(width,5,3)
    layer:DigitalMapper_V2_3 = model.digital_mappers[-1]
    uniqued:torch.Tensor = layer.get_max_index().unique()
    print(uniqued)
    flag_useful = uniqued.lt(width)
    print(flag_useful)
    useful_uniqued = uniqued[flag_useful]
    print(useful_uniqued)
    print()
    
    layer = model.digital_mappers[-2]
    pure_max = layer.get_max_index()
    print(pure_max)
    chosen_max = pure_max[useful_uniqued]
    print(chosen_max)
    uniqued = chosen_max.unique()
    print(uniqued)
    flag_useful = uniqued.lt(width)
    print(flag_useful)
    useful_uniqued = uniqued[flag_useful]
    print(useful_uniqued)
    print()
    
    layer = model.digital_mappers[-3]
    pure_max = layer.get_max_index()
    print(pure_max)
    chosen_max = pure_max[useful_uniqued]
    print(chosen_max)
    uniqued = chosen_max.unique()
    print(uniqued)
    flag_useful = uniqued.lt(width)
    print(flag_useful)
    useful_uniqued = uniqued[flag_useful]
    print(useful_uniqued)
    print()
    
    #now set breakpoint inside this function. The result may be a bit different, since the function deduplicates.
    model._deduplicate(print_final_max_index_count=True)
    pass

if 'direct stack test WITH HALFWAY WIDEN!!!' and True:
    batch = 30
    width = 5
    extra_width = 5
    num_layers = 3
    lr = 0.1#0.01(3k,4k4,7k4)0.03(760,2k5,17k+)0.06(2k,10k+,420k+)0.1(68,79,220,440,4k+)0.3(83,190,220,4k+,4k+)1(22,110,480,4k+,4k+)
    is_cuda = True################
    is_half = True
    re_rand_target_weight_every = 10#10(110,160,240,1k3)5000(91,450,670,17k+)
    deduplicate_every = 5
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # fds=432
    
    def print_config_after_finish():
        #print("in_features:", in_features, "/out_features:", out_features, "/num_layers:", num_layers)
        print("batch:", batch, "/width:", width, "/extra_width:", extra_width)
        return
    
    pt = Print_Timing(first=1, max_gap=50, density=1)
    
    (input, target_ori) = data_gen_for_directly_stacking_test_same_dim_no_duplicated(batch, width)
    target = target_ori
    #print(target)

    model = test_directly_stacking_multiple_digital_mappers_with_halfway_widen(width,extra_width,num_layers)
    #model.scale_the_scaling_ratio_for_raw_weight(3.)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if is_cuda:
        model.cuda()
        input = input.cuda()
        target_ori = target_ori.cuda()
        target = target.cuda()
        pass
    if is_half:
        model.to(torch.float16)
        input = input.half()
        target_ori = target_ori.half()
        target = target.half()
    for _ in range(5):
        model.besides_stepping(deduplicate = True,print_final_max_index_count=True)
        pass
    
    
    
    
    #################################################################training loop
    for epoch in range(1231231):
        if epoch%1000 == 1000-1:
            
            for layer in model.digital_mappers:
                print(layer.raw_weight_o_i)
                pass
            pass
        
        if epoch%deduplicate_every == deduplicate_every-1:
            model.besides_stepping(deduplicate = True,print_all_max_index_count=False)
            #print("deduplicate")
            pass
        # re rand the temp weight for target
        if epoch%re_rand_target_weight_every == re_rand_target_weight_every-1:
            random_number = torch.rand([1,width], device=model.digital_mappers[0].raw_weight_o_i.device)
            random_number = torch.pow(random_number,3)
            target = target_ori*random_number
            pass
        
        
        model.train()
        pred = model(input)
        if "shape" and False:
            print(pred.shape, "pred.shape")
            print(target.shape, "target.shape")
            pass
        if "print pred" and False:
            if 0 == epoch :
                print(pred[:5], "pred")
                print(target[:5], "target")
                pass
            pass
        if "print pred" and False:
            print(pred[:5], "pred")
            pass
        if True and "print acc":
            with torch.inference_mode():
                (acc, perfect) = bitwise_acc(pred, target)
                if perfect:
                    print("FINISHED, ep:", epoch+1)
                    print("FINISHED, ep:", epoch+1)
                    print("FINISHED, ep:", epoch+1)
                    print_config_after_finish()
                    print(pred[:2,:5], "pred", "    __line 888")
                    print(target[:2,:5], "target")
                    break
                if pt.check(epoch):
                    print(epoch+1, f"    ep/acc    {acc:.3f}")
                    model._print_max_index_count()
                    pass
                pass
            pass#if True and "print acc":
        
        optimizer.zero_grad()
        pred.backward(target)#intentional.
        if "print the grad" and False:
            if epoch%iter_per_print == iter_per_print-1:
                print(model.digital_mappers[0].raw_weight_o_i.grad, "0th layer    grad")
                print(model.digital_mappers[1].raw_weight_o_i.grad, "1 layer    grad")
                print(model.digital_mappers[-1].raw_weight_o_i.grad, "-1 layer    grad")
                pass
            pass
        if "print the weight" and False:
            layer:DigitalMapper_V2_3 = model.digital_mappers[0]
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
        if "print_zero_grad_ratio" and False:
            if pt.check(epoch):
                model.print_non_zero_grad_ratio()
                pass
            pass
        optimizer.step()
    
    for layer in model.digital_mappers:
        print(layer.raw_weight_o_i)
   
    pass