from typing import Any, List, Tuple, Optional
from enum import Enum, auto
import sys
from pathlib import Path
import math
import torch
sys.path.append(str(Path(__file__).parent.parent))


#from util import debug_Rank_1_parameter_to_List_float, int_into_floats, floats_into_int
from pytorch_yagaodirac_v2.Util import bitwise_acc, data_gen_for_directly_stacking_test
from pytorch_yagaodirac_v2.Util import data_gen_for_directly_stacking_test_same_dim_no_duplicated
from pytorch_yagaodirac_v2.Util import debug_strong_grad_ratio, make_grad_noisy
from pytorch_yagaodirac_v2.Util import Print_Timing
from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2
from pytorch_yagaodirac_v2.training_ended_sound import play_noise
from pytorch_yagaodirac_v2.torch_vec import torch_vec
#from Binarize import Binarize

'''
2.6的计划。
简单的说。最早的1.4版本和2.5版本的一个根本区别是，1.2到1.4是有softmax的做法，很传统，无法得到足够尖锐的结果。
1.4其实是误差回传。
2.5是纯二值化的做法，很尖锐，但是目标回传的性能不是很好。
现在遇到的问题是，如果可以利用1.4的那种信息反向渗透的能力给2.5探路，用2.5的能力来固化和得到最终的离散化的
前向传播，那么有可能就是最终版本。
现在的问题是，1.4的做法能不能和目标回传兼容。
这个2.6版本就是要测试这个。
现在的计划是，参数保护是直接做成小于0，极限值给一个-50，反正softmax，-50应该已经看不到了。
前向传播路径上不用二值化层，看看效果，之后再决定。

以下是以前的笔记

# 2.5的计划。
# 现在是答案回传，回传过程当中乘的一个类似权重的东西也是稠密的，这个反向传播虽然没有被验证过，但是根据我的经验，这个是没什么问题的。
# 新版本要做一个绝对不会重复的行为来决定每一次前向传播的时候的最终的对应关系。
# 但是这个太慢了，所以要做一些比较激进的但是不稳定的尝试。
# 如此一来，2.4的所有去重就不需要了。
# 参数依然保护在-1到1之间，或者类似区间。

#之前的2.4的问题。
#这个版本也不行。不行的原因是，两种重叠很难处理。
#第一个是一个输入被多个输出选择，这个是一定会发生的，代码里面用的办法是，把其中一个扔开。
#第二个是一个输出选择多个输入。我本来以为这个不会发生，因为必须最大的生权重必须要又多个完全相等，我以为浮点数不会发生这种事情。
#结果还是发生了。解决方案是，在生权重上看，最大的，因为会保护成1，所以检测比1-epi大就行，加上一个很小的随机数。
#但是随机数在每一个epoch的结果很可能不同，它会导致另外一个去重会被反复激活，于是变化最集中的区域永远无法设计出一个稳定机制。
#于是，我决定用笨办法。

# 然后是2.3版的笔记。2.4版依然适用。
# 回传的东西，如果是0，那么表示没有被后面的层选用。
# 如果不是0，符号表示后面需要的正确答案，绝对值表示强度。
# target是答案乘以强度，直接用autograd.backward给进去，而不用loss.backward()。
# 自定义回传的时候，顺着主梯度链传的其实是答案和对应强度，而不是梯度。
# 而w得到的是梯度，那个地方用了乘法，其实是，当x和答案（g）一致的时候，是对的，要强化。不一致表示错误，要弱化。
# 乘以-1是因为，pytorch的更新是减去梯度。神经网络都这么干的。

# 之前2.3版本里面有一个问题，就是，答案回传的时候只在彻底选中的路线上传，会导致，
# 如果输入更宽，而且有一些是肯定错误的选项，答案路径会被浪费掉，导致不是没一个答案路径都会追到输入。
# 总错误输入足够多的时候，就会有固定的概率无法训练。
# 于是，这个2.4版，准备利用稠密权重的那种手感来解决这个问题。
# 总的来说，2.4会充分利用之前研究全连接层的时候，稠密权重会帮助梯度回传的经验。

# 改进空间。
# 因为整个体系的数字是很普通的，不需要真的用gramo去保护，可以另外单独写一个东西，效率还有可能更高。
# 不需要梯度的时候，自定义的通路里面还可以有一些简化。
'''

    


class DigitalMapperFunction_v2_6(torch.autograd.Function):
    r'''
    forward input list:
    >>> x = args[0]# shape must be [batch, in_features]
    >>> raw_weight:torch.Tensor = args[1]# shape must be [out_features, in_features], must requires grad.
    >>> place_holder_for_answer_strength_o shape[out_features]
    
    backward input list:
    >>> g_in #shape of g_in must be [batch, out_features]
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        input_b_i:torch.Tensor = args[0]# shape must be [batch, in_features]
        raw_weight_o_i:torch.Tensor = args[1]# shape must be [out_features, in_features]
        place_holder_for_answer_strength_o:torch.Tensor = args[2]# value doesn't matter. shape[out_features]
                
        softmax_of_raw_weight_o_i = raw_weight_o_i.softmax(dim=1)
        
        input_reshaped_b_i_1 = input_b_i.reshape([input_b_i.shape[0],input_b_i.shape[1],1])
        
        output_before_reshape_b_o_1:torch.Tensor = softmax_of_raw_weight_o_i.matmul(input_reshaped_b_i_1)
        output_b_o = output_before_reshape_b_o_1.squeeze(dim=2)
        #output.requires_grad_(input_b_i.requires_grad and raw_weight_o_i.requires_grad)
        w_requires_grad = torch.tensor([raw_weight_o_i.requires_grad])
        ctx.save_for_backward(input_b_i, softmax_of_raw_weight_o_i, w_requires_grad)
        return output_b_o

    @staticmethod
    def backward(ctx, g_in_b_o):
        #shape of g_in must be [batch, out_features]
        input_b_i:torch.Tensor
        softmax_of_raw_weight_o_i:torch.Tensor
        w_requires_grad:torch.Tensor
        (input_b_i, softmax_of_raw_weight_o_i, w_requires_grad) = ctx.saved_tensors
        
        grad_for_x_b_i:Tuple[torch.tensor|None] = None
        grad_for_raw_weight_o_i:Tuple[torch.tensor|None] = None
        
        #input_requires_grad = torch.tensor([input_b_i.requires_grad], device=input_b_i.device)
        
        if w_requires_grad:
            g_in_reshaped_b_o_1:torch.Tensor = g_in_b_o[:,:,None]
            input_reshaped_b_1_i:torch.Tensor = input_b_i[:,None,:]
            
            grad_for_raw_weight__before_sum__b_o_i:torch.Tensor = g_in_reshaped_b_o_1*(input_reshaped_b_1_i*-1)
            #print(grad_for_raw_weight__before_sum__b_o_i)
            grad_for_raw_weight_o_i = grad_for_raw_weight__before_sum__b_o_i.sum(dim=0, keepdim=False)
            #print(grad_for_raw_weight_o_i)
            
            answer_strength_o = g_in_b_o.abs().sum(dim=0, keepdim=False)
            pass
        
        if input_b_i.requires_grad:
            grad_for_x_b_i = g_in_b_o.matmul(softmax_of_raw_weight_o_i)
            pass
            
        return grad_for_x_b_i, grad_for_raw_weight_o_i, answer_strength_o
    
    
    # input_b_i:torch.Tensor = args[0]# shape must be [batch, in_features]
    # raw_weight_o_i:torch.Tensor = args[1]# shape must be [out_features, in_features]
    # the dummy to get answer str back.
    pass  # class


if '''main check 2.6. 应该正常了。''' and False:
    b=2
    o=3
    i=5
    #x = torch.rand([b,i], requires_grad=True)
    #x = torch.tensor([[11.,12,13,14,15],[21,22,23,24,25]], requires_grad=True)
    #x = torch.tensor([[1.,-1,1,1,-1,],[1.,1,1,1,1,],], requires_grad=True)
    x = torch.tensor([[0.,0,0,0,1],[0,0,0,0,11],], requires_grad=True)
    #w = torch.rand([o,i], requires_grad=True)
    #w = torch.tensor([[0.,0,0,0,1],[0.,0,1,0,0],[0.,1,0,0,0],], requires_grad=True)
    w = torch.tensor([[-11.,-11,-11,-0.9,0],[-11,-1,0.,-11,-11],[-1,0.,-11,-11,-11],], requires_grad=True)
    print(w.softmax(dim=1), "softmax")
    #w = torch.tensor([[0.,0.1,0,0,0],[0.,0.1,0,0,0],[0.,0.1,0,0,0],], requires_grad=True)
    #mapping = torch.tensor([0,2,1])
    place_holder_answer_strength = torch.zeros([3], dtype=w.dtype, requires_grad=True)
    pred:torch.Tensor = DigitalMapperFunction_v2_6.apply(x, w,place_holder_answer_strength)
    print(pred, "pred")
    #g = torch.tensor([[111.,112,113],[121,122,123]])
    #g = torch.tensor([[1001.,1002,1003],[1004,1005,1006]])
    #g = torch.tensor([[1.,-1,1],[1.,1,1]])
    g = torch.tensor([[1.,0,0],[0,1.1,0]])
    pred.backward(g)#torch.tensor([[1.31, -1.32]]))
    
    # print(x.shape == x.grad.shape)
    # print(w.shape == w.grad.shape)
    print(x.grad, "x.grad is the correct answer with strength")
    print(w.grad, "w.grad, neg for correct mapping")
    #print(w.grad.abs().sum(dim=1,keepdim=False), "this is not answer strength")
    print(place_holder_answer_strength.grad, "answer_strength")
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
        


class DigitalMapper_v2_6(torch.nn.Module):
    r'''This layer is designed to be used between digital layers.
    The input should be in STANDARD range so to provide meaningful output
    in STANDARD range. It works for both 01 and np styles.

    Notice: unlike most layers in this project, this layer is stateful.
    In other words, it has inner param in neural network path.

    Remember to concat a constant 0. and 1. to the input before sending into this layer.
    In other words, provide Vdd and Vss as signal to the chip.
    '''
    #__constants__ = []

    def __init__(self, in_features: int, out_features: int, \
                     gramo_for_each_output:bool, \
                    #is_out_mapper_in_DSPU:bool, \
                 #needs_out_gramo = True, \
                    # auto_print_difference:bool = False, \
                    # out_gramo_scale_factor = 1., \
                    # gramo_for_raw_weight_scale_factor = 1., \
                    #protect_param_every____training:int = 5, \
                    debug_allow_any_shape = False, \
                    debug_number_in_model = -1, \
                    device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if in_features<2 and not debug_allow_any_shape:
            raise Exception('If this is intentional, search "if statement in python" on google.com.')

        self.debug_number_in_model = debug_number_in_model

        if out_features>in_features and not debug_allow_any_shape:
            raise Exception("out dim must be <= in dim. For debug purpose, set debug_allow_any_shape=True.")
        self.in_features = torch.nn.Parameter(torch.tensor([in_features], dtype = torch.int64), requires_grad=False)
        self.out_features = torch.nn.Parameter(torch.tensor([out_features], dtype = torch.int64), requires_grad=False)
        self.gramo_for_each_output = torch.nn.Parameter(torch.tensor([gramo_for_each_output], dtype = torch.bool), requires_grad=False)
            
        if in_features == out_features:
            #it's a transparent layer.
            self.raw_weight_o_i = torch.nn.Parameter(torch.empty((0), **factory_kwargs))
        else:
            self.raw_weight_o_i = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            #self.mapping_index_previous_o = torch.nn.Parameter(torch.empty(size=[out_features], dtype=torch.int64, device = device), requires_grad=False)
            #self.mapping_index_previous_o.data.fill_(-1)#this never equals in the first epoch.
            self.grad_is_answer_strength_previous_o = torch.nn.Parameter(torch.zeros([out_features], **factory_kwargs), requires_grad=True)
            self.grad_is_answer_strength_previous_o.grad = torch.zeros([out_features], **factory_kwargs)
            
            self.__reset_parameters__the_plain_rand01_style()
            #self.update_mapping_index()
            pass
    
            #这两个暂时没有用上。
        self.raw_weight_max = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.raw_weight_min = torch.nn.Parameter(torch.tensor([-1125.]), requires_grad=False)#to test. This is a hyperparam. Test it.
        #or this ??? self.raw_weight_min = torch.nn.Parameter(torch.tensor([-1./100]), requires_grad=False)
        
        # 2 param used in randomly making top elements picked in forward path.
        self.neg_epi_for_float_eq = torch.nn.Parameter(torch.tensor([-0.00001]), requires_grad=False)
        if dtype == torch.float16:
            self.small_number = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=False)
        else:
            self.small_number = torch.nn.Parameter(torch.tensor([0.00003]), requires_grad=False)
            pass
        
        self.gramo_for_raw_weight = GradientModification_v2()
        self.out_gramo = GradientModification_v2()
        
        pass

    def __reset_parameters__the_plain_rand01_style(self) -> None:
        '''copied from torch.nn.Linear'''
        self.raw_weight_o_i.data = torch.rand_like(self.raw_weight_o_i)# they should be <0.
        pass

    def get_plain_max_index_from_raw(self)->torch.Tensor:
        if self.out_features == self.in_features:
            raise Exception("untested")
            return torch.linspace(0,self.in_features-1,self.in_features,dtype=torch.int64)
            
        with torch.no_grad():
            
            the_max_index = self.raw_weight_o_i.max(dim=1,keepdim=False).indices
            return the_max_index

    def ____get_mapping_index_previous_o(self)->torch.Tensor:
        raise Exception("do I need this?")
        if self.out_features == self.in_features:
            raise Exception("untested")
            return torch.linspace(0,self.in_features-1,self.in_features,dtype=torch.int64)
            
        if self.mapping_index_previous_o[0]<0:
            raise Exception("UNREACHABLE CODE!!!!!!!! train at least 1 epoch, otherwise this is not initialized.")
        with torch.no_grad():
            #raise Exception("untested")
            return self.mapping_index_previous_o.data.detach()

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

    def get_answer_strength(self)->torch.Tensor:
        return self.grad_is_answer_strength_previous_o.grad

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
            
            #this branch statement is intentional.
            if self.gramo_for_each_output:
                w_after_gramo = self.gramo_for_raw_weight(self.raw_weight_o_i)
            else:
                w_after_gramo = self.gramo_for_raw_weight(self.raw_weight_o_i.reshape(1,-1)).reshape(self.out_features, self.in_features)
                pass
            
            #self.update_mapping_index()
                
            #print(self.mapping_index_previous_o.data, "line 609")
                
            #print(w_after_gramo.dtype, "  __line 395")
                
            x = DigitalMapperFunction_v2_6.apply(x, w_after_gramo, self.grad_is_answer_strength_previous_o)
            
            x = self.out_gramo(x)
            
            # if torch.isnan(x).any():
            #     fds = 432
            #     pass
            
            return x
        
        else:#eval mode.
            raise Exception("to do.")
            
            x = input[:, self.mapping_index_previous_o]
            return x

    def ____update_mapping_index(self):
        raise Exception (" do I need this anymore?")
        answer_str_sort_index_previous_o = self.__get_answer_str_sort_index_previous_o()
    
            #to do :mapping_index_trying_now_o = torch.empty([self.out_features])
            #debug
            #mapping_index_trying_now_o.fill_(-1)
            
            #some short cut test? with mapping_index_trying_now_o and mapping_index_previous_o
            
        raw_weight_copied_o_i:torch.Tensor = self.raw_weight_o_i.detach().clone()
            
            #the last stable fallback.
        for for_which_output in answer_str_sort_index_previous_o:
            max_index_here = raw_weight_copied_o_i[for_which_output].argmax()
            self.mapping_index_previous_o[for_which_output] = max_index_here
            raw_weight_copied_o_i[:,max_index_here] = -222
            pass
        pass

    def __get_answer_str_sort_index_previous_o(self):
        answer_str = self.get_answer_strength()
        result = answer_str.sort(descending=True).indices#descending=True important.
        
        return result
        #return self.get_answer_strength().sort(descending=True).indices
    #end of function.

    def extra_repr(self) -> str:
        #return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'
        return f'In_features={self.in_features}, out_features={self.out_features}'

    def protect_raw_weight(self, report_nan_and_inf = False):
        '''
        this function is designed only to be called in the forward. 
        Do not call this function directly.
        Moves everything between 1 and -1????
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
            
            
            #offset to align max of each row to 0
            #this alignment fits with softmax.
            max_element = self.raw_weight_o_i.max(dim = 1,keepdim=True).values
            #print(self.raw_weight_o_i.data)
            self.raw_weight_o_i -= max_element
            
            #clamp elements <-15 back to -15.
            flag_too_small = self.raw_weight_o_i.lt(self.raw_weight_min)
            self.raw_weight_o_i.data =    flag_too_small*self.raw_weight_min+ \
                            flag_too_small.logical_not()*self.raw_weight_o_i
            
            
            # The last part of this function. 
            # Because when extreme elements duplicate, the max(also min) returns the first one.
            # If they get the same update, only the first one is visible in forward path.
            # This part is design to defend againt this weak point.
            # Some random number is added to the top elements so they are visible to the forward path randomly.
            flag_too_close_to_1 = self.raw_weight_o_i.data.gt(self.neg_epi_for_float_eq)
            some_rand = torch.randn_like(self.raw_weight_o_i, device=self.raw_weight_o_i.device, dtype=self.raw_weight_o_i.dtype)*self.small_number
            to_add_to_raw_weight = some_rand*flag_too_close_to_1
            self.raw_weight_o_i.data += to_add_to_raw_weight
            pass
        pass
    # end of function.

    def debug__get_strong_grad_ratio(self, log10_diff = 0., \
            epi_for_w = 0.01, epi_for_g = 0.01)->float:
        result = debug_strong_grad_ratio(self.raw_weight_o_i, log10_diff, epi_for_w, epi_for_g)
        return result

    def deduplicate_v2_6(self, chosen_index:Tuple[torch.Tensor|None] = None)->Tuple[torch.Tensor|None]:
        #possible optimization:
        # since the duplication relationship of plain max index should be equivalent to the relationship from
        # "difference between plain max and the non-duplicating mapping relationship of previous",
        # simply compare the difference and kick off the different element.
        # But this version is already decent. Let's use it for now.
        
        if self.out_features == self.in_features:
            raise Exception("untested")
            return chosen_index
        with torch.no_grad():
            answer_str_sort_index_previous_o = self.__get_answer_str_sort_index_previous_o()#descending=True important.
            
            max_index_plain = self.get_plain_max_index_from_raw()
            #buff = torch.empty([max_index_plain.shape[0]], device=self.raw_weight_o_i.device, dtype=max_index_plain.dtype)
            buff = torch_vec(1, init_cap=max_index_plain.shape[0],device=self.raw_weight_o_i.device ,dtype=torch.int64)#or 32?
            buff.data.fill_(-1) #for debug purpose.
            for_which_output = answer_str_sort_index_previous_o[0]
            #buff[0] = max_index_plain[for_which_output] 
            buff.pushback(max_index_plain[for_which_output])
            #len = torch.tensor([1])
            for i in range(1,max_index_plain.shape[0]):
                #print(buff[:len], "line 748")
                #print(self.raw_weight_o_i.data)
                for_which_output = answer_str_sort_index_previous_o[i]
                max_index_now = max_index_plain[for_which_output]
                flag = buff.get_useful().eq(max_index_now)
                if flag.any():
                    self.raw_weight_o_i.data[for_which_output,max_index_now] -= 100.#to test: test this -1
                    # with lr == 10# 1(220,300,340,420)10(12,12,19,65)
                    # with lr == 1.# 0.1()1(32,56,100+0.97)10(34,34,38,49)100(32,36,41,68)
                    # with lr == 0.1# 0.01(800+unstable 0.82)0.1(270,270,330,400+unstable0.97)1(880,2k,3k2+0.97)
                    #11111111111111111111111
                    继续。
                    
                    #to test: test this.
                else:
                    buff.pushback(max_index_now)
                    #buff[len.item()] = max_index_now
                    #len+=1
                    pass
                pass
            
            if chosen_index is None:
                temp1 = self.get_plain_max_index_from_raw()
                temp3 = temp1.unique()
                return temp3
            else:
                temp1 = self.get_plain_max_index_from_raw()
                temp2 = temp1[chosen_index]
                temp3 = temp2.unique()
                return temp3
        #end of function
     
       
    @staticmethod
    def rand_weight_for_target(original_target:torch.Tensor)->torch.Tensor:
        raise Exception ("untested.")
        '''
        random_number = torch.rand([1,out_features], device=target_ori.device)
        random_number = torch.pow(random_number,3)
        '''
        random_number = torch.rand_like([1,out_features], device=target_ori.device)
#        random_number = torch.pow(random_number, check out the test result in both tests.)
        result = original_target*random_number
        return result
    
    pass
fast_traval____end_of_digital_mapper_layer_class = 432

# a = torch.tensor([21,31,11,41,61,51])
# b = a.sort(descending=True).indices
# print(b)
# fds=432

if 'deduplicate v2.6 test' and False:
    
    layer = DigitalMapper_v2_6(5,4,0.5,False)
    print(layer.raw_weight_o_i.shape)
    layer.raw_weight_o_i.data = torch.tensor([[0.1,0.1,0.4,0.7,0.9],[0.1,0.1,0.4,0.7,0.9],
                                              [0.1,0.1,0.4,0.7,0.9],[0.1,0.1,0.4,0.7,0.9],])
    
    print(layer.raw_weight_o_i.shape)
    layer.grad_is_answer_strength_previous_o.grad = torch.tensor([1.,2.,3.,2.])
    print(layer.deduplicate_v2_6())
    print(layer.raw_weight_o_i.data)
    
    chosen_index = torch.tensor([2,3])
    print(layer.deduplicate_v2_6(chosen_index))
    print(layer.raw_weight_o_i.data)
    pass
    
if 'get_max_index test 2.6没有这个.' and False:
    gramo_for_each_output = True # This doesn't do anything when output only has 1 element. 
    layer = DigitalMapper_v2_6(3,2,0.5,gramo_for_each_output)
    print(layer.get_mapping_index_previous_o())
    
    layer.raw_weight_o_i.data = torch.tensor([[0.,0.2,0.],[0.4,0.,0.]])
    input = torch.tensor([[1., -1, -1,],[-1, 1, -1,],[-1, -1, 1,]])
    print(layer(input))
    layer.eval()
    print(layer(input))
    print(layer.get_mapping_index_previous_o())
    pass

if 'param protection test' and False:
    layer = DigitalMapper_v2_6(4,3, gramo_for_each_output=False)
    layer.raw_weight_o_i.data = torch.tensor([[-2.,3,1,-1,],[-4.,5,1,-1,],
                                                 [-3.,-2,0,0,],])
    layer.protect_raw_weight()
    print(layer.raw_weight_o_i.data)
    pass
    
if '''basic single layer test''' and False:
    gramo_for_each_output = True # This doesn't do anything when output only has 1 element. 
    layer = DigitalMapper_v2_6(3,1,0.5,gramo_for_each_output)
    print(layer.raw_weight_o_i.shape)
    layer.raw_weight_o_i.data = torch.tensor([[-1.,-1., 1.]])
    print(layer.raw_weight_o_i.shape)
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
    #print(layer.mapping_index_previous_o.data, "mapping_index_previous_o")2.6没有这个。
    print(layer.grad_is_answer_strength_previous_o.grad, "answer_strength")
    pass

if '''basic 2 layer test.''' and False:
    #gramo_for_each_output matters. It doesn't do anything useful in v2.3
    gramo_for_each_output = False
    layer1 = DigitalMapper_v2_6(3,2,0.5,gramo_for_each_output)
    layer1.raw_weight_o_i.data = torch.tensor([[0., 0., 5.],[0., 5., 0.]])
    layer2 = DigitalMapper_v2_6(2,1,0.5,gramo_for_each_output)
    layer2.raw_weight_o_i.data = torch.tensor([[0., 5.]])
    input = torch.tensor([[-1., -1., 1.]], requires_grad=True)
    target = torch.tensor([[1.]]) 
    optim_them = []
    optim_them.extend(layer1.parameters())
    optim_them.extend(layer2.parameters())
    optim = torch.optim.SGD(optim_them, lr=0.01)

    mid_result:torch.Tensor = layer1(input)
    mid_result.retain_grad()
    pred:torch.Tensor = layer2(mid_result)
    print(pred, "pred")
    pred.backward(target)
    print(layer1.raw_weight_o_i.grad, "l1 weight grad")
    print(layer2.raw_weight_o_i.grad, "l2 weight grad")
    print(mid_result.grad, "mid_result.grad")
    print(input.grad, "input grad")
    pass

if '''basic 2 layer test 2. Answer with weight.''' and False:
    gramo_for_each_output = False
    layer1 = DigitalMapper_v2_6(4,3,0.5,gramo_for_each_output)
    layer1.raw_weight_o_i.data = torch.tensor([[5.,0,0,0],[0., 5.,0,0],[0., 0.,5,0]])
    layer2 = DigitalMapper_v2_6(3,2,0.5,gramo_for_each_output)
    layer2.raw_weight_o_i.data = torch.tensor([[0., 5,0],[0., 5,0]])
    input = torch.tensor([[-1.,1.,1., 1.]], requires_grad=True)
    target = torch.tensor([[1.,-1.2]]) 
    optim_them = []
    optim_them.extend(layer1.parameters())
    optim_them.extend(layer2.parameters())
    optim = torch.optim.SGD(optim_them, lr=0.01)

    mid_result = layer1(input)
    mid_result.retain_grad()
    pred:torch.Tensor = layer2(mid_result)
    print(pred, "pred")
    pred.backward(target)
    print(layer1.raw_weight_o_i.grad, "l1 weight grad")
    print(layer2.raw_weight_o_i.grad, "l2 weight grad")
    print(input.grad, "input grad")
    print(mid_result.grad, "mid_result grad")
    print(layer1.grad_is_answer_strength_previous_o.grad, "l1 answer_strength")
    print(layer2.grad_is_answer_strength_previous_o.grad, "l2 answer_strength")
    pass

if '1 layer real training.' and False:
    gramo_for_each_output = False
    layer = DigitalMapper_v2_6(2,1,0.5,gramo_for_each_output)
    layer.raw_weight_o_i.data = torch.tensor([[5., 0.]])
    input = torch.tensor([[-1., 1.]])#, requires_grad=True)
    target = torch.tensor([[1.]]) 
    optim = torch.optim.SGD(layer.parameters(), lr=1.)
    print(layer(input), "pred before training")
    for _ in range(5):
        pred:torch.Tensor = layer(input)
        print(layer.raw_weight_o_i.data, "raw weight")
        optim.zero_grad()
        pred.backward(target)
        optim.step()
        pass
    layer.protect_raw_weight()
    print(layer.raw_weight_o_i.data, "raw weight")
    print(layer(input), "   <-result")
    pass

if '2 layer real training.不一定说明问题，batch只有1.。。' and False:
    gramo_for_each_output = False
    layer1 = DigitalMapper_v2_6(4,3,gramo_for_each_output)
    layer1.raw_weight_o_i.data = torch.tensor([[1.,0,0,0],[0., 1.,0,0],[0., 0.,1,0]])
    layer2 = DigitalMapper_v2_6(3,2,gramo_for_each_output)
    layer2.raw_weight_o_i.data = torch.tensor([[0., 1,0],[0., 1,0]])
    
    input = torch.tensor([[-1.,1.,1.,1.]])#, requires_grad=True)
    target = torch.tensor([[1.,-0.5]]) 
    optim_them = []
    optim_them.extend(layer1.parameters())
    optim_them.extend(layer2.parameters())
    optim = torch.optim.SGD(optim_them, lr=1.)
    
    print(layer1.raw_weight_o_i.data, "l1 before")
    print(layer2.raw_weight_o_i.data, "l2 before")
    for _ in range(4):
        print("--------------------------------")
        layer1.deduplicate_v2_6()
        layer2.deduplicate_v2_6()
        mid_result = layer1(input)
        pred:torch.Tensor = layer2(mid_result)
        print(mid_result.data, "mid_result")
        print(pred.data, "pred")
        print("-----------")
        print(layer1.raw_weight_o_i.data, "l1 weight")
        print(layer2.raw_weight_o_i.data, "l2 weight")
        optim.zero_grad()
        
        pred.backward(target)
        print(layer2.raw_weight_o_i.grad, "l2 weight.grad")
        optim.step()
        pass
    layer1.protect_raw_weight()
    layer2.protect_raw_weight()
    print(layer1.raw_weight_o_i.data, "l1 after")
    print(layer2.raw_weight_o_i.data, "l2 after")
    print(layer2(layer1(input)), "   the dense way handles this case properly.")
    pass

if 'xxxxxxxxxxxx not for 2.5 xxxxxxxxxxxxxx 2 layer real training WITHOUT deduplicating.' and False:
    gramo_for_each_output = False
    layer1 = DigitalMapper_v2_4(4,3,gramo_for_each_output)
    layer1.raw_weight_o_i.data = torch.tensor([[1.,0,0,0],[0., 1.,0,0],[0., 0.,1,0]])
    layer2 = DigitalMapper_v2_4(3,2,gramo_for_each_output)
    layer2.raw_weight_o_i.data = torch.tensor([[1., 0,0],[1., 0,0]])
    
    input = torch.tensor([[1.,1.,1., -1.]])#, requires_grad=True)
    target = torch.tensor([[1.,-0.5]]) 
    optim_them = []
    optim_them.extend(layer1.parameters())
    optim_them.extend(layer2.parameters())
    optim = torch.optim.SGD(optim_them, lr=0.1)
    
    print(layer1.raw_weight_o_i.data)
    print(layer2.raw_weight_o_i.data)
    for _ in range(10):
        #print(layer1.raw_weight_o_i.data)
        pred:torch.Tensor = layer2(layer1(input))
        optim.zero_grad()
        pred.backward(target)
        optim.step()
        pass
    print(layer1.raw_weight_o_i.data)
    print(layer2.raw_weight_o_i.data)
    print(layer2(layer1(input)), "   <-both should the same as the strongest answer")
    print("now uncomment the raise instruction.")
    pass

if 'training loop test.应该是正常了。' and False:
    batch = 100
    n_in = 10
    n_out = 5
    input, target = data_gen_for_directly_stacking_test(batch,n_in, n_out, no_duplicated = True)
    # print(input)
    # print(target)
    layer = DigitalMapper_v2_6(n_in, n_out, 0.5, False)
    optim = torch.optim.SGD(layer.parameters(), lr = 0.1)
    for epoch in range(10000):
        pred = layer(input)  
        (acc, perfect) = bitwise_acc(pred, target,print_out=True)
        if perfect:
            print("finished epoch:", epoch)
            break
        optim.zero_grad()      
        pred.backward(target)
        make_grad_noisy(layer, 1.1)
        optim.step()
        layer.deduplicate_v2_6()
        pass        
    pass        



#below are 2 stacking tests.


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
            gramo_for_each_output = False, \
            #auto_print_difference:bool = False, \
            device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.digital_mappers = torch.nn.ParameterList([])
        self.shape_config = shape_config
        self.gramo_for_each_output = gramo_for_each_output
        for i in range(shape_config.__len__()-1):
            in_features = shape_config[i]
            out_features = shape_config[i+1]
            self.digital_mappers.append(DigitalMapper_v2_6(in_features,out_features,
                gramo_for_each_output,debug_number_in_model=i))
            pass
        
        self.index_of_last_update = torch.nn.Parameter(torch.empty([shape_config.__len__()-1, shape_config[0]], dtype=torch.int64), requires_grad=False)
        self.index_of_last_init_ed = False
        pass

    def ______a():
        '''
    # def reset_scaling_ratio_for_raw_weight(self):
    #     layer:DigitalMapper_v2______
    #     for layer in self.digital_mappers:
    #         layer.reset_scaling_ratio_for_raw_weight()
    #         pass
    #     pass
    # def scale_the_scaling_ratio_for_raw_weight(self, by:float):
    #     layer:DigitalMapper_v2______
    #     for layer in self.digital_mappers:
    #         layer.scale_the_scaling_ratio_for_raw_weight(by)
    #         pass
    #     pass
    # def print_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01):
    #     temp_list:List[float] = []
    #     layer:DigitalMapper_v2______
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
            #print(layer.raw_weight_o_i.dtype, "  __line 898")
            x = layer(x)
        return x
    
    #def print_strong_grad_ratio(self):
    #    for layer in self.digital_mappers:
    def _________print_debug_info(self):
        with torch.no_grad():
            to_print = []
            layer:DigitalMapper_v2_5
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.max())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_v2_5
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.min())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_v2_5
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.isnan().sum())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_v2_5
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
            layer:DigitalMapper_v2_5
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
        layer:DigitalMapper_v2_5
        for i, layer in enumerate(self.digital_mappers):
            layer.gramo_for_raw_weight.set_scaling_factor(factor[i])
            pass
        fds=432

    def __save_max_index(self):
        r'''this function is only used in xxxx__print_diff. DO NOT directly call this.'''
        layer:DigitalMapper_v2_5
        with torch.no_grad():
            for i, layer in enumerate(self.digital_mappers):
                temp = layer.get_mapping_index_previous_o()
                self.index_of_last_update[i,:temp.shape[0]] = temp
                pass
            pass
        pass#end of function.

    def xxxx__print_diff(self, epoch:int):
        # self.index_of_last_update = torch.nn.Parameter(torch.empty([shape_config.__len__()-1, shape_config[0]], dtype=torch.int64), requires_grad=False)
        # self.index_of_last_init_ed = False
        with torch.no_grad():
            layer:DigitalMapper_v2_5
            printed_something = False
            if self.index_of_last_init_ed:
                already_printed_epoch = False
                for i, layer in enumerate(self.digital_mappers):
                    new_result = layer.get_mapping_index_previous_o()
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
            layer:DigitalMapper_v2_6 = self.digital_mappers[-1]
            #temp1 = layer.get_mapping_index_previous_o()
            temp1 = layer.get_plain_max_index_from_raw()
            previous_index = temp1.unique()
            layer_number = self.digital_mappers.__len__()-1
            if layer_number%10==0:
                print(f"(layer {layer_number}):", end="")
                pass
            print(f"{previous_index.shape[0]}, ", end="")

            for i in range(self.digital_mappers.__len__()-2,-1,-1):
                layer = self.digital_mappers[i]
                #temp1 = layer.get_mapping_index_previous_o()
                temp1 = layer.get_plain_max_index_from_raw()
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
    
    def _deduplicate(self, print_final_max_index = False, print_final_max_index_count = False):
        r'''only used in besides_stepping. Do not call this function directly.'''
    #I assume the input dim is always bigger than the output dim, because I believe this is the real case in the integrated situation.
    #If the in and out dims are the same, simply paste the input to the output, it should work perfectly.
    #if the input dim is smaller than output, why do I need this shape. This is the case in some test.
        layer:DigitalMapper_v2_6
        chosen_index:Tuple[torch.Tensor|None] = None
        for i in range(self.digital_mappers.__len__()-1,-1,-1):
            layer = self.digital_mappers[i]
            chosen_index = layer.deduplicate_v2_6(chosen_index)
            pass
        if print_final_max_index:
            print("print_final_max_index", chosen_index)
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

if 'deduplicating 2.6. 2.6版本是一个基于softmax的，deduplicate的行为不是那种很硬的。这个测试意义不大。' and False:
    #notice, in 2.5, the init makes sure the get_max_index returns non duplicating result.
    # so the unique function doesn't do anything, except for sorting.
    model = test_directly_stacking_multiple_digital_mappers([10,8,6,4])
    layer:DigitalMapper_v2_6 = model.digital_mappers[-1]
    #uniqued = layer.get_mapping_index_previous_o().unique()
    uniqued = layer.get_plain_max_index_from_raw().unique()
    print(uniqued)
    print()
    
    layer = model.digital_mappers[-2]
    pure_max = layer.get_plain_max_index_from_raw()
    print(pure_max)
    chosen_max = pure_max[uniqued]
    print(chosen_max)
    uniqued = chosen_max.unique()
    print(uniqued)
    print()
    
    layer = model.digital_mappers[-3]
    pure_max = layer.get_plain_max_index_from_raw()
    print(pure_max)
    chosen_max = pure_max[uniqued]
    print(chosen_max)
    uniqued = chosen_max.unique()
    print(uniqued)
    
    #now set breakpoint inside this function. The result may be a bit different, since the function deduplicates.
    model._deduplicate(print_final_max_index=True)
    pass

if 'xxxxxxxxxx not for 2.5 xxxxxxxxxxxxx 2 static methods test' and False:
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

if 'skipped. Maybe I should do it before move on.       print_max_index test'and False:
    the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(1111,111,21)
    model = test_directly_stacking_multiple_digital_mappers(the_config,0.5)
    #model._deduplicate()
    model._print_max_index_count()
    pass


fast_traval____direct_stack_test = 432
if 'direct stack test' and True:
    for _ in range(4):
        is_half = False#false(4,5,6,6),True(4k+<0.7)
        
        gramo_for_each_output = True#????????????false(3,5,5,7))true(4,5,5,6)
        re_rand_target_weight_every =5
        deduplicate_every = 1
        #????????????re:de(result)
        #1:1(59,64,64)2(64,110,110,140)
        #2:1(39,44,46,89)2(65,100,110,110)
        #3:1(41,44,51,120)
        #4:1(51,52,57,88)
        #5:1(39,41,51)2(49,72,86)
        #7:1(54,57,68,80)
        #10:1(52,56,71,120)
        
        is_cuda = True
        batch = 10000
        
        lr = 1.#10
        #11111111111111111111111
        #lr 0.1(50,51,51,60)1(10,11,13,13)10(4,5,5,5)100(4,4,5,6)1k(3,5,6,8)10k(3,5,9,10)100k(4,4,8,10)
        
        in_features = 100
        out_features = 20
        num_layers = 4
        #10/3/3(3,5,5,8)
        #10/5/3(6,10,13,14,unstable 0.9)
        #20/5/3(6,7,8,10)
        #50/5/3(4,5,6,7)
        #100/5/3(4,5,5,7)
        #100/10/3(13,15,15,22)
        #100/20/3(16,36,unstable 0.95)
        #100/20/4(<100, unstable 0.87)
        
        def print_config_after_finish():
            #print("re_rand_target_weight_every:", re_rand_target_weight_every, "/deduplicate_every:", deduplicate_every)
            #print("lr:", lr)
            print("in_features:", in_features, "/out_features:", out_features, "/num_layers:", num_layers)
            return
        
        pt = Print_Timing(first=1, max_gap=200, density=0.5)
        
        (input, target_ori) = data_gen_for_directly_stacking_test(batch, in_features, out_features, no_duplicated = True)
        target = target_ori
        #print(target)

        the_config = test_directly_stacking_multiple_digital_mappers.gen_shape_config(in_features, out_features, num_layers)
        model = test_directly_stacking_multiple_digital_mappers(the_config, gramo_for_each_output)
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
            model.besides_stepping(deduplicate = True,print_final_max_index_count=False)
            pass
        
        
        #print(model.digital_mappers[0].raw_weight_o_i.dtype, "  __line 1207")
        
        #################################################################training loop
        for epoch in range(1231231):
            if epoch%deduplicate_every == deduplicate_every-1:
                model.besides_stepping(deduplicate = True,print_all_max_index_count=False)
                #print("deduplicate")
                pass
            # re rand the temp weight for target
            if epoch%re_rand_target_weight_every == re_rand_target_weight_every-1:
                '''
                random_number = torch.rand([1,out_features], device=target_ori.device)
                random_number = torch.pow(random_number,3)
                '''
                random_number = torch.rand([1,out_features], device=target_ori.device)
                random_number = torch.pow(random_number,0.01)
                #0.01(11,16,20)0.1(13,14,22)1(15,17,23)2(19,22,40)10(34,36,40)
                target = target_ori*random_number
                pass
            #print(model.digital_mappers[0].raw_weight_o_i.dtype, "  __line 1207")
            
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
            optimizer.zero_grad()
            
            with torch.inference_mode():
                (acc, perfect) = bitwise_acc(pred, target)
                if perfect:
                    print("FINISHED, ep:", epoch+1)
                    print("FINISHED, ep:", epoch+1)
                    print("FINISHED, ep:", epoch+1)
                    print_config_after_finish()
                    print_config_after_finish()
                    print(pred[:1,:7], "pred", "    __line 1260")
                    print(target_ori[:1,:7], "target")
                    break
                if pt.check(epoch):
                    print(epoch+1, f"    ep/acc    {acc:.3f}    line 1260")
                    #model._print_max_index_count()
                    pass
                pass
            
            pred.backward(target)#intentional.
            # if epoch >1000:
            #     for layer in model.digital_mappers:
            #         print(layer.raw_weight_o_i.grad)
            #         pass        
            #     pass        
            
            if "print the grad" and False:
                if pt.check(epoch):
                    print(model.digital_mappers[0].raw_weight_o_i.grad, "0th layer    grad")
                    print(model.digital_mappers[1].raw_weight_o_i.grad, "1 layer    grad")
                    print(model.digital_mappers[-1].raw_weight_o_i.grad, "-1 layer    grad")
                    pass
                pass
            if "print the weight" and False:
                layer:DigitalMapper_v2_6 = model.digital_mappers[0]
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
    pass
    play_noise()

#result below~
'''




'''




class test_directly_stacking_multiple_digital_mappers_with_halfway_widen(torch.nn.Module):
    def __init__(self, width:int, extra_width:int, layer_count:int, \
            gramo_for_each_output = False, \
            device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.width = torch.nn.Parameter(torch.tensor([width],dtype=torch.int64), requires_grad=False)
        self.extra_width = torch.nn.Parameter(torch.tensor([extra_width],dtype=torch.int64), requires_grad=False)
        #self.total_width = torch.nn.Parameter(torch.tensor([width+extra_width],dtype=torch.int64), requires_grad=False)
        total_width = width+extra_width
        self.gramo_for_each_output = gramo_for_each_output
        
        self.digital_mappers = torch.nn.ParameterList([])
        for i in range(layer_count):
            self.digital_mappers.append(DigitalMapper_v2_6(total_width,width,
                gramo_for_each_output,debug_number_in_model=i))
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
            layer:DigitalMapper_v2_5
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.max())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_v2_5
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.min())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_v2_5
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.isnan().sum())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
            
            to_print = []
            layer:DigitalMapper_v2_5
            for layer in self.digital_mappers:
                to_print.append(layer.raw_weight_o_i.isinf().sum())
                pass
            pass
            for number in to_print:
                print(f"{number:.3f} ", end="")            
                pass
        
    def ________a():
        '''
    def print_non_zero_grad_ratio(self):
        with torch.no_grad():
            to_print = []
            layer:DigitalMapper_v2______
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
        layer:DigitalMapper_v2______
        for i, layer in enumerate(self.digital_mappers):
            layer.gramo_for_raw_weight.set_scaling_factor(factor[i])
            pass
        fds=432

    def __save_max_index(self):
        r''this function is only used in xxxx__print_diff. DO NOT directly call this.''
        layer:DigitalMapper_v2______
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
            layer:DigitalMapper_v2______
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
            layer:DigitalMapper_v2_5
            __rough_current_index:torch.Tensor = torch.empty([0],device=self.digital_mappers[0].raw_weight_o_i.device)
            is_last_layer = True
            for i in range(self.digital_mappers.__len__()-1,-1,-1):
                layer = self.digital_mappers[i]
                mapping_index_of_layer = layer.get_mapping_index_previous_o()
                if is_last_layer:
                    is_last_layer = False
                    __rough_current_index = mapping_index_of_layer.unique()
                else:
                    mapping_index_directly_from_layer = mapping_index_of_layer[useful_previous_index]
                    __rough_current_index = mapping_index_directly_from_layer.unique()
                    pass
                flag_useful = __rough_current_index.lt(self.width)
                useful_current_index = __rough_current_index[flag_useful]
                useful_previous_index = useful_current_index
                if i % 10 == 0:
                    print(f"(layer {i}):", end="")
                    pass
                print(f"{useful_current_index.shape[0]}, ", end="")
                #previous_index = current_index
                pass
            print()
            return
    #end of function.
    
    def _deduplicate(self, print_final_max_index_count = False):
        r'''only used in besides_stepping. Do not call this function directly.'''
    #I assume the input dim is always bigger than the output dim, because I believe this is the real case in the integrated situation.
    #If the in and out dims are the same, simply paste the input to the output, it should work perfectly.
    #if the input dim is smaller than output, why do I need this shape. This is the case in some test.
        layer:DigitalMapper_v2_5
        useful_chosen:Tuple[torch.Tensor|None] = None
        for i in range(self.digital_mappers.__len__()-1,-1,-1):
            layer = self.digital_mappers[i]
            chosen_index:torch.Tensor = layer.deduplicate_v2_5(useful_chosen)
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

if '_print_max_index_count test' and False:
    width = 4
    extra_width = 1
    model = test_directly_stacking_multiple_digital_mappers_with_halfway_widen(width,extra_width,4,0.05)
    model._print_max_index_count()
    pass


if 'basic shape test' and False:
    batch = 11
    width = 2
    extra_width = 3
    (input, target) = data_gen_for_directly_stacking_test(batch, width, width, no_duplicated = True)
    model = test_directly_stacking_multiple_digital_mappers_with_halfway_widen(width,extra_width,4,0.05)
    print(model(input))
    pass

if 'should be correct????? besides_stepping' and False:
    width = 5
    model = test_directly_stacking_multiple_digital_mappers_with_halfway_widen(width,5,3,0.05)
    layer:DigitalMapper_v2_5 = model.digital_mappers[-1]
    uniqued:torch.Tensor = layer.get_mapping_index_previous_o().unique()
    print(uniqued)
    flag_useful = uniqued.lt(width)
    print(flag_useful)
    useful_uniqued = uniqued[flag_useful]
    print(useful_uniqued)
    print()
    
    layer = model.digital_mappers[-2]
    pure_max = layer.get_mapping_index_previous_o()
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
    pure_max = layer.get_mapping_index_previous_o()
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

fast_traval____direct_stack_test____with_halfway_widen = 432
if 'direct stack test WITH HALFWAY WIDEN!!!' and True:
    for _ in range(4):
        is_half = False#False(65,230,240)True(1k+x2)
        batch = 1000##########################
        lr = 0.06#0.3#0.01#0.0001(2k+)0.001(5k+almost)
        #lr 0.01(760,1k8,2k2)0.02(120,140,160,420)0.03(86,87,160,220)0.06(60,64,85,100)0.1(70,76,140,210)0.3(97,160,210,280)1(2k+x2)
        is_cuda = True################
        re_rand_target_weight_every = 3
        deduplicate_every = 3
        #re:de
        #1: 1(84,230,260,430)2(75,84,88,98) 3(54,61,85,96) 10(44,60,95,120)
        #2: 1(              )                 3(53,59,98,140) 10(50,60,60,60)
        #3: 1(62,150,380,390)2(48,69,100,150)!3(38,56,81,84)4(48,65,69,180)5(44,79,82,85) 10(59,66,72,100)
        #4:                                   3(55,83,97,99)
        #5:                                   3(56,60,66,70)
        #10: 1(180,350,370,600)3(92,94,96,1120) 10(33,67,77,110)
        
        alpha = 0.003#0.1#0.0001(54,69,83,170)0.001(40,57,65,69,70,71,71,120)
        #alpha !0.003(38,39,43,45,61,63,64,83)0.006(51,54,95,140)
        #alpha 0.01 (38,45,45,48,53,63,73,93)0.03(47,53,58,63,69,71,110,160)
        #alpha 0.1  (48,51,60,64,65,72,76,84)1(58,68,98,100)
        
        gramo_for_each_output = True#True(36,56,86,99), False(1k+x2)
        
        width = 15
        extra_width = 50
        num_layers = 7
        #width/extra width/layers
        #15/100/4(36,56,86,99)
        #15/100/6(88,110,140,270)
        #15/100/8(330,410,570)
                
        def print_config_after_finish():
            #print("re_rand_target_weight_every:", re_rand_target_weight_every, "/deduplicate_every:", deduplicate_every)
            #print("lr:", lr)
            print("alpha:", alpha)
            #print("width:", width, "/extra_width:", extra_width, "/num_layers:", num_layers)
            return

        pt = Print_Timing(first=1, max_gap=200, density=1)
        
        (input, target_ori) = data_gen_for_directly_stacking_test_same_dim_no_duplicated(batch, width)
        target = target_ori
        #print(target)

        model = test_directly_stacking_multiple_digital_mappers_with_halfway_widen(width,extra_width,num_layers,alpha, gramo_for_each_output)
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
            model.besides_stepping(deduplicate = True,print_final_max_index_count=False)
            pass
        
        #################################################################training loop
        for epoch in range(1231231):
            if epoch%deduplicate_every == deduplicate_every-1:
                model.besides_stepping(deduplicate = True,print_all_max_index_count=False)
                #print("deduplicate")
                pass
            # re rand the temp weight for target
            if epoch%re_rand_target_weight_every == re_rand_target_weight_every-1:
                random_number = torch.rand([1,width], device=model.digital_mappers[0].raw_weight_o_i.device)
                random_number = torch.pow(random_number,0.3)
                #0.01(150,250,320,550,)0.1(130,140,210,230)0.3(100,130,180,180)
                #1(120,190,200,250)3(230,330,330,360)
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
            
            
            
            #if epoch>1000 and epoch%200 == 0:
            # layer:DigitalMapper_v2_4
            # print(epoch, "      ",epoch, "      ",epoch)
            # for layer in model.digital_mappers:
            #     print(layer.raw_weight_o_i.data)
            #     pass
            #pass
            
            
            
            if True and "print acc":
                with torch.inference_mode():
                    (acc, perfect) = bitwise_acc(pred, target)
                    if perfect:
                        print("FINISHED, ep:", epoch+1)
                        print("FINISHED, ep:", epoch+1)
                        print("FINISHED, ep:", epoch+1)
                        print_config_after_finish()
                        print_config_after_finish()
                        print(target_ori[:1,:7], "target")
                        print(pred[:1,:7], "pred", "    __line 1757")
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
                layer:DigitalMapper_v2______ = model.digital_mappers[0]
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
        
        # for layer in model.digital_mappers:
        #     print(layer.raw_weight_o_i)
    
        pass
    pass
    play_noise()
    
'''
test round 2
I went through all the test from top to bottom. So below relies on upper, but upper relies on old below.
#################################################
is_half = False#False(65,230,240)True(1k+x2)
batch = 1000##########################

lr = 0.06#0.3#0.01#0.0001(2k+)0.001(5k+almost)
#lr 0.01(760,1k8,2k2)0.02(120,140,160,420)0.03(86,87,160,220)0.06(60,64,85,100)0.1(70,76,140,210)0.3(97,160,210,280)1(2k+x2)
    
is_cuda = True################
re_rand_target_weight_every = 3
deduplicate_every = 3
#re:de
#1: 1(84,230,260,430)2(75,84,88,98) 3(54,61,85,96) 10(44,60,95,120)
#2: 1(              )                 3(53,59,98,140) 10(50,60,60,60)
#3: 1(62,150,380,390)2(48,69,100,150)!3(38,56,81,84)4(48,65,69,180)5(44,79,82,85) 10(59,66,72,100)
#4:                                   3(55,83,97,99)
#5:                                   3(56,60,66,70)
#10: 1(180,350,370,600)3(92,94,96,1120) 10(33,67,77,110)
#inf: 1() 3() 10()

alpha = 0.003#0.1#0.0001(54,69,83,170)0.001(40,57,65,69,70,71,71,120)
#alpha !0.003(38,39,43,45,61,63,64,83)0.006(51,54,95,140)
#alpha 0.01 (38,45,45,48,53,63,73,93)0.03(47,53,58,63,69,71,110,160)
#alpha 0.1  (48,51,60,64,65,72,76,84)1(58,68,98,100)

gramo_for_each_output = True#True(36,56,86,99), False(1k+x2)

width = 15
extra_width = 50
num_layers = 8
#width/extra width/layers
#15/100/4(36,56,86,99)
#15/100/6(88,110,140,270)
#15/100/8(330,410,570)



test round 1
I went through all the test from top to bottom. So below relies on upper, but upper relies on old below.
#################################################
is_half = False#False(570,2k1,2k7)True(no good, acc<0.55)
batch = 1000##########################
lr = 0.3#0.01#0.1#0.0001(very slow,3k+2x,acc<0.87)0.001(3k2,4k+almost)0.01(180,190,320,1k2)
#lr 0.03(78,160,270,1k4,)0.1(30,36,68,110)0.3(17,22,24,54)0.6(10,69,94,160)
# 1(13,20,120,340)10(36,130,140,820)
is_cuda = True################
re_rand_target_weight_every = 2#2
deduplicate_every = 3#3
#re:de
#1: 1(24,26,79,120) 2(21,22,29,47) 3(12,30,34,48) 5(13,29,31,34)
#2: 1(21,37,110,130)2(33,47,52,76) 3(19,24,29,32) 5(10,16,38,58)
#5: 1(14,43,61,80)  2(28,30,34,110)3(15,27,83,120)5(17,23,30,130)
#10:1(19,36,51,230) 2(17,35,81,92) 3(14,18,67,74) 5(10,24,58,220)
#inf:1(17,25,28,57) 2()          3(10,15,1k+x2) 5()

alpha = 0.1#0.1#0.0001(12,22,150,220)0.0003(22,25,28,130)0.001(13,18,21,33)0.003(10,37,46,49)
#alpha 0.01(15,19,19,26)0.03(14,18,25,26)0.1(12,14,18,24)0.3(18,28,28,44)
#alpha 1(12,22,35,35)3(14,69,180,330)10(99,160,240,280)
gramo_for_each_output = True#False(not stable), True(stable?)

width = 15
extra_width = 50
num_layers = 4
#width/extra width/layers
#15/100/4(for the round 2)

#5/10/5(20,27,37,37)
#5/10/6(40,120,150,460)
#5/10/7(49,58,61,140)
#5/10/8(1k4,3k+x2)
#5/10/10(3k+)
#10/10/10(3k+)

#10/5/5(20,23,26,93)
#15/5/5(36,39,74,99)
#20/5/5(31,41,60,78)
#25/5/5(130,200,500,1k)
#30/5/5(120,480,540,1k5)

#5/10/5(25,26,30,32)
#5/20/5(21,22,22,50)
#5/30/5(23,28,28,61)
#5/50/5(85,120,270,490)
#5/70/5(24,130,150,180)

#10/10/5(22,31,72,84)
#15/15/5(33,57,74,150)
#20/20/5(320,340,350,670)
#25/25/5(3k+x2)

#7/7/4(20,21,26,32)
#10/10/4(30,32,35,54)
#10/10/7(74,88,93,100)
#15/10/7(140,240,500,590)
#20/15/7(,720,1k3,3k+)
#10/25/7(130,4k+)

'''
    