from typing import Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, _either_1_or_neg1, _tensor_shape_check, \
        iota, str_the_list
from pytorch_yagaodirac_v2.Random import rand_sign
from DNN2026.DNN_util import _test___DNN_forward___full_safety, _test___binary_accuracy___full_safety, \
        partly_reasonable_label_from_input
import torch
#from DNN2026.DNN_util import 
def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######



'''This file is only a release version of this DigitalMapper_layer__2026 class. 
All the test are done in the prototype test file.'''








'''auto grad function class'''
'''auto grad function class'''
'''auto grad function class'''
class autograd_function_class_for__DigitalMapper_layer__2026(torch.autograd.Function):
    r'''
    forward input list:
    >>> input_posneg1___b_i(if the input is not posneg1, then the output is not guarunteed to be posneg1)
    >>> raw_weight___o_i (make sure this is output of get_useful())
    >>> SOME_HYPER_PARAM___s(>0.)
    
    backward input list:
    >>> target___b_o (shape must be [batch, out_features]. can be any value.)
    '''
    @staticmethod
    def forward(input_posneg1___b_i:torch.Tensor, raw_weight___o_i:torch.Tensor, 
                SOME_HYPER_PARAM___s:torch.Tensor)->torch.Tensor:
        # input___b_i:torch.Tensor = args[0]# shape must be [batch, in_features]
        # raw_weight___o_i:torch.Tensor = args[1]# shape must be [out_features, in_features]

        #<  safety
        #the only safety is if the input is already posneg1( positive or negative 1). 
        # I plan to do it outside in the module class.
        

        #<  real payload
        #copied from the function _test___DNN_forward___full_safety section "real payload"
        index_of_max_of_raw_weight___o = raw_weight___o_i.max(dim=1).indices
        output_posneg1___b_o = input_posneg1___b_i[:, index_of_max_of_raw_weight___o]

        return output_posneg1___b_o

    @staticmethod
    def setup_context(ctx, inputs, output)->None:
        input_posneg1___b_i:torch.Tensor = inputs[0]
        raw_weight___o_i:torch.Tensor = inputs[1]
        SOME_HYPER_PARAM___s:torch.Tensor = inputs[2]
        #output___b_o:torch.Tensor = output
        ctx.save_for_backward(input_posneg1___b_i, raw_weight___o_i, SOME_HYPER_PARAM___s)

    @staticmethod
    def backward(ctx:Any,  *grad_outputs:Any) -> Any:    #tuple[torch.Tensor|None, torch.Tensor|None]: 
        #copied from function _algo_test__backward_function
        #some safety checks are removed.
        #shape of g_in must be [batch, out_features]
        input_posneg1___b_i:torch.Tensor
        raw_weight___o_i:torch.Tensor
        target___b_o:torch.Tensor = grad_outputs[0]

        #debug code         ##################################################################################################################################
        assert isinstance(target___b_o, torch.Tensor)

        (input_posneg1___b_i, raw_weight___o_i, SOME_HYPER_PARAM___s) = ctx.saved_tensors
        #debug code         ##################################################################################################################################


        #<  init results to None
        grad_like_for___input___b_i     :torch.Tensor|None = None
        grad_like_for___raw_weight___o_i:torch.Tensor|None = None

        #<  real payload
        if "raw_weight___o_i.requires_grad" or "input___b_i.requires_grad":
            target___b_o_EXPANDi = target___b_o.reshape(shape=[target___b_o.shape[0], target___b_o.shape[1], 1]). \
                    expand(size=[-1, -1, input_posneg1___b_i.shape[1]])
            pass# if "raw_weight___o_i.requires_grad" or "input___b_i.requires_grad":


        if "raw_weight___o_i.requires_grad":
            input_posneg1___b_oEXPAND_i = input_posneg1___b_i.reshape(shape=[input_posneg1___b_i.shape[0], 1, input_posneg1___b_i.shape[1]]). \
                    expand(size=[-1, target___b_o.shape[1], -1])

            grad_like_for___raw_weight___before_sum___b_o_i = input_posneg1___b_oEXPAND_i*target___b_o_EXPANDi
            del input_posneg1___b_oEXPAND_i
            grad_like_for___raw_weight___o_i = grad_like_for___raw_weight___before_sum___b_o_i.sum(dim=0)
            del grad_like_for___raw_weight___before_sum___b_o_i

            #控制变量的范围是优化器的事情.
            pass# if "raw_weight___o_i.requires_grad":


        if "input___b_i.requires_grad":
            
            #<  recomputation
            # output_posneg1___b_o, index_of_max_of_raw_weight___o = _test___DNN_forward___full_safety( \
            #         input_posneg1___b_i = input_posneg1___b_i, raw_weight___o_i = raw_weight___o_i, input_is_already_posneg1=True)
            #manually unroll.
            index_of_max_of_raw_weight___o = raw_weight___o_i.max(dim=1).indices
            output_posneg1___b_o = input_posneg1___b_i[:, index_of_max_of_raw_weight___o]
            #del index_of_max_of_raw_weight___o later


            #<  accuracy
            # accuracy___o, return_value_name = _test___binary_accuracy___full_safety(target___b_o=target___b_o, 
            #                                         output_posneg1___b_o=output_posneg1___b_o, mean_per= "per_output" )
            # assert return_value_name == "accuracy___o"
            # assert _tensor_shape_check(accuracy___o, out_dim)

            target_posneg1___b_o = target___b_o.gt(0.).to(torch.float32)*2. -1.  # DNN___to_posneg1
            assert _either_1_or_neg1(target_posneg1___b_o)######################################################################################

            # _test___binary_accuracy___full_safety
            element_mul_of_target_and_output___b_o = target_posneg1___b_o * output_posneg1___b_o
            element_mul_of_target_and_output___b_o = element_mul_of_target_and_output___b_o.to(torch.float32)
            accuracy___o = element_mul_of_target_and_output___b_o.mean(dim=0)
            del target_posneg1___b_o, output_posneg1___b_o, element_mul_of_target_and_output___b_o
            accuracy___o = (accuracy___o +1.)*0.5

            # assert False, "the sharpness-controlled softmax also is not tested."
            #<  soft part of the backward mapping.
            assert SOME_HYPER_PARAM___s > 0.#############################################################移出去
            sharpen_factor__from_accuracy___o:torch.Tensor = accuracy___o*SOME_HYPER_PARAM___s
            assert sharpen_factor__from_accuracy___o.dtype == accuracy___o.dtype##################################################
            assert sharpen_factor__from_accuracy___o.ge(0.).all()##########################################################
            sharpen_factor__from_accuracy___o_EXPANDi = sharpen_factor__from_accuracy___o. \
                    reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])
            sharpened_raw_weight___o_i = raw_weight___o_i*sharpen_factor__from_accuracy___o_EXPANDi
            soft_part_of_the_one_hot___o_i = sharpened_raw_weight___o_i.softmax(dim=1)
            del sharpen_factor__from_accuracy___o, sharpen_factor__from_accuracy___o_EXPANDi
            del sharpened_raw_weight___o_i
            #   a bit assertion       ###############################################################################
            _assert_only___sum_of_soft_part_of_the_one_hot___o = soft_part_of_the_one_hot___o_i.sum(dim=1)
            assert _tensor_equal(_assert_only___sum_of_soft_part_of_the_one_hot___o, 
                                    torch.ones_like(_assert_only___sum_of_soft_part_of_the_one_hot___o))

            #<  hard part of the backward mapping.
            out_dim = target___b_o.shape[1]
            iota_of_out = torch.linspace(start=0,end=out_dim-1,steps=out_dim,
                                            dtype=torch.int32, device=target___b_o.device)
            hard_part_of_the_one_hot___o_i = torch.zeros_like(soft_part_of_the_one_hot___o_i, device=target___b_o.device)
            hard_part_of_the_one_hot___o_i[iota_of_out, index_of_max_of_raw_weight___o] = 1.
            del index_of_max_of_raw_weight___o, iota_of_out
            #   a bit assertion      ###############################################################################
            _assert_only___sum_of_hard_part_of_the_one_hot___o = soft_part_of_the_one_hot___o_i.sum(dim=1)
            assert _tensor_shape_check(_assert_only___sum_of_hard_part_of_the_one_hot___o, out_dim)
            assert _tensor_equal(_assert_only___sum_of_hard_part_of_the_one_hot___o, 
                                    torch.ones_like(_assert_only___sum_of_hard_part_of_the_one_hot___o))

            assert _tensor_equal(_assert_only___sum_of_hard_part_of_the_one_hot___o, 
                                    torch.ones_like(_assert_only___sum_of_hard_part_of_the_one_hot___o))
            
            #linear interpolation.
            accuracy___o_EXPANDi = accuracy___o.reshape(shape=[-1, 1]).expand(size=[-1, raw_weight___o_i.shape[1]])
            #   accurate output makes the corresponding target go through a  sharp  mapping. 
            # inaccurate output makes the corresponding target go through a blurred mapping. 
            the_one_hot_like___o_i =       accuracy___o_EXPANDi  * hard_part_of_the_one_hot___o_i + \
                                (1. - accuracy___o_EXPANDi) * soft_part_of_the_one_hot___o_i
            del accuracy___o, accuracy___o_EXPANDi
            del hard_part_of_the_one_hot___o_i, soft_part_of_the_one_hot___o_i
            
            #the backward mapping relationship
            the_one_hot___EXPANDb_o_i = the_one_hot_like___o_i. \
                    reshape(shape=[1, the_one_hot_like___o_i.shape[0], the_one_hot_like___o_i.shape[1]]). \
                    expand(size=[target___b_o_EXPANDi.shape[0], -1, -1])
            
            grad_like_for___input___before_sum___b_o_i = the_one_hot___EXPANDb_o_i*target___b_o_EXPANDi
            del the_one_hot_like___o_i, the_one_hot___EXPANDb_o_i
            grad_like_for___input___b_i = grad_like_for___input___before_sum___b_o_i.sum(dim=1)
            del grad_like_for___input___before_sum___b_o_i
            assert isinstance(grad_like_for___input___b_i, torch.Tensor)

            #更精细的层间控制是gramo的事情。（如果你不熟悉gramo，不一定是某一个特定的gramo，不一定是绿色五角星那个）
            pass

        return grad_like_for___input___b_i, grad_like_for___raw_weight___o_i, None # 手动维护类型吧。实在写不来了。。
    
    
    pass  # class








'''2个申请内存的函数单独拿出来，方便以后调整。'''
def _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in(
        extra_in_dim:int, 
        in_dim_now:int, out_dim_now:int, )->int:
    '''return new_in_dim'''
    total_in_dim_needed = extra_in_dim+in_dim_now
    min_new_nelement = total_in_dim_needed*out_dim_now
    ONE_M = 1<<20
    if min_new_nelement<(ONE_M):
        return total_in_dim_needed*2
    ONE_G = 1<<30
    if min_new_nelement<(ONE_G):
        return int(total_in_dim_needed*1.25)
    return int(total_in_dim_needed*1.1)
    #end of function

def _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out(
        extra_out_dim:int, 
        in_dim_now:int, out_dim_now:int, )->int:
    total_out_dim_needed = extra_out_dim+out_dim_now
    min_new_nelement = in_dim_now*total_out_dim_needed
    ONE_M = 1<<20
    if min_new_nelement<(ONE_M):
        return total_out_dim_needed*2
    ONE_G = 1<<30
    if min_new_nelement<(ONE_G):
        return int(total_out_dim_needed*1.25)
    return int(total_out_dim_needed*1.1)
    #end of function

'''随机初始化的的函数，单独拿出来，方便以后调整'''
def _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style( \
        out_features:int, in_features:int, device = None, dtype = None) -> torch.Tensor:
    result = torch.rand(size=[out_features, in_features], device=device, dtype=dtype)*-1.
    return result



from collections.abc import Iterator
class DigitalMapper_layer__2026(torch.nn.Module):
    in_dim         :int
    out_dim        :int
    _init_to_nan   :bool
    _raw_weight___oCAP_iCAP     :torch.nn.parameter.Parameter
    some_hyper_param            :torch.nn.parameter.Parameter
    _always_check_input_is_posneg1__in_forward :bool

    #customizable functions.
    _random_init_algo               :function
    _calc_bigger_capacity__for_in   :function
    _calc_bigger_capacity__for_out  :function
    # _calc_bigger_capacity

    def __init__(self, in_features: int, out_features: int, some_hyper_param = 1., 
                init_capacity__for_in = 16, init_capacity__for_out = 16, init_to_nan = True, \
                _dtype_for_raw_weight = torch.float32, 
                _always_check_input_is_posneg1__in_forward = True, 
                    device=None,
                    #dtype=None
                    ) -> None:  
        
        #this dtype is only for a inner memory in training. It must be float point number.
        #<  pytorch format
        #factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #<  safety
        assert _dtype_for_raw_weight in [torch.float, torch.float32, torch.float64, torch.float16, torch.bfloat16]
        if init_capacity__for_in < in_features:
            init_capacity__for_in = in_features
            pass
        if init_capacity__for_out < out_features:
            init_capacity__for_out = out_features
            pass

        self.in_dim = in_features
        self.out_dim = out_features
        self._init_to_nan = init_to_nan
        self._always_check_input_is_posneg1__in_forward = _always_check_input_is_posneg1__in_forward
        self._raw_weight___oCAP_iCAP = torch.nn.Parameter(torch.empty(
                size=[init_capacity__for_out, init_capacity__for_in], dtype=_dtype_for_raw_weight, device=device, 
                        requires_grad = True))#, **factory_kwargs), 
        
        
        assert self._raw_weight___oCAP_iCAP.dtype in [torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16]
        if self._init_to_nan:
            with torch.no_grad():
                self._raw_weight___oCAP_iCAP.fill_(torch.nan)
                pass
            pass# if self._init_to_nan:

        if isinstance(some_hyper_param, float):
            self.some_hyper_param = torch.nn.Parameter(torch.tensor(some_hyper_param, dtype=torch.float64, device=device, 
                        requires_grad = False),#, **factory_kwargs), 
                        requires_grad = False)
            pass
        elif isinstance(some_hyper_param, torch.Tensor):
            self.some_hyper_param = torch.nn.Parameter(some_hyper_param.detach().clone(), requires_grad = False)
            pass
        self.some_hyper_param.data = self.some_hyper_param.to(self._raw_weight___oCAP_iCAP.device)
        #if this is a higher precision, the final result may get effected. It doesn't help. So let's keep it simple.
        self.some_hyper_param.data = self.some_hyper_param.to(self._raw_weight___oCAP_iCAP.dtype)
        assert self.some_hyper_param.data.requires_grad == False
        assert self.some_hyper_param.shape.__len__() == 0#not important

        #<  modulized functions.
        self._random_init_algo = _only_for_DigitalMapper_layer__2026_to_use__reset_parameters__the_plain_rand01_style
        with torch.no_grad():
            self._raw_weight___oCAP_iCAP[:self.out_dim, :self.in_dim] = \
                    self._random_init_algo(out_features, in_features, 
                            device=device, dtype=self._raw_weight___oCAP_iCAP.dtype)
            pass
        self._calc_bigger_capacity__for_in = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_in
        self._calc_bigger_capacity__for_out = _only_for_DigitalMapper_layer__2026_to_use____calc_bigger_capacity__for_out
        pass

    '''parameters function.         This only gives out the raw_weight___o_i.'''
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        r"""This only gives out the raw_weight___o_i.

        Copied from pytorch code."""
        for param in [self._raw_weight___oCAP_iCAP]:
            yield param


    '''plain shape related.'''
    def capacity_of_in_dim(self)->int:
        '''get'''
        return self._raw_weight___oCAP_iCAP.shape[1] 
    def capacity_of_out_dim(self)->int:
        '''get'''
        return self._raw_weight___oCAP_iCAP.shape[0] 

    
    if "idk if it's still useful" and False:

        def get_one_hot_format(self)->torch.Tensor:
            iota_of_out_dim___o = iota(self.out_dim)
            index_of_max___o = self.raw_weight.max(dim=1).indices

            one_hot___o_i = torch.zeros_like(self._raw_weight___)
            one_hot___o_i[iota_of_out_dim___o, index_of_max___o] = 1.
            assert False, "untested."
            return one_hot___o_i

        def debug_get_zero_grad_ratio(self, directly_print_out:float = False)->float:
            with torch.no_grad():
                result = 0.
                if not self.raw_weight.grad is None:
                    flags = self.raw_weight.grad.eq(0.)
                    total_amount = flags.sum().item()
                    result = float(total_amount)/self.raw_weight.nelement()
                if directly_print_out:
                    print("get_zero_grad_ratio:", result)
                return result


        def debug_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01, \
                                    print_out = False)->float:
            #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
            if self.raw_weight.grad is None:
                if print_out:
                    print(0., "inside debug_micro_grad_ratio function __line 1082")
                    pass
                return 0.

            the_device=self.raw_weight.device
            epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
            raw_weight_abs = self.raw_weight.abs()
            flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

            epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
            raw_weight_grad_abs = self.raw_weight.grad.abs()
            flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

            ten = torch.tensor([10.], device=the_device)
            log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
            corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
            flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

            flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
            result = (flag_useful_g.sum().to(torch.float32)/self.raw_weight.nelement()).item()
            if print_out:
                print(result, "inside debug_micro_grad_ratio function __line 1082")
                pass
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
            return
        pass

    '''forward function for neural net like.'''
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''input.shape must be [ batch, _ ]. Use the input container.'''
        #<  rename with the shape
        #<  safety
        input___b_i = input
        if self._always_check_input_is_posneg1__in_forward:
            assert _either_1_or_neg1(input___b_i)
            pass

        #<  real payload
        output___b_o = autograd_function_class_for__DigitalMapper_layer__2026.apply(input___b_i, 
                                        self.get_useful_part_of_raw_weight(), self.some_hyper_param)

        return output___b_o
        #end of function.
    
    '''index tool'''
    def get_max_index(self)->torch.Tensor:
        '''return index_of_max_of_useful_part___o
        '''
        _temp_useful_part = self.get_useful_part_of_raw_weight()
        index_of_max_of_useful_part___o = _temp_useful_part.max(dim=1).indices#copied from _test___DNN_forward___full_safety
        return index_of_max_of_useful_part___o

    def backward_index_quiry(self, output_slot_list:torch.Tensor)->torch.Tensor:
        index_of_max_of_useful_part___o = self.get_max_index()
        result = index_of_max_of_useful_part___o[output_slot_list]
        assert False, "untested      都必须比in dim小"
        return result


    ''' get useful part         squeeze'''
    def get_useful_part_of_raw_weight(self)->torch.Tensor:
        result = self._raw_weight___oCAP_iCAP[:self.out_dim,:self.in_dim]
        return result
    def _get_useful_part_of_raw_weight_grad(self)->torch.Tensor|None:
        if self._raw_weight___oCAP_iCAP.grad is None:
            return None
        result = self._raw_weight___oCAP_iCAP.grad[:self.out_dim,:self.in_dim]
        return result
    def set_useful_part_of_raw_weight(self, input:torch.Tensor, no_grad = True)->None:
        assert input.shape == torch.Size([self.out_dim, self.in_dim])
        if no_grad:
            with torch.no_grad():
                self._raw_weight___oCAP_iCAP[:self.out_dim,:self.in_dim] = input
                return
            pass
        else:#with grad
            self._raw_weight___oCAP_iCAP[:self.out_dim,:self.in_dim] = input
            return
        #end of function.    
    def get_useful_part_of_raw_weight___and_squeeze(self, squeeze_in = False, squeeze_out = False)->torch.Tensor:
        self._squeeze(squeeze_in = squeeze_in, squeeze_out = squeeze_out)
        #result = self._raw_weight___oCAP_iCAP[:self.out_dim,:self.in_dim]
        return self.get_useful_part_of_raw_weight()
    
    def _squeeze(self, squeeze_in = False, squeeze_out = False):
        '''This function is designed for inner use inside this class. 

        If you need to control the timing and you know what you are doing, feel free to do anything.'''
        #<  safety
        assert squeeze_in or squeeze_out, "No real payload is asked. Why do you call this function? Or if you know what you are doing, comment this line out."

        #<  real payload
        # calc new capacity.
        _temp_new_out_capacity = self.capacity_of_out_dim()
        if squeeze_out:
            _temp_new_out_capacity = self.out_dim
            pass
        _temp_new_in_capacity = self.capacity_of_in_dim()
        if squeeze_in:
            _temp_new_in_capacity = self.in_dim
            pass

        _temp_new_memory = torch.empty(size=[_temp_new_out_capacity,_temp_new_in_capacity], 
                    dtype=self._raw_weight___oCAP_iCAP.dtype, device=self._raw_weight___oCAP_iCAP.device)
        if self._init_to_nan:
            _temp_new_memory.fill_(torch.nan)
            pass
        _temp_new_memory[:self.out_dim, :self.in_dim] = self._raw_weight___oCAP_iCAP.data[:self.out_dim, :self.in_dim]
        with torch.no_grad():
            self._raw_weight___oCAP_iCAP.data = _temp_new_memory
            pass
        return

    '''add input slot'''
    def add_input_slot__to_the_tail(self, how_many = 0, new_raw_weight_part:torch.Tensor = torch.empty(size=[0]))->None:
        '''The param combination is either (0, some tensor), or (some number, empty tensor). 
        
        If new_raw_weight_part is not empty, its shape must be [out_dim, extra_in_dim]'''
        #<  wash the param.
        if how_many == 0:
            if new_raw_weight_part.nelement() == 0:
                assert False, "Bad param combination. Either how_many > 0, or new_raw_weight_part is provided."
                pass# if new_raw_weight_part.nelement() == 0:
            assert new_raw_weight_part.shape.__len__() == 2
            how_many = new_raw_weight_part.shape[1]
            pass# if how_many == 0:
        else:# how_many != 0:
            #assert how_many>0#duplicated. 
            assert new_raw_weight_part.nelement() == 0, "Bad param combination. Both are provided. Remove one of them."

            new_raw_weight_part = self._random_init_algo(self.out_dim, how_many, device=self._raw_weight___oCAP_iCAP.device, dtype=self._raw_weight___oCAP_iCAP.dtype)
            pass# else of if how_many == 0:
        assert new_raw_weight_part.shape[0] == self.out_dim

        
        #<  real payload

        with torch.no_grad():
                
            _size_after = self.in_dim + how_many
            if _size_after > self.capacity_of_in_dim():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity__for_in(
                        extra_in_dim = how_many, in_dim_now = self.in_dim, out_dim_now = self.out_dim)

                _temp___new_container = torch.empty(size=[self._raw_weight___oCAP_iCAP.shape[0], _temp___new_capacity],
                        dtype=self._raw_weight___oCAP_iCAP.dtype, device=self._raw_weight___oCAP_iCAP.device)
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:self.out_dim, :self.in_dim] = self.get_useful_part_of_raw_weight()
                self._raw_weight___oCAP_iCAP.data = _temp___new_container
                pass

            self._raw_weight___oCAP_iCAP.data[:self.out_dim, self.in_dim:self.in_dim + how_many] = new_raw_weight_part
            self.in_dim = _size_after
            return
            #end of function


    '''output slot'''
    def add_output_slot__to_the_tail(self, how_many = 0, new_raw_weight_part:torch.Tensor = torch.empty(size=[0]))->None:
        '''The param combination is either (0, some tensor), or (some number, empty tensor). 
        
        If new_raw_weight_part is not empty, its shape must be [out_dim, extra_in_dim]'''
        #<  wash the param.
        if how_many == 0:
            if new_raw_weight_part.nelement() == 0:
                assert False, "Bad param combination. Either how_many > 0, or new_raw_weight_part is provided."
                pass# if new_raw_weight_part.nelement() == 0:
            assert new_raw_weight_part.shape.__len__() == 2
            how_many = new_raw_weight_part.shape[0]
            pass# if how_many == 0:
        else:# how_many != 0:
            #assert how_many>0#duplicated. 
            assert new_raw_weight_part.nelement() == 0, "Bad param combination. Both are provided. Remove one of them."

            new_raw_weight_part = self._random_init_algo(how_many, self.in_dim, device=self._raw_weight___oCAP_iCAP.device, dtype=self._raw_weight___oCAP_iCAP.dtype)
            pass# else of if how_many == 0:
        assert new_raw_weight_part.shape[1] == self.in_dim
        
        #<  real payload

        with torch.no_grad():
                
            _size_after = self.out_dim + how_many
            if _size_after > self.capacity_of_out_dim():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity__for_out(
                        extra_out_dim = how_many, in_dim_now = self.in_dim, out_dim_now = self.out_dim)

                _temp___new_container = torch.empty(size=[_temp___new_capacity, self._raw_weight___oCAP_iCAP.shape[1]],
                        dtype=self._raw_weight___oCAP_iCAP.dtype, device=self._raw_weight___oCAP_iCAP.device)
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:self.out_dim, :self.in_dim] = self.get_useful_part_of_raw_weight()
                self._raw_weight___oCAP_iCAP.data = _temp___new_container
                pass

            self._raw_weight___oCAP_iCAP.data[self.out_dim:self.out_dim + how_many, :self.in_dim] = new_raw_weight_part
            self.out_dim = _size_after
            return
        pass# a dead pass to denote the end of function

    def keep_output_slot(self, keep_which:torch.Tensor, squeeze_the_input_dim = False)->None:
        '''This function also squeeze the memory to minimum.'''
        assert keep_which.shape.__len__() == 1
        assert keep_which.dtype == torch.bool
        #<  real payload
        with torch.no_grad():

            _temp__useful_part = self.get_useful_part_of_raw_weight()
            _temp__useful_part = _temp__useful_part[keep_which,:]
            if squeeze_the_input_dim:
                self._raw_weight___oCAP_iCAP.data = _temp__useful_part
                pass
            else:# not to squeeze the input dim.
                _temp___keep_which___in_int = keep_which.to(torch.int32)
                how_many_to_keep = int(_temp___keep_which___in_int.sum().to(torch.int32).item())
                self._raw_weight___oCAP_iCAP.data = torch.empty(size=[how_many_to_keep, self.capacity_of_in_dim()])
                if self._init_to_nan:
                    self._raw_weight___oCAP_iCAP.data.fill_(torch.nan)
                    pass
                self._raw_weight___oCAP_iCAP.data[:, :self.in_dim] = _temp__useful_part
                pass
            self.out_dim = self._raw_weight___oCAP_iCAP.shape[0]
            return 
        #end of function
    def remove_output_slot(self, remove_which:torch.Tensor, squeeze_the_input_dim = False)->None:
            self.keep_output_slot(remove_which.logical_not(), 
                    squeeze_the_input_dim = squeeze_the_input_dim)
            return

    '''stringify'''
    def extra_repr(self) -> str:
        if self._always_check_input_is_posneg1__in_forward:
            return f'Output is pos/neg 1. In_features={self.in_features}, out_features={self.out_features}'
        return f'In_features={self.in_features}, out_features={self.out_features}'
        

    # def __repr__(self):
    #     return f"{self.get_useful().__repr__()}, size:{self._size}, DNN input container 2026"
    # def __str__(self):
    #     return f"{self.get_useful().__str__()}, size:{self._size}, DNN input container 2026"

    pass# end of class.




'''the optimizer'''
'''the optimizer'''
'''the optimizer'''
class optim_for___DigitalMapper_layer__2026(torch.nn.Module):#torch.optim.Optimizer):
    '''I need the useful shape information.
    The torch.optim.Optimizer only accepts torch.Tensor. It needs a lot hack
    to get this DigitalMapper_layer__2026 to work along with it.
    The reason to choose torch.nn.Module is I want the convenience 
    when I move it between devices and save/load it.'''

    learning_rate___s:torch.nn.Parameter
    digitalmapping_layers:torch.nn.ParameterList
    epsilon:torch.nn.Parameter
    def __init__(self, DigitalMapper_layers:list[DigitalMapper_layer__2026], 
                    learning_rate___s=0.01, epsilon = 0.01, device = None, dtype = None):
        super().__init__()
        #<  safety
        assert epsilon > 0., "Bad param"
        assert learning_rate___s> 0., "Bad param"
        for DigitalMapper_layer in DigitalMapper_layers:
            assert isinstance(DigitalMapper_layer, DigitalMapper_layer__2026), \
                        "this is different from the pytorch optim. It must be list[DigitalMapper_layer__2026]."
            pass

        #<  real payload
        self.digitalmapping_layers = torch.nn.ParameterList(DigitalMapper_layers)
        #learning_rate___s
        if isinstance(learning_rate___s, float):
            self.learning_rate___s = torch.nn.Parameter(torch.tensor(learning_rate___s, device=device, dtype=dtype), 
                    requires_grad = False)
            pass
        else:#torch.tensor
            learning_rate___s = learning_rate___s.to(dtype).to(device)
            self.learning_rate___s = torch.nn.Parameter(torch.tensor(learning_rate___s), 
                    requires_grad = False)
            pass
        assert self.learning_rate___s.requires_grad == False
        #epsilon
        if isinstance(epsilon, float):
            self.epsilon = torch.nn.Parameter(torch.tensor(epsilon, device=device, dtype=dtype), 
                    requires_grad = False)
            pass
        else:#torch.tensor
            epsilon = epsilon.to(dtype).to(device)
            self.epsilon = torch.nn.Parameter(torch.tensor(epsilon), 
                    requires_grad = False)
            pass
        assert self.epsilon.requires_grad == False
        return

    def forward(self):
        assert False, "This tool is designed as an optimizer, not a layer. Please search torch.optim.Optimizer for reference."
    def parameters(self, recurse = True):
        assert False, "This tool is designed as an optimizer, not a layer. Please search torch.optim.Optimizer for reference."
        return super().parameters(recurse) #unreachable!!!!!!!
    def zero_grad(self, set_to_none: bool = True) -> None:
        for digitalmapping_layer in self.digitalmapping_layers:
            assert isinstance(digitalmapping_layer, DigitalMapper_layer__2026)
            digitalmapping_layer._raw_weight___oCAP_iCAP.grad = None
            pass

    @torch.no_grad() # Important: disable gradient tracking within the optimizer step
    def step(self, safety_check = False)->None:#, closure=None):
        #https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-optimizers
        '''Bc I don't use this closure style, and I have no idea how it works.
        Just in case, let me turn it off. 
        Fyi, https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-optimizers'''

        for digitalmapping_layer in self.digitalmapping_layers:
            assert isinstance(digitalmapping_layer, DigitalMapper_layer__2026)

            if digitalmapping_layer._raw_weight___oCAP_iCAP.grad is None:
                continue # Skip parameters without gradients

            grad_like_for_raw_weight___o_i = digitalmapping_layer._get_useful_part_of_raw_weight_grad()
            if grad_like_for_raw_weight___o_i is None:
                continue
            
            #<  展开
            raw_weight___o_i:torch.Tensor = digitalmapping_layer.get_useful_part_of_raw_weight()

            #<  real payload
            _temp___max___o:torch.Tensor = grad_like_for_raw_weight___o_i.max(dim=1).values
            _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
            del _temp___max___o, _temp___max___o_EXPANDi
            if safety_check:
                assert inner___grad_like_for_raw_weight___o_i.le(0.).all()#################
                pass

            _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
            _temp___mean_of_abs___o = _temp___mean_of_abs___o.max(self.epsilon)
            #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
            if safety_check:
                assert _temp___mean_of_abs___o.ge(self.epsilon).all()
                pass

            _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
            del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi

            new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * self.learning_rate___s
            if safety_check:
                assert new___raw_weight___before_tanh___o_i.le(0.).all()
                pass
            new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
            #</  展开


            digitalmapping_layer.set_useful_part_of_raw_weight(new___raw_weight___o_i)
        return



