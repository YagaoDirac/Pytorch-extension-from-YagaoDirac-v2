


'''
Documentation!!!

The docs for DDNN, the Digital Deep Neural Network

[[[Range denotation]]]

Since it's impossible to do backward propagation with boolean values,
all the tools in this project use some analogy(real number, or floating point number)
way to represent.
This requires the defination of range.
In this project, 3 ranges are used.
1st is non clamped real number, or the plain floating point number,
any f32, f64, f16, bf16, f8 are all in this sort.
2nd is 01, any number in the 1st format but in the range of 0. to 1..
3rd is np, negetive 1 to positive 1.
But in code, the 1st is processed the same way as the 3rd. 
So, in code, only 01 and np are present.
But the difference between the code for 01 and np is very little. 
I only want the denotation to be in class names, so it's less error prone.

If you send wrong input to any layer, it's undefined behavior.
It's your responsibility to make sure about this.

The relationship between different range of representations:
(float means real number which is -inf to +inf, np means -1 to +1, 01 means 0 to +1)
Binarize layers:
from 01: 01(Binarize_01), np(Binarize_01_to_np)
from np:  01(Binarize_np_to_01), np(Binarize_np)
from real number, it's the same as "np as input"
to binarize into: 01(Binarize_np_to_01), np(Binarize_np)

Gate layers:
from 01: 01 (AND_01, OR_01,XOR_01)
from np: np (AND_np, OR_np,XOR_np)

The following 2 layers only accept input in a very strict range.
Before using them, scale down the range of input to the corresbonding range.
ADC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01  (with AND2_01, OR2_01, ???_01)
(Notice, no np version ADC)

DAC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01
(Notice, no np version DAC)

Analogy signal, real number, should be processed with binarize layer
(Binarize_np_to_01 or Binarize_np) or any ADC layer to become digital, 
then processed with other digital layers.
This is the same as the convention of electronics.



[[[The standard range]]]

In this project, the standard range is defined as: if a range is exact (0., 1.) as a 01 range,
or exact(-1., 1.) as a np range, if we don't care about the error of floating point number calculation,
it's a standard range.

unfinished docs
现在有2个办法解决这个问题了。门层里面只有一个可以得到标准输出范围。

Boundate.scale_back_to_... functions help you scale any tensor back to a standard range.
Why is this important, the reason is that, only binarize layers can handle non-standard range.
All the other layers, gates, adc, dac are sensitive to this detail.
In this project, I provide binarize layers without automatically scale the output back
to standard range, because in most cases, uses need to stack them in customized ways.
I provided an example of  unfinished docs....

Another naming convention. If a layer provides output in non standard range,
its name ends with some weird notification.



[[[More Binarized output and real output range]]]

Since the output and input of Binarize_01 are in the same range, and share the same meaning,
it's possible to stack it a few times to get a more binarized result.
This also applies for Binarize_np.
Then, it's critical to balance the layers count against the big_number param.
Also, the real output range of each layer is not the theorimatic output range of 
sigmoid or tanh, because the input is not -inf to +inf.
This leads to the Boundary_calc function. It calculates the real output range.
This function comes alone with 2 helper functions, 
Boundary_calc_helper_wrapper and Boundary_calc_helper_unwrapper.

To get better precision, scale_back_to_np    [unfinished docs]



[[[Slightly better way to use binarize layers]]]
Basically, the binarize layer do this:
1, offset the input if the input is in a 01 range, because the middle value is 0.5.
2, scale the input up using the factor of big_number.
3, calculate sigmoid or tanh.
4, feed the result into a gramo layer.
Sometimes we want a gramo layer before sigmoid or tanh function to protect the gradient.
But no matter how you stack binarize layers, you still need to manually add 
a gramo layer in the beginning, this is true for gates layers. Because gates layers 
use a very small big_number for the first binarize layer, which also shrink the gradient.



[unfinished docs]
[[[Gates layers]]]
First, the most important thing you really MUST remember to use gates layers properly.
All the gates layers in this project only accepts exactly (0., 1.) or (-1., 1.) range as input.
If you have some input in (0.1, 0.8), it's undefined behavior.
Use the scale_back_to_01 and scale_back_to_np function to scale the input while correct the range.
If the input range is not semetric around the mid value, the scaling function will change
the relationship against the mid value a bit. 
If the input range is (0.1, 0.8), then, logically, 0.45 is the mid value.
If you assume 0.5 as the mid value, ok I didn't prepare a tool for you. 
Since the real range of output in such cases is a bit undefined.
Should I use (0., 1.) as the output range, or (0., 0.875)?
Maybe I would do something in the future.

If the input is in the exact mid, the output is useless.
I simply choose a random way to handle this, it only guarantees no nan or inf is provided.
The XOR and NXOR use multiplication in training mode,
which makes them 2 more sensitive to "out of range input" than others,
if you use out of range range as input.


Second, to protect the gradient, 
you may need to modify the big_number of the first binarize layer in any gates layer.
The principle is that, bigger input dim leads smaller big_number.
If AND2 needs 6., AND3 needs something less than 6., maybe a 4. or 3..
AND, NAND, OR, NOR use similar formula, while XOR, NXOR use another formula,
and the formula affects the choice of big_number.
Example in the documentation string in any gates layer.


If the input of a gate is too many, then, the greatest gradient is a lot times
bigger than the smallest one. If the greatest-grad element needs n epochs to train,
and its gradient is 100x bigger than the smallest, then the least-grad element needs 
100n epochs to train. This may harm the trainability of the least-grad element by a lot.
Basically, the gates layers are designed to use as AND2, AND3, but never AND42 or AND69420.




unfinished docs



'''




from typing import Any, List, Tuple, Optional, Self
import sys
import math
import torch

#my customized.
from util import make_grad_noisy, __line__str, bitwise_acc, debug_Rank_1_parameter_to_List_float
from util import int_into_floats, data_gen_full_adder
from Binarize import Binarize
from Digital_mapper_v1_4 import DigitalMapper_V1_4
from GatesLayer import single_input_gate_np, AND_np, OR_np, XOR_np





class DigitalSignalProcessingUnit_layer(torch.nn.Module):
    r'''DSPU, Digital Signal Processing Unit.
    It's a mapper followed by a compound gate layer.
    
    unfinished docs
    It accepts standard binarized range as input, and 
    also outputs standard binarized range.
    
    This example shows how to handle pure digital signal.
    Only standard binary signals are allowed.
    All the gates are 2-input, both-output.
    
    Basic idea:
    >>> x = digital_mapper(input)# the in_mapper
    >>> AND_head = and_gate(input)
    >>> OR_head = or_gate(input)
    >>> XOR_head = xor_gate(input)
    >>> output_unmapped = concat(AND_head, OR_head, XOR_head, 0s, 1s)
    >>> output = digital_mapper(output_unmapped)# the out_mapper
    '''
    #__constants__ = ['',]
    
    # protect_param_every____training:int
    # training_count:int

    def __init__(self, in_features: int, \
        SIG_single_input_gates: int, and_gates: int, or_gates: int, xor_gates: int, \
            is_last_layer:bool, \
                    training_ghost_weight_probability = 0.65, \
                        scaling_ratio_scaled_by = 1., \
            check_dimentions:bool = True, \
                    #protect_param_every____training:int = 20, \
            #scaling_ratio_for_gramo_in_mapper:float = 200., \
                    device=None, dtype=None) -> None:       #, debug_info = ""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if check_dimentions:
            raise Exception("to do")
        
        
        self.in_mapper_in_features = in_features
        self.in_mapper_out_features = SIG_single_input_gates+(and_gates+or_gates+xor_gates)*2
        self.SIG_count = SIG_single_input_gates
        self.and_gate_count = and_gates
        self.or_gate_count = or_gates
        self.xor_gate_count = xor_gates
        self.update_gate_input_index()
        self.is_last_layer = is_last_layer
        
        # Creates the layers.
        self.in_mapper = DigitalMapper_V1_4(self.in_mapper_in_features, self.in_mapper_out_features, training_ghost_weight_probability = training_ghost_weight_probability)#,
                                       #debug_info = debug_info)
        self.in_mapper.scale_the_scaling_ratio_for_raw_weight(scaling_ratio_scaled_by)
        
        output_mode = 1#1_is_both
        self.single_input_gate = single_input_gate_np(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
        self.and_gate = AND_np(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
        self.or_gate = OR_np(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
        self.xor_gate = XOR_np(2, output_mode_0_is_self_only__1_is_both__2_is_opposite_only=output_mode)
       
        # mapper already has a gramo.
        #self.out_gramo = GradientModification()# I guess this should be default param.
       
        # For the param protection.
        #self.protect_param_every____training = protect_param_every____training 
        #self.training_count = 1 
        #end of function        
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    
    def get_in_features(self)->int:
        return self.in_mapper_in_features
    def get_out_features(self)->int:
        if self.is_last_layer:
            return self.SIG_count+self.in_mapper_out_features+2#with -1 and 1.
        else:
            return self.SIG_count+self.in_mapper_out_features
        #old version. return self.in_mapper_out_features+2
    def get_mapper_raw_weight(self)->torch.nn.Parameter:
        return self.in_mapper.raw_weight
    
    def get_mapper_scaling_ratio_for_inner_raw(self)->torch.nn.parameter.Parameter:
        '''simply gets from the inner layer.'''
        return self.in_mapper.get_scaling_ratio()
    def set_scaling_ratio_for_inner_raw(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        self.in_mapper.set_scaling_ratio(scaling_ratio)
    def reset_scaling_ratio_for_inner_raw(self):
        '''simply sets the inner layer.'''
        self.in_mapper.reset_scaling_ratio_for_raw_weight()
    def scale_the_scaling_ratio_for_inner_raw(self, by:float):
        '''simply sets the inner layer.'''
        self.in_mapper.scale_the_scaling_ratio_for_raw_weight(by)
    
    def get_gates_big_number(self)->Tuple[torch.nn.parameter.Parameter,
                            torch.nn.parameter.Parameter,torch.nn.parameter.Parameter]:
        big_number_of_and_gate = self.and_gate.Binarize_doesnot_need_gramo.big_number_list
        big_number_of_or_gate = self.or_gate.Binarize_doesnot_need_gramo.big_number_list
        big_number_of_xor_gate = self.xor_gate.Binarize_doesnot_need_gramo.big_number_list
        return (big_number_of_and_gate,big_number_of_or_gate,big_number_of_xor_gate)
    def set_gates_big_number(self, big_number_for_and_gate:List[float],
                    big_number_for_or_gate:List[float], big_number_for_xor_gate:List[float]):
        '''use any number <=0. to tell the code not to modify the corresponding one.'''
        self.and_gate.Binarize_doesnot_need_gramo.set_big_number_with_float(big_number_for_and_gate)
        self.or_gate.Binarize_doesnot_need_gramo.set_big_number_with_float(big_number_for_or_gate)
        self.xor_gate.Binarize_doesnot_need_gramo.set_big_number_with_float(big_number_for_xor_gate)
        #end of function.
             
    def set_auto_print_difference_between_epochs(self, set_to:bool = True):
        '''Simply sets the inner.'''
        self.in_mapper.set_auto_print_difference_between_epochs(set_to)        
        pass

    def get_zero_grad_ratio(self, directly_print_out:float = False)->float:
        return self.in_mapper.get_zero_grad_ratio(directly_print_out)
    
    
    def debug_strong_grad_ratio(self, log10_diff = 0., epi_for_w = 0.01, epi_for_g = 0.01,) -> float:
        return self.in_mapper.debug_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g)
        
    def after_step(self):
        self.in_mapper.after_step()
        pass
    
    def debug__old_func__can_convert_into_eval_only_mode(self)->Tuple[torch.Tensor, torch.Tensor]:
        return self.in_mapper.can__old_func__convert_into_eval_only_mode()
    
    def set_eval_mode(self, eval_mode_0_is_raw__1_is_sharp):
        self.in_mapper.eval_mode_0_is_raw__1_is_sharp = eval_mode_0_is_raw__1_is_sharp
        pass
    
    
    
    def update_gate_input_index(self):
        pos_to = 0
        
        pos_from = pos_to
        pos_to += self.SIG_count#no *2
        self.SIG_from = pos_from
        self.SIG_to = pos_to
            
        pos_from = pos_to
        pos_to += self.and_gate_count*2
        self.and_gate_from = pos_from
        self.and_gate_to = pos_to
        
        pos_from = pos_to
        pos_to += self.or_gate_count*2
        self.or_gate_from = pos_from
        self.or_gate_to = pos_to
        
        pos_from = pos_to
        pos_to += self.xor_gate_count*2
        self.xor_gate_from = pos_from
        self.xor_gate_to = pos_to
        pass
        #end of function.
    def print_gate_input_index(self):
        result_str = f'Single input gate:{self.SIG_from}>>>{self.SIG_to}, '
        result_str += f'and gate:{self.and_gate_from}>>>{self.and_gate_to}, '
        result_str += f'or gate:{self.or_gate_from}>>>{self.or_gate_to}, '
        result_str += f'xor gate:{self.xor_gate_from}>>>{self.xor_gate_to}.'
        print(result_str)        
        pass
    
    def print_param_overlap_ratio(self):
        raise Exception("untested. maybe unfinished.")
        '''Important: this function doesn't call the in_mapper version directly.
        The reason is that, the meaning of overlapping are different in 2 cases.
        Only the SIG part is the same as the in_mapper.'''
        with torch.no_grad():
            the_max_index = self.in_mapper.raw_weight.data.max(dim=1).indices
            the_dtype = torch.int32
            if self.SIG_count>1:
                total_overlap_count = 0
                total_possible_count = self.SIG_count*(self.SIG_count-1)//2
                for i in range(self.SIG_count-1):
                    host_index = torch.tensor([self.SIG_from+(i)], dtype=the_dtype)
                    guest_index = torch.linspace(self.SIG_from+(i+1)*2,
                                            self.SIG_from+(self.SIG_count-1)*2,
                                            self.SIG_count-i-1, dtype=the_dtype)
                    flag_overlapped = the_max_index[guest_index].eq(the_max_index[host_index])
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("SIG_overlap_ratio:", 
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.SIG_gate_count>0:
            
            if self.and_gate_count>1:
                total_overlap_count = 0
                total_possible_count = self.and_gate_count*(self.and_gate_count-1)//2
                for i in range(self.and_gate_count-1):
                    host_index = torch.tensor([self.and_gate_from+(i)*2], dtype=the_dtype)
                    guest_index = torch.linspace(self.and_gate_from+(i+1)*2,
                                            self.and_gate_from+(self.and_gate_count-1)*2,
                                            self.and_gate_count-i-1, dtype=the_dtype)
                    flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
                    flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
                    flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("AND_overlap_ratio:", 
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.and_gate_count>0:
                
            if self.or_gate_count>1:
                total_overlap_count = 0
                total_possible_count = self.or_gate_count*(self.or_gate_count-1)//2
                for i in range(self.or_gate_count-1):
                    host_index = torch.tensor([self.or_gate_from+(i)*2], dtype=the_dtype)
                    guest_index = torch.linspace(self.or_gate_from+(i+1)*2,
                                            self.or_gate_from+(self.or_gate_count-1)*2,
                                            self.or_gate_count-i-1, dtype=the_dtype)
                    flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
                    flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
                    flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("OR_overlap_ratio:", 
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.or_gate_count>0:
            
            if self.xor_gate_count>1:
                total_overlap_count = 0
                total_possible_count = self.xor_gate_count*(self.xor_gate_count-1)//2
                for i in range(self.xor_gate_count-1):
                    host_index = torch.tensor([self.xor_gate_from+(i)*2], dtype=the_dtype)
                    guest_index = torch.linspace(self.xor_gate_from+(i+1)*2,
                                            self.xor_gate_from+(self.xor_gate_count-1)*2,
                                            self.xor_gate_count-i-1, dtype=the_dtype)
                    flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
                    flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
                    flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
                    #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                    total_overlap_count += int(flag_overlapped.sum().item())
                    pass
                overlap_ratio = float(total_overlap_count)/total_possible_count
                print("XOR_overlap_ratio:", 
                        f'{overlap_ratio:.4f}',", ", total_overlap_count,
                        "/", total_possible_count)
                pass#if self.xor_gate_count>0:
            pass
        pass
    #end of function
        
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''This example only shows the training path.'''
        
        # some extra safety. It's safe to comment this part out.
        if len(self.and_gate.Binarize_doesnot_need_gramo.gramos)>0:
            raise Exception("self.and_gate has gramo !!!")
        if len(self.or_gate.Binarize_doesnot_need_gramo.gramos)>0:
            raise Exception("self.or_gate has gramo !!!")
        if len(self.xor_gate.Binarize_doesnot_need_gramo.gramos)>0:
            raise Exception("self.xor_gate has gramo !!!")
        
        #param protection
        # if self.training_count<self.protect_param_every____training:
        #     self.training_count +=1
        # else:
        #     self.training_count = 1
            # with torch.no_grad():
            #     if self.print_overlapping_param:
            #         #print("if self.print_overlapping_param   __line__   3174")
            #         #print(self.in_mapper.raw_weight.data) 
            #         #SIG = single input gate.
            #         the_max_index = self.in_mapper.raw_weight.data.max(dim=1).indices
            #         #print(the_max_index)
            #         #single input gate, or the NOT gate.
            #         the_dtype = torch.int32
                    
            #         if self.SIG_count>1:
            #             total_overlap_count = 0
            #             total_possible_count = self.SIG_count*(self.SIG_count-1)//2
            #             for i in range(self.SIG_count-1):
            #                 host_index = torch.tensor([self.SIG_from+(i)], dtype=the_dtype)
            #                 guest_index = torch.linspace(self.SIG_from+(i+1)*2,
            #                                         self.SIG_from+(self.SIG_count-1)*2,
            #                                         self.SIG_count-i-1, dtype=the_dtype)
            #                 flag_overlapped = the_max_index[guest_index].eq(the_max_index[host_index])
            #                 #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
            #                 total_overlap_count += int(flag_overlapped.sum().item())
            #                 pass
            #             overlap_ratio = float(total_overlap_count)/total_possible_count
            #             print("SIG_overlap_ratio:", 
            #                     f'{overlap_ratio:.4f}',", ", total_overlap_count,
            #                     "/", total_possible_count)
            #             pass#if self.SIG_gate_count>0:
                    
            #         if self.and_gate_count>1:
            #             total_overlap_count = 0
            #             total_possible_count = self.and_gate_count*(self.and_gate_count-1)//2
            #             for i in range(self.and_gate_count-1):
            #                 host_index = torch.tensor([self.and_gate_from+(i)*2], dtype=the_dtype)
            #                 guest_index = torch.linspace(self.and_gate_from+(i+1)*2,
            #                                         self.and_gate_from+(self.and_gate_count-1)*2,
            #                                         self.and_gate_count-i-1, dtype=the_dtype)
            #                 flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
            #                 flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
            #                 flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
            #                 #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
            #                 total_overlap_count += int(flag_overlapped.sum().item())
            #                 pass
            #             overlap_ratio = float(total_overlap_count)/total_possible_count
            #             print("AND_overlap_ratio:", 
            #                     f'{overlap_ratio:.4f}',", ", total_overlap_count,
            #                     "/", total_possible_count)
            #             pass#if self.and_gate_count>0:
                        
            #         if self.or_gate_count>1:
            #             total_overlap_count = 0
            #             total_possible_count = self.or_gate_count*(self.or_gate_count-1)//2
            #             for i in range(self.or_gate_count-1):
            #                 host_index = torch.tensor([self.or_gate_from+(i)*2], dtype=the_dtype)
            #                 guest_index = torch.linspace(self.or_gate_from+(i+1)*2,
            #                                         self.or_gate_from+(self.or_gate_count-1)*2,
            #                                         self.or_gate_count-i-1, dtype=the_dtype)
            #                 flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
            #                 flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
            #                 flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
            #                 #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
            #                 total_overlap_count += int(flag_overlapped.sum().item())
            #                 pass
            #             overlap_ratio = float(total_overlap_count)/total_possible_count
            #             print("OR_overlap_ratio:", 
            #                     f'{overlap_ratio:.4f}',", ", total_overlap_count,
            #                     "/", total_possible_count)
            #             pass#if self.or_gate_count>0:
                    
            #         if self.xor_gate_count>1:
            #             total_overlap_count = 0
            #             total_possible_count = self.xor_gate_count*(self.xor_gate_count-1)//2
            #             for i in range(self.xor_gate_count-1):
            #                 host_index = torch.tensor([self.xor_gate_from+(i)*2], dtype=the_dtype)
            #                 guest_index = torch.linspace(self.xor_gate_from+(i+1)*2,
            #                                         self.xor_gate_from+(self.xor_gate_count-1)*2,
            #                                         self.xor_gate_count-i-1, dtype=the_dtype)
            #                 flag_first_input_eq = the_max_index[guest_index].eq(the_max_index[host_index])
            #                 flag_second_input_eq = the_max_index[guest_index+1].eq(the_max_index[host_index+1])
            #                 flag_overlapped = flag_first_input_eq.logical_and(flag_second_input_eq)
            #                 #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
            #                 total_overlap_count += int(flag_overlapped.sum().item())
            #                 pass
            #             overlap_ratio = float(total_overlap_count)/total_possible_count
            #             print("XOR_overlap_ratio:", 
            #                     f'{overlap_ratio:.4f}',", ", total_overlap_count,
            #                     "/", total_possible_count)
            #             pass#if self.xor_gate_count>0:
                    
            #             pass
                    
            pass#end of param protection.
        
        x = input
        x = self.in_mapper(x)
        not_head:torch.Tensor = self.single_input_gate(
            x[:, self.SIG_from:self.SIG_to])
        #print(not_head, "not_head")
            
        and_head:torch.Tensor = self.and_gate(x[:, self.and_gate_from:self.and_gate_to])
        #print(and_head, "and_head")
        
        or_head = self.or_gate(x[:, self.or_gate_from:self.or_gate_to])  
        #print(or_head, "or_head")
        
        xor_head = self.xor_gate(x[:, self.xor_gate_from:self.xor_gate_to])
        #print(xor_head, "xor_head")
        
        if not self.is_last_layer:
            x = torch.concat([not_head, and_head, or_head, xor_head],dim=1)
            return x
        else:
            neg_ones = torch.empty([input.shape[0],1])
            neg_ones = neg_ones.fill_(-1.)
            neg_ones = neg_ones.to(and_head.device).to(and_head.dtype)
            ones = torch.ones([input.shape[0],1])
            ones = ones.to(and_head.device).to(and_head.dtype)
            x = torch.concat([not_head, and_head, or_head, xor_head, neg_ones, ones],dim=1)
            return x
    #end of function
    
    def extra_repr(self) -> str:
        result = f'In features:{self.in_mapper.in_features}, out features:{self.get_out_features()}, AND2:{self.and_gate_count}, OR2:{self.or_gate_count}, XOR2:{self.xor_gate_count}'
        return result
    
    def get_info(self, directly_print:bool=False) -> str:
        result = f'In features:{self.in_mapper.in_features}, out features:{self.get_out_features()}, AND2:{self.and_gate_count}, OR2:{self.or_gate_count}, XOR2:{self.xor_gate_count}'
        if directly_print:
            print(result)
        return result
    
    pass
fast_traval____end_of_DSPU_layer = 432
'''This is not a test. This is actually a reminder. The input and output sequence of gates 
layers are not very intuitive. If the gates are denoted as a,b,c, inputs are a1,a2,b1,b2, 
outputs are ao,-ao,bo,-bo, then the real sequence is:
input: a1,a2,b1,b2,c1,c2,d1,d2
output: a,b,c,d,-a,-b,-c,-d,-1,1
The last 2 are from neg_ones and ones. They are a bit similar to the bias in FCNN(wx+b).
'''
input = torch.tensor([[1., 2., 3.]], requires_grad=True)
layer = DigitalSignalProcessingUnit_layer(in_features=3,SIG_single_input_gates=2,and_gates=0,
            or_gates=0,xor_gates=0, is_last_layer=True, check_dimentions=False, 
            training_ghost_weight_probability=0.5)
print(layer.in_mapper.raw_weight.data.shape)
layer.in_mapper.raw_weight.data = torch.tensor([[ 1.,  0.,  0.],[ 0.,  1.,  0.],])
layer.after_step()
print(layer.in_mapper.raw_weight.data.shape)
layer.print_gate_input_index()
print(layer(input), "should be 1,2,-1,-2,-1,1, the last -1,1 are neg_ones and ones")

input = torch.tensor([[-1., 1.]], requires_grad=True)
layer = DigitalSignalProcessingUnit_layer(in_features=3,SIG_single_input_gates=2,and_gates=0,
            or_gates=0,xor_gates=0, is_last_layer=True, check_dimentions=False, 
            training_ghost_weight_probability=0.5)
layer.in_mapper.raw_weight.data = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],])
layer.after_step()
print(layer.in_mapper.raw_weight.data.shape)
layer.print_gate_input_index()
print(layer(input), "should be -1-1-1 1 1 1,-1,1, the last -1,1 are neg_ones and ones")

fds=432



# '''finished???? It can only tell how many overlap. 
# parameter overlapping and protection test.'''
# input = torch.tensor([[1., 2.]], requires_grad=True)
# layer = DigitalSignalProcessingUnit_layer(in_features=2,
#                 SIG_single_input_gates=0,and_gates=3,or_gates=0,xor_gates=0, 
#                 check_dimentions=False, protect_param_every____training=0)
# # 6 SIGs or 3 of any other gate.
# #print(layer.in_mapper.raw_weight.data.shape)
# layer.in_mapper.raw_weight.data = torch.tensor([
#     [ 0.,  1.],[ 0.,  1.],[ 0.,  1.],[ 0.,  1.],[ 0.,  1.],[ 0.,  1.],])
# layer.print_param_overlap_ratio()

# input = torch.tensor([[-1.,-1.,-1.,]])#this doesn't matter.
# layer = DigitalSignalProcessingUnit_layer(in_features=2,
#                         SIG_single_input_gates=5,and_gates=3,or_gates=4,xor_gates=5,
#                         check_dimentions=False, protect_param_every____training=0)
# #print(layer.in_mapper.raw_weight.data.shape)
# layer.in_mapper.raw_weight.data = torch.tensor([
#         [5,2,3],[1,5,3],[1,2,3],[1,2,3],[1,2,3],
#         [1,2,3],[1,2,3],[1,2,3],[1,6,3],[6,2,3],[1,2,3],
#         [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,6,3],[1,2,3],[1,6,3],[1,2,3],
#         [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,5,3],[1,2,3],[1,6,3],
#                                                 ],dtype=torch.float32)
# layer.print_param_overlap_ratio()

# fds=432






# '''This test shows the gramo in digital_mapper is the only gramo in main grad pass.
# '''
# input = torch.tensor([[1.]], requires_grad=True)
# layer = DigitalSignalProcessingUnit_layer(in_features=1,single_input_gates=1,and_gates=1,
#                                             or_gates=1,xor_gates=1)
# #print(len(layer.and_gate.Binarize.gramos), __line__str())
# output = layer(input)
# print(output, "should be 1., -1.,  1., -1.,  1., -1., -1.,  1.")
# if False:
#     g_in = torch.zeros_like(output)
#     g_in[0,0] = 1.
#     pass
# if True:
#     g_in = torch.ones_like(output)*111111
#     g_in[0,0] = 1.
#     pass
# torch.autograd.backward(output, g_in, inputs=input)
# print(input.grad, "input.grad", __line__str())
# fds=432

# '''basic test. eval mode.'''
# input = torch.tensor([[1.]])
# layer = DigitalSignalProcessingUnit_layer(1,1,1,1,1)
# layer.eval()
# output = layer(input)
# print(output, "should be 1., -1.,  1., -1.,  1., -1., -1.,  1.")
# fds=432

# layer = DigitalSignalProcessingUnit_layer(2,3,5,7,11)
# layer.get_info(True)
# print("out feature should be ", 3*2+5*2+7*2+11*2)
# fds=432



class DSPU(torch.nn.Module):
    r'''n DSPU layers in a row. Interface is the same as the single layer version.
    
    It contains an extra mapper layer in the end. 
    
    It's a ((mapper+gates)*n)+mapper structure.
    
    More info see DigitalSignalProcessingUnit.
    '''
    #__constants__ = ['',]
    def __init__(self, in_features: int, out_features:int, \
                single_input_gates: int, and_gates: int, or_gates: int, \
                    xor_gates: int, num_layers:int, \
                    scaling_ratio_for_gramo_in_mapper:float = 200., \
                    scaling_ratio_for_gramo_in_mapper_for_first_layer:Optional[float] = None, \
                    scaling_ratio_for_gramo_in_mapper_for_out_mapper:Optional[float] = None, \
                    protect_param_every____training:int = 20, \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        if scaling_ratio_for_gramo_in_mapper_for_first_layer is None:
            scaling_ratio_for_gramo_in_mapper_for_first_layer = scaling_ratio_for_gramo_in_mapper
            
        if scaling_ratio_for_gramo_in_mapper_for_out_mapper is None:
            scaling_ratio_for_gramo_in_mapper_for_out_mapper = scaling_ratio_for_gramo_in_mapper
        
        
        #debug_print_raw_param_in_training_pass
        self.first_layer = DigitalSignalProcessingUnit_layer(
            in_features, single_input_gates, and_gates, or_gates, xor_gates, 
            scaling_ratio_for_gramo_in_mapper=scaling_ratio_for_gramo_in_mapper_for_first_layer, 
            check_dimentions=False)#, debug_info = "first layer")
        
        mid_width = self.first_layer.get_out_features()
        self.mid_layers = torch.nn.modules.container.ModuleList( \
                [DigitalSignalProcessingUnit_layer(mid_width, single_input_gates, 
                                                   and_gates, or_gates, xor_gates, 
                        scaling_ratio_for_gramo_in_mapper=scaling_ratio_for_gramo_in_mapper,
                        check_dimentions=False)   
                    for i in range(num_layers-1)])
        self.last_layer = 
        self.num_layers = num_layers
           
        self.out_mapper = DigitalMapper_V1_4(mid_width+2, out_features,
                                       )# , debug_info="out mapper")
        

        self.protect_param_every____training = protect_param_every____training 
        self.training_count = 1
        self.set_protect_param_every____training(protect_param_every____training)
        #end of function        
           
    def set_protect_param_every____training(self, set_to= 20):
        self.protect_param_every____training = set_to 
        self.first_layer.protect_param_every____training = set_to
        for layer in self.second_to_last_layers:
            layer.protect_param_every____training = set_to
            pass
        pass 
        
    # def set_auto_print_overlapping_param(self, set_to:bool = True):
    #     self.first_layer.print_overlapping_param = set_to
    #     for layer in self.second_to_last_layers:
    #         layer.print_overlapping_param = set_to
    #         pass
    #     pass
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    def get_mapper_scaling_ratio_for_inner_raw_from_all(self)->List[torch.nn.parameter.Parameter]:
        result:List[torch.nn.parameter.Parameter] = []
        result.append(self.first_layer.get_mapper_scaling_ratio_for_inner_raw())
        layer: DigitalSignalProcessingUnit_layer
        for layer in self.second_to_last_layers:
            result.append(layer.get_mapper_scaling_ratio_for_inner_raw())
            pass
        result.append(self.out_mapper.get_scaling_ratio())
        return result
    def print_mapper_scaling_ratio_for_inner_raw_from_all(self):
        result = self.get_mapper_scaling_ratio_for_inner_raw_from_all()
        print("Scaling ratio of the inner raw weight:")
        print("First layer:", result[0].item())
        list_for_middle_layers:List[float] = []
        for i_ in range(len(self.second_to_last_layers)):
            the_scaling_ratio_here = result[i_+1]
            list_for_middle_layers.append(the_scaling_ratio_here.item())     
            pass
        if len(list_for_middle_layers)>0:
            print("Mid layers: ", list_for_middle_layers)
            pass
        print("Last layer:", result[-1].item())
        return 
    
    # 3 setters for the inner gramo
    def set_scaling_ratio_for_inner_raw_for_first_layer(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        self.first_layer.set_scaling_ratio_for_inner_raw(scaling_ratio)
        pass
    def set_scaling_ratio_for_inner_raw_for_middle_layers(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        for layer in self.second_to_last_layers:
            layer.set_scaling_ratio_for_inner_raw(scaling_ratio)
            pass
        pass
    def set_scaling_ratio_for_inner_raw_for_out_mapper(self, scaling_ratio:float):
        '''simply sets the inner layer.'''
        self.out_mapper.set_scaling_ratio_for_inner_raw(scaling_ratio)
        pass
    
    
    def get_gates_big_number_from_all(self)->List[Tuple[torch.nn.parameter.Parameter,
                        torch.nn.parameter.Parameter, torch.nn.parameter.Parameter]]:
        result:List[Tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter,
                         torch.nn.parameter.Parameter]] = []
        result.append(self.first_layer.get_gates_big_number())
        for layer in self.second_to_last_layers:
            result.append(layer.get_gates_big_number())
            pass
        return result
    def print_gates_big_number_from_all(self):
        L_T_P:List[Tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter,
                         torch.nn.parameter.Parameter]]
        L_T_P = self.get_gates_big_number_from_all()
        
        print("big_number_print:------------------")
        for i, layer_info in enumerate(L_T_P):
            str_to_print = f"layer_{i}: and gate:"
            str_to_print += str(debug_Rank_1_parameter_to_List_float(layer_info[0]))
            str_to_print = f", or gate:"
            str_to_print += str(debug_Rank_1_parameter_to_List_float(layer_info[1]))
            str_to_print = f", xor gate:"
            str_to_print += str(debug_Rank_1_parameter_to_List_float(layer_info[2]))
            pass
        print(str_to_print)
        return 
    
    def set_auto_print_difference_between_epochs_for_second_to_last_DSPU_layers(self, set_to:bool = True):
        for layer in self.second_to_last_layers:
            layer.set_auto_print_difference_between_epochs(set_to)
            pass    
        pass  
    
    def set_auto_print_difference_between_epochs(self, set_to:bool = True, for_whom:int = 1):
        '''The param:for_whom:
        >>> 0 all
        >>> 1 important(first, second, out_mapper)
        >>> 2 first
        >>> 3 second
        >>> 4 second to last(DSPU_layer)
        >>> 5 out_mapper
        '''
        DSPU_layer:DigitalSignalProcessingUnit_layer
        if 0 == for_whom:#all
            self.first_layer.set_auto_print_difference_between_epochs(set_to)
            self.set_auto_print_difference_between_epochs_for_second_to_last_DSPU_layers(set_to)
            self.out_mapper.set_auto_print_difference_between_epochs(set_to)
        if 1 == for_whom:#important(first, second, out_mapper)
            self.first_layer.set_auto_print_difference_between_epochs(set_to)
            if len(self.second_to_last_layers)>0:
                DSPU_layer = self.second_to_last_layers[0]
                DSPU_layer.set_auto_print_difference_between_epochs(set_to)
            self.out_mapper.set_auto_print_difference_between_epochs(set_to)
        if 2 == for_whom:#first
            self.first_layer.set_auto_print_difference_between_epochs(set_to)
        if 3 == for_whom:#second
            if len(self.second_to_last_layers)>0:
                DSPU_layer = self.second_to_last_layers[0]
                DSPU_layer.set_auto_print_difference_between_epochs(set_to)
        if 4 == for_whom:#second to last(DSPU_layer)
            self.set_auto_print_difference_between_epochs_for_second_to_last_DSPU_layers(set_to)
        if 5 == for_whom:#out_mapper
            self.out_mapper.set_auto_print_difference_between_epochs(set_to)
        pass
    
    def reset_auto_print_difference_between_epochs(self):
        self.set_auto_print_difference_between_epochs(False, 0)
        pass
    
    # def set_gates_big_number_for_all(self, big_number_of_and_gate:float,
    #                 big_number_of_or_gate:float, big_number_of_xor_gate:float):
    #     '''use any number <=0. not to modify the corresponding one.'''
    #     self.first_layer.set_gates_big_number(big_number_for_and_gate = big_number_of_and_gate,
    #                                         big_number_for_or_gate = big_number_of_or_gate, 
    #                                         big_number_for_xor_gate = big_number_of_xor_gate)
    #     for layer in self.second_to_last_layers:
    #         layer.set_gates_big_number(big_number_of_and_gate = big_number_of_and_gate,
    #                                 big_number_of_or_gate = big_number_of_or_gate, 
    #                                 big_number_of_xor_gate = big_number_of_xor_gate)
    #         pass
    #     pass
        #end of function
    
    
    
    def get_zero_grad_ratio(self, directly_print_out:float = False)->List[float]:
        result:List[float] = []
        result.append(self.first_layer.get_zero_grad_ratio(directly_print_out))
        for layer in self.second_to_last_layers:
            result.append(layer.get_zero_grad_ratio(directly_print_out))
            pass
        result.append(self.out_mapper.get_zero_grad_ratio(directly_print_out))
        return result
    
    def print_param_overlap_ratio(self):
        self.first_layer.print_param_overlap_ratio()
        for layer in self.second_to_last_layers:
            layer.print_param_overlap_ratio()
            pass
        pass
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''This example only shows the training path.'''
        x = input
        #fds = self.first_layer.in_mapper.inner_raw.raw_weight.dtype
        x = self.first_layer(x)
        #print(x, "xxxxxxxxxxxxxxxxxxxxxx")

        for layer in self.second_to_last_layers:
            x = layer(x)
            pass
        x = self.out_mapper(x)
        return x
    #end of function
    
    def get_info(self, directly_print:bool=False) -> str:
        
        result_str = f'{self.num_layers} DSPUs in a row. In features:'
        result_str += str(self.in_features)
        result_str += ", out features:"
        result_str += str(self.out_features)
        result_str += ", AND2:"
        result_str += str(self.first_layer.and_gate_count)
        result_str += ", OR2:"
        result_str += str(self.first_layer.or_gate_count)
        result_str += ", XOR2:"
        result_str += str(self.first_layer.xor_gate_count)
        if directly_print:
            print(result_str)
        return result_str
    
    pass
# layer = DSPU(11,7,2,3,4,5)
# layer.get_info(True)
# fds=432

def DSPU_adder_test(in_a:int, in_b:int, in_c:int, model:DSPU):
    device = next(model.parameters()).device
    string = f'{in_a}+{in_b}+{in_c} is evaluated as: '
    a = torch.tensor([[in_a]], device=device)
    b = torch.tensor([[in_b]], device=device)
    c = torch.tensor([[in_c]], device=device)
    target = a+b+c
    a = int_into_floats(a,1, False)    
    b = int_into_floats(b,1, False)    
    c = int_into_floats(c,1, False)    
    input = torch.concat([a,b,c], dim=1)
    input = input.requires_grad_()
    target = int_into_floats(target,1+1, False)    
    print(string, model(input), "should be", target)





# input = torch.ones([1,10], requires_grad=True)
# layer = single_input_gate_np(1)
# output = layer(input)
# g_in = torch.zeros_like(output)
# g_in[0,:5] = 1.
# torch.autograd.backward(output, g_in, inputs=input)
# print(g_in)
# print(input.grad)
# fds=432

# '''and, or, xor takes any grad from its output and duplicate it to both input'''
# input = torch.ones([1,10], requires_grad=True)
# layer = AND_np(2,1)
# output = layer(input)
# g_in = torch.zeros_like(output)
# g_in[0,:3] = 1.
# torch.autograd.backward(output, g_in, inputs=input)
# print(g_in)
# print(input.grad)
# fds=432

#DigitalMapper_eval_only_v2






# '''a basic test.
# This test actually shows how to print out the difference before/after step.

# The result of this test is a bit crazy. Half of the times, it finishes in 1 epoch.
# '''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# target = torch.tensor([[-1.], [1.], [1.]])
# model = DSPU(input.shape[1],target.shape[1],2,1,1,1,num_layers=1, scaling_ratio_for_gramo_in_mapper=200.)
# model.get_info(directly_print=True)
# model.set_auto_print_overlapping_param(True)
# model.set_protect_param_every____training(0)
# #model.set_auto_print_difference_between_epochs()

# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# iter_per_print = 1#111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#       print(pred.shape)
#       print(target.shape)
#       fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred, "pred")
#             print(target, "target")
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "make_grad_noisy":
#         if epoch%iter_per_print == iter_per_print-1:
#             make_grad_noisy(model, 2.)
#             pass
#         pass
    
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.first_layer.in_mapper.raw_weight.grad, "grad")
#             #print(model.second_to_last_layers[0])
#             print(model.out_mapper.raw_weight.grad, "grad")
#             pass
#         pass
#     if False and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             #layer = model.out_mapper
#             layer = model.first_layer.in_mapper
#             print(layer.raw_weight, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "first_layer.in_mapper   after update")
            
#             layer = model.out_mapper
#             print(layer.raw_weight, "out_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "out_mapper   after update")
            
#             pass    
#         pass    
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     if True and "print acc":
#         if epoch%iter_per_print == iter_per_print-1:
#             model.eval()
#             pred = model(input)
#             #print(pred, "pred", __line__str())
#             #print(target, "target")
#             acc = bitwise_acc(pred, target)
#             #print(epoch+1, "    ep/acc    ", acc)
#             if 1. == acc:
#                 print(epoch+1, "    ep/acc    ", acc)
#                 break
#             pass
#         pass

# fds=432



# '''does it fit in half adder.

# All the hyperparameters are somewhat tuned by me.'''
# batch = 100#10000
# (input, target) = data_gen_half_adder_1bit(batch, is_output_01=False,is_cuda=False)
# model = DSPU(input.shape[1],target.shape[1],2,1,1,1,num_layers=1, #1,2,3 are ok
#              scaling_ratio_for_gramo_in_mapper=1000.,
#              scaling_ratio_for_gramo_in_mapper_for_out_mapper=2000.)
# model.get_info(directly_print=True)
# #model.set_auto_print_difference_between_epochs()
# #model.set_auto_print_difference_between_epochs(True,5)
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# iter_per_print = 1#111
# print_count = 1555
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#         print(pred.shape, "pred.shape")
#         print(target.shape, "target.shape")
#         fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred, "pred")
#             print(target, "target")
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "make_grad_noisy":
#         make_grad_noisy(model, 2.)
#         pass
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.first_layer.in_mapper.raw_weight.grad, "grad")
#             #print(model.second_to_last_layers[0])
#             print(model.out_mapper.raw_weight.grad, "grad")
#             pass
#         pass
#     if False and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             #layer = model.out_mapper
#             layer = model.first_layer.in_mapper
#             print(layer.raw_weight, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "first_layer.in_mapper   after update")
            
#             layer = model.out_mapper
#             print(layer.raw_weight, "out_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "out_mapper   after update")
            
#             pass    
#         pass    
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     if True and "print acc":
#         if epoch%iter_per_print == iter_per_print-1:
#             model.eval()
#             pred = model(input)
#             #print(pred, "pred", __line__str())
#             #print(target, "target")
#             acc = bitwise_acc(pred, target)
#             print(epoch+1, "    ep/acc    ", acc)
#             if 1. == acc:
#                 print(epoch+1, "    ep/acc    ", acc)
#                 # print(pred[:5], "pred", __line__str())
#                 # print(target[:5], "target")
#                 break
#             pass
#         pass

# fds=432



# '''4 layers should be able to fit in AND3'''
# batch = 5#10000
# input = torch.randint(0,2,[batch,3])
# target = input.all(dim=1,keepdim=True)
# input = input*2-1
# input = input.to(torch.float32)
# target = target*2-1
# target = target.to(torch.float32)
# # print(input)
# # print(target)

# model = DSPU(input.shape[1],target.shape[1],8,4,4,4,num_layers=4, 
#              scaling_ratio_for_gramo_in_mapper=1000.,
#              scaling_ratio_for_gramo_in_mapper_for_out_mapper=2000.)
# model.get_info(directly_print=True)
# #model.set_auto_print_difference_between_epochs()
# #model.set_auto_print_difference_between_epochs(True,5)
# # model.set_auto_print_overlapping_param(True)
# # model.set_protect_param_every____training(0)
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# model.cuda()
# input = input.cuda()
# target = target.cuda()
# iter_per_print = 1#111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#         print(pred.shape, "pred.shape")
#         print(target.shape, "target.shape")
#         fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred, "pred")
#             print(target, "target")
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "make_grad_noisy":
#         make_grad_noisy(model, 1.5)
#         pass
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.first_layer.in_mapper.raw_weight.grad, "grad")
#             #print(model.second_to_last_layers[0])
#             print(model.out_mapper.raw_weight.grad, "grad")
#             pass
#         pass
#     if False and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             #layer = model.out_mapper
#             layer = model.first_layer.in_mapper
#             print(layer.raw_weight, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "first_layer.in_mapper   after update")
            
#             layer = model.out_mapper
#             print(layer.raw_weight, "out_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "out_mapper   after update")
            
#             pass    
#         pass    
#     if True and "print zero grad ratio":
#         result = model.get_zero_grad_ratio()
#         print("print zero grad ratio: ", result)
#         pass
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     if True and "print acc":
#         if epoch%iter_per_print == iter_per_print-1:
#             model.eval()
#             pred = model(input)
#             #print(pred, "pred", __line__str())
#             #print(target, "target")
#             acc = bitwise_acc(pred, target)
#             print(epoch+1, "    ep/acc    ", acc)
#             if 1. == acc:
#                 print(epoch+1, "    ep/acc    ", acc)
#                 # print(pred[:5], "pred", __line__str())
#                 # print(target[:5], "target")
#                 break
#             pass
#         pass

# fds=432




# 门层里面不应该gramo。应该在dspu层里面统一gramo，统一二值化。
# 宽度解决很多问题？或者说在补谁的问题？
# 反正mapper层还有一个做法。



#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。
#压制参数。

def test_config_dispatcher(bit:int, layer:int, gates:int, batch:int, lr:float, noise_base:float):
    return(bit, layer, gates, batch, lr, noise_base)

'''does it fit in FULL adder. Probably YES!!!'''

'''
old
f16, bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.5>>> 1k2,2k ,0k2,0k3,1k5,1k3
f16, bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.2>>> 2k8,0k5,0k3,0k4,0k3,1k5
f16, bit=2,layer=6,gates=8,batch=100,lr=0.001,noise_base=1.2>>> random.
f16, bit=2,layer=6,gates=8,batch=100,lr=0.0001,noise_base=1.2>>> random.
f16, bit=2,layer=6,gates=8,batch=100,lr=0.01,noise_base=1.2>>> random.
f16, bit=2,layer=6,gates=8,batch=100,lr=0.001,noise_base=2.>>> random.???


to do:
UNIT:::: K epochs.
f16, bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.5  not*3>>> 0k3,6k7,0k2,0k1,5k
f16, bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.2  not*3>>> 0k4,0k9,1k, 0k3,1k7
f16, bit=2,layer=6,gates=8,batch=100,lr=0.001,noise_base=1.2  not*3>>> random
f16, bit=2,layer=6,gates=16,batch=100,lr=0.001,noise_base=1.2  not*3>>> random??

'''

# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个
# 重复选线，要踢一个


(bit, layer, gates, batch, lr, noise_base) = test_config_dispatcher(
    bit=1,layer=3,gates=8,batch=50,lr=0.001,noise_base=1.2)
is_f16 = True
iter_per_print = 111#1111
print_count = 5555

(input, target) = data_gen_full_adder(bit,batch, is_output_01=False, is_cuda=True)
# print(input[:5])
# print(target[:5])

model = DSPU(input.shape[1],target.shape[1],gates*2,gates,gates,gates,layer, 
             scaling_ratio_for_gramo_in_mapper=1000.,
             scaling_ratio_for_gramo_in_mapper_for_first_layer=330.,
             scaling_ratio_for_gramo_in_mapper_for_out_mapper=1000.,)#1000,330,1000
model.get_info(directly_print=True)
#model.set_auto_print_difference_between_epochs(True,4)
#model.set_auto_print_difference_between_epochs(True,5)

#model.set_protect_param_every____training(0)

model.cuda()
if is_f16:
    input = input.to(torch.float16)
    target = target.to(torch.float16)
    model.half()
    pass

# model_DSPU.print_mapper_scaling_ratio_for_inner_raw_from_all()
# model_DSPU.print_gates_big_number_from_all()
# fds=432


# print(model_DSPU.get_mapper_scaling_ratio_for_inner_raw_from_all())
# print(model_DSPU.get_gates_big_number_from_all())
#model_DSPU.first_layer.set_scaling_ratio()


#model.set_auto_print_difference_between_epochs(True,5)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model.cuda()
input = input.cuda()
target = target.cuda()
#iter_per_print = 1#111
#print_count = 1555
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
    if True and "make_grad_noisy":
        make_grad_noisy(model, noise_base)
        pass
    if False and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            print(model.first_layer.in_mapper.raw_weight.grad, "first_layer   grad")
            print(model.second_to_last_layers[0].in_mapper.raw_weight.grad, "second_to_last_layers[0]   grad")
            print(model.out_mapper.raw_weight.grad, "out_mapper   grad")
            pass
        pass
    if False and "print the weight":
        if epoch%iter_per_print == iter_per_print-1:
            #layer = model.out_mapper
            layer = model.first_layer.in_mapper
            print(layer.raw_weight, "first_layer.in_mapper   before update")
            optimizer.step()
            print(layer.raw_weight, "first_layer.in_mapper   after update")
            
            layer = model.model.second_to_last_layers[0]
            print(layer.raw_weight, "second_to_last_layers[0]   before update")
            optimizer.step()
            print(layer.raw_weight, "second_to_last_layers[0]   after update")
            
            layer = model.out_mapper
            print(layer.raw_weight, "out_mapper   before update")
            optimizer.step()
            print(layer.raw_weight, "out_mapper   after update")
            
            pass    
        pass    
    if True and "print zero grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            result = model.get_zero_grad_ratio()
            print("print zero grad ratio: ", result)
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    optimizer.step()
    if True and "print param overlap":
        every = 100
        if epoch%every == every-1:
            model.print_param_overlap_ratio()
            pass
        pass
    if True and "print acc":
        if epoch%iter_per_print == iter_per_print-1:
            model.eval()
            pred = model(input)
            #print(pred, "pred", __line__str())
            #print(target, "target")
            acc = bitwise_acc(pred, target)
            print(epoch+1, "    ep/acc    ", acc)
            if 1. == acc:
                print(epoch+1, "    ep/acc    ", acc)
                print(pred[:5], "pred", __line__str())
                print(target[:5], "target")
                break
            pass
        pass

fds=432










































'''
to do list
g(raw_weight)/raw_weight确认一下。
output mode
big number?
'''
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
#不慌。。还没开始。。
# class ADC(torch.nn.Module):
#     r""" """
#     def __init__(self, first_big_number:float, \
#                  output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
#                 device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
        
#         if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
#             raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
#         self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
            
#         # The intermediate result will be binarized with following layers.
#         self.Binarize1 = Binarize_01__non_standard_output(1.)
#         self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
#         '''The only param is the big_number. 
#         Bigger big_number leads to more binarization and slightly bigger result range.
#         More layers leads to more binarization.
#         '''
#         # you may also needs some more Binarize layers.
#         #self.Binarize2 = Binarize_01(5.)
#         #self.Binarize3 = Binarize_01(5.)
#         pass

    
        
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         if self.training:
#             # If you know how pytorch works, you can comment this checking out.
#             if not input.requires_grad:
#                 raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
#             if len(input.shape)!=2:
#                 raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, all the inputs]")
            
#             x = input
#             # no offset is needed for OR
#             x = x-0.5
#             x = x.prod(dim=1, keepdim=True)
#             x = -x+0.5
            
#             x = self.Binarize1(x)
            
#             # you may also needs some more Binarize layers.
#             # x = self.Binarize2(x)
#             # x = self.Binarize3(x)

#             if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                 return x
#             if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                 opposite = 1.-x
#                 return torch.concat([x,opposite])
#             if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                 opposite = 1.-x
#                 return opposite
#             raise Exception("unreachable code.")

            
#         else:#eval mode
#             with torch.inference_mode():
#                 x = input.gt(0.5)
#                 x = x.to(torch.int8)
#                 #overflow doesn't affect the result. 
#                 x = x.sum(dim=1, keepdim=True)
#                 x = x%2
                
#                 if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                     x = x.to(input.dtype)
#                     return x
#                 if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                     opposite = x.logical_not()
#                     return torch.concat([x,opposite]).to(input.dtype)
#                 if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
#                     opposite = x.logical_not()
#                     return opposite.to(input.dtype)
#                 raise Exception("unreachable code.")
#         #end of function
        
#     pass






















'''
Progression:
Binarize layers: 6.
Basically done. Tests are dirty.

IM HERE!!!

Gate layers:
from float, you can go to: np, 01 (AND2_all_range, OR2_all_range, ???_all_range)
(But this layer doesn't stack, or doesn't work after get stacked.)
|||||||||||||||||||||||||||||||||||||
from 01: 01 (AND2_01, OR2_01, ???_01)
from np: np (AND2_np, OR2_np, ???_np)


可能需要一个计算最终二值化的极限值的辅助函数。


The following 2 layers only accept input in a very strict range.
Before using them, scale down the range of input to the corresbonding range.
ADC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01  (with AND2_01, OR2_01, ???_01)

DAC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01

'''








'''to do list:
unfinished docs
__all__
set big number 可能要允许设置小于1的数。。不然门层的可训练性可能会出问题。

数字层间的选线器

门层的第一个二值化层的big number可能要测试。
二值化层的big number和输出的关系。
'''









