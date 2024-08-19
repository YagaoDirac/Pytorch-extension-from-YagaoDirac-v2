
from typing import Any, List, Tuple, Optional, Self
import math
import torch

#my customized.
from Binarize import Binarize




class single_input_gate_np(torch.nn.Module):
    r""" 
    unfinished docs
    """
                 
                 
    def __init__(self, output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        pass
   
    # def accepts_non_standard_range(self)->bool:
    #     return False
    # def outputs_standard_range(self)->bool:
    #     return True
    # def outputs_non_standard_range(self)->bool:
    #     return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            
            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x
                else:
                    opposite = x.neg()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'Single input gate(SAME/NOT) layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass

# '''basic test'''
# input = torch.tensor([[-2., -1, 0, 1,2]], requires_grad=True)
# layer = single_input_gate_np(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# output = layer(input)
# print(output, "output", __line__str())
# g_in = torch.ones_like(output)
# g_in[0,1] = 0.
# torch.autograd.backward(output, g_in, inputs=input)
# print(input.grad, "input.grad")

# input = torch.tensor([[-2., -1, 0, 1,2]])
# layer = single_input_gate_np(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# layer.eval()
# print(layer(input), "layer(input)")
# fds=432



class AND_np(torch.nn.Module):
    r""" 
    unfinished docs
    
    AND gate. 
    Only accepts STANDARD 01 range.
    Output is customizable. But if it's not standard, is it useful ?
    
    Input shape: [batch, gate_count*input_count]
    
    Output shape: [batch, gate_count*input_count*(1 or 2)](depends on what you need)
    
    Usually, reshape the output into [batch, all the output], 
    concat it with output of other gates, and a 0 and an 1.
    
    
    unfinished docs.
    first big number和input数量的关系，必须写上。
    
    
    
    Takes any number of inputs. Range is (0., 1.).
    1. is True.
    
    Notice: training mode uses arithmetic ops and binarize layers,
    while eval mode uses simple comparison.
    
    example:
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],], requires_grad=True)
    >>> bound_of_a = Boundary_calc_helper_wrapper(0.1, 0.9)
    >>> a = scale_back_to_01(a, bound_of_a)
    >>> b = torch.tensor([[0.2],[0.2],[0.8],], requires_grad=True)
    >>> bound_of_b = Boundary_calc_helper_wrapper(0.2, 0.8)
    >>> b = scale_back_to_01(b, bound_of_b)
    >>> '''(optional) c = torch.tensor([[0.3],[0.3],[0.7],], requires_grad=True)'''
    >>> '''bound_of_c = Boundary_calc_helper_wrapper(0.3, 0.7)'''
    >>> '''c = scale_back_to_01(c, bound_of_c)'''
    >>> input = torch.concat((a,b), 1)# dim = 1 !!!
    >>> ''' or input = torch.concat((a,b,c), 1) '''
    >>> layer = AND_01()
    >>> result = layer(input)
    
    The evaluating mode is a bit different. It's a simply comparison.
    It relies on the threshold. If the input doesn't use 0.5 as threshold, offset it.
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],]) # requires_grad doesn't matter in this case.
    >>> b = torch.tensor([[0.2],[0.2],[0.8],])
    >>> c = torch.tensor([[0.3],[0.3],[0.7],])
    >>> input = torch.concat((a,b,c), 1) 
    >>> layer = AND_01()
    >>> layer.eval()
    >>> result = layer(input)
    
    These 2 tests show how to protect the gradient.
    Basically the big_number of the first binarize layer is the key.
    By making it smaller, you make the gradient of inputs closer over each elements.
    For AND gate, if it gets a lot False as input, the corresponding gradient is very small.

    >>> a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
    >>> b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
    >>> input = torch.concat((a,b,b), 1) 
    >>> layer = AND_01()
    >>> layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
    >>> result = layer(input)
    >>> print(result)
    >>> g_in = torch.ones_like(result)
    >>> torch.autograd.backward(result, g_in,inputs= input)
    >>> print(input.grad)
    >>> #more~
    >>> input = torch.concat((a,b,b,b,b,b,b,b,b,b,b,b,b), 1) 
    >>> layer = AND_01()
    >>> layer.Binarize1.set_big_number(0.3, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
    >>> result = layer(input)
    >>> print(result)
    >>> g_in = torch.ones_like(result)
    >>> torch.autograd.backward(result, g_in,inputs= input)
    >>> print(input.grad)
    
    example for calculate the output range:
    (remember to modify the function if you modified the forward function)
    
    >>> a = torch.tensor([[0.],[1.],], requires_grad=True)
    >>> input = torch.concat((a,a), 1) 
    >>> layer = AND_01()
    >>> result = layer(input)
    >>> print(result)
    >>> print(layer.get_output_range(2), "the output range. Should be the same as the result.")

    If the performance doesn't meet your requirement, 
    modify the binarize layers. More explaination in the source code.
    """
    
                 
                 
                 #first_big_number:float = 3., 
    def __init__(self, input_per_gate:int = 2, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        
        if input_per_gate<2:
            raise Exception("Param:input_per_gate should >=2.")
        self.input_per_gate = input_per_gate
        # self.input_per_gate = torch.nn.Parameter(torch.tensor([input_per_gate]), requires_grad=False)
        # self.input_per_gate.requires_grad_(False)
        
        # The intermediate result will be binarized with following layers.
        #self.Binarize = Binarize.create_01_to_01()
        self.Binarize_doesnot_need_gramo = Binarize.create_analog_to_np(False)
        #self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
       
        # No matter what happens, this layer should be designed to output standard binarized result.
        # Even if the sigmoid is avoidable, you still need a final binarize layer.
        # But, Binarize_01_to_01 does this for you, so you don't need an individual final_binarize layer.
        # self.Final_Binarize = Binarize_01_Forward_only()
        pass
   
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            x = x.sum(dim=2, keepdim=False)#dim=2
            #back to rank-2
            
            offset = torch.tensor([self.input_per_gate], dtype=x.dtype, device=x.device).neg()+1.
            x = x + offset
            
            # binarize 
            x = self.Binarize_doesnot_need_gramo(x)
            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.all(dim=2, keepdim=False)
                x = x.to(torch.int8)
                x = x*2-1
                x = x.to(input.dtype)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x
                else:
                    opposite = x.neg()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'AND/NAND layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass

# '''Does this gate layer protect the grad?'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# print(input)
# print(input.shape)
# layer = AND_np(input_per_gate=2)
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432


# '''Not important. The shape of eval path'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b,a,b,a,b,a,b), dim=1) 
# print(input.shape)
# layer = AND_01(input_per_gate=2)
# layer.eval()
# result = layer(input)
# print(result.shape)
# fds=432


# '''a basic test'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# layer = AND_01()
# layer.Binarize1.set_big_number(3.)#So you can adjust the range of grad.
# result = layer(input)
# print(result, "should be 0 0 1")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# fds=432


# '''This old test shows a trick. You can use slightly non standard input to help
# you track the behavior of the tested layer.'''
# '''These 2 tests show how to protect the gradient.
# Basically the big_number of the first binarize layer is the key.
# By making it smaller, you make the gradient of inputs closer over each elements.
# For AND gate, if it gets a lot false as input, the corresponding gradient is very small.
# '''
# a = torch.tensor([[0.001],[0.992],[0.993],], requires_grad=True)
# b = torch.tensor([[0.004],[0.005],[0.996],], requires_grad=True)
# input = torch.concat((a,b,b,b), dim=1) 
# layer = AND_01(input_per_gate=4, first_big_number=1.1)
# # layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
# result = layer(input)
# print(result, "should be 0., 0., 1.")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad, should be 3x of a's")
# # more~

# old test but tells something.
# '''If the input of a gate is too many, then, the greatest gradient is a lot times
# bigger than the smallest one. If the greatest-grad element needs n epochs to train,
# and its gradient is 100x bigger than the smallest, then the least-grad element needs 
# 100n epochs to train. This may harm the trainability of the least-grad element by a lot.
# Basically, the gates layers are designed to use as AND2, AND3, but never AND42 or AND69420.'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,b,b,b,b,b,b,b,b,b), dim=1) 
# layer = AND_01(input_per_gate=11, first_big_number=0.35)
# result = layer(input)
# print(result)
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad, should be 10x of a's")

# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = AND_01(input_per_gate=3)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 0., 1.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = AND_01()#first big number 5?
# print(layer(input), "should be 0., 0., 1.")
# layer.eval()
# print(layer(input), "should be 0., 0., 1.")
# layer = AND_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0., 0., 1., 1., 1., 0.")
# layer.eval()
# print(layer(input), "should be 0., 0., 1., 1., 1., 0.")
# layer = AND_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 1., 1., 0.")
# layer.eval()
# print(layer(input), "should be 1., 1., 0.")

# '''__str__'''
# layer = AND_01()
# print(layer)

# fds=432




class OR_np(torch.nn.Module):
    r""" unfinished docs"""
    def __init__(self, input_per_gate:int = 2, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        
        if input_per_gate<2:
            raise Exception("Param:input_per_gate should >=2.")
        self.input_per_gate = input_per_gate
        # self.input_per_gate = torch.nn.Parameter(torch.tensor([input_per_gate]), requires_grad=False)
        # self.input_per_gate.requires_grad_(False)
        
        # The intermediate result will be binarized with following layers.
        #self.Binarize = Binarize.create_01_to_01()
        self.Binarize_doesnot_need_gramo = Binarize.create_analog_to_np(False)
        #self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
       
        # No matter what happens, this layer should be designed to output standard binarized result.
        # Even if the sigmoid is avoidable, you still need a final binarize layer.
        # But, Binarize_01_to_01 does this for you, so you don't need an individual final_binarize layer.
        # self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            x = x.sum(dim=2, keepdim=False)#dim=2
            #back to rank-2
            
            #offset = float(self.input_per_gate)-1.
            offset = torch.tensor([self.input_per_gate], dtype=x.dtype, device=x.device)-1.
            x = x + offset
            
            # binarize 
            x = self.Binarize_doesnot_need_gramo(x)
            
            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.any(dim=2, keepdim=False)
                x = x.to(torch.int8)
                x = x*2-1
                x = x.to(input.dtype)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x
                else:
                    opposite = x.neg()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'OR/NOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass


# '''Does this gate layer protect the grad?'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# print(input)
# print(input.shape)
# layer = OR_np(input_per_gate=2)
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432


# '''Not important. The shape of eval path'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,a,b,a,b,a,b), dim=1) 
# print(input.shape)
# layer = OR_01(input_per_gate=2)
# layer.eval()
# result = layer(input)
# print(result.shape)
# fds=432

# '''a basic test'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# layer = OR_01()
# layer.Binarize1.set_big_number(3.)#So you can adjust the range of grad.
# result = layer(input)
# print(result, "should be 0 1 1")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# fds=432



# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = OR_01(input_per_gate=3)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 1., 1.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = OR_01()#first big number 5?
# print(layer(input), "should be 0., 1., 1.")
# layer.eval()
# print(layer(input), "should be 0., 1., 1.")
# layer = OR_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0., 1., 1., 1., 0., 0.")
# layer.eval()
# print(layer(input), "should be 0., 1., 1., 1., 0., 0.")
# layer = OR_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 1., 0., 0.")
# layer.eval()
# print(layer(input), "should be 1., 0., 0.")

# '''__str__'''
# layer = OR_01()
# print(layer)

# fds=432





class XOR_np(torch.nn.Module):
    r""" unfinished docs"""
    def __init__(self, input_per_gate:int = 2, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        
        if input_per_gate<2:
            raise Exception("Param:input_per_gate should >=2.")
        self.input_per_gate = input_per_gate
        # self.input_per_gate = torch.nn.Parameter(torch.tensor([input_per_gate]), requires_grad=False)
        # self.input_per_gate.requires_grad_(False)
        
        # The intermediate result will be binarized with following layers.
        #self.Binarize = Binarize.create_analog_to_01()
        #self.Binarize_doesnot_need_gramo = Binarize.create_analog_to_np(False)
        
        #self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
       
        # No matter what happens, this layer should be designed to output standard binarized result.
        # Even if the sigmoid is avoidable, you still need a final binarize layer.
        # But, Binarize_01_to_01 does this for you, so you don't need an individual final_binarize layer.
        # self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
     
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            #no offset for XOR_np
            x = x.prod(dim=2)
            #back to rank-2

            raise Exception("the following line is removed. untested.")
            #x = self.Binarize_doesnot_need_gramo(x)#hey, you don't need it.
            
            # Now x is actually the XNOR
            # The output part is different from AND and OR styles.
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([opposite, x], dim=1)
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
          
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.to(torch.int8)
                #overflow doesn't affect the result. 
                x = x.sum(dim=2, keepdim=False)
                x = x%2
                x = x*2-1
                x = x.to(input.dtype)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x.to(input.dtype)
                else:
                    opposite = x.neg()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1).to(input.dtype)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite.to(input.dtype)
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'OR/NOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass

# '''Does this gate layer protect the grad?'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# print(input)
# print(input.shape)
# layer = XOR_np(input_per_gate=2)
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432


# '''odd ones results in 1.'''
# input = torch.tensor([[1., 1.]], requires_grad=True)
# layer = XOR_01(input_per_gate=2)
# print(layer(input), "should be 0")
# input = torch.tensor([[1., 1., 1.]], requires_grad=True)
# layer = XOR_01(input_per_gate=3)
# print(layer(input), "should be 1")
# input = torch.tensor([[1., 1., 1., 1.]], requires_grad=True)
# layer = XOR_01(input_per_gate=4)
# print(layer(input), "should be 0")
# fds=432


# '''Not important. The shape of eval path'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,a,b,a,b,a,b), dim=1) 
# print(input.shape)
# layer = XOR_01(input_per_gate=2)
# layer.eval()
# result = layer(input)
# print(result.shape)
# fds=432

# '''a basic test'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# layer = XOR_01()
# layer.Binarize1.set_big_number(3.)#So you can adjust the range of grad.
# result = layer(input)
# print(result, "should be 0 1 0")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# fds=432


# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = XOR_01(input_per_gate=3)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 1., 1.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = XOR_01()#first big number 5?
# print(layer(input), "should be 0., 1., 0.")
# layer.eval()
# print(layer(input), "should be 0., 1., 0.")
# layer = XOR_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0., 1., 0., 1., 0., 1.")
# layer.eval()
# print(layer(input), "should be 0., 1., 0., 1., 0., 1.")
# layer = XOR_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 1., 0., 1.")
# layer.eval()
# print(layer(input), "should be 1., 0., 1.")

# '''__str__'''
# layer = XOR_01()
# print(layer)

# fds=432

