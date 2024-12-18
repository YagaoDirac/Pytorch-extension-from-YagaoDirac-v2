'''This is the 3rd trial for digital mapper layer.
This time, I only add some extra ghost weight to the top weight element.
The original weight is not modified by any other ways except for the real backward propagation.

This is also a soft style of digital mapper. The target is also a hard result.
The last version(2nd) basically failed because I added too much extra protections to it.
Too much extra tools make the entire code very hard to twist. 
The lesson I learnt from that version is, try not to touch the original
param from plain deep learning. If backward pass does the job, it's very likely to be
correct not to touch it.'''


from typing import Any, List, Tuple, Optional, Self
import torch
import math

import sys
# def __line__int():
#     return sys._getframe(1).f_lineno
def __line__str():
    return "    Line number: "+str(sys._getframe(1).f_lineno)
#print('This is line', __line__()) 


# __all__ = [
#     'make_grad_noisy',
#     'GradientModification',
#     'MirrorLayer',
#     'MirrorWithGramo',
#     'GradientModificationFunction', #Should I expose this?
#     'Linear_gramo', #Should I rename this one? Or somebody help me with the naming?
#     ]

def make_grad_noisy(model:torch.nn.Module, noise_base:float = 1.5):
    for p in model.parameters():
        if p.requires_grad and (not p.grad is None):
            temp = torch.randn_like(p.grad)
            noise_factor = torch.pow(noise_base, temp)
            with torch.no_grad():
                #p.grad = p.grad.detach().clone().mul(noise_factor)
                p.grad = p.grad.detach().mul(noise_factor)
                pass
            pass
        pass
    pass

# p = torch.nn.Parameter(torch.tensor([42.]))
# p.grad = torch.tensor([1.])
# p.grad = p.grad.detach().clone().mul(torch.tensor([1.23]))
# print(p.grad)
# fds=432


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

        x_needs_grad = torch.tensor([x.requires_grad])
        ctx.save_for_backward(scaling_ratio, epi, div_me_when_g_too_small, x_needs_grad)
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        scaling_ratio:torch.Tensor
        requires_grad:torch.Tensor
        scaling_ratio, epi, div_me_when_g_too_small, requires_grad, = ctx.saved_tensors
        if requires_grad.logical_not():
            return None

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
    
    def __init__(self, scaling_ratio:float = 1., \
                       epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio]), requires_grad=False)
        self.scaling_ratio.requires_grad_(False)
        self.epi=torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
        self.epi.requires_grad_(False)
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small]), requires_grad=False)
        self.div_me_when_g_too_small.requires_grad_(False)
        #raise Exception("untested")
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        # if not x.requires_grad:
        #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

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
    
    def extra_repr(self) -> str:
        return f'scaling_ratio={self.scaling_ratio.item():.4e}, epi={self.epi.item():.4e}, div_me_when_g_too_small={self.div_me_when_g_too_small.item():.4e}'





class Binarize_Forward_only_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        input:torch.Tensor = args[0]
        output_is_01:torch.Tensor = args[1]#this is a bool.
        
        x = input
        
        threshold = output_is_01*0.5
        
        dtype = x.dtype
        x = x.gt(threshold)
        x = x.to(dtype)
        
        the_True = torch.tensor([True], device=output_is_01.device)
        
        if output_is_01.ne(the_True):
            x = x*2. - 1.
            pass
            
        input_needs_grad = torch.tensor([input.requires_grad])
        ctx.save_for_backward(input_needs_grad)
        return x

    @staticmethod
    def backward(ctx, g):
        input_needs_grad:torch.Tensor
        input_needs_grad,  = ctx.saved_tensors
        if input_needs_grad:
            return g, None
        else:
            return None, None

    pass  # class



class Binarize_Forward_only(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It clamp the forward inter layer data, while doesn't touch the backward propagation.
    """
    def __init__(self, output_is_01:bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.output_is_01 = torch.nn.Parameter(torch.tensor([output_is_01]), requires_grad=False)
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return Binarize_Forward_only_Function.apply(x, self.output_is_01)




class Binarize(torch.nn.Module):
    r"""This layer is not designed to be used by users.
    """
    __constants__ = ['big_number_list',
                     'input_is_01', 'output_is_01',]
    #output_when_ambiguous: float
    big_number_list: torch.nn.parameter.Parameter
    input_is_01:bool
    output_is_01:bool
    gramos:torch.nn.modules.container.ModuleList

    def __init__(self, input_is_01:bool, output_is_01:bool, layer_count:int, \
                    needs_gramo:bool, \
                    big_number_list:torch.nn.Parameter = \
                        torch.nn.Parameter(torch.tensor([5.]), requires_grad = False), \
                    device=None, dtype=None, \
                    scaling_ratio_list:List[float] = [1.], epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #safety:
        # if output_when_ambiguous<-1. or output_when_ambiguous>1. :
        #     raise Exception("Param:output_when_ambiguous is out of range. If you mean to do this, use the set_output_when_ambiguous function after init this layer.")
        
        if layer_count<1:
            raise Exception("Param:layer_count must be at least 1.")
        if len(big_number_list.shape)!=1:
            raise Exception("Param:big_number_list must be in rank-1 shape. Or simply use the default value.")
        if big_number_list.shape[0]<1:
            raise Exception("Param:big_number needs at least 1 element. Or simply use the default value.")
        if len(scaling_ratio_list)!=1:
            raise Exception("Param:scaling_ratio_list must be in rank-1 shape. Or simply use the default value.")
        if scaling_ratio_list[0]<1:
            raise Exception("Param:scaling_ratio_list needs at least 1 element. Or simply use the default value.")
        # if big_number_list.shape[0]!=layer_count or len(scaling_ratio_list)!=layer_count:
        #     raise Exception("Param:scaling_ratio_list and big_number_list must have the same shape, and equal to the layer_count")
        
        self.input_is_01 = input_is_01
        self.output_is_01 = output_is_01
        self.layer_count = layer_count
        
        self.big_number_list = big_number_list
        self.big_number_list.requires_grad_(False)
        
        # self.output_when_ambiguous = 0.
        # if output_is_01:
        #     self.output_when_ambiguous = 0.5

        # this layer also needs the info for gramo.
        self.gramos = torch.nn.ModuleList()
        if needs_gramo:
            self.gramos = torch.nn.ModuleList(
                [GradientModification(scaling_ratio_list[i],epi,div_me_when_g_too_small)
                for i in range(big_number_list.shape[0])]
            )
        
        self.auto_print_x_before_Binarize = False
        
        self.Final_Binarize = Binarize_Forward_only(output_is_01)
        # the sequence between gramo and Binarize_Forward_only doesn't matter at all.
        # gramo only affect the backward, while Binarize_Forward_only only affect the forward.
        pass

    # @staticmethod
    # def create_analog_to_01(big_number_list:List[float] = [1.5], \
    #                         device=None, dtype=None, \
    #                 scaling_ratio_list:List[float] = [1.], epi=1e-5, \
    #                    div_me_when_g_too_small = 1e-3, ):
    #     result = Binarize(False, #  input is 01
    #                   True,  # output is 01
    #                   layer_count = 1,
    #                     scaling_ratio_list = scaling_ratio_list, epi = epi,
    #                     div_me_when_g_too_small = div_me_when_g_too_small)
    #     result.set_big_number_with_float(big_number_list)
    #     return result
    @staticmethod
    def create_analog_to_np(needs_gramo: bool, big_number_list:List[float] = [1.5], \
                            device=None, dtype=None, \
                    scaling_ratio_list:List[float] = [1.], epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ):
        result = Binarize(False, #  input is 01
                      False,  # output is 01
                      layer_count = 1,
                      needs_gramo = needs_gramo,
                        scaling_ratio_list = scaling_ratio_list, epi = epi,
                        div_me_when_g_too_small = div_me_when_g_too_small)
        result.set_big_number_with_float(big_number_list)
        return result
    # @staticmethod
    # def create_01_to_01(big_number_list:List[float] = [1.5], \
    #                         device=None, dtype=None, \
    #                 scaling_ratio_list:List[float] = [1.], epi=1e-5, \
    #                    div_me_when_g_too_small = 1e-3, ):
    #     result = Binarize(True, #  input is 01
    #                   True,  # output is 01
    #                   layer_count = 1,
    #                     scaling_ratio_list = scaling_ratio_list, epi = epi,
    #                     div_me_when_g_too_small = div_me_when_g_too_small)
    #     result.set_big_number_with_float(big_number_list)
    #     return result
    # @staticmethod
    # def create_01_to_np(big_number_list:List[float] = [1.5], \
    #                         device=None, dtype=None, \
    #                 scaling_ratio_list:List[float] = [1.], epi=1e-5, \
    #                    div_me_when_g_too_small = 1e-3, ):
    #     result = Binarize(True, #  input is 01
    #                   False,  # output is 01
    #                   layer_count = 1,
    #                     scaling_ratio_list = scaling_ratio_list, epi = epi,
    #                     div_me_when_g_too_small = div_me_when_g_too_small)
    #     result.set_big_number_with_float(big_number_list)
    #     return result
    
    
    # def set_output_when_ambiguous(self, output:float):
    #     if self.output_is_01:
    #         if output<0. or output>1.:
    #             raise Exception("The output can only be between 0 and 1.")
    #         pass
    #     else:
    #         if output<-1. or output>1.:
    #             raise Exception("The output can only be between -1 and 1.")
    #         pass
        
    #     self.output_when_ambiguous = output
    #     pass

    def set_big_number_with_float(self, big_number_list:List[float], \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''I_know_Im_setting_a_value_which_may_be_less_than_1 is 
            designed only for the first binarize layer in gates layers. 
            If inputs doesn't accumulate enough gradient, you should 
            consider using some 0.3 as big_number to protect the 
            trainability of the intput-end of the model.
        '''
        if len(big_number_list)<1:
            raise Exception("Param:big_number needs at least 1 element. Or simply use the default value.")
            pass
        if len(big_number_list)!=self.layer_count:
            raise Exception("I didn't test. Maybe you can modify self.layer_count first.")
            #raise Exception("Temparorily, this new list has to be the same shape as the old one. If you need to change the layer count, make a new Binarize layer.")
            pass
        
        for i, e in enumerate(big_number_list):
            if I_know_Im_setting_a_value_which_may_be_less_than_1:
                # This case is a bit special. I made this dedicated for the gates layers.
                if e<=0.:
                    raise Exception(f"Param:big_number the [{i}] element({e}) is not big enough.")
                    pass
                
            else:# The normal case
                if e<1.:
                    raise Exception(f"Param:big_number the [{i}] element({e}) is not big enough.Use I_know_Im_setting_a_value_which_may_be_less_than_1 = True if you know what you are doing.")
                    pass
        self.big_number_list:torch.nn.Parameter = torch.nn.Parameter(torch.tensor(big_number_list), requires_grad=False)
        self.big_number_list.requires_grad_(False)
        #self.update_output_range()
        pass
    
    def set_scaling_ratio(self, scaling_ratio_list:List[float])->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
       
        if len(scaling_ratio_list)!=self.layer_count:
            raise Exception("I didn't test. Maybe you can modify self.layer_count first.")
            #raise Exception("Temparorily, this new list has to be the same shape as the old one. If you need to change the layer count, make a new Binarize layer.")
        
        for i in range(len(scaling_ratio_list)):
            if scaling_ratio_list[i]<=0.:
                raise Exception("Must be > 0.")
            pass
        
        #some old variable.
        layer:GradientModification
        if len(self.gramos)>0:
            layer = self.gramos[0]
        else:
            layer = GradientModification()
        epi = layer.epi.item()
        div_me_when_g_too_small = layer.div_me_when_g_too_small.item()
        
        # clear()
        for i in range(len(self.gramos)):
            self.gramos.pop(-1)
        
        for scaling_ratio in scaling_ratio_list:
            self.gramos.append(GradientModification(scaling_ratio,
                                                    epi,div_me_when_g_too_small))
        pass

    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return False
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    def __get_offset_according_to_input_mode_for_training_pass(self)->float:
        if self.input_is_01:
            return -0.5
        else:
            return 0.
        
    def __get_offset_according_to_input_mode_for_eval_pass(self)->float:
        offset = 0.
        if self.input_is_01:
            offset = -0.5
            pass
        if self.output_is_01:
            offset += 0.5
            pass
        return offset
        
        #untested?
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

            if self.big_number_list.requires_grad:
                raise Exception("Unreachable path.")
            if len(self.big_number_list.shape)>1:
                raise Exception("The big number list must be rank-1.")
            big_number_len = self.big_number_list.shape[0]
            if len(self.gramos)>0:
                gramos_len = len(self.gramos)
                if big_number_len!=self.layer_count or gramos_len!=self.layer_count:
                    raise Exception("They must equal.")

            x = input
            
            # The offset. This only works for 01 as input.
            offset = self.__get_offset_according_to_input_mode_for_training_pass()
            if 0. != offset:
                x = x + offset
                pass
            the_gramo_layer: GradientModification
            for i in range(self.layer_count-1):
                if len(self.gramos)>0:
                    the_gramo_layer = self.gramos[i]
                    x = the_gramo_layer(x)
                    pass
                big_number = self.big_number_list[i]
                if big_number!=1.:
                    x = x*big_number
                    pass
                # All the intermediate calculation use the np style.
                x = torch.tanh(x)
                pass
            
         
            if len(self.gramos)>0:
                the_gramo_layer = self.gramos[-1]
                x = the_gramo_layer(x)
                pass
            big_number = self.big_number_list[-1]
            if big_number!=1.:
                x = x*big_number
                pass
            if self.output_is_01:
                x = torch.sigmoid(x)
            else:
                x = torch.tanh(x)
                pass
            
            if self.auto_print_x_before_Binarize:
                print(x)
            x = self.Final_Binarize(x)
            return x
        
        else:#eval mode
            '''I know the bit wise operations are much faster. 
            But for now I'll keep it f32 for simplisity.
            '''
            
            x = input
            
            # The offset. This only works for 01 as input.
            offset = self.__get_offset_according_to_input_mode_for_eval_pass()
            if 0. != offset:
                x = x + offset
                pass
            x = self.Final_Binarize(x)
            return x
        # end of function.
        
    
    def extra_repr(self) -> str:
        input_str = "[0. to 1.]"
        if not self.input_is_01:
            input_str = "[-1. to 1.] or (-inf. to +inf.)"
            pass
        output_str = "(0. to 1.)" if self.output_is_01 else "(-1. to 1.)"
        mode_str = "training" if self.training else "evaluating"
        
        result = f'Banarize layer, theorimatic input range:{input_str}, theorimatic output range:{output_str}, in {mode_str} mode'
        return result

    def debug_info(self, verbose:int = 0)->str:
        result = "Input is "
        input_str = "01" if self.input_is_01 else "np"
        result += input_str
        result += ", output is "
        output_str = "01" if self.output_is_01 else "np"
        result += output_str
        result += ". "
        if verbose>=1:
            result += "Big numbers are: "
            for i in range(len(self.big_number_list)):
                result += str(self.big_number_list[i].item())
                if i<len(self.big_number_list)-1:
                    result += ", "
                else:
                    result += ". "
                    pass
                pass
            pass
        l:GradientModification
        if verbose>=2:
            result += "Scaling ratios are: "
            for i, l in enumerate(self.gramos):
                result += "{:.6f}".format(l.scaling_ratio.item())
                if i<len(self.gramos)-1:
                    result += ", "
                else:
                    result += ". "
                    pass
                pass
            pass
        if verbose>=3:
            result += "Epi are: "
            for i, l in enumerate(self.gramos):
                result += "{:.6f}".format(l.epi.item())
                if i<len(self.gramos)-1:
                    result += ", "
                else:
                    result += ". "
                    pass
                pass
            result += "div_me_when_g_too_small are: "
            for i, l in enumerate(self.gramos):
                result += "{:.6f}".format(l.div_me_when_g_too_small.item())
                if i<len(self.gramos)-1:
                    result += ", "
                else:
                    result += ". "
                    pass
                pass
            pass
        return result
        
    pass






































# class DigitalMapper_eval_only_v2(torch.nn.Module):
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






笔记
here is the reason for this version NOT to work.
I forgot to design the sharpen process.
The 1.2 way doesn't work.
But now, if I change the 1.2 way a bit, it should work.

This is the 1.3 version. I tried to control the top cooked weight to 0.52
when it's bigger than 0.6. This protects the grad.
In 1.2, if the multi layer version doesn't finish in 500 epoch, 
it never finishes. Because, some layer always turns very sharp, 
and the grad doesn't flow back properly.
So I had this 1.3 version.

In 1.4 version, I'll combine both.

class DigitalMapper_V1_3(torch.nn.Module):
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
                    auto_print_difference:bool = False, \
                    scaling_ratio_for_learning_gramo:Optional[float] = None, \
                    protect_param_every____training:int = 20, \
                    #raw_weight_boundary_for_f32:float = 15., \
                        shaper_factor = 1.01035, \
                    device=None, dtype=None) -> None:   #, debug_info = ""
                        #shaper_factor = 1.0035, \
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.out_iota = torch.nn.Parameter(torch.linspace(0,out_features-1, out_features, dtype=torch.int32), requires_grad=False)

        # self.raw_weight_boundary_for_f32 = torch.nn.Parameter(torch.tensor([raw_weight_boundary_for_f32]), requires_grad=False)
        # self.raw_weight_boundary_for_f32.requires_grad_(False)
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.__reset_parameters()
        #self.backup_raw_weight = torch.nn.Parameter(self.raw_weight.detach().clone(), requires_grad=False)
        self.raw_weight_max = torch.nn.Parameter(torch.tensor([50.]), requires_grad=False)
        self.raw_weight_min = torch.nn.Parameter(torch.tensor([-150.]), requires_grad=False)
        
        # self.with_ghost__sharped_mode = False
        # self.with_ghost = False
        
        #self.ghost_weight_length = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        temp_the_log_of_in_features = torch.log(torch.tensor([in_features]))
        #temp_max_length = temp_the_log_of_in_features+50.# but 20 should also be enough.
        #self.ghost_weight_length_max = torch.nn.Parameter(temp_max_length, requires_grad=False)
        #self.ghost_weight_length_step = torch.nn.Parameter(temp_the_log_of_in_features*0.005, requires_grad=False)#or 0.02?
        #self.ghost_weight_length.data = self.ghost_weight_length.to(self.raw_weight.dtype)#?does this do anything?
        
        if scaling_ratio_for_learning_gramo is None:
            scaling_ratio_for_learning_gramo = temp_the_log_of_in_features.item()*100
            pass
        self.gramo_for_raw_weight = GradientModification(scaling_ratio=scaling_ratio_for_learning_gramo)
        
        self.set_auto_print_difference_between_epochs(auto_print_difference)
        
        self.out_binarize_does_NOT_need_gramo = Binarize.create_analog_to_np(needs_gramo=False)
        self.out_gramo = GradientModification()

        #to keep track of the training.
        self.protect_param_every____training = protect_param_every____training 
        self.protect_param__training_count = 0 

        #self.last_acc = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        
        #threshold to be able to convert into eval only mode.
        self.can_convert_to_eval_only_mode__the_threshold = torch.nn.Parameter(torch.tensor([0.51], **factory_kwargs), requires_grad=False)
        self.can_convert_to_eval_only_mode__the_threshold.requires_grad_(False)
        self.debug_least_strength_last_time = 0.
        pass

    def __reset_parameters(self) -> None:
        '''copied from torch.nn.Linear'''
        torch.nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        pass
        
    def accepts_non_standard_range(self)->bool:
        return False#emmm unfinished.
    def outputs_standard_range(self)->bool:
        return True    
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    
    def set_scaling_ratio_for_raw_weight(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.gramo_for_raw_weight.set_scaling_ratio(scaling_ratio)
        pass
    def set_can_convert_to_eval_only_mode__the_threshold(self, the_threshold:float):
        if the_threshold<=0.5:
            raise Exception("Param:the_threshold must > 0.5")
        if the_threshold>0.9:
            raise Exception("Trust me.")
        self.can_convert_to_eval_only_mode__the_threshold = torch.nn.Parameter(torch.tensor([the_threshold], requires_grad=False))
        self.can_convert_to_eval_only_mode__the_threshold.requires_grad_(False)
        pass
    
    # def set_shaper_factor(self, shaper_factor:float, I_know_what_Im_doing = False):
    #     if shaper_factor<=1.:
    #         raise Exception("Param:shaper_factor must > 1.")
    #     if not I_know_what_Im_doing and shaper_factor>1.2:
    #         raise Exception("I believe this is wrong. But if you know what you are doint, comment this checking out.")
            
    #     self.shaper_factor.data = torch.tensor([shaper_factor], requires_grad=False)
    #     self.shaper_factor.requires_grad_(False)
    #     pass
    
    def set_auto_print_difference_between_epochs(self, set_to:bool = True):
        with torch.no_grad():    
            if not set_to:
                self.raw_weight_before = torch.nn.Parameter(torch.empty([0,], requires_grad=False))
                self.raw_weight_before.requires_grad_(False)
                # use self.raw_weight_before.nelement() == 0 to test it.
                pass
            if set_to:
                if self.raw_weight is None:
                    raise Exception("This needs self.raw_weight first. Report this bug to the author, thanks.")
                if self.raw_weight.nelement() == 0:
                    raise Exception("Unreachable code. self.raw_weight contains 0 element. It's so wrong.")
                #if not hasattr(self, "raw_weight_before") or self.raw_weight_before is None:
                if self.raw_weight_before is None:
                    self.raw_weight_before = torch.nn.Parameter(torch.empty_like(self.raw_weight), requires_grad=False)
                    self.raw_weight_before.requires_grad_(False)
                    pass
                self.raw_weight_before.data = self.raw_weight.detach().clone()
                pass
            pass
    
    # old code
    # def try_increasing_ghost_weight_length(self, factor = 1.):
    #     if 0. == self.ghost_weight_length:
    #         the_log = torch.log(torch.tensor([self.in_features], dtype=torch.float32)).item()
    #         the_log10 = the_log/torch.log(torch.tensor([10.], dtype=torch.float32))
    #         log10_minus_1 = the_log10-1.5
    #         log10_minus_1 = log10_minus_1.to(self.ghost_weight_length_step.device)
    #         if log10_minus_1>self.ghost_weight_length_step:
    #             self.ghost_weight_length.data = log10_minus_1*factor
    #             self.ghost_weight_length.data = self.ghost_weight_length.to(self.raw_weight.dtype)
    #             return
    #     self.ghost_weight_length.data = self.ghost_weight_length + self.ghost_weight_length_step*factor
    #     if self.ghost_weight_length > self.ghost_weight_length_max:
    #         self.ghost_weight_length.data = self.ghost_weight_length_max
    #         pass
    #     self.ghost_weight_length.data = self.ghost_weight_length.to(self.raw_weight.dtype)
    #     pass
    # def try_decreasing_ghost_weight_length(self, factor = 1.):
    #     self.ghost_weight_length.data = self.ghost_weight_length - self.ghost_weight_length_step*factor
    #     if self.ghost_weight_length < 0.:
    #         self.ghost_weight_length.data = self.ghost_weight_length*0.
    #         pass
    #     self.ghost_weight_length.data = self.ghost_weight_length.to(self.raw_weight.dtype)
    #     pass
    
    @staticmethod
    def bitwise_acc(a:torch.Tensor, b:torch.Tensor, print_out:bool = False)->float:
        with torch.no_grad():    
            temp = a.eq(b)
            temp = temp.sum().to(torch.float32)
            acc = temp/float(a.shape[0]*a.shape[1])
            acc_float = acc.item()
            if print_out:
                print("{:.4f}".format(acc_float), "<- the accuracy")
                pass
            return acc_float
        
        
    def get_softmax_format(self, with_ghost_weight :Optional[bool] = None):
        raise Exception("unfinished.")
        r'''When the input param:with_ghost_weight is None, it uses self.with_ghost.'''
        if with_ghost_weight is None:
            with_ghost_weight = self.with_ghost
            pass
        
        # previous_with_ghost = self.with_ghost
        # previous_with_ghost_            
        #raise Exception("untested.")
        with torch.no_grad():
            if with_ghost_weight:
                w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
                the_max_index = self.raw_weight.max(dim=1,keepdim=False).indices
                ghost_weight = torch.zeros_like(self.raw_weight)
                ghost_weight[self.out_iota, the_max_index] = self.ghost_weight_length
                w_with_ghost = w_after_gramo + ghost_weight
                w_after_softmax = w_with_ghost.softmax(dim=1)
                return w_after_softmax   
            else:
                result = self.raw_weight.softmax(dim=1)
                return result   
        
        #print(self.raw_weight.mul(self.update_progress_factor()).softmax(dim=1))
        #end of function.
        
    # duplicated.
    # def get_index_format(self)->torch.Tensor:
    #     index_of_max_o = self.raw_weight.max(dim=1).indices
    #     return index_of_max_o
    def get_one_hot_format(self)->torch.Tensor:
        with torch.no_grad():
            #raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
            out_features_s = self.raw_weight.shape[0]
            out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
            #print(out_features_iota, "out_features_iota")
            index_of_max_o = self.raw_weight.max(dim=1).indices
            #print(index_of_max_o, "index_of_max_o")
            
            one_hot_o_i = torch.zeros_like(self.raw_weight)
            one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
            return one_hot_o_i
    
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

    # def set_with_ghost(self, set_to:bool):
    #     self.with_ghost = set_to
    #     if (not set_to) and self.with_ghost__sharped_mode:
    #         the_info = '''The self.with_ghost_test_sharped_mode priotizes over self.with_ghost
    #         to control the sharpen behavior. Now self.with_ghost_test_sharped_mode is True, 
    #         setting self.with_ghost to False doesn't change the behavior.'''
    #         print(the_info)
    #         pass
    #     pass
    # def set_with_ghost__sharp_mode(self, set_to:bool):
    #     '''Do NOT train the layer in sharp mode.'''
    #     self.with_ghost__sharped_mode = set_to
    #     pass

    def forward(self, input:torch.Tensor)->torch.Tensor:
        if len(input.shape)!=2:
            raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")
        
        # raw weight changes???
        #use self.raw_weight_before.nelement() == 0 to test it.
        if self.raw_weight_before.nelement() != 0:
            ne_flag = self.raw_weight_before.data.ne(self.raw_weight)
            if ne_flag.any()>0:
                to_report_from = self.raw_weight_before[ne_flag]
                to_report_from = to_report_from[:16]
                to_report_to = self.raw_weight[ne_flag]
                to_report_to = to_report_to[:16]
                line_number_info = "    Line number: "+str(sys._getframe(1).f_lineno)
                print("Raw weight changed, from:\n", to_report_from, ">>>to>>>\n", 
                        to_report_to, line_number_info)
            else:
                print("Raw weight was not changed in the last stepping")
                pass
            self.raw_weight_before.data = self.raw_weight.detach().clone()
            pass
        
            
        x = input
        x = x.unsqueeze(dim=2)
        
        # old code.
        #w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
        #final_raw_weight = self.get_final_raw_weight(self.training)

        w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
        w_after_softmax = w_after_gramo.softmax(dim=1)
        x = w_after_softmax.matmul(x)
        
        x = x.squeeze(dim = 2)
        
        x = self.out_binarize_does_NOT_need_gramo(x)
        x = self.out_gramo(x)#here is the only gramo.
        return x
    
    # def get_final_raw_weight(self, training:bool)->torch.Tensor:
    #     the_length = self.get_ghost_weight_length()
    #     if training:
    #         w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
    #         the_result = DigitalMapper_V1_2.apply_ghost_weight(w_after_gramo,
    #                     the_length,self.out_iota, self.get_max_index(), 
    #                     training = training)
    #         return the_result
    #     else:
    #         with torch.no_grad():
    #             the_result = DigitalMapper_V1_2.apply_ghost_weight(self.raw_weight,
    #                     the_length,self.out_iota, self.get_max_index(), 
    #                     training = training)
    #             return the_result
    #     #end of function.
    
    # def set_ghost_weight_length_with_float(self, set_to:float):
    #     self.ghost_weight_length.data = torch.tensor([set_to]).to(self.raw_weight.dtype)
    #     self.ghost_weight_length.requires_grad_(False)
    #     pass
    
    # def get_ghost_weight_length(self)->torch.nn.parameter.Parameter:
    #     if self.with_ghost__sharped_mode:
    #         return self.ghost_weight_length_max
    #     elif self.with_ghost:
    #         return self.ghost_weight_length
    #     else:
    #         return torch.nn.Parameter(torch.zeros([1,]), requires_grad=False)
    #     #end of function.
    
    def get_max_index(self)->torch.Tensor:
        with torch.no_grad():
            the_max_index = self.raw_weight.max(dim=1,keepdim=False).indices
            return the_max_index
    
    # @staticmethod
    # def apply_ghost_weight(the_tensor : torch.Tensor, \
    #         ghost_weight_length:torch.nn.parameter.Parameter, \
    #         out_iota: torch.tensor, the_max_index: torch.tensor, training = False)->torch.Tensor:
    #     if training:
    #         if ghost_weight_length != 0.:
    #             result = the_tensor.clone()
    #             result[out_iota, the_max_index] = result[out_iota, the_max_index]+ghost_weight_length
    #             pass
    #         else:
    #             result = the_tensor
    #             pass
    #         return result
    #     else:#eval mode.
    #         with torch.no_grad():
    #             if ghost_weight_length != 0.:
    #                 result = the_tensor.clone()
    #                 result[out_iota, the_max_index] = result[out_iota, the_max_index]+ghost_weight_length
    #                 pass
    #             else:
    #                 result = the_tensor
    #                 pass
    #             return result
    #     #end of function.
    
    # @staticmethod
    # def weight_after_softmax(input:torch.Tensor, training = False)->torch.Tensor:
    #     改
    #     if training:
    #         after_softmax = input.softmax(dim=1)
    #         return after_softmax
    #     else:
    #         with torch.no_grad():
    #             after_softmax = input.softmax(dim=1)
    #             return after_softmax
                
    
    # def before_test_sharped_mode(self):
    #     self.test_sharped_mode = True
    #     pass
    # def after_test_sharped_mode(self):
    #     self.test_sharped_mode = False
    #     pass
    
    def extra_repr(self) -> str:
        return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'
 
    def debug__get_strong_grad_ratio(self, log10_diff = 3., epi_for_w = 0.01, epi_for_g = 0.00001, \
                                print_out = False)->float:
        with torch.no_grad():
            #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
            if self.raw_weight.grad is None:
                if print_out:
                    print(0., "inside debug_micro_grad_ratio function __line 1082")
                    pass
                return 0.
            
            the_device=self.raw_weight.device
            epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
            flag_w_big_enough = self.raw_weight.gt(epi_for_w_tensor)
            
            epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
            flag_g_big_enough = self.raw_weight.grad.gt(epi_for_g_tensor)
            
            ten = torch.tensor([10.], device=the_device)
            log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
            corresponding_g = self.raw_weight.grad*torch.pow(ten, log10_diff_tensor)
            flag_w_lt_corresponding_g = self.raw_weight.lt(corresponding_g)
            
            flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
            result_tensor = flag_useful_g.sum().to(torch.float32)/self.raw_weight.nelement()
            result = result_tensor.item()
            if print_out:
                print(result, "inside debug_micro_grad_ratio function __line 1082")
                pass
            return result
 
    def can_convert_into_eval_only_mode(self, print_least_strength_now = False)->Tuple[torch.Tensor, torch.Tensor]:
        r'''output:
        >>> [0] can(True) or cannot(False)
        >>> [1] The least "max strength" of each output. Only for debug.
        '''
        with torch.no_grad():
            w_after_softmax = self.raw_weight.softmax(dim=1)
            max_of_each_output = w_after_softmax.amax(dim=1)
            least_strength = max_of_each_output.amin()
            least_strength = least_strength.to(self.raw_weight.device)
            #print(least_strength)
            result_flag = least_strength>self.can_convert_to_eval_only_mode__the_threshold
            result_flag = result_flag.to(self.raw_weight.device)
            
            if print_least_strength_now:
                least_strength_now = least_strength.item()
                print(least_strength_now,"least_strength_now  __line 1098")
                
                # old code.
                # if least_strength_now<0.52 and least_strength_now>0.2:
                #     if least_strength_now == self.debug_least_strength_last_time:
                #         print(least_strength_now,"least_strength_now")
                #         pass
                #     pass
                # self.debug_least_strength_last_time = least_strength_now
                pass
                
            return (result_flag, least_strength)
        
    # def convert_into_eval_only_mode(self)->DigitalMapper_eval_only:
    #     after_softmax = self.inner_raw.raw_weight.softmax(dim=1)
    #     flag = after_softmax.gt(0.7)#0.7 or at least 0.5.
    #     #print(flag, "flag")
    #     _check_flag = flag.any(dim=1)
    #     _check_flag_all = _check_flag.all()
    #     if not _check_flag_all.item():
    #         raise Exception("The mapper can NOT figure out a stable result. Train the model more.")
        
    #     argmax = flag.to(torch.int8).argmax(dim=1)
    #     #print(argmax, "argmax")
    #     result = DigitalMapper_eval_only(self.in_features, self.out_features, argmax)
    #     #print(result)
    #     return result
    #     #raise Exception("Not implemented yet. This feature only helps in deployment cases, but nobody cares about my project. It's never gonne be deployed.")
    
    # def before_step(self):
    #     self.backup_raw_weight.data = self.raw_weight.detach().clone()
    #     self.backup_raw_weight.data.requires_grad_(False)
    #     pass
    def after_step(self):
        '''the algorithm:
        the top raw weight element is a, the exp(a) is A
        the exp(...) or all other sums up to B
        so:
        A == k*(A+B), where k is the softmax result of the top element.
        (I'm not going to figure k out.)
        
        Then, I need a new A' to make the k back to 0.52, it's like:
        A' == 0.52*(A'+B)
        B is known,
        0.48A' == 0.52B
        A' == 1.08B
        then, log(A') is what I need.
        
        The code is like, i manually calc ed the softmax, so I can access the intermidiate result.
        The max index of the raw weight is also of the cooked weight.
        By cooking, I mean the softmax operation.
        '''
        with torch.no_grad():
            the_max_index = self.raw_weight.max(dim=1,keepdim=True).indices
            the_max_index = the_max_index.squeeze(dim=1)
            
            #flag_add_true_minus_false = the_max.values.lt(10.)
            
            exp_of_raw_weight = torch.exp(self.raw_weight)
            sum_of_exp = exp_of_raw_weight.sum(dim=1, keepdim=True)
            the_softmax = exp_of_raw_weight/sum_of_exp
            #ref_the_softmax = self.raw_weight.softmax(dim=1)
            
            top_cooked_weight = the_softmax[self.out_iota,the_max_index]
            top_cooked_weight = top_cooked_weight.unsqueeze(dim=1)
            flag_weight_too_hard:torch.Tensor = top_cooked_weight.gt(0.6)
            
            #exp_of_the_amax = torch.exp(the_max.values)
            
            top_of_exp = exp_of_raw_weight[self.out_iota,the_max_index]
            top_of_exp = top_of_exp.unsqueeze(dim=1)
            sum_of_exp_with_out_top_one = sum_of_exp - top_of_exp
            The_A_prime = sum_of_exp_with_out_top_one*1.08
            temp = flag_weight_too_hard*(The_A_prime)+(flag_weight_too_hard.logical_not())*top_of_exp
            temp = temp.squeeze(dim=1)
            temp = temp.log()
            self.raw_weight.data[self.out_iota,the_max_index] = temp
            
            #then step2. 
            
            # param protection!!!
            if self.protect_param__training_count<self.protect_param_every____training:
                self.protect_param__training_count+=1
            else:
                self.protect_param__training_count = 1
                self.protect_raw_weight()
                pass
            pass
        pass
    #end of function
        
    def protect_raw_weight(self):
        with torch.no_grad():
            #step 1, move to prevent too big element.
            #a = self.raw_weight.max(dim=1,keepdim=True).values
            the_device = self.raw_weight.device
            move_left = self.raw_weight.max(dim=1,keepdim=True).values-self.raw_weight_max
            move_left = move_left.maximum(torch.tensor([0.], device=the_device))
            move_left = move_left.to(self.raw_weight.device).to(self.raw_weight.dtype)
            self.raw_weight.data = self.raw_weight-move_left
            
            #step 2, cap to prevent too small element.
            flag_too_small = self.raw_weight.lt(self.raw_weight_min)
            temp = flag_too_small*self.raw_weight_min + flag_too_small.logical_not() *self.raw_weight.data
            temp = temp.to(self.raw_weight.device).to(self.raw_weight.dtype)
            self.raw_weight.data = temp
            
            return
    # end of function.
    
    pass


# '''can convert to eval only mode'''
# layer = DigitalMapper_V1_3(3,2)
# layer.raw_weight.data = torch.tensor([[0.5, 0.,0.,], [1., 0.,0.,],])
# print(layer.can_convert_into_eval_only_mode())
# layer.raw_weight.data = torch.tensor([[1.5, 0.,0.,], [1., 0.,0.,],])
# print(layer.can_convert_into_eval_only_mode())
# fds=432


# '''the core protection of v1_3. What would it be in v1_4?'''
# layer = DigitalMapper_V1_3(4,2, protect_param_every____training=0)
# optim = torch.optim.SGD(layer.parameters(), lr=1 )
# #layer.raw_weight.data = torch.tensor([[0.5, 0.,0.,], [1., 0.,0.,],])
# layer.raw_weight.data = torch.tensor([[-0.5186,  0.0922,  0.5863,  0.4118],
#         [ 0.1524,  0.0863,  0.3626,  0.5213]])
# optim.zero_grad()
# #layer.before_step()
# #layer.raw_weight.grad = torch.tensor([[-0.5, 0.,0.,],[-1., 0.,0.,],])#pseudo backward.
# optim.step()
# print(layer.raw_weight.data)
# layer.after_step()
# print(layer.raw_weight.data)
# fds=432


# '''micro grad test'''
# layer = DigitalMapper_V1_3(4,1, every?)
# layer.raw_weight.data = torch.tensor([[0.1,0.001,0.1     ,1.1,],])
# layer.raw_weight.grad = torch.tensor([[0.1,0.1,  0.000001,0.001,],])
# layer.debug__get_strong_grad_ratio(print_out=True)
# print("should be 0.25")
# fds=432


# # '''param protection'''
# layer = DigitalMapper_V1_3(3,1)
# layer.raw_weight.data = torch.tensor([[100.,0.,-300]])
# layer.protect_raw_weight()
# print(layer.raw_weight)
# fds=432


# '''basic 2 pass test.'''
# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_3(3,1)
# layer.raw_weight.data = torch.tensor([[0.6,0,0],])
# #print(layer.raw_weight.shape)
# pred = layer(input)
# #print(pred.shape)
# print(pred, "pred")
# g_in = torch.ones_like(pred)*10#0000 dosen't change any grad.
# torch.autograd.backward(pred, g_in, inputs=[input,layer.raw_weight])
# print(input.grad, "input.grad")
# print(layer.raw_weight.grad, "layer.raw_weight.grad")

# input = torch.tensor([[1.,-1,-1]], requires_grad=True)
# layer = DigitalMapper_V1_3(3,1)
# layer.raw_weight.data = torch.tensor([[0.8,0,0],])
# pred = layer(input)
# #print(pred.shape)
# print(pred, "pred")
# g_in = torch.ones_like(pred)*10000#0000 dosen't change any grad.
# torch.autograd.backward(pred, g_in, inputs=[input,layer.raw_weight])
# print(input.grad, "input.grad")
# print(layer.raw_weight.grad, "layer.raw_weight.grad")

# fds=432



# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# ????????????????
# '''can_convert_into_eval_only_mode function shows if the difference between 
# param is already big enough, and if it's safe to convert into pure binary mode.'''
# layer = DigitalMapper_V1_1(2,3)
# layer.raw_weight.data = torch.tensor([[0.,0.8],[0.,5.],[0.,5.],])
# print(layer.can_convert_into_eval_only_mode(), "can_convert_into_eval_only_mode()")
# print(torch.softmax(torch.tensor([[0.,0.8]]),dim=1)[0,1])
# layer = DigitalMapper_V1_1(2,3)
# layer.raw_weight.data = torch.tensor([[0.,0.85],[0.,5.],[0.,5.],])
# print(layer.can_convert_into_eval_only_mode(), "can_convert_into_eval_only_mode()")
# print(torch.softmax(torch.tensor([[0.,0.85]]),dim=1)[0,1])
# fds=432



def data_gen_for_digital_mapper_directly_test(batch:int, n_in:int, n_out:int, dtype = torch.float32)->Tuple[torch.Tensor, torch.Tensor]:
    input = torch.randint(0,2,[batch, n_in], dtype=torch.int8)
    input = input*2-1
    input = input.to(dtype)
    answer_index = torch.randint(0,n_in,[n_out])
    target = input[:, answer_index]
    return input, target

'''some real training'''
batch = 2
n_in = 4
n_out = 3
(input, target) = data_gen_for_digital_mapper_directly_test(batch,n_in,n_out)
input.requires_grad_()
#input = torch.Tensor([[1., 1.],[1., -1.],[-1., 1.],[-1., -1.],])
#target = torch.Tensor([[1.],[1.],[-1.],[-1.],])
# print(input)
# print(target)

model = DigitalMapper_V1_3(input.shape[1],target.shape[1],protect_param_every____training=5)
#model.set_scaling_ratio_for_raw_weight(500.)#old.
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
if False:
    for name, p in zip(model._parameters, model.parameters()):
        print(name, p)

iter_per_print = 1#1111
print_count = 333
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
    if False and "make_grad_noisy":
        make_grad_noisy(model, 1.5)
        pass
    if True and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            print(model.raw_weight.grad[:2,:7], "grad")
            pass
        pass
    if True and "print the weight":
        if epoch%iter_per_print == iter_per_print-1:
            layer = model
            print(layer.raw_weight[:2,:7], "first_layer.in_mapper   before update")
            optimizer.step()
            print(layer.raw_weight[:2,:7], "first_layer.in_mapper   after update")
            layer.after_step()############################
            print(layer.raw_weight[:2,:7], "first_layer.in_mapper   after THE PROTECTION")
            pass    
        pass    
    if False and "print zero grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            result = model.debug_get_zero_grad_ratio()
            print("print zero grad ratio: ", result)
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    optimizer.step()
    layer.after_step()############################
    if False and "print param overlap":
        every = 1
        if epoch%every == every-1:
            model.print_param_overlap_ratio()
            pass
        pass
    if epoch%iter_per_print == iter_per_print-1:
        with torch.inference_mode():
            model.eval()
            pred = model(input)
            #print(pred, "pred", __line__str())
            #print(target, "target")
            acc = DigitalMapper_V1_3.bitwise_acc(pred, target)
            
            if 1. != acc:
                print(epoch+1, "    ep/raw mode acc    ", acc)
            else:#FINISHED
                print(pred[:2,:7], "pred", __line__str())
                print(target[:2,:7], "target")
                print(model.can_convert_into_eval_only_mode(), "model.can_convert_into_eval_only_mode")
                print(epoch+1, "Training finished    __line  1256")
                break
        pass
    
    pass# the training loop.
fds=432



class dry_stack_test_for_digital_mapper_v1_2(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_width:int, num_layers:int, \
                    # auto_print_difference:bool = False, \
                    # scaling_ratio_for_learning_gramo:float = 100., \
                    # protect_param_every____training:int = 20, \
                    # raw_weight_boundary_for_f32:float = 15., \
                    #     shaper_factor = 1.0035, \
                    device=None, dtype=None) -> None:   #, debug_info = ""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if num_layers<2:
            raise Exception("............................")
        
        self.first_layer = DigitalMapper_V1_2(in_features,mid_width)
        self.mid_layers = torch.nn.ModuleList([
            DigitalMapper_V1_2(mid_width,mid_width)
            for _ in range(num_layers-2)
        ])
        self.last_layer = DigitalMapper_V1_2(mid_width,out_features)
        
        self.all_layers:List[DigitalMapper_V1_2] = []
        self.all_layers.append(self.first_layer)
        for layer in self.mid_layers:
            self.all_layers.append(layer)
            pass
        self.all_layers.append(self.last_layer)
        
        self.sharpen_ratio = 0.
        pass
    #end of function
    def forward(self, input:torch.Tensor)->torch.Tensor:
        x = input
        x = self.first_layer(x)
        for layer in self.mid_layers:
            x = layer(x)
            pass
        x = self.last_layer(x)
        return x
    @staticmethod
    def get_rand_scalar()->float:
        res = torch.rand([1,]).item()
        return res
    @staticmethod
    def get_rand_bool(p:float)->bool:
        r = torch.rand([1,]).item()
        return r<p
    
    def set_acc(self, acc:float):
        something = torch.tensor([0.],device=self.first_layer.raw_weight.device)
        if acc>0.5:
            temp = (acc-0.5)*2.
            something = torch.tensor([temp],device=self.first_layer.raw_weight.device)
            pass
        
        r = torch.rand([1,],device=self.first_layer.raw_weight.device)
        layer:DigitalMapper_V1_2
        
        drag_factor = torch.tensor([0.995],device=self.first_layer.raw_weight.device)
        if r<something:
            self.sharpen_ratio = drag_factor*self.sharpen_ratio+(1.-drag_factor)*something
            self.sharpen_ratio = self.sharpen_ratio+0.01
            if self.sharpen_ratio>0.4:#0.75:
                self.sharpen_ratio = torch.tensor([0.4],device=self.first_layer.raw_weight.device)
                pass
            pass
        else:
            self.sharpen_ratio = drag_factor*self.sharpen_ratio+(1.-drag_factor)*something
            self.sharpen_ratio = self.sharpen_ratio*0.95-0.01
            if self.sharpen_ratio<0.0:#0.
                self.sharpen_ratio= torch.tensor([0.0],device=self.first_layer.raw_weight.device)
                pass
            pass
        
        
        something2 = torch.tensor([0.],device=self.first_layer.raw_weight.device)
        if acc>0.5:
            temp = (acc-0.5)*2.
            temp = temp*temp
            something2 = torch.tensor([temp],device=self.first_layer.raw_weight.device)
            pass
        for layer in self.all_layers:
            rr = torch.rand([1,],device=self.first_layer.raw_weight.device)
            if rr<something2:
                layer.try_increasing_ghost_weight_length()
            else:
                layer.try_decreasing_ghost_weight_length(5.)
                pass
            pass
           
        self.re_rand_with_ghost()
        pass
    #end of function.

    def re_rand_with_ghost(self):
        for layer in self.all_layers:
            rand_bool = dry_stack_test_for_digital_mapper_v1_2.get_rand_bool(self.sharpen_ratio)
            layer.set_with_ghost(rand_bool)
            pass
        
        
        # self.first_layer.set_with_ghost(dry_stack_test_for_digital_mapper_v1_2.get_rand_bool(self.sharpen_ratio))
        # for layer in self.mid_layers:
        #     layer.set_with_ghost(dry_stack_test_for_digital_mapper_v1_2.get_rand_bool(self.sharpen_ratio))
        #     pass
        # self.last_layer.set_with_ghost(dry_stack_test_for_digital_mapper_v1_2.get_rand_bool(self.sharpen_ratio))
        
        # old code.
        # layer:DigitalMapper_V1_1
        # some_threshold = 0.52
        # layer = self.first_layer
        # if acc!=1. or layer.can_convert_into_eval_only_mode()[1]<some_threshold:
        # #if not layer.can_convert_into_eval_only_mode()[0]:
        #     layer.set_acc(acc)
        #     pass
        # for layer in self.mid_layers:
        #     if acc!=1. or layer.can_convert_into_eval_only_mode()[1]<some_threshold:
        #     #if not layer.can_convert_into_eval_only_mode()[0]:
        #         layer.set_acc(acc)
        #         pass
        #     pass
        # layer = self.last_layer
        # if acc!=1. or layer.can_convert_into_eval_only_mode()[1]<some_threshold:
        # #if not layer.can_convert_into_eval_only_mode()[0]:
        #     layer.set_acc(acc)
        #     pass
        # pass
    def set_scaling_ratio_for_raw_weight(self, scaling_ratio:float):
        for layer in self.all_layers:
            layer.set_scaling_ratio_for_raw_weight(scaling_ratio)
            pass
        # self.first_layer.set_scaling_ratio_for_raw_weight(scaling_ratio)
        # layer:DigitalMapper_V1_2
        # for layer in self.mid_layers:
        #     layer.set_scaling_ratio_for_raw_weight(scaling_ratio)
        #     pass
        # self.last_layer.set_scaling_ratio_for_raw_weight(scaling_ratio)
        pass
    
    def print_zero_grad_ratio(self):
        result = self.first_layer.debug_get_zero_grad_ratio()
        print("First layer: zero grad ratio: ", result)
        layer:DigitalMapper_V1_2
        for i, layer in enumerate(self.mid_layers):
            result = layer.debug_get_zero_grad_ratio()
            print(f"{i+2}th layer: zero grad ratio: ", result)
            pass
        result = self.last_layer.debug_get_zero_grad_ratio()
        print("Last layer: zero grad ratio: ", result)
        pass
    
    def print_strong_grad_ratio(self, log10_diff = 3., epi_for_w = 0.01, epi_for_g = 0.00001,):
        result = self.first_layer.debug_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g)
        print("First layer: zero grad ratio: ", result.item())
        layer:DigitalMapper_V1_2
        for i, layer in enumerate(self.mid_layers):
            result = layer.debug_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g)
            print(f"{i+2}th layer: zero grad ratio: ", result.item())
            pass
        result = self.last_layer.debug_strong_grad_ratio(log10_diff, epi_for_w, epi_for_g)
        print("Last layer: zero grad ratio: ", result.item())
        pass

    def can_convert_into_eval_only_mode(self, epoch:int, print_result = False)->Tuple[torch.Tensor, torch.Tensor]:
        temp_list:List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.all_layers:
            temp_list.append(layer.can_convert_into_eval_only_mode())
            pass
        
        # temp_list.append(self.first_layer.can_convert_into_eval_only_mode())
        # layer:DigitalMapper_V1_2
        # for layer in self.mid_layers:
        #     temp_list.append(layer.can_convert_into_eval_only_mode())
        #     pass
        # temp_list.append(self.last_layer.can_convert_into_eval_only_mode())
        
        if print_result:
            for obj in temp_list:
                print(f"{obj[1].item():.3f}", end=",,,")
                pass
            print("    ", epoch, "    from dry stack test can_convert_into_eval_only_mode function.")
            pass
        
        the_flag = torch.tensor([True], device=self.first_layer.raw_weight.device)
        the_number = torch.tensor([1.], device=self.first_layer.raw_weight.device)
        for temp in temp_list:
            the_flag = the_flag.logical_and(temp[0])
            the_number = the_number.minimum(temp[1])
            pass
        return (the_flag, the_number)
    
    # def before_step(self):
    #     self.first_layer.before_step()
    #     for layer in self.mid_layers:
    #         layer.before_step()
    #         pass
    #     self.last_layer.before_step()
    #     pass
    
    def before_test_sharped_mode(self):
        for layer in self.all_layers:
            layer.set_with_ghost__sharp_mode(True)
            pass
        
        # self.first_layer.set_with_ghost(True)
        # for layer in self.mid_layers:
        #     layer.set_with_ghost(True)
        #     pass
        # self.last_layer.set_with_ghost(True)
        pass
        
    def after_test_sharped_mode(self):
        for layer in self.all_layers:
            layer.set_with_ghost__sharp_mode(False)
            pass
        #self.re_rand_with_ghost()
        pass    
    
    def print_ghost_length(self):
        print("ghost_length:    ", end="")
        temp_list:List[float] = []
        for layer in self.all_layers:
            temp_list.append(layer.ghost_weight_length)
            pass
        
        # temp_list.append(self.first_layer.ghost_weight_length)
        # for layer in self.mid_layers:
        #     temp_list.append(layer.ghost_weight_length)
        #     pass
        # temp_list.append(self.last_layer.ghost_weight_length)
        for temp in temp_list:
            print(f"{temp.item():.3f}", end=", ")
            pass
        print("   __line 1566")
        pass
    
    pass




'''tests the dry stack.'''
batch = 100
n_in = 50
n_out = 11
mid_width = 142
num_layers = 4
iter_per_print = 100#1111
print_count = 333333
# batch = 10
# n_in = 8
# n_out = 4
# mid_width = 6
# num_layers = 2
(input, target) = data_gen_for_digital_mapper_directly_test(batch,n_in,n_out)
input.requires_grad_()
#input = torch.Tensor([[1., 1.],[1., -1.],[-1., 1.],[-1., -1.],])
#target = torch.Tensor([[1.],[1.],[-1.],[-1.],])
# print(input, "input")
# print(target, "target")

model = dry_stack_test_for_digital_mapper_v1_2(input.shape[1],target.shape[1],mid_width,num_layers=num_layers)
loss_function = torch.nn.MSELoss()
#model.set_scaling_ratio_for_raw_weight(2000.)
model.last_layer.set_scaling_ratio_for_raw_weight(model.last_layer.gramo_for_raw_weight.scaling_ratio*50.)
#print(model.first_layer.gramo_for_raw_weight.scaling_ratio)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
if True and "print parameters":
    if True:# and "only the training params":
        for name, p in zip(model._parameters, model.parameters()):
            if p.requires_grad:
                print(name, p)
                pass
            pass
        pass
    else:# prints all the params.
        for name, p in zip(model._parameters, model.parameters()):
            print(name, p)
            pass
        pass
    
model.cuda()
input = input.cuda()
target = target.cuda()
#print(model.first_layer.ghost_weight_length.dtype)
model.half()
#print(model.first_layer.ghost_weight_length.dtype)
input = input.to(torch.float16)
target = target.to(torch.float16)
#print(model.first_layer.raw_weight.requires_grad)
#fds=432

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
            print(pred[:5,:5], "pred")
            print(target[:5,:5], "target")
            pass
        pass
    loss = loss_function(pred, target)
    optimizer.zero_grad()
    loss.backward()
    #if epoch>19:
    #print(model.first_layer.raw_weight_boundary_for_f32.item())
    #print(model.first_layer.raw_weight_boundary_for_f32.requires_grad)
    if False and "make_grad_noisy":
        make_grad_noisy(model, 1.05)
        pass
    #print(model.first_layer.raw_weight_boundary_for_f32.item())
    if False and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            print(model.first_layer.raw_weight.grad[:3,:7], "grad")
            print(model.last_layer.raw_weight.grad[:3,:7], "grad")
            pass
        pass
    if False and "print the weight":
        if epoch%iter_per_print == iter_per_print-1:
            layer = model.first_layer
            print(layer.raw_weight[:3,:7], "first_layer.in_mapper   before update")
            optimizer.step()
            print(layer.raw_weight[:3,:7], "first_layer.in_mapper   after update")
            
            layer = model.last_layer
            print(layer.raw_weight[:3,:7], "first_layer.in_mapper   before update")
            optimizer.step()
            print(layer.raw_weight[:3,:7], "first_layer.in_mapper   after update")
            pass    
        pass    
    if True and "print strong grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            model.print_strong_grad_ratio()
            # 看这里
            # here is the reason this v1_2 failed.
            # It still push the softmax too hard, which prevents backward pass.
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    #print(model.first_layer.raw_weight.requires_grad,"    __line 1720")
#    model.before_step()
    optimizer.step()
    if False and "print param overlap":
        every = 1
        if epoch%every == every-1:
            model.print_param_overlap_ratio()
            pass
        pass
    if False and "likely useless....print max of each output":
        if epoch>500:
            if epoch%iter_per_print == iter_per_print-1:
                layer = model.first_layer
                the_max_values = layer.raw_weight.max(dim=1).values
                print(the_max_values, "first_layer max of each output")
                for i, layer in enumerate(model.mid_layers):
                    the_max_values = layer.raw_weight.max(dim=1).values
                    print(the_max_values, i+2, "_layer max of each output")
                    pass
                layer = model.last_layer
                the_max_values = layer.raw_weight.max(dim=1).values
                print(the_max_values, "last_layer max of each output")
                pass    
            pass    
        pass    
    
    if 2222==epoch:
        model.print_ghost_length()
        print(model.first_layer.raw_weight[:2,:6], "model.first_layer.raw_weight")
        print(model.first_layer.can_convert_into_eval_only_mode(), "model.first_layer")
        print(model.last_layer.raw_weight[:2,:6], "model.last_layer.raw_weight")
        print(model.last_layer.can_convert_into_eval_only_mode(), "model.last_layer")
        fds=432
        pass
    
    if True and "print with_ghost flags.":
        for i, layer in enumerate(model.all_layers):
            if layer.with_ghost or layer.with_ghost__sharped_mode:
                print(epoch, "ep/",i ," with_ghost:",layer.with_ghost,model.first_layer.with_ghost__sharped_mode, "   __line 1626")
                pass
            pass
        pass
    
    with torch.inference_mode():
        #every = 10
        #if epoch%every == every-1:
        model.eval()
        pred = model(input)
        #print(pred, "pred", __line__str())
        #print(target, "target")
        acc = DigitalMapper_V1_2.bitwise_acc(pred, target)
        model.set_acc(acc)
        
        if model.sharpen_ratio>0.1 and model.sharpen_ratio<0.7:
            print(epoch, "   epoch/    ", f"{model.sharpen_ratio.item():.2f}", "   model.sharpen_ratio    __line 1708")
            pass
        
        model.before_test_sharped_mode()
        model.eval()
        pred = model(input)
        sharp_mode_acc = DigitalMapper_V1_2.bitwise_acc(pred, target)

        
        
        if 1. == sharp_mode_acc:
            # finished = model.can_convert_into_eval_only_mode(epoch, print_result=True)
            # if finished[0].logical_not():
            #     if epoch%iter_per_print == iter_per_print-1:
            #         print(epoch+1, "    ep/acc    ", acc,"    is param hard enough:", f"{finished[1].item():.4f}", "    __line 1515")
            #         pass
            #     pass
            # else:
                #congrats. Training finished.
            print(pred[:2,:7], "pred", __line__str())
            print(target[:2,:7], "target")
            print(pred.ne(target).sum().item(), "   wrong in total.")
            print(epoch, "epoch.   Finished.")
            model.print_ghost_length()
            break
        model.after_test_sharped_mode()
            
        if acc!=1. or sharp_mode_acc !=1.:
            every = 1
            if epoch%every == every-1:
                print(epoch+1, "    ep/acc    ", f"{acc:.3f}", "    /   sharp mode acc",f"{sharp_mode_acc:.3f}")
                pass
            pass
        

fds=432


# 要做的事情：
# 1，加一个针对更新强度的debug功能。
# 2，保护到最大熟权重在0.52，不能大了，小了都还好。
# 3，保护生权重的范围。


