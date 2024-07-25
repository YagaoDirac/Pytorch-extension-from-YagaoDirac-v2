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





class DigitalMapper_V1_1(torch.nn.Module):
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
                    scaling_ratio_for_learning_gramo:float = 1000., \
                    protect_param_every____training:int = 20, raw_weight_boundary_for_f32:float = 15., \
                    device=None, dtype=None) -> None:   #, debug_info = ""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if raw_weight_boundary_for_f32<5. :
            raise Exception("In my test, it goes to almost 6. in 4000 epochs. If you know what you are doing, comment this checking out.")

        self.in_features = in_features
        self.out_features = out_features

        self.raw_weight_boundary_for_f32 = torch.nn.Parameter(torch.tensor([raw_weight_boundary_for_f32]), requires_grad=False)
        self.raw_weight_boundary_for_f32.requires_grad_(False)
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.__reset_parameters()
        self.gramo_for_raw_weight = GradientModification(scaling_ratio=scaling_ratio_for_learning_gramo)
        self.set_auto_print_difference_between_epochs(auto_print_difference)
        
        self.raw_progress = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.progress_factor = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        
        self.Final_Binarize_doesnt_need_gramo = Binarize.create_analog_to_np(needs_gramo=False)

        #to keep track of the training.
        self.protect_param_every____training = protect_param_every____training 
        self.training_count = 0 
        
        pass

    def __reset_parameters(self) -> None:
        '''copied from torch.nn.Linear'''
        torch.nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        pass
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True    
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    
    def set_scaling_ratio_for_raw_weight(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.gramo_for_raw_weight.set_scaling_ratio(scaling_ratio)
        pass
    
    def print_raw_weight_after_softmax(self):
        print(self.raw_weight.mul(self.update_progress_factor()).softmax(dim=1))
        pass
    
    
    def set_auto_print_difference_between_epochs(self, set_to:bool = True):
        if not set_to:
            self.raw_weight_before = torch.nn.Parameter(torch.empty([0,]))
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
    
    
    def update_progress_factor(self):
        '''factor最低0.1， 最高9999？'''
        if self.raw_progress<0.6:
            draging_factor = 0.2
            draging_to = 0.1
        elif self.raw_progress<0.8:
            draging_factor = 0.05
            draging_to = 1.
            pass
        elif self.raw_progress<0.9:
            draging_factor = 0.05
            draging_to = 2.
            pass
        elif self.raw_progress<0.95:
            draging_factor = 0.05
            draging_to = 3.
            pass
        elif self.raw_progress<0.98:
            draging_factor = 0.02
            draging_to = 4.
            pass
        elif self.raw_progress<0.99:
            draging_factor = 0.02
            draging_to = 6.
            pass
        else:
            draging_factor = 0.01
            draging_to = 8.
            pass
        self.progress_factor.data = (1-draging_factor)*self.progress_factor+draging_factor*draging_to
        pass        
        #raise Exception("")
        
    def set_acc(self, acc:float):
        self.raw_progress.data = torch.tensor([acc], dtype=self.raw_progress.dtype, device=self.raw_progress.device)
        pass
    @staticmethod
    def bitwise_acc(a:torch.Tensor, b:torch.Tensor, print_out:bool = False)->float:
        temp = a.eq(b)
        temp = temp.sum().to(torch.float32)
        acc = temp/float(a.shape[0]*a.shape[1])
        acc_float = acc.item()
        if print_out:
            print("{:.4f}".format(acc_float), "<- the accuracy")
            pass
        return acc_float
        
            
    def get_index_format(self)->torch.Tensor:
        index_of_max_o = self.raw_weight.max(dim=1).indices
        return index_of_max_o
    def get_one_hot_format(self)->torch.Tensor:
        #raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
        out_features_s = self.raw_weight.shape[0]
        out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
        #print(out_features_iota, "out_features_iota")
        index_of_max_o = self.raw_weight.max(dim=1).indices
        #print(index_of_max_o, "index_of_max_o")
        
        one_hot_o_i = torch.zeros_like(self.raw_weight)
        one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
        return one_hot_o_i
    
    def get_zero_grad_ratio(self, directly_print_out:float = False)->float:
        result = 0.
        if not self.raw_weight.grad is None:
            flags = self.raw_weight.grad.eq(0.)
            total_amount = flags.sum().item()
            result = float(total_amount)/self.raw_weight.nelement()
        if directly_print_out:
            print("get_zero_grad_ratio:", result)
        return result
        
    def forward(self, input:torch.Tensor)->torch.Tensor:
        if len(input.shape)!=2:
            raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")
        
        # zero grad ratio!!!
        # use self.raw_weight_before.nelement() == 0 to test it.
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
            #pass
        
        # param protection!!!
        if self.training:
            if not self.training_count>=self.protect_param_every____training:
                self.training_count+=1
            else:
                self.training_count = 0
                with torch.no_grad():
                    #this automatic modification may mess sometime.
                    boundary = self.raw_weight_boundary_for_f32
                    if self.raw_weight.dtype == torch.float64:
                        boundary.data = boundary.data*2.
                        pass
                    if self.raw_weight.dtype == torch.float16:
                        boundary.data = boundary.data*0.5
                        pass
                    
                    flag = self.raw_weight.gt(boundary)
                    temp = flag*boundary
                    temp = temp.to(self.raw_weight.dtype)
                    temp = temp + self.raw_weight.data*(flag.logical_not())
                    self.raw_weight.data = temp

                    boundary.data = boundary.data * -1.
                    flag = self.raw_weight.lt(boundary)
                    temp = flag*boundary
                    temp = temp.to(self.raw_weight.dtype)
                    temp = temp + self.raw_weight.data*(flag.logical_not())
                    self.raw_weight.data = temp
                    
                    mean = self.raw_weight.mean(dim=1,keepdim=True)
                    self.raw_weight.data = self.raw_weight.data-mean
                    pass
                pass
            pass
            
        x = input
        x = x.unsqueeze(dim=2)
        
        w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
        self.update_progress_factor()
        w_after_mul = w_after_gramo.mul(self.progress_factor)
        w_after_softmax = w_after_mul.softmax(dim=1)
        x = w_after_softmax.matmul(x)
        
        x = x.squeeze(dim = 2)
        
        x = self.Final_Binarize_doesnt_need_gramo(x)
        return x

    def extra_repr(self) -> str:
        return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'
 
    def can_convert_into_eval_only_mode(self)->Tuple[torch.Tensor, torch.Tensor]:
        r'''output:
        >>> [0] can(True) or cannot(False)
        >>> [1] The least "max strength" of each output. Only for debug.
        '''
        after_softmax = self.inner_raw.raw_weight.softmax(dim=1)
        
        #print(after_softmax[:5])
        max_strength_for_each_output = after_softmax.amax(dim=1, keepdim=True)
        #print(max_strength_for_each_output)
        least_strength = max_strength_for_each_output.amin()
        #print(least_strength)
        result_flag = least_strength>0.7
        return (result_flag, least_strength)
        
        #the_least_of_after_softmax = after_softmax.max(dim=1)
        
        # old code
        # flag = after_softmax.gt(0.7)#0.7 or at least 0.5.
        # _check_flag = flag.any(dim=1)
        # _check_flag_all = _check_flag.all()
        
        #return _check_flag_all.item()
        
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
    
    pass


是不是应该手动加一个gramo？
'''basic 2 pass test.'''
input = torch.tensor([[-1.,1]], requires_grad=True)
layer = DigitalMapper_V1_1(2,3)
layer.raw_weight.data = torch.tensor([[1.,2],[1.,2],[1.,2],])
#print(layer.raw_weight.shape)
pred = layer(input)
#print(pred.shape)
print(pred)
g_in = torch.ones_like(pred)*10
torch.autograd.backward(pred, g_in, inputs=[input,layer.raw_weight])
print(input.grad)
print(layer.raw_weight.grad)
fds=432












            
        
        
        
        
        
        
        
        
        
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



# '''Test for all the modes, and eval only layer.'''
# print("All 4 prints below should be equal.")
# x = torch.tensor([[5., 6., 7.], [8., 9., 13]], requires_grad=True)
# layer = DigitalMapper_V2(3,5)
# print(layer(x))
# print(x[:,layer.get_index_format()])
# print(layer.get_one_hot_format().matmul(x[:,:,None]).squeeze(dim=-1))
# fds=432
# eval_only_layer = layer.get_eval_only()
# print(eval_only_layer(x))
# print("All 4 prints above should be equal.")

# fds=432


# '''basic test. Also, the eval mode.'''
# layer = DigitalMapper_V2(2,3)
# # print(layer.raw_weight.data.shape)
# layer.raw_weight.data=torch.Tensor([[2., -2.],[ 0.1, 0.],[ -2., 2.]])
# # print(layer.raw_weight.data)
# input = torch.tensor([[1., 0.]], requires_grad=True)
# print(layer(input), "should be 1 1 0")

# layer = DigitalMapper_V2(2,3)
# layer.eval()
# layer.raw_weight.data=torch.Tensor([[2., -2.],[ 0.1, 0.],[ -2., 2.]])
# input = torch.tensor([[1., 0.]], requires_grad=True)
# print(layer(input), "should be 1 1 0")
# fds=432



# '''some real training'''
# model = DigitalMapper_V2(2,1)
# model.set_scaling_ratio(200.)
# loss_function = torch.nn.MSELoss()
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# #input = torch.Tensor([[1., 0.],[0., 1.]])
# input = input.requires_grad_()
# target = torch.Tensor([[1.],[1.],[0.],[0.],])
# #target = torch.Tensor([[1.],[0.],])
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
# # for p in model.parameters():
# #     print(p)

# iter_per_print = 3#1111
# print_count = 5
# for epoch in range(iter_per_print*print_count):
    
#     model.train()
#     pred = model(input)
    
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(pred.shape, "pred")
#     #     print(target.shape, "target")
    
#     loss = loss_function(pred, target)
#     if True and epoch%iter_per_print == iter_per_print-1:
#         print(loss.item())
#         pass
#     optimizer.zero_grad()
#     loss.backward()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight.grad, "grad")
        
#     #model.raw_weight.grad = model.raw_weight.grad*-1.
#     #optimizer.param_groups[0]["lr"] = 0.01
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, model.raw_weight.grad, "before update")
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, "raw weight itself")
#     #     print(model.raw_weight.grad, "grad")
#     #     fds=432
#     if True and epoch%iter_per_print == iter_per_print-1:
#         print(model.raw_weight)
#         optimizer.step()
#         print(model.raw_weight)
#         pass
    
#     optimizer.step()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, "after update")

#     model.eval()
#     if epoch%iter_per_print == iter_per_print-1:
#         # print(loss, "loss")
#         # if True:
#         #     print(model.raw_weight.softmax(dim=1), "eval after softmax")
#         #     print(model.raw_weight, "eval before softmax")
#         #     print("--------------")
#         pass

# model.eval()
# print(model(input), "should be", target)

# fds=432
