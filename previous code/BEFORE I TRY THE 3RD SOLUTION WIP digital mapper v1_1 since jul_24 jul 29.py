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
                    scaling_ratio_for_learning_gramo:float = 100., \
                    protect_param_every____training:int = 20, \
                    #raw_weight_boundary_for_f32:float = 15., \
                        shaper_factor = 1.01035, \
                    device=None, dtype=None) -> None:   #, debug_info = ""
                        #shaper_factor = 1.0035, \
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if in_features>160_000:
            raise_str = '''Now, the param are designed to cap the raw_weight at 12. 
            Although in the test, the second significant weight can eventually to to -9, 
            the softmax of the biggest weight can go e^21, which is very big.
            But the design doesn't guarantee it. I guess the biggest softmax will always
            be at least e^12, which is a bit bigger than 160_000. If the raw_weight 
            elements are too close to each other, the top softmax doesn't exceed 0.5,
            it's not possible to become pure digital, though it still works. 
            If you know what you are doing, use the test after this class to test your config.
            If you know what you are doing, but don't know how to use that test code, find me
            on twitter, discord, or github.'''
            raise Exception(raise_str)

        # if raw_weight_boundary_for_f32<5. :
        #     raise Exception("In my test, it goes to almost 6. in 4000 epochs. If you know what you are doing, comment this checking out.")

        self.critical_hit_threshold_min = 0.8
        self.critical_hit_threshold_original = 0.999
        self.critical_hit_threshold = self.critical_hit_threshold_original


        self.debug_least_strength_last_time = 0.

        self.in_features = in_features
        self.out_features = out_features

        # self.raw_weight_boundary_for_f32 = torch.nn.Parameter(torch.tensor([raw_weight_boundary_for_f32]), requires_grad=False)
        # self.raw_weight_boundary_for_f32.requires_grad_(False)
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.shaper_factor = torch.nn.Parameter(torch.tensor([shaper_factor], **factory_kwargs), requires_grad=False)
        self.shaper_factor.requires_grad_(False)
        
        self.__reset_parameters()
        self.gramo_for_raw_weight = GradientModification(scaling_ratio=scaling_ratio_for_learning_gramo)
        self.gramo_for_raw_weight_scaling_factor_original = torch.nn.Parameter(torch.tensor([scaling_ratio_for_learning_gramo]), requires_grad=False)
        self.gramo_for_raw_weight_scaling_factor_cap = torch.nn.Parameter(self.gramo_for_raw_weight_scaling_factor_original*10., requires_grad=False)
        
        self.set_auto_print_difference_between_epochs(auto_print_difference)
        
        # self.raw_progress = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        # self.progress_factor = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        
        self.Final_Binarize_doesnt_need_gramo = Binarize.create_analog_to_np(needs_gramo=False)
        self.out_gramo = GradientModification()

        #to keep track of the training.
        self.protect_param_every____training = protect_param_every____training 
        self.training_count = 0 
        
        self.soft_clamp_threshold_up = torch.nn.Parameter(torch.tensor([15.]), requires_grad=False)
        self.soft_clamp_threshold_low = torch.nn.Parameter(torch.tensor([-30.]), requires_grad=False)
        self.soft_clamp_margin = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        self.soft_clamp_epi = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=False)
        #temp1 = self.soft_clamp_threshold_up-self.soft_clamp_margin
        #temp_sharpen_help_max = 
        #self.sharpen_help_max = torch.nn.Parameter(torch.tensor([temp_sharpen_help_max]), requires_grad=False)

        
        self.last_acc = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.skip_hard_clamp = torch.nn.Parameter(torch.tensor([0], dtype = torch.int32),
                                                  requires_grad=False)
        self.skip_hard_clamp.requires_grad_(False)
        
        
        
        #threshold to be able to convert into eval only mode.
        self.the_threshold__can_convert_to_eval_only_mode = torch.nn.Parameter(torch.tensor([0.51], requires_grad=False, **factory_kwargs))
        self.the_threshold__can_convert_to_eval_only_mode.requires_grad_(False)
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
    def set_the_threshold(self, the_threshold:float):
        if the_threshold<=0.5:
            raise Exception("Param:the_threshold must > 0.5")
        if the_threshold>0.9:
            raise Exception("Trust me.")
        self.the_threshold__can_convert_to_eval_only_mode = torch.nn.Parameter(torch.tensor([the_threshold], requires_grad=False))
        self.the_threshold__can_convert_to_eval_only_mode.requires_grad_(False)
        pass
    
    def set_shaper_factor(self, shaper_factor:float, I_know_what_Im_doing = False):
        if shaper_factor<=1.:
            raise Exception("Param:shaper_factor must > 1.")
        if not I_know_what_Im_doing and shaper_factor>1.2:
            raise Exception("I believe this is wrong. But if you know what you are doint, comment this checking out.")
            
        self.shaper_factor.data = torch.tensor([shaper_factor], requires_grad=False)
        self.shaper_factor.requires_grad_(False)
        pass
    
    def set_auto_print_difference_between_epochs(self, set_to:bool = True):
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
    
    def set_acc(self, acc:float, protect_param = False, threshold = 0.52):
        '''When the acc is greater than 0.9, the raw_weight scales a bit. 
        This leads the result of softmax goes a bit sharper.'''
        self.last_acc = torch.nn.Parameter(torch.tensor([acc]), requires_grad=False)
        
        with torch.no_grad():
            if 1. == acc:
                if self.can_convert_into_eval_only_mode(False)[1]<threshold:
                #if acc>=0.995:
                    self.raw_weight.data = self.raw_weight.data *self.shaper_factor
                    pass
                pass
        
            # if 1. == acc:
            #     temp = self.gramo_for_raw_weight_scaling_factor_mul_factor
            #     self.gramo_for_raw_weight.set_scaling_ratio()
            
        
            # when the biggest number repeats, this helps make them different.
            #?????????????no strong reason, but very unlikely to help at all.
            if 1. == acc:
                #print(self.raw_weight.data.shape)
                #print(self.raw_weight[:2])
                the_max_index = self.raw_weight.data.max(dim = 1).indices
                #print(the_max_index)
                self.raw_weight.data[:,the_max_index]=self.raw_weight.data[:,the_max_index]+0.001
                #0.0001better???, 
                #0.001ok, 
                #0.01,2,4ok, 
                #0.1unlikely ok
                #print(self.raw_weight[:2])
                #fds= 432
                self.skip_hard_clamp.data = torch.tensor([2])
                self.skip_hard_clamp.requires_grad_(False)
                pass
                
        if protect_param:
            self.protect_raw_weight()
        # old code.        
        #self.raw_progress.data = torch.tensor([acc], dtype=self.raw_progress.dtype, device=self.raw_progress.device)
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
        
        
        
    def get_softmax_format(self):
        #print(self.raw_weight.mul(self.update_progress_factor()).softmax(dim=1))
        result = self.raw_weight.softmax(dim=1)
        return result
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
        
    def protect_raw_weight(self):
        with torch.no_grad():
            # the hard clamp part.
            #print(self.raw_weight_boundary_for_f32.item())
            
            # if self.raw_weight_boundary_for_f32<=0.:
            #     raise Exception(" programming is hard......")
            
            # hard clamp~
            # if 1. != self.last_acc:
            #     if self.skip_hard_clamp>0:
            #         self.skip_hard_clamp.data = self.skip_hard_clamp.data-1
            #         self.skip_hard_clamp.requires_grad_(False)
            #         pass
            #     else:
                    
            boundary = torch.tensor([-50.], device=self.raw_weight.device)
            if self.raw_weight.dtype == torch.float64:
                boundary.data = boundary.data*2.
                pass
            # if self.raw_weight.dtype == torch.float16:
            #     boundary.data = boundary.data*0.5
            #     pass
                    
            # flag = self.raw_weight.gt(boundary)
            # # print(flag.sum(), "/", flag.nelement(), "     __line 853")
            # # if flag.sum()>0:
            # #     print(self.raw_weight[flag], "hard clamp positive.  __line 853")
            # #     pass
            # temp = flag*boundary
            # temp = temp.to(self.raw_weight.dtype)
            # temp = temp + self.raw_weight.data*(flag.logical_not())
            # self.raw_weight.data = temp

            #boundary.data = boundary.data * -1.
            flag = self.raw_weight.lt(boundary)
            # print(flag.sum(), "/", flag.nelement(), "     __line 853")
            # if flag.sum()>0:
            #     print(self.raw_weight[flag], "hard clamp negative.  __line 853")
            #     pass
            temp = flag*boundary
            temp = temp.to(self.raw_weight.dtype)
            temp = temp + self.raw_weight.data*(flag.logical_not())
            self.raw_weight.data = temp
            #end of hard clamp!!!

            # soft clamping !!!
            # some soft clamping from the previous hardmax version of digital mapper layer.
            # the positive part
            target_exceed = self.soft_clamp_threshold_up-self.soft_clamp_threshold_low-self.soft_clamp_margin
            gt_up = self.raw_weight.gt(self.soft_clamp_threshold_up)
            # if gt_up.sum()>0:
            #     print(self.raw_weight[gt_up], "soft clamp positive.  __line 853")
            #     pass
            at_least_smt_gt_t2 = gt_up.any(dim=1)
            if at_least_smt_gt_t2.any().item():
                the_max_value = self.raw_weight.max(dim=1, keepdim=True).values
                gt_low = self.raw_weight.gt(self.soft_clamp_threshold_low)
                at_least_smt_gt_t2_expanded = at_least_smt_gt_t2[:, None].expand(-1, self.raw_weight.shape[1])
                modify_these = gt_low.logical_and(at_least_smt_gt_t2_expanded)
                exceed = the_max_value-self.soft_clamp_threshold_low
                exceed = exceed.abs()+self.soft_clamp_epi#or exceed.mul(at_least_smt_gt_t2)+epi
                mul_me = target_exceed/exceed
                self.raw_weight.data = modify_these*((self.raw_weight-self.soft_clamp_threshold_low)*mul_me+self.soft_clamp_threshold_low)+(modify_these.logical_not())*self.raw_weight
            pass
            # if gt_t2.sum()>0:
            #     print(self.raw_weight[gt_t2], "soft clamp positive. AFTER!!!  __line 853")
            #     pass
        
            #at least something has to be positive for every output.
            the_max_for_each_output = self.raw_weight.max(dim=1,keepdim=True).values
            flag_no_one_gt_one = the_max_for_each_output.lt(1.)
            do_i_need_to_do_this = flag_no_one_gt_one.any()
            if do_i_need_to_do_this:
                self.raw_weight.data = flag_no_one_gt_one*(self.raw_weight-the_max_for_each_output+1.)+(flag_no_one_gt_one.logical_not())*self.raw_weight
                pass
            #end of #at least something has to be positive for every output.
            
            pass
        pass
    #end of function.

    def print_param_overlap_ratio(self):
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
            pass
        
        # param protection!!!
        if self.training:
            if self.training_count<self.protect_param_every____training:
                self.training_count+=1
            else:
                self.training_count = 0
                self.protect_raw_weight()
                pass
            pass
            
        x = input
        x = x.unsqueeze(dim=2)
        
        w_after_gramo:torch.Tensor = self.gramo_for_raw_weight(self.raw_weight)
        #self.update_progress_factor()
        w_after_softmax = w_after_gramo.softmax(dim=1)
        x = w_after_softmax.matmul(x)
        
        x = x.squeeze(dim = 2)
        
        x = self.Final_Binarize_doesnt_need_gramo(x)
        x = self.out_gramo(x)#here is the only gramo.
        return x
    
    @staticmethod
    def max_weight_after_softmax(input:torch.Tensor)->torch.Tensor:
        with torch.no_grad():
            after_softmax = input.softmax(dim=1)
                #print(after_softmax)
            max_strength_for_each_output = after_softmax.amax(dim=1, keepdim=True)
                #print(max_strength_for_each_output, "max_strength_for_each_output", "line 909")
            return max_strength_for_each_output
    
    def before_step(self):
        
        '''This function provides a 10% chance to land a critital hit on the layer which deals 250% more damage.'''
        if not self.raw_weight.grad is None:
            if self.last_acc>self.critical_hit_threshold:
                with torch.no_grad():
                    the_max_index = self.raw_weight.max(dim=1, keepdim=False).indices
                    #print(the_max_index, "the_max_index   line 1119")
                    out_features_iota = torch.linspace(0,self.out_features-1,self.out_features,dtype=torch.int32)
                    check_these_grad = self.raw_weight.grad[out_features_iota, the_max_index]
                    #print(check_these_grad, "check_these_grad")
                    flag_neg_grad = check_these_grad.lt(-0.001)
                    #print(flag_neg_grad, "flag_neg_grad")
                    result_of_max_weight_after_softmax = DigitalMapper_V1_1.max_weight_after_softmax(self.raw_weight.data)
                    #print(result_of_max_weight_after_softmax, "result_of_max_weight_after_softmax")
                    flag_top_maxsoft_big_enough = result_of_max_weight_after_softmax.gt(0.48)
                    flag_top_maxsoft_big_enough = flag_top_maxsoft_big_enough.squeeze(dim = 1)
                    #print(flag_top_maxsoft_big_enough, "flag_top_maxsoft_big_enough")
                    hit_this_one = flag_neg_grad.logical_and(flag_top_maxsoft_big_enough)
                    #print(hit_this_one, "hit_this_one")
                    if hit_this_one.any():
                        temp = hit_this_one*(-35.)+hit_this_one.logical_not()*self.raw_weight.data[out_features_iota, the_max_index]
                        temp = temp.to(self.raw_weight.dtype)
                        
                        debug_index_before = self.raw_weight.data.max(dim=1).indices
                        self.raw_weight.data[out_features_iota, the_max_index] = temp
                        debug_index_after = self.raw_weight.data.max(dim=1).indices
                        debug_amount_changed = debug_index_before.ne(debug_index_after).sum()
                        if debug_amount_changed>0:
                            print(debug_amount_changed.item(), "debug_amount_changed   __line 1084")
                            pass
                        
                        self.critical_hit_threshold = self.critical_hit_threshold_original
                    else:
                        drag_factor = 0.002
                        self.critical_hit_threshold = (1-drag_factor)*self.critical_hit_threshold+drag_factor*self.critical_hit_threshold_min
                        pass
                    
                    
                pass
            pass
        pass
    
    def extra_repr(self) -> str:
        return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'
 
    def can_convert_into_eval_only_mode(self, print_repeating_result = False)->Tuple[torch.Tensor, torch.Tensor]:
        r'''output:
        >>> [0] can(True) or cannot(False)
        >>> [1] The least "max strength" of each output. Only for debug.
        '''
        with torch.no_grad():
            # after_softmax = self.raw_weight.softmax(dim=1)
            # #print(after_softmax)
            # max_strength_for_each_output = after_softmax.amax(dim=1, keepdim=True)
            # #print(max_strength_for_each_output, "max_strength_for_each_output", "line 909")
            
            result_of_max_strength_for_each_output = self.max_weight_after_softmax(self.raw_weight)
            least_strength = result_of_max_strength_for_each_output.amin()
            least_strength = least_strength.to(self.raw_weight.device)
            #print(least_strength)
            result_flag = least_strength>self.the_threshold__can_convert_to_eval_only_mode
            result_flag = result_flag.to(self.raw_weight.device)
            
            if print_repeating_result:
                least_strength_now = least_strength.item()
                if least_strength_now<0.52 and least_strength_now>0.2:
                    if least_strength_now == self.debug_least_strength_last_time:
                        print(least_strength_now,"least_strength_now")
                        pass
                    pass
                self.debug_least_strength_last_time = least_strength_now
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
    
    pass




# ''' top softmax == 0.5 problem'''
# layer = DigitalMapper_V1_1(1,4)
# layer.raw_weight.data = torch.ones_like(layer.raw_weight.data)
# layer.set_acc()






# '''before_step function test'''
# layer = DigitalMapper_V1_1(3,4)
# layer.set_acc(1.)
# print(layer.raw_weight.shape)
# layer.raw_weight.data = torch.tensor([
#     [0.042,12.,0.042,],[12.,0.042,0.042,],[0.1,0.042,0.042,],[0.1,0.042,0.042,],])
# layer.raw_weight.grad = torch.tensor([
#     [0.042,-1., 0.042,],[-1.,0.042,0.042,],[1., 0.042,0.042,],[-1.,0.042,0.042,],])
# layer.before_step()
# print(layer.raw_weight.data)
# fds=432
# '''individual reference code. But they are actually very different.'''
# a = torch.tensor([[12.,0,],[12.,0,],[1.,0.],[ 1.,0.],])
# b = torch.tensor([[1., 0,],[-1.,0,],[1.,0,],[-1.,0,],])
# the_max_index = a.max(dim=1, keepdim=False).indices
# print(the_max_index, "the_max_index   line 1119")
# out_features_iota = torch.linspace(0,4-1,4,dtype=torch.int32)
# print(out_features_iota, "out_features_iota")
# check_these_grad = b[out_features_iota, the_max_index]
# print(check_these_grad, "check_these_grad")
# hit_this_one = check_these_grad.lt(0.)
# print(hit_this_one, "hit_this_one")
# fds=432



# '''the 3rd part of the param protection. It makes sure at least something is >=1. for each output.'''
# layer = DigitalMapper_V1_1(3,2)
# print(layer.raw_weight.shape)
# layer.raw_weight.data = torch.tensor([[0.9, 0., -1],[1.9, 0., -1],])
# print(layer.raw_weight.shape)
# layer.protect_raw_weight()
# print(layer.raw_weight.shape)
# print(layer.raw_weight)
# fds=432


# '''sharpen test.
# 1, Too wide + all positive, passed.
# 2, Too wide + all negative.'''
# layer = DigitalMapper_V1_1(4444,1)
# layer.raw_weight.data = torch.ones_like(layer.raw_weight.data)
# layer.raw_weight.data[0,0] = torch.tensor([[1.01]])
# # layer.raw_weight.data = torch.ones_like(layer.raw_weight.data)*-1.
# # layer.raw_weight.data[0,:2] = torch.tensor([[-0.98,-0.99]])
# layer.set_shaper_factor(1.0035,True)
# at_least = 1
# for epoch in range(at_least+1511):
#     every = 100
#     layer.set_acc(1.)
#     if epoch >at_least and epoch%every == every-1:
#         print(epoch,"---------------------")
#         print(layer.raw_weight[0,:5])
#         print(layer.can_convert_into_eval_only_mode())
#         pass
#     layer.protect_raw_weight()
#     if epoch >at_least and epoch%every == every-1:
#         print(layer.can_convert_into_eval_only_mode()[1], "The number __line 1077")
#         pass
#     pass
# fds=432


# '''basic 2 pass test.'''
# input = torch.tensor([[-1.,1]], requires_grad=True)
# layer = DigitalMapper_V1_1(2,3)
# layer.raw_weight.data = torch.tensor([[1.,2],[1.,2],[1.,2],])
# #print(layer.raw_weight.shape)
# pred = layer(input)
# #print(pred.shape)
# print(pred, "pred")
# g_in = torch.ones_like(pred)*10#0000 dosen't change any grad.
# torch.autograd.backward(pred, g_in, inputs=[input,layer.raw_weight])
# print(input.grad, "input.grad")
# print(layer.raw_weight.grad, "layer.raw_weight.grad")
# fds=432

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

# '''set_acc() makes the selection a bit sharper. When it's sharp enough, the 
# can_convert_into_eval_only_mode function return true.'''
# layer = DigitalMapper_V1_1(2,1)
# layer.raw_weight.data = torch.tensor([[1.,2],])
# print(layer.get_softmax_format(), "layer.get_softmax_format   before")
# layer.set_acc(0.9)
# print(layer.raw_weight.data, "layer.raw_weight.data")
# print(layer.get_softmax_format(), "layer.get_softmax_format   after")

# layer = DigitalMapper_V1_1(2,1)
# layer.raw_weight.data = torch.tensor([[0.,0.847],])
# layer.set_the_threshold(0.7)
# print(layer.can_convert_into_eval_only_mode(), "layer.can_convert_into_eval_only_mode   before")
# layer.set_acc(0.9)
# print(layer.can_convert_into_eval_only_mode(), "layer.can_convert_into_eval_only_mode   before")
# fds=432





# '''Param protection test. Real case vs equivalent individual code.'''
# layer = DigitalMapper_V1_1(3,8,protect_param_every____training=0)
# layer.raw_weight.data = torch.tensor([
#     [-6., -3., 3.],[-6., -3., 6.],[-3., 0., 11.],[-6., 2., 12.],
#     [-13., 0., 13.],[3., -3., -6.],[-3., -7., -12.],[-5., -10., -15.],
#     ])
# layer(torch.tensor([[1., 2., 3.]], requires_grad=True))
# print(layer.raw_weight)
# fds=432
# '''Then ,the individual code.'''
# epi = 0.01
# threshold1 = 3.
# threshold2 = 7.
# margin = 1.

# target_exceed = threshold2+threshold1-margin
# test_weight = torch.tensor([
#     [-6., -3., 3.],[-6., -3., 6.],[-3., 0., 11.],[-6., 2., 12.],
#     [-13., 0., 13.],[3., -3., -6.],[-3., -7., -12.],[-5., -10., -15.],
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
    
#     exceed = the_max_value+threshold1
#     exceed = exceed.abs()+epi#or exceed.mul(at_least_smt_gt_t2)+epi
#     #print(exceed, "exceed")
    
#     mul_me = target_exceed/exceed
#     print(mul_me, "mul_me")
#     test_weight = modify_these*((test_weight+threshold1)*mul_me-threshold1)+(modify_these.logical_not())*test_weight
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
    
#     exceed = the_min_value-threshold1
#     exceed = exceed.abs()+epi#or exceed.mul(at_least_smt_gt_t2)+epi
#     #print(exceed, "exceed")
    
#     mul_me_negative = target_exceed/exceed
#     print(mul_me_negative, "mul_me_negative")
#     test_weight = modify_these_negative*((test_weight-threshold1)*mul_me_negative+threshold1) \
#         +(modify_these_negative.logical_not())*test_weight
#     print(test_weight, "test_weight")
# pass
# print("--------------SUMMARIZATION-----------")
# print(layer.raw_weight, "real code.")
# print(test_weight, "test code.")
# print(layer.raw_weight.eq(test_weight))
# print("they are a bit different. The test code only has soft clamping, but the real code has an extra hard clamp which it inherits from the old version.")
# fds=432

# layer = DigitalMapper_V1_1(6,1)
# layer.raw_weight.data = torch.tensor([
#     [-12., -7., -2., 2., 7., 12.]])
# layer.protect_raw_weight()
# print(layer.raw_weight, "from this")
# print(layer.raw_weight[0,5]/layer.raw_weight[0,4], "================the ratio.")
# layer.set_shaper_factor(1.2)
# for _ in range(3):
#     layer.set_acc(0.9)
#     pass
# print(layer.raw_weight, "after sets acc")
# layer.protect_raw_weight()
# print(layer.raw_weight, "after protects param.")
# print(layer.raw_weight[0,5]/layer.raw_weight[0,4], "================the ratio.")
# fds=432







# '''This is an old test. The old version doesn't have a hard binarize layer. 
# Now, only 2(or maybe 3?) of them still work.
# Test for all the modes, and eval only layer.'''
# print("All 4 prints below should be equal.")
# x = torch.tensor([[5., 6., 7.], [8., 9., 13]], requires_grad=True)
# layer = DigitalMapper_V1_1(3,5)
# #print(layer(x))#the hardmax version doesn't have a binarize layer in it.
# print(x[:,layer.get_index_format()])
# print(layer.get_one_hot_format().matmul(x[:,:,None]).squeeze(dim=-1))
# fds=432
# #eval_only_layer = layer.get_eval_only()#not implemented yet. But planned.
# #print(eval_only_layer(x))
# print("All 4 prints above should be equal.")

# fds=432


# ''' probably not needed anymore'''
# '''basic test. Also, the eval mode.'''
# layer = DigitalMapper_V1_1(2,3)
# # print(layer.raw_weight.data.shape)
# layer.raw_weight.data=torch.Tensor([[2., -2.],[ 0.1, 0.],[ -2., 2.]])
# # print(layer.raw_weight.data)
# input = torch.tensor([[1., -1.]], requires_grad=True)
# print(layer(input), "should be 1 1 0")

# layer = DigitalMapper_V1_1(2,3)
# layer.eval()
# layer.raw_weight.data=torch.Tensor([[2., -2.],[ 0.1, 0.],[ -2., 2.]])
# input = torch.tensor([[1., -1.]], requires_grad=True)
# print(layer(input), "should be 1 1 -1")
# fds=432

def data_gen_for_digital_mapper_directly_test(batch:int, n_in:int, n_out:int, dtype = torch.float32)->Tuple[torch.Tensor, torch.Tensor]:
    input = torch.randint(0,2,[batch, n_in], dtype=torch.int8)
    input = input*2-1
    input = input.to(dtype)
    answer_index = torch.randint(0,n_in,[n_out])
    target = input[:, answer_index]
    return input, target

# '''some real training'''
# batch = 100
# n_in = 64
# n_out = 32
# (input, target) = data_gen_for_digital_mapper_directly_test(batch,n_in,n_out)
# input.requires_grad_()
# #input = torch.Tensor([[1., 1.],[1., -1.],[-1., 1.],[-1., -1.],])
# #target = torch.Tensor([[1.],[1.],[-1.],[-1.],])
# #print(input)
# #print(target)

# model = DigitalMapper_V1_1(input.shape[1],target.shape[1])
# model.set_scaling_ratio_for_raw_weight(500.)
# model.set_shaper_factor(1.1)
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# if False:
#     for name, p in zip(model._parameters, model.parameters()):
#         print(name, p)

# iter_per_print = 1#1111
# print_count = 333
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
#             print(pred[:5], "pred")
#             print(target[:5], "target")
#             pass
#         pass
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if False and "make_grad_noisy":
#         make_grad_noisy(model, 1.5)
#         pass
#     if False and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(model.raw_weight.grad, "grad")
#             pass
#         pass
#     if False and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             layer = model
#             print(layer.raw_weight, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "first_layer.in_mapper   after update")
#             pass    
#         pass    
#     if False and "print zero grad ratio":
#         if epoch%iter_per_print == iter_per_print-1:
#             result = model.get_zero_grad_ratio()
#             print("print zero grad ratio: ", result)
#             pass
#         pass
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     if False and "print param overlap":
#         every = 1
#         if epoch%every == every-1:
#             model.print_param_overlap_ratio()
#             pass
#         pass
#     if True and "print acc":
#         if epoch%iter_per_print == iter_per_print-1:
#             with torch.inference_mode():
#                 model.eval()
#                 pred = model(input)
#                 #print(pred, "pred", __line__str())
#                 #print(target, "target")
#                 acc = DigitalMapper_V1_1.bitwise_acc(pred, target)
#                 model.set_acc(acc)
#                 if 1. != acc:
#                     print(epoch+1, "    ep/acc    ", acc)
#                 else:
#                     #print(epoch+1, "    ep/acc    ", acc)
#                     finished = model.can_convert_into_eval_only_mode()
#                     print(finished, "is param hard enough __line 1273")
#                     if finished[0]:
#                         print(pred[:5].T, "pred", __line__str())
#                         print(target[:5].T, "target")
#                         break
#                         pass
#                     pass
#                 pass
#             pass
#         pass
# fds=432



class dry_stack_test_for_digital_mapper_v1_1(torch.nn.Module):
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
        
        self.first_layer = DigitalMapper_V1_1(in_features,mid_width)
        self.mid_layers = torch.nn.ModuleList([
            DigitalMapper_V1_1(mid_width,mid_width)
            for _ in range(num_layers-2)
        ])
        self.last_layer = DigitalMapper_V1_1(mid_width,out_features)
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
    def set_acc(self, acc:float):
        layer:DigitalMapper_V1_1
        layer = self.first_layer
        layer.set_acc(acc)
        for layer in self.mid_layers:
            layer.set_acc(acc)
            pass
        layer = self.last_layer
        layer.set_acc(acc)
        pass
        
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
        self.first_layer.set_scaling_ratio_for_raw_weight(scaling_ratio)
        layer:DigitalMapper_V1_1
        for layer in self.mid_layers:
            layer.set_scaling_ratio_for_raw_weight(scaling_ratio)
            pass
        self.last_layer.set_scaling_ratio_for_raw_weight(scaling_ratio)
        pass
    
    def print_zero_grad_ratio(self):
        result = self.first_layer.get_zero_grad_ratio()
        print("First layer: zero grad ratio: ", result)
        layer:DigitalMapper_V1_1
        for i, layer in enumerate(self.mid_layers):
            result = layer.get_zero_grad_ratio()
            print(f"{i+2}th layer: zero grad ratio: ", result)
            pass
        result = self.last_layer.get_zero_grad_ratio()
        print("Last layer: zero grad ratio: ", result)
    
    def can_convert_into_eval_only_mode(self, epoch:int, print_repeating_result = False)->Tuple[torch.Tensor, torch.Tensor]:
        temp_list:List[Tuple[torch.Tensor, torch.Tensor]] = []
        temp_list.append(self.first_layer.can_convert_into_eval_only_mode(print_repeating_result))
        layer:DigitalMapper_V1_1
        for layer in self.mid_layers:
            temp_list.append(layer.can_convert_into_eval_only_mode(print_repeating_result))
            pass
        temp_list.append(self.last_layer.can_convert_into_eval_only_mode(print_repeating_result))
        pass
        for obj in temp_list:
            print(f"{obj[1].item():.3f}", end=",,,")
            pass
        print("    ", epoch, "    from dry stack test can_convert_into_eval_only_mode function.")
        the_flag = torch.tensor([True], device=self.first_layer.raw_weight.device)
        the_number = torch.tensor([1.], device=self.first_layer.raw_weight.device)
        for temp in temp_list:
            the_flag = the_flag.logical_and(temp[0])
            the_number = the_number.minimum(temp[1])
            pass
        return (the_flag, the_number)
    
    def before_step(self):
        self.first_layer.before_step()
        for layer in self.mid_layers:
            layer.before_step()
            pass
        self.last_layer.before_step()
        pass
    
    pass














'''tests the dry stack.'''
batch = 10
n_in = 32
n_out = 8
mid_width = 32
num_layers = 4
iter_per_print = 100#1111
print_count = 3333
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

model = dry_stack_test_for_digital_mapper_v1_1(input.shape[1],target.shape[1],mid_width,num_layers=num_layers)
loss_function = torch.nn.MSELoss()
model.set_scaling_ratio_for_raw_weight(2000.)
#model.first_layer.set_scaling_ratio_for_raw_weight(2000.)
#print(model.first_layer.gramo_for_raw_weight.scaling_ratio)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
if True and "print parameters":
    if True and "only the training params":
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
model.half()
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
    if True and "make_grad_noisy":
        make_grad_noisy(model, 1.25)
        pass
    #print(model.first_layer.raw_weight_boundary_for_f32.item())
    if False and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            print(model.first_layer.raw_weight.grad, "grad")
            print(model.last_layer.raw_weight.grad, "grad")
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
    if False and "print zero grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            model.print_zero_grad_ratio()
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    #print(model.first_layer.raw_weight.requires_grad,"    __line 1720")
    model.before_step()
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
    
    if True and "print acc":
        with torch.inference_mode():
            #every = 10
            #if epoch%every == every-1:
            model.eval()
            pred = model(input)
            #print(pred, "pred", __line__str())
            #print(target, "target")
            acc = DigitalMapper_V1_1.bitwise_acc(pred, target)
            
            # if 1. == acc and epoch>700:
            #     if epoch%50 == 42:
            #         temp = model.first_layer.raw_weight[0].detach().clone()
            #         temp2 = temp.sort(descending=True).values
            #         print(temp2[:4], "after sortation.")
            #         fds=432
                
            if 1.!=acc:
                print(epoch+1, "    ep/acc    ", acc)
                pass
            model.set_acc(acc)
            if epoch%iter_per_print == iter_per_print-1:
                if 1. != acc:
                    print(epoch+1, "    ep/acc    ", acc)
            if 1. == acc:
                finished = model.can_convert_into_eval_only_mode(epoch)
                if finished[0].logical_not():
                    if epoch%iter_per_print == iter_per_print-1:
                        print(epoch+1, "    ep/acc    ", acc,"    is param hard enough:", f"{finished[1].item():.4f}", "    __line 1515")
                        pass
                    pass
                else:
                    print(pred[:2,:7], "pred", __line__str())
                    print(target[:2,:7], "target")
                    print(pred.ne(target).sum(), "   wrong in total.")
                    print(epoch, "epoch. Finished.")
                    break
                    
                pass
            pass
        pass

fds=432



检查一下哪些地方可以进参数保护。
检查一下哪些地方可以进参数保护。
检查一下哪些地方可以进参数保护。
检查一下哪些地方可以进参数保护。

检查训练数据。如果重叠得多，可能会不一样。
critical hit可能只能对100%的用。
检查参数保护是不是没给翻身的机会。锐化的时候是不是不应该牵扯负数。
