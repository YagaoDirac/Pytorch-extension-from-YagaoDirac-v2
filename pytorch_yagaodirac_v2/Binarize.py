from typing import Any, List, Tuple, Optional, Self
import math
import torch

#my customized.
#from ParamMo import GradientModification_v2

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



# '''Does this gate layer protect the grad?'''
# input = torch.tensor([[-1., -1.],[-1., 1.],[1., 1.]], requires_grad=True)
# pred = Binarize_Forward_only_Function.apply(input, torch.tensor([False]))
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[input])
# print(input.grad)
# fds=432

# '''some basic test'''
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output_is_01 = torch.tensor([True])
# output = Binarize_Forward_only_Function.apply(input, output_is_01)
# print(output, "should be 0., 0., 0., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")
# fds=432
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output_is_01 = torch.tensor([False])
# output = Binarize_Forward_only_Function.apply(input, output_is_01)
# print(output, "should be -1 -1 +1 +1 +1")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")
# fds=432



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


# layer = Binarize_Forward_only(True)
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "should be 0., 0., 0., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")
# fds=432
# layer = Binarize_Forward_only(False)
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "should be -1 -1, 1., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")
# fds=432






class Binarize(torch.nn.Module):
    r"""This layer is not designed to be used by users.
    You should use the 6 following layers instead, they are:
    >>> Binarize_float_to_01(unfinished docs. deleted?)
    >>> Binarize_float_to_np(unfinished docs. deleted?)
    >>> Binarize_01
    >>> Binarize_01_to_np
    >>> Binarize_np
    >>> Binarize_np_to_01
    
    This layer is the base layer of all Binarize layers.
    
    The first 2(Binarize_float_to_01 and Binarize_float_to_np) accepts any range(-inf to +inf) as input,
    then Binarize_01 and Binarize_01_to_np accepts 0. to 1.,
    then Binarize_np and Binarize_np_to_01 accepts -1. to 1..
    But because they all use sigmoid or tanh, they all accepts any real number as input. 
    So you don't have to worry about running into inf or nan.
    
    Binarize_float_to_01, Binarize_01 and Binarize_np_to_01 output 0. to 1.,
    while
    Binarize_float_to_np, Binarize_01_to_np and Binarize_np output -1. to 1..
    But in practice, the real output range depends on all the other params, the input range, the big_number.
    But no matter what happens, the output range is guarunteed to be with in (0., 1.) and (-1., 1.) respectively, and the boundaries are all excluded, because they are direct output of either sigmoid or tanh.
    
    The eval mode is a bit different. The output is directly from step function, in other words, a simple comparison.
    The output can only be the boundary value. For output mode of 01, it's either 0., or 1.. For np it's -1. or 1..
    But if the input equals to the threshold, the behavior is different, it's directly from a preset number.
    To set it, use set_output_when_ambiguous function. The threshold is directly inferred from the input range, 0.5 for 01, 0. for others.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
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


# '''When doesn't need gramo'''
# #input = torch.tensor([[-2.,-1.,-0.,1.,]], requires_grad=True)
# input = torch.tensor([[0.,]], requires_grad=True)
# layer = Binarize.create_analog_to_np(False, big_number_list=[1.23])
# pred = layer(input)
# g_in = torch.ones_like(pred)*100
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "input.grad WITHOUT gramo", __line__str())
# input = torch.tensor([[0.,]], requires_grad=True)
# layer = Binarize.create_analog_to_np(True, big_number_list=[1.23])
# pred = layer(input)
# g_in = torch.ones_like(pred)*100
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "input.grad WITH gramo", __line__str())
# fds=432


# '''concat passes the grad.'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# g_in_1 = torch.ones_like(input)
# torch.autograd.backward(input, g_in_1, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432

# '''Does this gate layer protect the grad?'''
# input = torch.tensor([[-1., -1.],[-1., 1.],[1., 1.]], requires_grad=True)
# layer = Binarize.create_analog_to_np()
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[input])
# print(input.grad)
# fds=432


# ''' All 4 configs.'''
# model = Binarize.create_01_to_01()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be 0 0 0 0 1")
# model = Binarize.create_01_to_01()
# model.eval()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be 0 0 0 0 1")
# model = Binarize.create_01_to_np()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be -1 -1 -1 -1 1")
# model = Binarize.create_01_to_np()
# model.eval()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be -1 -1 -1 -1 1")

# model = Binarize.create_analog_to_01()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be 0 0 1 1 1")
# model = Binarize.create_analog_to_01()
# model.eval()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be 0 0 1 1 1")
# model = Binarize.create_analog_to_np()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be -1 -1 1 1 1")
# model = Binarize.create_analog_to_np()
# model.eval()
# input = torch.tensor([[-1., 0., 0.25, 0.5, 1.]], requires_grad=True)
# print(model(input), "should be -1 -1 1 1 1")
# fds=432


# '''All the big numbers in the list are properly iterated.
# No print. Test with breakpoint.'''
# model = Binarize(True,True,3)
# model.set_big_number_with_float([2., 3., 4.])
# model.set_scaling_ratio([5., 6., 7.])
# input = torch.tensor([[0., 0.25, 0.5, 1.]], requires_grad=True)
# pred = model(input)
# fds=432


# '''some extra test to provide a intuition of how to control and protect the gradient.'''
# model = Binarize.create_analog_to_np()
# input = torch.tensor([[0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]], requires_grad=True)
# #input = torch.tensor([[0., 0., 0., 0.]], requires_grad=True)
# g_in = torch.ones_like(input)

# #model.layer_count = 1
# model.set_big_number_with_float([1.])
# model.set_scaling_ratio([1.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.35(sqrt(1/8)), but decreasing.")

# model.set_scaling_ratio([2.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.7, but decreasing.")

# model.set_big_number_with_float([2.])
# model.set_scaling_ratio([1.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.35(sqrt(1/8)), but decreasing MUCH MORE.")

# model.layer_count = 2
# model.set_big_number_with_float([1., 1.])
# model.set_scaling_ratio([1., 1.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.35(sqrt(1/8)), but decreasing MUCH MORE.")

# model.layer_count = 3
# model.set_big_number_with_float([1., 1., 1.])
# model.set_scaling_ratio([1., 1., 1.])
# print(model.debug_info(2))
# input.grad = None
# pred = model(input)
# torch.autograd.backward(pred, g_in, inputs=input)
# print(input.grad, "should be around 0.35(sqrt(1/8)), but decreasing MUCH MORE.")
# fds=432






# example
# example
# example
# example
# example
# example
# example
# example
# example
# '''np to 01 3 times, in this implementation, it's np to 01 to 01 to 01.'''
# '''Because the forward path yields the same result as that one before, 
# the only difference is grad. The param:big_number affects the grad by a lot.'''
# class example_Binarize_analog_to_01_3times(torch.nn.Module):
#     r"""This example layer shows how to use the binarize layer.
#     """
#     def __init__(self, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         self.Binarize = Binarize.create_analog_to_01()
#         self.Binarize.set_big_number_with_float([0.75, 1., 2.], I_know_Im_setting_a_value_which_may_be_less_than_1=True)
#         pass
#     # 3 optional function.
#     # def accepts_non_standard_range(self)->bool:
#     #     return True
#     # def outputs_standard_range(self)->bool:
#     #     return True
#     # def outputs_non_standard_range(self)->bool:
#     #     return not self.outputs_standard_range()
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         x = input
#         x = self.Binarize(x)
#         return x

#     pass


# input = torch.tensor([-3., -1., 0., 1., 2.], requires_grad=True)
# input = input.unsqueeze(dim=0)
# layer = example_Binarize_analog_to_01_3times()
# output = layer(input)
# print(output, "output, should be 0 0 0 1 1")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs=input)
# print(input.grad, "grad")
# fds=432



# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.
# good old test with good old explainations.


# '''
# !!! This part is old docs. I don't want to delete it, bacause it tells something. 
# This old test made me add the range system into this project.
# The data shown here was got before that.

# Although this test shows the ability for more layers to binarize the result more, 
# but in fact this behavior is not guaranteed. It heavily depends on the big_number param.
# In this example, the big_number for all the 3 layers are set to 3., 6., 6.. 
# If you set a much smaller big_number, the result will be "blurred".
# A always working trick is that, set breakpoint in the forward function,
# and check out what each layer does to the intermediate result.

# This test happens to show another interesting feature.
# The results are:
# input:         -1.,   -0.5,    0.,     ...
# 1 layer output:0.0474, 0.1824, 0.5000, ...
# 3 layer output:0.0674, 0.0977, 0.5000, ...
# 1 layer grad:  0.0606, 0.2001, 0.3354, ...
# 3 layer grad:  0.0077, 0.0692, 0.7294, ...
# The 3 layer version binarize the -0.5 more(0.0977<0.1824),
# but gets relarively smaller grad(0.0692/0.7294<0.2001/0.3354).
# BUT!!!
# The 3 layer version binarize the -1. LESS!!!(0.0674>0.0474),
# wihle still gets relarively smaller grad(0.0077/0.7294<0.0606/0.3354).

# The grad is affected by gradient modification layer.

# The argument is that, it's not reliable to tell the 
# relationship of grad by only read the relationship of outputs.

# Also, notice, if we only consider the range. If we send a range(0, 1)
# to sigmoid, the result is [0.5, 0.7311). The range is getting smaller.
# With big_number, we can scale the range back to something near to the input.
# But, no matter what happens, is we only multiply the output by a scale number,
# and send it to sigmoid, the result is definitely not gonna fill all the range(0, 1).
# The parts near to the 2 boundaries are lost.
# Even if the result is binarized more, it's not possible to tell by only reading the range.
# But the relationship of grad tells everything.


# Now, the scale_back to standard range feature is added through the project.
# Let's compare the old result vs the new result:
# input:                -1.,   -0.5,    0.,     ...
# OLD 1 layer output:0.0474, 0.1824, 0.5000, ...
# NEW                0.0000, 0.1491, 0.5000, ...

# OLD 3 layer output:0.0674, 0.0977, 0.5000, ...
# NEW                0.0000, 0.0350, 0.5000, ...

# OLD 1 layer grad:  0.0606, 0.2001, 0.3354, ...
# NEW                0.0606, 0.2001, 0.3354, ...

# OLD 3 layer grad:  0.0077, 0.0692, 0.7294, ...
# NEW                0.0077, 0.0692, 0.7294, ...

# The grad is not modified at all, which is probably due to gramo, and probably what I want.
# '''

# old test.
# layer = Binarize_np_to_01__non_standard_output(3.)
# input:torch.Tensor = torch.tensor([-1., -0.5, 0., 0.5, 1.], requires_grad=True,)
# input = input.unsqueeze(0)
# output = layer(input)
# r = layer.get_output_range(BoundaryPair.make_np())
# output = r.scale_back_to_01(output)
# print(output, "processed with 1 layer.")
# g_in = torch.ones_like(output, dtype=torch.float32)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad from 1 layer.")
# print(layer.get_output_range(BoundaryPair.make_np()))

# model = example_Binarize_np_to_01_3times()
# input = torch.tensor([-1., -0.5, 0., 0.5, 1.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# output = model(input)
# print(output, "processed with 3 layer.")
# g_in = torch.ones_like(output, dtype=torch.float32)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad from 3 layer.")


# fds=432

