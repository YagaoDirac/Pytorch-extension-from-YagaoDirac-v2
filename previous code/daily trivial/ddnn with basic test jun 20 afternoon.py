'''
The first part is a plain copy from the MIG python file.
Explaination and tests can be found in the original file.

THE NEW CODE STARTS FROM AROUND LINE 350 !!!
THE NEW CODE STARTS FROM AROUND LINE 350 !!!
THE NEW CODE STARTS FROM AROUND LINE 350 !!!
THE NEW CODE STARTS FROM AROUND LINE 350 !!!
THE NEW CODE STARTS FROM AROUND LINE 350 !!!
THE NEW CODE STARTS FROM AROUND LINE 350 !!!
'''
from typing import Any, List
import torch
import math

# __all__ = [
#     'GradientModification',
#     'MirrorLayer',
#     'MirrorWithGramo',
#     'GradientModificationFunction', #Should I expose this?
#     'Linear_gramo', #Should I rename this one? Or somebody help me with the naming?
#     ]

class GradientModificationFunction(torch.autograd.Function):
    r'''input param list:
    x:torch.Tensor,(must be set as require_grad = True)
    learning_rate:float, 
    epi=torch.tensor([1e-12]),
    div_me_when_g_too_small = torch.tensor([1e3])
    
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

        ctx.save_for_backward(scaling_ratio, epi, div_me_when_g_too_small)
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        scaling_ratio:torch.Tensor
        scaling_ratio, epi, div_me_when_g_too_small = ctx.saved_tensors

        #the shape should only be rank2 with[batch, something]
        # original_shape = g.shape
        # if len(g.shape) == 1:
        #     g = g.unsqueeze(1)
        # protection against div 0    
        length = g.mul(g).sum(dim=1,).sqrt()
        too_small = length.le(epi).to(torch.float32)
        div_me = length*(too_small*-1.+1)+div_me_when_g_too_small*too_small
        div_me = div_me.unsqueeze(dim=1)
        g_out:torch.Tensor = g/div_me
        
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
    scaling_ratio:torch.Tensor
    def __init__(self, scaling_ratio:float = 1., \
                       epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio]), requires_grad=False)
        self.epi=torch.nn.Parameter(torch.tensor([epi], requires_grad=False))
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small], requires_grad=False))
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if not x.requires_grad:
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction.apply(x, self.scaling_ratio, self.epi, \
                                                   self.div_me_when_g_too_small)
    def set_scaling_ratio(self, scaling_ratio:float)->None:
        self.scaling_ratio = torch.nn.Parameter(torch.tensor([scaling_ratio], requires_grad=False))
        pass



# This is actually test code. But it looks very useful... 
# I should say, don't use this one. A better one is at the end. You'll love it.
class Linear_gramo(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,\
                            device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear = torch.nn.Linear(in_features, out_features, bias, device, dtype)
        self.gramo = GradientModification()

    def forward(self, x:torch.tensor)->torch.Tensor:
        #maybe I should handle the shape here.
        x = self.linear(x)
        x = self.gramo(x)
        return x
    


# I copied the torch.nn.Linear code and modified it.


class MirrorLayer(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]

    check torch.nn.Linear for other help
    """
    __constants__ = ['in_features', 'out_features', 'auto_merge_duration']
    in_features: int
    out_features: int
    half_weight: torch.Tensor
    half_weight_mirrored: torch.Tensor

    auto_merge_duration:int
    update_count:int

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, auto_merge_duration:int = 20) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.half_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.half_weight_mirrored = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        #to keep track of the training.
        self.auto_merge_duration:int = auto_merge_duration
        self.update_count:int = 0
        pass

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.half_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.half_weight_mirrored.copy_(self.half_weight)
            pass

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.half_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            pass
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.update_count>=self.auto_merge_duration:
            self.update_count = 0
            with torch.no_grad():
                self.half_weight = (self.half_weight+self.half_weight_mirrored)/2.
                self.half_weight_mirrored.copy_(self.half_weight)
                pass
            pass

        head1:torch.Tensor = torch.nn.functional.linear(input + 0.5, self.half_weight)
        head2:torch.Tensor = torch.nn.functional.linear(input - 0.5, self.half_weight_mirrored, self.bias)
        return head1+head2

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def convert_to_plain_Linear(self)->torch.nn.Linear:
        has_bias = bool(self.bias)
        result:torch.nn.Linear = torch.nn.Linear(self.in_features, self.out_features, has_bias)
        with torch.no_grad():
            result.weight = self.half_weight+self.half_weight_mirrored
            result.bias.copy_(self.bias)
            pass
        return result
    
    def copy_from_plain_Linear(self, intput_linear_layer:torch.nn.Linear)->None:
        with torch.no_grad():
            self.half_weight = intput_linear_layer.weight/2.
            self.half_weight_mirrored.copy_(self.half_weight)
            if intput_linear_layer.bias:
                self.bias.copy_(intput_linear_layer.bias)
                pass
            pass

    pass



# The Mirror part ends.
# Now please welcome, Mirror with Gramo.
# Emmm, somebody please rename this.
# Maybe, Mig?

'''I recommend you use this layer only. Forget about the 2 above. 
This layer is very similar to the previous one.
Basically, if you know how to combine those 2, you can do it yourself.
But mathmatically, you should put the gramo at the very end of mirror. 
In other words, you need 1 gramo for every mirror, not 2.
'''

class MirrorWithGramo(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]

    check torch.nn.Linear for other help
    """
    __constants__ = ['in_features', 'out_features', 'auto_merge_duration']
    in_features: int
    out_features: int
    half_weight: torch.Tensor
    half_weight_mirrored: torch.Tensor

    auto_merge_duration:int
    update_count:int

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, auto_merge_duration:int = 20,\
                 scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.half_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.half_weight_mirrored = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        #to keep track of the training.
        self.auto_merge_duration:int = auto_merge_duration
        self.update_count:int = 0

        #this layer also needs the info for gramo.
        self.gramo = GradientModification(scaling_ratio,epi,div_me_when_g_too_small)
        pass

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.half_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.half_weight_mirrored.copy_(self.half_weight)
            pass

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.half_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            pass
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.update_count>=self.auto_merge_duration:
            self.update_count = 0
            with torch.no_grad():
                self.half_weight = (self.half_weight+self.half_weight_mirrored)/2.
                self.half_weight_mirrored.copy_(self.half_weight)
                pass
            pass

        head1:torch.Tensor = torch.nn.functional.linear(input + 0.5, self.half_weight)
        head2:torch.Tensor = torch.nn.functional.linear(input - 0.5, self.half_weight_mirrored, self.bias)
        
        # Basically the only difference from the previous one.
        output = self.gramo(head1+head2)
        return output

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def convert_to_plain_Linear(self)->torch.nn.Linear:
        has_bias = bool(self.bias)
        result:torch.nn.Linear = torch.nn.Linear(self.in_features, self.out_features, has_bias)
        with torch.no_grad():
            result.weight = self.half_weight+self.half_weight_mirrored
            result.bias.copy_(self.bias)
            pass
        return result
    
    def copy_from_plain_Linear(self, intput_linear_layer:torch.nn.Linear)->None:
        with torch.no_grad():
            self.half_weight = intput_linear_layer.weight/2.
            self.half_weight_mirrored.copy_(self.half_weight)
            if intput_linear_layer.bias:
                self.bias.copy_(intput_linear_layer.bias)
                pass
            pass

    pass




































#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////








'''
Here is the end of copying the MIG code.
From here all are DDNN code. DDNN ==  Digital Deep Neural Network.
'''



'''
Notice.
In most cases we use two specifications of analogily representation 
for digital meaning, (-1, 1) and (0, 1). Although the conversion is 
easy, but it's still very easy to mess.
My solution is 2 seperated path for each. The naming convention is:
???_np(negetive 1 to positive 1)
???_01(0 to 1)

I plan to make:
a series of Binarize layers.
AND2, OR2, XOR2, AND3 layers. They can be extended easily following the provided pattern.
Then, ADC and DAC layers.
'''












'''
The relationship between different range of representations:
(float means real number which is -inf to +inf, np means -1 to +1, 01 means 0 to +1)
Binarize layers:
from float, you can go to: 01(Binarize_float_to_01), np(Binarize_float_to_np)
from 01: np, 01
from np: np, 01

Gate layers:
from float, you can go to: np, 01 (AND2_all_range, OR2_all_range, ???_all_range)
(But this layer doesn't stack, or doesn't work after get stacked.)

from 01: 01 (AND2_01, OR2_01, ???_01)
from np: np (AND2_np, OR2_np, ???_np)

The following 2 layers only accept input in a very strict range.
Before using them, scale down the range of input to the corresbonding range.
ADC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01  (with AND2_01, OR2_01, ???_01)
from np: np

DAC:
(the shape changes from (batch,1) to (batch,n) )
from 01, you can go to: 01
from np: np


'''









#最后又调了一下这个类，需要把所有验证全部来一遍。


class Binarize_base(torch.nn.Module):
    r"""This layer is the base layer of all Binarize layers.
    In general, I don't recommend you use this layer directly. 
    You should use the 6 following layers instead, they are:
    
    Binarize_float_to_01
    
    Binarize_float_to_np
    
    Binarize_01
    
    Binarize_01_to_np
    
    Binarize_np
    
    Binarize_np_to_01
    
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
    __constants__ = ['output_when_ambiguous','big_number','is_output_01',
                     'input_is_0_for_01__1_for_np__2_for_float', 'output_dtype']
    output_when_ambiguous: float
    big_number: float
    output_is_01: bool
    input_is_0_for_01__1_for_np__2_for_float:int
    

    def __init__(self, input_is_0_for_01__1_for_np__2_for_real_number:int, \
                    big_number:float, is_output_01:bool, device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #safety:
        if input_is_0_for_01__1_for_np__2_for_real_number not in [0,1,2]:
            raise Exception("Param:input_is_0_for_01__1_for_np__2_for_float can only be 0, 1 or 2. 0 is for (0. to +1.), 1 is for (-1. to +1.), 2 is for real number (-inf to +inf).")
        # if output_when_ambiguous<-1. or output_when_ambiguous>1. :
        #     raise Exception("Param:output_when_ambiguous is out of range. If you mean to do this, use the set_output_when_ambiguous function after init this layer.")
        if big_number<1.:
            raise Exception("Param:big_number is not bit enough.")
            pass
        
        self.input_is_0_for_01__1_for_np__2_for_float = input_is_0_for_01__1_for_np__2_for_real_number
        self.big_number = big_number
        self.output_is_01 = is_output_01
        
        self.output_when_ambiguous = 0.
        if self.output_is_01:
            self.output_when_ambiguous = 0.5
            pass
        
        if None == dtype:
            self.output_dtype = torch.float32
        else:
            self.output_dtype = dtype
            pass

        #this layer also needs the info for gramo.
        self.gramo = GradientModification(scaling_ratio,epi,div_me_when_g_too_small)
        pass


    def set_output_when_ambiguous(self, output:float):
        if self.output_is_01:
            if output<0. or output>1.:
                raise Exception("The output can only be between 0 and 1.")
            pass
        else:
            if output<-1. or output>1.:
                raise Exception("The output can only be between -1 and 1.")
            pass
        self.output_when_ambiguous = output
        pass

    #the older version.
    # def set_output_when_ambiguous(self, output:float, is_output_01:bool, extra_range:float = 0.):
    #     #safety
    #     if extra_range<0.:
    #         raise Exception("Param:extra_range can only be >=0.")
    #     if is_output_01:
    #         if output<0.-extra_range or output>1.+extra_range:
    #             raise Exception("Param:output out of range.")
    #         pass
    #     else:#then output is np(-1 to +1)
    #         if output<1.-extra_range or output>1.+extra_range:
    #             raise Exception("Param:output out of range.")
    #         pass
        
    #     self.output_when_ambiguous = output
    #     pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Both modes need this.
        ambiguous_at = 0.
        if 0 == self.input_is_0_for_01__1_for_np__2_for_float:
            ambiguous_at = 0.5
            pass
        
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

            x = input
            
            # The offset. This only works for 01 as input.
            if 0. != ambiguous_at:
                x = x - ambiguous_at
                
            # this scaling also applies to real number as input.
            if self.big_number!=1.:
                x = x*self.big_number
                pass
            
            if self.output_is_01:
                result = self.gramo(torch.sigmoid(x))
            else:
                result = self.gramo(torch.tanh(x))
                pass
            return result
        
        else:#eval mode
            '''I know the bit wise operations are much faster. 
            But for now I'll keep it f32 for simplisity.
            '''
            with torch.inference_mode():
                
                    
                '''The optimization:
                If output is 01, and output_when_ambiguous is 0., only > is 1.
                If output is 01, and output_when_ambiguous is 1., only >= is 1.
                If output is 01, and output_when_ambiguous is not 0. or 1., it needs 2 parts, the == and >.
                If output is np, and output_when_ambiguous is -1., it needs <= and >.
                If output is np, and output_when_ambiguous is  0., it needs < and >.
                If output is np, and output_when_ambiguous is  1., it needs < and >=.
                If output is np, and output_when_ambiguous is not -1. or 1., it needs all 3 ops.
                '''
                if self.output_is_01:
                    if 0. == self.output_when_ambiguous:
                        return input.gt(ambiguous_at).to(self.output_dtype)
                    if 1. == self.output_when_ambiguous:
                        return input.ge(ambiguous_at).to(self.output_dtype)
                    result = input.eq(ambiguous_at).to(self.output_dtype)*self.output_when_ambiguous
                    result += input.gt(ambiguous_at).to(self.output_dtype)
                    return result
                else:# output is np
                    if -1. == self.output_when_ambiguous:
                        result = input.le(ambiguous_at).to(self.output_dtype)*-1.
                        result += input.gt(ambiguous_at).to(self.output_dtype)
                        return result
                    if 0. == self.output_when_ambiguous:
                        result = input.lt(ambiguous_at).to(self.output_dtype)*-1.
                        result += input.gt(ambiguous_at).to(self.output_dtype)
                        return result
                    if 1. == self.output_when_ambiguous:
                        result = input.lt(ambiguous_at).to(self.output_dtype)*-1.
                        result += input.ge(ambiguous_at).to(self.output_dtype)
                        return result
                    result = input.lt(ambiguous_at).to(self.output_dtype)*-1.
                    result += input.eq(ambiguous_at).to(self.output_dtype)*self.output_when_ambiguous
                    result += input.gt(ambiguous_at).to(self.output_dtype)
                    return result

                # some old code. nvm.                        
                # flag = input.eq(ambiguous_at).to(self.output_dtype)
                # head_output_when_ambiguous = (flag*self.output_when_ambiguous).to(self.output_dtype)
                # head_other = (1-flag)*input.gt(ambiguous_at).to(self.output_dtype)
                # return head_output_when_ambiguous+head_other
        #End of The function

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass

    def extra_repr(self) -> str:
        input_str = "[0. to 1.]"
        if 1 == self.input_is_0_for_01__1_for_np__2_for_float:
            input_str = "[-1. to 1.]"
            pass
        if 2 == self.input_is_0_for_01__1_for_np__2_for_float:
            input_str = "(-inf. to +inf.)"
            pass
        output_str = "(0. to 1.)" if 0 == self.output_is_01 else "(-1. to 1.)"
        mode_str = "training" if self.training else "evaluating"
        
        result = f'Banarize layer, input range:{input_str}, output range:{output_str}, in {mode_str} mode'
        return result

    pass




# '''01 to 01'''
# trivial_big_number = 1.
# layer = Binarize_base(0, 0.5, trivial_big_number, True,)
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), "should be <0.5, ==0.5, >0.5")
# layer.big_number = 9.
# print(layer(input), "should be 0.01_, 0.5, 0.99_")

# '''01 to np'''
# trivial_big_number = 1.
# layer = Binarize_base(0, 0.5, trivial_big_number, False,)
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), "should be <0., 0., >0.")

# ''' np as input is basically the same as real number. No test for it.

# jklfds = 894032;




# '''This tests all the output formula of eval mode.'''
# trivial_big_number = 9.
# two_for_input_range = 2

# layer = Binarize_base(two_for_input_range, 0., trivial_big_number, True,)
# layer.eval()
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be 0., 0., 1.")

# layer = Binarize_base(two_for_input_range, 1., trivial_big_number, True,)
# layer.eval()
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be 0., 1., 1.")

# layer = Binarize_base(two_for_input_range, 0.25, trivial_big_number, True,)
# layer.eval()
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be 0., 0.25, 1.")

# layer = Binarize_base(two_for_input_range, -1., trivial_big_number, False,)
# layer.eval()
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be -1., -1., 1.")

# layer = Binarize_base(two_for_input_range, 0., trivial_big_number, False,)
# layer.eval()
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be -1., 0., 1.")

# layer = Binarize_base(two_for_input_range, 1., trivial_big_number, False,)
# layer.eval()
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be -1., 1., 1.")

# layer = Binarize_base(two_for_input_range, 0.25, trivial_big_number, False,)
# layer.eval()
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be -1., 0.25, 1.")

# jfklds= 584903




class Binarize_float_to_01(torch.nn.Module):
    r"""This layer accepts any range as input, 
    provides a binarized result in the range of 0 to 1, 
    0 and 1 are both excluded in training mode, 
    but included in eval mode.
    
    When input is right in the mid(0.), the default output is set to 0.5.
    Use set_output_when_ambiguous function to modify it.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
    """
    # __constants__ = []

    def __init__(self, device=None, dtype=None, scaling_ratio:float = 1., \
                    epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        output_is_01 = True
        self.base = Binarize_base(2,9.,output_is_01,scaling_ratio = scaling_ratio,
                                  epi = epi, div_me_when_g_too_small = div_me_when_g_too_small)
        pass

    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass

    pass



# '''the tests below are mainly about the gradient modification layers.'''
# layer = Binarize_float_to_01()
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.0012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.25, the same as the dirivitive of sigmoid at 0")

# layer = Binarize_float_to_01()
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.0000012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should much smaller than 0.25")

# layer = Binarize_float_to_01()
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25*0.7 or 0.17_")

# layer = Binarize_float_to_01()
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.5, 0.1]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be smaller than 0.25, but the first should be near 0.25, the second should be very small.")


# '''This test is about the batch. Since I specify the shape for gramo layer as [batch count, length with in each batch].
# Because I already tested the gramo layer in MIG, I don't plan to redo them here.'''
# layer = Binarize_float_to_01()
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25*0.7 or 0.17_")
# layer = Binarize_float_to_01()
# input = torch.tensor([[0.], [0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12], [0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25, since they are in different batches.")

# layer = Binarize_float_to_01()
# input:torch.Tensor = torch.linspace(-3., 3., 7, dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "first 3 are <0.5, then 0.5, then >0.5.")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")


# '''This part tests the output_when_ambiguous. It means, when the input it 0, which doesn't indicate towards any direction. The default output is 0.5. But it's customizable.'''
# layer = Binarize_float_to_01()
# input = torch.tensor([[0.]], requires_grad=True)
# print(layer(input), "should be 0.5")
# layer.eval()
# output = layer(input)
# print(layer(input), "should also be 0.5")
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be 0.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0.25")
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be 0., 0.25, 1.")

#'''The output'''
# layer = Binarize_float_to_01()
# print(layer)

# jkldfs=456879





class Binarize_float_to_np(torch.nn.Module):
    r"""This layer accepts any range as input, 
    provides a binarized result in the range of -1 to 1, 
    -1 and 1 are both excluded in training mode, 
    but included in eval mode.
    
    When input is right in the mid(0.), the default output is set to 0..
    Use set_output_when_ambiguous function to modify it.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
    """
    #__constants__ = []

    def __init__(self, device=None, dtype=None, scaling_ratio:float = 1., \
                    epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        output_is_01 = False
        self.base = Binarize_base(2,9.,output_is_01,scaling_ratio = scaling_ratio,
                                  epi = epi, div_me_when_g_too_small = div_me_when_g_too_small)
        pass

    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass

    pass



# layer = Binarize_float_to_np()
# input:torch.Tensor = torch.linspace(-3., 3., 7, dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "first 3 are <0., then 0., then >0..")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")

# '''This part tests the output_when_ambiguous. It means, when the input it 0, which doesn't indicate towards any direction. The default output is 0.5. But it's customizable.'''
# layer = Binarize_float_to_np()
# input = torch.tensor([[0.]], requires_grad=True)
# print(layer(input), "should be 0.")
# layer.eval()
# output = layer(input)
# print(layer(input), "should also be 0.")
# layer.set_output_when_ambiguous(-1.)
# print(layer(input), "should be -1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0.25")
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), "should be -1., 0.25, 1.")

# jkldfs=456879




class Binarize_01(torch.nn.Module):
    r"""This layer accepts the range of 0 to 1, 
    provides a binarized result in the range of 0 to 1, 
    0 and 1 are both excluded in training mode, 
    but included in eval mode.
    
    When input is right in the mid(0.5), the default output is set to 0.5.
    Use set_output_when_ambiguous function to modify it.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
    """
    #__constants__ = []

    def __init__(self, device=None, dtype=None, scaling_ratio:float = 1., \
                    epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        output_is_01 = True
        self.base = Binarize_base(0,9.,output_is_01,scaling_ratio = scaling_ratio,
                                  epi = epi, div_me_when_g_too_small = div_me_when_g_too_small)
        pass

    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass

    pass



# layer = Binarize_01()
# input:torch.Tensor = torch.tensor([-5., 0., 0.25, 0.5, 0.75, 1., 6.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "the middle one should be 0.5")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")

# '''This part tests the output_when_ambiguous. 
# It means, when the input it 0.5, which doesn't indicate towards any direction. 
# The default output is 0.5. But it's customizable.'''
# layer = Binarize_01()
# input = torch.tensor([[0.5]], requires_grad=True)
# print(layer(input), "should be 0.5")
# layer.eval()
# output = layer(input)
# print(layer(input), "should also be 0.5")
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be 0.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0.25")
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), "should be 0., 0.25, 1.")

# jkldfs=456879



#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.
#Below is untested.



class Binarize_01_to_np(torch.nn.Module):
    r"""This layer accepts the range of 0 to 1, 
    provides a binarized result in the range of -1 to 1, 
    -1 and 1 are both excluded in training mode, 
    but included in eval mode.
    
    When input is right in the mid(0.5), the default output is set to 0..
    Use set_output_when_ambiguous function to modify it.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
    """
    #__constants__ = []

    def __init__(self, device=None, dtype=None, scaling_ratio:float = 1., \
                    epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        output_is_01 = False
        self.base = Binarize_base(0,9.,output_is_01,scaling_ratio = scaling_ratio,
                                  epi = epi, div_me_when_g_too_small = div_me_when_g_too_small)
        pass

    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass

    pass



# layer = Binarize_01_to_np()
# input:torch.Tensor = torch.tensor([-5., 0., 0.25, 0.5, 0.75, 1., 6.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "the middle one should be 0.")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")

'''This part tests the output_when_ambiguous. 
It means, when the input it 0.5, which doesn't indicate towards any direction. 
The default output is 0.. But it's customizable.'''
layer = Binarize_01_to_np()
input = torch.tensor([[0.5]], requires_grad=True)
print(layer(input), "should be 0.")
layer.eval()
output = layer(input)
print(layer(input), "should also be 0.")
layer.set_output_when_ambiguous(0.)
print(layer(input), "should be -1.")
layer.set_output_when_ambiguous(1.)
print(layer(input), "should be 1.")
layer.set_output_when_ambiguous(0.25)
print(layer(input), "should be 0.25")
input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
print(layer(input), "should be -1., 0.25, 1.")

jkldfs=456879
继续。







    # Binarize_np
    # Binarize_np_to_01





















































# old code. 

# class Binarize_01(torch.nn.Module):
#     r"""This layer is designed to be used INSIDE DIGITAL LAYER 
#     to help the layer make the outout more binarized.
    
#     To get the maximum training performance, you should balance between use
#     "LESS this layers but GREATER big_number" with "MORE this layers but SMALLER big_number".
    
#     This layer accepts [0. to 1.] as input.
#     Both 0. and 1. are programmingly accepted, but not mathmatically accepted.
#     provides a binarized result in the range of 0 to 1, 
#     0 and 1 are both excluded in training mode, 
#     but may be included in eval mode depending on how the output_when_ambiguous is set.
    
#     But the upper description is only theory. In practice[unfinished docs]
    
#     I only tested float number as input(f8, f16, f32, f64, bf16),
#     if you provide int as input, this is likely to go wrong.
#     """
#     __constants__ = ['big_number']
#     output_when_ambiguous: float
#     big_number:float

#     def __init__(self, device=None, dtype=None, scaling_ratio:float = 1., 
#                     epi=1e-5, output_when_ambiguous = 0.5, \
#                      big_number:float = 9., \
#                        div_me_when_g_too_small = 1e-3, ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         self.output_when_ambiguous = output_when_ambiguous
#         self.big_number = big_number

#         #this layer also needs the info for gramo.
#         self.gramo = GradientModification(scaling_ratio,epi,div_me_when_g_too_small)
#         pass

# #    def reset_parameters(self) -> None:

#     def set_output_when_ambiguous(self, output:float):
#         if output<0 or output>1:
#             raise Exception("The output can only be between 0 and 1.")
#         self.output_when_ambiguous = output
#         pass

#     def set_big_number(self, big_number:float):
#         if big_number<1.:
#             raise Exception("The big number should be BIG!")
#         self.big_number = big_number
#         pass

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         if self.training:
#             x = (input-0.5)*self.big_number
#             output = self.gramo(torch.sigmoid(x))
#             return output
#         else:
#             with torch.inference_mode():
#                 if 0. == self.output_when_ambiguous:
#                     return input.gt(0.5).to(input.dtype)
#                 if 1. == self.output_when_ambiguous:
#                     return input.ge(0.5).to(input.dtype)
#                 flag = input.eq(0.5).to(input.dtype)
#                 head_output_when_ambiguous = (flag*self.output_when_ambiguous).to(input.dtype)
#                 head_other = (1-flag)*input.gt(0.5).to(input.dtype)
#                 return head_output_when_ambiguous+head_other
#         #End of The function

#     def set_scaling_ratio(self, scaling_ratio:float)->None:
#         self.gramo.set_scaling_ratio(scaling_ratio)
#         pass

#     def extra_repr(self) -> str:
#         mode_str = "train" if self.training else{"eval"}
#         return f'feature count={self.feature_count}, mode{mode_str}'

#     pass




#This tests the Binarize layer.
# layer = Binarize_01(1)
# input:torch.Tensor = torch.linspace(0.1, 0.9, 5, dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# print(input, "This is the input.")
# output = layer(input)
# print(output, "after 1 binarize layer.")
# output = layer(output)
# print(output, "after 2 binarize layer.")
# output = layer(output)
# print(output, "after 3 binarize layer.")
# output = layer(output)
# print(output, "after 4 binarize layer.")
# output = layer(output)
# print(output, "after 5 binarize layer.")

# jkldfs=456879





class example_Binarize_01_ntimes(torch.nn.Module):
    r"""This layer shows how to use the binarize layer.
    """
    __constants__ = ['big_number']
    output_when_ambiguous: float
    big_number:float
    #train_mode: bool

    def __init__(self, feature_count: int, device=None, dtype=None, 
                 scaling_ratio:float = 1., epi=1e-5, output_when_ambiguous = 0.5, \
                     big_number:float = 9., \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.output_when_ambiguous = output_when_ambiguous
        self.big_number = big_number

        #this layer also needs the info for gramo.
        self.binarize1 = Binarize_01()
        self.gramo2 = GradientModification(scaling_ratio,epi,div_me_when_g_too_small)
        self.gramo3 = GradientModification(scaling_ratio,epi,div_me_when_g_too_small)
        pass

#    def reset_parameters(self) -> None:

    def set_output_when_ambiguous(self, output:float):
        if output<0 or output>1:
            raise Exception("The output can only be between 0 and 1.")
        self.output_when_ambiguous = output
        pass

    def set_big_number(self, big_number:float):
        if big_number<1.:
            raise Exception("The big number should be BIG!")
        self.big_number = big_number
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = (input-0.5)*self.big_number
            output = self.gramo(torch.sigmoid(x))
            return output
        else:
            with torch.inference_mode():
                if 0. == self.output_when_ambiguous:
                    return input.gt(0.5).to(input.dtype)
                if 1. == self.output_when_ambiguous:
                    return input.ge(0.5).to(input.dtype)
                flag = input.eq(0.5).to(input.dtype)
                head_output_when_ambiguous = (flag*self.output_when_ambiguous).to(input.dtype)
                head_other = (1-flag)*input.gt(0.5).to(input.dtype)
                return head_output_when_ambiguous+head_other
        #End of The function

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass

    def extra_repr(self) -> str:
        mode_str = "train" if self.training else{"eval"}
        return f'feature count={self.feature_count}, mode{mode_str}'

    pass





















'''to do list:
unfinished docs
有一些feature count不需要。
考虑一下合并某些层。
__all__


'''


























































































































































































































