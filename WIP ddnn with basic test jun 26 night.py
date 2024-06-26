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
from typing import Any, List, Tuple
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
        else:
            self.update_count+=1

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
        else:
            self.update_count+=1

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

















class BoundaryPair:
    r"""Good game starts with good bp.
    BoundaryPare helps managing the real range of digital signals.
    More info in the source code.
    More more info in the "with tests" version.
    >>> bf = BoundaryPair.from_float(0.1, 0.9)
    """
    __constants__ = ['lower','upper',]
    lower:torch.Tensor
    upper:torch.Tensor
    def __init__(self, lower:torch.Tensor, upper:torch.Tensor)->None:
        if len(lower.shape)!=1 or lower.shape[0]!=1:
            raise Exception("Param:lower should be a scalar.")
        if len(upper.shape)!=1 or upper.shape[0]!=1:
            raise Exception("Param:upper should be a scalar.")
        if lower.item()>=upper.item():
            raise Exception("Lower should be less than upper.")

        self.lower = lower        
        self.upper = upper        
        pass
    
    def __str__(self)->str:
        return f'BoundaryPair({self.lower.item():.4f}, {self.upper.item():.4f})'
    
    @staticmethod
    def from_float(lower:float, upper:float):
        if lower>=upper:
            raise Exception("Lower should be less than upper.")
        result = BoundaryPair(torch.tensor([lower]), torch.tensor([upper]))
        return result
    
    @staticmethod
    def make_01():
        return BoundaryPair.from_float(0., 1.)
    @staticmethod
    def make_np():
        return BoundaryPair.from_float(-1., 1.)

    def __calc_boundary_base(self, input_is_01: bool, output_is_01: bool, big_number:float, steps: int = 1):
        r'''This function is not designed to be called by users.
        
        This function helps calculate the boundary of a specific 
        step within binarization process.
        
        If the input for a binarize layer is within the range of (0, 1), 
        the binarize layer firstly offset it to (-0.5, 0.5), then multiplies
        it with the big_number param, and gets the range of (-0.5*big_number, 0.5*big_number).
        Because we care about the backward propagation, the big_number is usually less than 20.
        In a lot of tests, I simply use a number around 5 as the big_number.
        This leads the output of a binarize layer not to fill all the (0, 1),
        which is the theorimatic output range of sigmoid. 
        When we need the binarized result as analog signal or real number and
        do the up coming calculation, we need to know the possible range for 
        every specific step, in order to get the maximum possible precision.

        This class provides 4 functions based on this function, they are:     
        >>> calc_boundary_01
        >>> calc_boundary_np
        >>> calc_boundary_non01_to_01
        >>> calc_boundary_01_to_np
        '''
            
            #old version
            # # The offset. This only works for 01 as input.
            # if 0. != offset:
            #     x = x + offset
                
            # # this scaling also applies to real number as input.
            # if self.big_number!=1.:
            #     x = x*self.big_number
            #     pass
            
            # if self.output_is_01:
            #     result = self.gramo(torch.sigmoid(x))
            # else:
            #     result = self.gramo(torch.tanh(x))
        
        # It's &self, not &mut self. This comment is in Rust syntax.
        lower_bound = self.lower
        upper_bound = self.upper
        
        offset = 0.
        if input_is_01:
            offset = -0.5
        
        # Step 1. 
        if output_is_01:#sigmoid
            lower_bound = torch.sigmoid((lower_bound + offset)*big_number)
            upper_bound = torch.sigmoid((upper_bound + offset)*big_number)
            pass
        else:#tanh
            lower_bound = torch.tanh((lower_bound + offset)*big_number)
            upper_bound = torch.tanh((upper_bound + offset)*big_number)
            pass
        
        if 1 == steps:
            return BoundaryPair(lower_bound, upper_bound)
        
        #The offset may vary between the 1st layer and other layers.
        if output_is_01:#sigmoid
            offset = -0.5
        else:#tanh
            offset = 0.
            pass
        # Step 2 to n
        if output_is_01:#sigmoid
            for _ in range(steps-1):
                lower_bound = torch.sigmoid((lower_bound + offset)*big_number)
                upper_bound = torch.sigmoid((upper_bound + offset)*big_number)
                pass
            pass
        else:#tanh
            for _ in range(steps-1):
                lower_bound = torch.tanh((lower_bound + offset)*big_number)
                upper_bound = torch.tanh((upper_bound + offset)*big_number)
                pass
            pass
        return BoundaryPair(lower_bound, upper_bound)

    def calc_boundary_01(self, big_number:float, steps: int = 1):
        r'''
        >>> calc_boundary_01(..., steps = 3) 
        is equivalent to 
        >>> calc_boundary_01(..., steps = 1) 3 times
        '''
        return self.__calc_boundary_base(True,True, big_number, steps=steps)
    def calc_boundary_np(self, big_number:float, steps: int = 1):
        r'''
        >>> calc_boundary_np(..., steps = 3) 
        is equivalent to 
        >>> calc_boundary_np(..., steps = 1) 3 times
        '''
        return self.__calc_boundary_base(False,False, big_number, steps=steps)
    def calc_boundary_non01_to_01(self, big_number:float, steps: int = 1):
        r'''
        >>> calc_boundary_01_to_np(..., steps=3)
        is equivalent to
        >>> calc_boundary_01_to_np(...).calc_boundary_np(...).calc_boundary_np(...)
        Because the first calc_boundary_01_to_np returns the range in the np format.
        The 2 after calls should be to calc_boundary_np.
        '''
        return self.__calc_boundary_base(False,True, big_number, steps=steps)
    def calc_boundary_01_to_np(self, big_number:float, steps: int = 1):
        r'''
        >>> calc_boundary_non01_to_01(..., steps=3)
        is equivalent to
        >>> calc_boundary_non01_to_01(...).calc_boundary_01(...).calc_boundary_01(...)
        Because the first calc_boundary_non01_to_01 returns the range in the 01 format.
        The 2 after calls should be to calc_boundary_01.
        '''
        return self.__calc_boundary_base(True,False, big_number, steps=steps)


    def scale_back_to_01(self, input:torch.Tensor)->torch.Tensor:
        r"""
        example:
        >>> input = torch.tensor([0.1,0.9])
        >>> bp = BoundaryPair.from_float(0.1, 0.9)
        >>> print(bp.scale_back_to_01(input), "should be 0., 1.")
        """
        range_length = self.upper - self.lower
        x = input - self.lower
        x /= range_length
        return x

    def scale_back_to_np(self, input:torch.Tensor)->torch.Tensor:
        r"""
        example:
        >>> input = torch.tensor([0.1,0.9])
        >>> bp = BoundaryPair.from_float(0.1, 0.9)
        >>> print(bp.scale_back_to_np(input), "should be -1., 1.")
        """
        mid_point = (self.lower + self.upper)/2.
        half_range_length = (self.upper - self.lower)/2.
        x = input - mid_point
        x /= half_range_length
        return x



# input = torch.tensor([0.1,0.9])
# bp = BoundaryPair.from_float(0.1, 0.9)
# print(bp.scale_back_to_01(input), "should be 0., 1.")
# print(bp.scale_back_to_np(input), "should be -1., 1.")

# fds=432

# '''tanh(x) == sigmoid(2x)*2-1'''
# print(torch.sigmoid(torch.linspace(0.1,0.9,9,dtype=torch.float32)*2.)*2.-1.)
# print(torch.tanh(torch.linspace(0.1,0.9,9,dtype=torch.float32)))
# bp01 = BoundaryPair.from_float(0., 1.)
# bpnp = BoundaryPair.from_float(-1., 1.)
# sig_0_5 = torch.sigmoid(torch.Tensor([0.5])).item()
# sig_1 = torch.sigmoid(torch.Tensor([1.])).item()
# sig_2 = torch.sigmoid(torch.Tensor([2.])).item()
# tanh_0_5 = torch.tanh(torch.Tensor([0.5])).item()
# tanh_1 = torch.tanh(torch.Tensor([1.])).item()
# tanh_2 = torch.tanh(torch.Tensor([2.])).item()
# print(bp01.calc_boundary_01(1.), f'(0, 1) through sigmoid once. It should be {1-sig_0_5:.4f}, {sig_0_5:.4f}')
# print(bp01.calc_boundary_01(2.), f'(0, 1)*2 through sigmoid once. It should be {1-sig_1:.4f}, {sig_1:.4f}')
# print(bp01.calc_boundary_np(1.), f'(0, 1) through tanh once. It should be 0., {tanh_1:.4f}')
# print(bp01.calc_boundary_np(2.), f'(0, 1)*2 through tanh once. It should be 0., {tanh_2:.4f}')
# print(bpnp.calc_boundary_non01_to_01(1.), f'(-1, 1) through sigmoid once. It should be {1-sig_1:.4f}, {sig_1:.4f}')
# print(bpnp.calc_boundary_non01_to_01(2.), f'(-1, 1)*2 through sigmoid once. It should be {1-sig_2:.4f}, {sig_2:.4f}')
# print(bp01.calc_boundary_01_to_np(1.), f'(0, 1) through tanh once. It should be {-tanh_0_5:.4f}, {tanh_0_5:.4f}')
# print(bp01.calc_boundary_01_to_np(2.), f'(0, 1)*2 through tanh once. It should be {-tanh_1:.4f}, {tanh_1:.4f}')

# '''Tests for multiple steps calculation.'''
# bp01 = BoundaryPair.from_float(0., 1.)
# bpnp = BoundaryPair.from_float(-1., 1.)
# x = bp01.calc_boundary_01(2.)
# x = x.calc_boundary_01(2.)
# y = bp01.calc_boundary_01(2., steps=2)
# print(x, "should be ",y)
# x = bp01.calc_boundary_01(2.).calc_boundary_01(2.).calc_boundary_01(2.)
# y = bp01.calc_boundary_01(2., steps=3)
# print(x, "should be ",y)
# x = bp01.calc_boundary_01(5.).calc_boundary_01(5.).calc_boundary_01(5.)
# y = bp01.calc_boundary_01(5., steps=3)
# print(x, "should be ",y)
# x = bp01.calc_boundary_non01_to_01(5.).calc_boundary_01(5.).calc_boundary_01(5.)
# y = bp01.calc_boundary_non01_to_01(5., steps=3)
# print(x, "should be ",y)
# x = bpnp.calc_boundary_non01_to_01(5.).calc_boundary_01(5.).calc_boundary_01(5.)
# y = bpnp.calc_boundary_non01_to_01(5., steps=3)
# print(x, "should be ",y)
# x = bp01.calc_boundary_np(0.9).calc_boundary_np(0.9).calc_boundary_np(0.9)
# y = bp01.calc_boundary_np(0.9, steps=3)
# print(x, "should be ",y)
# x = bpnp.calc_boundary_np(0.9).calc_boundary_np(0.9).calc_boundary_np(0.9)
# y = bpnp.calc_boundary_np(0.9, steps=3)
# print(x, "should be ",y)
# x = bp01.calc_boundary_01_to_np(1.3).calc_boundary_np(1.3).calc_boundary_np(1.3)
# y = bp01.calc_boundary_01_to_np(1.3, steps=3)
# print(x, "should be ",y)

# fds=432





# old code.
# old code.
# old code.
# old code.
# old code.
# old code.
# def Boundary_calc_helper_wrapper(lower_bound:float, upper_bound:float)->Tuple[torch.Tensor,torch.Tensor]:
#     lower = torch.tensor([lower_bound], dtype=torch.float32)
#     upper = torch.tensor([upper_bound], dtype=torch.float32)
#     return (lower, upper)

# def Boundary_calc_helper_unwrapper(input:Tuple[torch.Tensor,torch.Tensor])->Tuple[float, float]:
#     return (input[0].item(), input[1].item())

# def Boundary_calc(output_is_01: bool, bound:Tuple[torch.Tensor,torch.Tensor], \
#                 steps: int = 1, big_number:float = 1.)->Tuple[torch.Tensor,torch.Tensor]:
#     '''This function helps calculate the boundary of a specific step within binarization process.
#     It's mainly used in ADC and DAC layers.
    
#     If the input for a binarize layer is within the range of (0, 1), 
#     the binarize layer firstly offset it to (-0.5, 0.5), then multiplies
#     it with the big_number param, and gets the range of (-0.5*big_number, 0.5*big_number).
#     Because we care about the backward propagation, the big_number is usually less than 20.
#     In a lot of tests, I simply use a number around 5 as the big_number.
#     This leads the output of a binarize layer not to fill all the (0, 1),
#     which is the theorimatic output range of sigmoid. When we need the binarized result 
#     as analog signal or real number and do the up coming calculation, we need to know
#     the possible range for every specific step, in order to get the maximum possible precision.
    
#     example:
#     >>> input = Boundary_calc_helper_wrapper(0., 1.)
#     >>> output = Boundary_calc(True, input, 3, 5.)
#     >>> print(f'input:(0, 1) through 01_range 3 times with big_number = 5. is {output}')
#     '''
    
#     if big_number<1.:
#         raise Exception("Param:big_number is not big enough.")
#         pass
    
#     lower_bound = bound[0]
#     upper_bound = bound[1]
    
#     offset = 0.
#     if output_is_01:
#         offset = 0.5
    
#     for _ in range(steps):
#         lower_bound = torch.sigmoid((lower_bound-offset)*big_number)
#         upper_bound = torch.sigmoid((upper_bound-offset)*big_number)
#     return (lower_bound, upper_bound)


# big_number=1.
# input = Boundary_calc_helper_wrapper(0., 1.)
# bound = Boundary_calc(True, input, 1, big_number)
# print(bound, f'(0, 1) through sigmoid once.')
# bound = Boundary_calc(True, bound, 1, big_number)
# print(bound, f'(0, 1) through sigmoid twice.')

# big_number=2.
# input = Boundary_calc_helper_wrapper(0., 1.)
# bound = Boundary_calc(True, input, 1, big_number)
# print(bound, f'(0, 1) through sigmoid once with big number of {big_number}.')
# bound = Boundary_calc(True, bound, 1, big_number)
# print(bound, f'(0, 1) through sigmoid twice with big number of {big_number}.')

# input = Boundary_calc_helper_wrapper(0., 1.)
# output = Boundary_calc(True, input, 3, 5.)
# print(f'input:(0, 1) through 01_range 3 times with big_number = 5. is {output}')

# jhklfds = 890234



# Do I need this function?
# def Boundary_blender(input:List[Tuple[torch.Tensor,torch.Tensor]])->Tuple[torch.Tensor,torch.Tensor]:
#     '''
#     >>> a = Boundary_calc_helper_wrapper(0.1, 0.9)
#     >>> b = Boundary_calc_helper_wrapper(0.2, 0.7)
#     >>> bounds = list()
#     >>> bounds.append(a)
#     >>> bounds.append(b)
#     >>> blended_bound = Boundary_blender(bounds)
#     '''
#     lower:torch.Tensor = torch.zeros([1],dtype=torch.float32)
#     upper:torch.Tensor = torch.zeros([1],dtype=torch.float32)
    
#     for b in input:
#         lower += b[0]
#         upper += b[1]
#         pass
#     lower/=len(input)
#     upper/=len(input)
#     return (lower, upper)


# a = Boundary_calc_helper_wrapper(0.1, 0.9)
# b = Boundary_calc_helper_wrapper(0.2, 0.7)
# bounds = list()
# bounds.append(a)
# bounds.append(b)
# blended_bound = Boundary_blender(bounds)
# print(blended_bound)

# fds = 432





class Clamp_ForwardOnly_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x:torch.Tensor = args[0]
        lower_bound:torch.Tensor = args[1]#scalar
        upper_bound:torch.Tensor = args[2]#scalar
        #if len(lower_bound.shape)!=1: this checking is in the wrapper class.
        #if lower_bound.shape[0]!=1: this checking is in the wrapper class.
        #if len(upper_bound.shape)!=1: this checking is in the wrapper class.
        #if upper_bound.shape[0]!=1: this checking is in the wrapper class.
        
        flag = x.gt(upper_bound)
        temp = flag*upper_bound
        temp = temp + x.data*(flag.logical_not())
        x.data = temp

        flag = x.lt(lower_bound)
        temp = flag*lower_bound
        temp = temp + x.data*(flag.logical_not())
        x.data = temp
        return x

    @staticmethod
    def backward(ctx, g):
        return g, None, None

    pass  # class


# lower_bound = torch.tensor([0.], requires_grad=False)
# upper_bound = torch.tensor([1.], requires_grad=False)
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = Clamp_ForwardOnly_Function.apply(input,lower_bound, upper_bound)
# print(output, "should be 0., 0., 0.5, 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")

# fds=432



class Clamp_ForwardOnly(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It clamp the forward inter layer data, while doesn't touch the backward propagation.
    """
    lower_bound:torch.Tensor
    upper_bound:torch.Tensor
    def __init__(self, lower_bound:float, upper_bound:float, \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if lower_bound>=upper_bound-0.5:
            raise Exception("I designed this in this way. If you know what you are doing, modify the code.")

        self.lower_bound = torch.tensor([lower_bound], requires_grad=False)
        self.upper_bound = torch.tensor([upper_bound], requires_grad=False)
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if not x.requires_grad:
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")


        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return Clamp_ForwardOnly_Function.apply(x, self.lower_bound, self.upper_bound)



# layer = Clamp_ForwardOnly(0., 1.)
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "should be 0., 0., 0.5, 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")

# fds=432





class Binarize_base(torch.nn.Module):
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
    __constants__ = ['output_when_ambiguous','big_number',
                     'input_is_01', 'output_is_01',]
    output_when_ambiguous: float
    big_number: float
    input_is_01:bool
    output_is_01: bool

    def __init__(self, input_is_01:bool, \
                    big_number:float, is_output_01:bool, device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        #safety:
        # if output_when_ambiguous<-1. or output_when_ambiguous>1. :
        #     raise Exception("Param:output_when_ambiguous is out of range. If you mean to do this, use the set_output_when_ambiguous function after init this layer.")
        if big_number<1.:
            raise Exception("Param:big_number is not big enough.")
            pass
        
        self.input_is_01 = input_is_01
        self.output_is_01 = is_output_01
        self.set_big_number(big_number)
        
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

    # def update_output_range(self):
    def get_output_range(self, input:BoundaryPair)->BoundaryPair:
        '''The range pass'''
        if self.input_is_01:
            if self.output_is_01:
                return input.calc_boundary_01(self.big_number)
            else:#output is np
                return  input.calc_boundary_01_to_np(self.big_number)
            pass
        else:#input is real number or np
            if self.output_is_01:
                return  input.calc_boundary_non01_to_01(self.big_number)
            else:#output is np
                return  input.calc_boundary_np(self.big_number)
            pass
        #end of function
        

    def set_output_when_ambiguous(self, output:float):
        if self.output_is_01:
            if output<0. or output>1.:
                raise Exception("The output can only be between 0 and 1.")
            pass
        else:
            if output<-1. or output>1.:
                raise Exception("The output can only be between -1 and 1.")
            pass
        # if output < self.output_range.lower.item():
        #     raise Exception("Param:output too small.")
        # if output > self.output_range.upper.item():
        #     raise Exception("Param:output too big.")
        
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

    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''I_know_Im_setting_a_value_which_may_be_less_than_1 is 
            designed only for the first binarize layer in gates layers. 
            If inputs doesn't accumulate enough gradient, you should 
            consider using some 0.3 as big_number to protect the 
            trainability of the intput-end of the model.
        '''
            
        if I_know_Im_setting_a_value_which_may_be_less_than_1:
            # This case is a bit special. I made this dedicated for the gates layers.
            if big_number<=0.:
                raise Exception(
                    '''Param:big_number is not big enough. 
                    ''')
                pass
            
        else:# The normal case
            if big_number<1.:
                raise Exception('''Param:big_number is not big enough. 
                                Use I_know_Im_setting_a_value_which_may_be_less_than_1 = True 
                                if you know what you are doing.''')
                pass
        self.big_number = big_number
        #self.update_output_range()
        pass
    
    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass

    def __get_offset(self)->float:
        if self.input_is_01:
            return -0.5
        else:
            return 0.
            pass
        #untested
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

            x = input
            offset = self.__get_offset()
            
            # The offset. This only works for 01 as input.
            if 0. != offset:
                x = x + offset
                
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
                
                ambiguous_at = self.__get_offset()*-1. #opposite of offset.
                # old code.
                # ambiguous_at = 0. #opposite of offset.
                # if 0 == self.input_is_0_for_01__1_for_np__2_for_float:
                #     ambiguous_at = 0.5
                #     pass
                
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

    
    def extra_repr(self) -> str:
        input_str = "[0. to 1.]"
        if not self.input_is_01:
            input_str = "[-1. to 1.] or (-inf. to +inf.)"
            pass
        output_str = "(0. to 1.)" if self.output_is_01 else "(-1. to 1.)"
        mode_str = "training" if self.training else "evaluating"
        
        result = f'Banarize layer, theorimatic input range:{input_str}, theorimatic output range:{output_str}, in {mode_str} mode'
        return result

    pass



# bp01 = BoundaryPair.from_float(0., 1.)
# '''01 to 01'''
# layer = Binarize_base(True, 1., True)
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), f'should be {layer.get_output_range(bp01).lower.item():.4f}, 0.5, {layer.get_output_range(bp01).upper.item():.4f}')
# layer.set_big_number(9.)
# print(layer(input), "should be 0.01_, 0.5, 0.99_")

# '''01 to np'''
# layer = Binarize_base(True, 1., False)
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), f'should be {layer.get_output_range(bp01).lower.item():.4f}, 0., {layer.get_output_range(bp01).upper.item():.4f}')
# layer.set_big_number(9.)
# print(layer(input), "should be -0.99_, 0., 0.99_")

# bpnp = BoundaryPair.from_float(-1., 1.)
# '''non 01 to 01'''
# layer = Binarize_base(False, 1., True)
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), f'should be {layer.get_output_range(bpnp).lower.item():.4f}, 0.5, {layer.get_output_range(bp01).upper.item():.4f}')
# layer.set_big_number(9.)
# print(layer(input), "should be 0.01_, 0.5, 0.99_")

# '''non 01 to np'''
# layer = Binarize_base(False, 1., False)
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), f'should be {layer.get_output_range(bpnp).lower.item():.4f}, 0., {layer.get_output_range(bp01).upper.item():.4f}')
# layer.set_big_number(9.)
# print(layer(input), "should be -0.99_, 0., 0.99_")

# ''' real number as input is the same as np. No test for it.''' 

# fds=432



# '''This tests all the output formula of eval mode.'''
# trivial_big_number = 9.
# input = torch.tensor([[-1., 0., 1.]])

# layer = Binarize_base(False, trivial_big_number, True,)
# layer.eval()
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be 0., 0., 1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 0., 1., 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0., 0.25, 1.")

# layer = Binarize_base(False, trivial_big_number, False,)
# layer.eval()
# layer.set_output_when_ambiguous(-1.)
# print(layer(input), "should be -1., -1., 1.")
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be -1., 0., 1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be -1., 1., 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be -1., 0.25, 1.")


# trivial_big_number = 9.
# bp01 = BoundaryPair.from_float(0., 1.)
# input = torch.tensor([[0., 0.5, 1.]])

# layer = Binarize_base(True, trivial_big_number, True,)
# layer.eval()
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be 0., 0., 1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be 0., 1., 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be 0., 0.25, 1.")

# layer = Binarize_base(True, trivial_big_number, False,)
# layer.eval()
# layer.set_output_when_ambiguous(-1.)
# print(layer(input), "should be -1., -1., 1.")
# layer.set_output_when_ambiguous(0.)
# print(layer(input), "should be -1., 0., 1.")
# layer.set_output_when_ambiguous(1.)
# print(layer(input), "should be -1., 1., 1.")
# layer.set_output_when_ambiguous(0.25)
# print(layer(input), "should be -1., 0.25, 1.")

# fds=432







#old code
#old code
#old code
#old code
# class Binarize_float_to_01(torch.nn.Module):
#     r"""This layer accepts any range as input, 
#     provides a binarized result in the range of 0 to 1, 
#     0 and 1 are both excluded in training mode, 
#     but included in eval mode.
    
#     When input is right in the mid(0.), the default output is set to 0.5.
#     Use set_output_when_ambiguous function to modify it.
    
#     Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
#     If you provide int as input, it's not gonna work.
    
#     The shape should be [batch size, length within each batch]
#     """
#     # __constants__ = []

#     def __init__(self, big_number = 9., device=None, dtype=None, \
#                     scaling_ratio:float = 1., epi=1e-5, \
#                        div_me_when_g_too_small = 1e-3, ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()

#         #this layer also needs the info for gramo.
#         output_is_01 = True
#         self.base = Binarize_base(2,big_number,output_is_01,scaling_ratio = scaling_ratio,
#                                   epi = epi, div_me_when_g_too_small = div_me_when_g_too_small)
#         pass

#     def set_output_when_ambiguous(self, output:float):
#         '''Simply set it to the inner layer.'''
#         self.base.set_output_when_ambiguous(output)
#         pass
    
#     def set_big_number(self, big_number:float, \
#                 I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
#         '''Simply set it to the inner layer.'''
#         self.base.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
#         pass

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         #if you want to modify this function, copy the forward function from the base class.
#         return self.base(input)

#     def set_scaling_ratio(self, scaling_ratio:float)->None:
#         '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
#         '''
#         self.base.gramo.set_scaling_ratio(scaling_ratio)
#         pass

#     pass



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




#old code
#old code
#old code
#old code
# class Binarize_float_to_np(torch.nn.Module):
#     r"""This layer accepts any range as input, 
#     provides a binarized result in the range of -1 to 1, 
#     -1 and 1 are both excluded in training mode, 
#     but included in eval mode.
    
#     When input is right in the mid(0.), the default output is set to 0..
#     Use set_output_when_ambiguous function to modify it.
    
#     Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
#     If you provide int as input, it's not gonna work.
    
#     The shape should be [batch size, length within each batch]
#     """
#     #__constants__ = []

#     def __init__(self, big_number = 9., device=None, dtype=None, \
#                     scaling_ratio:float = 1., epi=1e-5, \
#                        div_me_when_g_too_small = 1e-3, ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()

#         #this layer also needs the info for gramo.
#         output_is_01 = False
#         self.base = Binarize_base(2,big_number,output_is_01,scaling_ratio = scaling_ratio,
#                                   epi = epi, div_me_when_g_too_small = div_me_when_g_too_small)
#         pass

#     def set_output_when_ambiguous(self, output:float):
#         '''Simply set it to the inner layer.'''
#         self.base.set_output_when_ambiguous(output)
#         pass
    
#     def set_big_number(self, big_number:float, \
#                 I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
#         '''Simply set it to the inner layer.'''
#         self.base.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
#         pass
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         #if you want to modify this function, copy the forward function from the base class.
#         return self.base(input)

#     def set_scaling_ratio(self, scaling_ratio:float)->None:
#         '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
#         '''
#         self.base.gramo.set_scaling_ratio(scaling_ratio)
#         pass

#     pass



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





class Binarize_np_to_01__non_standard_output(torch.nn.Module):
    r"""This layer accepts the range of -1 to 1, 
    provides a binarized result in the range of 0 to 1, 
    0 and 1 are both excluded in training mode, 
    but included in eval mode.
    
    When input is right in the mid(0.), the default output is set to 0.5.
    Use set_output_when_ambiguous function to modify it.
    
    Only designed to accept input of any real number, f8, f16, f32, f64, bf16.
    If you provide int as input, it's not gonna work.
    
    The shape should be [batch size, length within each batch]
    """
    #__constants__ = []

    def __init__(self, big_number = 5., \
                            device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        self.base = Binarize_base(False, big_number, True, 
                                  scaling_ratio = scaling_ratio, epi = epi,
                                  div_me_when_g_too_small = div_me_when_g_too_small)
        pass

    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.base.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
    def get_output_range(self, input: BoundaryPair)->BoundaryPair:
        '''Gets the output range from inner layer'''
        return self.base.get_output_range(input)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    pass



# '''the tests below are mainly about the gradient modification layers.'''
# layer = Binarize_np_to_01__non_standard_output(1.)
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.0012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.25, the same as the dirivitive of sigmoid at 0")

# layer = Binarize_np_to_01__non_standard_output(5.)
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.0012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 1.25, sigmoid(0) is 0.25, big_number is 5.")

# '''An extra gramo layer before the binarize layer to protect the gradient.'''
# gramo = GradientModification()
# layer = Binarize_np_to_01__non_standard_output(1.)
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(gramo(input))
# g_in = torch.tensor([[0.0012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "now it's 1.")

# layer = Binarize_np_to_01__non_standard_output(1.)
# input = torch.tensor([[0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.0000012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, f'should be 0.0000012*1000*0.25({0.0000012*1000*0.25})')

# layer = Binarize_np_to_01__non_standard_output(1.)
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25*0.7 or 0.17_")

# layer = Binarize_np_to_01__non_standard_output(1.)
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.5, 0.05]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be smaller than 0.25, but the first should be near 0.25, the second should be very small. The first one should be 10x of the second one.")

# '''This test is about the batch. Since I specify the shape 
# for gramo layer as [batch count, length with in each batch].
# Because I already tested the gramo layer in MIG, I don't plan to redo them here.'''
# layer = layer = Binarize_np_to_01__non_standard_output(1.)
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25*0.7 or 0.17_")
# layer = layer = Binarize_np_to_01__non_standard_output(1.)
# input = torch.tensor([[0.], [0.]], requires_grad=True)
# output = layer(input)
# g_in = torch.tensor([[0.12], [0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "both should be 0.25, since they are in different batches.")

# '''shape doesn't matter.'''
# layer = Binarize_np_to_01__non_standard_output(1.)
# input:torch.Tensor = torch.linspace(-3., 3., 7, dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "first 3 are <0.5, then 0.5, then >0.5.")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")
# print(layer.get_output_range(BoundaryPair.from_float(-3., 3.)))

# layer = Binarize_np_to_01__non_standard_output(1.)
# input:torch.Tensor = torch.tensor([-5., -1., -0.5, 0., 0.5, 1., 5.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "the middle one should be 0.5")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")

# '''This part tests the output_when_ambiguous. It means, when the input it 0, which doesn't indicate towards any direction. The default output is 0.5. But it's customizable.'''
# layer = Binarize_np_to_01__non_standard_output(1.)
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

# '''This part tests the output_when_ambiguous. 
# It means, when the input it 0., which doesn't indicate towards any direction. 
# The default output is 0.5. But it's customizable.'''
# layer = Binarize_np_to_01__non_standard_output(1.)
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
# print(layer(input), "should be -0., 0.25, 1.")

# '''The output'''
# layer = Binarize_np_to_01__non_standard_output(1.)
# print(layer)

# fds=432



class Binarize_np__non_standard_output(torch.nn.Module):
    r"""This layer accepts the range of -1 to 1, 
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

    def __init__(self, big_number = 5., \
                            device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()


        #this layer also needs the info for gramo.
        self.base = Binarize_base(False, big_number, False,
                                  scaling_ratio = scaling_ratio, epi = epi, 
                                  div_me_when_g_too_small = div_me_when_g_too_small)
        pass

    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.base.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
    def get_output_range(self, input: BoundaryPair)->BoundaryPair:
        '''Gets the output range from inner layer'''
        return self.base.get_output_range(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    pass



# '''shape doesn't matter.'''
# layer = Binarize_np__non_standard_output(1.)
# input:torch.Tensor = torch.tensor([-2., -1., -0.5, 0., 0.5, 1., 3.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "the middle one should be 0.")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")
# print(layer.get_output_range(BoundaryPair.from_float(-2., 3.)))

# '''This part tests the output_when_ambiguous. 
# It means, when the input it 0., which doesn't indicate towards any direction. 
# The default output is 0.. But it's customizable.'''
# layer = Binarize_np__non_standard_output(1.)
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

# fds=432



class Binarize_01__non_standard_output(torch.nn.Module):
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

    def __init__(self, big_number = 5., \
                            device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        self.base = Binarize_base(True, big_number, True, 
                                  scaling_ratio = scaling_ratio, epi = epi,
                                  div_me_when_g_too_small = div_me_when_g_too_small)
        pass


    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.base.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
    def get_output_range(self, input: BoundaryPair)->BoundaryPair:
        '''Gets the output range from inner layer'''
        return self.base.get_output_range(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    pass



# '''shape doesn't matter.'''
# layer = Binarize_01__non_standard_output(1.)
# input:torch.Tensor = torch.tensor([-2., 0., 0.25, 0.5, 0.75, 1., 3.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "the middle one should be 0.5")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")
# print(layer.get_output_range(BoundaryPair.from_float(-2., 3.)))

# '''This part tests the output_when_ambiguous. 
# It means, when the input it 0.5, which doesn't indicate towards any direction. 
# The default output is 0.5. But it's customizable.'''
# layer = Binarize_01__non_standard_output(1.)
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

# fds=432




class Binarize_01_to_np__non_standard_output(torch.nn.Module):
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

    def __init__(self, big_number = 5., \
                            device=None, dtype=None, \
                    scaling_ratio:float = 1., epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        #this layer also needs the info for gramo.
        self.base = Binarize_base(True, big_number, False, 
                                  scaling_ratio = scaling_ratio, epi = epi,
                                  div_me_when_g_too_small = div_me_when_g_too_small)
        pass


    def set_output_when_ambiguous(self, output:float):
        '''Simply set it to the inner layer.'''
        self.base.set_output_when_ambiguous(output)
        pass

    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.base.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass

    def set_scaling_ratio(self, scaling_ratio:float)->None:
        '''This function set the param for the inner "Mirror with GRadient MOdification" layer.
        '''
        self.base.gramo.set_scaling_ratio(scaling_ratio)
        pass
    
    def get_output_range(self, input: BoundaryPair)->BoundaryPair:
        '''Gets the output range from inner layer'''
        return self.base.get_output_range(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if you want to modify this function, copy the forward function from the base class.
        return self.base(input)

    pass



# layer = Binarize_01_to_np__non_standard_output(3.)
# input:torch.Tensor = torch.tensor([0., 0.25, 0.5, 0.75, 1.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# # print( input)
# output = layer(input)
# print(output, "the middle one should be 0.")
# input = input.view(-1, 1)
# output = layer(input)
# print(output, "shape doesn't matter.")
# print(layer.get_output_range(BoundaryPair.make_01()))

# '''This part tests the output_when_ambiguous. 
# It means, when the input it 0.5, which doesn't indicate towards any direction. 
# The default output is 0.. But it's customizable.'''
# layer = Binarize_01_to_np__non_standard_output(1.)
# input = torch.tensor([[0.5]], requires_grad=True)
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
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), "should be -1., 0.25, 1.")

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
# class example_Binarize_np_to_01_3times(torch.nn.Module):
#     r"""This example layer shows how to use the binarize layer.
#     """
#     def __init__(self, I_know_input_range_is_not_standard_and_here_it_is: BoundaryPair = BoundaryPair.make_np(), \
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
        
#         self.input_range = I_know_input_range_is_not_standard_and_here_it_is
        
#         #self.optionalGramo = GradientModification()
#         self.Binarize1 = Binarize_np_to_01__non_standard_output(3.)
#         self.Binarize2 = Binarize_01__non_standard_output(6.)
#         self.Binarize3 = Binarize_01__non_standard_output(6.)
#         pass
    
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
        
#         # gramo doesn't affect this.
        
#         r = self.input_range
#         x = input
#         #x = self.optionalGramo(x)
#         r = self.Binarize1.get_output_range(r)
#         x = self.Binarize1(x)
#         r = self.Binarize2.get_output_range(r)
#         x = self.Binarize2(x)
#         r = self.Binarize3.get_output_range(r)
#         x = self.Binarize3(x)
#         x = r.scale_back_to_01(x)
#         return x

#     pass

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


# layer = Binarize_np_to_01__non_standard_output(3.)
# input:torch.Tensor = torch.tensor([-1., -0.5, 0., 0.5, 1.], dtype=torch.float32, requires_grad=True,)
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

# '''not a test for math. It's testing the pytorch api.'''

# a = torch.nn.Parameter(torch.Tensor([1.,]), requires_grad=True)
# a.grad = torch.Tensor([2,])
# b = torch.nn.Parameter(torch.Tensor([3.,]), requires_grad=True)
# with torch.no_grad():
#     b.copy_(a)
# #    b.grad = a.grad
# print(a, a.grad)
# print(b, b.grad)
# a.grad = torch.Tensor([4,])
# print(a, a.grad)
# print(b, b.grad)

# a = torch.nn.Parameter(torch.Tensor([[1., 2.],[1., 2.],[1., 2.],]), requires_grad=True)
# with torch.no_grad():
#     temp = torch.Tensor(a)
#     temp = temp-1.    
#     a.grad = temp
#     #print(a, a.grad)
#     b = a.mean(dim=1,keepdim=True)
#     #print(b, b.grad)
#     c = a-b
#     #print(c)
#     temp = torch.nn.Parameter(c, requires_grad=True)
#     temp.grad = a.grad
#     #print(temp, temp.grad)
#     a = temp
# print(a, a.grad)
# fds=432



# example
# example
# example
# example
class example_Binarize_01_3times(torch.nn.Module):
    r"""This example layer shows how to use the binarize layer.
    """
    def __init__(self, I_know_input_range_is_not_standard_and_here_it_is: BoundaryPair = BoundaryPair.make_np(), \
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.input_range = I_know_input_range_is_not_standard_and_here_it_is
        
        #self.optionalGramo = GradientModification()
        self.Binarize1 = Binarize_01__non_standard_output(6.)
        self.Binarize2 = Binarize_01__non_standard_output(6.)
        self.Binarize3 = Binarize_01__non_standard_output(6.)
        pass
    
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        # gramo doesn't affect this.
        
        r = self.input_range
        x = input
        #x = self.optionalGramo(x)
        r = self.Binarize1.get_output_range(r)
        x = self.Binarize1(x)
        r = self.Binarize2.get_output_range(r)
        x = self.Binarize2(x)
        r = self.Binarize3.get_output_range(r)
        x = self.Binarize3(x)
        x = r.scale_back_to_01(x)
        return x

    pass


# model = example_Binarize_01_3times()
# input = torch.tensor([0., 0.25, 0.5, 1.], dtype=torch.float32, requires_grad=True,)
# input = input.unsqueeze(0)
# output = model(input)
# print(output, "processed with 3 layer.")
# g_in = torch.ones_like(output, dtype=torch.float32)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad from 3 layer.")
# fds=432






class DigitalMapper(torch.nn.Module):
    r'''This layer is designed to be used between digital layers. 
    The input should be in STANDARD range so to provide meaningful output
    in STANDARD range. It works for both 01 and np styles.
    
    Notice: unlike most layers in this project, this layer is stateful.
    In other words, it has inner param in neural network path.
    
    Remember to concat a constant 0. and 1. to the input before sending into this layer.
    In other words, provide Vdd and Vss as signal to the chip.
    '''
    #__constants__ = []
    
    auto_merge_duration:int
    update_count:int
    
    def __init__(self, in_features: int, out_features: int, \
                    auto_merge_duration:int = 20, raw_weight_boundary_for_f32:float = 15., \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if raw_weight_boundary_for_f32<5. :
            raise Exception("In my test, it goes to almost 6. in 4000 epochs. If you know what you are doing, comment this checking out.")

        self.in_features = in_features
        self.out_features = out_features
        self.raw_weight_boundary_for_f32 = raw_weight_boundary_for_f32
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        self.gramo1 = GradientModification()
        self.gramo1.set_scaling_ratio(100.)
        #self.gramo2 = GradientModification()

        #to keep track of the training.
        self.auto_merge_duration:int = auto_merge_duration
        self.update_count:int = 0

    def reset_parameters(self) -> None:
        '''copied from torch.nn.Linear'''
        torch.nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        
    def forward(self, input:torch.Tensor)->torch.Tensor:
        if self.training:
            if self.update_count>=self.auto_merge_duration:
                self.update_count = 0
                with torch.no_grad():
                    boundary = self.raw_weight_boundary_for_f32
                    if self.raw_weight.dtype == torch.float64:
                        boundary *= 2.
                        pass
                    if self.raw_weight.dtype == torch.float16:
                        boundary *= 0.5
                        pass
                    
                    flag = self.raw_weight.gt(boundary)
                    temp = flag*boundary
                    temp = temp + self.raw_weight.data*(flag.logical_not())
                    self.raw_weight.data = temp

                    boundary*=-1.
                    flag = self.raw_weight.lt(boundary)
                    temp = flag*boundary
                    temp = temp + self.raw_weight.data*(flag.logical_not())
                    self.raw_weight.data = temp
                    
                    mean = self.raw_weight.mean(dim=1,keepdim=True)
                    self.raw_weight.data = self.raw_weight.data-mean
                    pass
                pass
            else:
                self.update_count+=1
            
        # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")

            x = input.unsqueeze(dim=2)
            
            w = self.gramo1(self.raw_weight)
            #w = self.raw_weight
            
            w_after_softmax = w.softmax(dim=1)
            #w_after_softmax = self.gramo2(w_after_softmax)
            x = w_after_softmax.matmul(x)
            
            #print(x.shape)
            x = x.squeeze(dim = 2)
            #print(x.shape)
            
            return x
            
        else:#eval mode.
            with torch.inference_mode():
                if len(input.shape)!=2:
                    raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")
                # The same as training path. Maybe I should merge them.
                x = input.unsqueeze(dim=2)
                w_after_softmax = self.raw_weight.softmax(dim=1)
                x = w_after_softmax.matmul(x)
                x = x.squeeze(dim = 2)
                return x          

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
 
    def convert_into_eval_only_mode(self):
        raise Exception("Not implemented yet. This feature only helps in deployment cases, but nobody cares about my project. It's never gonne be deployed.")
    
    pass




# layer = DigitalMapper(2,3)
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# input = input.requires_grad_()
# #print(input.softmax(dim=1))#dim = 1!!!
# print(layer(input))
# print(layer.raw_weight.softmax(dim=1))

# layer = DigitalMapper(2,3)
# layer.eval()
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# print(layer(input))
# print(layer.raw_weight.softmax(dim=1))

# fds=432

# in_feature = 2
# model = DigitalMapper(in_feature,1)
# loss_function = torch.nn.MSELoss()
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# #input = torch.Tensor([[1., 0.],[0., 1.]])
# input = input.requires_grad_()
# target = torch.Tensor([[1.],[1.],[0.],[0.],])
# #target = torch.Tensor([[1.],[0.],])
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
# # for p in model.parameters():
# #     print(p)

# iter_per_print = 1111
# print_count = 3
# for epoch in range(iter_per_print*print_count):
    
#     model.train()
#     pred = model(input)
    
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(pred, "pred")
#     #     print(target, "target")
    
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight.grad, "grad")
        
#     #model.raw_weight.grad = model.raw_weight.grad*-1.
#     #optimizer.param_groups[0]["lr"] = 0.01
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, model.raw_weight.grad, "before update")
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
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# print(model(input), "should be ([[1.],[1.],[0.],[0.],])")

# fds=432
    


# class clamp_test(torch.nn.Module):
    
#     def __init__(self, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()

#         self.raw_weight = torch.nn.Parameter(torch.Tensor([[50., 10., -50., ]]))
#         self.raw_weight.requires_grad_()
#         self.raw_weight.grad = torch.Tensor([[55., 11., -55., ]])

#         boundary = 30.
#         if self.raw_weight.dtype == torch.float32:
#             boundary = 15.
#             pass
#         if self.raw_weight.dtype == torch.float16:
#             boundary = 8.
#             pass
        
#         print(self.raw_weight, self.raw_weight.grad)
        
#         flag = self.raw_weight.gt(boundary)
#         print(flag)
#         temp = flag*boundary
#         print(temp)
#         temp = temp + self.raw_weight.data*(flag.logical_not())
#         print(temp)
#         self.raw_weight.data = temp
#         print(self.raw_weight, self.raw_weight.grad)

#         boundary*=-1.
#         flag = self.raw_weight.lt(boundary)
#         print(temp)
#         temp = flag*boundary
#         print(temp)
#         temp = temp + self.raw_weight.data*(flag.logical_not())
#         print(temp)
#         self.raw_weight.data = temp
#         print(self.raw_weight, self.raw_weight.grad)

#         fds=432

# a = clamp_test()





































#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。
#范围是错的。。





class AND_01(torch.nn.Module):
    r""" 
    AND gate. 
    Only accepts STANDARD 01 range.
    Output is customizable. But if it's not standard, is it useful ?
    
    Input shape: [batch, gate_count*input_count]
    
    Output shape: [batch, gate_count*input_count*(1 or 2)](depends on what you need)
    
    Usually, reshape the output into [batch, all the output], 
    concat it with output of other gates, and a 0 and an 1.
    
    
    unfinished docs.
    
    
    
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
    
    def __init__(self, first_big_number:float, \
                input_per_gate:int = 2, \
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
        
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01__non_standard_output(1.)
        self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
        '''The only param is the big_number. 
        Bigger big_number leads to more binarization and slightly bigger result range.
        More layers leads to more binarization.
        '''
        # you may also needs some more Binarize layers.
        #self.Binarize2 = Binarize_01(5.)
        #self.Binarize3 = Binarize_01(5.)
        pass
    
    # if you customize the layer and make the output in a non standard range, you should provide this function.
    # def get_output_range(self, input: torch.Tensor)->BoundaryPair:
    #     '''Make sure all the inputs are in the (0., 1.) range.
    #     Otherwise this function gives out wrong result.
    #     Use 
    #     >>> Boundary.scale_back_to_...(tensor with wrong range)'''
    #     #this function consists of 2 parts, the first part is fixed.
    #     lower = float(input.shape[1])*-1.+1.
    #     upper = 1.
    #     r = BoundaryPair.from_float(lower, upper)
        
    #     #this part is part 2. Make sure it's the same as the forward function.
    #     r = self.Binarize1.get_output_range(r)
    #     # r = self.Binarize2.get_output_range(r)
    #     # r = self.Binarize3.get_output_range(r)
        
    #     return r
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            x = x.sum(dim=2, keepdim=False)#dim=2
            #back to rank-2
            
            offset = float(self.input_per_gate)*-1.+1.
            x = x + offset
            
            # binarize 
            r = BoundaryPair.from_float(offset, 1.)
            r = self.Binarize1.get_output_range(r)
            x = self.Binarize1(x)
            
            # you may also needs some more Binarize layers.
            # r = self.Binarize2.get_output_range(r)
            # x = self.Binarize2(x)
            # r = self.Binarize3.get_output_range(r)
            # x = self.Binarize3(x)

            x = r.scale_back_to_01(x)

            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = 1.-x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite])
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.all(dim=1, keepdim=True)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x.to(input.dtype)
                else:
                    opposite = x.logical_not()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite]).to(input.dtype)
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
        
        result = f'AND/NAND layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass


# old test.
# a = torch.tensor([[0.1],[0.9],[0.9],], requires_grad=True)
# bound_of_a = BoundaryPair.from_float(0.1, 0.9)
# a = bound_of_a.scale_back_to_01(a)
# b = torch.tensor([[0.2],[0.2],[0.8],], requires_grad=True)
# bound_of_b = BoundaryPair.from_float(0.2, 0.8)
# b = bound_of_b.scale_back_to_01(b)
# input = torch.concat((a,b), 1)# dim = 1 !!!
# layer = AND_01(5.)
# print(layer(input), "should be 0.0_, 0.0_, 0.9_")

# '''These 2 tests show how to protect the gradient.
# Basically the big_number of the first binarize layer is the key.
# By making it smaller, you make the gradient of inputs closer over each elements.
# For AND gate, if it gets a lot false as input, the corresponding gradient is very small.
# '''
# '''input is not standard range, the output should also be a bit non standard.'''
# a = torch.tensor([[0.001],[0.992],[0.993],], requires_grad=True)
# b = torch.tensor([[0.004],[0.005],[0.996],], requires_grad=True)
# input = torch.concat((a,b,b,b), dim=1) 
# layer = AND_01(1., input_per_gate=4)
# # layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
# result = layer(input)
# print(result, "should be <0.5, <0.5, >0.5")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad, should be 3x of a's")
#more~
# '''If the input of a gate is too many, then, the greatest gradient is a lot times
# bigger than the smallest one. If the greatest-grad element needs n epochs to train,
# and its gradient is 100x bigger than the smallest, then the least-grad element needs 
# 100n epochs to train. This may harm the trainability of the least-grad element by a lot.
# Basically, the gates layers are designed to use as AND2, AND3, but never AND42 or AND69420.'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,b,b,b,b,b,b,b,b,b), dim=1) 
# layer = AND_01(0.5, input_per_gate=11)
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
# layer = AND_01(5., input_per_gate=3)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 0., 1.")

'''output mode'''
a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
input = torch.concat((a,b), 1) 
layer = AND_01(5.)
print(layer(input), "should be 0.0_, 0.0_, 0.9_")
layer.eval()
print(layer(input), "should be 0., 0., 1.")
layer = AND_01(5., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
print(layer(input), "should be 0.0_, 0.0_, 0.9_, 0.9_, 0.9_, 0.0_")
layer.eval()
print(layer(input), "should be 0., 0., 1., 1., 1., 0.")
layer = AND_01(5., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
print(layer(input), "should be 0.9_, 0.9_, 0.0_")
layer.eval()
print(layer(input), "should be 1., 1., 0.")

layer = AND_01(1.)
print(layer)

fds=432


raise Exception("继续")
































class OR_01(torch.nn.Module):
    r""" 
    OR gate. Takes any number of inputs. Range is (0., 1.).
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
    >>> layer = OR_01()
    >>> result = layer(input)
    
    The evaluating mode is a bit different. It's a simply comparison.
    It relies on the threshold. If the input doesn't use 0.5 as threshold, offset it.
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],]) # requires_grad doesn't matter in this case.
    >>> b = torch.tensor([[0.2],[0.2],[0.8],])
    >>> c = torch.tensor([[0.3],[0.3],[0.7],])
    >>> input = torch.concat((a,b,c), 1) 
    >>> layer = OR_01()
    >>> layer.eval()
    >>> result = layer(input)
    
    These 2 tests show how to protect the gradient.
    Basically the big_number of the first binarize layer is the key.
    By making it smaller, you make the gradient of inputs closer over each elements.
    For OR gate, if it gets a lot True as input, the corresponding gradient is very small.

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
    >>> layer = OR_01()
    >>> result = layer(input)
    >>> print(result)
    >>> print(layer.get_output_range(2), "the output range. Should be the same as the result.")

    If the performance doesn't meet your requirement, 
    modify the binarize layers. More explaination in the source code.
    """
    
    def __init__(self, first_big_number:float, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
            
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01__non_standard_output(1.)
        self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
        '''The only param is the big_number. 
        Bigger big_number leads to more binarization and slightly bigger result range.
        More layers leads to more binarization.
        '''
        # you may also needs some more Binarize layers.
        #self.Binarize2 = Binarize_01(5.)
        #self.Binarize3 = Binarize_01(5.)
        pass

    
    def get_output_range(self, input: torch.Tensor)->BoundaryPair:
        '''Make sure all the inputs are in the (0., 1.) range.
        Otherwise this function gives out wrong result.
        Use 
        >>> Boundary.scale_back_to_...(tensor with wrong range)'''
        #this function consists of 2 parts, the first part is fixed.
        lower = 0.
        upper = float(input.shape[1])
        r = BoundaryPair.from_float(lower, upper)
        
        #this part is part 2. Make sure it's the same as the forward function.
        r = self.Binarize1.get_output_range(r)
        # r = self.Binarize2.get_output_range(r)
        # r = self.Binarize3.get_output_range(r)
        
        return r
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, all the inputs]")
            
            # no offset is needed for OR
            x = input.sum(dim=1, keepdim=True)
            
            x = self.Binarize1(x)
            
            # you may also needs some more Binarize layers.
            # x = self.Binarize2(x)
            # x = self.Binarize3(x)

            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return torch.concat([x,opposite])
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return opposite
            raise Exception("unreachable code.")

            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.any(dim=1, keepdim=True)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    x = x.to(input.dtype)
                    return x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return torch.concat([x,opposite]).to(input.dtype)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return opposite.to(input.dtype)
                raise Exception("unreachable code.")
        #end of function
    
    def extra_repr(self) -> str:
        output_mode = "Self only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both self and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'OR/NOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass




# '''forward path and grad.'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = OR_01(5.)
# result = layer(input)
# print(result, "should be 0.0_, 0.9_, 0.9_")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")

# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = OR_01(5.)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 1., 1.")

# '''test for output range'''
# layer = OR_01(1.5)
# a = torch.tensor([[0.],[1.],], requires_grad=True)
# input = torch.concat((a,a), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")
# input = torch.concat((a,a,a), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = OR_01(5.)
# print(layer(input), "should be 0.0_, 0.9_, 0.9_")
# layer.eval()
# print(layer(input), "should be 0., 1., 1.")
# layer = OR_01(5., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0.0_, 0.9_, 0.9_, 0.9_, 0.0_, 0.0_")
# layer.eval()
# print(layer(input), "should be 0., 1., 1., 1., 0., 0.")
# layer = OR_01(5., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 0.9_, 0.0_, 0.0_")
# layer.eval()
# print(layer(input), "should be 1., 0., 0.")

# layer = OR_01(1.)
# print(layer)

# fds=432





class XOR_01(torch.nn.Module):
    r""" 
    OR gate. Takes any number of inputs. Range is (0., 1.).
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
    >>> layer = OR_01()
    >>> result = layer(input)
    
    The evaluating mode is a bit different. It's a simply comparison.
    It relies on the threshold. If the input doesn't use 0.5 as threshold, offset it.
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],]) # requires_grad doesn't matter in this case.
    >>> b = torch.tensor([[0.2],[0.2],[0.8],])
    >>> c = torch.tensor([[0.3],[0.3],[0.7],])
    >>> input = torch.concat((a,b,c), 1) 
    >>> layer = OR_01()
    >>> layer.eval()
    >>> result = layer(input)
    
    These 2 tests show how to protect the gradient.
    Basically the big_number of the first binarize layer is the key.
    By making it smaller, you make the gradient of inputs closer over each elements.
    For OR gate, if it gets a lot True as input, the corresponding gradient is very small.

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
    >>> layer = OR_01()
    >>> result = layer(input)
    >>> print(result)
    >>> print(layer.get_output_range(2), "the output range. Should be the same as the result.")

    If the performance doesn't meet your requirement, 
    modify the binarize layers. More explaination in the source code.
    """
    
    def __init__(self, first_big_number:float, \
                 output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
            
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01__non_standard_output(1.)
        self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
        '''The only param is the big_number. 
        Bigger big_number leads to more binarization and slightly bigger result range.
        More layers leads to more binarization.
        '''
        # you may also needs some more Binarize layers.
        #self.Binarize2 = Binarize_01(5.)
        #self.Binarize3 = Binarize_01(5.)
        pass

    
    def get_output_range(self, input: torch.Tensor)->BoundaryPair:
        '''Make sure all the inputs are in the (0., 1.) range.
        Otherwise this function gives out wrong result.
        Use 
        >>> Boundary.scale_back_to_...(tensor with wrong range)'''
        #this function consists of 2 parts, the first part is fixed.
        
        upper = torch.pow(torch.tensor([0.5]),torch.tensor(input.shape[1], dtype = torch.float32))
        lower = -1.*upper
        r = BoundaryPair(lower+0.5, upper+0.5)
        
        #this part is part 2. Make sure it's the same as the forward function.
        r = self.Binarize1.get_output_range(r)
        # r = self.Binarize2.get_output_range(r)
        # r = self.Binarize3.get_output_range(r)
        
        return r
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, all the inputs]")
            
            x = input
            # no offset is needed for OR
            x = x-0.5
            x = x.prod(dim=1, keepdim=True)
            x = -x+0.5
            
            x = self.Binarize1(x)
            
            # you may also needs some more Binarize layers.
            # x = self.Binarize2(x)
            # x = self.Binarize3(x)

            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return torch.concat([x,opposite])
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return opposite
            raise Exception("unreachable code.")

            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.to(torch.int8)
                #overflow doesn't affect the result. 
                x = x.sum(dim=1, keepdim=True)
                x = x%2
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    x = x.to(input.dtype)
                    return x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return torch.concat([x,opposite]).to(input.dtype)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return opposite.to(input.dtype)
                raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Self only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both self and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'XOR/NXOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
    pass




# '''forward path and grad.'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = XOR_01(10.)
# result = layer(input)
# print(result, "should be 0.0_, 0.9_, 0.0_")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")

# a = torch.tensor([[0.],[1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],[1.],], requires_grad=True)
# c = torch.tensor([[0.],[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,c), 1) 
# layer = XOR_01(25.)
# result = layer(input)
# print(result, "should be 0.0_, 0.9_, 0.0_, 0.9_")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b,c])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# print(b.grad, "c's grad")

# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.3],[0.7],])
# d = torch.tensor([[0.4],[0.4],[0.6],[0.6],])
# input = torch.concat((a,b,c,d), 1) 
# layer = XOR_01(5.)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 1., 1., 0.")

# '''test for output range'''
# layer = XOR_01(4.)
# a = torch.tensor([[0.],[1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],[1.],], requires_grad=True)
# c = torch.tensor([[0.],[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")
# input = torch.concat((a,b,c), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = XOR_01(10.)
# print(layer(input), "should be 0.0_, 0.9_, 0.0_")
# layer.eval()
# print(layer(input), "should be 0., 1., 0.")
# layer = XOR_01(10., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0.0_, 0.9_, 0.0_, 0.9_, 0.0_, 0.9_")
# layer.eval()
# print(layer(input), "should be 0., 1., 0., 1., 0., 1.")
# layer = XOR_01(10., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 0.9_, 0.0_, 0.9_")
# layer.eval()
# print(layer(input), "should be 1., 0., 1.")

# layer = XOR_01(1.)
# print(layer)

# fds=432














#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。
#前面门层的shape弄错了。v

# untested!!!!!
class exampleDigitalSignalProcessor_01(torch.nn.Module):
    r'''This example shows how to handle pure digital signal.
    Pre binarize the input into (0., 1.).
    '''
    #__constants__ = ['',]

    def __init__(self, in_features: int, out_features: int, gate_count: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gate_count = gate_count
        
        self.mapper1 = DigitalMapper(in_features, gate_count*6)
                
        self.and1 = AND_01(3.,output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
        self.or1 = OR_01(3.,output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
        self.xor1 = XOR_01(5.,output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
        
        mapper_in_features = gate_count*6+2
        self.mapper2 = DigitalMapper(mapper_in_features, out_features)
        
        #end of function        
        
        
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''This example only shows the training path.'''
        x = input
        r = BoundaryPair.make_01()
        
        #mapper doesn't affect range. r is still 01
        x = self.mapper1(x)
        
        r_and_or = self.and1.get_output_range(r)
        r_xor = self.xor1.get_output_range(r)
        and_head = r_and_or.scale_back_to_01(self.and1(x))
        or_head = r_and_or.scale_back_to_01(self.or1(x))                
        xor_head = r_xor.scale_back_to_01(self.xor1(x))
        zeros = torch.zeros([input.shape[0],1])
        ones = torch.zeros([input.shape[0],1])
        concat = torch.concat([and_head, or_head, xor_head, zeros, ones],dim=1)
        
        x = self.mapper2(x)
        
        return x
    #end of function
    
    pass
        
        


































class AND_np(torch.nn.Module):
    r""" 
    AND gate. Takes any number of inputs. Range is (-1., 1.).
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
    
    def __init__(self, first_big_number:float, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_np__non_standard_output(1.)
        self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
        '''The only param is the big_number. 
        Bigger big_number leads to more binarization and slightly bigger result range.
        More layers leads to more binarization.
        '''
        # you may also needs some more Binarize layers.
        #self.Binarize2 = Binarize_np(5.)
        #self.Binarize3 = Binarize_np(5.)
        pass
    
    def get_output_range(self, input: torch.Tensor)->BoundaryPair:
        '''Make sure all the inputs are in the (0., 1.) range.
        Otherwise this function gives out wrong result.
        Use 
        >>> Boundary.scale_back_to_...(tensor with wrong range)'''
        #this function consists of 2 parts, the first part is fixed.
        lower = float(input.shape[1])*-2.+1.
        upper = 1.
        r = BoundaryPair.from_float(lower, upper)
        
        #this part is part 2. Make sure it's the same as the forward function.
        r = self.Binarize1.get_output_range(r)
        # r = self.Binarize2.get_output_range(r)
        # r = self.Binarize3.get_output_range(r)
        
        return r
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, all the inputs]")
            
            raise Exception("继续。")
            offset = float(input.shape[1])*-1.+1.
            x = input.sum(dim=1, keepdim=True)
            x = x + offset
            x = self.Binarize1(x)
            
            # you may also needs some more Binarize layers.
            # x = self.Binarize2(x)
            # x = self.Binarize3(x)

            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return torch.concat([x,opposite])
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return opposite
            raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.all(dim=1, keepdim=True)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    x = x.to(input.dtype)
                    return x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return torch.concat([x,opposite]).to(input.dtype)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return opposite.to(input.dtype)
                raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Self only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both self and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'AND/NAND layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass



# a = torch.tensor([[0.1],[0.9],[0.9],], requires_grad=True)
# bound_of_a = BoundaryPair.from_float(0.1, 0.9)
# a = bound_of_a.scale_back_to_01(a)
# b = torch.tensor([[0.2],[0.2],[0.8],], requires_grad=True)
# bound_of_b = BoundaryPair.from_float(0.2, 0.8)
# b = bound_of_b.scale_back_to_01(b)
# input = torch.concat((a,b), 1)# dim = 1 !!!
# layer = AND_01(5.)
# print(layer(input), "should be 0.0_, 0.0_, 0.9_")
# print(layer.get_output_range(input))

# '''These 2 tests show how to protect the gradient.
# Basically the big_number of the first binarize layer is the key.
# By making it smaller, you make the gradient of inputs closer over each elements.
# For AND gate, if it gets a lot false as input, the corresponding gradient is very small.
# '''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,b), 1) 
# layer = AND_01(1.)
# # layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
# result = layer(input)
# print(result, "should be 0.0_, 0.0_, 0.9_")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad, should be 2x of a's")
# #more~
# '''If the input of a gate is too many, then, the greatest gradient is a lot times
# bigger than the smallest one. If the greatest-grad element needs n epochs to train,
# and its gradient is 100x bigger than the smallest, then the least-grad element needs 
# 100n epochs to train. This may harm the trainability of the least-grad element by a lot.
# Basically, the gates layers are designed to use as AND2, AND3, but never AND42 or AND69420.'''
# input = torch.concat((a,b,b,b,b,b,b,b,b,b,b), 1) 
# layer = AND_01(0.5)
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
# layer = AND_01(5.)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 0., 1.")

# '''test for output range'''
# layer = AND_01(1.5)
# a = torch.tensor([[0.],[1.],], requires_grad=True)
# input = torch.concat((a,a), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")
# input = torch.concat((a,a,a), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = AND_01(5.)
# print(layer(input), "should be 0.0_, 0.0_, 0.9_")
# layer.eval()
# print(layer(input), "should be 0., 0., 1.")
# layer = AND_01(5., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0.0_, 0.0_, 0.9_, 0.9_, 0.9_, 0.0_")
# layer.eval()
# print(layer(input), "should be 0., 0., 1., 1., 1., 0.")
# layer = AND_01(5., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 0.9_, 0.9_, 0.0_")
# layer.eval()
# print(layer(input), "should be 1., 1., 0.")

# layer = AND_01(1.)
# print(layer)

# fds=432





class OR_01____________(torch.nn.Module):
    r""" 
    OR gate. Takes any number of inputs. Range is (0., 1.).
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
    >>> layer = OR_01()
    >>> result = layer(input)
    
    The evaluating mode is a bit different. It's a simply comparison.
    It relies on the threshold. If the input doesn't use 0.5 as threshold, offset it.
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],]) # requires_grad doesn't matter in this case.
    >>> b = torch.tensor([[0.2],[0.2],[0.8],])
    >>> c = torch.tensor([[0.3],[0.3],[0.7],])
    >>> input = torch.concat((a,b,c), 1) 
    >>> layer = OR_01()
    >>> layer.eval()
    >>> result = layer(input)
    
    These 2 tests show how to protect the gradient.
    Basically the big_number of the first binarize layer is the key.
    By making it smaller, you make the gradient of inputs closer over each elements.
    For OR gate, if it gets a lot True as input, the corresponding gradient is very small.

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
    >>> layer = OR_01()
    >>> result = layer(input)
    >>> print(result)
    >>> print(layer.get_output_range(2), "the output range. Should be the same as the result.")

    If the performance doesn't meet your requirement, 
    modify the binarize layers. More explaination in the source code.
    """
    
    def __init__(self, first_big_number:float, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
            
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01__non_standard_output(1.)
        self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
        '''The only param is the big_number. 
        Bigger big_number leads to more binarization and slightly bigger result range.
        More layers leads to more binarization.
        '''
        # you may also needs some more Binarize layers.
        #self.Binarize2 = Binarize_01(5.)
        #self.Binarize3 = Binarize_01(5.)
        pass

    
    def get_output_range(self, input: torch.Tensor)->BoundaryPair:
        '''Make sure all the inputs are in the (0., 1.) range.
        Otherwise this function gives out wrong result.
        Use 
        >>> Boundary.scale_back_to_...(tensor with wrong range)'''
        #this function consists of 2 parts, the first part is fixed.
        lower = 0.
        upper = float(input.shape[1])
        r = BoundaryPair.from_float(lower, upper)
        
        #this part is part 2. Make sure it's the same as the forward function.
        r = self.Binarize1.get_output_range(r)
        # r = self.Binarize2.get_output_range(r)
        # r = self.Binarize3.get_output_range(r)
        
        return r
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, all the inputs]")
            
            # no offset is needed for OR
            x = input.sum(dim=1, keepdim=True)
            
            x = self.Binarize1(x)
            
            # you may also needs some more Binarize layers.
            # x = self.Binarize2(x)
            # x = self.Binarize3(x)

            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return torch.concat([x,opposite])
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return opposite
            raise Exception("unreachable code.")

            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.any(dim=1, keepdim=True)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    x = x.to(input.dtype)
                    return x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return torch.concat([x,opposite]).to(input.dtype)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return opposite.to(input.dtype)
                raise Exception("unreachable code.")
        #end of function
    
    def extra_repr(self) -> str:
        output_mode = "Self only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both self and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'OR/NOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass




# '''forward path and grad.'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = OR_01(5.)
# result = layer(input)
# print(result, "should be 0.0_, 0.9_, 0.9_")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")

# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = OR_01(5.)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 1., 1.")

# '''test for output range'''
# layer = OR_01(1.5)
# a = torch.tensor([[0.],[1.],], requires_grad=True)
# input = torch.concat((a,a), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")
# input = torch.concat((a,a,a), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = OR_01(5.)
# print(layer(input), "should be 0.0_, 0.9_, 0.9_")
# layer.eval()
# print(layer(input), "should be 0., 1., 1.")
# layer = OR_01(5., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0.0_, 0.9_, 0.9_, 0.9_, 0.0_, 0.0_")
# layer.eval()
# print(layer(input), "should be 0., 1., 1., 1., 0., 0.")
# layer = OR_01(5., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 0.9_, 0.0_, 0.0_")
# layer.eval()
# print(layer(input), "should be 1., 0., 0.")

# layer = OR_01(1.)
# print(layer)

# fds=432





class XOR_01_________________(torch.nn.Module):
    r""" 
    OR gate. Takes any number of inputs. Range is (0., 1.).
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
    >>> layer = OR_01()
    >>> result = layer(input)
    
    The evaluating mode is a bit different. It's a simply comparison.
    It relies on the threshold. If the input doesn't use 0.5 as threshold, offset it.
    
    >>> a = torch.tensor([[0.1],[0.9],[0.9],]) # requires_grad doesn't matter in this case.
    >>> b = torch.tensor([[0.2],[0.2],[0.8],])
    >>> c = torch.tensor([[0.3],[0.3],[0.7],])
    >>> input = torch.concat((a,b,c), 1) 
    >>> layer = OR_01()
    >>> layer.eval()
    >>> result = layer(input)
    
    These 2 tests show how to protect the gradient.
    Basically the big_number of the first binarize layer is the key.
    By making it smaller, you make the gradient of inputs closer over each elements.
    For OR gate, if it gets a lot True as input, the corresponding gradient is very small.

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
    >>> layer = OR_01()
    >>> result = layer(input)
    >>> print(result)
    >>> print(layer.get_output_range(2), "the output range. Should be the same as the result.")

    If the performance doesn't meet your requirement, 
    modify the binarize layers. More explaination in the source code.
    """
    
    def __init__(self, first_big_number:float, \
                 output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
            
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01__non_standard_output(1.)
        self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
        '''The only param is the big_number. 
        Bigger big_number leads to more binarization and slightly bigger result range.
        More layers leads to more binarization.
        '''
        # you may also needs some more Binarize layers.
        #self.Binarize2 = Binarize_01(5.)
        #self.Binarize3 = Binarize_01(5.)
        pass

    
    def get_output_range(self, input: torch.Tensor)->BoundaryPair:
        '''Make sure all the inputs are in the (0., 1.) range.
        Otherwise this function gives out wrong result.
        Use 
        >>> Boundary.scale_back_to_...(tensor with wrong range)'''
        #this function consists of 2 parts, the first part is fixed.
        
        upper = torch.pow(torch.tensor([0.5]),torch.tensor(input.shape[1], dtype = torch.float32))
        lower = -1.*upper
        r = BoundaryPair(lower+0.5, upper+0.5)
        
        #this part is part 2. Make sure it's the same as the forward function.
        r = self.Binarize1.get_output_range(r)
        # r = self.Binarize2.get_output_range(r)
        # r = self.Binarize3.get_output_range(r)
        
        return r
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, all the inputs]")
            
            x = input
            # no offset is needed for OR
            x = x-0.5
            x = x.prod(dim=1, keepdim=True)
            x = -x+0.5
            
            x = self.Binarize1(x)
            
            # you may also needs some more Binarize layers.
            # x = self.Binarize2(x)
            # x = self.Binarize3(x)

            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return torch.concat([x,opposite])
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return opposite
            raise Exception("unreachable code.")

            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.to(torch.int8)
                #overflow doesn't affect the result. 
                x = x.sum(dim=1, keepdim=True)
                x = x%2
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    x = x.to(input.dtype)
                    return x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return torch.concat([x,opposite]).to(input.dtype)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return opposite.to(input.dtype)
                raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Self only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both self and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'XOR/NXOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
    pass




# '''forward path and grad.'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = XOR_01(10.)
# result = layer(input)
# print(result, "should be 0.0_, 0.9_, 0.0_")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")

# a = torch.tensor([[0.],[1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],[1.],], requires_grad=True)
# c = torch.tensor([[0.],[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,c), 1) 
# layer = XOR_01(25.)
# result = layer(input)
# print(result, "should be 0.0_, 0.9_, 0.0_, 0.9_")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b,c])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# print(b.grad, "c's grad")

# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.3],[0.7],])
# d = torch.tensor([[0.4],[0.4],[0.6],[0.6],])
# input = torch.concat((a,b,c,d), 1) 
# layer = XOR_01(5.)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 1., 1., 0.")

# '''test for output range'''
# layer = XOR_01(4.)
# a = torch.tensor([[0.],[1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],[1.],], requires_grad=True)
# c = torch.tensor([[0.],[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")
# input = torch.concat((a,b,c), 1) 
# print(layer(input))
# print(layer.get_output_range(input), "the output range. Should be the same as the result.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = XOR_01(10.)
# print(layer(input), "should be 0.0_, 0.9_, 0.0_")
# layer.eval()
# print(layer(input), "should be 0., 1., 0.")
# layer = XOR_01(10., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0.0_, 0.9_, 0.0_, 0.9_, 0.0_, 0.9_")
# layer.eval()
# print(layer(input), "should be 0., 1., 0., 1., 0., 1.")
# layer = XOR_01(10., output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 0.9_, 0.0_, 0.9_")
# layer.eval()
# print(layer(input), "should be 1., 0., 1.")

# layer = XOR_01(1.)
# print(layer)

# fds=432









































































'''
to do list
output mode
big number?
'''
#不慌。。还没开始。。
class ADC(torch.nn.Module):
    r""" """
    def __init__(self, first_big_number:float, \
                 output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
            
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01__non_standard_output(1.)
        self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
        '''The only param is the big_number. 
        Bigger big_number leads to more binarization and slightly bigger result range.
        More layers leads to more binarization.
        '''
        # you may also needs some more Binarize layers.
        #self.Binarize2 = Binarize_01(5.)
        #self.Binarize3 = Binarize_01(5.)
        pass

    
    def get_output_range(self, input: torch.Tensor)->BoundaryPair:
        '''Make sure all the inputs are in the (0., 1.) range.
        Otherwise this function gives out wrong result.
        Use 
        >>> Boundary.scale_back_to_...(tensor with wrong range)'''
        #this function consists of 2 parts, the first part is fixed.
        
        upper = torch.pow(torch.tensor([0.5]),torch.tensor(input.shape[1], dtype = torch.float32))
        lower = -1.*upper
        r = BoundaryPair(lower+0.5, upper+0.5)
        
        #this part is part 2. Make sure it's the same as the forward function.
        r = self.Binarize1.get_output_range(r)
        # r = self.Binarize2.get_output_range(r)
        # r = self.Binarize3.get_output_range(r)
        
        return r
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, all the inputs]")
            
            x = input
            # no offset is needed for OR
            x = x-0.5
            x = x.prod(dim=1, keepdim=True)
            x = -x+0.5
            
            x = self.Binarize1(x)
            
            # you may also needs some more Binarize layers.
            # x = self.Binarize2(x)
            # x = self.Binarize3(x)

            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return torch.concat([x,opposite])
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                opposite = 1.-x
                return opposite
            raise Exception("unreachable code.")

            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.5)
                x = x.to(torch.int8)
                #overflow doesn't affect the result. 
                x = x.sum(dim=1, keepdim=True)
                x = x%2
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    x = x.to(input.dtype)
                    return x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return torch.concat([x,opposite]).to(input.dtype)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    opposite = x.logical_not()
                    return opposite.to(input.dtype)
                raise Exception("unreachable code.")
        #end of function
        
    pass
























































































































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







































































































































































































































