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






class Grad_Balancer_2out_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x:torch.Tensor = args[0]
        factor_for_path_1 = args[1]
        factor_for_path_2 = args[2]
        ctx.save_for_backward(factor_for_path_1, factor_for_path_2)
        
        x = torch.stack([x, x], dim=0)
        x = x.requires_grad_()
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        # factor_for_path_1:torch.Tensor
        # factor_for_path_2:torch.Tensor
        factor_for_path_1, factor_for_path_2 = ctx.saved_tensors
        
        return g[0]*factor_for_path_1+g[1]*factor_for_path_2, None, None

    pass  # class

# input = torch.tensor([1., 2., 3.], requires_grad=True)
# factor_for_path_1 = torch.tensor([0.1])
# factor_for_path_2 = torch.tensor([0.01])
# output = Grad_Balancer_2out_Function.apply(input, factor_for_path_1, factor_for_path_2)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# fds=432





class Grad_Balancer_2out(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It duplicates the forward path, 
    and multiplies the gradient from different backward path with a given weight.
    """
    def __init__(self, factor1:float, factor2:float, \
                    device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if factor1<=0.:
            raise Exception("Param:factor1 must > 0.")
        if factor2<=0.:
            raise Exception("Param:factor2 must > 0.")
        
        self.factor_for_path_1 = torch.Tensor([factor1])
        self.factor_for_path_2 = torch.Tensor([factor2])
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not x.requires_grad):
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return Grad_Balancer_2out_Function.apply(x, self.factor_for_path_1, self.factor_for_path_2)
    

# layer = Grad_Balancer_2out(0.1, 0.02)
# input = torch.tensor([1., 2., 3.], requires_grad=True)
# output = layer(input)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# fds=432





class Grad_Balancer_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x:torch.Tensor = args[0]
        factor = args[1]
        x = x.unsqueeze(dim=0)
        result = x
        
        for _ in range(1, len(factor)):
            result = torch.concat([result,x], dim=0)
        
        ctx.save_for_backward(factor)
        
        result = result.requires_grad_()
        return result

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        (factor,) = ctx.saved_tensors#this gives a TUPLE!!!
        g_out = torch.zeros_like(g[0])
        
        for i in range(len(factor)):
            g_out += g[i]*(factor[i].item())
            
        return g_out, None

    pass  # class

# input = torch.tensor([1., 2.], requires_grad=True)
# factor = torch.tensor([0.1, 0.02, 0.003])
# output = Grad_Balancer_Function.apply(input, factor)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# input = torch.tensor([[1., 2.], [3., 4.], ], requires_grad=True)
# factor = torch.tensor([0.1, 0.02, 0.003])
# output = Grad_Balancer_Function.apply(input, factor)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# fds=432




class Grad_Balancer(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It duplicates the forward path, 
    and multiplies the gradient from different backward path with a given weight.
    """
    def __init__(self, weight_tensor_for_grad:torch.Tensor = torch.Tensor([1., 1.]), \
                    device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if len(weight_tensor_for_grad.shape)!=1:
            raise Exception("Param:weight_tensor_for_grad should be a vector.")
        for i in range(len(weight_tensor_for_grad)):
            if weight_tensor_for_grad[i]<=0.:
                raise Exception(f'The [{i}] element in the factor tensor is <=0.. It must be >0..')
            
        self.weight_tensor_for_grad = weight_tensor_for_grad
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not x.requires_grad):
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return Grad_Balancer_Function.apply(x, self.weight_tensor_for_grad)

# factor = torch.tensor([0.1, 0.02, 0.003])
# layer = Grad_Balancer(factor)
# input = torch.tensor([1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "output")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "grad")

# fds=432





class Binarize_01_Forward_only_Function(torch.autograd.Function):
    r'''This class is not designed to be used directly.
    A critical safety check is in the wrapper class.    
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x:torch.Tensor = args[0]
        
        dtype = x.dtype
        x = x.gt(0.5)
        x = x.to(dtype)
        return x

    @staticmethod
    def backward(ctx, g):
        return g

    pass  # class

# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = Binarize_01_Forward_only_Function.apply(input)
# print(output, "should be 0., 0., 0., 1., 1.")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "all should be 1s")

# fds=432



class Binarize_01_Forward_only(torch.nn.Module):
    r"""This is a wrapper class. It helps you use the inner functional properly.
    
    It clamp the forward inter layer data, while doesn't touch the backward propagation.
    """
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        if self.training and (not x.requires_grad):
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

        #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return Binarize_01_Forward_only_Function.apply(x)

# layer = Binarize_01_Forward_only()
# input = torch.tensor([-1., 0., 0.5, 1., 2.], requires_grad=True)
# output = layer(input)
# print(output, "should be 0., 0., 0., 1., 1.")
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

    def __init__(self, input_is_01:bool, big_number:float, \
                    device=None, dtype=None, \
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
        self.set_big_number(big_number)
        
        self.output_when_ambiguous = 0.5

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

    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return False
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

    def __get_offset(self)->float:
        if self.input_is_01:
            return -0.5
        else:
            return 0.
            pass
        
        #untested?
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
            
            result = self.gramo(torch.sigmoid(x))
            return result
        
        else:#eval mode
            '''I know the bit wise operations are much faster. 
            But for now I'll keep it f32 for simplisity.
            '''
            with torch.inference_mode():
                output_dtype = input.dtype
                    
                '''The optimization:
                If output is 01, and output_when_ambiguous is 0., only > is 1.
                If output is 01, and output_when_ambiguous is 1., only >= is 1.
                If output is 01, and output_when_ambiguous is not 0. or 1., it needs 2 parts, the == and >.
                '''
                
                ambiguous_at = self.__get_offset()*-1. #opposite of offset.
                
            
                if 0. == self.output_when_ambiguous:
                    return input.gt(ambiguous_at).to(output_dtype)
                if 1. == self.output_when_ambiguous:
                    return input.ge(ambiguous_at).to(output_dtype)
                result = input.eq(ambiguous_at).to(output_dtype)*self.output_when_ambiguous
                result += input.gt(ambiguous_at).to(output_dtype)
                return result
                
        #End of The function

    
    def extra_repr(self) -> str:
        input_str = "[0. to 1.]"
        if not self.input_is_01:
            input_str = "[-1. to 1.] or (-inf. to +inf.)"
            pass
        mode_str = "training" if self.training else "evaluating"
        
        result = f'Banarize layer, theorimatic input range:{input_str}, in {mode_str} mode'
        return result

    pass



# bp01 = BoundaryPair.from_float(0., 1.)
# '''01 to 01'''
# layer = Binarize_base(True, 1., True)
# input = torch.tensor([[0., 0.5, 1.]], requires_grad=True)
# print(layer(input), f'should be {layer.get_output_range(bp01).lower.item():.4f}, 0.5, {layer.get_output_range(bp01).upper.item():.4f}')
# layer.set_big_number(9.)
# print(layer(input), "should be 0.01_, 0.5, 0.99_")

# bpnp = BoundaryPair.from_float(-1., 1.)
# '''non 01 to 01'''
# layer = Binarize_base(False, 1., True)
# input = torch.tensor([[-1., 0., 1.]], requires_grad=True)
# print(layer(input), f'should be {layer.get_output_range(bpnp).lower.item():.4f}, 0.5, {layer.get_output_range(bp01).upper.item():.4f}')
# layer.set_big_number(9.)
# print(layer(input), "should be 0.01_, 0.5, 0.99_")

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






class Binarize_analog_to_01_non_standard(torch.nn.Module):
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

    def __init__(self, big_number = 1.5, \
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
    
    # def get_output_range(self, input: BoundaryPair)->BoundaryPair:
    #     '''Gets the output range from inner layer'''
    #     return self.base.get_output_range(input)
    
    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return False
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
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




class Binarize_01_to_01_non_standard(torch.nn.Module):
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

    def __init__(self, big_number = 3., \
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
    
    # def get_output_range(self, input: BoundaryPair)->BoundaryPair:
    #     '''Gets the output range from inner layer'''
    #     return self.base.get_output_range(input)

    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return False
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()

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





class Binarize_analog_to_01(torch.nn.Module):
    r"""This example layer shows how to use the binarize layer.
    """
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        #only the first binarize layer is analog to 01. Others must all be 01 to 01.
        self.Binarize = Binarize_analog_to_01_non_standard(1.5)
        self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.Binarize.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass
    
    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.Binarize(x)
        x = self.Final_Binarize(x)
        return x

    pass

# input = torch.tensor([-3., -1., 0., 1., 2.], requires_grad=True)
# input = input.unsqueeze(dim=0)
# layer = Binarize_analog_to_01()
# output = layer(input)
# print(output, "output, should be 0 0 0 1 1")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs=input)
# print(input.grad, "grad")

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
'''Because the forward path yields the same result as that one before, 
the only difference is grad. The param:big_number affects the grad by a lot.'''
class example_Binarize_analog_to_01_3times(torch.nn.Module):
    r"""This example layer shows how to use the binarize layer.
    """
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        #self.optionalGramo = GradientModification()#Probably not needed.
        #only the first binarize layer is analog to 01. Others must all be 01 to 01.
        self.Binarize1 = Binarize_analog_to_01_non_standard(1.5)
        self.Binarize2_optional = Binarize_01_to_01_non_standard(4.)
        self.Binarize3_optional = Binarize_01_to_01_non_standard(4.)
        self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        # gramo doesn't affect this.
        
        x = input
        #x = self.optionalGramo(x)#Probably not needed.
        x = self.Binarize1(x)
        x = self.Binarize2_optional(x)
        x = self.Binarize3_optional(x)
        x = self.Final_Binarize(x)
        return x

    pass


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





class Binarize_01_to_01(torch.nn.Module):
    r"""This example layer shows how to use the binarize layer.
    """
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        #only the first binarize layer is analog to 01. Others must all be 01 to 01.
        self.Binarize = Binarize_01_to_01_non_standard(10.)
        self.Final_Binarize = Binarize_01_Forward_only()
        pass
    
    def set_big_number(self, big_number:float, \
                I_know_Im_setting_a_value_which_may_be_less_than_1:bool = False):
        '''Simply set it to the inner layer.'''
        self.Binarize.set_big_number(big_number, I_know_Im_setting_a_value_which_may_be_less_than_1)
        pass
    
    def accepts_non_standard_range(self)->bool:
        return True
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.Binarize(x)
        x = self.Final_Binarize(x)
        return x

    pass

# input = torch.tensor([[0., 0.1, 0.2, 0.5, 1.]], requires_grad=True)
# #input = input.unsqueeze(dim=0)
# layer = Binarize_01_to_01()
# output = layer(input)
# print(output, "output, should be 0 0 0 0 1")
# g_in = torch.ones_like(output)
# torch.autograd.backward(output, g_in,inputs=input)
# print(input.grad, "grad")

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
                    #this automatic modification may mess sometime.
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


# '''basic test'''
# layer = DigitalMapper(2,3)
# # print(layer.raw_weight.data)
# # print(layer.raw_weight.data.shape)
# layer.raw_weight.data=torch.Tensor([[2., -2.],[ 0., 0.],[ -2., 2.]])
# input = torch.Tensor([[1., 0.]])
# input = input.requires_grad_()
# #print(input.softmax(dim=1))#dim = 1!!!
# print(layer(input), "should be 0.9_, 0.5, 0.0_")
# print(layer.raw_weight.softmax(dim=1))
# fds=432

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
    


# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111
# 111111111111111111





class AND_01(torch.nn.Module):
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
        
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01_to_01()
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
            x = self.Binarize1(x)

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



# '''a basic test'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
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
        
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_01_to_01()
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
            
            # no offset needed for OR gate.
            
            x = self.Binarize1(x)
            
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
                x = x.any(dim=1, keepdim=True)
                
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
        
        result = f'OR/NOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass



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
        
        # The intermediate result will be binarized with following layers.
        self.Binarize1 = Binarize_analog_to_01()
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
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            if not input.requires_grad:
                raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])

            x = x-0.5
            
            # this redundant *2 makes sure the grad is scaled to the same level as AND and OR path.
            # If you know what you are doing, you can comment this line out.
            x = x*2. 

            #back to rank-2
            x = x.prod(dim=2)
            # now, x is NXOR with threshold of 0.
            #x = -x+0.5 # some optimization needed in the future 
            
            x = self.Binarize1(x)
            
            # Now x is actually the XNOR
            # The output part is different from AND and OR styles.
            if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = 1.-x
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([opposite, x])
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
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
        
        result = f'OR/NOR layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass




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




# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
# untested!!!!!
class example_DigitalSignalProcessingUnit(torch.nn.Module):
    r'''This example shows how to handle pure digital signal.
    Only standard binary signals are allowed.
    
    All the gates are 2-input, both-output.
    
    Basic idea:
    >>> x = digital_mapper(input)
    >>> AND_ = and_gate(input)
    >>> OR_ = or_gate(input)
    >>> XOR_ = xor_gate(input)
    >>> output = concat(AND_, OR_, XOR_, 0, 1)
    
    '''
    #__constants__ = ['',]

    def __init__(self, in_features: int, and_gates: int, or_gates: int, xor_gates: int, \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.mapper_in_features = in_features
        self.and_gates = and_gates
        self.or_gates = or_gates
        self.xor_gates = xor_gates
        self.mapper_out_features = (and_gates+or_gates+xor_gates)*2
        
        self.mapper = DigitalMapper(self.mapper_in_features, self.mapper_out_features)
                
        self.and1 = AND_01(1.5,output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
        self.or1 = OR_01(1.5,output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
        self.xor1 = XOR_01(5.,output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
        #end of function        
        
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
     
    def forward(self, input:torch.Tensor)->torch.Tensor:
        '''This example only shows the training path.'''
        x = input
        x = self.mapper(x)
        
        and_head = self.and1(x)
        or_head = self.or1(x)              
        xor_head = self.xor1(x)
        zeros = torch.zeros([input.shape[0],1])
        ones = torch.zeros([input.shape[0],1])
        x = torch.concat([and_head, or_head, xor_head, zeros, ones],dim=1)
        return x
    #end of function
    
    pass


'''a basic test'''
a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
input = torch.concat((a,b), dim=1) 
layer = example_DigitalSignalProcessingUnit(2,不慌。))
layer.Binarize1.set_big_number(3.)#So you can adjust the range of grad.
result = layer(input)
print(result, "should be 0 1 0")
g_in = torch.ones_like(result)
torch.autograd.backward(result, g_in, inputs = [a,b])
print(a.grad, "a's grad")
print(b.grad, "b's grad")
fds=432









raise Exception("继续")



























'''
to do list
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









