
'''
Known issue.
When using customized backward function, 
the backward propagation seems to need all the require_grad of torch.Tensor set to True.
This rule looks including the y/label/ground truth/target. 
So, make sure the y is also set to require gradient.

All the 3 layers(gradient modification, mirror, mirror with gramo) need learning rate to work properly.
Set learning rate to the layers when ever learning rate changes. Or simply set lr to the layers in every iteration.
A simple way is to iterate all the model.Parameters(), if has(the layer, "set_learning_rate")is True, call it.
To access the learning rate, the code looks like:
optimizer.param_groups[0]["lr"]
You can also set to this member to change it directly. I believe it's safe.

Search for convert_to_plain_Linear function.
This function provides a equivalence plain Linear layer( or the Dense layer in Tensorflow).
You basically want to convert the mirror (both w/ and w/o the gramo enhancement) to a plain Linear one before you do the ONNX part.
The reason is, a mirror is basically 2 Linear, but does the same thing.
Mirror is only good for training, but 1x slower in predicting while needs 1x more memory.
'''

from typing import Any, Optional
import torch
import math

__all__ = [
    'GradientModification',
    'MirrorLayer',
    'MirrorWithGramo'
    ]


class GradientModificationFunction(torch.autograd.Function):
    r'''input param list:
    x:torch.Tensor,(must be set as require_grad = True)
    learning_rate:float, 
    epi=torch.tensor([1e-12]),
    suppression_factor = torch.tensor([1e3])
    
    retur type: torch.Tensor
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
    #def forward(ctx: Any, x:torch.Tensor, learning_rate:float, epi=torch.tensor([1e-12]), \
    #                 suppression_factor = torch.tensor([1e3]), *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, learning_rate:float, epi=torch.tensor([1e-12]), \
        #             suppression_factor = torch.tensor([1e3]))->torch.Tensor:
        #but python grammar punched me.

        #重要，如果输入的x 的 require_grad是false，就会出问题。报错是，什么，0号arg不需要grad，也没有grad fn。
        x = args[0]
        learning_rate = args[1]
        epi = args[2]
        suppression_factor = args[3]

        ctx.save_for_backward(learning_rate, epi, suppression_factor)
        return x

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        learning_rate:torch.Tensor
        learning_rate, epi, suppression_factor = ctx.saved_tensors

        print(learning_rate, epi, suppression_factor)

        mean = g.mean(dim=0, keepdim=True)
        _centralized = g - mean
        std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is slightly recommended by me.
        std_too_small = std < epi
        std = (std - std * std_too_small) + std_too_small * (epi* suppression_factor)
        _normalized = _centralized / std
        if learning_rate != 1.:
            return _normalized * learning_rate#, None, None, None
        else:
            return _normalized#, None, None, None
        pass
    pass  # class


class GradientModification(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """
    learning_rate:float
    def __init__(self, epi=torch.tensor([1e-12]), \
                 suppression_factor = torch.tensor([1e3]), \
                  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epi=epi
        self.suppression_factor = suppression_factor
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if not x.requires_grad:
            raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
          
        if not hasattr(self, "learning_rate"):
            raise Exception("Assign the learning rate for this layer. \n The code is like:\nmodel object var name.set_learning_rate(optimizer.param_groups[0][\"lr\"])")
          
        #forward(ctx, x:torch.Tensor, learning_rate:float, epi=torch.tensor([1e-12]), \
        #suppression_factor = torch.tensor([1e3]))->torch.Tensor:
        return GradientModificationFunction.apply(torch.tensor([x]), self.learning_rate, self.epi, \
                                                   self.suppression_factor)
    def set_learning_rate(self, learning_rate:float):
        self.learning_rate = learning_rate
        pass

    


# class test_gramo(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True,\
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.linear = torch.nn.Linear(in_features, out_features, bias, device, dtype)
#         self.gramo = GradientModification()

#     def set_learning_rate(self, learning_rate:float)->None:
#         self.gramo.set_learning_rate(learning_rate)

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         x = self.linear(x)
#         x = self.gramo(x)
#         return x
    
# model = test_gramo(1,1,True)
# loss_function = torch.nn.L1Loss()
# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
# model.set_learning_rate(optimizer.param_groups[0]["lr"])

# for epoch in range(1):
#     model.train()
#     input = torch.tensor([1.], requires_grad=True)
#     pred:torch.Tensor = model(input)
#     loss:torch.Tensor = loss_function(pred, torch.tensor([0.], requires_grad=True))
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

# jkldfs=345789







'''I copied the torch.nn.Linear code and modified it.
'''

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

    auto_merge_duration:int
    update_count:int
    learning_rate:float

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
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
        self.auto_merge_duration:int = 20
        self.update_count:int = 0

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.half_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.half_weight_mirrored.copy_(self.half_weight)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.half_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def set_learning_rate(self, learning_rate:float)->None:
        self.learning_rate = learning_rate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "learning_rate"):
            raise Exception("Assign the learning rate for this layer. \n The code is like:\nmodel object var name.set_learning_rate(optimizer.param_groups[0][\"lr\"])")
        
        if self.update_count>=self.auto_merge_duration:
            self.update_count = 0
            with torch.no_grad():
                self.half_weight = (self.half_weight+self.half_weight_mirrored)/2.
                self.half_weight_mirrored.copy_(self.half_weight)
                pass
            pass

        head1:torch.Tensor = torch.nn.functional.linear(input+self.learning_rate, self.half_weight)
        head2:torch.Tensor = torch.nn.functional.linear(input-self.learning_rate, self.half_weight, self.bias)
        return head1+head2

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def convert_to_plain_Linear(self)->torch.nn.Linear:
        has_bias = bool(self.bias)
        result:torch.nn.Linear = torch.nn.Linear(self.in_features, self.out_features, has_bias)
        result.weight = self.half_weight+self.half_weight_mirrored
        result.bias = self.bias
        return result
    pass



# model = MirrorLayer(1,1,True)
# loss_function = torch.nn.L1Loss()
# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
# model.set_learning_rate(optimizer.param_groups[0]["lr"])

# for epoch in range(1):
#     model.train()
#     pred:torch.Tensor = model(torch.tensor([1.]))
#     loss:torch.Tensor = loss_function(pred, torch.tensor([0.]))
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

# jkldfs=345789







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

    auto_merge_duration:int
    update_count:int
    learning_rate:float

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
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
        self.auto_merge_duration:int = 20
        self.update_count:int = 0

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.half_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.half_weight_mirrored.copy_(self.half_weight)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.half_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def set_learning_rate(self, learning_rate:float)->None:
        self.learning_rate = learning_rate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "learning_rate"):
            raise Exception("Assign the learning rate for this layer. \n The code is like:\nmodel object var name.set_learning_rate(optimizer.param_groups[0][\"lr\"])")
        
        if self.update_count>=self.auto_merge_duration:
            self.update_count = 0
            with torch.no_grad():
                self.half_weight = (self.half_weight+self.half_weight_mirrored)/2.
                self.half_weight_mirrored.copy_(self.half_weight)
                pass
            pass

        head1:torch.Tensor = torch.nn.functional.linear(input+1., self.half_weight)
        head2:torch.Tensor = torch.nn.functional.linear(input-1., self.half_weight, self.bias)
        output = GradientModificationFunction.apply(head1+head2)
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def convert_to_plain_Linear(self)->torch.nn.Linear:
        has_bias = bool(self.bias)
        result:torch.nn.Linear = torch.nn.Linear(self.in_features, self.out_features, has_bias)
        result.weight = self.half_weight+self.half_weight_mirrored
        result.bias = self.bias
        return result
    pass



# model = MirrorLayer(1,1,True)
# loss_function = torch.nn.L1Loss()
# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
# model.set_learning_rate(optimizer.param_groups[0]["lr"])

# for epoch in range(1):
#     model.train()
#     pred:torch.Tensor = model(torch.tensor([1.]))
#     loss:torch.Tensor = loss_function(pred, torch.tensor([0.]))
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

# jkldfs=345789

