from typing import Optional
import torch
import math

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


if False:
    model = MirrorLayer(1,1,True)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    model.set_learning_rate(optimizer.param_groups[0]["lr"])

    for epoch in range(1):
        model.train()
        pred:torch.Tensor = model(torch.tensor([1.]))
        loss:torch.Tensor = loss_function(pred, torch.tensor([0.]))
        optimizer.zero_grad()
        loss.backward()
        #optimizer.param_groups[0]["lr"] = 0.01
        optimizer.step()
        model.eval()

    jkldfs=345789
