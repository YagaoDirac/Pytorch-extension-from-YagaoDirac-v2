
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









class DigitalMapper(torch.nn.Module):
    r'''里面那个softmax，不排除内部数值过大，会导致无法训练，可能还需要额外的安全，比如clamp'''
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



