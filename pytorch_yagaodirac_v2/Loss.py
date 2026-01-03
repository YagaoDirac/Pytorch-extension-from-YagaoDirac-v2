
from typing import Any, Optional
#import math
import torch

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal


def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
if "test" and False:
    assert __DEBUG_ME__()
    pass





def assert_param_shape__batch_dim(param:torch.Tensor):
    assert param.shape.__len__() == 2, "Only accept rank-2 tensor. The shape should be[batch, something]"
    pass




class BCELoss_outputs_real_probabilityFunction(torch.autograd.Function):
    r'''
    Warning: this is the Function class. Do not use it directly. Use class SigmoidBCELoss_outputs_real_probability.
    
    This tool works as a replacement of torch.nn.BCELoss. It makes the training process stabler.
    
    The input only accepts in range [0,1]. The target can only be 0 or 1.
    
    Interface is a bit different from the torch.nn.BCELoss.
    
    The grad is calculated only with input and target. Also, grad is in range [0, 1-no_grad_zone_size].
    By default, no_grad_zone_size is 0.1, so the grad range is [0, 0.9]. Now you know how to tweak lr.
        
    input param:
    >>> input:torch.Tensor
    >>> target:torch.Tensor
    >>> weight:torch.Tensor|None,
    >>> no_grad_zone_size:torch.Tensor
    
    return param: 
    >>> result:torch.Tensor, 1-acc,It's in range [0,1]
    '''
    @staticmethod
    #def forward(*args: Any, **kwargs: Any)->Any:
    def forward(input:torch.Tensor,target:torch.Tensor, weight:torch.Tensor|None,\
                    no_grad_zone_size:torch.Tensor,\
                                                *args: Any, **kwargs: Any)->Any:
        '''
        input param:
        >>> input:torch.Tensor
        >>> target:torch.Tensor
        >>> weight:torch.Tensor|None,
        >>> no_grad_zone_size:torch.Tensor

        return param: 
        >>> result:torch.Tensor, 1-acc
        '''    
        
        
        assert_param_shape__batch_dim(input)#this is my convention. If you don't like it, commend it out.
        _input_lt_threshold = input.ge(0.5)
        result_raw = _input_lt_threshold.logical_xor(target)
        result = result_raw.mean(dim=1, dtype=torch.float64)#basically this is the end of neural net.
        result.requires_grad_(input.requires_grad)
        assert result.shape.__len__() == 1
        assert result.shape[0] == input.shape[0]
        return result     

    @staticmethod
    def setup_context(ctx:torch.autograd.function.FunctionCtx, inputs, output):
        #ctx:torch.autograd.function.FunctionCtx
        
        input:torch.Tensor = inputs[0]
        target:torch.Tensor = inputs[1]
        #weight:torch.Tensor|None, = inputs[2]  moved below
        no_grad_zone_size:torch.Tensor = inputs[3]
        
        #result = output[0]
        
        weight:torch.Tensor = inputs[2]
        
        if weight is None:
            #ctx.save_for_backward(input, no_grad_zone_size,result, _target_gt_threshold)
            ctx.save_for_backward(input, target, no_grad_zone_size)
            pass
        else:
            #ctx.save_for_backward(input, no_grad_zone_size, result, _target_gt_threshold, \
            ctx.save_for_backward(input, target, no_grad_zone_size,   weight)
            pass
        return
        #return super().setup_context(ctx, inputs, output)

    @staticmethod
    #def backward(ctx:torch.autograd.function.FunctionCtx, g_in_b_o):#->tuple[Optional[torch.Tensor], None, None, None]:
    def backward(ctx, g_in_b):#->tuple[Optional[torch.Tensor], None, None, None]:
        #super().backward()
        # input:torch.Tensor
        # no_grad_zone_size:torch.Tensor 
        #result:torch.Tensor
        # target:torch.Tensor
        # weight:torch.Tensor|None
        
        match ctx.saved_tensors.__len__():
            case 3:
                (input, target, no_grad_zone_size)= ctx.saved_tensors
                weight = None
                pass
            case 4:
                (input, target, no_grad_zone_size,   weight) = ctx.saved_tensors
                pass
            case _:
                assert False, "unreachable"
            # end of match.
        
        if input.requires_grad == False:
            return None,None,None,None
        
        #_total_dist = 1.
        #input_gt_0_5 = input.gt(0.5)
        
        grad = torch.zeros_like(input)
        #grad[_target_gt_threshold] = input[_target_gt_threshold]-no_grad_zone_size
        grad[ target] = torch.minimum(torch.tensor(0.), input[ target]-(1-no_grad_zone_size))
        #grad[~_target_gt_threshold] = input[~_target_gt_threshold]-(1-no_grad_zone_size)
        grad[~target] = torch.maximum(torch.tensor(0.), input[~target]-no_grad_zone_size)
        #xxxxxxxx grad.sub_(no_grad_zone_size)
        #xxxxxxxxx why is this ??? grad[~grad.gt(0.)] = 0. #-inf and nan returns false from gt().
        
        if g_in_b.ne(1.).any():
            g_in_b = g_in_b.unsqueeze(dim=-1).expand(-1,grad.shape[1])
            assert g_in_b.shape == grad.shape
            grad.mul_(g_in_b)
            pass
        if weight is not None:
            grad.mul_(weight)
            pass
        
        return grad,None,None,None

    pass  # class

if '''BCELoss_outputs_real_probabilityFunction''' and __DEBUG_ME__() and True:
    a = torch.tensor( [[0 ,0.1,0.2,0.9,1.]], requires_grad=True)
    target = torch.zeros_like(a, dtype = torch.bool)
    weight = None
    no_grad_zone_size = torch.tensor(0.)
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert _tensor_equal(b, torch.tensor([0.4]))
    b.backward()#inputs = ?
    #g_in = torch.tensor([1.], dtype=torch.float16)
    #torch.autograd.backward(b, g_in, inputs = a)
    assert a.grad is not None
    assert _tensor_equal(a.grad, torch.tensor([[0., 0.1, 0.2, 0.9, 1]]))#, epsilon=1e-7)


    #target
    a = torch.tensor( [[0 ,0.1,0.2,0.9,1.]], requires_grad=True)
    target = torch.ones_like(a, dtype = torch.bool)
    weight = None
    no_grad_zone_size = torch.tensor(0.)
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert _tensor_equal(b, torch.tensor([0.6]))
    b.backward()#inputs = ?
    assert a.grad is not None
    assert _tensor_equal(a.grad, torch.tensor([[-1., -0.9, -0.8, -0.1, 0]]))#, epsilon=1e-7)
    
    a = torch.tensor( [[0 ,0.1,0.2,0.9,1.]], requires_grad=True)
    target = torch.tensor([[False,False,False,True,True]])
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert _tensor_equal(b, torch.tensor([0.]))
    b.backward()#inputs = ?
    assert a.grad is not None
    assert _tensor_equal(a.grad, torch.tensor([[0., 0.1, 0.2, -0.1, 0]]))#, epsilon=1e-7)
    
    
    # no_grad_zone_size
    a = torch.tensor( [[0 ,0.1,0.2,0.9,1.]], requires_grad=True)
    target = torch.zeros_like(a, dtype = torch.bool)
    no_grad_zone_size = torch.tensor(0.1)
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert _tensor_equal(b, torch.tensor([0.4]))
    b.backward()#inputs = ?
    assert a.grad is not None
    assert _tensor_equal(a.grad, torch.tensor([[0., 0, 0.1, 0.8, 0.9]]))#, epsilon=1e-7)

    
    #weight = torch.tensor
    a = torch.tensor([[0 ,0.1,0.2,0.9,1.]], requires_grad=True)
    target = torch.zeros_like(a, dtype = torch.bool)
    weight = torch.tensor([[11. , 12, 13,14,15]])
    no_grad_zone_size = torch.tensor(0.)
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert _tensor_equal(b, torch.tensor([0.4]))
    b.backward()#inputs = ?
    assert a.grad is not None
    assert _tensor_equal(a.grad, torch.tensor([[0., 1.2, 2.6, 12.6, 15]]))#, epsilon=1e-7)

    
    # no grad
    a = torch.tensor([[0 ,0.1,0.2,0.9,1.]], requires_grad=False)
    target = torch.zeros_like(a, dtype = torch.bool)
    weight = None
    no_grad_zone_size = torch.tensor(0.)
    assert a.requires_grad == False
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert b.grad_fn is None

    
    #g_in, although I don't recommend you do this. This tool is not designed to do so.
    a = torch.tensor( [[0 ,0.1,0.2,0.9,1.]], requires_grad=True)
    target = torch.zeros_like(a, dtype = torch.bool)
    weight = None
    no_grad_zone_size = torch.tensor(0.)
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert _tensor_equal(b, torch.tensor([0.4]))
    g_in = torch.tensor([123.])
    torch.autograd.backward(b, g_in, inputs=a)
    assert a.grad is not None
    assert _tensor_equal(a.grad, torch.tensor([[0., 12.3, 24.6, 110.7, 123]]))#, epsilon=1e-7)
    
    
    #batch
    a = torch.tensor( [[0 ,0.1,0.2,0.9,1.],[0 ,0,0,1,1]], requires_grad=True)
    target = torch.tensor( [[False,False,False,False,True],[False,True,True,True,True]])
    weight = None
    no_grad_zone_size = torch.tensor(0.)
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert _tensor_equal(b, torch.tensor([0.2, 0.4]))
    b.backward(torch.ones_like(b))#inputs = ?
    assert a.grad is not None
    assert _tensor_equal(a.grad, torch.tensor([[0., 0.1, 0.2, 0.9, 0],[0,-1,-1,0,0],]))#, epsilon=1e-7)
    
    for _ in range(15):
        a = torch.rand(size=[32,16])
        a.requires_grad_(True)
        assert a.requires_grad
        target = torch.randint_like(a, low=0, high=2, dtype=torch.bool)
        weight = None
        no_grad_zone_size = torch.tensor(0.)
        b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
        assert b.ge(0.).all()# b is in range [0,1]
        assert b.le(1.).all()
        b.backward(torch.ones_like(b))#inputs = ?
        assert a.grad is not None
        _a_ge_0 = a.grad.ge(0.)
        assert _a_ge_0.ne(target).all()
        _a_le_0 = a.grad.le(0.)
        assert _a_le_0.eq(target).all()
        pass
    
    pass

if '''dtype adaption.''' and __DEBUG_ME__() and True:
    a = torch.tensor( [[0 ,0.1,0.2,0.9,1.]], requires_grad=True)
    original_dtype = a.dtype
    target = torch.zeros_like(a, dtype = torch.bool)
    weight = torch.tensor([[11. , 12, 13,14,15]], dtype=torch.float64)
    no_grad_zone_size = torch.tensor(0.1, dtype=torch.float64)
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert b.dtype == torch.float64
    g_in =  torch.tensor([123.], dtype=torch.float64)
    b.backward(g_in)#inputs = ?
    assert a.grad is not None
    assert a.grad.dtype == original_dtype
    pass

if '''device adaption''' and __DEBUG_ME__() and True:
    #torch.cuda.empty_cache()
    a = torch.tensor([[0 ,0.2,0.9]], requires_grad=True).cuda()
    a.grad = torch.tensor([[33.,44,55]], device='cuda')
    target = torch.tensor([[True,False,True]]).cuda()
    #assert a.shape == target.shape
    weight = torch.tensor([[11., 12, 13]]).cuda()
    no_grad_zone_size = torch.tensor(0.1).cuda()
    b = BCELoss_outputs_real_probabilityFunction.apply(a, target, weight, no_grad_zone_size)
    assert _float_equal(b.item(), 1./3)
    assert b.grad_fn is not None
    assert b.device == torch.device('cuda', index=0)
    assert b.device.type == 'cuda'
    #torch.Tensor.backward(b,g_in,inputs=a)
    b.backward(inputs = a)
    #torch.cuda.synchronize()
    assert a.grad is not None
    assert a.grad.device == torch.device('cuda', index=0)
    assert a.grad.device.type == 'cuda'
    pass



class BCELoss_outputs_real_probability(torch.nn.modules.loss._WeightedLoss):
    weight:torch.nn.parameter.Parameter|None
    no_grad_zone_size:torch.nn.parameter.Parameter
    safety_check:bool
    def __init__(self, no_grad_zone_size:float|torch.Tensor = 0.1, weight: Optional[torch.Tensor] = None,
                    safety_check = True,) -> None:
        #format first. copy pasted from torch.nn.BCELoss
        size_average = None
        reduce = None
        reduction = None
        super().__init__(size_average, reduce, reduction)
        
        if weight is not None:
            self.weight = torch.nn.parameter.Parameter(weight, requires_grad=False)
            pass
        else:
            self.weight = None
            pass
        
        assert no_grad_zone_size>=0.
        assert no_grad_zone_size<=0.5
        if isinstance(no_grad_zone_size, torch.Tensor):
            self.no_grad_zone_size = torch.nn.parameter.Parameter(no_grad_zone_size.detach().clone(), requires_grad=False)
            pass
        elif isinstance(no_grad_zone_size, float):
            self.no_grad_zone_size = torch.nn.parameter.Parameter(torch.tensor(no_grad_zone_size), requires_grad=False)
            pass
        else:
            assert False, "unreachable"
        
        self.safety_check = safety_check
        pass

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        if self.safety_check:
            assert self.no_grad_zone_size.shape == torch.Size([])
            assert self.no_grad_zone_size.ge(0.)
            assert input.le(1.).all()
            assert input.ge(0.).all()
            target_gt_0_5 = target.gt(0.5)
            assert target[target_gt_0_5].eq(1.).all()
            assert target[~target_gt_0_5].eq(0.).all()
            pass
        
        
        if target.dtype == torch.bool:
            target_for_inner = target
            pass
        elif target.dtype == torch.float:
            target_for_inner = target.gt(0.5)
            pass
        else:
            assert False, "target.dtype can only be bool or float"
        
        temp_result_tuple = BCELoss_outputs_real_probabilityFunction.apply(
                                input, target_for_inner, self.weight, self.no_grad_zone_size)
        return temp_result_tuple
    
    def set_weight(self, new_weight:torch.Tensor|None= None):
        if new_weight is None:
            self.weight = None
            return
        if isinstance(new_weight, torch.Tensor):
            if self.weight is None:
                self.weight = torch.nn.parameter.Parameter(new_weight.detach().clone(), requires_grad=False)
                return
            else:
                self.weight.data = new_weight.detach().clone().requires_grad_(False)
                return 
        assert False, "unreachable"
        pass#/function
    
    def set_no_grad_zone_size(self, no_grad_zone_size:float|torch.Tensor):
        assert no_grad_zone_size>=0.
        assert no_grad_zone_size<=0.5
        if isinstance(no_grad_zone_size, torch.Tensor):
            self.no_grad_zone_size.data = no_grad_zone_size.detach().clone().requires_grad_(False)
            self.no_grad_zone_size.requires_grad_(False)
            pass
        elif isinstance(no_grad_zone_size, float):
            self.no_grad_zone_size.data = torch.tensor(no_grad_zone_size)
            self.no_grad_zone_size.requires_grad_(False)
            pass
        else:
            assert False, "unreachable"
        pass#/function
    
    def set_safety_check(self, safety_check:bool):
        self.safety_check = safety_check
        pass#/function
            
    pass#end of class

if '''all the setters''' and __DEBUG_ME__() and True:
    model_BCELoss_outputs_real_probability = BCELoss_outputs_real_probability()
    assert model_BCELoss_outputs_real_probability.weight is None
    assert model_BCELoss_outputs_real_probability.no_grad_zone_size.requires_grad == False
    assert isinstance(model_BCELoss_outputs_real_probability.safety_check, bool)
    
    model_BCELoss_outputs_real_probability.set_weight(torch.tensor([0.123, 0.234]))
    assert isinstance(model_BCELoss_outputs_real_probability.weight, torch.nn.parameter.Parameter)
    assert _tensor_equal(model_BCELoss_outputs_real_probability.weight, torch.tensor([0.123, 0.234]))
    assert model_BCELoss_outputs_real_probability.weight.requires_grad is not None
    assert isinstance(model_BCELoss_outputs_real_probability.weight.requires_grad, bool)
    assert model_BCELoss_outputs_real_probability.weight.requires_grad == False
    
    model_BCELoss_outputs_real_probability.set_weight(None)
    assert model_BCELoss_outputs_real_probability.weight is None
    
    model_BCELoss_outputs_real_probability.set_no_grad_zone_size(0.345)
    assert _float_equal(model_BCELoss_outputs_real_probability.no_grad_zone_size.item(), 0.345)
    assert model_BCELoss_outputs_real_probability.no_grad_zone_size.requires_grad == False
    
    model_BCELoss_outputs_real_probability.set_no_grad_zone_size(torch.tensor(0.456))
    assert _float_equal(model_BCELoss_outputs_real_probability.no_grad_zone_size.item(), 0.456)
    assert model_BCELoss_outputs_real_probability.no_grad_zone_size.requires_grad == False
    pass

if '''dtype adaption.''' and __DEBUG_ME__() and True:
    input = torch.tensor([[1.]], requires_grad=True)
    target = torch.tensor([[False]])
    model_BCELoss_outputs_real_probability = BCELoss_outputs_real_probability()
    model_BCELoss_outputs_real_probability.to(torch.float64)#!!!!!!!!!!!!!!!!!!!!!!
    #model.to(torch.float16)

    optimizer = torch.optim.SGD([input], lr=0.1)
    for epoch in range(1):
        model_BCELoss_outputs_real_probability.train()
        one_minus_acc = model_BCELoss_outputs_real_probability(input, target)
        assert one_minus_acc.dtype == torch.float64# this loss set to fp64
        optimizer.zero_grad()
        one_minus_acc.backward(inputs = input)#inputs = ?
        #optimizer.param_groups[0]["lr"] = 0.01
        assert input.grad is not None
        assert _float_equal(input.grad.item(), 0.9)
        assert input.grad.dtype == input.dtype

        optimizer.step()
        assert _float_equal(input.item(), 0.91)#1- 0.9*0.1
        
        model_BCELoss_outputs_real_probability.eval()
        pass
    pass

if '''init test ???????????????????????????????????''' and __DEBUG_ME__() and True:
    "this is a torch.nn.modules.loss._WeightedLoss. Idk how to test this part."
    # layer_BCELoss_outputs_real_probability = BCELoss_outputs_real_probability(device='cuda')
    # assert layer_BCELoss_outputs_real_probability.scaling_factor.device == torch.device('cuda', index=0)
    # assert layer_BCELoss_outputs_real_probability.scaling_factor.dtype == torch.float32
    # layer_BCELoss_outputs_real_probability = BCELoss_outputs_real_probability(dtype=torch.float64)
    # assert layer_BCELoss_outputs_real_probability.scaling_factor.dtype == torch.float64
    # layer_BCELoss_outputs_real_probability = BCELoss_outputs_real_probability(dtype=torch.float32)
    # assert layer_BCELoss_outputs_real_probability.scaling_factor.dtype == torch.float32
    # layer_BCELoss_outputs_real_probability = BCELoss_outputs_real_probability(dtype=torch.float16)
    # assert layer_BCELoss_outputs_real_probability.scaling_factor.dtype == torch.float32
    pass

if "how is it different from the vanilla pytorch version?"and __DEBUG_ME__() and True:
    "vanilla pytorch version"
    bceloss_no_reduction = torch.nn.BCELoss(reduction="none")
    input = torch.tensor([0., 0.25, 0.5, 0.75, 1.], requires_grad=True)
    target = torch.zeros_like(input)
    loss = bceloss_no_reduction(input, target)
    assert _tensor_equal(loss, torch.tensor([0, 0.2877, 0.6931, 1.3863, 100]))
    loss.backward(torch.ones_like(loss), inputs = input)
    assert input.grad is not None
    assert _tensor_equal(input.grad, torch.tensor([0, 1.3333, 2, 4, 1e12]))
    # Notice the last element in grad, it's 1e12. 
    # Most elements are in range 0 to 10, but the wrongest one provides 1e12.
    # What lr should you use?
    # yagao be like:1e12 bro. Do you see you model flying in the universe?
    
    # bc this, I made my version.
    "yagaodirac version(at least this version)"
    layer_BCELoss_outputs_real_probability = BCELoss_outputs_real_probability()
    input = torch.tensor([[0., 0.25, 0.5, 0.75, 1.]], requires_grad=True)
    target = torch.zeros_like(input)
    loss = layer_BCELoss_outputs_real_probability(input, target)
    assert _tensor_equal(loss, torch.tensor([0.5]), epsilon=0.1001)# 0.4 or 0.6. Trivial.
    loss.backward(torch.ones_like(loss), inputs = input)
    assert input.grad is not None
    assert _tensor_equal(input.grad   , torch.tensor([[0, 0.15, 0.4, 0.65, 0.9]]))
    assert _tensor_equal(input.grad*5., torch.tensor([[0, 0.75, 2  , 3.25, 4.5]]))
    pass





assert False, '''todo list:
I need a new softmax. 

test new bce and softmax with real training case.
'''

