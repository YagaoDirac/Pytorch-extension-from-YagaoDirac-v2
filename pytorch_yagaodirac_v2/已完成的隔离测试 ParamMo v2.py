from typing import Any, List, Tuple, Optional, Self
import math
import torch



class GradientModificationFunction_v2(torch.autograd.Function):
    r'''input param:
    >>> x:torch.Tensor (must be set as require_grad = True)
    >>> scaling_factor = torch.tensor([1.])
    >>> epi = torch.tensor([1e-5])
    >>> div_me_when_g_too_small = torch.tensor([1e-3])

    retur type: torch.Tensor
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, scaling_factor:float = torch.tensor([1.]), \
        #               epi=torch.tensor([1e-5]), \
        #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
        #but python grammar punched me.
        x:torch.Tensor = args[0]
        scaling_factor = args[1]
        epi = args[2]
        mul_me_when_g_too_small = args[3]
        # the default values:
        # scaling_factor = torch.tensor([1.])
        # epi = torch.tensor([0.00001])
        # div_me_when_g_too_small = torch.tensor([0.001])
        # the definition of the 3 param are different from the previous version
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")
        
        x_needs_grad = torch.tensor([x.requires_grad])
        ctx.save_for_backward(scaling_factor, epi, mul_me_when_g_too_small, x_needs_grad)
        return x

    @staticmethod
    def backward(ctx, g_in_b_o):
        #super().backward()
        scaling_factor:torch.Tensor
        x_needs_grad:torch.Tensor
        scaling_factor, epi, mul_me_when_g_too_small, x_needs_grad, = ctx.saved_tensors
        if x_needs_grad.logical_not():
            return None, None, None, None

        out_features_as_float:torch.Tensor = torch.tensor([g_in_b_o.shape[-1]], dtype=torch.float64, device=g_in_b_o.device)
        #mul_me_when_g_too_small = mul_me_when_g_too_small_per_element#*out_features_as_float

        avg_length_per_element_b_1:torch.Tensor = (g_in_b_o.mul(g_in_b_o).sum(dim=1,keepdim = True)/out_features_as_float).sqrt()
        mul_me_when_g_is_ok_raw_b_1 = scaling_factor/avg_length_per_element_b_1
        
        mul_me_when_g_is_ok_raw_b_1.nan_to_num_(mul_me_when_g_too_small.item())
        not_too_big_flag = mul_me_when_g_is_ok_raw_b_1.lt(mul_me_when_g_too_small*1000)#is this needed?
        mul_me_when_g_is_ok_b_1 = not_too_big_flag*mul_me_when_g_is_ok_raw_b_1+not_too_big_flag.logical_not()*mul_me_when_g_too_small
        
        too_small_b_1:torch.Tensor = avg_length_per_element_b_1.le(epi)#*out_features_as_float)
        
        mul_me_b_1 = too_small_b_1.logical_not()*mul_me_when_g_is_ok_b_1+ too_small_b_1*mul_me_when_g_too_small
        mul_me_b_1 = mul_me_b_1.to(g_in_b_o.dtype)
        g_out_b_o:torch.Tensor = g_in_b_o*mul_me_b_1

        return g_out_b_o, None, None, None

    pass  # class



if '''dim irrelated gramo''' and False:
    scaling_factor = torch.tensor([1.], dtype=torch.float64)
    epi=torch.tensor([1e-3], dtype=torch.float32)
    mul_me_when_g_too_small = torch.tensor([10], dtype=torch.float16)
    a = torch.zeros([5,2], requires_grad=True, dtype=torch.float16)
    b = GradientModificationFunction_v2.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.1,0.2],[0.01,0.02,],[0.001,0.002],[1e-4,2e-4],[1e-5,2e-5]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    fds=432
    a = torch.zeros([5,1], requires_grad=True, dtype=torch.float16)
    b = GradientModificationFunction_v2.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.1],[0.01],[0.001],[1e-4],[1e-5]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    fds=432
    a = torch.zeros([5,10], requires_grad=True, dtype=torch.float16)
    b = GradientModificationFunction_v2.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.1],[0.01],[0.001],[1e-4],[1e-5]], dtype=torch.float16).repeat([1,10])
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad[:,1])
    pass

if '''dtype adaption.''' and False:
    scaling_factor = torch.tensor([1.], dtype=torch.float64)
    epi=torch.tensor([1e-5], dtype=torch.float32)
    mul_me_when_g_too_small = torch.tensor([1e3], dtype=torch.float16)
    a = torch.tensor([[0.]], requires_grad=True, dtype=torch.float16)
    original_dtype = a.dtype
    b = GradientModificationFunction_v2.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
    g_in = torch.tensor([[1.]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    #print(g[0])
    print(a.grad.dtype, "should be ", original_dtype)
    pass

if '''device adaption''' and False:
    scaling_factor = torch.tensor([1.]).cuda()
    epi=torch.tensor([1e-5]).cuda()
    mul_me_when_g_too_small = torch.tensor([1e3]).cuda()
    a = torch.tensor([[0.]], requires_grad=True).cuda()
    b = GradientModificationFunction_v2.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[1.]]).cuda()
    torch.autograd.backward(b, g_in,inputs= a)
    #print(g[0])
    print(a.grad.device, "should be cuda")
    pass

if '''NOT NEEDED when g_in is too small.''' and False:
    scaling_factor = torch.tensor([1.])
    epi=torch.tensor([1e-5])
    mul_me_when_g_too_small = torch.tensor([1e3])
    a = torch.tensor([[0.]], requires_grad=True)
    b = GradientModificationFunction_v2.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.0000012]])
    torch.autograd.backward(b, g_in,inputs= a)
    #print(g[0])
    print(a.grad, "should be ", g_in*mul_me_when_g_too_small)
    pass

if '''NOT NEEDED when g_in is NOT too small.''' and False:
    scaling_factor = torch.tensor([1.])
    epi=torch.tensor([1e-5])
    mul_me_when_g_too_small = torch.tensor([1e3])
    input = torch.tensor([[0.]], requires_grad=True)
    output = GradientModificationFunction_v2.apply(input, scaling_factor,epi,mul_me_when_g_too_small)
    ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
    g_in = torch.tensor([[0.12]])
    torch.autograd.backward(output, g_in,inputs= input)
    print(input.grad, "should be 1.")
    pass

if '''NOT NEEDED Computation is limited inside each batch.''' and False:
    '''The shape is [batches, inside a batch]. '''
    scaling_factor = torch.tensor([1.])
    epi=torch.tensor([1e-5])
    mul_me_when_g_too_small = torch.tensor([1e3])
    input = torch.tensor([[0., 0.]], requires_grad=True)
    output = GradientModificationFunction_v2.apply(input, scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.1, 0.1]])
    torch.autograd.backward(output, g_in,inputs= input)
    print(input.grad, "should be 1, 1")

    input = torch.tensor([[0.], [0.]], requires_grad=True)
    output = GradientModificationFunction_v2.apply(input, scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.12], [0.12]])
    torch.autograd.backward(output, g_in,inputs= input)
    print(input.grad, "should be 1., 1.")

    input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
    output = GradientModificationFunction_v2.apply(input, scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.1, 0.173], [0.12, 0.12]])
    torch.autograd.backward(output, g_in,inputs= input)
    print(input.grad, "should be 0.7, 1.3?, 1, 1")

    '''Other 3 input besides the main input x.'''
    input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
    #output = GradientModificationFunction.apply(input, scaling_factor,     epi,                  mul_me_when_g_too_small)
    output = GradientModificationFunction_v2.apply(input, torch.tensor([1.]), torch.tensor([0.001]),torch.tensor([10]))
    g_in = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
    torch.autograd.backward(output, g_in,inputs= input)
    print(input.grad, "should be 0.001, 0.001, 1, 1")

    input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
    #output = GradientModificationFunction.apply(input, scaling_factor,     epi,                  mul_me_when_g_too_small)
    output = GradientModificationFunction_v2.apply(input, torch.tensor([2.]), torch.tensor([0.001]),torch.tensor([10]))
    g_in = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
    torch.autograd.backward(output, g_in,inputs= input)
    print(input.grad, "should be 0.002, 0.002, 2, 2")
    pass



class GradientModification_v2(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """

    def __init__(self, scaling_factor:float = 1., \
                       epi=1e-5, \
                       mul_me_when_g_too_small = 1e3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_factor = torch.nn.Parameter(torch.tensor([scaling_factor]), requires_grad=False)
        self.scaling_factor.requires_grad_(False)
        self.epi=torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
        self.epi.requires_grad_(False)
        self.mul_me_when_g_too_small = torch.nn.Parameter(torch.tensor([mul_me_when_g_too_small]), requires_grad=False)
        self.mul_me_when_g_too_small.requires_grad_(False)
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_factor:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction_v2.apply(x, self.scaling_factor, self.epi, \
                                                   self.mul_me_when_g_too_small)
    def set_scaling_factor(self, scaling_factor:float)->None:
        the_device = self.scaling_factor.device
        the_dtype = self.scaling_factor.dtype
        self.scaling_factor.data = torch.tensor([scaling_factor], device=the_device, dtype=the_dtype)
        self.scaling_factor.requires_grad_(False)
        pass
    def scale_scaling_factor(self, by:float)->None:
        self.set_scaling_factor((self.scaling_factor*by).item())
        pass
    
    def set_epi(self, epi:float)->None:
        the_device = self.epi.device
        the_dtype = self.epi.dtype
        self.epi.data = torch.tensor([epi], device=the_device, dtype=the_dtype)
        self.epi.requires_grad_(False)
        pass
    def set_mul_me_when_g_too_small(self, mul_me_when_g_too_small:float)->None:
        the_device = self.mul_me_when_g_too_small.device
        the_dtype = self.mul_me_when_g_too_small.dtype
        self.mul_me_when_g_too_small.data = torch.tensor([mul_me_when_g_too_small], device=the_device, dtype=the_dtype)
        self.mul_me_when_g_too_small.requires_grad_(False)
        pass

    def extra_repr(self) -> str:
        return f'scaling_factor={self.scaling_factor.item():.4e}, epi={self.epi.item():.4e}, mul_me_when_g_too_small={self.mul_me_when_g_too_small.item():.4e}'


if '''all the setters''' and False:
    model = GradientModification_v2()
    print(model.scaling_factor.requires_grad, "should be False")
    print(model.epi.requires_grad, "should be False")
    print(model.mul_me_when_g_too_small.requires_grad, "should be False")
    model.set_scaling_factor(0.123)
    print(model.scaling_factor, "should be 0.123")
    print(model.scaling_factor.requires_grad, "should be False")
    model.set_epi(0.234)
    print(model.epi, "should be 0.234")
    print(model.epi.requires_grad, "should be False")
    model.set_mul_me_when_g_too_small(0.345)
    print(model.mul_me_when_g_too_small, "should be 0.345")
    print(model.mul_me_when_g_too_small.requires_grad, "should be False")
    pass

if '''dtype adaption.''' and False:
    input = torch.tensor([[1.]], requires_grad=True)
    target = torch.tensor([[0.]])
    model = GradientModification_v2()
    model.to(torch.float64)
    #model.to(torch.float16)

    loss_function = torch.nn.L1Loss()# the L1Loss function only provides the direction. It's the dirivitive of abs.
    optimizer = torch.optim.SGD([input], lr=0.1)
    for epoch in range(1):
        model.train()
        pred = model(input)
        print(pred.dtype, "pred.dtype should be f32")
        loss = loss_function(pred, target)
        print(loss.dtype, "loss.dtype should be f32")
        optimizer.zero_grad()
        loss.backward()
        #optimizer.param_groups[0]["lr"] = 0.01
        print(input.grad, "should be 1.")
        print(input.grad.dtype, "input.grad.dtype should be f32")

        optimizer.step()
        print(input, "should be 0.9")
        
        model.eval()
        pass
    pass

# if '''when grad is too small.''' and False:
#     input = torch.tensor([[0.]], requires_grad=True)
#     target = torch.tensor([[0.00012]])
#     model = GradientModification_v2(epi=0.001,mul_me_when_g_too_small=10)###

#     loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
#     optimizer = torch.optim.SGD([input], lr=0.1)
#     for epoch in range(1):
#         model.train()
#         pred = model(input)
#         loss = loss_function(pred, target)
#         the grad input for the backward is -2*0.002. It's small enough to be *10
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.param_groups[0]["lr"] = 0.01
#         print(input.grad, "should be -0.0024")

#         optimizer.step()
#         print(input.item(), "should be 0.00024")
        
#         model.eval()
#         pass
#     pass

# if '''when grad is too small while dim is not 1.''' and True:
#     input = torch.tensor([[0.,0.]], requires_grad=True)
#     target =  torch.tensor([[0.00012,0.00012]])
#     model = GradientModification_v2(epi=0.001,mul_me_when_g_too_small=10)###

#     loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
#     optimizer = torch.optim.SGD([input], lr=0.1)
#     for epoch in range(1):
#         model.train()
#         pred = model(input)
#         loss = loss_function(pred, target)
#         the grad input for the backward is -2*0.002/2. . It's small enough to be *10
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.param_groups[0]["lr"] = 0.01
#         print(input.grad, "should be -0.0024")

#         optimizer.step()
#         print(input, "should be 0.00024")
        
#         model.eval()
#         pass

#     '''same test but dim is different.''' 
#     input = torch.tensor([[0., 0., 0.]], requires_grad=True)
#     target = torch.tensor([[0.00012, 0.00012, 0.00012]])
#     model = GradientModification_v2(epi=0.001,mul_me_when_g_too_small=10)###

#     loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
#     optimizer = torch.optim.SGD([input], lr=0.1)
#     for epoch in range(1):
#         model.train()
#         pred = model(input)
#         loss = loss_function(pred, target)
#         the grad input for the backward is -2*0.00012/3. . It's small enough to be *10
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.param_groups[0]["lr"] = 0.01
#         print(input.grad, "should be -0.0024")

#         optimizer.step()
#         print(input, "should be 0.00024")
        
#         model.eval()
#         pass

#     '''same test but dim is different.'''
#     print("---------------")
#     raise Exception()
    
#     input = torch.tensor([[0.,0.],[0.,0.]], requires_grad=True)
#     target = torch.tensor([[0.0001,0.0001],[0.0001,0.0001]])
#     model = GradientModification_v2(epi=0.001, mul_me_when_g_too_small=10)

#     loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
#     optimizer = torch.optim.SGD([input], lr=0.1)
#     for epoch in range(1):
#         model.train()
#         pred = model(input)
#         loss = loss_function(pred, target)
#         the grad input for the backward of first 2 elements backward is -2*0.0001/4. . It's smaller than 0.001. Then it's mul ed by 10.
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.param_groups[0]["lr"] = 0.01
#         print(torch.tensor([[0.0001,0.0002],[0.1,0.2]]))
#         print(input.grad, "should be -0.0024")

#         optimizer.step()
#         print(input, "should be 0.00024")
        
#         model.eval()
#         pass


# '''a batch has big grad, the other one has small grad.'''
# input = torch.tensor([[0.,0.],[0.,0.]], requires_grad=True)
# target = torch.tensor([[0.0001,0.0001],[0.1,0.1]])
# model = GradientModification_v2(epi=0.001, mul_me_when_g_too_small=0.1)

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     the grad input for the backward of first 2 elements backward is -2*0.0001/4. . It's smaller than 0.001. Then it's mul ed by 10.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.param_groups[0]["lr"] = 0.01
#     print(torch.tensor([[0.0001,0.0002],[0.1,0.2]]))
#     print(input.grad, "first 2 should be -0.0005, last 2 are -0.7")

#     optimizer.step()
#     print(input, "should be 0.00005, last 2 are 0.07")
    
#     model.eval()
#     pass

# fds=432













