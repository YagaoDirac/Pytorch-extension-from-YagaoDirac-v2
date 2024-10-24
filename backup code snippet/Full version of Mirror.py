''' This file is not temparorily completely testes.
The Mirror looks like useless. I'm so sad.

Anyway, the only useful is the GradientModification layer.



All the main parts are probably corrected. But all the examples and 
integrated tests are nor validated after the latest update. 
Any of them may be wrong, or not work. But you still can use them as tutorials.

1 extra info I would like to provide. 
Sigmoid may have a chance to work in this project.
If:
x = sigmoid(x)
x = gramo(x)
x -= 0.5
x *= 5. (or 3. to 10.)
x = w*x+b ( or mirror style)
x = next sigmoid(x)
Notice that, 
x -= 0.5
x *= 5. (or 3. to 10.)
are not used in old tests.
I got new inspiration from my next project of DDNN.
I plan to come back and redo all the tests in late jul or aug.
'''


'''
Known issue.

Idk if this is a pytorch bug, or intentional.
If a nn.Parameter doesn't need grad, the code can only be:
p = torch.nn.Parameter(...)
p = p.requires_grad_(False)
If you directly specify the require_grad in the constructor function call, it's ignored.

When using customized backward function, 
2 ways are accepted, the 1st is :
loss = loss_func(some param, torch.tensor(some data, requires_grad = True))
The 2dn way is :
target = torch.tensor(some data)
loss = loss_func(some param, target)
By default, torch.tensor set the requires_grad to False. 
So, some black magic happens behind the scene.

To access the learning rate, the code looks like:
optimizer.param_groups[0]["lr"]
You can also set to this member to change it directly. I believe it's safe.
Most of the basic test are done with SGD for simplicity.

Search for convert_to_plain_Linear function.
This function provides a equivalence plain Linear layer( or the Dense layer in Tensorflow).
You basically want to convert the mirror (both w/ and w/o the gramo enhancement) to a plain Linear one before you do the ONNX part.
The reason is, a mirror is basically 2 Linear, but does the same thing.
Mirror is only good for training, but 1x slower in predicting while needs 1x more memory.
'''

'''
Known conclusions:

I'm not after the latest researchment for a while. 
At least several years ago, "assigning different lr for different layers" was an open question.
I believe we can close it now. It's very helpful, and powerful. 
Not only the lr should be modified for each layer, the relationship between the strength of grad of weight and bias should also
 be haddled carefully. If the updating strength of bias is greater then that one of the weight, the model doesn't train in a 
 proper way. So, it helps a lot to check all the inner param and their grad. If this issue happens, the output looks like a 
 torch.ones_like(...)*some value, for instance[0.123, 0.123, 0.123...](it's repeating). It's supposed to be something like 
 [0.123, 0.321, 0.6969, 0.4242...]

I don't have time for any further test. Here's an untested workflow.
1) Write the class (torch.nn.Module). Train it a bit.
2) Add a callback function to modify the bias.grad before optimizer.step().
3) Check if the output is repeating the same number for different input, if so, bias.grad*=0.1 with the callback function in the
 last step. Or maybe in some cases, it's *=0.01. Now the model should output differently for different input.
Get weight, weight.grad, bias, bias.grad. Calc absolute value, sort, remove the smallest 20%, calc the log, then avg.
Now you should get 4 numbers. Let's call them w,wg,b,bg. The relationship should be wg/w == bg/b, and for different layers, 
 the wg/w should be the same.
Now get back to the definition of model. With gramo or mig, you can use the scaling_ratio param when create a gramo or mig to
 align the wg/w between layers. Usually, the layers in the mid are very similar at wg/w, so you can use them as reference 
 safely. Then the callback function you created in step 2 can help aligning the bg/b to the layer's wg/w.
End)
This can be seen in the last test. I did it manually. But I think it's time to write some code to automate it.




For all test:
Aligning the avg length of gradients for each layer is optional, it helps in some cases. 
In all the useful test, I only carefully aligned it for the last test.

When creating any gramo layer, set the scaling_ratio param. You can use the result in a recent paper from XAI.
Idk if I should call it scaling ratio or factor. Maybe factor is a better name.

In most cases:
Sigmoid doesn't work when the test is scaled. It works for small test but it's too small to be any useful. 
With sigmoid, the loss value also drops, but the output doesn't look correct in some cases. 
With what I got from all the test, relu basically works better.

This file contains 3 building blocks for deep learning.
They are completely different from resnet. 
They guarantee the trainability in 2 ways, the 3rd one is a combination of previous 2.
According to the test, the combination doesn't look better than the second one. Idk if I messed with any test.


Gramo(GRA-dient MO-dificatino layer):
(Notice, by gramo, I mean the only gramo alone. The combination is explained below in another section.)
(Notice 2, this layer by itself doesn't learn anything. It feels like an activation function, but it doesn't do anything
 in the forward path.)
This is the first building block in this file.
It normalize the grad for each batch. When the length of grad is too small, it simply scale it a bit but if
 I set correctly, the result for super small input is always smaller than 1. Then right before the return
 directive, the grad is multiplied by scaling_ratio.
This layer is supposed to be used right after any learnable layer, for inst Linear, Conv.
Since this layer protects the grad through the model while it doesn't touch the foreward path, it helps with
 training super deep models. I simply stacked 10 Linear/sigmoid and in another test 10 Linear/relu, and
 it trains very quickly. Which is already much better, if the tests were correct enough.

Linear Gramo Sigmoid/ReLU:
I tested this one with Linear(or Dense in tensorflow). The linear,gramo,sigmoid works fine.
Sigmoid softly mutes some neutrons behind it much more heavy than others, while gramo doesn't help in this case.
If I want to solve this problem, I probably need to pow(g,0.1), then normalize it, then pow(g,10) back.
Or maybe some new activation function like x/(1+x), or x*x/(1+x*x) to soften the grad. But this new
 activation is not widely tested. I don't plan to do this now.
Relu mutes neutrons hard. If too many are muted, the model doesn't train properly. Sometimes it needs the
 initialization to be good enough. But generally, according to my test, if the *input and model* are 
  both wide enough, it trains properly. The linear,gramo,relu doesn't work at width of 4.
Since the width of input is also counted as the width of the model, we really need a way to widen the input(duplicating may help)
 to make it work with relu.


Mirror(The magic Mirror):
Notice. Before this version, it works like 0.5*(w*(x+delta)+w*(x-delta))+b. But now, it works like half_w*(x+delta)+half_w*(x-delta)+b.
So, maybe I should name it in other ways.
The idea is that, the grad for w is g_in*x. If g_in is always in useful range, but there's absolutely
 no way to guaruntee all the x are in useful range. When the x is too close to 0, the w gets no useful grad.
To solve this problem, the simpliest way is to shift x a bit. When x is too close to 0(bad), x+delta is 
 basically at a distance of delta from 0(good). But this breaks the forward path. The formula shown above
  solved the problem. 2 w are different in code, 2 half_w are also different from each other in code. So
   at least one of them gets a useful grad.
When x is outside [-delta, delta], it's the same as a plain Linear. When x is in that range, it looks like
 the g is delta*2.. I use a 0.5 for delta so to align the learning strength to plain Linear.
This tool replaces the Linear(or Dense in Tensorflow). 

Mirror Sigmoid/ReLU:
Relu works. The output looks very correct.
(Sigmoid doesn't. The loss value decreases a bit but the outputs don't look correct.)
I remember the 10 layer test(with relu) only comsumed 3000 epochs, which is almost 10x faster than the combination.


The last one, the combination of the previous 2.
The MIG(MI-rror with G-ramo)(HELP ME NAME IT)
It looks slower than the previous one. 
But it trains almost the same speed with 10 layers as with 7 layers. 
Maybe we can stack much more layers directly and expect it to train at a decent speed.

'''

'''
About the test:
Notice: I only have 1 pcs of 1660(6G).
All the test are done with width 64 for most layers.

test_stacked_Linear_gramo_3:
It's a stacked Linear with gramo test.
I didn't push it to the extreme. I believe the result is good enough.
For no more than 6 layers, it's basically an instant training.
The test for <=6 layers was on cpu, then I move it to gpu for bigger test afterward.

test_multi_mirror__relu:
It's a stacked Mirror test.
For no more than 8 layers, it's basically within 1000 epochs.
Even for 10 layers, it's about 2000 epochs to train it.
The result is very crazy.
All the test from 3 layer to 10 layers are all on cpu.

test_multi_MIG__relu:
It's the final test of the combination.
It takes more epochs than the 2nd one. 

Maybe the Mirror withOUT gramo is the best.. I expect the last one to be the best, but the test almost drove me crazy.
'''


from typing import Any, List, Optional, Self
import torch
import math

__all__ = [
    'make_grad_noisy',
    'GradientModification',
    'MirrorLayer',
    'MirrorWithGramo',
    'GradientModificationFunction', #Should I expose this?
    'Linear_with_gramo', #Should I rename this one? Or somebody help me with the naming?
    ]


def make_grad_noisy(model:torch.nn.Module, noise_base:float = 1.2):
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



#This one is the old version. I believe it's mathmatically wrong.
# class GradientModificationFunction(torch.autograd.Function):
#     r'''input param list:
#     x:torch.Tensor,(must be set as require_grad = True)
#     learning_rate:float, 
#     epi=torch.tensor([1e-12]),
#     div_me_when_g_too_small = torch.tensor([1e3])
    
#     retur type: torch.Tensor
#     '''
#     @staticmethod
#     def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
#     #def forward(ctx: Any, x:torch.Tensor, learning_rate:float, epi=torch.tensor([1e-12]), \
#     #                 div_me_when_g_too_small = torch.tensor([1e3]), *args: Any, **kwargs: Any)->Any:
#         #I tried to write like:
#         #def forward(ctx, x:torch.Tensor, learning_rate:float, epi=torch.tensor([1e-12]), \
#         #             div_me_when_g_too_small = torch.tensor([1e3]))->torch.Tensor:
#         #but python grammar punched me.

#         #重要，如果输入的x 的 require_grad是false，就会出问题。报错是，什么，0号arg不需要grad，也没有grad fn。
#         x = args[0]
#         learning_rate = args[1]
#         epi = args[2]
#         div_me_when_g_too_small = args[3]

#         ctx.save_for_backward(learning_rate, epi, div_me_when_g_too_small)
#         return x

#     @staticmethod
#     def backward(ctx, g):
#         #super().backward()
#         learning_rate:torch.Tensor
#         learning_rate, epi, div_me_when_g_too_small = ctx.saved_tensors

#         print(learning_rate, epi, div_me_when_g_too_small)

#         mean = g.mean(dim=0, keepdim=True)
#         _centralized = g - mean
#         std = _centralized.std(dim=0, unbiased=False, keepdim=True)  # unbiased = False is slightly recommended by me.
#         std_too_small = std < epi
#         std = (std - std * std_too_small) + std_too_small * (epi* div_me_when_g_too_small)
#         _normalized = _centralized / std
#         if learning_rate != 1.:
#             return _normalized * learning_rate#, None, None, None
#         else:
#             return _normalized#, None, None, None
#         pass
#     pass  # class



#the old one is from mid in 2022
#the new one is from early in 2024.

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


# '''dtype adaption.'''
# scaling_ratio = torch.tensor([1.], dtype=torch.float64)
# epi=torch.tensor([1e-5], dtype=torch.float32)
# div_me_when_g_too_small = torch.tensor([1e-3], dtype=torch.float16)
# a = torch.tensor([[0.]], requires_grad=True, dtype=torch.float16)
# original_dtype = a.dtype
# b = GradientModificationFunction.apply(a,scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[1.]], dtype=torch.float16)
# torch.autograd.backward(b, g_in,inputs= a)
# #print(g[0])
# print(a.grad.dtype, "should be ", original_dtype)
# fds=432

# '''when g_in is too small.'''
# scaling_ratio = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# a = torch.tensor([[0.]], requires_grad=True)
# b = GradientModificationFunction.apply(a,scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.0000012]])
# torch.autograd.backward(b, g_in,inputs= a)
# #print(g[0])
# print(a.grad, "should be ", g_in/div_me_when_g_too_small)

# '''when g_in is NOT too small.'''
# scaling_ratio = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 1.")
# fds=432

# '''The shape is [batches, inside a batch]. Computation is limited inside each batch.'''
# scaling_ratio = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.7, 0.7")

# input = torch.tensor([[0.], [0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# g_in = torch.tensor([[0.12], [0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 1., 1.")

# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# g_in = torch.tensor([[0.1, 0.173], [0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.5, 0.86, 0.7, 0.7")
# fds=432

# '''Other 3 input besides the main input x.'''
# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# #output = GradientModificationFunction.apply(input, scaling_ratio,     epi,                  div_me_when_g_too_small)
# output = GradientModificationFunction.apply(input, torch.tensor([1.]), torch.tensor([0.001]),torch.tensor([0.1]))
# g_in = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.001, 0.001, 0.7, 0.7")

# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# #output = GradientModificationFunction.apply(input, scaling_ratio,     epi,                  div_me_when_g_too_small)
# output = GradientModificationFunction.apply(input, torch.tensor([2.]), torch.tensor([0.001]),torch.tensor([0.1]))
# g_in = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.002, 0.002, 1.4, 1.4")
# fds=432





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
        self.scaling_ratio.requires_grad_(False)
        self.epi=torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
        self.epi.requires_grad_(False)
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small]), requires_grad=False)
        self.div_me_when_g_too_small.requires_grad_(False)
        #raise Exception("untested")
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

# '''all the setters'''
# model = GradientModification()
# print(model.scaling_ratio.requires_grad, "should be False")
# print(model.epi.requires_grad, "should be False")
# print(model.div_me_when_g_too_small.requires_grad, "should be False")
# model.set_scaling_ratio(0.123)
# print(model.scaling_ratio, "should be 0.123")
# print(model.scaling_ratio.requires_grad, "should be False")
# model.set_epi(0.234)
# print(model.epi, "should be 0.234")
# print(model.epi.requires_grad, "should be False")
# model.set_div_me_when_g_too_small(0.345)
# print(model.div_me_when_g_too_small, "should be 0.345")
# print(model.div_me_when_g_too_small.requires_grad, "should be False")
# fds=432


# '''dtype adaption.'''
# input = torch.tensor([[1.]], requires_grad=True)
# target = torch.tensor([[0.]])
# model = GradientModification()
# model.to(torch.float64)
# #model.to(torch.float16)

# loss_function = torch.nn.L1Loss()# the L1Loss function only provides the direction. It's the dirivitive of abs.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     print(pred.dtype, "pred.dtype should be f32")
#     loss = loss_function(pred, target)
#     print(loss.dtype, "loss.dtype should be f32")
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     print(input.grad, "should be 1.")
#     print(input.grad.dtype, "input.grad.dtype should be f32")

#     optimizer.step()
#     print(input, "should be 0.9")
    
#     model.eval()
#     pass
# fds=432


# '''when grad is too small.'''
# input = torch.tensor([[0.]], requires_grad=True)
# target = torch.tensor([[0.00012]])
# model = GradientModification(epi=0.001,div_me_when_g_too_small=0.1)###

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     #the grad input for the backward is -2*0.002. It's small enough to be *10
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     print(input.grad, "should be -0.0024")

#     optimizer.step()
#     print(input, "should be 0.00024")
    
#     model.eval()
#     pass
# fds=432


# '''same test but dim is different.'''
# input = torch.tensor([[0.,0.]], requires_grad=True)
# target =  torch.tensor([[0.00012,0.00012]])
# model = GradientModification(epi=0.001,div_me_when_g_too_small=0.1)###

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     #the grad input for the backward is -2*0.002/2. . It's small enough to be *10
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     print(input.grad, "should be -0.0012")

#     optimizer.step()
#     print(input, "should be 0.00012")
    
#     model.eval()
#     pass


# '''same test but dim is different.'''
# input = torch.tensor([[0., 0., 0.]], requires_grad=True)
# target = torch.tensor([[0.00012, 0.00012, 0.00012]])
# model = GradientModification(epi=0.001,div_me_when_g_too_small=0.1)###

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     #the grad input for the backward is -2*0.00012/3. . It's small enough to be *10
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     print(input.grad, "should be -0.0008")

#     optimizer.step()
#     print(input, "should be 0.00008")
    
#     model.eval()
#     pass


# '''same test but dim is different.'''
# input = torch.tensor([[0.,0.],[0.,0.]], requires_grad=True)
# target = torch.tensor([[0.0001,0.0001],[0.0001,0.0001]])
# model = GradientModification(epi=0.001, div_me_when_g_too_small=0.1)

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     #the grad input for the backward of first 2 elements backward is -2*0.0001/4. . It's smaller than 0.001. Then it's mul ed by 10.
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     #print(torch.tensor([[0.0001,0.0002],[0.1,0.2]]))
#     print(input.grad, "should be -0.0005")

#     optimizer.step()
#     print(input, "should be 0.00005")
    
#     model.eval()
#     pass


# '''a batch has big grad, the other one has small grad.'''
# input = torch.tensor([[0.,0.],[0.,0.]], requires_grad=True)
# target = torch.tensor([[0.0001,0.0001],[0.1,0.1]])
# model = GradientModification(epi=0.001, div_me_when_g_too_small=0.1)

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     #the grad input for the backward of first 2 elements backward is -2*0.0001/4. . It's smaller than 0.001. Then it's mul ed by 10.
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     #print(torch.tensor([[0.0001,0.0002],[0.1,0.2]]))
#     print(input.grad, "first 2 should be -0.0005, last 2 are -0.7")

#     optimizer.step()
#     print(input, "should be 0.00005, last 2 are 0.07")
    
#     model.eval()
#     pass

# fds=432





# This is actually test code. But it looks very useful... 
# I should say, don't use this one. A better one is at the end. You'll love it.
class Linear_with_gramo(torch.nn.Module):
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
    

# '''basic test'''
# model = Linear_with_gramo(1,1,False)
# model.linear.weight = torch.nn.Parameter(torch.ones_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# target = torch.tensor([[0.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# ### model.set_learning_rate(optimizer.param_groups[0]["lr"])

# #for epoch in range(1):
# #model.train()
# pred = model(input)# w = x = 1.. No b or b = 0..
# loss = loss_function(pred, target)
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be 1.")

# optimizer.step()
# print(model.linear.weight, "should be 0.9")
# fds=432


# '''safe test. safe to skip'''
# model = Linear_with_gramo(1,1,False)
# model.linear.weight = torch.nn.Parameter(torch.zeros_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# target = torch.tensor([[1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# #for epoch in range(1):
# pred = model(input)# w = 0., x = 1.. b = 0..target = 1.
# loss = loss_function(pred, target)
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be -1.")

# optimizer.step()
# print(model.linear.weight, "should be 0.1")
# fds=432


# '''2 same batches work like 2 epochs.'''
# model = Linear_with_gramo(1,1,False)
# model.linear.weight = torch.nn.Parameter(torch.zeros_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.],[1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# #for epoch in range(1):
# pred = model(input)# x = 1.. w = b = 0..
# loss = loss_function(pred, torch.tensor([[1.],[1.]], requires_grad=True))
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be -1-1 == -2")

# optimizer.step()
# print(model.linear.weight, "should be 0.2")
# fds=432


# '''1 batch, but more dimentions in it.'''
# model = Linear_with_gramo(2,1,False)
# model.linear.weight = torch.nn.Parameter(torch.zeros_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1., 1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# #for epoch in range(1):
# pred = model(input)# x = 1.. w = b = 0..
# loss = loss_function(pred, torch.tensor([[1.]], requires_grad=True))
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be both -1")

# optimizer.step()
# print(model.linear.weight, "should be 0.1")
# fds=432


# '''Only the out_features(out dimention) affects the grad.'''
# in_features = 4#this one doesn't affect the result.
# out_features = 3# this one does.
# lr=0.1
# model = Linear_with_gramo(in_features, out_features,False)
# model.linear.weight = torch.nn.Parameter(torch.zeros_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.ones([1, in_features,])
# target = torch.ones([1, out_features,])
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# #for epoch in range(1):
# pred = model(input)# x = 1.. w = b = 0..
# loss = loss_function(pred, target)
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, f"should be all -{math.sqrt(1/out_features)}")

# optimizer.step()
# print(model.linear.weight, f"should be {lr*math.sqrt(1/out_features)}0.07")
# fds=432


# '''Gramo doesn't help when any element of x(the input) is 0. 
# The corresponding elements in W doesn't get any non-0 grad.
# After this test, let me introduce the solution for this case.'''
# model = Linear_with_gramo(1,1,False)
# model.linear.weight = torch.nn.Parameter(torch.ones_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[0.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# #for epoch in range(1):
# pred = model(input)# x = 0.. w = 1., b = 0..
# loss = loss_function(pred, torch.tensor([[1., 1.]], requires_grad=True))
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be all 0")

# print(model.linear.weight, "should be the same")
# optimizer.step()
# print(model.linear.weight, "should be the same")
# fds=432



# 继续
# 继续
# 继续
# 继续
# 继续
# 继续
# 继续
# 继续
# 继续
# 继续
# 继续
# 继续
# class example_Linear_gramo_sigmoid_layer(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.width_inside = 64
#         self.linear = torch.nn.Linear(in_features, self.width_inside, True, device, dtype)
#         self.gramo = GradientModification(scaling_ratio=0.1)
#         self.sigmoid = torch.nn.Sigmoid()
#     def forward(self, x:torch.tensor)->torch.Tensor:
#         x = self.linear(x)
#         x = self.gramo(x)
#         x = self.sigmoid(x)
#         return x





#redo this one
# model = test_stacked_Linear_gramo__sigmoid(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# target = torch.tensor([[0.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(10):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()
#     print(loss)
#     if True:
#             print("pred  ", pred.T)
# print("target", target.T)

# jklfds =432




# in_feature = 16
# model = test_stacked_Linear_gramo__sigmoid(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 4
# input = torch.rand([data_amount, in_feature])+0.3
# #target = torch.rand([data_amount, 1])+0.1
# target = (input.pow(1.5).sum(dim=1)*0.05+0.).unsqueeze(1)
# # print(input)
# # print(target.shape)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# iter_per_print = 1
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         # print(model.linear1.weight.grad[0:1])
#         # print(model.linear2.weight.grad[0:1])
#         # print(model.linear3.weight.grad[0:1])
#         # print(model.linear3.bias.grad[0:1])
#         # print(model.linear3.weight[0:1])
#         # print(model.linear3.bias[0:1])
#         # print("--------------")
#         if True:
#             print("pred  ", pred.T)
# print("target", target.T)

# fdsfds = 654456456



# '''
# A not very comfirmed comclusion. Since relu mutes part of the neutrons behind it, this gramo doesn't help in such cases.
# I believe more data helps, but according to my test, gramo seems to never help with muted neutrons. 
# '''

# class test_stacked_Linear_gramo__relu(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.width_inside = 64
#         self.linear1 = torch.nn.Linear(in_features, self.width_inside, True, device, dtype)
#         self.linear2 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear3 = torch.nn.Linear(self.width_inside, out_features, True, device, dtype)

#         self.gramo1 = GradientModification(scaling_ratio=0.01)
#         self.gramo2 = GradientModification(scaling_ratio=0.01)
#         self.gramo3 = GradientModification(scaling_ratio=0.01)

#         self.relu1 = torch.nn.ReLU()
#         self.relu2 = torch.nn.ReLU()

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         x = self.linear1(x)
#         x = self.gramo1(x)
#         x = self.relu1(x)
#         x = self.linear2(x)
#         x = self.gramo2(x)
#         x = self.relu2(x)
#         x = self.linear3(x)
#         x = self.gramo3(x)
#         return x


# in_feature = 64
# model = test_stacked_Linear_gramo__relu(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 10
# input = torch.rand([data_amount, in_feature])+0.3
# # target = torch.rand([data_amount, 1])+0.1
# target = ((input.pow(1.5)-input.pow(2.5)).sum(dim=1)).unsqueeze(1)
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# iter_per_print = 20
# print_count = 5
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if False:
#             print(model.linear1.weight.grad[0:2])
#             print(model.linear2.weight.grad[0:2])
#             print(model.linear3.weight.grad[0:2])
#             print("--------------")
#         if True:
#             print("pred  ", pred.T)
# print("target", target.T)

# jkldfs=345789




# '''this one doesn't look work, but it actually works. 
# It converges super quickly.
# '''

# class test_stacked_Linear_gramo_2(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.width_inside = 64
#         self.linear1  = torch.nn.Linear(in_features, self.width_inside, True, device, dtype)
#         self.linear2  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear3  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear4  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear5  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear6  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear7  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear8  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear9  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear10 = torch.nn.Linear(self.width_inside, out_features, True, device, dtype)

#         self.epi = 0.0000001
#         self.div_me_when_ = 0.000005
#         self.gramo1  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo2  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo3  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo4  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo5  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo6  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo7  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo8  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo9  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo10 = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)


#         # self.gramo1  = GradientModification(scaling_ratio=in_features/self.width_inside)
#         # self.gramo1  = GradientModification(scaling_ratio=0.2*math.sqrt(self.width_inside/in_feature))
#         # self.gramo1  = GradientModification(scaling_ratio=math.pow(self.width_inside, 0.25)/math.sqrt(in_feature))
#         # #ok, I'm giving up with this formula.
#         # scaling_ratio_for_mid = math.pow(self.width_inside, 0.25)/math.sqrt(self.width_inside)
#         # self.gramo2  = GradientModification(scaling_ratio=scaling_ratio_for_mid)
#         # self.gramo3  = GradientModification(scaling_ratio=scaling_ratio_for_mid)
#         # self.gramo4  = GradientModification(scaling_ratio=scaling_ratio_for_mid)
#         # self.gramo5  = GradientModification(scaling_ratio=scaling_ratio_for_mid)
#         # self.gramo6  = GradientModification(scaling_ratio=scaling_ratio_for_mid)
#         # self.gramo7  = GradientModification(scaling_ratio=scaling_ratio_for_mid)
#         # self.gramo8  = GradientModification(scaling_ratio=scaling_ratio_for_mid)
#         # self.gramo9  = GradientModification(scaling_ratio=scaling_ratio_for_mid)
#         # self.gramo10 = GradientModification(scaling_ratio=out_features/self.width_inside)

#         if True:
#             self.act1  = torch.nn.Sigmoid()
#             self.act2  = torch.nn.Sigmoid()
#             self.act3  = torch.nn.Sigmoid()
#             self.act4  = torch.nn.Sigmoid()
#             self.act5  = torch.nn.Sigmoid()
#             self.act6  = torch.nn.Sigmoid()
#             self.act7  = torch.nn.Sigmoid()
#             self.act8  = torch.nn.Sigmoid()
#             self.act9  = torch.nn.Sigmoid()

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         x = self.linear1(x)
#         x = self.gramo1(x)
#         x = self.act1(x)
#         x = self.linear2(x)
#         x = self.gramo2(x)
#         x = self.act2 (x)
#         x = self.linear3(x)
#         x = self.gramo3(x)
#         x = self.act3 (x)
#         x = self.linear4(x)
#         x = self.gramo4(x)
#         x = self.act4 (x)
#         x = self.linear5(x)
#         x = self.gramo5(x)
#         x = self.act5 (x)
#         x = self.linear6(x)
#         x = self.gramo6(x)
#         x = self.act6 (x)
#         x = self.linear7(x)
#         x = self.gramo7(x)
#         x = self.act7 (x)
#         x = self.linear8(x)
#         x = self.gramo8(x)
#         if True:#inside this, they should be act->linear->gramo
#             x = self.act8 (x)
#             x = self.linear9(x)
#             x = self.gramo9(x)
#     # according to my test, if you return x here, it's not gonna meet the epi thing.
#     # but I know you are gonna push this toy to the extreme. If you add the 10th layer, you can modify the epi to a smaller number.
#     # And you maybe want to test this toy at float64. You know wat i mean, lol.
#         x = self.act9 (x)
#         x = self.linear10(x)
#         x = self.gramo10(x)
#         return x#Hey, move it here.


# in_feature = 64
# model = test_stacked_Linear_gramo_2(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 20
# input = torch.rand([data_amount, in_feature])+0.3
# #target = input.pow(1.5)
# target = torch.rand([data_amount, 1])+0.1
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# iter_per_print = 1
# print_count = 5
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if False:
#             print(model.linear1.weight.grad[0:1])
#             print(model.linear2.weight.grad[0:1])
#             # print(model.linear3.weight.grad[0:1])
#             # print(model.linear4.weight.grad[0:1])
#             # print(model.linear5.weight.grad[0:1])
#             # print(model.linear6.weight.grad[0:1])
#             # print(model.linear7.weight.grad[0:1])
#             # print(model.linear8.weight.grad[0:1])
#             # print(model.linear9.weight.grad[0:1])
#             print(model.linear10.weight.grad[0:1])
#             print("--------------")

# jkldfs=345789






# class test_stacked_Linear_gramo_3(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.width_inside = 64
#         self.linear1  = torch.nn.Linear(in_features, self.width_inside, True, device, dtype)
#         self.linear2  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear3  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear4  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear5  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear6  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear7  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear8  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear9  = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear10 = torch.nn.Linear(self.width_inside, out_features, True, device, dtype)


#         self.epi = 0.00001#0.0000001
#         self.div_me_when_ = 0.001#0.000001
#         self.gramo1  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo2  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo3  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo4  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo5  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo6  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo7  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo8  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo9  = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)
#         self.gramo10 = GradientModification(epi=self.epi, div_me_when_g_too_small=self.div_me_when_)


#         if True:
#             self.act1  = torch.nn.ReLU()
#             self.act2  = torch.nn.ReLU()
#             self.act3  = torch.nn.ReLU()
#             self.act4  = torch.nn.ReLU()
#             self.act5  = torch.nn.ReLU()
#             self.act6  = torch.nn.ReLU()
#             self.act7  = torch.nn.ReLU()
#             self.act8  = torch.nn.ReLU()
#             self.act9  = torch.nn.ReLU()

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         a_small_number = 0.1#if you set this to like 0.001, the model doesn't work properly.
#         x = self.linear1(x)
#         x = self.gramo1(x)
#         x = self.act1(x)
#         x = x*a_small_number
#         x = self.linear2(x)
#         x = self.gramo2(x)
#         x = self.act2 (x)
#         x = x*a_small_number
#         x = self.linear3(x)
#         x = self.gramo3(x)
#         x = self.act3 (x)
#         x = x*a_small_number
#         x = self.linear4(x)
#         x = self.gramo4(x)
#         x = self.act4 (x)
#         x = x*a_small_number
#         x = self.linear5(x)
#         x = self.gramo5(x)
#         x = self.act5 (x)
#         x = x*a_small_number
#         x = self.linear6(x)
#         x = self.gramo6(x)
#         x = self.act6 (x)
#         x = x*a_small_number
#         x = self.linear7(x)
#         x = self.gramo7(x)
#         x = self.act7 (x)
#         x = x*a_small_number
#         return self.gramo10(self.linear10(x))#move this line.
#         x = self.linear8(x)
#         x = self.gramo8(x)
#         if True:#inside this, they should be act->linear->gramo
#             x = self.act8 (x)
#             x = x*a_small_number
#             x = self.linear9(x)
#             x = self.gramo9(x)
#     # according to my test, if you return x here, it's not gonna meet the epi thing.
#     # but I know you are gonna push this toy to the extreme. If you add the 10th layer, you can modify the epi to a smaller number.
#     # And you maybe want to test this toy at float64. You know wat i mean, lol.
#         x = self.act9 (x)
#         x = x*a_small_number
#         pass




# in_feature = 64
# model = test_stacked_Linear_gramo_3(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 10
# input = torch.rand([data_amount, in_feature])+0.3
# #target = input.pow(1.5)
# #target = torch.rand([data_amount, 1])+0.1
# target = ((input.pow(1.5)-input.pow(2.5)).sum(dim=1)).unsqueeze(1)
# # print(input)
# # print(target)
# #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#for 5 linear
# #optimizer = torch.optim.SGD(model.parameters(), lr=0.002)#for 7 linear
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)#for ? linear

# iter_per_print = 2572
# print_count = 5
# #1000 epochs lr0.01 works for 5 linear.
# #15000 epochs lr0.002 works for 7 linear.
# #20000 epochs lr0.0005 works for 8 linear.
# model = model.cuda()
# input = input.cuda()
# target = target.cuda()
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     if model.linear10.bias.grad is not None:
#         model.linear10.bias.grad*=0.02

#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if False:
#             print(model.linear1.weight.grad[0:1])
#             print(model.linear2.weight.grad[0:1])
#             print(model.linear3.weight.grad[0:1])
#             print(model.linear4.weight.grad[0:1])
#             print(model.linear5.weight.grad[0:1])
#             print(model.linear6.weight.grad[0:1])
#             print(model.linear7.weight.grad[0:1])
#             print(model.linear8.weight.grad[0:1])
#             print(model.linear9.weight.grad[0:1])
#             print(model.linear10.weight.grad[0:1])
#             print("--------------")
#         # print(model.linear9.bias[0:3])
#         # print(model.linear9.bias.grad[0:3])
#         # print(model.linear10.bias[0:1])
#         # print(model.linear10.bias.grad[0:1])

#         if True:
#             print("pred  ", pred.T)
#             print("target", target.T)
# jkldfs=345789







# Gramo part ends.
# Now it's Mirror part.
    






# Again, I believe this version is mathmatically wrong. New version below.

# '''I copied the torch.nn.Linear code and modified it.
# '''

# class MirrorLayer(torch.nn.Module):
#     r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
#     To access the learning rate, you usually need some thing like:
#     lr:float = optimizer.param_groups[0]["lr"]

#     check torch.nn.Linear for other help
#     """
#     __constants__ = ['in_features', 'out_features', 'auto_merge_duration']
#     in_features: int
#     out_features: int
#     half_weight: torch.Tensor

#     auto_merge_duration:int
#     update_count:int
#     learning_rate:float

#     def __init__(self, in_features: int, out_features: int, bias: bool = True,
#                  device=None, dtype=None, auto_merge_duration:int = 20) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.half_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
#         self.half_weight_mirrored = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
#         if bias:
#             self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#         #to keep track of the training.
#         self.auto_merge_duration:int = auto_merge_duration
#         self.update_count:int = 0

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         torch.nn.init.kaiming_uniform_(self.half_weight, a=math.sqrt(5))
#         with torch.no_grad():
#             self.half_weight_mirrored.copy_(self.half_weight)

#         if self.bias is not None:
#             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.half_weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             torch.nn.init.uniform_(self.bias, -bound, bound)

#     def set_learning_rate(self, learning_rate:float)->None:
#         self.learning_rate = learning_rate

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         if not hasattr(self, "learning_rate"):
#             raise Exception("Assign the learning rate for this layer. \n The code is like:\nmodel object var name.set_learning_rate(optimizer.param_groups[0][\"lr\"])")
        
#         if self.update_count>=self.auto_merge_duration:
#             self.update_count = 0
#             with torch.no_grad():
#                 self.half_weight = (self.half_weight+self.half_weight_mirrored)/2.
#                 self.half_weight_mirrored.copy_(self.half_weight)
#                 pass
#             pass

#         head1:torch.Tensor = torch.nn.functional.linear(input+self.learning_rate, self.half_weight)
#         head2:torch.Tensor = torch.nn.functional.linear(input-self.learning_rate, self.half_weight, self.bias)
#         return head1+head2

#     def extra_repr(self) -> str:
#         return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

#     def convert_to_plain_Linear(self)->torch.nn.Linear:
#         has_bias = bool(self.bias)
#         result:torch.nn.Linear = torch.nn.Linear(self.in_features, self.out_features, has_bias)
#         result.weight = self.half_weight+self.half_weight_mirrored
#         result.bias = self.bias
#         return result
#     pass











#let me comment this part out for the test.
#If this part is not uncommended, tell me to fix it.





# # Here's the new version.
# '''I copied the torch.nn.Linear code and modified it.
# '''

class MirrorLayer(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]

    check torch.nn.Linear for other help
    """
    __constants__ = ['in_features', 'out_features', 'auto_merge_duration']
    in_features: int
    out_features: int
    
    half_weight: torch.nn.Parameter
    half_weight_mirrored: torch.nn.Parameter
    bias: Optional[torch.nn.Parameter]
    
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
        if self.update_count<self.auto_merge_duration:
            self.update_count += 1
        else:
            self.update_count = 0
            with torch.no_grad():
                
                # print("--------------")
                # print(self.half_weight)
                # print(self.half_weight.dtype)
                # print(self.half_weight_mirrored)
                # print(self.half_weight.grad)
                # print(self.half_weight_mirrored.grad)
                grad1 = self.half_weight.grad
                grad2 = self.half_weight_mirrored.grad
                if self.bias:
                    #bias_data = self.bias.data
                    self.bias.data = self.half_weight-self.half_weight_mirrored+self.bias.data
                    pass
                self.half_weight = self.half_weight.requires_grad_(False)
                self.half_weight_mirrored = self.half_weight_mirrored.requires_grad_(False)
                
                self.half_weight.data = (self.half_weight.data+self.half_weight_mirrored.data)/2.
                self.half_weight_mirrored.copy_(self.half_weight)

                self.half_weight = self.half_weight.requires_grad_(True)
                self.half_weight_mirrored = self.half_weight_mirrored.requires_grad_(True)
                self.half_weight.grad = grad1
                self.half_weight_mirrored.grad = grad2
                # print("--------------")
                # print(self.half_weight)
                # print(self.half_weight.dtype)
                # print(self.half_weight_mirrored)
                # print(self.half_weight.grad)
                # print(self.half_weight_mirrored.grad)
                #raise Exception ("device and dtype are not tested. It only works on cpu and f32 for now.")
                pass
            pass
            
        head1:torch.Tensor = torch.nn.functional.linear(input + 1., self.half_weight)
        head2:torch.Tensor = torch.nn.functional.linear(input - 1., self.half_weight_mirrored, self.bias)
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
    
    def copy_from_plain_Linear(self, input_linear_layer:torch.nn.Linear)->None:
        with torch.no_grad():
            self.half_weight = input_linear_layer.weight/2.
            self.half_weight_mirrored.copy_(self.half_weight)
            if input_linear_layer.bias:
                self.bias.copy_(input_linear_layer.bias)
                pass
            pass
        
    def I_tested_this_layer_with_cpu_f16(self)->str:
        return "pytorch doesn't support f16 matmul on cpu. Move the test to gpu."
        
    pass


# '''dtype test. f16+cpu+matmul is not supported. This test can only be done in gpu.'''
# model = MirrorLayer(1,1,bias=False,auto_merge_duration=0)
# model.to(torch.float16)
# model.cuda()
# input = torch.tensor([[1.]], dtype=torch.float16).cuda()
# pred = model(input)
# g_in = torch.tensor([[1.]], dtype=torch.float16).cuda()
# torch.autograd.backward(pred, g_in, inputs=next(model.parameters()))
# print("If you see this line printed, the test passed.")
# fds=432


# '''dtype test. f16+cpu+matmul is not supported. This test can only be done in gpu.'''
# model = MirrorLayer(1,1,bias=True,auto_merge_duration=0)
# with torch.no_grad():
#     model.half_weight+=1.
#     model.bias+=5.
#     pass
# print(model.half_weight)
# print(model.half_weight_mirrored)
# print(model.bias)
# input = torch.tensor([[1.]])
# pred1 = model(input)
# print(model.half_weight)
# print(model.half_weight_mirrored)
# print(model.bias)
# pred2 = model(input)
# print(pred1, "should be ", pred2)
# fds=432




# '''this test shown me why this Mirror layer is a wrong design.
# It doesn't do anything useful.

# Mirror protects grad when x(input) is very close to 0.
# But it doesn't make sure the '''
# in_feature = 1
# model = MirrorLayer(in_feature,1,bias=False,auto_merge_duration=3)
# loss_function = torch.nn.MSELoss()
# data_amount = 1
# input = torch.ones([data_amount, in_feature])*0.#as close to 0 as possible.
# target = torch.ones([data_amount, 1])
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# iter_per_print = 11
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     #print(model.update_count, "model.update_count")
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     if True and 'print the weight':
#         print(model.half_weight.data, "model.half_weight")
#         print(model.half_weight_mirrored.data, "model.half_weight_mirrored")
#         pass
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if False:
#             print(model.half_weight.grad)
#             print(model.half_weight_mirrored.grad)
#             print("--------------")
#         if True:
#             print("pred  ", pred.T)
#             print("target", target.T)
# fds=432



# '''????????????????'''
# in_feature = 4
# model = MirrorLayer(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 2
# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# iter_per_print = 200
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if False:
#             # this part is actually very interesting. If you have time, uncomment the next 2 lines.
#             # print(model.half_weight.grad)
#             # print(model.half_weight_mirrored.grad)
#             pass
#         if True:
#             print("pred  ", pred.T)
#             print("target", target.T)

# jkldfs=345789





# class test_multi_mirror__sigmoid(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.auto_merge_ = 1
#         self.mirror1 = MirrorLayer(in_features, self.width_inside,       bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror2 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror3 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror4 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror5 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror6 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror7 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror8 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror9 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror10 = MirrorLayer(self.width_inside, out_features,     bias, device, dtype, self.auto_merge_, *args, **kwargs)

#         if True:
#             self.act1  = torch.nn.Sigmoid()
#             self.act2  = torch.nn.Sigmoid()
#             self.act3  = torch.nn.Sigmoid()
#             self.act4  = torch.nn.Sigmoid()
#             self.act5  = torch.nn.Sigmoid()
#             self.act6  = torch.nn.Sigmoid()
#             self.act7  = torch.nn.Sigmoid()
#             self.act8  = torch.nn.Sigmoid()
#             self.act9  = torch.nn.Sigmoid()
#         pass

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         a_small_number :float = 0.001
#         x = self.mirror1 (x) 
#         x = self.act1(x)
#         x = x*a_small_number
#         x = self.mirror2 (x) 
#         x = self.act2(x)
#         x = x*a_small_number
#         x = self.mirror3 (x) 
#         x = self.act3(x)
#         x = x*a_small_number
#         x = self.mirror4 (x) 
#         x = self.act4(x)
#         x = x*a_small_number
#         x = self.mirror5 (x) 
#         x = self.act5(x)
#         x = x*a_small_number
#         x = self.mirror6 (x) 
#         x = self.act6(x)
#         x = x*a_small_number
#         x = self.mirror7 (x) 
#         x = self.act7(x)
#         x = x*a_small_number
#         x = self.mirror8 (x) 
#         x = self.act8(x)
#         x = x*a_small_number
#         x = self.mirror9 (x) 
#         x = self.act9(x)
#         x = x*a_small_number
#         x = self.mirror10(x) 
#         return x#move this line up and down. Each time 3 lines.




# in_feature = 64
# model = test_multi_mirror__sigmoid(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 20
# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# iter_per_print = 1
# print_count = 5
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if False:
#             # this part is actually very interesting. If you have time, uncomment this part.
#             # if model.mirror1.half_weight.grad is not None:
#             #     print(model.mirror1.half_weight.grad[0:1])
#             # if model.mirror2.half_weight.grad is not None:
#             #     print(model.mirror2.half_weight.grad[0:1])
#             # if model.mirror10.half_weight.grad is not None:
#             #     print(model.mirror10.half_weight.grad[0:1])
#             #below is another way to access the grad. It's actually safe.
#             print(model.mirror1.half_weight.grad[0:1])
#             print(model.mirror2.half_weight.grad[0:1])
#             print(model.mirror10.half_weight.grad[0:1])

#         if False:
#             print(model.mirror1.half_weight[0:1])
#             print(model.mirror2.half_weight[0:1])
#             print(model.mirror10.half_weight[0:1])
#             # print(model.half_weight_mirrored.grad)
#             pass

# jkldfs=345789






# class test_multi_mirror__relu(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.auto_merge_ = 1
#         self.mirror1 = MirrorLayer(in_features, self.width_inside,       bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror2 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror3 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror4 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror5 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror6 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror7 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror8 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror9 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror10 = MirrorLayer(self.width_inside, out_features,     bias, device, dtype, self.auto_merge_, *args, **kwargs)

#         if True:
#             self.act1  = torch.nn.ReLU()
#             self.act2  = torch.nn.ReLU()
#             self.act3  = torch.nn.ReLU()
#             self.act4  = torch.nn.ReLU()
#             self.act5  = torch.nn.ReLU()
#             self.act6  = torch.nn.ReLU()
#             self.act7  = torch.nn.ReLU()
#             self.act8  = torch.nn.ReLU()
#             self.act9  = torch.nn.ReLU()
#         pass

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         a_small_number :float = 1.00
#         x = self.mirror1 (x) 
#         x = self.act1(x)
#         x = x*a_small_number
#         x = self.mirror2 (x) 
#         x = self.act2(x)
#         x = x*a_small_number
#         x = self.mirror3 (x) 
#         x = self.act3(x)
#         x = x*a_small_number
#         x = self.mirror4 (x) 
#         x = self.act4(x)
#         x = x*a_small_number
#         x = self.mirror5 (x) 
#         x = self.act5(x)
#         x = x*a_small_number
#         x = self.mirror6 (x) 
#         x = self.act6(x)
#         x = x*a_small_number
#         x = self.mirror7 (x) 
#         x = self.act7(x)
#         x = x*a_small_number
#         x = self.mirror8 (x) 
#         x = self.act8(x)
#         x = x*a_small_number
#         x = self.mirror9 (x) 
#         x = self.act9(x)
#         x = x*a_small_number
#         return self.mirror10(x)#move this line up and down. Each time 3 lines.




# in_feature = 64
# model = test_multi_mirror__relu(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 20
# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# iter_per_print = 500
# print_count = 3
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if False:
#             #below is another way to access the grad. It's actually safe.
#             print(model.mirror1.half_weight.grad[0:1])
#             print(model.mirror2.half_weight.grad[0:1])
#             print(model.mirror10.half_weight.grad[0:1])

#         if False:
#             print(model.mirror1.half_weight[0:1])
#             print(model.mirror2.half_weight[0:1])
#             print(model.mirror10.half_weight[0:1])
#             # print(model.half_weight_mirrored.grad)
#             pass
#         if True:
#             print("pred  ", pred.T)
#             print("target", target.T)
# jkldfs=345789






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
                raise Exception ("device and dtype are not tested. It only works on cpu and f32 for now.")
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
                #继续
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





# in_feature = 1
# model = MirrorWithGramo(in_feature,1,auto_merge_duration=1)
# loss_function = torch.nn.MSELoss()
# data_amount = 1
# input = torch.ones([data_amount, in_feature])*0.00001
# target = torch.ones([data_amount, 1])*0.00001
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# iter_per_print = 1
# print_count = 5
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if False:
#             print(model.half_weight.grad, "should be 0.5 or -0.5")
#             print(model.half_weight_mirrored.grad, "should be -0.5 or 0.5")
#             # 2 grad should be opposite.
#             print("--------------")

# jkldfs=345789




# in_feature = 64
# model = MirrorWithGramo(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 20
# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

# iter_per_print = 1
# print_count = 5
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if True:
#             # this part is actually very interesting. If you have time, uncomment the next 2 lines.
#             # print(model.half_weight.grad)
#             # print(model.half_weight_mirrored.grad)
#             # the first one is always greater than the second one.
#             pass

# jkldfs=345789




# class test_multi_MIG__sigmoid(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.auto_merge_ = 1
#         self.mig1  = MirrorWithGramo(in_features, self.width_inside,       bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig2  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig3  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig4  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig5  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig6  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig7  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig8  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig9  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig10 = MirrorWithGramo(self.width_inside, out_features,     bias, device, dtype, self.auto_merge_, *args, **kwargs)

#         if True:
#             self.act1  = torch.nn.Sigmoid()
#             self.act2  = torch.nn.Sigmoid()
#             self.act3  = torch.nn.Sigmoid()
#             self.act4  = torch.nn.Sigmoid()
#             self.act5  = torch.nn.Sigmoid()
#             self.act6  = torch.nn.Sigmoid()
#             self.act7  = torch.nn.Sigmoid()
#             self.act8  = torch.nn.Sigmoid()
#             self.act9  = torch.nn.Sigmoid()
#         pass

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         x = self.mig1 (x) 
#         x = self.act1(x)
#         x = self.mig2 (x) 
#         x = self.act2(x)
#         x = self.mig3 (x) 
#         x = self.act3(x)
#         x = self.mig4 (x) 
#         x = self.act4(x)
#         x = self.mig5 (x) 
#         x = self.act5(x)
#         x = self.mig6 (x) 
#         x = self.act6(x)
#         x = self.mig7 (x) 
#         x = self.act7(x)
#         x = self.mig8 (x) 
#         x = self.act8(x)
#         x = self.mig9 (x) 
#         x = self.act9(x)
#         x = self.mig10(x) 
#         return x#move this line up and down. Each time 3 lines.




# in_feature = 64
# model = test_multi_MIG__sigmoid(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 4
# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)+0.0001
# # print(input)
# # print(target)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00002)

# iter_per_print = 10
# print_count = 5
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(loss)
#         if True:
#             print("pred", pred.T)
#             print("target", target.T)
#         if False:
#             print(model.mig1.half_weight.grad[0:1])
#             print(model.mig1.half_weight_mirrored.grad[0:1])
#             # print(model.mig2.half_weight.grad[0:1])
#             print(model.mig10.half_weight.grad[0:1])
#             print(model.mig10.half_weight_mirrored.grad[0:1])
#             pass

#         if True:
#             # print(model.mig1.half_weight[0:1])
#             # print(model.mig2.half_weight[0:1])
#             print(",,,", model.mig10.half_weight[0:1])
#             # print(model.half_weight_mirrored.grad)
#             pass

# jkldfs=345789


#it's still very wierd. The loss deceases, but the input side layers are not trained..

# Known issue. I assume the bias is always true. So, not safety check for bias == false.

# class test_multi_MIG__relu(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.auto_merge_ = 1
#         self.mig1  = MirrorWithGramo(in_features, self.width_inside,       bias, device, dtype, self.auto_merge_, 0.5,\
#                                       *args, **kwargs)
#         self.mig2  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig3  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig4  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig5  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig6  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig7  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig8  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig9  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mig10 = MirrorWithGramo(self.width_inside, out_features,     bias, device, dtype, self.auto_merge_, 0.01 \
#                                      , *args, **kwargs)

#         self.biases:List[torch.nn.Parameter] = []
       
#         self.biases.append(self.mig1.bias)
#         self.biases.append(self.mig2.bias)
#         self.biases.append(self.mig3.bias)
#         self.biases.append(self.mig4.bias)
#         self.biases.append(self.mig5.bias)
#         self.biases.append(self.mig6.bias)
#         self.biases.append(self.mig7.bias)
#         self.biases.append(self.mig8.bias)
#         self.biases.append(self.mig9.bias)
#         self.biases.append(self.mig10.bias)
#         self.bias_scale_factor :List[float] = [1. for i in range(10)]
#         self.bias_scale_factor[1-1] = 1.
#         self.bias_scale_factor[2-1] = 1.
#         self.bias_scale_factor[3-1] = 1.
#         self.bias_scale_factor[4-1] = 1.
#         self.bias_scale_factor[5-1] = 1.
#         self.bias_scale_factor[6-1] = 1.
#         self.bias_scale_factor[7-1] = 1.
#         self.bias_scale_factor[8-1] = 1.
#         self.bias_scale_factor[9-1] = 1.
#         self.bias_scale_factor[10-1] = 3.
#         #0.02 works for 3 to 6 layers. But when I tested 7 layers, it's very slow.

#         if True:
#             self.act1  = torch.nn.ReLU()
#             self.act2  = torch.nn.ReLU()
#             self.act3  = torch.nn.ReLU()
#             self.act4  = torch.nn.ReLU()
#             self.act5  = torch.nn.ReLU()
#             self.act6  = torch.nn.ReLU()
#             self.act7  = torch.nn.ReLU()
#             self.act8  = torch.nn.ReLU()
#             self.act9  = torch.nn.ReLU()
#         pass

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         a_small_number :float = 1.
#         x = self.mig1 (x) 
#         x = self.act1(x)
#         x = x*a_small_number
#         x = self.mig2 (x) 
#         x = self.act2(x)
#         x = x*a_small_number
#         x = self.mig3 (x) 
#         x = self.act3(x)
#         x = x*a_small_number
#         x = self.mig4 (x) 
#         x = self.act4(x)
#         x = x*a_small_number
#         x = self.mig5 (x) 
#         x = self.act5(x)
#         x = x*a_small_number
#         x = self.mig6 (x) 
#         x = self.act6(x)
#         x = x*a_small_number
#         x = self.mig7 (x) 
#         x = self.act7(x)
#         x = x*a_small_number
#         x = self.mig8 (x) 
#         x = self.act8(x)
#         x = x*a_small_number
#         x = self.mig9 (x) 
#         x = self.act9(x)
#         x = x*a_small_number
#         return self.mig10(x) #move this line up and down. Each time 3 lines.
    
#     def before_optim_step(self)->None:
#         for i in range(len(self.biases)):
#             bias_param:torch.nn.Parameter = self.biases[i]
#             if bias_param is not None:
#                 if bias_param.grad is not None:
#                     bias_param.grad *= self.bias_scale_factor[i]



# # s = '''print("mig{0}.half_weight", model.mig{0}.half_weight[:1][0,:7])
# # print("mig{0}.half_weight.grad", model.mig{0}.half_weight.grad[:1][0,:7])
# # print("mig{0}.bias", model.mig{0}.bias[0,:5])
# # print("mig{0}.bias.grad", model.mig{0}.bias.grad[0,:5])'''
# # for i in range(10):
# #     print(s.format(i+1))


# in_feature = 64
# model = test_multi_MIG__relu(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 20
# if True:
#     input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
#     #target = input.pow(1.5)
#     target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# else:
#     input = torch.rand([data_amount, in_feature])+0.3
#     #target = input.pow(1.5)
#     #target = torch.rand([data_amount, 1])+0.1
#     target = ((input.pow(1.5)-input.pow(2.5)).sum(dim=1)).unsqueeze(1)
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.000002)
# #00001 works for 4 mig layers
# #000005 works for 6 mig layers
# # I modified the bias scale factor here.
# #000002 works for 8 to 10 mig layers


# iter_per_print = 1000#500
# print_count = 333#5
# #5000 epochs for 4 mig layers.
# #7000 epochs for 6 mig layers.
# # I modified the bias scale factor here.
# #30000 epochs for 7 mig layers.
# #20000 epochs for 8 to 10 mig layers.
# # model = model.cuda()
# # input = input.cuda()
# # target = target.cuda()
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     model.before_optim_step()# This line is important.
#     optimizer.step()
#     model.eval()

#     if epoch%iter_per_print == iter_per_print-1:
#         print(epoch+1, "loss", loss)
#         if False:
#             #below is another way to access the grad. It's actually safe.
#             # print(model.mig1.half_weight.grad[0:1])
#             # print(model.mig2.half_weight.grad[0:1])
#             print(model.mig1.half_weight[0:1])
#             print(model.mig1.half_weight.grad[0:1])
#             print(model.mig1.half_weight_mirrored[0:1])
#             print(model.mig1.half_weight_mirrored.grad[0:1])
#             print(model.mig1.bias[0:1])
#             print(model.mig1.bias.grad[0:1])

#         if True:
#             # print(model.mig1.half_weight[0:1])
#             # print(model.mig2.half_weight[0:1])
#             # print(model.mig10.half_weight[0:1])
#             # print(model.mig10.bias)
#             # print(model.mig10.half_weight_mirrored.grad)
#             pass
#         if True:
#             pass
# print("pred  ", pred.T)

# print("target", target.T)

# if False:
#     print("mig10.half_weight", model.mig10.half_weight[:1][0,:16])
#     print("mig1.half_weight", model.mig1.half_weight[:1][0,:16])
#     print("mig2.half_weight", model.mig2.half_weight[:1][0,:7])
#     print("mig3.half_weight", model.mig3.half_weight[:1][0,:7])
#     print("mig4.half_weight", model.mig4.half_weight[:1][0,:7])
#     print("mig5.half_weight", model.mig5.half_weight[:1][0,:7])
#     print("mig6.half_weight", model.mig6.half_weight[:1][0,:7])
#     print("mig10.half_weight.grad", model.mig10.half_weight.grad[:1][0,:16])
#     print("mig1.half_weight.grad", model.mig1.half_weight.grad[:1][0,:16])
#     print("mig2.half_weight.grad", model.mig2.half_weight.grad[:1][0,:7])
#     print("mig3.half_weight.grad", model.mig3.half_weight.grad[:1][0,:7])
#     print("mig4.half_weight.grad", model.mig4.half_weight.grad[:1][0,:7])
#     print("mig5.half_weight.grad", model.mig5.half_weight.grad[:1][0,:7])
#     print("mig6.half_weight.grad", model.mig6.half_weight.grad[:1][0,:7])


#     print("mig10.bias", model.mig10.bias)
#     print("mig1.bias", model.mig1.bias[0:16])
#     print("mig2.bias", model.mig2.bias[0:5])
#     print("mig3.bias", model.mig3.bias[0:5])
#     print("mig4.bias", model.mig4.bias[0:5])
#     print("mig5.bias", model.mig5.bias[0:5])
#     print("mig6.bias", model.mig6.bias[0:5])
#     print("mig10.bias.grad", model.mig10.bias.grad)
#     print("mig1.bias.grad", model.mig1.bias.grad[0:16])
#     print("mig2.bias.grad", model.mig2.bias.grad[0:5])
#     print("mig3.bias.grad", model.mig3.bias.grad[0:5])
#     print("mig4.bias.grad", model.mig4.bias.grad[0:5])
#     print("mig5.bias.grad", model.mig5.bias.grad[0:5])
#     print("mig6.bias.grad", model.mig6.bias.grad[0:5])


#     jlkdfs=45698

#     print("mig7.half_weight", model.mig7.half_weight[:1][0,:7])
#     print("mig7.half_weight.grad", model.mig7.half_weight.grad[:1][0,:7])
#     print("mig7.bias", model.mig7.bias[0:5])
#     print("mig7.bias.grad", model.mig7.bias.grad[0:5])
#     print("mig8.half_weight", model.mig8.half_weight[:1][0,:7])
#     print("mig8.half_weight.grad", model.mig8.half_weight.grad[:1][0,:7])
#     print("mig8.bias", model.mig8.bias[0:5])
#     print("mig8.bias.grad", model.mig8.bias.grad[0:5])
#     print("mig9.half_weight", model.mig9.half_weight[:1][0,:7])
#     print("mig9.half_weight.grad", model.mig9.half_weight.grad[:1][0,:7])
#     print("mig9.bias", model.mig9.bias[0:5])
#     print("mig9.bias.grad", model.mig9.bias.grad[0:5])

# jkldfs=345789











