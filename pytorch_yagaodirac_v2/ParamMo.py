# 后续计划。
# 简化版本的gramo
# x的归一化。
# 直接堆叠测试。
# 结合relu，tanh，sigmoid测试。


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
Now get back to the definition of model. With gramo or mig, you can use the scaling_factor param when create a gramo or mig to
 align the wg/w between layers. Usually, the layers in the mid are very similar at wg/w, so you can use them as reference 
 safely. Then the callback function you created in step 2 can help aligning the bg/b to the layer's wg/w.
End)
This can be seen in the last test. I did it manually. But I think it's time to write some code to automate it.




For all test:
Aligning the avg length of gradients for each layer is optional, it helps in some cases. 
In all the useful test, I only carefully aligned it for the last test.

When creating any gramo layer, set the scaling_factor param. You can use the result in a recent paper from XAI.


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
 directive, the grad is multiplied by scaling_factor.
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


from typing import Any, List, Tuple, Optional, Self
import math
import torch


# __all__ = [
#     'make_grad_noisy',
#     'GradientModification',
#     'MirrorLayer',
#     'MirrorWithGramo',
#     'GradientModificationFunction', #Should I expose this?
#     'Linear_gramo', #Should I rename this one? Or somebody help me with the naming?
#     ]






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
        div_me_when_g_too_small = args[3]
        # the default values:
        # scaling_factor = torch.tensor([1.])
        # epi = torch.tensor([0.00001])
        # div_me_when_g_too_small = torch.tensor([0.001])
        # the definition of the 3 param are different from the previous version
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")

        x_needs_grad = torch.tensor([x.requires_grad])
        ctx.save_for_backward(scaling_factor, epi, div_me_when_g_too_small, x_needs_grad)
        return x

    @staticmethod
    def backward(ctx, g_in_b_o):
        #super().backward()
        scaling_factor:torch.Tensor
        requires_grad:torch.Tensor
        scaling_factor, epi, div_me_when_g_too_small, requires_grad, = ctx.saved_tensors
        if requires_grad.logical_not():
            return None, None, None, None

        #the shape should only be rank2 with[batch, something]
        # original_shape = g.shape
        # if len(g.shape) == 1:
        #     g = g.unsqueeze(1)
        # protection against div 0
        #torch.Tensor.sum()
        length_b_1:torch.Tensor = g_in_b_o.mul(g_in_b_o).sum(dim=1,keepdim = True).sqrt()
        too_small_b_1:torch.Tensor = length_b_1.le(epi)
        div_me_b_1 = too_small_b_1.logical_not()*length_b_1 + too_small_b_1*div_me_when_g_too_small
        div_me_b_1 = div_me_b_1.to(g_in_b_o.dtype)
        g_out_b_o:torch.Tensor = g_in_b_o/div_me_b_1

        scaling_factor = scaling_factor.to(g_in_b_o.dtype)
        if 1.!=scaling_factor.item():
            g_out_b_o *= scaling_factor
            pass

        return g_out_b_o, None, None, None

    pass  # class



# '''dtype adaption.'''
# scaling_factor = torch.tensor([1.], dtype=torch.float64)
# epi=torch.tensor([1e-5], dtype=torch.float32)
# div_me_when_g_too_small = torch.tensor([1e-3], dtype=torch.float16)
# a = torch.tensor([[0.]], requires_grad=True, dtype=torch.float16)
# original_dtype = a.dtype
# b = GradientModificationFunction.apply(a,scaling_factor,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[1.]], dtype=torch.float16)
# torch.autograd.backward(b, g_in,inputs= a)
# #print(g[0])
# print(a.grad.dtype, "should be ", original_dtype)
# fds=432

# '''when g_in is too small.'''
# scaling_factor = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# a = torch.tensor([[0.]], requires_grad=True)
# b = GradientModificationFunction.apply(a,scaling_factor,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.0000012]])
# torch.autograd.backward(b, g_in,inputs= a)
# #print(g[0])
# print(a.grad, "should be ", g_in/div_me_when_g_too_small)

# '''when g_in is NOT too small.'''
# scaling_factor = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 1.")
# fds=432

# '''The shape is [batches, inside a batch]. Computation is limited inside each batch.'''
# scaling_factor = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small)
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.7, 0.7")

# input = torch.tensor([[0.], [0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small)
# g_in = torch.tensor([[0.12], [0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 1., 1.")

# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small)
# g_in = torch.tensor([[0.1, 0.173], [0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.5, 0.86, 0.7, 0.7")
# fds=432

# '''Other 3 input besides the main input x.'''
# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# #output = GradientModificationFunction.apply(input, scaling_factor,     epi,                  div_me_when_g_too_small)
# output = GradientModificationFunction.apply(input, torch.tensor([1.]), torch.tensor([0.001]),torch.tensor([0.1]))
# g_in = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.001, 0.001, 0.7, 0.7")

# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# #output = GradientModificationFunction.apply(input, scaling_factor,     epi,                  div_me_when_g_too_small)
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

    def __init__(self, scaling_factor:float = 1., \
                       epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_factor = torch.nn.Parameter(torch.tensor([scaling_factor]), requires_grad=False)
        self.scaling_factor.requires_grad_(False)
        self.epi=torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
        self.epi.requires_grad_(False)
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small]), requires_grad=False)
        self.div_me_when_g_too_small.requires_grad_(False)
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_factor:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction.apply(x, self.scaling_factor, self.epi, \
                                                   self.div_me_when_g_too_small)
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
    def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
        the_device = self.div_me_when_g_too_small.device
        the_dtype = self.div_me_when_g_too_small.dtype
        self.div_me_when_g_too_small.data = torch.tensor([div_me_when_g_too_small], device=the_device, dtype=the_dtype)
        self.div_me_when_g_too_small.requires_grad_(False)
        pass

    def extra_repr(self) -> str:
        return f'scaling_factor={self.scaling_factor.item():.4e}, epi={self.epi.item():.4e}, div_me_when_g_too_small={self.div_me_when_g_too_small.item():.4e}'


# '''all the setters'''
# model = GradientModification()
# print(model.scaling_factor.requires_grad, "should be False")
# print(model.epi.requires_grad, "should be False")
# print(model.div_me_when_g_too_small.requires_grad, "should be False")
# model.set_scaling_factor(0.123)
# print(model.scaling_factor, "should be 0.123")
# print(model.scaling_factor.requires_grad, "should be False")
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








class XModificationFunction(torch.autograd.Function):
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
        x_in:torch.Tensor = args[0]
        scaling_factor = args[1]
        epi = args[2]
        div_me_when_g_too_small = args[3]
        # the default values:
        # scaling_factor = torch.tensor([1.])
        # epi = torch.tensor([0.00001])
        # div_me_when_g_too_small = torch.tensor([0.001])
        # the definition of the 3 param are different from the previous version
        
        if len(x_in.shape)!=2:
            raise Exception("XModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")

        length:torch.Tensor = x_in.mul(x_in).sum(dim=1,).sqrt()
        too_small:torch.Tensor = length.le(epi)
        div_me = too_small.logical_not()*length + too_small*div_me_when_g_too_small
        div_me = div_me.unsqueeze(dim=1)
        div_me = div_me.to(x_in.dtype)
        x_out:torch.Tensor = x_in/div_me

        scaling_factor = scaling_factor.to(x_in.dtype)
        if 1.!=scaling_factor.item():
            x_out *= scaling_factor
            pass
        
        return x_out

    @staticmethod
    def backward(ctx, g):
        #super().backward()
        return g, None, None, None

    pass  # class



# '''dtype adaption.'''
# scaling_factor = torch.tensor([1.], dtype=torch.float64)
# epi=torch.tensor([1e-5], dtype=torch.float32)
# div_me_when_g_too_small = torch.tensor([1e-3], dtype=torch.float16)
# a = torch.tensor([[0.]], dtype=torch.float16)
# original_dtype = a.dtype
# print(XModificationFunction.apply(a,scaling_factor,epi,div_me_when_g_too_small))
# print("should be ", original_dtype)
# fds=432

# '''when x_in is too small.'''
# scaling_factor = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0.0000012]])
# print(XModificationFunction.apply(input,scaling_factor,epi,div_me_when_g_too_small))
# print("should be ", input/div_me_when_g_too_small)

# '''when x_in is NOT too small.'''
# scaling_factor = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0.12]])
# print(XModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small))
# print("should be 1.")
# fds=432

# '''The shape is [batches, inside a batch]. Computation is limited inside each batch.'''
# scaling_factor = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# input = torch.tensor([[0.12, 0.12]])
# print(XModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small))
# print("should be 0.7, 0.7")

# input = torch.tensor([[0.12], [0.12]])
# print(XModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small))
# print("should be 1., 1.")


# input = torch.tensor([[0.1, 0.173], [0.12, 0.12]])
# print(XModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small))
# print("should be 0.5, 0.86, 0.7, 0.7")
# fds=432

# '''Other 3 input besides the main input x.'''
# input = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# print(XModificationFunction.apply(input, torch.tensor([1.]), torch.tensor([0.001]),torch.tensor([0.1])))
# print("should be 0.001, 0.001, 0.7, 0.7")

# input = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# print(XModificationFunction.apply(input, torch.tensor([2.]), torch.tensor([0.001]),torch.tensor([0.1])))
# print("should be 0.002, 0.002, 1.4, 1.4")
# fds=432




class XModification(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """
    def __init__(self, scaling_factor:float = 1., \
                       epi=1e-5, \
                       div_me_when_g_too_small = 1e-3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_factor = torch.nn.Parameter(torch.tensor([scaling_factor]), requires_grad=False)
        self.scaling_factor.requires_grad_(False)
        self.epi=torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
        self.epi.requires_grad_(False)
        self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small]), requires_grad=False)
        self.div_me_when_g_too_small.requires_grad_(False)
        #raise Exception("untested")
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.

        if len(x.shape)!=2:
            raise Exception("XModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_factor:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return XModificationFunction.apply(x, self.scaling_factor, self.epi, \
                                                   self.div_me_when_g_too_small)
    def set_scaling_factor(self, scaling_factor:float)->None:
        the_device = self.scaling_factor.device
        the_dtype = self.scaling_factor.dtype
        self.scaling_factor.data = torch.tensor([scaling_factor], device=the_device, dtype=the_dtype)
        self.scaling_factor.requires_grad_(False)
        pass
    def set_epi(self, epi:float)->None:
        the_device = self.epi.device
        the_dtype = self.epi.dtype
        self.epi.data = torch.tensor([epi], device=the_device, dtype=the_dtype)
        self.epi.requires_grad_(False)
        pass
    def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
        the_device = self.div_me_when_g_too_small.device
        the_dtype = self.div_me_when_g_too_small.dtype
        self.div_me_when_g_too_small.data = torch.tensor([div_me_when_g_too_small], device=the_device, dtype=the_dtype)
        self.div_me_when_g_too_small.requires_grad_(False)
        pass

    def extra_repr(self) -> str:
        return f'scaling_factor={self.scaling_factor.item():.4e}, epi={self.epi.item():.4e}, div_me_when_g_too_small={self.div_me_when_g_too_small.item():.4e}'

    pass#end of class
#No tests currently.



class DoubleModification(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.xmo = XModification()
        self.gramo = GradientModification()
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.xmo(x)
        x = self.gramo(x)
        return x
    pass #end of class.




class ReLU_with_offset(torch.nn.Module):
    r"""y = max(1, x)
    """
    def __init__(self, offset:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.offset = torch.nn.Parameter(torch.tensor([offset]), requires_grad=False)
        #raise Exception ('untested.')
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        #tensor_one = torch.tensor([1.], dtype=x.dtype, device=x.device)
        result = torch.maximum(x, self.offset)
        return result
    pass #end of class.

# layer = ReLU_with_offset(0.5123)
# input = torch.linspace(0.,1.,10)
# output = layer(input)
# fds=432