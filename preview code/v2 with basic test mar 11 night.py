
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

'''
Known conclusions:

If you see this line, tell me to rewrite the conclusion section.
I added activation function for the last layer and got confused for 1 hour. While being confused, I wrote this section.



Gramo(GRA-dient MO-dificatino layer):
(Notice, by gramo, I mean the only gramo alone. The combination is explained below in another section.)

Gramo with linear and Sigmoid:
I tested this one with Linear(or Dense in tensorflow). The linear,gramo,sigmoid works fine ONLY with and narrow model,
 if you align the gradients each layer gets properly.
I believe the issue is because, sigmoid mutes some neutrons behind it much more heavy than others, while gramo doesn't help in this case.
If I want to solve this problem, I probably need to pow(g,0.1), then normalize it, then pow(g,10) back.
But relu seems not to have this problem.
Maybe this is a todo-list.
Bacially you can make the grad every element gets as close to others as possible. I showed this in a test code. The trick is to use
 scaling_ratio.
Btw, Idk if I should call it scaling ratio or factor. Maybe factor is a better name.

Gramo with linear and ReLU:
Gramo doesn't work properly with a narrow relu. The linear,gramo,relu doesn't work at width of 4.
Since the width of input is also counted as the width of the model, we really need a way to widen the input(duplicating may help)
 to make it work with relu.
Anyway, it works in some cases. But I don't get a very confirmed idea how to guaruntee it works.


Mirror(The magic Mirror):
Notice. Before this version, it works like 0.5*(w*(x+delta)+w*(x-delta))+b. But now, it works like half_w*(x+delta)+half_w*(x-delta)+b.
So, maybe I should name it in other ways.











'''


from typing import Any, Optional
import torch
import math

__all__ = [
    'GradientModification',
    'MirrorLayer',
    'MirrorWithGramo',
    'GradientModificationFunction', #Should I expose this?
    'Linear_gramo', #Should I rename this one? Or somebody help me with the naming?
    ]

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


# scaling_ratio = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])
# a = torch.tensor([0.], requires_grad=True)
# b = GradientModificationFunction.apply(a,scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([0.0000012])
# torch.autograd.backward(b, g_in,inputs= a)
# #print(g[0])
# print(a.grad)

# jkldfs=456879

# scaling_ratio = torch.tensor([1.])
# epi=torch.tensor([1e-5])
# div_me_when_g_too_small = torch.tensor([1e-3])

#print(1.!=torch.tensor([1.]).item())

# input = torch.tensor([[0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.0000012]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.0012")

# input = torch.tensor([[0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 1.")

# input = torch.tensor([[0., 0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.7, 0.7")

# input = torch.tensor([[0.], [0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.12], [0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 1., 1.")

# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.1, 0.173], [0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.5, 0.86, 0.7, 0.7")

# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# #output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# output = GradientModificationFunction.apply(input, torch.tensor([1.]), torch.tensor([0.001]),torch.tensor([0.1]))
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.001, 0.001, 0.7, 0.7")

# input = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
# #output = GradientModificationFunction.apply(input, scaling_ratio,epi,div_me_when_g_too_small)
# output = GradientModificationFunction.apply(input, torch.tensor([2.]), torch.tensor([0.001]),torch.tensor([0.1]))
# ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
# g_in = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# torch.autograd.backward(output, g_in,inputs= input)
# print(input.grad, "should be 0.002, 0.002, 1.4, 1.4")


# jkldfs=456879









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
        self.scaling_ratio = torch.tensor([scaling_ratio], requires_grad=False)
        self.epi=torch.tensor([epi], requires_grad=False)
        self.div_me_when_g_too_small = torch.tensor([div_me_when_g_too_small], requires_grad=False)
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
        self.scaling_ratio = torch.tensor([scaling_ratio], requires_grad=False)
        pass

    




# input = torch.tensor([[1.]], requires_grad=True)
# model = GradientModification()

# loss_function = torch.nn.L1Loss()# the L1Loss function only provides the direction. It's the dirivitive of abs.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.]], requires_grad=True))
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     print(input.grad, "should be 1.")

#     optimizer.step()
#     print(input, "should be 0.9")
    
#     model.eval()
#     pass



# input = torch.tensor([[0.]], requires_grad=True)
# model = GradientModification(epi=0.001,div_me_when_g_too_small=0.1)###

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.00012]], requires_grad=True))
#     #the grad input for the backward is -2*0.002. It's small enough to be *10
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     print(input.grad, "should be -0.0024")

#     optimizer.step()
#     print(input, "should be 0.00024")
    
#     model.eval()
#     pass




# input = torch.tensor([[0.,0.]], requires_grad=True)
# model = GradientModification(epi=0.001,div_me_when_g_too_small=0.1)###

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.00012,0.00012]], requires_grad=True))
#     #the grad input for the backward is -2*0.002/2. . It's small enough to be *10
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     print(input.grad, "should be -0.0012")

#     optimizer.step()
#     print(input, "should be 0.00012")
    
#     model.eval()
#     pass




# input = torch.tensor([[0., 0., 0.]], requires_grad=True)
# model = GradientModification(epi=0.001,div_me_when_g_too_small=0.1)###

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.00012, 0.00012, 0.00012]], requires_grad=True))
#     #the grad input for the backward is -2*0.00012/3. . It's small enough to be *10
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     print(input.grad, "should be -0.0008")

#     optimizer.step()
#     print(input, "should be 0.00008")
    
#     model.eval()
#     pass




# input = torch.tensor([[0.,0.],[0.,0.]], requires_grad=True)
# model = GradientModification(epi=0.001, div_me_when_g_too_small=0.1)

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.0001,0.0001],[0.0001,0.0001]], requires_grad=True))
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




# input = torch.tensor([[0.,0.],[0.,0.]], requires_grad=True)
# model = GradientModification(epi=0.001, div_me_when_g_too_small=0.1)

# loss_function = torch.nn.MSELoss()#the grad of MSE is 2*x. So it provides the direction also the error.
# optimizer = torch.optim.SGD([input], lr=0.1)
# for epoch in range(1):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.0001,0.0001],[0.1,0.1]], requires_grad=True))
#     #the grad input for the backward of first 2 elements backward is -2*0.0001/4. . It's smaller than 0.001. Then it's mul ed by 10.
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     #print(torch.tensor([[0.0001,0.0002],[0.1,0.2]]))
#     print(input.grad, "first 2 should be -0.0005")

#     optimizer.step()
#     print(input, "should be 0.00005")
    
#     model.eval()
#     pass

# jkldfs=345789



#good, this works. Now let's go downward.





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
    




# model = Linear_gramo(1,1,False)
# model.linear.weight = torch.nn.Parameter(torch.ones_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# ### model.set_learning_rate(optimizer.param_groups[0]["lr"])
# # for item in model.parameters():
# #     print(item, "inside model.parameters()")
# #     pass

# #for epoch in range(1):
# #model.train()
# pred = model(input)# w = x = 1.. No b or b = 0..
# loss = loss_function(pred, torch.tensor([[0.]], requires_grad=True))
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be 1.")

# optimizer.step()
# print(model.linear.weight, "should be 0.9")

# #model.eval()


   
# model = Linear_gramo(1,1,False)
# model.linear.weight = torch.nn.Parameter(torch.zeros_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# #for epoch in range(1):
# pred = model(input)# x = 1.. w = b = 0..
# loss = loss_function(pred, torch.tensor([[1.]], requires_grad=True))
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be -1.")

# optimizer.step()
# print(model.linear.weight, "should be 0.1")



# model = Linear_gramo(1,1,False)
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



# model = Linear_gramo(2,1,False)
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



# model = Linear_gramo(2,2,False)
# model.linear.weight = torch.nn.Parameter(torch.zeros_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1., 1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# #for epoch in range(1):
# pred = model(input)# x = 1.. w = b = 0..
# loss = loss_function(pred, torch.tensor([[1., 1.]], requires_grad=True))
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be all -0.7")

# optimizer.step()
# print(model.linear.weight, "should be 0.07")



# model = Linear_gramo(2,2,False)
# model.linear.weight = torch.nn.Parameter(torch.zeros_like(model.linear.weight))
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1., 1.],[1., 1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# #for epoch in range(1):
# pred = model(input)# x = 1.. w = b = 0..
# loss = loss_function(pred, torch.tensor([[1., 1.]], requires_grad=True))
# optimizer.zero_grad()
# loss.backward()
# #optimizer.param_groups[0]["lr"] = 0.01
# print(model.linear.weight.grad, "should be all -0.7*2 == -1.4")

# optimizer.step()
# print(model.linear.weight, "should be 0.14")



# All the test above work.







### this is also test code. Comment it out.
# class test_stacked_Linear_gramo__sigmoid(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.width_inside = 64
#         self.linear1 = torch.nn.Linear(in_features, self.width_inside, True, device, dtype)
#         self.linear2 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear3 = torch.nn.Linear(self.width_inside, out_features, True, device, dtype)

#         self.gramo1 = GradientModification(scaling_ratio=0.1)
#         self.gramo2 = GradientModification(scaling_ratio=0.05)
#         self.gramo3 = GradientModification(scaling_ratio=0.01)

#         self.sigmoid1 = torch.nn.Sigmoid()
#         self.sigmoid2 = torch.nn.Sigmoid()

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         x = self.linear1(x)
#         x = self.gramo1(x)
#         x = self.sigmoid1(x)
#         x = self.linear2(x)
#         x = self.gramo2(x)
#         x = self.sigmoid2(x)
#         x = self.linear3(x)
#         x = self.gramo3(x)
#         return x



# model = test_stacked_Linear_gramo__sigmoid(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# target = torch.tensor([[0.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(10):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()



# in_feature = 16
# model = test_stacked_Linear_gramo__sigmoid(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 10
# input = torch.rand([data_amount, in_feature])+0.3
# target = torch.rand([data_amount, 1])+0.1
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

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
#         # print(model.linear1.weight.grad[0:1])
#         # print(model.linear2.weight.grad[0:1])
#         # print(model.linear3.weight.grad[0:1])
#         # print("--------------")


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


# in_feature = 16
# model = test_stacked_Linear_gramo__relu(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 10
# input = torch.rand([data_amount, in_feature])+0.3
# target = torch.rand([data_amount, 1])+0.1
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
#             print(model.linear1.weight.grad[0:2])
#             print(model.linear2.weight.grad[0:2])
#             print(model.linear3.weight.grad[0:2])
#             print("--------------")

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

#         self.width_inside = 16
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
#         self.div_me_when_ = 0.000001
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
# model = test_stacked_Linear_gramo_3(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 20
# input = torch.rand([data_amount, in_feature])+0.3
# #target = input.pow(1.5)
# target = torch.rand([data_amount, 1])+0.1
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

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
#             print(model.linear3.weight.grad[0:1])
#             print(model.linear4.weight.grad[0:1])
#             print(model.linear5.weight.grad[0:1])
#             print(model.linear6.weight.grad[0:1])
#             print(model.linear7.weight.grad[0:1])
#             print(model.linear8.weight.grad[0:1])
#             print(model.linear9.weight.grad[0:1])
#             print(model.linear10.weight.grad[0:1])
#             print("--------------")

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
            result.bias = self.bias
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
# model = MirrorLayer(in_feature,1,auto_merge_duration=1)
# loss_function = torch.nn.MSELoss()
# data_amount = 1
# input = torch.ones([data_amount, in_feature])*0.00001
# target = torch.ones([data_amount, 1])*0.00001
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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
#         if True:
#             print(model.half_weight.grad)
#             print(model.half_weight_mirrored.grad)
#             print("--------------")

# jkldfs=345789




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
#             pass

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
# model = test_multi_mirror__relu(in_feature,1)
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
            result.bias = self.bias
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




# class test_multi_mirror__sigmoid(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.auto_merge_ = 1
#         self.mirror1  = MirrorWithGramo(in_features, self.width_inside,       bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror2  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror3  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror4  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror5  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror6  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror7  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror8  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror9  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror10 = MirrorWithGramo(self.width_inside, out_features,     bias, device, dtype, self.auto_merge_, *args, **kwargs)

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




# in_feature = 2
# model = test_multi_mirror__sigmoid(in_feature,1)
# loss_function = torch.nn.MSELoss()
# data_amount = 1
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
#             # this part is actually very interesting. If you have time, uncomment this part.
#             # if model.mirror1.half_weight.grad is not None:
#             #     print(model.mirror1.half_weight.grad[0:1])
#             # if model.mirror2.half_weight.grad is not None:
#             #     print(model.mirror2.half_weight.grad[0:1])
#             # if model.mirror10.half_weight.grad is not None:
#             #     print(model.mirror10.half_weight.grad[0:1])
#             #below is another way to access the grad. It's actually safe.
#             # print(model.mirror1.half_weight.grad[0:1])
#             # print(model.mirror1.half_weight_mirrored.grad[0:1])
#             # #print(model.mirror2.half_weight.grad[0:1])
#             # print(model.mirror10.half_weight.grad[0:1])
#             # print(model.mirror10.half_weight_mirrored.grad[0:1])
#             pass

#         if True:
#             print(model.mirror1.half_weight[0:1])
#             print(model.mirror2.half_weight[0:1])
#             print(model.mirror10.half_weight[0:1])
#             # print(model.half_weight_mirrored.grad)
#             pass

# jkldfs=345789


# it's still very wierd. The loss deceases, but the input side layers are not trained..




# class test_multi_mirror__relu(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.auto_merge_ = 1
#         self.mirror1  = MirrorWithGramo(in_features, self.width_inside,       bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror2  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror3  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror4  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror5  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror6  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror7  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror8  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror9  = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, self.auto_merge_, *args, **kwargs)
#         self.mirror10 = MirrorWithGramo(self.width_inside, out_features,     bias, device, dtype, self.auto_merge_, *args, **kwargs)

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
# model = test_multi_mirror__relu(in_feature,1)
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

