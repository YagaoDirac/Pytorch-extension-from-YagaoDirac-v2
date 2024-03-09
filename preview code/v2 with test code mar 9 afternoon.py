
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





# 
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








#this is also test code. Comment it out.
# class test_stacked_Linear_gramo(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.width_inside = 4
#         self.linear1 = torch.nn.Linear(in_features, self.width_inside, True, device, dtype)
#         self.linear2 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear3 = torch.nn.Linear(self.width_inside, out_features, True, device, dtype)

#         self.gramo1 = GradientModification()
#         self.gramo2 = GradientModification()
#         self.gramo3 = GradientModification()

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         x = self.linear1(x)
#         x = self.gramo1(x)
#         x = self.linear2(x)
#         x = self.gramo2(x)
#         x = self.linear3(x)
#         x = self.gramo3(x)
#         return x



# model = test_stacked_Linear_gramo(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(10):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([0.], requires_grad=True))
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()



# model = test_stacked_Linear_gramo(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# for epoch in range(30):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([0.], requires_grad=True))
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()



# jkldfs=345789





# class test_stacked_Linear_gramo_2(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         self.width_inside = 64
#         self.linear1 = torch.nn.Linear(in_features, self.width_inside, True, device, dtype)
#         self.linear2 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear3 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear4 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear5 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear6 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear7 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear8 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear9 = torch.nn.Linear(self.width_inside, self.width_inside, True, device, dtype)
#         self.linear10 = torch.nn.Linear(self.width_inside, out_features, True, device, dtype)

#         self.gramo1 = GradientModification()
#         self.gramo2 = GradientModification()
#         self.gramo3 = GradientModification()
#         self.gramo4 = GradientModification()
#         self.gramo5 = GradientModification()
#         self.gramo6 = GradientModification()
#         self.gramo7 = GradientModification()
#         self.gramo8 = GradientModification()
#         self.gramo9 = GradientModification()
#         self.gramo10 = GradientModification()

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         x = self.linear1(x)
#         x = self.gramo1(x)
#         x = self.linear2(x)
#         x = self.gramo2(x)
#         x = self.linear3(x)
#         x = self.gramo3(x)
#         x = self.linear4(x)
#         x = self.gramo4(x)
#         x = self.linear5(x)
#         x = self.gramo5(x)
#         x = self.linear6(x)
#         x = self.gramo6(x)
#         x = self.linear7(x)
#         x = self.gramo7(x)
#         x = self.linear8(x)
#         x = self.gramo8(x)
#         x = self.linear9(x)
#         x = self.gramo9(x)
#         return x
#     # according to my test, if you return x here, it's not gonna meet the epi thing.
#     # but I know you are gonna push this toy to the extreme. If you add the 10th layer, you can modify the epi to a smaller number.
#     # And you maybe want to test this toy at float64. You know wat i mean, lol.
#         x = self.linear10(x)
#         x = self.gramo10(x)
#         return x


# model = test_stacked_Linear_gramo_2(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[1.]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# for epoch in range(10):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([0.], requires_grad=True))
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()



# jkldfs=345789











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





# Here's the new version.
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
        head2:torch.Tensor = torch.nn.functional.linear(input - 0.5, self.half_weight, self.bias)
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



# model = MirrorLayer(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[0.00001]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(3):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.00001]], requires_grad=True))
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()




# model = MirrorLayer(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[0.00001]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# for epoch in range(30):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.00001]], requires_grad=True))
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()



# jkldfs=345789





# class test_multi_mirror(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.mirror1 = MirrorLayer(in_features, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror2 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror3 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror4 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror5 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror6 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror7 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror8 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror9 = MirrorLayer(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror10 = MirrorLayer(self.width_inside, out_features, bias, device, dtype, *args, **kwargs)
#         pass

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         a_small_number :float = 0.001
#         x = self.mirror1 (x) 
#         x = x*a_small_number
#         x = self.mirror2 (x) 
#         x = x*a_small_number
#         x = self.mirror3 (x) 
#         x = x*a_small_number
#         x = self.mirror4 (x) 
#         x = x*a_small_number
#         x = self.mirror5 (x) 
#         x = x*a_small_number
#         x = self.mirror6 (x) 
#         x = x*a_small_number
#         x = self.mirror7 (x) 
#         x = x*a_small_number
#         x = self.mirror8 (x) 
#         x = x*a_small_number
#         x = self.mirror9 (x) 
#         x = x*a_small_number
#         x = self.mirror10(x) 
#         return x#move this line up and down



# model = test_multi_mirror(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[0.00001]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(50):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.1]], requires_grad=True))
#     if epoch%5 == 4:
#         print(loss)
#         pass
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
        head2:torch.Tensor = torch.nn.functional.linear(input - 0.5, self.half_weight, self.bias)
        
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







# model = MirrorLayer(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[0.00001]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# for epoch in range(3):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.00001]], requires_grad=True))
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()




# model = MirrorLayer(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[0.00001]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# for epoch in range(30):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.00001]], requires_grad=True))
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()






# give this one a go. It's WILD.

# class test_multi_mirrorgramo(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, \
#                             device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.width_inside = 64
#         self.mirror_with_gramo1 = MirrorWithGramo(in_features, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo2 = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo3 = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo4 = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo5 = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo6 = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo7 = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo8 = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo9 = MirrorWithGramo(self.width_inside, self.width_inside, bias, device, dtype, *args, **kwargs)
#         self.mirror_with_gramo10 = MirrorWithGramo(self.width_inside, out_features, bias, device, dtype, *args, **kwargs)
#         pass

#     def forward(self, x:torch.tensor)->torch.Tensor:
#         a_small_number :float = 0.001
#         x = self.mirror_with_gramo1 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo2 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo3 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo4 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo5 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo6 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo7 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo8 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo9 (x) 
#         x = x*a_small_number
#         x = self.mirror_with_gramo10(x) 
#         return x



# model = test_multi_mirrorgramo(1,1)
# loss_function = torch.nn.MSELoss()

# input = torch.tensor([[0.00001]])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# for epoch in range(10):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, torch.tensor([[0.1]], requires_grad=True))
#     if epoch%1 == 0:
#         print(loss)
#         pass
#     optimizer.zero_grad()
#     loss.backward()
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     model.eval()



# jkldfs=345789




