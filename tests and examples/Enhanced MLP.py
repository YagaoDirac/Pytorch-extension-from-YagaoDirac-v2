from typing import Any, List, Tuple, Optional, Self
import math
import torch
from ParamMo import GradientModification, XModification

继续

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








