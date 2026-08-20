import torch

# class bug_report_model(torch.nn.Module):
#     def __init__(self, param:torch.Tensor|None = None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if param is None:
#             self.param = torch.nn.Parameter(torch.randn(size=[3,2]))
#             pass
#         else:
#             self.param = torch.nn.Parameter(param)
#             pass

#         assert self.param.requires_grad == True
#         return
#     def forward(self, input:torch.Tensor)->torch.Tensor:
#         output = input@(self.param.T)
#         return output

#     def break_the_shape(self)->None:
#         self.param.data = torch.randn(size=[5,2])
#         return
#     #end of class

# assert torch.__version__ == '2.12.1+cu132'

# batch = 7
# #infra
# the_ori_model = bug_report_model()
# #epoch 0
# output___batch_3:torch.Tensor = the_ori_model(torch.randn(size=[batch, 2]))
# assert output___batch_3.shape == torch.Size([batch, 3])
# output___batch_3.backward(gradient=torch.randn_like(output___batch_3))
# del output___batch_3
# #I don't need the grad. No optim.step()
# the_ori_model.param.grad = None#zero_grad
# #a completely reshape of the inner data.
# the_ori_model.break_the_shape()

# #epoch 1
# output___batch_5:torch.Tensor = the_ori_model(torch.randn(size=[batch, 2]))
# assert output___batch_5.shape == torch.Size([batch, 5])
# #output___batch_5.backward(gradient=torch.randn_like(output___batch_5)) this line doesn't work. 
# del output___batch_5

# the_new_model = bug_report_model(the_ori_model.param.data)
# del the_ori_model

# #epoch 1 but with the new object
# new___output___batch_5:torch.Tensor = the_new_model(torch.randn(size=[batch, 2]))
# assert new___output___batch_5.shape == torch.Size([batch, 5])
# assert the_new_model.param.grad is None
# new___output___batch_5.backward(gradient=torch.randn_like(new___output___batch_5))# this line works.
# assert the_new_model.param.grad is not None

# print("new model object works.")












'''Ok now we have a part 2. Enjoy'''
class can_I_replace_the_element_inside_a_param_list__without_refresh_the_entire_object(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.param_list = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(size=[3,2]))])
        assert self.param_list[0].requires_grad == True

        return
    def forward(self, input:torch.Tensor)->torch.Tensor:
        output = input@(self.param_list[0].T)
        return output

    def break_the_shape(self)->None:
        self.param_list[0] = torch.nn.Parameter(torch.rand(size=[5,2]))
        return
    #end of class


batch = 7
#infra
the_ori_model = can_I_replace_the_element_inside_a_param_list__without_refresh_the_entire_object()
#epoch 0
output___batch_3:torch.Tensor = the_ori_model(torch.randn(size=[batch, 2]))
assert output___batch_3.shape == torch.Size([batch, 3])
output___batch_3.backward(gradient=torch.randn_like(output___batch_3))
del output___batch_3
#I don't need the grad. No optim.step()
the_ori_model.param_list[0].grad = None#zero_grad
#a completely reshape of the inner data.
the_ori_model.break_the_shape()

#epoch 1
output___batch_5:torch.Tensor = the_ori_model(torch.randn(size=[batch, 2]))
assert output___batch_5.shape == torch.Size([batch, 5])
output___batch_5.backward(gradient=torch.randn_like(output___batch_5))# this line works. 
print("The elements of a param_list can be replaced with totally different shape and still works.")








