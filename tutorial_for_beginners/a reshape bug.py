import torch

class bug_report_model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.param = torch.nn.Parameter(torch.randn(size=[3,2]))
        assert self.param.requires_grad == True
        return
    def forward(self, input:torch.Tensor)->torch.Tensor:
        output = input@(self.param.T)
        return output

    def break_the_shape(self)->None:
        self.param.data = torch.randn(size=[5,2])
        return
    #end of class

assert torch.__version__ == '2.12.1+cu132'

batch = 7
the_model = bug_report_model()
output___batch_out:torch.Tensor = the_model(torch.randn(size=[batch, 2]))
assert output___batch_out.shape == torch.Size([batch, 3])
output___batch_out.backward(gradient=torch.randn_like(output___batch_out))
the_model.param.grad = None#zero_grad
the_model.break_the_shape()
output___batch_out:torch.Tensor = the_model(torch.randn(size=[batch, 2]))
assert output___batch_out.shape == torch.Size([batch, 5])
output___batch_out.backward(gradient=torch.randn_like(output___batch_out))


