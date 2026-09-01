import torch


class the_model_class(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = torch.nn.Parameter(torch.tensor([[1.,2],[3,4]]))
        assert self.param.requires_grad == True
        return
    def forward(self, input):
        return self.param+input
    def change_param(self):
        self.param = torch.nn.Parameter(torch.tensor([[1.,2],[3,4],[5,6]]))
        return
    def change_param___THE_WRONG_WAY(self):
        self.param.data = torch.tensor([[1.,2],[3,4],[5,6]])
        return
    def change_param___DOESNT_WORK_AT_ALL(self):
        self.param = 1.
        return
    pass#end of class

def correct_way()->None:
        
    model = the_model_class()
    assert model.param.shape == torch.Size([2,2])

    input = torch.tensor([[1.,2],[3,4]], requires_grad=True)
    output:torch.Tensor = model(input)
    assert output.shape == torch.Size([2,2])
    assert output.eq(torch.tensor([[2.,4],[6,8]])).all()

    output.backward(gradient=torch.ones_like(output), inputs=[input, model.param])
    assert model.param.grad is not None
    assert input.grad is not None


    model.change_param()
    assert model.param.shape == torch.Size([3,2])

    input = torch.tensor([[1.,2],[3,4],[5,6]], requires_grad=True)
    output = model(input)
    assert output.shape == torch.Size([3,2])
    assert output.eq(torch.tensor([[2., 4],[6, 8],[10, 12]])).all()

    output.backward(gradient=torch.ones_like(output), inputs=[input, model.param])
    assert model.param.grad is not None
    assert input.grad is not None

    return
correct_way()


def wrong_way()->None:
        
    model = the_model_class()
    assert model.param.shape == torch.Size([2,2])

    input = torch.tensor([[1.,2],[3,4]], requires_grad=True)
    output:torch.Tensor = model(input)
    assert output.shape == torch.Size([2,2])
    assert output.eq(torch.tensor([[2.,4],[6,8]])).all()

    output.backward(gradient=torch.ones_like(output), inputs=[input, model.param])
    assert model.param.grad is not None
    assert input.grad is not None


    model.change_param___THE_WRONG_WAY()
    assert model.param.shape == torch.Size([3,2])

    input = torch.tensor([[1.,2],[3,4],[5,6]], requires_grad=True)
    output = model(input)
    assert output.shape == torch.Size([3,2])
    assert output.eq(torch.tensor([[2., 4],[6, 8],[10, 12]])).all()

    output.backward(gradient=torch.ones_like(output), inputs=[input, model.param])#this line doesn't work.
    assert model.param.grad is not None
    assert input.grad is not None

    return
wrong_way()











        


