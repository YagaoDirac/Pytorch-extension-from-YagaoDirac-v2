import torch

def __DEBUG_ME__():
    return __name__ == "__main__"
if "test" and False:
    assert __DEBUG_ME__()
    pass
    


torch.nn.Linear#press f12 and copy paste the format.

"I don't know if you can find any example like this. "
"Thif format example can help you understand how to diy in pytorch."
class Null_Model_as_example(torch.nn.Module):
    myinfo:str
    plus_me:torch.nn.parameter.Parameter
    def __init__(self, device=None, dtype=None,) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        #init here.
        self.myinfo = "myinfo"
        self.plus_me = torch.nn.parameter.Parameter(torch.tensor([10.],**factory_kwargs))
        pass
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #do the job here.
        x = input
        x = x+self.plus_me
        return x

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"my info={self.myinfo}, plus?={self.plus_me}"

    def __repr__(self):
        return f"this is an example. it pluses 1 to the input."

if "train loop example" and __DEBUG_ME__() and True:
    model_null = Null_Model_as_example()
    optim = torch.optim.SGD(params=model_null.parameters(), lr=0.25)
    #lr=0.25 for this example. In real case, it's usually 1e-4 to 1e-5
    assert optim.param_groups[0]['lr'] == 0.25
    
    input = torch.tensor([123.])
    assert input.requires_grad == False
    
    predict:torch.Tensor = model_null(input) #line 1
    #loss = something.                  #line 2
    optim.zero_grad()                   #line 3
    predict.backward()                  #line 4 
    #in real case, loss is calculated with predict, so do loss.backward() instead.
    optim.step()                        #line 5
    #5 lines in total.
    
    assert input.item() == 123.
    assert model_null.plus_me.item() == 10-1*0.25
    pass
    
    
        
        
        
        
        
        
        
class Model_minimum_training_example(torch.nn.Module):
    linear_1:torch.nn.Linear
    relu_1:torch.nn.ReLU
    linear_2:torch.nn.Linear
    def __init__(self, device=None, dtype=None,) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        #init here.
        self.myinfo = "myinfo"
        self.linear_1 = torch.nn.Linear(1,4,True)
        self.relu_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(4,1,True)
        pass
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #do the job here.
        x = input
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        return x

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"not implemented"

    def __repr__(self):
        return f"not implemented"
    
if "minimum example of training." and __DEBUG_ME__() and True:
    #step 1, dataset
    input = torch.linspace(start=0., end=2., steps=5).reshape([-1, 1])
    with torch.no_grad():
        ground_truth = input.exp()
        pass
    
    #step 2, training infra
    model_minimum_training_examplel = Model_minimum_training_example()
    optim = torch.optim.SGD(params=model_minimum_training_examplel.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()

    #step 3, train it.    
    for epoch in range(2):
        predict = model_minimum_training_examplel(input)    #line 1
        loss = loss_function(predict, ground_truth)         #line 2
        optim.zero_grad()                                   #line 3
        loss.backward()                                     #line 4 
        optim.step()                                        #line 5
        pass
    pass
    
    
    
    
    
class Model_read_params_please(torch.nn.Module):
    linear_1:torch.nn.Linear
    relu_1:torch.nn.ReLU
    linear_2:torch.nn.Linear
    def __init__(self, device=None, dtype=None,) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        #init here.
        self.myinfo = "myinfo"
        self.linear_1 = torch.nn.Linear(1,4,True)
        self.relu_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(4,1,True)
        pass
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #do the job here.
        x = input
        x = self.linear_1(x)
        print("after linear layer 1", x)
        x = self.relu_1(x)
        print("after relu layer 1", x)
        x = self.linear_2(x)
        return x

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"not implemented"

    def __repr__(self):
        return f"not implemented"
    
if "read all the param in here" and __DEBUG_ME__() and True:
    #step 1, dataset
    input = torch.linspace(start=0., end=2., steps=5).reshape([-1, 1])
    with torch.no_grad():
        ground_truth = input.exp()
        pass
    
    #step 2, training infra
    model_read_params_please = Model_read_params_please()
    optim = torch.optim.SGD(params=model_read_params_please.parameters(), lr=0.1)
    loss_function = torch.nn.MSELoss()

    #step 3, train it.    
    for epoch in range(2):
        predict = model_read_params_please(input)   #line 1
        print("predict", predict)
        loss = loss_function(predict, ground_truth) #line 2
        print("loss", loss)
        optim.zero_grad()                           #line 3
        loss.backward()                             #line 4 
        print("grad of linear layer 1", model_read_params_please.linear_1.weight.grad, model_read_params_please.linear_1.bias.grad)
        print("grad of linear layer 2", model_read_params_please.linear_2.weight.grad, model_read_params_please.linear_2.bias.grad)
        print("param of linear layer 1 before step", model_read_params_please.linear_1.weight, model_read_params_please.linear_1.bias)
        print("param of linear layer 2 before step", model_read_params_please.linear_2.weight, model_read_params_please.linear_2.bias)
        optim.step()                                #line 5
        print("param of linear layer 1 after step", model_read_params_please.linear_1.weight, model_read_params_please.linear_1.bias)
        print("param of linear layer 2 after step", model_read_params_please.linear_2.weight, model_read_params_please.linear_2.bias)
        
        pass
    pass