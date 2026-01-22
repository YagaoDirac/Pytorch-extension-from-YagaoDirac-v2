'''
In this file, Im gonna show you all the known issue and nearly-issue.

Imo, they are bugs, but they are probably bc the early days. 
So, the best way is not to expect them to change or wait or what, 
the best way is to know all the issues and not to run into it.
'''
import torch

if "Does it require grad?" and True:
    tensor_a = torch.tensor([42.])#default is false
    assert tensor_a.requires_grad == False
    tensor_a = torch.tensor([42.], requires_grad=True)
    assert tensor_a.requires_grad == True
    param_a = torch.nn.parameter.Parameter(torch.tensor([42.]))#default is true.
    assert param_a.requires_grad == True
    
    #the following line trolls a lot people.
    param_a = torch.nn.parameter.Parameter(torch.tensor([42.], requires_grad=False))
    assert param_a.requires_grad == True
    
    #this is the correct way.
    param_a = torch.nn.parameter.Parameter(torch.tensor([42.]), requires_grad=False)
    assert param_a.requires_grad == False
    param_a = torch.nn.parameter.Parameter(torch.tensor([42.], requires_grad=True), requires_grad=False)
    assert param_a.requires_grad == False
    
    "also, if the dtype is int, bool, it's not allowed to require_grad."
    
    1w 类里面的情况。
    数据类型的。
    
    pass
    
    
    
if "Does it get grad?" and True:
    "if it's cuda, the inputs= must be specified."
    
    a = torch.tensor([42.], requires_grad=True)
    b = a+42.
    b.backward()
    assert a.grad is not None
    assert a.grad.eq(1.)
    
    a = torch.tensor([42.], requires_grad=True).cuda()
    b = a+42.
    b.backward()
    assert a.grad is None#!!!!!!!!!!!!!!!!
    assert a.grad is None#!!!!!!!!!!!!!!!!
    assert a.grad is None#!!!!!!!!!!!!!!!!
    "They call it ***expected behavior*** https://github.com/pytorch/pytorch/issues/171415"
    
    a = torch.tensor([42.], requires_grad=True).cuda()
    b = a+42.
    b.backward(inputs=a)
    assert a.grad is not None
    
    
    
    a = torch.tensor([42.], requires_grad=True)
    b = a+42.
    torch.autograd.backward(b, torch.tensor([1.]))
    assert a.grad is not None
    
    a = torch.tensor([42.], requires_grad=True).cuda()
    b = a+42.
    torch.autograd.backward(b, torch.tensor([1.]).cuda())
    assert a.grad is None#!!!!!!!!!!!!!!!!
    assert a.grad is None#!!!!!!!!!!!!!!!!
    assert a.grad is None#!!!!!!!!!!!!!!!!
    "They call it ***expected behavior*** https://github.com/pytorch/pytorch/issues/171415"
    
    a = torch.tensor([42.], requires_grad=True).cuda()
    b = a+42.
    torch.autograd.backward(b, torch.tensor([1.]).cuda(), inputs=a)
    assert a.grad is not None
    
    pass




how to add, remove, modify a param in module class.




class param_test(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p11 = torch.nn.parameter.Parameter(torch.tensor([11.]))
        self.register_parameter("p12", torch.nn.parameter.Parameter(torch.tensor([12.])))
        
        self.p21 = torch.nn.parameter.Parameter(torch.tensor([21.]))
        self.register_parameter("p22", torch.nn.parameter.Parameter(torch.tensor([22.])))
        self.p21 = None
        self.p22 = None
        
        self.p31 = torch.nn.parameter.Parameter(torch.tensor([31.]))
        self.register_parameter("p32", torch.nn.parameter.Parameter(torch.tensor([32.])))
        self.register_parameter("p31", None)
        self.register_parameter("p32", None)
        
        self.p41 = torch.nn.parameter.Parameter(torch.tensor([41.]))
        self.register_parameter("p42", torch.nn.parameter.Parameter(torch.tensor([42.])))
        self.p41 = torch.nn.parameter.Parameter(torch.tensor([41.1]))
        self.p42 = torch.nn.parameter.Parameter(torch.tensor([42.1]))
        
        self.p51 = torch.nn.parameter.Parameter(torch.tensor([51.]))
        self.register_parameter("p52", torch.nn.parameter.Parameter(torch.tensor([52.])))
        self.p51.data = torch.tensor([51.1])
        self.p52.data = torch.tensor([52.1])
        
        pass
    pass

pt = param_test()
aaa = [x for x in pt.parameters()]




'''
also, if you write customized autograd.Function, you see more issues.
I'll add when needed.
'''
