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



'''
also, if you write customized autograd.Function, you see more issues.
I'll add when needed.
'''
