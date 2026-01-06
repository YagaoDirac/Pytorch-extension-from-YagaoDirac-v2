import math
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, _float_equal, iota
def __DEBUG_ME__()->bool:
    return __name__ == "__main__"


'''
The result.
The formula of cross entropy is different from the bce.
Bce has both y*log(p) and (1-y)*log(1-p).
But CrossEntropy only has y*log(p).
Idk how it the formula of gradient was derived, but it looks simple, which is good.

The core behavior of cross entropy is very similar to all the sigmoid-likd layers, 
it has a balance point in the middle. Cross entropy only cares the top element, 
if it's softmax is near 0.5 or greater than 0.5, it's grad is small, but never zero.
If it's smaller than 0.5, grad is bit but never exceeds some maximum value.
The balance point works like a threshold. When the softmax of the top element goes
over 0.5, it basically stops the training. This looks neat, but if all the other points
in dataset are also considered, this balance point is too weak.

The output of cross entropy doesn't mean too much. Only the relativity means something.

I personally recommend you to make your new version. Define all the behavior, but 
inhirite all the direction(sign) from the vanilla cross entropy.

I'm working on some new version. Let's wish it would work.

The next time you interview any intern, ask "what is 0.3133 and 1.3133, 
and what is 0.7311 and 0.2689. Also, what is 0.2119." Enjoy.
'''


if "basic test" and False:
    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    input_ori__b_d  = torch.tensor([[1.]], requires_grad=True)
    target_ori__b_d = torch.tensor([[1.]])
    output_ori__b = loss(input_ori__b_d, target_ori__b_d)
    assert _float_equal(output_ori__b.item(), 0.)

    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    input_ori__b_d  = torch.tensor([[1.,0]], requires_grad=True)
    target_ori__b_d = torch.tensor([[1.,0]])
    output_ori__b = loss(input_ori__b_d, target_ori__b_d)
    assert _float_equal(output_ori__b.item(), 0.3133)
    output_exp = (-output_ori__b).exp().item()
    assert _float_equal(output_exp, math.exp(1.)/(math.exp(1.)+1))

    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    input_ori__b_d  = torch.tensor([[1.,0,0]], requires_grad=True)
    target_ori__b_d = torch.tensor([[1.,0,0]])
    output_ori__b = loss(input_ori__b_d, target_ori__b_d)
    assert _float_equal(output_ori__b.item(), 0.5514)
    output_exp = (-output_ori__b).exp().item()
    assert _float_equal(output_exp, math.exp(1.)/(math.exp(1.)+1+1))

    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    input_ori__b_d  = torch.tensor([[1.,0,0,0]], requires_grad=True)
    target_ori__b_d = torch.tensor([[1.,0,0,0]])
    output_ori__b = loss(input_ori__b_d, target_ori__b_d)
    assert _float_equal(output_ori__b.item(), 0.7437)
    output_exp = (-output_ori__b).exp().item()
    assert _float_equal(output_exp, math.exp(1.)/(math.exp(1.)+1+1+1))
    
    pass



if "gradient formula" and False:
    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    input  = torch.tensor([[1.,0,0]], requires_grad=True)
    target = torch.tensor([[1.,0,0]])
    output = loss(input, target)
    assert _float_equal(output.item(), 0.5514)
    output_exp = (-output).exp().item()
    assert _float_equal(output_exp, math.exp(1.)/(math.exp(1.)+2.))
    output.backward(inputs = input)
    assert input.grad is not None
    assert _tensor_equal(input.grad, [[-2./(math.exp(1.)+2.),  1./(math.exp(1.)+2.),  1./(math.exp(1.)+2.)]])
    assert _tensor_equal(input.grad, [[-0.4239,  0.2119,  0.2119]])

    _temp_softmax = input.softmax(dim=1)
    assert _tensor_equal(_temp_softmax, [[   math.exp(1.)/(math.exp(1.)+2.), 1./(math.exp(1.)+2.), 1./(math.exp(1.)+2.)]])
    _result = _temp_softmax-target#this line or the commented part. They both provide the same result.

    # _remove_the_selected_element = _temp_softmax*(1.-target)
    # assert _tensor_equal(_remove_the_selected_element, [[                0,  1./(math.exp(1.)+2.), 1./(math.exp(1.)+2.)]])
    # _neg_sum = -1.*_remove_the_selected_element.sum(dim=-1)
    # assert _tensor_equal(_neg_sum, [                  -2./(math.exp(1.)+2.)])
    # _part_of_selected_element = torch.zeros_like(input)
    # _part_of_selected_element += target*_neg_sum
    # assert _tensor_equal(_part_of_selected_element, [[-2./(math.exp(1.)+2.),              0,                   0]])
    # _result = _remove_the_selected_element+_part_of_selected_element
    assert _tensor_equal(_result, [[                  -2./(math.exp(1.)+2.), 1./(math.exp(1.)+2.), 1./(math.exp(1.)+2.)]])
    assert _tensor_equal(_result, input.grad)




    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    input  = torch.tensor([[1.,0,0]], requires_grad=True)
    target = torch.tensor([[0.,1,0]])
    output = loss(input, target)
    assert _float_equal(output.item(), 1.5514)
    output_exp = (-output).exp().item()
    assert _float_equal(output_exp, 1./(math.exp(1.)+2.))
    output.backward(inputs = input)
    assert input.grad is not None
    assert _tensor_equal(input.grad, [[math.exp(1.)/(math.exp(1.)+2.),  -(math.exp(1.)+1.)/(math.exp(1.)+2.),  1./(math.exp(1.)+2.)]])
    assert _tensor_equal(input.grad, [[ 0.5761, -0.7881,  0.2119]])

    _temp_softmax = input.softmax(dim=1)
    assert _tensor_equal(_temp_softmax, [[  math.exp(1.)/(math.exp(1.)+2.),         1./(math.exp(1.)+2.),        1./(math.exp(1.)+2.)]])
    _result = _temp_softmax-target#this line or the commented part. They both provide the same result.

    # _remove_the_selected_element = _temp_softmax*(1.-target)
    # assert _tensor_equal(_remove_the_selected_element, [[
    #                                         math.exp(1.)/(math.exp(1.)+2.),                0,                    1./(math.exp(1.)+2.)]])
    # _neg_sum = -1.*_remove_the_selected_element.sum(dim=-1)
    # assert _tensor_equal(_neg_sum, [                                        -(math.exp(1.)+1.)/(math.exp(1.)+2.)])
    # _part_of_selected_element = torch.zeros_like(input)
    # _part_of_selected_element += target*_neg_sum
    # assert _tensor_equal(_part_of_selected_element, [[             0,       -(math.exp(1.)+1.)/(math.exp(1.)+2.),              0]])
    # _result = _remove_the_selected_element+_part_of_selected_element
    assert _tensor_equal(_result, [[        math.exp(1.)/(math.exp(1.)+2.), -(math.exp(1.)+1.)/(math.exp(1.)+2.), 1./(math.exp(1.)+2.)]])
    assert _tensor_equal(_result, input.grad)



    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    input  = torch.tensor([[1.,0,0]], requires_grad=True)+123.#offset doesn't affect anything.
    target = torch.tensor([[0.,1,0]])
    output = loss(input, target)
    assert _float_equal(output.item(), 1.5514)
    output_exp = (-output).exp().item()
    assert _float_equal(output_exp, 1./(math.exp(1.)+2.))
    output.backward(inputs = input)
    assert input.grad is not None
    assert _tensor_equal(input.grad, [[math.exp(1.)/(math.exp(1.)+2.),  -(math.exp(1.)+1.)/(math.exp(1.)+2.),  1./(math.exp(1.)+2.)]])
    assert _tensor_equal(input.grad, [[ 0.5761, -0.7881,  0.2119]])

    pass



if "training dynamics" and False:
    # result is, the pytorch version of cross entropy balances around 0.5. If the top element is calculated into 0.5 after softmax, 
    # it's the balance point. If the element is greater than that point, then the grad is very small, if it's smaller than that 
    # point, then the grad is big and capped to some maximum value.

    "dynamics"
    dim = 30000
    input  = torch.zeros(size=[1,dim], requires_grad=True)
    target = torch.zeros(size=[1,dim])
    target[0,0] = 1.
    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    optim = torch.optim.SGD(params=[input], lr=0.5)
    for epoch in range(1,201):
        with torch.no_grad():
            input.add_(torch.randn_like(input)*1.)
            pass
        # input.requires_grad_()
        assert input.requires_grad
        
        output = loss(input, target)
        
        if epoch% 10 == 0 or epoch<10:
            print(f"epoch = {epoch}", end=", ")
            _temp_list_input = input[0].tolist()
            print(f"input = {_temp_list_input[0]:.3f}, {_temp_list_input[1]:.3f}, {_temp_list_input[2]:.3f}", end=", ")
            _temp_list_softmax = input.softmax(dim=1)[0].tolist()
            print(f"softmax = {_temp_list_softmax[0]:.3f}, {_temp_list_softmax[1]:.3f}, {_temp_list_softmax[2]:.3f}", end=", ")
            _temp_list_output = output.tolist()
            print(f"loss = {_temp_list_output[0]:.3f}", end=", ")
            pass
        
        optim.zero_grad()
        output.backward(inputs = input)
        
        if epoch% 10 == 0 or epoch<10:
            _temp_list_input_grad = input.grad[0].tolist()
            print(f"input.grad = {_temp_list_input_grad[0]:.3f}, {_temp_list_input_grad[1]:.3f}, {_temp_list_input_grad[2]:.3f}")
            pass
        
        optim.step()
        pass


    # In this comparison test, if the top element doesn't result less than 0.9 after softmax, then it's added with 
    # something similar to lr. The result shows it's locked to a much greater balance point.

    "dynamics"
    dim = 30000
    input  = torch.zeros(size=[1,dim])
    target = torch.zeros(size=[1,dim])
    target[0,0] = 1.
    loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
    lr = 0.5
    for epoch in range(1,201):
        with torch.no_grad():
            input.add_(torch.randn_like(input)*1.)
            pass
        # input.requires_grad_()
        assert input.requires_grad == False
        
        output = loss(input, target)
        
        if epoch% 10 == 0 or epoch<10:
            print(f"epoch = {epoch}", end=", ")
            _temp_list_input = input[0].tolist()
            print(f"input = {_temp_list_input[0]:.3f}, {_temp_list_input[1]:.3f}, {_temp_list_input[2]:.3f}", end=", ")
            _temp_list_softmax = input.softmax(dim=1)[0].tolist()
            print(f"softmax = {_temp_list_softmax[0]:.3f}, {_temp_list_softmax[1]:.3f}, {_temp_list_softmax[2]:.3f}", end=", ")
            _temp_list_output = output.tolist()
            print(f"loss = {_temp_list_output[0]:.3f}", end=", ")
            pass
        
        if output>0.1:#with this condition, it's similar to crossentropy but stabler.
            with torch.no_grad():
                input[0,0]+=lr
                pass
            pass
            
        if epoch% 10 == 0 or epoch<10:
            print(f"input.grad = {lr:.3f}, {0.:.3f}, {0.:.3f}")
            pass
        
        pass

    pass



if "repeat the formula again.":
    if "target as full tensor.":
        loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
        input_ori__b_d = torch.tensor([[1.,0,0],[1.,0,0],], requires_grad=True)
        target_ori__b_d = torch.tensor([[1.,0,0],[0.,0,1],])
        output_ori__b = loss(input_ori__b_d, target_ori__b_d)
        assert _tensor_equal(output_ori__b, [0.5514,  1.5514])

        _log_softmax__b_d = -(input_ori__b_d.detach().log_softmax(dim=1))
        assert _tensor_equal(_log_softmax__b_d, torch.tensor([[  0.5514, 1.5514, 1.5514],
                                                                [0.5514, 1.5514, 1.5514]]))
        assert _tensor_equal(output_ori__b[0], _log_softmax__b_d[0,0])# 0.5514
        assert _tensor_equal(output_ori__b[1], _log_softmax__b_d[1,2])# 1.5514


        output_ori__b.backward(gradient=torch.ones_like(output_ori__b), inputs=input_ori__b_d)
        _softmax__b_d = input_ori__b_d.detach().softmax(dim=1)
        assert _tensor_equal(_softmax__b_d, torch.tensor([[  0.5761, 0.2119, 0.2119],
                                                            [0.5761, 0.2119, 0.2119]]))
        assert input_ori__b_d.grad is not None
        assert _tensor_equal(input_ori__b_d.grad, torch.tensor([[-0.4239,  0.2119,  0.2119],
                                                                [ 0.5761,  0.2119, -0.7881]]))
        assert _tensor_equal(input_ori__b_d.grad, _softmax__b_d-target_ori__b_d)

        for _ in range(123):
            loss = torch.nn.CrossEntropyLoss(reduction="none")
            batch = 21
            dim = 31
            input_ori__b_d = torch.randn(size=[batch,dim], requires_grad=True)
            #target = torch.tensor([0])
            _temp__dont_use__target_ori__b = torch.empty(size=[batch], dtype=torch.long).random_(dim)
            target_ori__b_d = torch.zeros_like(input_ori__b_d)
            iota_of__b = iota(batch).to(torch.long)
            target_ori__b_d[iota_of__b, _temp__dont_use__target_ori__b[iota_of__b]] = 1.
            output_ori__b = loss(input_ori__b_d, target_ori__b_d)
            output_ori__b.backward(gradient=torch.ones_like(output_ori__b), inputs=input_ori__b_d)
            
            #cracking
            input__b_d = input_ori__b_d.detach().clone()
            target__b_d = target_ori__b_d.detach().clone()
            log_softmax_of_input__b_d = -(input__b_d.log_softmax(dim=1))
            output_raw__b_d:torch.Tensor = log_softmax_of_input__b_d*target__b_d
            output__b = output_raw__b_d.max(dim=1).values
            assert _tensor_equal(output__b, output_ori__b)
            softmax_of_input__b_d = input__b_d.softmax(dim=1)
            grad__b_d = softmax_of_input__b_d-target__b_d
            assert input_ori__b_d.grad is not None
            assert _tensor_equal(grad__b_d, input_ori__b_d.grad)
            pass
        pass#if "target as full tensor.":



    if "target as index.":
        loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
        input_ori__b_d = torch.tensor([[1.,0,0],[1.,0,0],], requires_grad=True)
        target_ori__b = torch.tensor([0,2])
        output_ori__b = loss(input_ori__b_d, target_ori__b)
        #assert isinstance(output_ori__b, torch.Tensor)
        assert _tensor_equal(output_ori__b, [0.5514,  1.5514])
        _log_softmax__b_d = -(input_ori__b_d.detach().log_softmax(dim=1))
        assert _tensor_equal(_log_softmax__b_d, torch.tensor([[  0.5514, 1.5514, 1.5514],
                                                                [0.5514, 1.5514, 1.5514]]))
        assert _tensor_equal(output_ori__b[0], _log_softmax__b_d[0,0])# 0.5514
        assert _tensor_equal(output_ori__b[1], _log_softmax__b_d[1,2])# 1.5514

        output_ori__b.backward(gradient=torch.ones_like(output_ori__b), inputs=input_ori__b_d)
        _softmax__b_d = input_ori__b_d.detach().softmax(dim=1)
        assert _tensor_equal(_softmax__b_d, torch.tensor([[  0.5761, 0.2119, 0.2119],
                                                            [0.5761, 0.2119, 0.2119]]))
        grad__b_d = _softmax__b_d.detach().clone()
        iota_of__b = iota(target_ori__b.shape[0])
        grad__b_d[iota_of__b, target_ori__b[iota_of__b]] = grad__b_d[iota_of__b, target_ori__b[iota_of__b]] -1.
        assert input_ori__b_d.grad is not None
        assert _tensor_equal(input_ori__b_d.grad, torch.tensor([[-0.4239,  0.2119,  0.2119],
                                                                [ 0.5761,  0.2119, -0.7881]]))
        assert _tensor_equal(grad__b_d, input_ori__b_d.grad)

        for _ in range(123):
            loss = torch.nn.CrossEntropyLoss(reduction="none")
            batch = 22
            dim = 32
            input_ori__b_d = torch.randn(size=[batch,dim], requires_grad=True)
            target_ori__b = torch.empty(size=[batch],dtype=torch.int64).random_(to=dim)
            output_ori__b = loss(input_ori__b_d, target_ori__b)
            output_ori__b.backward(gradient=torch.ones_like(output_ori__b), inputs=input_ori__b_d)
            
            #cracking
            input__b_d = input_ori__b_d.detach().clone()
            target__b = target_ori__b.detach().clone()
            log_softmax_of_input__b_d = -(input__b_d.log_softmax(dim=1))
            iota_of__b = iota(batch)
            output__b = log_softmax_of_input__b_d[iota_of__b, target__b[iota_of__b]]
            assert _tensor_equal(output__b, output_ori__b)
            softmax_of_input__b_d = input__b_d.softmax(dim=1)
            grad__b_d = softmax_of_input__b_d.detach().clone()
            grad__b_d[iota_of__b, target__b[iota_of__b]] = grad__b_d[iota_of__b, target__b[iota_of__b]] -1.
            assert input_ori__b_d.grad is not None
            assert _tensor_equal(grad__b_d, input_ori__b_d.grad)
            pass
        pass#if "target as full tensor.":

    pass






