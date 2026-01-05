import math
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
from pytorch_yagaodirac_v2.Util import _tensor_equal, _float_equal, iota
def __DEBUG_ME__()->bool:
    return __name__ == "__main__"












#<grad formula>
#<grad formula>
#<grad formula>

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

#</grad formula>
#</grad formula>
#</grad formula>











#<dynamics>
#<dynamics>
#<dynamics>
if "dynamics" and False:
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
#</dynamics>
#</dynamics>
#</dynamics>













# old code??
# loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
# input_ori__b_d  = torch.tensor([[0.,0]], requires_grad=True)
# target_ori__b_d = torch.tensor([[1.,0]])
# output_ori:torch.Tensor = loss(input_ori__b_d, target_ori__b_d)
# assert _float_equal(output_ori.item(), 0.6931)#log(2)
# assert _float_equal(output_ori.exp().item(), 2.)#
# loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
# input_ori__b_d  = torch.tensor([[0.,0,0,0]], requires_grad=True)
# target_ori__b_d = torch.tensor([[1.,0,0,0]])
# output_ori = loss(input_ori__b_d, target_ori__b_d)
# assert _float_equal(output_ori.item(), 1.3863)#
# assert _float_equal(output_ori.exp().item(), 4.)#




#<formual cracking>
#<formual cracking>
#<formual cracking>


# In this section, I wrote equivalent formula as the pytorch version.
# So you can see how the formula/algorithm actually works.

loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
input_ori__b_d  = torch.tensor([[1.,0]], requires_grad=True)
target_ori__b_d = torch.tensor([[1.,0]])
output_ori = loss(input_ori__b_d, target_ori__b_d)
assert _float_equal(output_ori.item(), 0.3133)#0.3133

loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
input_ori__b_d  = torch.tensor([[1.,0]], requires_grad=True)
target_ori__b_d = torch.tensor([[0,1.]])
output_ori = loss(input_ori__b_d, target_ori__b_d)
assert _float_equal(output_ori.item(), 1.3133)

a1 = torch.tensor([1.,0]).exp()
a_sum = a1[0]+a1[1]
a3 = a1/a_sum
assert _tensor_equal(a3,torch.tensor([ 0.7311, 0.2689]))
a4 = a3.log()
assert _tensor_equal(a4,torch.tensor([-0.3133,-1.3133]))
a5_0 = -a4[0]
assert _float_equal(a5_0, 0.3133)#0.3133
a5_1 = -a4[1]
assert _float_equal(a5_1, 1.3133)

loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
input_ori__b_d  = torch.tensor([[1.,0,0]], requires_grad=True)
target_ori__b_d = torch.tensor([[1.,0,0]])
output_ori = loss(input_ori__b_d, target_ori__b_d)
assert _float_equal(output_ori.item(), 0.5514)#0.5514

a1 = torch.tensor([1.,0,0]).exp()
a_sum = a1[0]+a1[1]+a1[2]
a3 = a1/a_sum
assert _tensor_equal(a3,torch.tensor([0.5761, 0.2119, 0.2119]))
a4 = a3.log()
assert _tensor_equal(a4,torch.tensor([-0.5514, -1.5514, -1.5514]))
a5_0 = -a4[0]
assert _float_equal(a5_0, 0.5514)#0.5514
a5_1 = -a4[1]
assert _float_equal(a5_1, 1.5514)
a5_2 = -a4[2]
assert _float_equal(a5_2, 1.5514)



# 1) exp
# 2) sum
# 3) div
# 4) log
# 5) pick(mul)












#??????????????????????
#??????????????????????
#??????????????????????
#??????????????????????
loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
input_ori__b_d  = torch.tensor([[1.]], requires_grad=True)
target_ori__b_d = torch.tensor([[1.]])
output_ori = loss(input_ori__b_d, target_ori__b_d)
assert _float_equal(output_ori.item(), 0.)

loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
input_ori__b_d  = torch.tensor([[1.,0]], requires_grad=True)
target_ori__b_d = torch.tensor([[1.,0]])
output_ori = loss(input_ori__b_d, target_ori__b_d)
assert _float_equal(output_ori.item(), 0.3133)#0.2689
output_exp = (-output_ori).exp().item()
assert _float_equal(output_exp, math.exp(1.)/(math.exp(1.)+1))

loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
input_ori__b_d  = torch.tensor([[1.,0,0]], requires_grad=True)
target_ori__b_d = torch.tensor([[1.,0,0]])
output_ori = loss(input_ori__b_d, target_ori__b_d)
assert _float_equal(output_ori.item(), 0.5514)#0.576
output_exp = (-output_ori).exp().item()
assert _float_equal(output_exp, math.exp(1.)/(math.exp(1.)+1+1))

loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
input_ori__b_d  = torch.tensor([[1.,0,0,0]], requires_grad=True)
target_ori__b_d = torch.tensor([[1.,0,0,0]])
output_ori = loss(input_ori__b_d, target_ori__b_d)
assert _float_equal(output_ori.item(), 0.7437)#0.576
output_exp = (-output_ori).exp().item()
assert _float_equal(output_exp, math.exp(1.)/(math.exp(1.)+1+1+1))
#??????????????????????
#??????????????????????
#??????????????????????
#??????????????????????
















































# target as full tensor.
loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
input_ori__b_d = torch.tensor([[1.,0,0],[1.,0,0],], requires_grad=True)
target_ori__b_d = torch.tensor([[1.,0,0],[0.,0,1],])
output_ori:torch.Tensor = loss(input_ori__b_d, target_ori__b_d)
assert _tensor_equal(output_ori, [0.5514,  1.5514])
ref_log_softmax = torch.tensor([1.,0,0]).log_softmax(dim=0)
assert _tensor_equal(output_ori[0], -ref_log_softmax[0])# 0.5514
assert _tensor_equal(output_ori[1], -ref_log_softmax[2])# 1.5514

input = input_ori__b_d.detach().clone()
log_softmax_of_input = input.log_softmax(dim=1)
target = target_ori__b_d.detach().clone()
result_raw__b_d:torch.Tensor = log_softmax_of_input*target
result__b = (-result_raw__b_d).max(dim=1).values
assert _tensor_equal(result__b, output_ori)

for _ in range(123):
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    batch = 21
    dim = 31
    input_ori__b_d = torch.randn(size=[batch,dim])
    #target = torch.tensor([0])
    _temp__target_ori__b = torch.empty(size=[batch], dtype=torch.long).random_(dim)
    target_ori__b_d = torch.zeros_like(input_ori__b_d)
    iota_of__b = iota(batch).to(torch.long)
    target_ori__b_d[iota_of__b, _temp__target_ori__b[iota_of__b]] = 1.
    output_ori = loss(input_ori__b_d, target_ori__b_d)
    
    
    #cracking
    input__b_d = input_ori__b_d.detach().clone()
    log_softmax_of_input__b_d = input__b_d.log_softmax(dim=1)
    target__b_d = target_ori__b_d.detach().clone()
    output_raw__b_d:torch.Tensor = log_softmax_of_input__b_d*target__b_d
    output__b_1 = (-output_raw__b_d).max(dim=1).values
    assert _tensor_equal(output__b_1, output_ori)
    pass

fds = 432


1w 继续。



















# target as index tensor.
loss = torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.)
batch = 1
dim = 3
input_ori__b_d = torch.tensor([[1.,0,0],[1.,0,0],], requires_grad=True)
target_ori__b = torch.tensor([0,2])
output_ori:torch.Tensor = loss(input_ori__b_d, target_ori__b_d)
assert _tensor_equal(output_ori, [0.5514,  1.5514])
ref_log_softmax = torch.tensor([1.,0,0]).log_softmax(dim=0)
assert _tensor_equal(output_ori[0], -ref_log_softmax[target_ori__b[0]])# 0.5514
assert _tensor_equal(output_ori[1], -ref_log_softmax[target_ori__b[1]])# 1.5514

#input = torch.tensor([[1.,0]], requires_grad=True)
input = input_ori__b_d.detach().clone()
log_softmax_of_input = input.log_softmax(dim=1)
target = target_ori__b.detach().clone()
result_raw__b_d:torch.Tensor = log_softmax_of_input*target
result__b = -result_raw__b_d[target]
assert _tensor_equal(result__b, output_ori)

#iota_batch__b = iota(batch, dtype=torch.int32)

for _ in range(123):
    # Example of target with class indices
    loss = torch.nn.CrossEntropyLoss()
    #input = torch.tensor([[1.5,0]], requires_grad=True)
    batch = 1
    dim = 3
    input_ori__b_d = torch.randn(size=[batch,dim])
    #target = torch.tensor([0])
    target_ori__b = torch.empty(size=[batch], dtype=torch.long).random_(dim)
    output_ori = loss(input_ori__b_d, target_ori__b)
    #assert _float_equal(output.item(), 0.3133)

    #input = torch.tensor([[1.,0]], requires_grad=True)
    
    
    # 1) exp
    # 2) sum
    # 3) div
    # 4) log
    # 5) pick(mul)

    
    input = input_ori__b_d.detach().clone()
    input_exp__b_d = input.exp()
    input_exp_sum__b = input_exp__b_d.sum(dim=1)
    input_exp__over__input_exp_sum__b_d = input_exp__b_d/(input_exp_sum__b.reshape([-1,1]).expand([-1,dim]))
    one_minus__input_exp__over__input_exp_sum__b_d = 1.-input_exp__over__input_exp_sum__b_d
    target__b_d = torch.zeros_like(input)
    iota_batch__b = iota(batch, dtype=torch.int32)
    target__b_d[iota_batch__b,target_ori__b] = 1.
    one_minus__target__b_d = 1.-target__b_d
    part1 = target__b_d*(input_exp__over__input_exp_sum__b_d.log())
    part2 = one_minus__target__b_d*(one_minus__input_exp__over__input_exp_sum__b_d.log())
    before_output = part1#+part2   ?????????????????
    output2 = -before_output.mean()
    assert _tensor_equal(output2, output_ori)
    pass

fds = 432
































































