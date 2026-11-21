from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _bool_equal___0_as_false, all_False, \
        _tensor_equal, _tensor_shape_check, iota, is_list_int, _float_equal
#         iota, str_the_list


import random
import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######





'''一点说明
https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html

我觉得原版里面的 λ (weight decay) 的逻辑是错的。如果要让权重慢慢的往 0 收缩，感觉有很多方法，
唯独不需要像网页上写的公式那样，而且我觉得公式都是错的，因为 t-1 的模型参数 (theta) 对 g 做了贡献以后，
g 对 b 做贡献， 然后 b 又对 g 做了一次贡献，也就是这个 weight decay 实际上的效果是超过 1 次的，
根据 μ (momentum) 的具体取值会有一些不同。总之就是增加了耦合。

我不知道是不是传统上的某些数学公式导致的。pytorch的一些细节上出错也不是一次两次了，虽说瑕不掩瑜。

现在的计划是在最后做一个关于 λ (weight decay)的等效性验证，但是不作为重点。

另外， pytorch 的文档里面还有一个 nesterov 变量。我表示完全不知道这个是干嘛的。不牵扯。'''






if "prototype before into a function" and __DEBUG_ME__() and False:
    def ____prototype____extract_info_from_sgd():

        #  init
        the_param_0 = torch.tensor([11.], requires_grad=True)
        the_param_0.grad = torch.tensor([21.])
        the_param_0.aaaaa = "===debug tag 0"
        the_param_1 = torch.tensor([12.], requires_grad=True)
        the_param_1.grad = torch.tensor([22.])
        the_param_1.aaaaa = "===debug tag 1"

        lr = 0.1
        momentum = 0.123
        dampening = 0.321
        weight_decay = 0.555
        the_sgd = torch.optim.SGD(lr=lr, params=[the_param_0, the_param_1], momentum=momentum, dampening=dampening, weight_decay = weight_decay)


        # assert  this assert is only for step 0
        #  optim.param_groups   doesn't do anything...
        assert the_sgd.param_groups.__len__() == 1
        _temp___optim_param_groups___0 = the_sgd.param_groups[0]
        assert _temp___optim_param_groups___0 is not None

        #  lr
        optim_param_groups___0___lr = _temp___optim_param_groups___0.get("lr")
        assert type(optim_param_groups___0___lr) == float
        assert optim_param_groups___0___lr == lr
        #  momentum
        optim_param_groups___0___momentum = _temp___optim_param_groups___0.get("momentum")
        assert type(optim_param_groups___0___momentum) == float
        assert optim_param_groups___0___momentum == momentum
        #  dampening
        optim_param_groups___0___dampening = _temp___optim_param_groups___0.get("dampening")
        assert type(optim_param_groups___0___dampening) == float
        assert optim_param_groups___0___dampening == dampening
        #  weight_decay
        optim_param_groups___0___weight_decay = _temp___optim_param_groups___0.get("weight_decay")
        assert type(optim_param_groups___0___weight_decay) == float
        assert optim_param_groups___0___weight_decay == weight_decay
        #  params
        _temp___optim_param_groups___0___params = _temp___optim_param_groups___0.get("params")
        assert type(_temp___optim_param_groups___0___params) == list
        assert _temp___optim_param_groups___0___params.__len__() == 2

        optim_param_groups___0___params___0 = _temp___optim_param_groups___0___params[0]
        assert isinstance(optim_param_groups___0___params___0, torch.Tensor)
        assert optim_param_groups___0___params___0 == the_param_0
        assert optim_param_groups___0___params___0 is the_param_0

        optim_param_groups___0___params___1 = _temp___optim_param_groups___0___params[1]
        assert isinstance(optim_param_groups___0___params___1, torch.Tensor)
        assert optim_param_groups___0___params___1 == the_param_1
        assert optim_param_groups___0___params___1 is the_param_1
        #  state
        assert the_sgd.state.get(optim_param_groups___0___params___0) == None
        assert the_sgd.state.get(optim_param_groups___0___params___0) is None
        assert the_sgd.state[optim_param_groups___0___params___0] == {}
        assert the_sgd.state[optim_param_groups___0___params___0].__len__() == 0
        assert (the_sgd.state[optim_param_groups___0___params___0] is {}) == False  #  false. It's bc python.

        assert the_sgd.state.get(optim_param_groups___0___params___1) == None
        assert the_sgd.state.get(optim_param_groups___0___params___1) is None
        assert the_sgd.state[optim_param_groups___0___params___1] == {}
        assert the_sgd.state[optim_param_groups___0___params___1].__len__() == 0
        assert (the_sgd.state[optim_param_groups___0___params___1] is {}) == False  #  false. It's bc python.


        for _iii in range(3):
            #  step
            #  step
            #  step
            the_sgd.step()

            # assert  
            #  optim.param_groups   doesn't do anything...
            assert the_sgd.param_groups.__len__() == 1
            _temp___optim_param_groups___0 = the_sgd.param_groups[0]
            assert _temp___optim_param_groups___0 is not None

            #  lr
            optim_param_groups___0___lr = _temp___optim_param_groups___0.get("lr")
            assert type(optim_param_groups___0___lr) == float
            assert optim_param_groups___0___lr == lr
            #  momentum
            optim_param_groups___0___momentum = _temp___optim_param_groups___0.get("momentum")
            assert type(optim_param_groups___0___momentum) == float
            assert optim_param_groups___0___momentum == momentum
            #  dampening
            optim_param_groups___0___dampening = _temp___optim_param_groups___0.get("dampening")
            assert type(optim_param_groups___0___dampening) == float
            assert optim_param_groups___0___dampening == dampening
            #  weight_decay
            optim_param_groups___0___weight_decay = _temp___optim_param_groups___0.get("weight_decay")
            assert type(optim_param_groups___0___weight_decay) == float
            assert optim_param_groups___0___weight_decay == weight_decay
            #  params
            optim_param_groups___0___param_list = _temp___optim_param_groups___0.get("params")
            assert type(optim_param_groups___0___param_list) == list
            assert optim_param_groups___0___param_list.__len__() == 2

            some_list:list[torch.Tensor] = []

            for optim_param_groups___0___param in optim_param_groups___0___param_list:
                assert isinstance(optim_param_groups___0___param, torch.Tensor)
                # assert optim_param_groups___0___param == the_param_0 or optim_param_groups___0___param == the_param_1
                # assert optim_param_groups___0___param is the_param_0 or optim_param_groups___0___param is the_param_1

                optim_state___the_param = the_sgd.state.get(optim_param_groups___0___param)
                if optim_state___the_param is not None:
                    assert type(optim_state___the_param) == dict

                    the_momentum_buffer = optim_state___the_param["momentum_buffer"]
                    assert isinstance(the_momentum_buffer, torch.Tensor)
                    #1w这儿好像还不对。继续跑
                    # 不知道为什么不对。后面看公式吧。assert the_momentum_buffer == the_param_0.grad or the_momentum_buffer == the_param_1.grad  #  prototype only
                    some_list.append(the_momentum_buffer)

                    pass

                pass

            # this part of similar to step 0 special part. But reversed.
            _temp___optim_state___the_param = the_sgd.state.get(the_param_0)

            assert the_sgd.state.get(optim_param_groups___0___params___0) != None
            assert the_sgd.state[optim_param_groups___0___params___0] != {}
            assert the_sgd.state[optim_param_groups___0___params___0].__len__() != 0

            assert the_sgd.state.get(optim_param_groups___0___params___1) != None
            assert the_sgd.state[optim_param_groups___0___params___1] != {}
            assert the_sgd.state[optim_param_groups___0___params___1].__len__() != 0

            pass# for _iii

        return 
    
    ____prototype____extract_info_from_sgd()
    pass

def extract_info_from_sgd(the_sgd:torch.optim.SGD) \
            ->tuple[list[torch.Tensor], list[torch.Tensor], float, float, float, float]:
    '''return param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay'''

    #  optim.param_groups   doesn't do anything...
    assert the_sgd.param_groups.__len__() == 1
    _temp___optim_param_groups___0 = the_sgd.param_groups[0]
    assert _temp___optim_param_groups___0 is not None

    #  lr
    lr = _temp___optim_param_groups___0.get("lr")
    assert type(lr) == float
    #  momentum
    momentum = _temp___optim_param_groups___0.get("momentum")
    assert type(momentum) == float
    #  dampening
    dampening = _temp___optim_param_groups___0.get("dampening")
    assert type(dampening) == float
    #  weight_decay
    weight_decay = _temp___optim_param_groups___0.get("weight_decay")
    assert type(weight_decay) == float
    #  params
    param_list = _temp___optim_param_groups___0.get("params")
    assert isinstance(param_list, list)
    assert type(param_list) == list

    momentum_buffer_list:list[torch.Tensor] = []

    for param in param_list:
        assert isinstance(param, torch.Tensor)   #  maybe redundent.

        optim_state___the_param = the_sgd.state.get(param)
        if optim_state___the_param is not None:
            assert type(optim_state___the_param) == dict
            if optim_state___the_param.__len__() > 1:
                assert False, "unreachable"
                pass
            elif optim_state___the_param.__len__() == 1:
                the_momentum_buffer = optim_state___the_param["momentum_buffer"]
                assert isinstance(the_momentum_buffer, torch.Tensor)
                momentum_buffer_list.append(the_momentum_buffer)
                pass
            else:
                assert optim_state___the_param.__len__() == 0
                pass

            pass

        pass

    assert type(param_list) == list
    assert type(momentum_buffer_list) == list

    return param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay
if "test" and __DEBUG_ME__() and True:
    def ____test____extract_info_from_sgd():
        if "equilavance          not a full test." and False:
            #  init
            the_param_0 = torch.tensor([11.], requires_grad=True)
            the_param_0.grad = torch.tensor([21.])
            the_param_0.aaaaa = "===debug tag 0"
            the_param_1 = torch.tensor([12.], requires_grad=True)
            the_param_1.grad = torch.tensor([22.])
            the_param_1.aaaaa = "===debug tag 1"

            lr = 0.1
            momentum = 0.123
            dampening = 0.321
            weight_decay = 0.555
            the_sgd = torch.optim.SGD(lr=lr, params=[the_param_0, the_param_1], momentum=momentum, dampening=dampening, weight_decay = weight_decay)


            # assert  this assert is only for step 0
            #  optim.param_groups   doesn't do anything...
            assert the_sgd.param_groups.__len__() == 1
            _temp___optim_param_groups___0 = the_sgd.param_groups[0]
            assert _temp___optim_param_groups___0 is not None

            #  lr
            optim_param_groups___0___lr = _temp___optim_param_groups___0.get("lr")
            assert type(optim_param_groups___0___lr) == float
            assert optim_param_groups___0___lr == lr
            #  momentum
            optim_param_groups___0___momentum = _temp___optim_param_groups___0.get("momentum")
            assert type(optim_param_groups___0___momentum) == float
            assert optim_param_groups___0___momentum == momentum
            #  dampening
            optim_param_groups___0___dampening = _temp___optim_param_groups___0.get("dampening")
            assert type(optim_param_groups___0___dampening) == float
            assert optim_param_groups___0___dampening == dampening
            #  weight_decay
            optim_param_groups___0___weight_decay = _temp___optim_param_groups___0.get("weight_decay")
            assert type(optim_param_groups___0___weight_decay) == float
            assert optim_param_groups___0___weight_decay == weight_decay
            #  params
            _temp___optim_param_groups___0___params = _temp___optim_param_groups___0.get("params")
            assert type(_temp___optim_param_groups___0___params) == list
            assert _temp___optim_param_groups___0___params.__len__() == 2

            optim_param_groups___0___params___0 = _temp___optim_param_groups___0___params[0]
            assert isinstance(optim_param_groups___0___params___0, torch.Tensor)
            assert optim_param_groups___0___params___0 == the_param_0
            assert optim_param_groups___0___params___0 is the_param_0

            optim_param_groups___0___params___1 = _temp___optim_param_groups___0___params[1]
            assert isinstance(optim_param_groups___0___params___1, torch.Tensor)
            assert optim_param_groups___0___params___1 == the_param_1
            assert optim_param_groups___0___params___1 is the_param_1
            #  state
            assert the_sgd.state.get(optim_param_groups___0___params___0) == None
            assert the_sgd.state.get(optim_param_groups___0___params___0) is None
            assert the_sgd.state[optim_param_groups___0___params___0] == {}
            assert the_sgd.state[optim_param_groups___0___params___0].__len__() == 0
            assert (the_sgd.state[optim_param_groups___0___params___0] is {}) == False  #  false. It's bc python.

            assert the_sgd.state.get(optim_param_groups___0___params___1) == None
            assert the_sgd.state.get(optim_param_groups___0___params___1) is None
            assert the_sgd.state[optim_param_groups___0___params___1] == {}
            assert the_sgd.state[optim_param_groups___0___params___1].__len__() == 0
            assert (the_sgd.state[optim_param_groups___0___params___1] is {}) == False  #  false. It's bc python.

            #  func ver
            #  func ver
            param_list, momentum_buffer_list, func_ver__lr, func_ver__momentum, func_ver__dampening, func_ver__weight_decay = \
                    extract_info_from_sgd(the_sgd=the_sgd)
            assert param_list == [the_param_0, the_param_1]
            assert param_list[0] is the_param_0
            assert param_list[1] is the_param_1
            assert momentum_buffer_list.__len__() == 0
            assert func_ver__lr == lr
            assert func_ver__momentum == momentum
            assert func_ver__dampening == dampening
            assert func_ver__weight_decay == weight_decay


            #  prototype ver,   continue
            #  prototype ver,   continue
            for _iii in range(3):
                #  step
                #  step
                #  step
                the_sgd.step()

                # assert  
                #  optim.param_groups   doesn't do anything...
                assert the_sgd.param_groups.__len__() == 1
                _temp___optim_param_groups___0 = the_sgd.param_groups[0]
                assert _temp___optim_param_groups___0 is not None

                #  lr
                optim_param_groups___0___lr = _temp___optim_param_groups___0.get("lr")
                assert type(optim_param_groups___0___lr) == float
                assert optim_param_groups___0___lr == lr
                #  momentum
                optim_param_groups___0___momentum = _temp___optim_param_groups___0.get("momentum")
                assert type(optim_param_groups___0___momentum) == float
                assert optim_param_groups___0___momentum == momentum
                #  dampening
                optim_param_groups___0___dampening = _temp___optim_param_groups___0.get("dampening")
                assert type(optim_param_groups___0___dampening) == float
                assert optim_param_groups___0___dampening == dampening
                #  weight_decay
                optim_param_groups___0___weight_decay = _temp___optim_param_groups___0.get("weight_decay")
                assert type(optim_param_groups___0___weight_decay) == float
                assert optim_param_groups___0___weight_decay == weight_decay
                #  params
                optim_param_groups___0___param_list = _temp___optim_param_groups___0.get("params")
                assert type(optim_param_groups___0___param_list) == list
                assert optim_param_groups___0___param_list.__len__() == 2

                some_list:list[torch.Tensor] = []

                for optim_param_groups___0___param in optim_param_groups___0___param_list:
                    assert isinstance(optim_param_groups___0___param, torch.Tensor)
                    # assert optim_param_groups___0___param == the_param_0 or optim_param_groups___0___param == the_param_1
                    # assert optim_param_groups___0___param is the_param_0 or optim_param_groups___0___param is the_param_1

                    optim_state___the_param = the_sgd.state.get(optim_param_groups___0___param)
                    if optim_state___the_param is not None:
                        assert type(optim_state___the_param) == dict

                        the_momentum_buffer = optim_state___the_param["momentum_buffer"]
                        assert isinstance(the_momentum_buffer, torch.Tensor)
                        #1w这儿好像还不对。继续跑
                        some_list.append(the_momentum_buffer)

                        pass

                    pass

                # this part of similar to step 0 special part. But reversed.
                _temp___optim_state___the_param = the_sgd.state.get(the_param_0)

                assert the_sgd.state.get(optim_param_groups___0___params___0) != None
                assert the_sgd.state[optim_param_groups___0___params___0] != {}
                assert the_sgd.state[optim_param_groups___0___params___0].__len__() != 0

                assert the_sgd.state.get(optim_param_groups___0___params___1) != None
                assert the_sgd.state[optim_param_groups___0___params___1] != {}
                assert the_sgd.state[optim_param_groups___0___params___1].__len__() != 0


                #  func ver 
                #  func ver
                param_list, momentum_buffer_list, func_ver__lr, func_ver__momentum, func_ver__dampening, func_ver__weight_decay = \
                        extract_info_from_sgd(the_sgd=the_sgd)
                assert param_list == [the_param_0, the_param_1]
                assert param_list[0] is the_param_0
                assert param_list[1] is the_param_1
                assert momentum_buffer_list == some_list
                assert momentum_buffer_list[0] is some_list[0]
                assert momentum_buffer_list[1] is some_list[1]
                assert func_ver__lr == lr
                assert func_ver__momentum == momentum
                assert func_ver__dampening == dampening 
                assert func_ver__weight_decay == weight_decay 

                pass# for _iii

            pass#/test

        if "ran-able test" and False:
            for with_grad in [True, False]:
                for _iii in range(33):
                    ori___lr           = random.random()+0.1
                    ori___momentum     = random.random()+0.1
                    ori___dampening    = random.random()+0.1
                    ori___weight_decay = random.random()+0.1
                    ori___param_list = []
                    for _iii in range(random.randint(2,5)):
                        ori___param_list.append(torch.rand(size=[random.randint(5,10), random.randint(5,10), random.randint(5,10)]))
                        pass
                    for _iii in range(random.randint(2,5)):
                        ori___param_list.append(torch.rand(size=[random.randint(5,10), random.randint(5,10)]))
                        pass
                    for _iii in range(random.randint(2,5)):
                        ori___param_list.append(torch.rand(size=[random.randint(5,10)]))
                        pass
                    if with_grad:
                        for param in ori___param_list:
                            param.grad = torch.randn_like(param)
                            pass
                        pass#if with_grad:
                    random.shuffle(ori___param_list)
                    the_sgd = torch.optim.SGD(params=ori___param_list, lr=ori___lr, momentum=ori___momentum, \
                                dampening=ori___dampening, weight_decay=ori___weight_decay)

                    param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=the_sgd)
                    assert param_list == ori___param_list
                    assert momentum_buffer_list.__len__() == 0
                    assert lr           == ori___lr
                    assert momentum     == ori___momentum
                    assert dampening    == ori___dampening
                    assert weight_decay == ori___weight_decay

                    the_sgd.step()
                    
                    param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=the_sgd)
                    assert param_list == ori___param_list
                    if with_grad:
                        assert momentum_buffer_list.__len__() == param_list.__len__()
                        pass
                    else:
                        assert momentum_buffer_list.__len__() == 0
                        pass
                    assert lr           == ori___lr
                    assert momentum     == ori___momentum
                    assert dampening    == ori___dampening
                    assert weight_decay == ori___weight_decay

                    the_sgd.step()
                    
                    param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=the_sgd)
                    assert param_list == ori___param_list
                    if with_grad:
                        assert momentum_buffer_list.__len__() == param_list.__len__()
                        pass
                    else:
                        assert momentum_buffer_list.__len__() == 0
                        pass
                    assert lr           == ori___lr
                    assert momentum     == ori___momentum
                    assert dampening    == ori___dampening
                    assert weight_decay == ori___weight_decay

                    pass#for _iii
                pass# for with_grad

            pass#/test

        if "some manual value test" and True:
            ori___lr = 0.
            ori___momentum = 0.3
            ori___dampening = 0.
            ori___weight_decay = 0.#keep this 0.
            assert ori___weight_decay == 0.

            the_param = torch.tensor([10.], requires_grad=True)
            the_param.grad = torch.tensor([2.])

            the_sgd = torch.optim.SGD(lr=ori___lr, params=[the_param], momentum=ori___momentum, dampening=ori___dampening,
                                                                                            weight_decay=ori___weight_decay)

            param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=the_sgd)
            assert momentum_buffer_list.__len__() == 0
            # prin(f"param {param_list[0]}   grad {param_list[0].grad}   buffer {momentum_buffer_list}   lr {lr}   momentum {momentum}")
            the_sgd.step()

            param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=the_sgd)
            assert momentum_buffer_list.__len__() == 1
            momentum_buffer = momentum_buffer_list[0]
            assert isinstance(momentum_buffer, torch.Tensor)
            assert _tensor_equal(momentum_buffer, the_param.grad)  #  !!!
            # prin(f"param {param_list[0]}   grad {param_list[0].grad}   buffer {momentum_buffer_list}   lr {lr}   momentum {momentum}")
            the_sgd.step()

            param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=the_sgd)
            assert momentum_buffer_list.__len__() == 1
            momentum_buffer = momentum_buffer_list[0]
            assert isinstance(momentum_buffer, torch.Tensor)
            assert _tensor_equal(momentum_buffer, the_param.grad*ori___momentum + the_param.grad)  #  !!!
            # prin(f"param {param_list[0]}   grad {param_list[0].grad}   buffer {momentum_buffer_list}   lr {lr}   momentum {momentum}")
            the_sgd.step()

            param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=the_sgd)
            assert momentum_buffer_list.__len__() == 1
            momentum_buffer = momentum_buffer_list[0]
            assert isinstance(momentum_buffer, torch.Tensor)
            assert _tensor_equal(momentum_buffer, (the_param.grad*ori___momentum + the_param.grad)*ori___momentum + the_param.grad)  #  !!!
            # prin(f"param {param_list[0]}   grad {param_list[0].grad}   buffer {momentum_buffer_list}   lr {lr}   momentum {momentum}")
            the_sgd.step()
            pass#/test

        return 
    ____test____extract_info_from_sgd()
    pass











def maybe_pytorch_sgd_step(model_param:torch.Tensor, hidden_state___the_momentum_buffer:torch.Tensor, first_time:bool, 
                        lr = 0., momentum = 0., dampening = 0., weight_decay = 0.)->tuple[torch.Tensor, torch.Tensor]:
    '''return new_model_param, new_hidden_state___the_momentum_buffer'''


    gard__plus__weight_decay__mul__model_param = model_param.grad + weight_decay*model_param

    if hidden_state___the_momentum_buffer is None:
        assert False, "param:hidden_state___the_momentum_buffer can not be None. Set it to torch.zeros_like(param:model_param)"
        pass
    if first_time:
        new_hidden_state___the_momentum_buffer = gard__plus__weight_decay__mul__model_param.detach().clone() 
        # There's no "(1. - dampening)*".  This is exactly the reason you should NEVER trust a math tool if you don't know the formula.
        pass
    else:
        new_hidden_state___the_momentum_buffer = momentum*hidden_state___the_momentum_buffer + (1. - dampening)*gard__plus__weight_decay__mul__model_param
        pass

    assert isinstance(new_hidden_state___the_momentum_buffer, torch.Tensor)

    grad_to_update_param = new_hidden_state___the_momentum_buffer#.detach().clone()  no need. 
    new_model_param = model_param - lr*grad_to_update_param
    del grad_to_update_param
    return new_model_param, new_hidden_state___the_momentum_buffer

if "test" and __DEBUG_ME__() and False:
    def ____test____maybe_pytorch_sgd():

        if "equilavance" and True:
            #  hyper param
            for _iii in range(66):
                ori___lr = random.random()*0.5+0.1
                assert ori___lr > 0. and ori___lr < 1.
                #ori___lr = 0.5  
                ori___momentum = random.random()*0.5+0.1
                assert ori___momentum > 0. and ori___momentum < 1.
                #ori___momentum = 0.4  
                ori___dampening = random.random()*0.5+0.1
                assert ori___dampening > 0. and ori___dampening < 1.
                #ori___dampening = 0.1  #    !!!!!!!!!!!!!!  现在在研究这个
                ori___weight_decay = random.random()*0.5+0.1
                assert ori___weight_decay > 0. and ori___weight_decay < 1.
                #ori___weight_decay = 0.   #random.random()*0.5+0.1

                # prin(f"hyper   lr {ori___lr}   momentum {ori___momentum}   dampening {ori___dampening}   weight_decay {ori___weight_decay}")
                #  ori
                ori___param = torch.randn(size=[10])
                ori___grad = torch.randn_like(ori___param)

                #  pytorch ver
                #  step 0
                pytorch_ver___param = ori___param.detach().clone()
                pytorch_ver___param.requires_grad_()
                pytorch_ver___param.grad = ori___grad.detach().clone()
                
                pytorch_ver___param___step_0 = pytorch_ver___param.detach().clone()
                pytorch_ver___param___step_0.grad = pytorch_ver___param.grad.detach().clone()

                pytorch_ver___sgd = torch.optim.SGD(lr=ori___lr, params=[pytorch_ver___param], momentum=ori___momentum, dampening=ori___dampening,
                                                                                                weight_decay=ori___weight_decay)


                param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=pytorch_ver___sgd)
                assert param_list.__len__() == 1
                assert param_list[0] is pytorch_ver___param
                assert momentum_buffer_list.__len__() == 0
                assert lr           == ori___lr
                assert momentum     == ori___momentum
                assert dampening    == ori___dampening
                assert weight_decay == ori___weight_decay
                # prin(f"param___step_0 {pytorch_ver___param___step_0}   param___step_0.grad {pytorch_ver___param___step_0.grad}")


                pytorch_ver___sgd.step()# 0 -> 1
                pytorch_ver___param___step_1 = pytorch_ver___param.detach().clone()
                pytorch_ver___param___step_1.grad = pytorch_ver___param.grad.detach().clone()
                assert _tensor_equal(pytorch_ver___param___step_0, pytorch_ver___param___step_1) == False
                pytorch_ver___param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=pytorch_ver___sgd)
                assert pytorch_ver___param_list == [pytorch_ver___param]
                assert pytorch_ver___param_list[0] is pytorch_ver___param
                assert momentum_buffer_list.__len__() == 1
                assert lr           == ori___lr
                assert momentum     == ori___momentum
                assert dampening    == ori___dampening
                assert weight_decay == ori___weight_decay
                pytorch_ver___momentum_buffer___step_1 = momentum_buffer_list[0].detach().clone()
                # prin(f"param___step_1 {pytorch_ver___param___step_1}   param___step_1.grad {pytorch_ver___param___step_1.grad}   momentum {pytorch_ver___momentum_buffer___step_1}")


                pytorch_ver___sgd.step()# 1 -> 2
                pytorch_ver___param___step_2 = pytorch_ver___param.detach().clone()
                pytorch_ver___param___step_2.grad = pytorch_ver___param.grad.detach().clone()
                assert _tensor_equal(pytorch_ver___param___step_1, pytorch_ver___param___step_2) == False
                pytorch_ver___param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=pytorch_ver___sgd)
                assert pytorch_ver___param_list == [pytorch_ver___param]
                assert pytorch_ver___param_list[0] is pytorch_ver___param
                assert momentum_buffer_list.__len__() == 1
                assert lr           == ori___lr
                assert momentum     == ori___momentum
                assert dampening    == ori___dampening
                assert weight_decay == ori___weight_decay
                pytorch_ver___momentum_buffer___step_2 = momentum_buffer_list[0].detach().clone()
                assert _tensor_equal(pytorch_ver___momentum_buffer___step_1, pytorch_ver___momentum_buffer___step_2) == False
                # prin(f"param___step_2 {pytorch_ver___param___step_2}   param___step_2.grad {pytorch_ver___param___step_2.grad}   momentum {pytorch_ver___momentum_buffer___step_2}")


                pytorch_ver___sgd.step()# 2 -> 3
                pytorch_ver___param___step_3 = pytorch_ver___param.detach().clone()
                pytorch_ver___param___step_3.grad = pytorch_ver___param.grad.detach().clone()
                assert _tensor_equal(pytorch_ver___param___step_2, pytorch_ver___param___step_3) == False
                pytorch_ver___param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=pytorch_ver___sgd)
                assert pytorch_ver___param_list == [pytorch_ver___param]
                assert pytorch_ver___param_list[0] is pytorch_ver___param
                assert momentum_buffer_list.__len__() == 1
                assert lr           == ori___lr
                assert momentum     == ori___momentum
                assert dampening    == ori___dampening
                assert weight_decay == ori___weight_decay
                pytorch_ver___momentum_buffer___step_3 = momentum_buffer_list[0].detach().clone()
                assert _tensor_equal(pytorch_ver___momentum_buffer___step_2, pytorch_ver___momentum_buffer___step_3) == False
                # prin(f"param___step_3 {pytorch_ver___param___step_3}   param___step_3.grad {pytorch_ver___param___step_3.grad}   momentum {pytorch_ver___momentum_buffer___step_3}")


                pytorch_ver___sgd.step()# 3 -> 4
                pytorch_ver___param___step_4 = pytorch_ver___param.detach().clone()
                pytorch_ver___param___step_4.grad = pytorch_ver___param.grad.detach().clone()
                assert _tensor_equal(pytorch_ver___param___step_3, pytorch_ver___param___step_4) == False
                pytorch_ver___param_list, momentum_buffer_list, lr, momentum, dampening, weight_decay = extract_info_from_sgd(the_sgd=pytorch_ver___sgd)
                assert pytorch_ver___param_list == [pytorch_ver___param]
                assert pytorch_ver___param_list[0] is pytorch_ver___param
                assert momentum_buffer_list.__len__() == 1
                assert lr           == ori___lr
                assert momentum     == ori___momentum
                assert dampening    == ori___dampening
                assert weight_decay == ori___weight_decay
                pytorch_ver___momentum_buffer___step_4 = momentum_buffer_list[0].detach().clone()
                assert _tensor_equal(pytorch_ver___momentum_buffer___step_3, pytorch_ver___momentum_buffer___step_4) == False
                # prin(f"param___step_4 {pytorch_ver___param___step_4}   param___step_4.grad {pytorch_ver___param___step_4.grad}   momentum {pytorch_ver___momentum_buffer___step_4}")





                #  maybe ver
                #  maybe ver
                maybe_pytorch_ver___param___step_0 = ori___param.detach().clone()
                maybe_pytorch_ver___param___step_0.requires_grad_()
                maybe_pytorch_ver___param___step_0.grad = ori___grad.detach().clone()
                assert _tensor_equal(pytorch_ver___param___step_0,      maybe_pytorch_ver___param___step_0)
                assert _tensor_equal(pytorch_ver___param___step_0.grad, maybe_pytorch_ver___param___step_0.grad)
                # prin(f"pytorch   param___step_0 {      pytorch_ver___param___step_0}   param___step_0.grad {      pytorch_ver___param___step_0.grad}")
                # prin(f"maybe ver param___step_0 {maybe_pytorch_ver___param___step_0}   param___step_0.grad {maybe_pytorch_ver___param___step_0.grad}")

                maybe_pytorch_ver___param___step_1, maybe_pytorch_ver___momentum_buffer___step_1 = \
                        maybe_pytorch_sgd_step(model_param=maybe_pytorch_ver___param___step_0, 
                                hidden_state___the_momentum_buffer = torch.zeros_like(maybe_pytorch_ver___param___step_0), 
                                first_time=True, 
                                lr=ori___lr, momentum=ori___momentum, dampening=ori___dampening, weight_decay=ori___weight_decay) # 0 -> 1
                maybe_pytorch_ver___param___step_1.requires_grad_()
                maybe_pytorch_ver___param___step_1.grad = ori___grad.detach().clone()
                
                assert _tensor_equal(          pytorch_ver___param___step_1,                maybe_pytorch_ver___param___step_1)
                assert _tensor_equal(          pytorch_ver___param___step_1.grad,           maybe_pytorch_ver___param___step_1.grad)
                assert _tensor_equal(pytorch_ver___momentum_buffer___step_1,      maybe_pytorch_ver___momentum_buffer___step_1)
                # prin(f"pytorch   param___step_1 {      pytorch_ver___param___step_1.item():.4f}   param___step_1.grad {      pytorch_ver___param___step_1.grad}   momentum {      pytorch_ver___momentum_buffer___step_1}")
                # prin(f"maybe ver param___step_1 {maybe_pytorch_ver___param___step_1.item():.4f}   param___step_1.grad {maybe_pytorch_ver___param___step_1.grad}   momentum {maybe_pytorch_ver___momentum_buffer___step_1}")



                maybe_pytorch_ver___param___step_2, maybe_pytorch_ver___momentum_buffer___step_2 = \
                        maybe_pytorch_sgd_step(model_param=maybe_pytorch_ver___param___step_1, 
                                hidden_state___the_momentum_buffer = pytorch_ver___momentum_buffer___step_1, 
                                first_time=False, 
                                lr=ori___lr, momentum=ori___momentum, dampening=ori___dampening, weight_decay=ori___weight_decay) # 1 -> 2
                maybe_pytorch_ver___param___step_2.requires_grad_()
                maybe_pytorch_ver___param___step_2.grad = ori___grad.detach().clone()
                
                assert _tensor_equal(          pytorch_ver___param___step_2,                maybe_pytorch_ver___param___step_2)
                assert _tensor_equal(          pytorch_ver___param___step_2.grad,           maybe_pytorch_ver___param___step_2.grad)
                assert _tensor_equal(pytorch_ver___momentum_buffer___step_2,      maybe_pytorch_ver___momentum_buffer___step_2)
                # prin(f"pytorch   param___step_2 {      pytorch_ver___param___step_2.item():.4f}   param___step_2.grad {      pytorch_ver___param___step_2.grad}   momentum {      pytorch_ver___momentum_buffer___step_2}")
                # prin(f"maybe ver param___step_2 {maybe_pytorch_ver___param___step_2.item():.4f}   param___step_2.grad {maybe_pytorch_ver___param___step_2.grad}   momentum {maybe_pytorch_ver___momentum_buffer___step_2}")



                maybe_pytorch_ver___param___step_3, maybe_pytorch_ver___momentum_buffer___step_3 = \
                        maybe_pytorch_sgd_step(model_param=maybe_pytorch_ver___param___step_2, 
                                hidden_state___the_momentum_buffer = pytorch_ver___momentum_buffer___step_2, 
                                first_time=False, 
                                lr=ori___lr, momentum=ori___momentum, dampening=ori___dampening, weight_decay=ori___weight_decay) # 2 -> 3
                maybe_pytorch_ver___param___step_3.requires_grad_()
                maybe_pytorch_ver___param___step_3.grad = ori___grad.detach().clone()
                
                assert _tensor_equal(          pytorch_ver___param___step_3,                maybe_pytorch_ver___param___step_3)
                assert _tensor_equal(          pytorch_ver___param___step_3.grad,           maybe_pytorch_ver___param___step_3.grad)
                assert _tensor_equal(pytorch_ver___momentum_buffer___step_3,      maybe_pytorch_ver___momentum_buffer___step_3)
                # prin(f"pytorch   param___step_3 {      pytorch_ver___param___step_3.item():.4f}   param___step_3.grad {      pytorch_ver___param___step_3.grad}   momentum {      pytorch_ver___momentum_buffer___step_3}")
                # prin(f"maybe ver param___step_3 {maybe_pytorch_ver___param___step_3.item():.4f}   param___step_3.grad {maybe_pytorch_ver___param___step_3.grad}   momentum {maybe_pytorch_ver___momentum_buffer___step_3}")



                maybe_pytorch_ver___param___step_4, maybe_pytorch_ver___momentum_buffer___step_4 = \
                        maybe_pytorch_sgd_step(model_param=maybe_pytorch_ver___param___step_3, 
                                hidden_state___the_momentum_buffer = maybe_pytorch_ver___momentum_buffer___step_3, 
                                first_time=False, 
                                lr=ori___lr, momentum=ori___momentum, dampening=ori___dampening, weight_decay=ori___weight_decay) # 3 -> 4
                maybe_pytorch_ver___param___step_4.requires_grad_()
                maybe_pytorch_ver___param___step_4.grad = ori___grad.detach().clone()
                
                assert _tensor_equal(          pytorch_ver___param___step_4,                maybe_pytorch_ver___param___step_4)
                assert _tensor_equal(          pytorch_ver___param___step_4.grad,           maybe_pytorch_ver___param___step_4.grad)
                assert _tensor_equal(pytorch_ver___momentum_buffer___step_4,      maybe_pytorch_ver___momentum_buffer___step_4)
                # prin(f"pytorch   param___step_4 {      pytorch_ver___param___step_4.item():.4f}   param___step_4.grad {      pytorch_ver___param___step_4.grad}   momentum {      pytorch_ver___momentum_buffer___step_4}")
                # prin(f"maybe ver param___step_4 {maybe_pytorch_ver___param___step_4.item():.4f}   param___step_4.grad {maybe_pytorch_ver___param___step_4.grad}   momentum {maybe_pytorch_ver___momentum_buffer___step_4}")

                pass#for _iii

            pass#/test

        return 
    ____test____maybe_pytorch_sgd()
    pass








'''conclusion:
momentum factor must be >= 0.
When momentum factor is 0, the dampening doesn't affect the result.
If momentum factor > 1., the update accelarates. Not ideal.
If the momentum factor is not 0., while the dampening factor is > 1., the update may go to the opposite direction. It's wrong.
'''
if "VISUAL        plot the values" and __DEBUG_ME__() and True:
    import matplotlib.pyplot as plt
    def ____VISUAL____plot_the_values():
        
        '''all the momentum == 0. overlap into one line.'''
        if "all the possible combinations. Including wrong ones." and False:
            plot_the_grad = True
            random_shift = 0.1
            #  plt init
            fig, ax = plt.subplots()

            #grad_list = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]
            grad_list = [2.] * 10
            _iota = iota(grad_list.__len__())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            start_param = 0.
            for momentum, linestyle, linewidth in zip([0., 0.1, 0.5, ], [":", "-.", "-"], [6,4,2]):
                #dampening = 0.
                for dampening, linecolor in zip([-0.1, 0., 0.1, 0.5, 1.5], ["red", "orange", "green", "blue", "black"]):
                    # if momentum == 0. and dampening != 0.:
                    #     continue
                    #  result container
                    result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                    result_container.fill_(torch.nan)
                    
                    #  calc
                    _temp_tensor = torch.tensor([start_param], requires_grad=True)
                    the_sgd = torch.optim.SGD(params = [_temp_tensor], lr = lr, momentum = momentum, dampening = dampening)
                    for ii in range(grad_list.__len__()):
                        _temp_tensor.grad = torch.tensor([grad_list[ii]])
                        the_sgd.step()
                        result_container[ii] = _temp_tensor
                        pass#for ii

                    assert all_False(result_container.isnan())
                    ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                            label = f"m {momentum:.1f} d {dampening:.1f}", linestyle = linestyle, linewidth = linewidth, c = linecolor, alpha = 0.7)
                    pass#for dampening
                pass#for momentum
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test


        '''when the grad is 1,0,0,0..., or only the first time the grad is non 0, the dampening doesn't affect the result.
        Reason is probably the pytorch sgd has a lil bit warm up.
        Dampening is a simple nerf of the input grad. Nothing fancy.
        
        The effect lasts around 7 steps.'''
        if "If the grad is 1,0,0,0....  or  0,0,1,0,0,0..." and False:
            plot_the_grad = True
            random_shift = 0.02
            #  plt init
            fig, ax = plt.subplots()

            grad_list = [0.] * 15
            #grad_list[0] = 2.
            #grad_list[1] = 2.
            grad_list[2] = 2.
            #basically [2., 0,0,0,0,0,...]

            _iota = iota(grad_list.__len__())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            start_param = 0.
            for momentum, linestyle, linewidth in zip([0., 0.1, 0.5, ], [":", "-.", "-"], [6,4,2]):
                #dampening = 0.
                for dampening, linecolor in zip([0., 0.1, 0.5], ["red", "orange", "green"]):#, "blue", "black"]):
                    if momentum == 0. and dampening != 0.:
                        continue
                    #  result container
                    result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                    result_container.fill_(torch.nan)
                    
                    #  calc
                    _temp_tensor = torch.tensor([start_param], requires_grad=True)
                    the_sgd = torch.optim.SGD(params = [_temp_tensor], lr = lr, momentum = momentum, dampening = dampening)
                    for ii in range(grad_list.__len__()):
                        _temp_tensor.grad = torch.tensor([grad_list[ii]])
                        the_sgd.step()
                        result_container[ii] = _temp_tensor
                        pass#for ii

                    assert all_False(result_container.isnan())
                    ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                            label = f"m {momentum:.1f} d {dampening:.1f}", linestyle = linestyle, linewidth = linewidth, c = linecolor, alpha = 0.7)
                    pass#for dampening
                pass#for momentum
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test


        '''These param in sgd doesn't affect the shape significantly.'''
        if "If the grad is random" and False:
            plot_the_grad = True
            random_shift = 0.02
            #  plt init
            fig, ax = plt.subplots()

            #  grad
            grad_list = torch.randn(size=[1000])
            grad_list[100] = 30.


            _iota = iota(grad_list.nelement())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            start_param = 0.
            for momentum, linestyle, linewidth in zip([0., 0.3, 0.6, ], [":", "-.", "-"], [3,2,1]):
                #dampening = 0.
                for dampening, linecolor in zip([0., 0.3, 0.6], ["red", "orange", "green"]):#, "blue", "black"]):
                    if momentum == 0. and dampening != 0.:
                        continue
                    #  result container
                    result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                    result_container.fill_(torch.nan)
                    
                    #  calc
                    _temp_tensor = torch.tensor([start_param], requires_grad=True)
                    the_sgd = torch.optim.SGD(params = [_temp_tensor], lr = lr, momentum = momentum, dampening = dampening)
                    for ii in range(grad_list.__len__()):
                        _temp_tensor.grad = torch.tensor([grad_list[ii]])
                        the_sgd.step()
                        result_container[ii] = _temp_tensor
                        pass#for ii

                    assert all_False(result_container.isnan())
                    ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                            label = f"m {momentum:.1f} d {dampening:.1f}", linestyle = linestyle, linewidth = linewidth, c = linecolor, alpha = 0.7)
                    pass#for dampening
                pass#for momentum
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test


        if "If the momentum == dampening " and False:
            plot_the_grad = True
            random_shift = 0.0
            #  plt init
            fig, ax = plt.subplots()

            #  grad
            grad_list = torch.randn(size=[1000])
            grad_list[100] = 20.


            _iota = iota(grad_list.nelement())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            start_param = 0.
            for momentum, linewidth, linecolor in zip([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ], 
                                                        [7, 6,5,4,3,2,1], 
                                            ["red", "orange", "#C0B600", "green", "blue", "darkorchid", "fuchsia"] ):
                dampening = momentum   #  v
                #dampening = 1.-(momentum)
                #  result container
                result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                result_container.fill_(torch.nan)
                
                #  calc
                _temp_tensor = torch.tensor([start_param], requires_grad=True)
                the_sgd = torch.optim.SGD(params = [_temp_tensor], lr = lr, momentum = momentum, dampening = dampening)
                for ii in range(grad_list.__len__()):
                    _temp_tensor.grad = torch.tensor([grad_list[ii]])
                    the_sgd.step()
                    result_container[ii] = _temp_tensor
                    pass#for ii

                assert all_False(result_container.isnan())
                ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                        label = f"m {momentum:.1f} d {dampening:.1f}", linewidth = linewidth, c = linecolor, alpha = 0.7)
                pass#for momentum
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test


        return 
    ____VISUAL____plot_the_values()















    def ____VISUAL____plot_the_values_________adam():
        '''conclusion: both beta param can only be [0., 1.). '''
        if "param possibility" and False:
            _temp_tensor = torch.tensor([0.], requires_grad=True)
            the_adam = torch.optim.Adam(params = [_temp_tensor], lr = 0.1, betas = ( 0. ,  0. ))
            the_adam = torch.optim.Adam(params = [_temp_tensor], lr = 0.1, betas = ( 0.5,  0. ))

            the_adam = torch.optim.Adam(params = [_temp_tensor], lr = 0.1, betas = ( 0. ,  0.4 ))
            the_adam = torch.optim.Adam(params = [_temp_tensor], lr = 0.1, betas = ( 0.5,  0.4 ))
            pass#/test


        '''conclusion: if grad is constant, adam does nothing. It's a plain optim.'''
        if "all the possible combinations." and False:
            plot_the_grad = True
            random_shift = 0.
            #  plt init
            fig, ax = plt.subplots()

            #grad_list = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]
            grad_list = [2.] * 10
            _iota = iota(grad_list.__len__())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            eps = 1e-7#1e-8
            start_param = 0.

            for amsgrad, linestyle in zip([False, True ], [":",  "-"]):
                for betas_1, linecolor in zip([0.0, 0.3, 0.6, 0.9], ["red", "orange", "green", "blue", "black"]):
                    for betas_2, linewidth in zip([0.0, 0.3, 0.6, 0.9], [4,3,2,1]):
                            
                        #  result container
                        result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                        result_container.fill_(torch.nan)
                        
                        #  calc
                        _temp_tensor = torch.tensor([start_param], requires_grad=True)
                        the_adam = torch.optim.Adam(params = [_temp_tensor], lr = lr, 
                                                    betas = (betas_1, betas_2), eps = eps, amsgrad = amsgrad,)
                        
                        for ii in range(grad_list.__len__()):
                            _temp_tensor.grad = torch.tensor([grad_list[ii]])
                            the_adam.step()
                            result_container[ii] = _temp_tensor
                            pass#for ii

                        assert all_False(result_container.isnan())
                        if amsgrad:
                            ams_str = "v"
                            pass
                        else:
                            ams_str = "x"
                            pass
                        
                        ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                                label = f"b1 {betas_1:.1f} b2 {betas_2:.1f} ams {ams_str}", 
                                linestyle = linestyle, linewidth = linewidth, c = linecolor, alpha = 0.7)
                    pass#for betas_2
                pass#for betas_1           
            pass#for amsgrad
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test


        '''conclusion: at least one of beta_2 and amsgrad must be there, otherwise the update behavior may be unstable.
        No matter even the eps == 1e-2'''
        if "grad is randn" and False:
            plot_the_grad = True
            random_shift = 0.
            #  plt init
            fig, ax = plt.subplots()

            grad_list = torch.randn(size=[1000])
            _iota = iota(grad_list.__len__())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            eps = 1e-3#1e-8
            start_param = 0.

            for amsgrad, linestyle in zip([False, True ], [":",  "-"]):
                for betas_1, linecolor in zip([0.0, 0.3, 0.6, 0.9], ["red", "orange", "green", "blue", "black"]):
                    for betas_2, linewidth in zip([0.0, 0.3, 0.6, 0.9], [4,3,2,1]):
                            
                        #  result container
                        result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                        result_container.fill_(torch.nan)
                        
                        #  calc
                        _temp_tensor = torch.tensor([start_param], requires_grad=True)
                        the_adam = torch.optim.Adam(params = [_temp_tensor], lr = lr, 
                                                    betas = (betas_1, betas_2), eps = eps, amsgrad = amsgrad,)
                        
                        for ii in range(grad_list.__len__()):
                            _temp_tensor.grad = torch.tensor([grad_list[ii]])
                            the_adam.step()
                            result_container[ii] = _temp_tensor
                            pass#for ii

                        assert all_False(result_container.isnan())
                        if amsgrad:
                            ams_str = "v"
                            pass
                        else:
                            ams_str = "x"
                            pass
                        
                        ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                                label = f"b1 {betas_1:.1f} b2 {betas_2:.1f} ams {ams_str}", 
                                linestyle = linestyle, linewidth = linewidth, c = linecolor, alpha = 0.7)
                    pass#for betas_2
                pass#for betas_1           
            pass#for amsgrad
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test
        










1w jixu  4个角的具体情况。  ams grad 永远是关。

        '''conclusion: at least one of beta_2 and amsgrad must be there, otherwise the update behavior may be unstable.'''
        if "4 corners without amsgrad" and True:
            plot_the_grad = True
            random_shift = 0.
            #  plt init
            fig, ax = plt.subplots()

            grad_list = torch.randn(size=[1000])
            _iota = iota(grad_list.__len__())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            eps = 1e-3#1e-8
            start_param = 0.

            for betas_1, linecolor in zip([0.0, 0.3, 0.6, 0.9], ["red", "orange", "green", "blue", "black"]):
                for betas_2, linewidth in zip([0.0, 0.3, 0.6, 0.9], [4,3,2,1]):
                        
                    #  result container
                    result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                    result_container.fill_(torch.nan)
                    
                    #  calc
                    _temp_tensor = torch.tensor([start_param], requires_grad=True)
                    the_adam = torch.optim.Adam(params = [_temp_tensor], lr = lr, 
                                                betas = (betas_1, betas_2), eps = eps, amsgrad = False)
                    
                    for ii in range(grad_list.__len__()):
                        _temp_tensor.grad = torch.tensor([grad_list[ii]])
                        the_adam.step()
                        result_container[ii] = _temp_tensor
                        pass#for ii

                    assert all_False(result_container.isnan())
                    # if amsgrad:
                    #     ams_str = "v"
                    #     pass
                    # else:
                    #     ams_str = "x"
                    #     pass
                    
                    ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                            label = f"b1 {betas_1:.1f} b2 {betas_2:.1f}", 
                            linewidth = linewidth, c = linecolor, alpha = 0.7)
                            #linestyle = linestyle, linewidth = linewidth, c = linecolor, alpha = 0.7)
                pass#for betas_2
            pass#for betas_1           
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test
        








        assert False, "jixu"


        '''when the grad is 1,0,0,0..., or only the first time the grad is non 0, the dampening doesn't affect the result.
        Reason is probably the pytorch sgd has a lil bit warm up.
        Dampening is a simple nerf of the input grad. Nothing fancy.
        
        The effect lasts around 7 steps.'''
        if "If the grad is 1,0,0,0....  or  0,0,1,0,0,0..." and False:
            plot_the_grad = True
            random_shift = 0.02
            #  plt init
            fig, ax = plt.subplots()

            grad_list = [0.] * 15
            #grad_list[0] = 2.
            #grad_list[1] = 2.
            grad_list[2] = 2.
            #basically [2., 0,0,0,0,0,...]

            _iota = iota(grad_list.__len__())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            start_param = 0.
            for momentum, linestyle, linewidth in zip([0., 0.1, 0.5, ], [":", "-.", "-"], [6,4,2]):
                #dampening = 0.
                for dampening, linecolor in zip([0., 0.1, 0.5], ["red", "orange", "green"]):#, "blue", "black"]):
                    if momentum == 0. and dampening != 0.:
                        continue
                    #  result container
                    result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                    result_container.fill_(torch.nan)
                    
                    #  calc
                    _temp_tensor = torch.tensor([start_param], requires_grad=True)
                    the_sgd = torch.optim.SGD(params = [_temp_tensor], lr = lr, momentum = momentum, dampening = dampening)
                    for ii in range(grad_list.__len__()):
                        _temp_tensor.grad = torch.tensor([grad_list[ii]])
                        the_sgd.step()
                        result_container[ii] = _temp_tensor
                        pass#for ii

                    assert all_False(result_container.isnan())
                    ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                            label = f"m {momentum:.1f} d {dampening:.1f}", linestyle = linestyle, linewidth = linewidth, c = linecolor, alpha = 0.7)
                    pass#for dampening
                pass#for momentum
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test


        '''These param in sgd doesn't affect the shape significantly.'''
        if "If the grad is random" and False:
            plot_the_grad = True
            random_shift = 0.02
            #  plt init
            fig, ax = plt.subplots()

            #  grad
            grad_list = torch.randn(size=[1000])
            grad_list[100] = 30.


            _iota = iota(grad_list.nelement())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            start_param = 0.
            for momentum, linestyle, linewidth in zip([0., 0.3, 0.6, ], [":", "-.", "-"], [3,2,1]):
                #dampening = 0.
                for dampening, linecolor in zip([0., 0.3, 0.6], ["red", "orange", "green"]):#, "blue", "black"]):
                    if momentum == 0. and dampening != 0.:
                        continue
                    #  result container
                    result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                    result_container.fill_(torch.nan)
                    
                    #  calc
                    _temp_tensor = torch.tensor([start_param], requires_grad=True)
                    the_sgd = torch.optim.SGD(params = [_temp_tensor], lr = lr, momentum = momentum, dampening = dampening)
                    for ii in range(grad_list.__len__()):
                        _temp_tensor.grad = torch.tensor([grad_list[ii]])
                        the_sgd.step()
                        result_container[ii] = _temp_tensor
                        pass#for ii

                    assert all_False(result_container.isnan())
                    ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                            label = f"m {momentum:.1f} d {dampening:.1f}", linestyle = linestyle, linewidth = linewidth, c = linecolor, alpha = 0.7)
                    pass#for dampening
                pass#for momentum
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test


        if "If the momentum == dampening " and False:
            plot_the_grad = True
            random_shift = 0.0
            #  plt init
            fig, ax = plt.subplots()

            #  grad
            grad_list = torch.randn(size=[1000])
            grad_list[100] = 20.


            _iota = iota(grad_list.nelement())+1
            if plot_the_grad:
                ax.plot(_iota, grad_list, label = "grad")
                pass

            #  test
            lr = 1.
            start_param = 0.
            for momentum, linewidth, linecolor in zip([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ], 
                                                        [7, 6,5,4,3,2,1], 
                                            ["red", "orange", "#C0B600", "green", "blue", "darkorchid", "fuchsia"] ):
                dampening = momentum   #  v
                #dampening = 1.-(momentum)
                #  result container
                result_container = torch.empty(size=[grad_list.__len__()], dtype=torch.float32)
                result_container.fill_(torch.nan)
                
                #  calc
                _temp_tensor = torch.tensor([start_param], requires_grad=True)
                the_sgd = torch.optim.SGD(params = [_temp_tensor], lr = lr, momentum = momentum, dampening = dampening)
                for ii in range(grad_list.__len__()):
                    _temp_tensor.grad = torch.tensor([grad_list[ii]])
                    the_sgd.step()
                    result_container[ii] = _temp_tensor
                    pass#for ii

                assert all_False(result_container.isnan())
                ax.plot(_iota+(random.random()*2.-1.)*random_shift, (result_container+(random.random()*2.-1.)*random_shift).tolist(), \
                        label = f"m {momentum:.1f} d {dampening:.1f}", linewidth = linewidth, c = linecolor, alpha = 0.7)
                pass#for momentum
            #  plt show.
            plt.legend()
            plt.show()
            pass#/test


        return 
    ____VISUAL____plot_the_values_________adam()











    pass



























assert False, '''
结论。
很闹心。
其实应该是一个很简单的事情。
反正，没自己写过，没自己验证过的东西就尽量不要用。

本来后面还计划自己实现一个的，实在没心情了。

就用最基本的就行了。

'''

