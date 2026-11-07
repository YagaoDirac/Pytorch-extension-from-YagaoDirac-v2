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

总之这个文档里面不会牵扯 λ (weight decay)。'''






if "prototype before into a function" and __DEBUG_ME__() and True:
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
        the_sgd = torch.optim.SGD(lr=lr, params=[the_param_0, the_param_1], momentum=momentum, dampening=dampening)


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

def extract_info_from_sgd(the_sgd:torch.optim.SGD)->tuple[list[torch.Tensor], list[torch.Tensor], float, float, float]:
    '''return param_list, momentum_buffer_list, lr, momentum, dampening'''

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

    return param_list, momentum_buffer_list, lr, momentum, dampening
if "test" and __DEBUG_ME__() and True:
    def ____test____extract_info_from_sgd():
        if "equilavance          not a full test." and True:
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
            the_sgd = torch.optim.SGD(lr=lr, params=[the_param_0, the_param_1], momentum=momentum, dampening=dampening)


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
            param_list, momentum_buffer_list, func_ver__lr, func_ver__momentum, func_ver__dampening = extract_info_from_sgd(the_sgd=the_sgd)
            assert param_list == [the_param_0, the_param_1]
            assert param_list[0] is the_param_0
            assert param_list[1] is the_param_1
            assert momentum_buffer_list.__len__() == 0
            assert func_ver__lr == lr
            assert func_ver__momentum == momentum
            assert func_ver__dampening == dampening


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
                param_list, momentum_buffer_list, func_ver__lr, func_ver__momentum, func_ver__dampening = extract_info_from_sgd(the_sgd=the_sgd)
                assert param_list == [the_param_0, the_param_1]
                assert param_list[0] is the_param_0
                assert param_list[1] is the_param_1
                assert momentum_buffer_list == some_list
                assert momentum_buffer_list[0] is some_list[0]
                assert momentum_buffer_list[1] is some_list[1]
                assert func_ver__lr == lr
                assert func_ver__momentum == momentum
                assert func_ver__dampening == dampening 

                pass# for _iii

            pass#/test

        if "ran-able test" and True:
            for with_grad in [True, False]:
                for _iii in range(33):
                    ori___lr = random.random()+0.1
                    ori___momentum = random.random()+0.1
                    ori___dampening = random.random()+0.1
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
                    the_sgd = torch.optim.SGD(params=ori___param_list, lr=ori___lr, momentum=ori___momentum, dampening=ori___dampening)

                    param_list, momentum_buffer_list, lr, momentum, dampening = extract_info_from_sgd(the_sgd=the_sgd)
                    assert param_list == ori___param_list
                    assert momentum_buffer_list.__len__() == 0
                    assert lr == ori___lr
                    assert momentum == ori___momentum
                    assert dampening == ori___dampening

                    the_sgd.step()
                    
                    param_list, momentum_buffer_list, lr, momentum, dampening = extract_info_from_sgd(the_sgd=the_sgd)
                    assert param_list == ori___param_list
                    if with_grad:
                        assert momentum_buffer_list.__len__() == param_list.__len__()
                        pass
                    else:
                        assert momentum_buffer_list.__len__() == 0
                        pass
                    assert lr == ori___lr
                    assert momentum == ori___momentum
                    assert dampening == ori___dampening

                    the_sgd.step()
                    
                    param_list, momentum_buffer_list, lr, momentum, dampening = extract_info_from_sgd(the_sgd=the_sgd)
                    assert param_list == ori___param_list
                    if with_grad:
                        assert momentum_buffer_list.__len__() == param_list.__len__()
                        pass
                    else:
                        assert momentum_buffer_list.__len__() == 0
                        pass
                    assert lr == ori___lr
                    assert momentum == ori___momentum
                    assert dampening == ori___dampening

                    pass#for _iii
                pass# for with_grad

            pass#/test










        if "some manual value test" and True:
            the_param = torch.tensor([10.], requires_grad=True)
            the_param.grad = torch.tensor([2.])

            lr = 0.
            momentum = 0.3
            dampening = 0.
            the_sgd = torch.optim.SGD(lr=lr, params=[the_param], momentum=momentum, dampening=dampening)

            param_list, momentum_buffer_list, lr, momentum, dampening = extract_info_from_sgd(the_sgd=the_sgd)
            print(f"param {param_list[0]}   grad {param_list[0].grad}   buffer {momentum_buffer_list}   lr {lr}   momentum {momentum}")
            the_sgd.step()

            param_list, momentum_buffer_list, lr, momentum, dampening = extract_info_from_sgd(the_sgd=the_sgd)
            print(f"param {param_list[0]}   grad {param_list[0].grad}   buffer {momentum_buffer_list}   lr {lr}   momentum {momentum}")
            the_sgd.step()

            param_list, momentum_buffer_list, lr, momentum, dampening = extract_info_from_sgd(the_sgd=the_sgd)
            print(f"param {param_list[0]}   grad {param_list[0].grad}   buffer {momentum_buffer_list}   lr {lr}   momentum {momentum}")
            the_sgd.step()

            param_list, momentum_buffer_list, lr, momentum, dampening = extract_info_from_sgd(the_sgd=the_sgd)
            print(f"param {param_list[0]}   grad {param_list[0].grad}   buffer {momentum_buffer_list}   lr {lr}   momentum {momentum}")
            the_sgd.step()
            pass#/test
        1w
        1w
        1wjixu 

        return 
    ____test____extract_info_from_sgd()
    pass



















def my_sgd(model_param:torch.Tensor, grad:torch.Tensor, lr:float, hidden_state:torch.Tensor|None = None, 
                        momentum = 0., dampening = 0., weight_decay = 0.)->tuple[torch.Tensor, torch.Tensor]:

    gard__plus__weight_decay__mul__model_param = grad+weight_decay*model_param

    if hidden_state is not None:
        hidden_state = momentum*hidden_state + (1. - dampening)*grad
        pass
    else:
        hidden_state = grad.clone()
        pass
    assert isinstance(hidden_state, torch.Tensor)

    grad_to_update_param = hidden_state
    new_model_param = model_param- lr*grad_to_update_param
    return new_model_param, hidden_state









fds=432




