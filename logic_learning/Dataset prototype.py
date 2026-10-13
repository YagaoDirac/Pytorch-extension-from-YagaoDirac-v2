from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _bool_equal___0_as_false, all_False, \
        _tensor_equal, _tensor_shape_check
#         iota, str_the_list
from logic_learning.Util_logic_learning import int_into_bool_tensor, int_list_into_bool_tensor, \
    bool_tensor_into_int
        #bool_tensor_into_posneg1
import random
import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######









if "algo prototype       relativity" and __DEBUG_ME__() and False:
    def ____prototype____relativity():
        if "prototype" and True:

            batch = 3
            length = 5
            # init
            input___b_l = int_list_into_bool_tensor(input_list=[0b10000,
                                                                0b11010,
                                                                0b10010,
                                                                ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            label___b = int_into_bool_tensor(input=0b111, length=batch)# no length for label
            assert _tensor_shape_check(label___b, batch)
            label___b_EXPANDl = label___b.reshape(shape=[label___b.shape[0], 1]).expand(size=[label___b.shape[0], length])
            assert _tensor_shape_check(label___b_EXPANDl, batch, length)
            # calc
            the_xor___b_l = input___b_l.logical_xor(label___b_EXPANDl)
            assert _tensor_shape_check(the_xor___b_l, batch, length)
            assert the_xor___b_l.dtype == torch.bool
            assert _bool_equal___0_as_false(the_xor___b_l, [[0,1,1,1,1],
                                                            [0,0,1,0,1],
                                                            [0,1,1,0,1],])
            
            sum_of_xor___l = the_xor___b_l.to(torch.int32).sum(dim=0)
            assert _tensor_shape_check(sum_of_xor___l, length)
            assert sum_of_xor___l.dtype == torch.int64#it feels like the sum converts dtype into int64.
            assert _tensor_equal(sum_of_xor___l, [0,2,3,1,3])

            sum_of_xnor___l = batch-sum_of_xor___l
            assert _tensor_shape_check(sum_of_xnor___l, length)
            assert sum_of_xnor___l.dtype == torch.int64
            assert _tensor_equal(sum_of_xnor___l, [3,1,0,2,0])

            relativity___l = sum_of_xnor___l.to(torch.float32)/batch
            assert _tensor_shape_check(relativity___l, length)
            assert relativity___l.dtype == torch.float32
            assert _tensor_equal(relativity___l, [1., 0.33, 0, 0.66, 0.], epsilon=0.01)
            pass#/test

        if "not a test, simplification.     get ready to convert into a function." and False and False:
            # param for prototype
            batch = 3
            length = 5
            # param
            input___b_l = int_list_into_bool_tensor(input_list=[0b10000,
                                                                0b11010,
                                                                0b10010,
                                                                ], length=length)
            label___b = int_into_bool_tensor(input=0b111, length=batch)# no length for label
            label___b_EXPANDl = label___b.reshape(shape=[label___b.shape[0], 1]).expand(size=[label___b.shape[0], length])
            # calc
            the_xor___b_l = input___b_l.logical_xor(label___b_EXPANDl)
            sum_of_xor___l = the_xor___b_l.to(torch.int32).sum(dim=0)
            sum_of_xnor___l = batch-sum_of_xor___l
            relativity___l = sum_of_xnor___l.to(torch.float32)/batch
            # return relativity___l
            pass#/test

        return
    ____prototype____relativity()
    pass

def _prototype____dataset____calc_relativity(input___b_l:torch.Tensor, label___b:torch.Tensor, safety_check = False)->torch.Tensor:
    'return relativity___l'
    if safety_check:
        assert input___b_l.shape.__len__() == 2
        assert label___b.shape.__len__() == 1
        assert input___b_l.shape[0] == label___b.shape[0]
        assert input___b_l.dtype == torch.bool
        assert label___b.dtype == torch.bool
        pass

    batch = label___b.shape[0]
    label___b_EXPANDl = label___b.reshape(shape=[batch, 1]).expand(size=[batch, input___b_l.shape[1]])
    # calc
    the_xor___b_l = input___b_l.logical_xor(label___b_EXPANDl)
    sum_of_xor___l = the_xor___b_l.to(torch.int32).sum(dim=0)
    sum_of_xnor___l = batch-sum_of_xor___l
    relativity___l = sum_of_xnor___l.to(torch.float32)/batch
    return relativity___l
if "test" and __DEBUG_ME__() and True:
    def ____test_____prototype____dataset____calc_relativity():
        if "equivalance        basic" and False:

            batch = 3
            length = 5
            # init
            input___b_l = int_list_into_bool_tensor(input_list=[0b10000,
                                                                0b11010,
                                                                0b10010,
                                                                ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            label___b = int_into_bool_tensor(input=0b111, length=batch)# no length for label
            assert _tensor_shape_check(label___b, batch)
            label___b_EXPANDl = label___b.reshape(shape=[label___b.shape[0], 1]).expand(size=[label___b.shape[0], length])
            assert _tensor_shape_check(label___b_EXPANDl, batch, length)
            # calc
            the_xor___b_l = input___b_l.logical_xor(label___b_EXPANDl)
            assert _tensor_shape_check(the_xor___b_l, batch, length)
            assert the_xor___b_l.dtype == torch.bool
            assert _bool_equal___0_as_false(the_xor___b_l, [[0,1,1,1,1],
                                                            [0,0,1,0,1],
                                                            [0,1,1,0,1],])
            
            sum_of_xor___l = the_xor___b_l.to(torch.int32).sum(dim=0)
            assert _tensor_shape_check(sum_of_xor___l, length)
            assert sum_of_xor___l.dtype == torch.int64#it feels like the sum converts dtype into int64.
            assert _tensor_equal(sum_of_xor___l, [0,2,3,1,3])

            sum_of_xnor___l = batch-sum_of_xor___l
            assert _tensor_shape_check(sum_of_xnor___l, length)
            assert sum_of_xnor___l.dtype == torch.int64
            assert _tensor_equal(sum_of_xnor___l, [3,1,0,2,0])

            relativity___l = sum_of_xnor___l.to(torch.float32)/batch
            assert _tensor_shape_check(relativity___l, length)
            assert relativity___l.dtype == torch.float32
            assert _tensor_equal(relativity___l, [1., 0.33, 0, 0.66, 0.], epsilon=0.01)
            #func equilavance
            #func equilavance
            func_result___relativity___l = _prototype____dataset____calc_relativity(input___b_l=input___b_l, label___b=label___b)
            assert _tensor_equal(func_result___relativity___l, relativity___l)

            pass#/test

        if "equivalance        randomly" and True:

            for batch in [1,2,3,5,11,25]:
                for length in [1,2,3,5,17,42]:
                    for _ in range((batch+3)*(length+3)+11):
                        # init

                        input_list = []
                        for __ in range(batch):
                            input_list.append(random.randint(0, (1<<length)-1))
                            pass
                        input___b_l = int_list_into_bool_tensor(input_list=input_list, length=length)
                        assert _tensor_shape_check(input___b_l, batch, length)

                        label___b = int_into_bool_tensor(input=random.randint(0,(1<<batch)-1), length=batch)# no length for label
                        assert _tensor_shape_check(label___b, batch)
                        label___b_EXPANDl = label___b.reshape(shape=[label___b.shape[0], 1]).expand(size=[label___b.shape[0], length])
                        assert _tensor_shape_check(label___b_EXPANDl, batch, length)
                        # calc
                        the_xor___b_l = input___b_l.logical_xor(label___b_EXPANDl)
                        assert _tensor_shape_check(the_xor___b_l, batch, length)
                        assert the_xor___b_l.dtype == torch.bool
                        
                        sum_of_xor___l = the_xor___b_l.to(torch.int32).sum(dim=0)
                        assert _tensor_shape_check(sum_of_xor___l, length)
                        assert sum_of_xor___l.dtype == torch.int64#it feels like the sum converts dtype into int64.
                        assert sum_of_xor___l.ge(0).all()
                        assert sum_of_xor___l.le(batch).all()

                        sum_of_xnor___l = batch-sum_of_xor___l
                        assert _tensor_shape_check(sum_of_xnor___l, length)
                        assert sum_of_xnor___l.dtype == torch.int64
                        assert sum_of_xnor___l.ge(0).all()
                        assert sum_of_xnor___l.le(batch).all()

                        relativity___l = sum_of_xnor___l.to(torch.float32)/batch
                        assert _tensor_shape_check(relativity___l, length)
                        assert relativity___l.dtype == torch.float32
                        assert relativity___l.ge(0.).all()
                        assert relativity___l.le(1.).all()
                        #func equilavance
                        #func equilavance
                        func_result___relativity___l = _prototype____dataset____calc_relativity(input___b_l=input___b_l, label___b=label___b)
                        assert _tensor_equal(func_result___relativity___l, relativity___l)

                        pass#for _
                    pass#for length 
                pass#for batch

            pass#/test
        return
    ____test_____prototype____dataset____calc_relativity()
    pass












if "plot a dataset" and __DEBUG_ME__() and True:
    def ____prototype____plot_a_dataset():

        length = 2
        batch = 2**length
        # init
        input_list = [x for x in range(batch)]
        assert input_list.__len__() == 4
        assert input_list[-1] == 3

        input___b_l = int_list_into_bool_tensor(input_list=input_list, length=length)
        assert _tensor_shape_check(input___b_l, batch, length)
        label___b = int_into_bool_tensor(input=0b1111, length=batch)# no length for label
        assert _tensor_shape_check(label___b, batch)

        #calc



        from matplotlib import pyplot as plt









