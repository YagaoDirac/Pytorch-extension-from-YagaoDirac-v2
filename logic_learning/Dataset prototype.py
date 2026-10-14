from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _bool_equal___0_as_false, all_False, \
        _tensor_equal, _tensor_shape_check, iota
#         iota, str_the_list
from logic_learning.Util_logic_learning import int_into_bool_tensor, int_list_into_bool_tensor, \
        bool_tensor_into_int_list, pos_of_first_one__int
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

if "algo prototype       sort the dataset" and __DEBUG_ME__() and True:
    def ____prototype____sort_the_dataset():
        
        
        if "prototype" and True:

            batch = 7
            length = 4

            input___b_l = int_list_into_bool_tensor(input_list=[13, 11,8,6,4,2,1], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            label___b = int_into_bool_tensor(input=0b1110000, length=batch)# no length for label
            assert _tensor_shape_check(label___b, batch)
            # calc
            addr_in_int_list = bool_tensor_into_int_list(input___b_l)
            assert _tensor_shape_check(addr_in_int_list, batch)
            assert addr_in_int_list.dtype == torch.int64
            addr_in_tensor_list = torch.tensor(addr_in_int_list)
            index = addr_in_tensor_list.sort().indices#decending?
            new_input___b_l = input___b_l[index]
            new_label___b   = label___b  [index]

            new_input_in_int_list = bool_tensor_into_int_list(new_input___b_l)
            assert _tensor_equal(new_input_in_int_list, [1,2,4,6,8,11,13])
            assert _bool_equal___0_as_false(new_label___b, [0,0,0,0,1,1,1])
            pass#/test

        if "not a test, simplification.     get ready to convert into a function.":# and False and False:
            #param
            input___b_l = int_list_into_bool_tensor(input_list=[13, 11,8,6,4,2,1], length=length)
            label___b = int_into_bool_tensor(input=0b1110000, length=batch)# no length for label

            # calc
            addr_in_int_list = bool_tensor_into_int_list(input___b_l)
            addr_in_tensor_list = torch.tensor(addr_in_int_list)
            index = addr_in_tensor_list.sort().indices#decending? no plan.
            new_input___b_l = input___b_l[index]
            new_label___b   = label___b  [index]
            
            pass#/test

        return
    ____prototype____sort_the_dataset()
    pass
def _prototype____dataset____sort(input___b_l:torch.Tensor, label___b:torch.Tensor, safety_check = False)->torch.Tensor:
    'return new_input___b_l, new_label___b'

    if safety_check:
        assert input___b_l.shape.__len__() == 2
        assert label___b.shape.__len__() == 1
        assert input___b_l.shape[0] == label___b.shape[0]
        assert input___b_l.dtype == torch.bool
        assert label___b.dtype == torch.bool
        pass

    # # func body
    # label___b_EXPANDl = label___b.reshape(shape=[label___b.shape[0], 1]).expand(size=[label___b.shape[0], input___b_l.shape[1]])
    # calc
    addr_in_int_list = bool_tensor_into_int_list(input___b_l)
    addr_in_tensor_list = torch.tensor(addr_in_int_list)
    index = addr_in_tensor_list.sort().indices#decending? no plan.
    new_input___b_l = input___b_l[index]
    new_label___b   = label___b  [index]
    return new_input___b_l, new_label___b
if "test" and __DEBUG_ME__() and True:
    def ____test______prototype____dataset____sort():
        if "equivalance        basic" and False:

            batch = 7
            length = 4

            input___b_l = int_list_into_bool_tensor(input_list=[13, 11,8,6,4,2,1], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            label___b = int_into_bool_tensor(input=0b1110000, length=batch)# no length for label
            assert _tensor_shape_check(label___b, batch)
            label___b_EXPANDl = label___b.reshape(shape=[label___b.shape[0], 1]).expand(size=[label___b.shape[0], length])
            assert _tensor_shape_check(label___b_EXPANDl, batch, length)
            # calc
            addr_in_int_list = bool_tensor_into_int_list(input___b_l)
            assert addr_in_int_list.__len__() == batch
            for _the_int in addr_in_int_list:
                assert type(_the_int) == int
                pass
            addr_in_tensor_list = torch.tensor(addr_in_int_list)
            index = addr_in_tensor_list.sort().indices#decending?
            new_input___b_l = input___b_l[index]
            new_label___b   = label___b  [index]

            new_input_in_int_list = bool_tensor_into_int_list(new_input___b_l)
            assert _tensor_equal(new_input_in_int_list, [1,2,4,6,8,11,13])
            assert _bool_equal___0_as_false(new_label___b, [0,0,0,0,1,1,1])
            #func equivalance
            #func equivalance
            func_ver___new_input___b_l, func_ver___new_label___b = _prototype____dataset____sort(input___b_l = input___b_l, label___b=label___b)
            assert _tensor_equal(           new_input_in_int_list, bool_tensor_into_int_list(func_ver___new_input___b_l))
            assert _bool_equal___0_as_false(new_label___b,                                   func_ver___new_label___b)

            pass#/test

        if "equivalance        randomly" and True:
            # hyperparam
            for batch in [1,2,3,5,14,31,67]:
                for length in [1,2,3,6,11]:#,23]:
                    #safety
                    two_to_the_power_of_length = 2**length
                    assert two_to_the_power_of_length< 10_000_000
                    if batch>two_to_the_power_of_length:
                        continue

                    for _ in range((batch+3)*(length+3)+11):

                        #  answer      addr
                        all_possible_addr = iota(two_to_the_power_of_length).tolist()
                        assert batch <= all_possible_addr.__len__()#safety

                        random.shuffle(all_possible_addr)
                        before_sorted__input_list:list[int] = all_possible_addr[0:batch]
                        after_sorted__input_list:list[int] = [x for x in before_sorted__input_list]
                        after_sorted__input_list.sort()
                        del all_possible_addr
                        #  answer      label
                        before_sorted__label_list:list[bool] = []
                        for _ in range(batch):
                            before_sorted__label_list.append(random.random()<0.5)
                            pass
                        after_sorted__label_list:list[bool] = []
                        for the_addr_after_sorted in after_sorted__input_list:
                            _temp_index = before_sorted__input_list.index(the_addr_after_sorted)
                            after_sorted__label_list.append(before_sorted__label_list[_temp_index])
                            pass
                        del the_addr_after_sorted, _temp_index

                        #  prototype ver        init

                        input___b_l = int_list_into_bool_tensor(input_list=before_sorted__input_list, length=length)
                        assert _tensor_shape_check(input___b_l, batch, length)
                        assert input___b_l.dtype == torch.bool
                        label___b = torch.tensor(before_sorted__label_list)
                        assert _tensor_shape_check(label___b, batch)
                        assert label___b.dtype == torch.bool
                        label___b_EXPANDl = label___b.reshape(shape=[label___b.shape[0], 1]).expand(size=[label___b.shape[0], length])
                        assert _tensor_shape_check(label___b_EXPANDl, batch, length)
                        #  prototype ver        calc
                        addr_in_int_list = bool_tensor_into_int_list(input___b_l)
                        assert _tensor_shape_check(addr_in_int_list, batch)
                        assert addr_in_int_list.dtype == torch.int64
                        addr_in_tensor_list = torch.tensor(addr_in_int_list)
                        index = addr_in_tensor_list.sort().indices#decending?
                        new_input___b_l = input___b_l[index]
                        new_label___b   = label___b  [index]

                        new_input_in_int_list = bool_tensor_into_int_list(new_input___b_l)
                        assert _tensor_equal(new_input_in_int_list, after_sorted__input_list)
                        assert _bool_equal___0_as_false(new_label___b, after_sorted__label_list)
                        #  func ver
                        #  func ver
                        func_ver___new_input___b_l, func_ver___new_label___b = _prototype____dataset____sort(input___b_l = input___b_l, label___b=label___b)
                        assert _tensor_equal(           new_input_in_int_list, bool_tensor_into_int_list(func_ver___new_input___b_l))
                        assert _bool_equal___0_as_false(new_label___b,                                   func_ver___new_label___b)

                        pass#for _ 
                    pass#for length
                pass#for batch

            pass#/test

        return
    ____test______prototype____dataset____sort()
    pass



if "matplotlib test" and False and False:
    from matplotlib import pyplot as plt
    data = [[0., 0,0,0,],
            [1., 0,0,0,],
            [2., 3,0,0,],]
    plt.imshow(data)#, cmap='hot', interpolation='nearest')
    plt.show()
    pass


if "plot a dataset          prototype" and __DEBUG_ME__() and True:
    def ____prototype____plot_a_dataset():
        if "prototype" and True:

            #assert False, "等dataset排序，这个逻辑只能适配已经排序的数据集。"

            length = 2
            batch = 2**length
            # init
            input_list = [x for x in range(batch)]
            assert input_list.__len__() == 4   #prototype
            assert input_list[-1] == 3         #prototype

            input___b_l = int_list_into_bool_tensor(input_list=input_list, length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            label___b = int_into_bool_tensor(input=0b1001, length=batch)# no length for label
            assert _tensor_shape_check(label___b, batch)

            #calc
            #  buffer size
            _temp___pos_of_first_one_of_batch = pos_of_first_one__int(batch-1)#batch-1 is the max of index.

            _temp___height___but_log2 = _temp___pos_of_first_one_of_batch//2
            _temp___width___but_log2 = _temp___pos_of_first_one_of_batch - _temp___height___but_log2
            del _temp___pos_of_first_one_of_batch
            height = 2 ** _temp___height___but_log2
            width  = 2 **  _temp___width___but_log2
            del _temp___height___but_log2, _temp___width___but_log2
            #  buffer init
            buffer_for_plt = torch.zeros(size=[height*width])
            assert buffer_for_plt.shape.__len__() == 1
            #  buffer write
            addr_in_int_list = bool_tensor_into_int_list(input___b_l)
            label_in_plot_color_index_like = torch.where(label___b, 3., 1.)
            buffer_for_plt[addr_in_int_list] = label_in_plot_color_index_like
            assert all_False(buffer_for_plt.eq(0.))#这个地方应该没有任何一个没有被赋值的东西"
            #  buffer reshape
            buffer_for_plt = buffer_for_plt.reshape(shape=[height, width])
            #  plot it
            from matplotlib import pyplot as plt
            plt.imshow(buffer_for_plt)#, cmap='hot', interpolation='nearest')
            plt.show()
            pass#/test



    ____prototype____plot_a_dataset()
    pass












