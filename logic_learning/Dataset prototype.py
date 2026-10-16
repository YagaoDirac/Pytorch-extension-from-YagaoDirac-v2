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
            addr_in_tensor_list = bool_tensor_into_int_list(input___b_l)
            assert _tensor_shape_check(addr_in_tensor_list, batch)
            assert addr_in_tensor_list.dtype == torch.int64
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
            addr_in_tensor_list = bool_tensor_into_int_list(input___b_l)
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
    addr_in_tensor_list = bool_tensor_into_int_list(input___b_l)
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
                        addr_in_tensor_list = bool_tensor_into_int_list(input___b_l)
                        assert _tensor_shape_check(addr_in_tensor_list, batch)
                        assert addr_in_tensor_list.dtype == torch.int64
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

if "to fold some feature test code" and False and False:
    if "matplotlib test" and False and False:
        from matplotlib import pyplot as plt
        data = [[0., 0,0,0,],
                [1., 0,0,0,],
                [2., 3,0,0,],]
        plt.imshow(data)#, cmap='hot', interpolation='nearest')
        plt.show()
        pass
    if "matplotlib test" and False and False:
        from matplotlib import pyplot as plt
        data = [[
                [1., 1., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
                            ]]
        plt.imshow(data)#, cmap='hot', interpolation='nearest')
        plt.show()
        pass
    if "torch.where func feature test" and False and False:
        'conclusion:  no guess is the best guess'
        condition = torch.tensor([True, False, True, True, False])
        value1 = torch.tensor([3.])
        value2 = torch.tensor([4.])
        result = torch.where(condition, value1, value2)
        assert _tensor_equal(result, [3,4,3,3,4])

        condition = torch.tensor([True, False])
        value1 = torch.tensor([3., 3.1])
        value2 = torch.tensor([4., 4.1])
        result = torch.where(condition, value1, value2)
        assert _tensor_equal(result, [3,  4.1])#the second one is 4.1

        condition = torch.tensor([True, False, True, True, False])
        value1 = torch.tensor([3., 3.1])
        value2 = torch.tensor([4., 4.1])
        #result = torch.where(condition, value1, value2)#doesn't work

        condition = torch.tensor([[True,True], [False,False], [True,True]])
        value1 = torch.tensor([[11, 12], [13, 14], [15, 16]])
        value2 = torch.tensor([[21, 22], [23, 24], [25, 26]])
        result = torch.where(condition, value1, value2)
        assert _tensor_equal(result, [[11, 12], [23, 24], [15, 16]])

        condition = torch.tensor([True, False, True]).reshape([3, 1]).expand([3, 2])
        value1 = torch.tensor([3., 3.1]).reshape([1, 2]).expand([3, 2])
        value2 = torch.tensor([4., 4.1]).reshape([1, 2]).expand([3, 2])
        result = torch.where(condition, value1, value2)
        assert _tensor_equal(result, [[3., 3.1], [4., 4.1], [3., 3.1]])


        pass
    pass
if "plot a dataset          prototype" and __DEBUG_ME__() and False:
    def ____prototype____plot_a_dataset():
        if "prototype              with plt.show" and False:
            # param
            safety = True
            force__log2_of_total_shown_points = 3   # or none or anything useless.
            heat_map_value_for_True :list[float] = [1., 0.75, 0.2]
            
            heat_map_value_for_False = [1., 0.1, 0.2]



            heat_map_value_for_NA =[0., 0.01, 0.02]
            heat_map_title = "T:yellow, F:red, NA:black"

            length = 2
            batch = 2**length
            # init
            input_list = [x for x in range(batch)]#prototype
            assert input_list.__len__() == 4   #prototype
            assert input_list[-1] == 3         #prototype

            if safety:
                all_unique = len(input_list) == len(set(input_list))
                assert all_unique
                pass

            _temp__max_of_input_list = max(input_list)

            input___b_l = int_list_into_bool_tensor(input_list=input_list, length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            label___b = int_into_bool_tensor(input=0b1001, length=batch)# no length for label
            assert _tensor_shape_check(label___b, batch)


            #calc
            #  buffer size           manual set
            _temp___pos_of_first_one_of_batch = pos_of_first_one__int(_temp__max_of_input_list)#batch-1 is the max of index.
            if type(force__log2_of_total_shown_points) == int and \
                    force__log2_of_total_shown_points > 0:
                if force__log2_of_total_shown_points >= _temp___pos_of_first_one_of_batch:
                    log2_of_total_shown_points = force__log2_of_total_shown_points
                    pass
                else: #force__log2_of_total_shown_points < _temp___pos_of_first_one_of_batch:
                    assert False, f"if you want to ignore param:force__log2_of_total_shown_points, {None \
                                    }set it to non-int or 0 or negative int. If it's positive int, {None \
                                    }it must be >= the minimum needed value. If you don't know what {None \
                                    }you are doint, leave it untouched (set it to None)."
                    pass
                pass
            else:
                log2_of_total_shown_points = _temp___pos_of_first_one_of_batch
                pass

            assert log2_of_total_shown_points>=_temp___pos_of_first_one_of_batch
            del _temp___pos_of_first_one_of_batch
            #  buffer size          the rest
            _temp___height___but_log2 = log2_of_total_shown_points//2
            _temp___width___but_log2 = log2_of_total_shown_points - _temp___height___but_log2
            del log2_of_total_shown_points
            height = 2 ** _temp___height___but_log2
            width  = 2 **  _temp___width___but_log2
            del _temp___height___but_log2, _temp___width___but_log2
            #  buffer init to NA
            heat_map_value_for_NA___into_tensor = torch.tensor(heat_map_value_for_NA, dtype=torch.float16)
            assert heat_map_value_for_NA___into_tensor.shape == torch.Size([3]), "Set heat_map_value_for_True as rgb color. Or leave it as defaul."
            assert heat_map_value_for_NA___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
            assert heat_map_value_for_NA___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
            _temp___heat_map_value_for_NA___EXPANDhw_3 = heat_map_value_for_NA___into_tensor.reshape(shape=[1, 3]).expand([height*width, 3])
            buffer_for_plt = _temp___heat_map_value_for_NA___EXPANDhw_3.detach().clone()
            del heat_map_value_for_NA___into_tensor, _temp___heat_map_value_for_NA___EXPANDhw_3
            assert _tensor_equal(buffer_for_plt[0], heat_map_value_for_NA)#prototype
            assert _tensor_equal(buffer_for_plt[1], heat_map_value_for_NA)#prototype
            assert _tensor_equal(buffer_for_plt[7], heat_map_value_for_NA)#prototype
            assert buffer_for_plt.shape.__len__() == 2
            assert _tensor_shape_check(buffer_for_plt, height*width, 3)

            #  buffer write             
            addr_in_tensor = bool_tensor_into_int_list(input___b_l)
            assert _tensor_equal(addr_in_tensor, input_list)#debug

            addr_in_tensor__for_True  = addr_in_tensor[label___b]
            assert _tensor_equal(addr_in_tensor__for_True , [0, 3])
            addr_in_tensor__for_False = addr_in_tensor[label___b.logical_not()]
            assert _tensor_equal(addr_in_tensor__for_False, [1, 2])

            heat_map_value_for_True___into_tensor = torch.tensor(heat_map_value_for_True , dtype=torch.float16)
            assert heat_map_value_for_True___into_tensor.shape == torch.Size([3]), "Set heat_map_value_for_True as rgb color. Or leave it as defaul."
            assert heat_map_value_for_True___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
            assert heat_map_value_for_True___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
            heat_map_value_for_True___EXPANDb_3  = heat_map_value_for_True___into_tensor.reshape([1, 3]).expand([addr_in_tensor__for_True .shape[0], 3])
            assert _tensor_equal(heat_map_value_for_True___EXPANDb_3, [ [1., 0.75, 0.2], [1., 0.75, 0.2], ],  epsilon=0.01)
            
            heat_map_value_for_False___into_tensor = torch.tensor(heat_map_value_for_False, dtype=torch.float16)
            assert heat_map_value_for_False___into_tensor.shape == torch.Size([3]), "set heat_map_value_for_True as rgb color. Or leave it as defaul."
            assert heat_map_value_for_False___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
            assert heat_map_value_for_False___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
            heat_map_value_for_False___EXPANDb_3 = heat_map_value_for_False___into_tensor.reshape([1, 3]).expand([addr_in_tensor__for_False.shape[0], 3])
            assert _tensor_equal(heat_map_value_for_False___EXPANDb_3, [[1., 0.1, 0.2],  [1., 0.1, 0.2],  ], epsilon=0.01)

            buffer_for_plt[addr_in_tensor__for_True ] = heat_map_value_for_True___EXPANDb_3
            buffer_for_plt[addr_in_tensor__for_False] = heat_map_value_for_False___EXPANDb_3

            assert _tensor_equal(buffer_for_plt, [  [1., 0.75, 0.2],    [1., 0.1, 0.2],  
                                                    [1., 0.1, 0.2],   [1., 0.75, 0.2], 
                                                    [0., 0.01, 0.02], #NA
                                                    [0., 0.01, 0.02], #NA
                                                    [0., 0.01, 0.02], #NA
                                                    [0., 0.01, 0.02], #NA
                                                                ], epsilon=0.01)



            if "old algo,      Now I have a better one.    Let's leave the old one here." and False and False:
                # #  buffer write             
                # addr_in_int_list = bool_tensor_into_int_list(input___b_l)
                # assert _tensor_equal(addr_in_int_list, input_list)#debug
                #     #  reshape a bit
                # label___b_EXPAND3 = label___b.reshape([-1, 1]).expand([-1, 3])
                # assert _bool_equal___0_as_false(label___b_EXPAND3, [[1,1,1], [0,0,0], [0,0,0], [1,1,1]])
                # heat_map_value_for_True___EXPANDb_3  = heat_map_value_for_True.reshape([1, -1]).expand_as(label___b_EXPAND3)
                # assert _tensor_equal(heat_map_value_for_True___EXPANDb_3, [ [1., 0.7, 0.], [1., 0.7, 0.], 
                #                                                             [1., 0.7, 0.], [1., 0.7, 0.],  ], epsilon=0.01)
                # heat_map_value_for_False___EXPANDb_3 = heat_map_value_for_False.reshape([1, -1]).expand_as(label___b_EXPAND3)
                # assert _tensor_equal(heat_map_value_for_False___EXPANDb_3, [[1., 0.1, 0.2], [1., 0.1, 0.2], 
                #                                                             [1., 0.1, 0.2], [1., 0.1, 0.2],   ], epsilon=0.01)
                #     #  mix
                # label_in_plot_color_index_like = torch.where(label___b_EXPAND3, heat_map_value_for_True___EXPANDb_3, \
                #                                                                 heat_map_value_for_False___EXPANDb_3)
                # assert _tensor_equal(label_in_plot_color_index_like, [[1., 0.7, 0.],    [1., 0.1, 0.2],  
                #                                                         [1., 0.1, 0.2],   [1., 0.7, 0.],     ], epsilon=0.01)
                #     #  put into buffer
                # buffer_for_plt[addr_in_int_list] = label_in_plot_color_index_like[iota(label_in_plot_color_index_like.shape[0])]
                # assert _tensor_equal(buffer_for_plt, [  [1., 0.7, 0.],    [1., 0.1, 0.2],  
                #                                         [1., 0.1, 0.2],   [1., 0.7, 0.], 
                #                                         [0., 0.01, 0.02], #NA
                #                                         [0., 0.01, 0.02], #NA
                #                                         [0., 0.01, 0.02], #NA
                #                                         [0., 0.01, 0.02], #NA
                #                                                     ], epsilon=0.01)
                pass

            #  buffer reshape
            buffer_for_plt = buffer_for_plt.reshape(shape=[height, width, 3])
            #  plot it
            from matplotlib import pyplot as plt
            plt.imshow(buffer_for_plt)#, cmap='hot', interpolation='nearest')
            plt.title(heat_map_title)
            plt.show()
            pass#/test

        if "not a test......   simplification, and get ready to turn into a function" and False and False:
            # param
            safety = True
            force__log2_of_total_shown_points:int|None = None   # or none or anything useless.
            heat_map_value_for_True :list[float] = [1., 1.,  0.23]
            heat_map_value_for_False:list[float] = [1., 0.1, 0.2 ]
            heat_map_value_for_NA   :list[float] = [0., 0.,  0.  ]
            heat_map_title = "T:yellow, F:red, NA:black"

            # length = 2
            # batch = 2**length
            # init

            if safety:
                all_unique = len(input_list) == len(set(input_list))
                assert all_unique
                pass

            _temp__max_of_input_list = max(input_list)

            input___b_l:torch.Tensor
            #assert _tensor_shape_check(input___b_l, batch, length)
            label___b:torch.Tensor
            #assert _tensor_shape_check(label___b, batch)


            #calc
            #  buffer size           manual set
            _temp___pos_of_first_one_of_batch = pos_of_first_one__int(_temp__max_of_input_list)#batch-1 is the max of index.
            if type(force__log2_of_total_shown_points) == int and \
                    force__log2_of_total_shown_points > 0:
                if force__log2_of_total_shown_points >= _temp___pos_of_first_one_of_batch:
                    log2_of_total_shown_points = force__log2_of_total_shown_points
                    pass
                else: #force__log2_of_total_shown_points < _temp___pos_of_first_one_of_batch:
                    assert False, f"if you want to ignore param:force__log2_of_total_shown_points, {None \
                                    }set it to non-int or 0 or negative int. If it's positive int, {None \
                                    }it must be >= the minimum needed value. If you don't know what {None \
                                    }you are doint, leave it untouched (set it to None)."
                    pass
                pass
            else:
                log2_of_total_shown_points = _temp___pos_of_first_one_of_batch
                pass

            assert log2_of_total_shown_points>=_temp___pos_of_first_one_of_batch
            del _temp___pos_of_first_one_of_batch
            #  buffer size          the rest
            _temp___height___but_log2 = log2_of_total_shown_points//2
            _temp___width___but_log2 = log2_of_total_shown_points - _temp___height___but_log2
            del log2_of_total_shown_points
            height = 2 ** _temp___height___but_log2
            width  = 2 **  _temp___width___but_log2
            del _temp___height___but_log2, _temp___width___but_log2
            #  buffer init to NA
            heat_map_value_for_NA___into_tensor = torch.tensor(heat_map_value_for_NA, dtype=torch.float16)
            assert heat_map_value_for_NA___into_tensor.shape == torch.Size([3]), "Set heat_map_value_for_True as rgb color. Or leave it as defaul."
            assert heat_map_value_for_NA___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
            assert heat_map_value_for_NA___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
            _temp___heat_map_value_for_NA___EXPANDhw_3 = heat_map_value_for_NA___into_tensor.reshape(shape=[1, 3]).expand([height*width, 3])
            buffer_for_plt = _temp___heat_map_value_for_NA___EXPANDhw_3.detach().clone()
            del heat_map_value_for_NA___into_tensor, _temp___heat_map_value_for_NA___EXPANDhw_3
            #assert buffer_for_plt.shape.__len__() == 2
            #assert _tensor_shape_check(buffer_for_plt, height*width, 3)

            #  buffer write             
            addr_in_tensor = bool_tensor_into_int_list(input___b_l)

            addr_in_tensor__for_True  = addr_in_tensor[label___b]
            addr_in_tensor__for_False = addr_in_tensor[label___b.logical_not()]

            heat_map_value_for_True___into_tensor = torch.tensor(heat_map_value_for_True , dtype=torch.float16)
            assert heat_map_value_for_True___into_tensor.shape == torch.Size([3]), "Set heat_map_value_for_True as rgb color. Or leave it as defaul."
            assert heat_map_value_for_True___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
            assert heat_map_value_for_True___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
            heat_map_value_for_True___EXPANDb_3  = heat_map_value_for_True___into_tensor.reshape([1, 3]).expand([addr_in_tensor__for_True .shape[0], 3])
            
            heat_map_value_for_False___into_tensor = torch.tensor(heat_map_value_for_False, dtype=torch.float16)
            assert heat_map_value_for_False___into_tensor.shape == torch.Size([3]), "set heat_map_value_for_True as rgb color. Or leave it as defaul."
            assert heat_map_value_for_False___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
            assert heat_map_value_for_False___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
            heat_map_value_for_False___EXPANDb_3 = heat_map_value_for_False___into_tensor.reshape([1, 3]).expand([addr_in_tensor__for_False.shape[0], 3])

            buffer_for_plt[addr_in_tensor__for_True ] = heat_map_value_for_True___EXPANDb_3
            buffer_for_plt[addr_in_tensor__for_False] = heat_map_value_for_False___EXPANDb_3

            #  buffer reshape
            buffer_for_plt = buffer_for_plt.reshape(shape=[height, width, 3])
            #  plot it
            from matplotlib import pyplot as plt
            plt.imshow(buffer_for_plt)#, cmap='hot', interpolation='nearest')
            plt.title(heat_map_title)
            plt.show()
            pass#/test

    ____prototype____plot_a_dataset()
    pass
def _prototype____dataset____plt_show(input___b_l:torch.Tensor, label___b:torch.Tensor, \
        force__log2_of_total_shown_points:int|None = None, safety = True, plt_show = True,
        heat_map_value_for_True :list[float] = [1., 1.,  0.23],
        heat_map_value_for_False:list[float] = [1., 0.1, 0.2 ],
        heat_map_value_for_NA   :list[float] = [0., 0.,  0.  ],
        heat_map_title = "T:yellow, F:red, NA:black")                   ->torch.Tensor:
    '''return buffer_for_plt

    The return value is only for debug purpose.
    '''
    
    addr_in_tensor = bool_tensor_into_int_list(input___b_l)

    if safety:
        input_list = addr_in_tensor.tolist()
        _temp___all_unique = len(input_list) == len(set(input_list))
        assert _temp___all_unique
        del _temp___all_unique

        assert input___b_l.shape.__len__() == 2
        assert label___b.shape.__len__() == 1
        assert input___b_l.shape[0] == label___b.shape[0]
        #assert _tensor_shape_check(input___b_l, batch, length)
        #assert _tensor_shape_check(label___b, batch)
        pass

    _temp__max_of_input_list = max(input_list)
    #calc
        #  buffer size           manual set
    _temp___pos_of_first_one_of_batch = pos_of_first_one__int(_temp__max_of_input_list)#batch-1 is the max of index.
    if type(force__log2_of_total_shown_points) == int and \
            force__log2_of_total_shown_points > 0:
        if force__log2_of_total_shown_points >= _temp___pos_of_first_one_of_batch:
            log2_of_total_shown_points = force__log2_of_total_shown_points
            pass
        else: #force__log2_of_total_shown_points < _temp___pos_of_first_one_of_batch:
            assert False, f"If you want to ignore param:force__log2_of_total_shown_points, {"" \
                            }set it to non-int or 0 or negative int. If it's positive int, {"" \
                            }it must be >= the minimum needed value. If you don't know what {"" \
                            }you are doint, leave it untouched (set it to None)."
            pass
        pass
    else:
        log2_of_total_shown_points = _temp___pos_of_first_one_of_batch
        pass

    assert log2_of_total_shown_points>=_temp___pos_of_first_one_of_batch
    del _temp___pos_of_first_one_of_batch
        #  buffer size          the rest
    _temp___height___but_log2 = log2_of_total_shown_points//2
    _temp___width___but_log2 = log2_of_total_shown_points - _temp___height___but_log2
    del log2_of_total_shown_points
    height = 2 ** _temp___height___but_log2
    width  = 2 **  _temp___width___but_log2
    del _temp___height___but_log2, _temp___width___but_log2
        #  buffer init to NA
    heat_map_value_for_NA___into_tensor = torch.tensor(heat_map_value_for_NA, dtype=torch.float16)
    assert heat_map_value_for_NA___into_tensor.shape == torch.Size([3]), "Set heat_map_value_for_True as rgb color. Or leave it as defaul."
    assert heat_map_value_for_NA___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
    assert heat_map_value_for_NA___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
    _temp___heat_map_value_for_NA___EXPANDhw_3 = heat_map_value_for_NA___into_tensor.reshape(shape=[1, 3]).expand([height*width, 3])
    buffer_for_plt = _temp___heat_map_value_for_NA___EXPANDhw_3.detach().clone()
    del heat_map_value_for_NA___into_tensor, _temp___heat_map_value_for_NA___EXPANDhw_3
    #assert buffer_for_plt.shape.__len__() == 2
    #assert _tensor_shape_check(buffer_for_plt, height*width, 3)
        #  buffer write             
    addr_in_tensor__for_True  = addr_in_tensor[label___b]
    addr_in_tensor__for_False = addr_in_tensor[label___b.logical_not()]

    heat_map_value_for_True___into_tensor = torch.tensor(heat_map_value_for_True , dtype=torch.float16)
    assert heat_map_value_for_True___into_tensor.shape == torch.Size([3]), "Set heat_map_value_for_True as rgb color. Or leave it as defaul."
    assert heat_map_value_for_True___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
    assert heat_map_value_for_True___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
    heat_map_value_for_True___EXPANDb_3  = heat_map_value_for_True___into_tensor.reshape([1, 3]).expand([addr_in_tensor__for_True .shape[0], 3])
    
    heat_map_value_for_False___into_tensor = torch.tensor(heat_map_value_for_False, dtype=torch.float16)
    assert heat_map_value_for_False___into_tensor.shape == torch.Size([3]), "set heat_map_value_for_True as rgb color. Or leave it as defaul."
    assert heat_map_value_for_False___into_tensor.le(1.).all(), "The range is [0., 1.], including 0., and 1."
    assert heat_map_value_for_False___into_tensor.ge(0.).all(), "The range is [0., 1.], including 0., and 1."
    heat_map_value_for_False___EXPANDb_3 = heat_map_value_for_False___into_tensor.reshape([1, 3]).expand([addr_in_tensor__for_False.shape[0], 3])

    buffer_for_plt[addr_in_tensor__for_True ] = heat_map_value_for_True___EXPANDb_3
    buffer_for_plt[addr_in_tensor__for_False] = heat_map_value_for_False___EXPANDb_3

    #  buffer reshape
    buffer_for_plt = buffer_for_plt.reshape(shape=[height, width, 3])
    #  plot it
    if plt_show:
        from matplotlib import pyplot as plt
        plt.imshow(buffer_for_plt)
        plt.title(heat_map_title)
        plt.show()
        pass
    return buffer_for_plt
if "test" and __DEBUG_ME__() and True:
    def ____test_____prototype____dataset____plt_show():
        if "equivalance         with plt.show" and False:
            '''compare this result against the upper one. If they look the same, then it's good'''

            length = 2
            batch = 2**length
            # init
            input_list = [x for x in range(batch)]#prototype
            assert input_list.__len__() == 4   #prototype
            assert input_list[-1] == 3         #prototype

            input___b_l = int_list_into_bool_tensor(input_list=input_list, length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            label___b = int_into_bool_tensor(input=0b1001, length=batch)# no length for label
            assert _tensor_shape_check(label___b, batch)
            buffer_for_plt = _prototype____dataset____plt_show(input___b_l=input___b_l, label___b=label___b, force__log2_of_total_shown_points=3, plt_show=False)
            assert _tensor_equal(buffer_for_plt, [[ [1, 1, 0.23],    [1., 0.1, 0.2],  
                                                    [1., 0.1, 0.2],   [1, 1, 0.23], ],
                                                    [[0., 0., 0.], #NA
                                                    [0., 0., 0.], #NA
                                                    [0., 0., 0.], #NA
                                                    [0., 0., 0.],] #NA
                                                                ], epsilon=0.01)
            
            assert _tensor_equal(buffer_for_plt[:,:,2], [[0.23, 0.2, 0.2, 0.23], [0., 0., 0., 0.]], epsilon=0.01)
            pass#/test

        if "manual cases     1  " and False:
            # T T F .
            # . F . T
            length = 3
            batch = 5
            input___b_l = int_list_into_bool_tensor(input_list=[0,1, 2 ,5, 7 ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            label___b = int_into_bool_tensor(input=0b11001, length=batch)# no length for label
            assert _tensor_shape_check(label___b, batch)
            buffer_for_plt = _prototype____dataset____plt_show(input___b_l=input___b_l, label___b=label___b, plt_show=False)
            assert _tensor_equal(buffer_for_plt[:,:,2], [[0.23, 0.23, 0.2, 0.], [0., 0.2, 0., 0.23]], epsilon=0.01)

            pass#/test

        if "randomly cases" and False:
            for _max_len in [1,2,3,5,10,30,100]: 
                for _ in range(6):

                    _max_number = _max_len*10
                    input_list = []
                    for _ in range(_max_len):
                        input_list.append(random.randint(0, _max_number))
                        pass
                    input_list = list(set(input_list))  #  remove the duplicated.
                    label_list = []
                    for _ in (input_list):
                        label_list.append(random.random()<0.5)
                        pass

                    #  calc
                    length = pos_of_first_one__int(max(input_list))
                    length = random.randint(length, length+3)
                    batch = input_list.__len__()

                    input___b_l = int_list_into_bool_tensor(input_list=input_list, length=length)
                    assert _tensor_shape_check(input___b_l, batch, length)
                    label___b = torch.tensor(label_list)# no length for label
                    assert _tensor_shape_check(label___b, batch)
                    ori__buffer_for_plt = _prototype____dataset____plt_show(input___b_l=input___b_l, label___b=label___b, plt_show=False)

                    for _ in range(3):
                        random_index = [x for x in range(batch)]
                        random.shuffle(random_index)
                        new__input___b_l=input___b_l[random_index]
                        new__label___b  =label___b  [random_index]
                        buffer_for_plt = _prototype____dataset____plt_show(input___b_l=new__input___b_l, label___b=new__label___b, plt_show=False)
                        assert _tensor_equal(ori__buffer_for_plt, buffer_for_plt, epsilon=0.01)
                        pass

                    pass#for _ 
                pass#for _max_len
            pass#/test

        if "how to break this function" and True:
            input___b_l = int_list_into_bool_tensor(input_list=[0,1,2,3,4,5,6], length=3)
            label___b = int_into_bool_tensor(input=0b1100110, length=7)# no length for label
            # buffer_for_plt = _prototype____dataset____plt_show(input___b_l=input___b_l, label___b=label___b, plt_show=False, \
            #                                                         force__log2_of_total_shown_points=1) # should not work
            pass#/test

        return 
    ____test_____prototype____dataset____plt_show()
    pass



1w 好像可以写类了。然后是特殊的xor数据集的生成，可以用可视化看看了。







