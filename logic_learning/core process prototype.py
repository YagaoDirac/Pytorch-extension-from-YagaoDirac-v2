from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _bool_equal___0_as_false, all_False, \
        _tensor_equal, _tensor_shape_check, iota, is_list_int
#         iota, str_the_list
from logic_learning.Util_logic_learning import int_into_bool_tensor, int_list_into_bool_tensor, \
        bool_tensor_into_int_list, pos_of_first_one__int, print_2d_bool_tensor
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











if "before made into any function" and False:
    def before_made_into_any_function():
        if "without addr" and True:
            batch = 4
            length = 4
            pos_to_test = 0

            iota_of_batch = iota(batch)

            input___b_l = int_list_into_bool_tensor([
                0b0000, 
                0b1001, 
                0b0110, 
                0b1111, 
                ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)

            the_critical_bit = input___b_l[:,pos_to_test]
            index_for_False = iota_of_batch[the_critical_bit.logical_not()]
            index_for_True = iota_of_batch[the_critical_bit]
            assert _tensor_equal(index_for_False, [0,2])
            assert _tensor_equal(index_for_True , [1,3])
            how_many_False = index_for_False.nelement()
            assert how_many_False == 2

            new_buffer = torch.empty_like(input___b_l)
            assert new_buffer.dtype == torch.bool

            new_buffer[:how_many_False] = input___b_l[index_for_False]
            new_buffer[how_many_False:] = input___b_l[index_for_True]
            assert _bool_equal___0_as_false(new_buffer, int_list_into_bool_tensor([ 0b0000, 
                                                                                    0b0110, 
                                                                                    0b1001, 
                                                                                    0b1111, ], length=length))
            pass#/test



        if "with addr" and True:
            #init
            batch = 4
            length = 4
            pos_to_test = 0
            input___b_l = int_list_into_bool_tensor([
                0b0110, 
                0b1111, 
                0b0000, 
                0b1001, 
                ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)

            free_addr_bits = torch.ones(size=[4], dtype=torch.bool)
            assert free_addr_bits.dtype == torch.bool
            assert free_addr_bits.all()# TTTT
            index_range__from = 0
            index_range__to = batch
            #iota_of_batch = iota(index_range__to-index_range__from)+index_range__from  应该是写错了
            iota_of_batch = iota(batch)
            assert _tensor_equal(iota_of_batch, [0,1,2,3])

            #  partly sort
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [0,1,2,3])
            the_critical_bit = input___b_l[index_range__from:index_range__to, pos_to_test]
            index_for_False = iota_for_here[the_critical_bit.logical_not()]
            index_for_True  = iota_for_here[the_critical_bit]
            assert _tensor_equal(index_for_False, [0,2])
            assert _tensor_equal(index_for_True , [1,3])

            how_many_False = index_for_False.nelement()
            assert how_many_False == 2
            how_many_True = index_for_True.nelement()
            assert how_many_True == 2
                # 3 step swaps.  Do not merge
            new_buffer = torch.empty(size=[index_range__to - index_range__from, length], dtype=torch.bool, device=input___b_l.device)
            new_buffer[:how_many_False] = input___b_l[index_for_False]
            new_buffer[how_many_False:] = input___b_l[index_for_True]
            input___b_l[index_range__from:index_range__to] = new_buffer
            assert _bool_equal___0_as_false(input___b_l, int_list_into_bool_tensor([ 0b0110, 
                                                                                    0b0000, 
                                                                                    0b1111, 
                                                                                    0b1001, ], length=length))
            #  /partly sort


            #go to false child
            free_addr_bits[pos_to_test] = False
            assert _bool_equal___0_as_false(free_addr_bits, [0,1,1,1])
            #update the mask first^^^^^^^, then update the pos vvvvvvvvv
            pos_to_test = 1#   only for this test, 用计算相关度的那个模块来替换这一行。

            index_range__from = 0
            index_range__to = index_range__from+how_many_False
            assert index_range__to == 2
            #iota_of_batch = iota_of_batch[:how_many_False]   1w iota应该不变的。
            #assert _tensor_equal(iota_of_batch, [0,1])
            #now inside child.

            
            #  partly sort
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [0,1])
            the_critical_bit = input___b_l[index_range__from:index_range__to, pos_to_test]
            index_for_False = iota_for_here[the_critical_bit.logical_not()]
            index_for_True  = iota_for_here[the_critical_bit]
            assert _tensor_equal(index_for_False, [1])
            assert _tensor_equal(index_for_True , [0])

            how_many_False = index_for_False.nelement()
            assert how_many_False == 1
            how_many_True = index_for_True.nelement()
            assert how_many_True == 1
                # 3 step swaps.  Do not merge
            new_buffer = torch.empty(size=[index_range__to - index_range__from, length], dtype=torch.bool, device=input___b_l.device)
            new_buffer[:how_many_False] = input___b_l[index_for_False]
            new_buffer[how_many_False:] = input___b_l[index_for_True]
            input___b_l[index_range__from:index_range__to] = new_buffer
            assert _bool_equal___0_as_false(input___b_l, int_list_into_bool_tensor([0b0000, 
                                                                                    0b0110, 
                                                                                    0b1111, 
                                                                                    0b1001, ], length=length))
            #  /partly sort

            pass#/test
        
                
        return
    before_made_into_any_function()
    pass

def _prototype____partly_sort(input___b_l:torch.Tensor, free_addr_bits:torch.Tensor, 
        iota_of_batch:torch.Tensor, index_range__from:int, index_range__to:int, pos_to_test:int, 
        assert___iota___by_batch :int|None = None, 
        assert___index_for_False:list[int]|None = None, 
        assert___index_for_True:list[int]|None = None, 
        assert___how_many_False:int|None = None, 
        assert___how_many_True:int|None = None, 
        assert___result:torch.Tensor|None = None, 
        )->tuple[torch.Tensor, torch.Tensor, int, int]:
    '''return input___b_l, free_addr_bits, how_many_False, how_many_True'''

    #  safety
    assert isinstance(input___b_l, torch.Tensor)
    assert input___b_l.dtype == torch.bool
    assert input___b_l.shape.__len__() == 2

    assert isinstance(free_addr_bits, torch.Tensor)
    assert free_addr_bits.dtype == torch.bool
    assert free_addr_bits.shape.__len__() == 1
    assert free_addr_bits[pos_to_test] == True

    if assert___iota___by_batch is not None:
        assert type(assert___iota___by_batch) == int
        assert assert___iota___by_batch>0
        assert iota_of_batch[-1] == assert___iota___by_batch-1
        assert (iota_of_batch[1:]-iota_of_batch[:-1]).eq(1).all()
        pass# if assert___iota___by_batch
    #  /safety
    #  partly sort

    iota_for_here = iota_of_batch[index_range__from:index_range__to]
        #  old code
        # assert is_list_int(assert___iota_for_here)
        # assert _tensor_equal(iota_for_here, assert___iota_for_here)
        # pass

    the_critical_bit = input___b_l[index_range__from:index_range__to, pos_to_test]
    index_for_False = iota_for_here[the_critical_bit.logical_not()]
    index_for_True  = iota_for_here[the_critical_bit]
    if assert___index_for_False is not None:
        assert is_list_int(assert___index_for_False)
        assert _tensor_equal(index_for_False, assert___index_for_False)
        pass    
    if assert___index_for_True is not None:
        assert is_list_int(assert___index_for_True)
        assert _tensor_equal(index_for_True , assert___index_for_True)
        pass

    how_many_False = index_for_False.nelement()
    if assert___how_many_False is not None:
        assert how_many_False == assert___how_many_False
        pass
    how_many_True = index_for_True.nelement()
    if assert___how_many_True is not None:
        assert how_many_True == assert___how_many_True
        pass

        # 3 step swaps.  Do not merge
    new_buffer = torch.empty(size=[index_range__to - index_range__from, input___b_l.shape[1]], dtype=torch.bool, device=input___b_l.device)
    new_buffer[:how_many_False] = input___b_l[index_for_False]
    new_buffer[how_many_False:] = input___b_l[index_for_True]
    input___b_l[index_range__from:index_range__to] = new_buffer
    if assert___result is not None:
        assert assert___result.dtype == torch.bool
        assert _bool_equal___0_as_false(input___b_l, assert___result)
        pass

    #  free_addr
    free_addr_bits[pos_to_test] = False

    return input___b_l, free_addr_bits, how_many_False, how_many_True
def _prototype____get_new_range(is_True_child:bool, ori__index_range__from:int, ori__index_range__to:int, 
                                how_many_False:int, safety = False)->tuple[int, int]:
    '''return index_range__from, index_range__to'''

    if safety:  
        assert ori__index_range__from >=0
        assert how_many_False>=0
        assert ori__index_range__from+how_many_False <= ori__index_range__to
        assert ori__index_range__from < ori__index_range__to# no equal, the range is at least 1 element.
        pass
    
    if is_True_child:
        index_range__from = ori__index_range__from + how_many_False
        index_range__to   = ori__index_range__to
        return index_range__from, index_range__to
    else:
        index_range__from = ori__index_range__from
        index_range__to   = ori__index_range__from + how_many_False
        return index_range__from, index_range__to
    pass#end of function
if "test" and __DEBUG_ME__() and True:
    def ____test_____prototype____get_new_range():
        index_range__from, index_range__to = _prototype____get_new_range(is_True_child = False, 
                ori__index_range__from = 0, ori__index_range__to = 5, how_many_False = 2, safety = True)
        assert index_range__from == 0
        assert index_range__to   == 2

        index_range__from, index_range__to = _prototype____get_new_range(is_True_child = True, 
                ori__index_range__from = 0, ori__index_range__to = 5, how_many_False = 2, safety = True)
        assert index_range__from == 2
        assert index_range__to   == 5


        index_range__from, index_range__to = _prototype____get_new_range(is_True_child = False, 
                ori__index_range__from = 3, ori__index_range__to = 5, how_many_False = 2, safety = True)
        assert index_range__from == 3
        assert index_range__to   == 5

        index_range__from, index_range__to = _prototype____get_new_range(is_True_child = True, 
                ori__index_range__from = 3, ori__index_range__to = 5, how_many_False = 2, safety = True)
        assert index_range__from == 5
        assert index_range__to   == 5


        index_range__from, index_range__to = _prototype____get_new_range(is_True_child = False, 
                ori__index_range__from = 3, ori__index_range__to = 5, how_many_False = 0, safety = True)
        assert index_range__from == 3
        assert index_range__to   == 3

        index_range__from, index_range__to = _prototype____get_new_range(is_True_child = True, 
                ori__index_range__from = 3, ori__index_range__to = 5, how_many_False = 0, safety = True)
        assert index_range__from == 3
        assert index_range__to   == 5

        #  safety check,       these examples should not work
        #_prototype____get_new_range(is_True_child = True, ori__index_range__from = -1, ori__index_range__to = 5, how_many_False = 0,  safety=True)
        #_prototype____get_new_range(is_True_child = True, ori__index_range__from =  2, ori__index_range__to = 5, how_many_False = -1, safety=True)
        #_prototype____get_new_range(is_True_child = True, ori__index_range__from =  2, ori__index_range__to = 5, how_many_False = 4,  safety=True)
        #_prototype____get_new_range(is_True_child = True, ori__index_range__from =  2, ori__index_range__to = 2, how_many_False = 0,  safety=True)

        return
    ____test_____prototype____get_new_range()
    pass









if "after into function" and True:
    def after_into_function():
        if "equivalance      without addr       1 step" and True:
            #init
            batch = 4
            length = 5
            pos_to_test = 0
            input___b_l = int_list_into_bool_tensor([
                0b01100, 
                0b11111, 
                0b00000, 
                0b10011, 
                ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)

            free_addr_bits = torch.ones(size=[length], dtype=torch.bool)
            assert free_addr_bits.dtype == torch.bool
            assert free_addr_bits.all()# TTTT
            index_range__from = 0
            index_range__to = batch
            #iota_of_batch = iota(index_range__to-index_range__from)+index_range__from  应该是写错了
            iota_of_batch = iota(batch)
            assert _tensor_equal(iota_of_batch, [0,1,2,3])

            #  partly sort
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [0,1,2,3])
            the_critical_bit = input___b_l[index_range__from:index_range__to, pos_to_test]
            index_for_False = iota_for_here[the_critical_bit.logical_not()]
            index_for_True  = iota_for_here[the_critical_bit]
            assert _tensor_equal(index_for_False, [0,2])
            assert _tensor_equal(index_for_True , [1,3])

            how_many_False = index_for_False.nelement()
            assert how_many_False == 2
            how_many_True = index_for_True.nelement()
            assert how_many_True == 2
                # 3 step swaps.  Do not merge
            new_buffer = torch.empty(size=[index_range__to - index_range__from, length], dtype=torch.bool, device=input___b_l.device)
            new_buffer[:how_many_False] = input___b_l[index_for_False]
            new_buffer[how_many_False:] = input___b_l[index_for_True]
            input___b_l[index_range__from:index_range__to] = new_buffer
            assert _bool_equal___0_as_false(input___b_l, int_list_into_bool_tensor([0b01100, 
                                                                                    0b00000, 
                                                                                    0b11111, 
                                                                                    0b10011, ], length=length))
            free_addr_bits[pos_to_test] = False
            assert _bool_equal___0_as_false(free_addr_bits, [0,1,1,1,1])


            #  func ver
            #  func ver
            #  func ver
                #init
            batch = 4
            length = 5
            pos_to_test = 0
            input___b_l = int_list_into_bool_tensor([
                0b01100, 
                0b11111, 
                0b00000, 
                0b10011, 
                ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)

            free_addr_bits = torch.ones(size=[length], dtype=torch.bool)
            assert free_addr_bits.dtype == torch.bool
            assert free_addr_bits.all()# TTTTT
            index_range__from = 0
            index_range__to = batch
            #iota_of_batch = iota(index_range__to-index_range__from)+index_range__from  应该是写错了
            iota_of_batch = iota(batch)
            assert _tensor_equal(iota_of_batch, [0,1,2,3])

            free_addr_bits = torch.ones(size=[length], dtype=torch.bool)
            func_ver___input___b_l, func_ver__free_addr_bits, func_ver___how_many_False, func_ver___how_many_True = \
                    _prototype____partly_sort(input___b_l = input___b_l, free_addr_bits = free_addr_bits, 
                            iota_of_batch = iota_of_batch, 
                            index_range__from = index_range__from, index_range__to = index_range__to, pos_to_test = pos_to_test, 
                            assert___iota___by_batch = batch,
                            assert___index_for_False = [0,2], assert___index_for_True = [1,3])
            assert _bool_equal___0_as_false(func_ver___input___b_l, int_list_into_bool_tensor([ 0b01100, 
                                                                                                0b00000, 
                                                                                                0b11111, 
                                                                                                0b10011, ], length=length))
            assert _bool_equal___0_as_false(func_ver__free_addr_bits, [0,1,1,1,1])
            assert func_ver___how_many_False == 2
            assert func_ver___how_many_True  == 2
            pass#/test

        if "2 steps, root and the false child" and True:
            #init
            batch = 4
            length = 5
            pos_to_test = 0
            input___b_l = int_list_into_bool_tensor([
                0b01100, 
                0b11111, 
                0b00000, 
                0b10011, 
                ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)

            free_addr_bits = torch.ones(size=[length], dtype=torch.bool)
            assert free_addr_bits.dtype == torch.bool
            assert free_addr_bits.all()# TTTT
            index_range__from = 0
            index_range__to = batch
            #iota_of_batch = iota(index_range__to-index_range__from)+index_range__from  应该是写错了
            iota_of_batch = iota(batch)
            assert _tensor_equal(iota_of_batch, [0,1,2,3])

            #  partly sort
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [0,1,2,3])
            the_critical_bit = input___b_l[index_range__from:index_range__to, pos_to_test]
            index_for_False = iota_for_here[the_critical_bit.logical_not()]
            index_for_True  = iota_for_here[the_critical_bit]
            assert _tensor_equal(index_for_False, [0,2])
            assert _tensor_equal(index_for_True , [1,3])

            how_many_False = index_for_False.nelement()
            assert how_many_False == 2
            how_many_True = index_for_True.nelement()
            assert how_many_True == 2
                # 3 step swaps.  Do not merge
            new_buffer = torch.empty(size=[index_range__to - index_range__from, length], dtype=torch.bool, device=input___b_l.device)
            new_buffer[:how_many_False] = input___b_l[index_for_False]
            new_buffer[how_many_False:] = input___b_l[index_for_True]
            input___b_l[index_range__from:index_range__to] = new_buffer
            assert _bool_equal___0_as_false(input___b_l, int_list_into_bool_tensor([0b01100, 
                                                                                    0b00000, 
                                                                                    0b11111, 
                                                                                    0b10011, ], length=length))
            free_addr_bits[pos_to_test] = False
            assert _bool_equal___0_as_false(free_addr_bits, [0,1,1,1,1])
            #  /partly sort


            #go to false child
            #update the mask first^^^^^^^, then update the pos vvvvvvvvv
            pos_to_test = 1#   only for this test, 用计算相关度的那个模块来替换这一行。

            # index_range__from keeps
            index_range__to = index_range__from+how_many_False
            assert index_range__to == 2
            #iota_of_batch = iota_of_batch[:how_many_False]   1w iota应该不变的。
            #assert _tensor_equal(iota_of_batch, [0,1])
            #now inside child.

            
            #  partly sort
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [0,1])
            the_critical_bit = input___b_l[index_range__from:index_range__to, pos_to_test]
            index_for_False = iota_for_here[the_critical_bit.logical_not()]
            index_for_True  = iota_for_here[the_critical_bit]
            assert _tensor_equal(index_for_False, [1])
            assert _tensor_equal(index_for_True , [0])

            how_many_False = index_for_False.nelement()
            assert how_many_False == 1
            how_many_True = index_for_True.nelement()
            assert how_many_True == 1
                # 3 step swaps.  Do not merge
            new_buffer = torch.empty(size=[index_range__to - index_range__from, length], dtype=torch.bool, device=input___b_l.device)
            new_buffer[:how_many_False] = input___b_l[index_for_False]
            new_buffer[how_many_False:] = input___b_l[index_for_True]
            input___b_l[index_range__from:index_range__to] = new_buffer
            assert _bool_equal___0_as_false(input___b_l, int_list_into_bool_tensor([0b00000, 
                                                                                    0b01100, 
                                                                                    0b11111, 
                                                                                    0b10011, ], length=length))
            free_addr_bits[pos_to_test] = False
            assert _bool_equal___0_as_false(free_addr_bits, [0,0,1,1,1])
            #  /partly sort


            #  func ver
            #  func ver
            #  func ver
            input___b_l = int_list_into_bool_tensor([
                            0b01100, 
                            0b11111, 
                            0b00000, 
                            0b10011, 
                            ], length=length)
            pos_to_test = 0
            free_addr_bits = torch.ones(size=[length], dtype=torch.bool)
            index_range__from = 0
            index_range__to = batch
            assert index_range__to == 4

            func_ver___input___b_l, func_ver___free_addr_bits, func_ver___how_many_False, func_ver___how_many_True = \
                    _prototype____partly_sort(input___b_l = input___b_l, free_addr_bits = free_addr_bits, 
                            iota_of_batch = iota_of_batch, 
                            index_range__from = index_range__from, index_range__to = index_range__to, pos_to_test = pos_to_test, 
                            assert___iota___by_batch = batch, 
                            assert___index_for_False = [0,2], assert___index_for_True = [1,3])
            assert _bool_equal___0_as_false(func_ver___input___b_l, int_list_into_bool_tensor([ 0b01100, 
                                                                                                0b00000, 
                                                                                                0b11111, 
                                                                                                0b10011, ], length=length))
            assert _bool_equal___0_as_false(func_ver___free_addr_bits, [0,1,1,1,1])
            assert func_ver___how_many_False == 2
            assert func_ver___how_many_True  == 2

            #  update the data
            input___b_l = func_ver___input___b_l
            free_addr_bits = func_ver___free_addr_bits
            pos_to_test = 1#  test only

            index_range__from, index_range__to = _prototype____get_new_range(is_True_child = False, 
                    ori__index_range__from = index_range__from, ori__index_range__to = index_range__to, 
                    how_many_False = func_ver___how_many_False, safety = True)
            assert index_range__from == 0
            assert index_range__to   == 2
            #  /update the data

            func_ver___input___b_l, func_ver___free_addr_bits, func_ver___how_many_False, func_ver___how_many_True = \
                    _prototype____partly_sort(input___b_l = input___b_l, free_addr_bits = free_addr_bits, 
                            iota_of_batch = iota_of_batch, 
                            index_range__from = index_range__from, index_range__to = index_range__to, pos_to_test = pos_to_test, 
                            assert___iota___by_batch = batch, 
                            assert___index_for_False = [1], assert___index_for_True = [0])
            assert _bool_equal___0_as_false(func_ver___input___b_l, int_list_into_bool_tensor([ 0b00000, 
                                                                                                0b01100, 
                                                                                                0b11111, 
                                                                                                0b10011, ], length=length))
            assert _bool_equal___0_as_false(func_ver___free_addr_bits, [0,0,1,1,1])
            assert func_ver___how_many_False == 1
            assert func_ver___how_many_True  == 1

            pass#/test

        if "2 steps, root and the true child" and True:
            #init
            batch = 4
            length = 5
            pos_to_test = 0
            input___b_l = int_list_into_bool_tensor([
                0b01100, 
                0b11111, 
                0b00000, 
                0b10011, 
                ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)

            free_addr_bits = torch.ones(size=[length], dtype=torch.bool)
            assert free_addr_bits.dtype == torch.bool
            assert free_addr_bits.all()# TTTT
            index_range__from = 0
            index_range__to = batch
            #iota_of_batch = iota(index_range__to-index_range__from)+index_range__from  应该是写错了
            iota_of_batch = iota(batch)
            assert _tensor_equal(iota_of_batch, [0,1,2,3])

            #  partly sort
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [0,1,2,3])
            the_critical_bit = input___b_l[index_range__from:index_range__to, pos_to_test]
            index_for_False = iota_for_here[the_critical_bit.logical_not()]
            index_for_True  = iota_for_here[the_critical_bit]
            assert _tensor_equal(index_for_False, [0,2])
            assert _tensor_equal(index_for_True , [1,3])

            how_many_False = index_for_False.nelement()
            assert how_many_False == 2
            how_many_True = index_for_True.nelement()
            assert how_many_True == 2
                # 3 step swaps.  Do not merge
            new_buffer = torch.empty(size=[index_range__to - index_range__from, length], dtype=torch.bool, device=input___b_l.device)
            new_buffer[:how_many_False] = input___b_l[index_for_False]
            new_buffer[how_many_False:] = input___b_l[index_for_True]
            input___b_l[index_range__from:index_range__to] = new_buffer
            assert _bool_equal___0_as_false(input___b_l, int_list_into_bool_tensor([0b01100, 
                                                                                    0b00000, 
                                                                                    0b11111, 
                                                                                    0b10011, ], length=length))
            free_addr_bits[pos_to_test] = False
            assert _bool_equal___0_as_false(free_addr_bits, [0,1,1,1,1])
            #  /partly sort


            #go to false child
            #update the mask first^^^^^^^, then update the pos vvvvvvvvv
            pos_to_test = 1#   only for this test, 用计算相关度的那个模块来替换这一行。

            # index_range__from keeps
            index_range__from = index_range__from+how_many_False
            assert index_range__from == 2
            assert index_range__to   == 4
            #iota_of_batch = iota_of_batch[:how_many_False]   1w iota应该不变的。
            #assert _tensor_equal(iota_of_batch, [0,1])
            #now inside child.

            
            #  partly sort
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [2,3])
            the_critical_bit = input___b_l[index_range__from:index_range__to, pos_to_test]
            index_for_False = iota_for_here[the_critical_bit.logical_not()]
            index_for_True  = iota_for_here[the_critical_bit]
            assert _tensor_equal(index_for_False, [3])
            assert _tensor_equal(index_for_True , [2])

            how_many_False = index_for_False.nelement()
            assert how_many_False == 1
            how_many_True = index_for_True.nelement()
            assert how_many_True == 1
                # 3 step swaps.  Do not merge
            new_buffer = torch.empty(size=[index_range__to - index_range__from, length], dtype=torch.bool, device=input___b_l.device)
            new_buffer[:how_many_False] = input___b_l[index_for_False]
            new_buffer[how_many_False:] = input___b_l[index_for_True]
            input___b_l[index_range__from:index_range__to] = new_buffer
            assert _bool_equal___0_as_false(input___b_l, int_list_into_bool_tensor([
                                                                                    0b01100, 
                                                                                    0b00000, 
                                                                                    0b10011, 
                                                                                    0b11111, 
                                                                                    ], length=length))
            free_addr_bits[pos_to_test] = False
            assert _bool_equal___0_as_false(free_addr_bits, [0,0,1,1,1])
            #  /partly sort


            #  func ver
            #  func ver
            #  func ver
            input___b_l = int_list_into_bool_tensor([
                            0b01100, 
                            0b11111, 
                            0b00000, 
                            0b10011, 
                            ], length=length)
            pos_to_test = 0
            free_addr_bits = torch.ones(size=[length], dtype=torch.bool)
            index_range__from = 0
            index_range__to = batch
            assert index_range__to == 4

            func_ver___input___b_l, func_ver___free_addr_bits, func_ver___how_many_False, func_ver___how_many_True = \
                    _prototype____partly_sort(input___b_l = input___b_l, free_addr_bits = free_addr_bits, 
                            iota_of_batch = iota_of_batch, 
                            index_range__from = index_range__from, index_range__to = index_range__to, pos_to_test = pos_to_test, 
                            assert___iota___by_batch = batch, 
                            assert___index_for_False = [0,2], assert___index_for_True = [1,3])
            assert _bool_equal___0_as_false(func_ver___input___b_l, int_list_into_bool_tensor([ 0b01100, 
                                                                                                0b00000, 
                                                                                                0b11111, 
                                                                                                0b10011, ], length=length))
            assert _bool_equal___0_as_false(func_ver___free_addr_bits, [0,1,1,1,1])
            assert func_ver___how_many_False == 2
            assert func_ver___how_many_True  == 2

            #  update the data
            input___b_l = func_ver___input___b_l
            free_addr_bits = func_ver___free_addr_bits
            pos_to_test = 1#  test only

            index_range__from, index_range__to = _prototype____get_new_range(is_True_child = True, 
                    ori__index_range__from = index_range__from, ori__index_range__to = index_range__to, 
                    how_many_False = func_ver___how_many_False, safety = True)
            assert index_range__from == 2
            assert index_range__to   == 4
            #  /update the data

            func_ver___input___b_l, func_ver___free_addr_bits, func_ver___how_many_False, func_ver___how_many_True = \
                    _prototype____partly_sort(input___b_l = input___b_l, free_addr_bits = free_addr_bits, 
                            iota_of_batch = iota_of_batch, 
                            index_range__from = index_range__from, index_range__to = index_range__to, pos_to_test = pos_to_test, 
                            assert___iota___by_batch = batch, 
                            assert___index_for_False = [3], assert___index_for_True = [2])
            assert _bool_equal___0_as_false(func_ver___input___b_l, int_list_into_bool_tensor([ 
                                                                                                0b01100, 
                                                                                                0b00000, 
                                                                                                0b10011, 
                                                                                                0b11111, 
                                                                                                ], length=length))
            assert _bool_equal___0_as_false(func_ver___free_addr_bits, [0,0,1,1,1])
            assert func_ver___how_many_False == 1
            assert func_ver___how_many_True  == 1

            pass#/test

        return
    after_into_function()












    def bigger_dataset_test():
        #  path:    root ( . )  false( . )   false( . )  
        #           ........................ true ( . )  
        #           ............true ( . )   false( . )                  
        #           ........................ true ( . )  


        if "3 steps, full process" and True:

            batch = 23
            iota_of_batch = iota(batch)
            length = 29

            answer_for_input___b_l = int_list_into_bool_tensor([
                            0b_00000_000000_0000_00000001_000001, #0
                            0b_00000_000000_0000_00000001_000011, 
                            0b_00000_000000_0000_00000001_000111, 
                            0b_00000_001000_0001_00000010_000001, 
                            0b_00000_001000_0001_00000010_000011, 
                            0b_00000_010000_0010_00000100_000001, #5
                            0b_00000_010000_0010_00000100_000011, 
                            0b_00000_010000_0010_00000100_000111, 
                            0b_00000_011000_0011_00001000_000001, 
                            0b_00000_011000_0011_00001000_000011, 
                            0b_00000_011000_0011_00001000_000111, #10
                            0b_00000_011000_0011_00001000_001111, 
                            0b_00000_100000_0100_00010000_000001, 
                            0b_00000_100000_0100_00010000_000011, 
                            0b_00000_100010_0101_00100000_000001, 
                            0b_00000_100010_0101_00100000_000011, #15
                            0b_00000_100010_0101_00100000_000111, 
                            0b_00000_100010_0101_00100000_001111, 
                            0b_00000_100100_0110_01000000_000001, 
                            0b_00000_100100_0110_01000000_000011, 
                            0b_00000_100110_0111_10000000_000001, #20
                            0b_00000_100110_0111_10000000_000011, 
                            0b_00000_100110_0111_10000000_000111, 
                            #  01234 567890 1234 56789012 345678
                            ], length=length)

            #  init
            input___b_l = int_list_into_bool_tensor([
                            0b_00000_000000_0000_00000001_000001, #0
                            0b_00000_000000_0000_00000001_000011, 
                            0b_00000_000000_0000_00000001_000111, 
                            0b_00000_001000_0001_00000010_000001, 
                            0b_00000_001000_0001_00000010_000011, 
                            0b_00000_010000_0010_00000100_000001, #5
                            0b_00000_010000_0010_00000100_000011, 
                            0b_00000_010000_0010_00000100_000111, 
                            0b_00000_011000_0011_00001000_000001, 
                            0b_00000_011000_0011_00001000_000011, 
                            0b_00000_011000_0011_00001000_000111, #10
                            0b_00000_011000_0011_00001000_001111, 
                            0b_00000_100000_0100_00010000_000001, 
                            0b_00000_100000_0100_00010000_000011, 
                            0b_00000_100010_0101_00100000_000001, 
                            0b_00000_100010_0101_00100000_000011, #15
                            0b_00000_100010_0101_00100000_000111, 
                            0b_00000_100010_0101_00100000_001111, 
                            0b_00000_100100_0110_01000000_000001, 
                            0b_00000_100100_0110_01000000_000011, 
                            0b_00000_100110_0111_10000000_000001, #20
                            0b_00000_100110_0111_10000000_000011, 
                            0b_00000_100110_0111_10000000_000111, 
                            #  01234 567890 1234 56789012 345678
                            ], length=length)
            assert _tensor_shape_check(input___b_l, batch, length)
            _temp___input_in_list = input___b_l.tolist()
            random.shuffle(_temp___input_in_list)
            input___b_l = torch.tensor(_temp___input_in_list)
            del _temp___input_in_list

            #  basic dataset test
            #  colomns
            assert all_False(input___b_l[:, 0:5])
            assert all_False(input___b_l[:, 10:12])
            assert all_False(input___b_l[:, 23:25])
            assert           input___b_l[:, 28].all()
            #  sum to 1
            assert input___b_l[:, 15:23].to(torch.int32).sum(dim=1).shape == torch.Size([23])
            assert input___b_l[:, 15:23].to(torch.int32).sum(dim=1).eq(1).all()

            _temp___some_special_rows = input___b_l[input___b_l[:, 27].logical_not()]
            assert _tensor_shape_check(_temp___some_special_rows, 8, length)
            _temp___some_special_chunk = _temp___some_special_rows[:, 15:23]
            assert _tensor_shape_check(_temp___some_special_chunk, 8, 8)
            #prin_2d_bool_tensor(_temp___some_special_chunk, true_as="1", false_as=".")
            assert _temp___some_special_chunk.to(torch.int32).sum(dim=0).shape == torch.Size([8])
            assert _temp___some_special_chunk.to(torch.int32).sum(dim=0).eq(1).all()
            #  /basic dataset test



            #  root ( 1 )

            free_addr_bits = int_into_bool_tensor(0b_11111_111111_1111_11111111_111111,length=length)
            index_range__from = 0
            index_range__to = batch
            pos_to_test = 5

            input___b_l, free_addr_bits, how_many_False, how_many_True = _prototype____partly_sort( \
                    input___b_l = input___b_l, free_addr_bits = free_addr_bits, iota_of_batch = iota_of_batch, 
                    index_range__from = index_range__from, index_range__to = index_range__to, pos_to_test = pos_to_test, 
                    assert___iota___by_batch = batch) #, 
                    # assert___index_for_False = [0,2], assert___index_for_True = [1,3])
            assert _bool_equal___0_as_false(free_addr_bits, \
                    int_into_bool_tensor(0b_11111_011111_1111_11111111_111111,length=length))
            assert how_many_False == 12
            assert how_many_True  == 11

            #  basic dataset test
            #  colomns
            assert all_False(input___b_l[:, 0:5])
            assert all_False(input___b_l[:, 10:12])
            assert all_False(input___b_l[:, 23:25])
            assert           input___b_l[:, 28].all()
            #  sum to 1
            assert input___b_l[:, 15:23].to(torch.int32).sum(dim=1).shape == torch.Size([23])
            assert input___b_l[:, 15:23].to(torch.int32).sum(dim=1).eq(1).all()

            _temp___some_special_rows = input___b_l[input___b_l[:, 27].logical_not()]
            assert _tensor_shape_check(_temp___some_special_rows, 8, length)
            _temp___some_special_chunk = _temp___some_special_rows[:, 15:23]
            assert _tensor_shape_check(_temp___some_special_chunk, 8, 8)
            #prin_2d_bool_tensor(_temp___some_special_chunk, true_as="1", false_as=".")
            assert _temp___some_special_chunk.to(torch.int32).sum(dim=0).shape == torch.Size([8])
            assert _temp___some_special_chunk.to(torch.int32).sum(dim=0).eq(1).all()
            #  /basic dataset test
            #  special dataset test     root
            assert all_False(input___b_l[0:12,     5])
            assert           input___b_l[12:batch, 5].all()

            assert all_False(input___b_l[0:12,     12])
            assert           input___b_l[12:batch, 12].all()

            assert all_False(input___b_l[0:12,     15:19])
            assert           input___b_l[0:12,        19:23].to(torch.int32).sum(dim=1).shape == torch.Size([12])
            assert           input___b_l[0:12,        19:23].to(torch.int32).sum(dim=1).eq(1).all()
            assert           input___b_l[12:batch, 15:19]   .to(torch.int32).sum(dim=1).shape == torch.Size([11])
            assert           input___b_l[12:batch, 15:19]   .to(torch.int32).sum(dim=1).eq(1).all()
            assert all_False(input___b_l[12:batch,    19:23])
            #  /special dataset test     root

            #  update the data
            index_range__from, index_range__to = _prototype____get_new_range(is_True_child = False, 
                    ori__index_range__from = index_range__from, ori__index_range__to = index_range__to, 
                    how_many_False = how_many_False, safety = True)
            assert index_range__from == 0
            assert index_range__to   == 12
        
            #  /update the data



















            #  root ( 1 )  false( 2 )
            pos_to_test = 6#  test only
            assert _bool_equal___0_as_false(free_addr_bits, int_into_bool_tensor( \
                    0b_11111_011111_1111_11111111_111111,length=length))
                    #  01234 567890

            input___b_l, free_addr_bits, how_many_False, how_many_True = _prototype____partly_sort( \
                    input___b_l = input___b_l, free_addr_bits = free_addr_bits, iota_of_batch = iota_of_batch, 
                    index_range__from = index_range__from, index_range__to = index_range__to, pos_to_test = pos_to_test, 
                    assert___iota___by_batch = batch) #, 
                    # assert___index_for_False = [0,2], assert___index_for_True = [1,3])
            assert _bool_equal___0_as_false(free_addr_bits, \
                    int_into_bool_tensor(0b_11111_001111_1111_11111111_111111,length=length))
            assert how_many_False == 5
            assert how_many_True  == 7

            #  basic dataset test
            #  colomns
            assert all_False(input___b_l[:, 0:5])
            assert all_False(input___b_l[:, 10:12])
            assert all_False(input___b_l[:, 23:25])
            assert           input___b_l[:, 28].all()
            #  sum to 1
            assert input___b_l[:, 15:23].to(torch.int32).sum(dim=1).shape == torch.Size([23])
            assert input___b_l[:, 15:23].to(torch.int32).sum(dim=1).eq(1).all()

            _temp___some_special_rows = input___b_l[input___b_l[:, 27].logical_not()]
            assert _tensor_shape_check(_temp___some_special_rows, 8, length)
            _temp___some_special_chunk = _temp___some_special_rows[:, 15:23]
            assert _tensor_shape_check(_temp___some_special_chunk, 8, 8)
            #prin_2d_bool_tensor(_temp___some_special_chunk, true_as="1", false_as=".")
            assert _temp___some_special_chunk.to(torch.int32).sum(dim=0).shape == torch.Size([8])
            assert _temp___some_special_chunk.to(torch.int32).sum(dim=0).eq(1).all()
            #  /basic dataset test
            #  special dataset test     root
            assert all_False(input___b_l[0:12,     5])
            assert           input___b_l[12:batch, 5].all()

            assert all_False(input___b_l[0:12,     12])
            assert           input___b_l[12:batch, 12].all()

            assert all_False(input___b_l[0:12,     15:19])
            assert           input___b_l[0:12,        19:23].to(torch.int32).sum(dim=1).shape == torch.Size([12])
            assert           input___b_l[0:12,        19:23].to(torch.int32).sum(dim=1).eq(1).all()
            assert           input___b_l[12:batch, 15:19]   .to(torch.int32).sum(dim=1).shape == torch.Size([11])
            assert           input___b_l[12:batch, 15:19]   .to(torch.int32).sum(dim=1).eq(1).all()
            assert all_False(input___b_l[12:batch,    19:23])
            #  /special dataset test     root
            #  special dataset test     root false
            assert all_False(input___b_l[0:5,   6])
            assert           input___b_l[5:12,  6].all()

            assert all_False(input___b_l[0:5,   13])
            assert           input___b_l[5:12,  13].all()

            assert all_False(input___b_l[0:5,      19:21])
            assert all_False(input___b_l[5:batch,  21:23])
            #  /special dataset test     root false

            #  update the data
            index_range__from, index_range__to = _prototype____get_new_range(is_True_child = False, 
                    ori__index_range__from = index_range__from, ori__index_range__to = index_range__to, 
                    how_many_False = how_many_False, safety = True)
            assert index_range__from == 0
            assert index_range__to   == 5
            #  /update the data






















            #  root ( 1 )  false( 2 )  false( 3 )
            pos_to_test = 7#  test only
            assert _bool_equal___0_as_false(free_addr_bits, int_into_bool_tensor( \
                    0b_11111_001111_1111_11111111_111111,length=length))
                    #  01234 567890

            input___b_l, free_addr_bits, how_many_False, how_many_True = _prototype____partly_sort( \
                    input___b_l = input___b_l, free_addr_bits = free_addr_bits, iota_of_batch = iota_of_batch, 
                    index_range__from = index_range__from, index_range__to = index_range__to, pos_to_test = pos_to_test, 
                    assert___iota___by_batch = batch) #, 
                    # assert___index_for_False = [0,2], assert___index_for_True = [1,3])
            assert _bool_equal___0_as_false(free_addr_bits, \
                    int_into_bool_tensor(0b_11111_000111_1111_11111111_111111,length=length))
            assert how_many_False == 3
            assert how_many_True  == 2

            #  basic dataset test
            #  colomns
            assert all_False(input___b_l[:, 0:5])
            assert all_False(input___b_l[:, 10:12])
            assert all_False(input___b_l[:, 23:25])
            assert           input___b_l[:, 28].all()
            #  sum to 1
            assert input___b_l[:, 15:23].to(torch.int32).sum(dim=1).shape == torch.Size([23])
            assert input___b_l[:, 15:23].to(torch.int32).sum(dim=1).eq(1).all()

            _temp___some_special_rows = input___b_l[input___b_l[:, 27].logical_not()]
            assert _tensor_shape_check(_temp___some_special_rows, 8, length)
            _temp___some_special_chunk = _temp___some_special_rows[:, 15:23]
            assert _tensor_shape_check(_temp___some_special_chunk, 8, 8)
            #prin_2d_bool_tensor(_temp___some_special_chunk, true_as="1", false_as=".")
            assert _temp___some_special_chunk.to(torch.int32).sum(dim=0).shape == torch.Size([8])
            assert _temp___some_special_chunk.to(torch.int32).sum(dim=0).eq(1).all()
            #  /basic dataset test
            #  special dataset test     root
            assert all_False(input___b_l[0:12,     5])
            assert           input___b_l[12:batch, 5].all()

            assert all_False(input___b_l[0:12,     12])
            assert           input___b_l[12:batch, 12].all()

            assert all_False(input___b_l[0:12,     15:19])
            assert           input___b_l[0:12,        19:23].to(torch.int32).sum(dim=1).shape == torch.Size([12])
            assert           input___b_l[0:12,        19:23].to(torch.int32).sum(dim=1).eq(1).all()
            assert           input___b_l[12:batch, 15:19]   .to(torch.int32).sum(dim=1).shape == torch.Size([11])
            assert           input___b_l[12:batch, 15:19]   .to(torch.int32).sum(dim=1).eq(1).all()
            assert all_False(input___b_l[12:batch,    19:23])
            #  /special dataset test     root
            #  special dataset test     root false
            assert all_False(input___b_l[0:5,   6])
            assert           input___b_l[5:12,  6].all()

            assert all_False(input___b_l[0:5,   13])
            assert           input___b_l[5:12,  13].all()

            assert all_False(input___b_l[0:5,      19:21])
            assert all_False(input___b_l[5:batch,  21:23])
            #  /special dataset test     root false
            #  special dataset test     root false false
            assert all_False(input___b_l[0:3,  6])
            1w继续
            1w继续
            1w继续
            1w继续
            assert           input___b_l[3:5,  6].all()

            assert all_False(input___b_l[0:5,   13])
            assert           input___b_l[5:12,  13].all()

            assert all_False(input___b_l[0:5,      19:21])
            assert all_False(input___b_l[5:batch,  21:23])
            #  /special dataset test     root false false




            pass#/test






        
                
        return
    bigger_dataset_test()
    pass





