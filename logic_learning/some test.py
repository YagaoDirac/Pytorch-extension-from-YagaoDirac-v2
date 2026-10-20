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



if "case1":
    def case1():
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
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [0,1,2,3])

            #  partly sort
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
            iota_for_here = iota_of_batch[index_range__from:index_range__to]
            assert _tensor_equal(iota_for_here, [0,1])
            #now inside child.

            
            #  partly sort
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
    case1()
    pass














