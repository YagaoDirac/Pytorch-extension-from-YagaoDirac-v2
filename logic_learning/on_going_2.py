'''
Author:YagaoDirac (also on X.com)

content
part1, infrastructures of logic algebra.
part2, logic learning.

a lot untested and unfinished. Simply search these 2 words to find them.

'''

from typing import Optional, TypeVar, Callable 

#import SupportsRichComparison
#from _typeshed import SupportsRichComparisonT, SupportsRichComparison
from enum import Enum
import random
import sys
import random
import time
from datetime import datetime


dataset:'Dataset'

def _line_()->int:
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######


    

if False:
    
    # _T = TypeVar("_T")
    # def dont_use___untested_yet___binary_search(the_list:list[_T], target:SupportsRichComparison, 
    #             key: Callable[[_T], SupportsRichComparison] = lambda item:item)->int:
    #     '''return the index. -1 for not found.'''
    #     '''type hint is copy pasted from the [].sort(). It needs 
    #     >>> from typing import TypeVar, Callable 
    #     >>> from _typeshed import SupportsRichComparison
    #     '''
    #     found = False
    #     left = 0
    #     right = dataset.__len__()-1
    #     mid:int = (left+right)//2
    #     while left<=right:
    #         if target<key(the_list[mid]):
    #             right = mid-1# two valid style.
    #             pass
    #         if target>key(the_list[mid]):
    #             left = mid+1# two valid style.
    #             pass
    #         if target == key(the_list[mid]):
    #             return mid
    #         pass#while of binary search.
    #     return -1
    pass
                    

def count_ones(input:int)->int:
    assert input>=0
    result = 0
    while input>0:
        result = result +1
        #tail
        input = input >>1
        pass
    return result
if "test" and False:
    assert count_ones(0b0) == 0
    assert count_ones(0b1) == 1
    assert count_ones(0b10) == 2
    assert count_ones(0b11) == 2
    assert count_ones(0b100) == 3
    assert count_ones(0b10000) == 5
    assert count_ones(0b10010) == 5
    assert count_ones(0b10110) == 5
    pass
    
def readable_binary(input:int, pad_to_length = 0)->str:
    raw_str = f"{input:b}"
    needs_zeros = pad_to_length - raw_str.__len__()
    if needs_zeros>0:
        return "0"*needs_zeros+raw_str
    else:
        return raw_str
    pass#end of function
if "test" and False:
    assert readable_binary(0b1) == "1"
    assert readable_binary(0b0) == "0"
    assert readable_binary(0b10) == "10"
    assert readable_binary(0b1,3) == "001"
    assert readable_binary(0b0,3) == "000"
    assert readable_binary(0b10,3) == "010"
    assert readable_binary(0b10,5) == "00010"
    pass



#still useful, but not used in the class.
def is_addr_valid(bit_in_use_mask:int, fixed_addr:int)->bool:
    '''Checks if fixed_addr has no 1s outside the bits marked by bit_in_use_mask
    
    Mainly for debug.'''
    reversed_mask = ~bit_in_use_mask
    bitwise_and = fixed_addr&(reversed_mask)
    result = 0 == bitwise_and
    return result
if "test" and False:
    assert is_addr_valid(int("0",2), int("0",2))
    #assert is_addr_valid(int("0",2), int("1",2))
    assert is_addr_valid(int("1",2), int("0",2))
    assert is_addr_valid(int("1",2), int("1",2))
    #assert is_addr_valid(int("1",2), int("10",2))
    assert is_addr_valid(int("10",2), int("10",2))
    #assert is_addr_valid(int("10",2), int("11",2))
    #assert is_addr_valid(int("10",2), int("1",2))
    assert is_addr_valid(int("10",2), int("0",2))
    pass
#not in use. But let's leave it here.
class OneTimeSetup_Bool:
    def __init__(self):
        self.uninit = True
        self.value = False
        pass
    def set(self, set_to = True):
        if __debug__:
            assert self.uninit, "already set before."
            self.uninit = False
            pass
        self.value = set_to
        return
        #end of function
    def reset(self):
        if __debug__:
            assert self.uninit, "already set before."
            self.uninit = False
            return
        return
        #end of function
    def get_inited(self)->bool:
        return not self.uninit
    def get_value_safe(self)->bool:
        if __debug__:
            assert not self.uninit, "not set yet."
            pass
        return self.value
    def __str__(self):
        if self.uninit:
            return "UnInit"
        else:
            return str(self.value)
    #end of class
if "test" and False:
    print(OneTimeSetup_Bool())
    
    print("---get_value_safe")
    a_OneTimeSetup_Bool = OneTimeSetup_Bool()
    #a.get_value_safe()
    a_OneTimeSetup_Bool.set()
    print(a_OneTimeSetup_Bool.get_value_safe())
    
    print("---test get_inited")
    a_OneTimeSetup_Bool = OneTimeSetup_Bool()
    print(a_OneTimeSetup_Bool.get_inited())
    a_OneTimeSetup_Bool.set()
    print(a_OneTimeSetup_Bool.get_inited())
    print("---test set and reset")
    a_OneTimeSetup_Bool = OneTimeSetup_Bool()
    a_OneTimeSetup_Bool.set()
    print(a_OneTimeSetup_Bool)
    a_OneTimeSetup_Bool = OneTimeSetup_Bool()
    a_OneTimeSetup_Bool.reset()
    print(a_OneTimeSetup_Bool)
    #only set once.
    a_OneTimeSetup_Bool = OneTimeSetup_Bool()
    a_OneTimeSetup_Bool.set()
    #a.set()
    a_OneTimeSetup_Bool = OneTimeSetup_Bool()
    a_OneTimeSetup_Bool.set()
    #a.reset()
    a_OneTimeSetup_Bool = OneTimeSetup_Bool()
    a_OneTimeSetup_Bool.reset()
    #a.set()
    a_OneTimeSetup_Bool = OneTimeSetup_Bool()
    a_OneTimeSetup_Bool.reset()
    #a.reset()
    pass




class Bitset:
    pass




class Dataset:
    max_input_bits:int
    data:list[tuple[int,bool]]
    is_sorted:bool
    
    @staticmethod
    def todo_list()->str:
        return '''
    rand_true_xor
    生成复杂xor专用函数。
    '''
    
    @staticmethod
    def rand_true_xor(max_input_bits:int, only_full = True, seed:Optional[int]=None)->tuple['Dataset',str]:
        '''return (result, instruction)'''
        #safety first
        
        assert only_full, "this version only handles this case."
        
        if seed:
            random.seed(time.time())
            pass
        
        assert max_input_bits>=3
        _free_count = max_input_bits-2
        count_list:list[int] = []
        if only_full:
            irr_bit_count = random.randint(0, _free_count)
            _free_count = _free_count - irr_bit_count
            xor_bit_count = _free_count +2
            one_bit_count = 0
            zero_bit_count = 0
            pass
        else:#full or non-full.
            for _ in range(3):
                _rand_int = random.randint(0, _free_count)
                _free_count = _free_count - _rand_int
                count_list.append(_rand_int)
                pass
            count_list.append(_free_count)
            
            for _ in range(3):
                random.shuffle(count_list)
                pass
            xor_bit_count = count_list.pop()+2
            irr_bit_count = count_list.pop()
            one_bit_count = count_list.pop()
            zero_bit_count = count_list.pop()
            pass#if only_full   else
        _temp_irr_bits = "." * xor_bit_count + "i" * irr_bit_count + "1" * one_bit_count + "0" * zero_bit_count
        #irr_bits_list:list[str] = _temp_irr_bits.split(sep="")
        irr_bits_list:list[str] = list(_temp_irr_bits)
        random.shuffle(irr_bits_list)
        irr_bits = "".join(irr_bits_list)
        
        _rand_flag = (random.random()<0.5)
        result = Dataset.inst_2_xor_like_dataset(irr_bits, not_after_xor=_rand_flag)
        if only_full:#safety
            assert result.data.__len__() == (1<<max_input_bits)
            pass
        
        return (result, irr_bits)
    
    @staticmethod
    def inst_2_xor_like_dataset(irr_bits:str, not_after_xor = False, )->'Dataset':
        result = Dataset.from_str("0")
        if not_after_xor:
            result = Dataset.from_str("1")
            pass
        
        for i_from_left_side in range(irr_bits.__len__()):
            i_from_right_side = irr_bits.__len__()-1-i_from_left_side
            char = irr_bits[i_from_right_side]
            #one_shift_by_i = 1<<i
            match char:
                case "0":#one more input bit, but no more data, it's non full now.
                    result.add_const_bit_into_addr(i_from_left_side, False)
                    pass
                case "1":#one more input bit, but no more data, it's non full now.
                    result.add_const_bit_into_addr(i_from_left_side, True)
                    pass
                case ".":
                    other = result.clone()
                    result.add_const_bit_into_addr(i_from_left_side, False)
                    other.add_const_bit_into_addr(i_from_left_side, True)
                    other.reverse()#<<<<<<<<<
                    result.extend(other)
                    pass
                case "i":
                    other = result.clone()
                    result.add_const_bit_into_addr(i_from_left_side, False)
                    other.add_const_bit_into_addr(i_from_left_side, True)
                    #result.reverse()#<<<<<<<<<
                    result.extend(other)
                    pass
                case _:
                    assert False, "unreachable"
            pass#for
        return result
    
    @staticmethod
    def _debug__new_empty(max_input_bits:int)->'Dataset':
        return Dataset(max_input_bits = max_input_bits)
    
    def __init__(self, max_input_bits:int, input:Optional[list[tuple[int,bool]]] = None, is_sorted:bool = False):
        self.max_input_bits = max_input_bits
        self.data = input or []
        if self.data.__len__()<2: 
            self.is_sorted = True
            pass
        else:
            self.is_sorted = is_sorted
            pass
        self.safety_check___sort_btw()
        pass#end of function.
    
    @staticmethod
    def rand__sorted(max_input_bits:int, p_False:float, p_True:float, seed:Optional[int]=None)->"Dataset":
        '''The result looks like addr:value. The addr is naturally sorted???
        
        For debug purpose.'''
        dataset:list[tuple[int,bool]] = []
        if seed:
            random.seed(time.time())
            pass
        p_both = p_True + p_False
        for i in range(1<<max_input_bits):
            r = random.random()
            if r < p_False:
                dataset.append((i, False))
            elif r < p_both:
                dataset.append((i, True))
            else:
                # irrelevant items are not in dataset
                pass
            pass
        return Dataset(max_input_bits, dataset)
    
    @staticmethod
    def from_str(input:str, recommend_length_according_to_only_relevent_items = False, increase_input_bit_count_to = -1)->"Dataset":
        '''return (dataset, min of field length)'''
        if input.__len__() == 0:
            assert increase_input_bit_count_to>=0
            return Dataset(increase_input_bit_count_to)
        
        
        raw_list:list[tuple[int,bool]] = []
        addr = 0
        for value in input:
            match value:
                case "1":
                    raw_list.append((addr,True))
                    addr = addr +1
                    pass
                case "0":
                    raw_list.append((addr,False))
                    addr = addr +1
                    pass
                case "_":
                    addr = addr +1
                    pass
                case " ":
                    pass
                case ",":
                    pass
                case "\n":
                    pass
                case _:
                    assert False, "unreachable, unknown char in the input. use only 1, 0, _ to represent true, false and irrelevant."
            pass
        
        if raw_list.__len__() == 0:
            assert increase_input_bit_count_to>=0
            return Dataset(increase_input_bit_count_to)
        
        min_of_field_len = 0
        if recommend_length_according_to_only_relevent_items:
            temp_last_addr = raw_list[-1][0]
            pass
        else:#including irr at tail of the string.
            temp_last_addr = addr-1
            pass
        min_of_field_len = count_ones(temp_last_addr)
        pass
        result = Dataset(min_of_field_len, raw_list, True)
        if increase_input_bit_count_to>0:
            result.increase_input_bit_count(increase_input_bit_count_to)
            pass
        return result

    def reverse(self):
        for i in range(self.data.__len__()):
            item = self.data[i]
            self.data[i] = (item[0], not item[1])
            pass
        pass
    
    def extend(self, other:'Dataset'):
        assert self.is_sorted
        assert other.is_sorted
        if other.data.__len__() == 0:
            return
        
        if self.max_input_bits < other.max_input_bits:
            self.max_input_bits = other.max_input_bits
            pass
        self.data.extend(other.data)
        self.is_sorted = False
        self.safety_check___sort_btw()        
        pass
    
    def clone(self)->'Dataset':
        _temp_list = []
        _temp_list.extend(self.data)
        result = Dataset(self.max_input_bits, _temp_list)
        return result

    def increase_input_bit_count(self, new_count:int):
        assert new_count>self.max_input_bits
        self.max_input_bits = new_count
        pass
    
    def get_input_bits(self)->int:
        return self.max_input_bits
    def get_output_bits(self)->int:
        return 1

    def add_addr(self, input_addr:int, output_bool:bool, safety_check = True):
        if safety_check:
            assert count_ones(input_addr)<= self.max_input_bits
            for _addr_result in self.data:
                assert _addr_result[0] != input_addr
                pass
            pass
        self.data.append((input_addr, output_bool))
        self.is_sorted = False
        pass
    
    def sort(self):
        self.data.sort(key=lambda item:item[0])
        self.is_sorted = True
        pass
    
    def find_addr(self, addr:int)->tuple[bool, int]:
        '''return (found, index)'''
        if not self.is_sorted:
            self.sort()
            pass
        #binary search
        found = False
        left = 0
        right = self.data.__len__()-1
        while left<=right:
            mid:int = (left+right)//2
            temp_addr = self.data[mid][0]
            if addr<temp_addr:
                right = mid-1# two valid style.
                continue
            elif addr>temp_addr:
                left = mid+1# two valid style.
                continue
            else:#guess_addr == dataset[mid][0]:
                found = True
                break
            pass#while of binary search.
        return (found, mid)
    
    def get_irr_addr___sorts_self(self)->tuple[bool, list[int]]:
        '''return (at least one item in dataset, the irr list). 
        
        If obj is empty, them all addresses are irr. In this case, no list is returned. 
        Manually handle it.'''
        if not self.is_sorted:
            self.sort()
            pass
        
        if self.data.__len__() == 0:
            return (False, [])
        index = 0
        irr_addr:list[int] = []
        addr = 0
        one_shift_self_max_input_bits = 1<<self.max_input_bits
        for addr in range(one_shift_self_max_input_bits):
            if addr == self.data[index][0]:
                index+=1
                if self.data.__len__() == index:
                    addr = addr + 1
                    break
                pass
            else:
                irr_addr.append(addr)
                pass
            pass
        for addr2 in range(addr, one_shift_self_max_input_bits):
            irr_addr.append(addr2)
            pass
        return (True, irr_addr)

    def get_addr_as_set(self)->set:
        result:set[int] = set()
        for item in self.data:
            result.add(item[0])
            pass
        return result
    
    def add_const_bit_into_addr(self, index_from_right_side:int, value:bool):
        '''xyz, add a T in 1st(from right side), is xyTz'''
        
        '''xyz, 1st, T. 
        xyz into xy0 and z. The mask is 110(1<<3 - 1<<1, 3is the max_input_bits) and 1(1<<1 -1).
        xy0 into xy00, shift by 1 bit.
        insert T0(T<<1, 1 is the index)
        
        wxyz, 2nd, T.
        wxyz into wx00 and yz. The mask is 1100(1<<4 - 1<<2, 4is the max_input_bits) and 11(1<<2 -1)
        wx00 into wx000, shift by 1 bit.
        insert T00(T<<2, 2 is the index)
        '''
        assert index_from_right_side <= self.max_input_bits
        assert index_from_right_side >=0
        
        insert_this = value<<index_from_right_side
        mask_for_high_bits = (1<<self.max_input_bits)-(1<<index_from_right_side)
        mask_for_low_bits = (1<<index_from_right_side)-1
        new_data:list[tuple[int,bool]] = []
        for item in self.data:
            high_part__before_shift = item[0] & mask_for_high_bits
            high_part = high_part__before_shift <<1 
            low_part = item[0] & mask_for_low_bits
            new_addr = high_part|insert_this|low_part
            new_data.append((new_addr, item[1]))
            pass
        self.data = new_data
        self.max_input_bits = self.max_input_bits +1
        pass
    
    def get_subset(self, bitmask:int, addr:int)->"Dataset":
        assert is_addr_valid(bitmask, addr)
        assert self.is_sorted
        new_raw_data:list[tuple[int,bool]] = []
        for item in self.data:
            masked_addr_of_item = item[0]&bitmask
            if masked_addr_of_item == addr:
                new_raw_data.append(item)
                pass
            pass
        return Dataset(self.max_input_bits, new_raw_data, True)
    
    def safety_check___sort_btw(self):
        if not self.is_sorted:
            self.sort()
            pass
        if self.data.__len__() == 0:
            return
        elif self.data.__len__() == 1:
            assert self.data[0][0]>=0
            pass
        else:
            for i in range(self.data.__len__()-1):
                item_1 = self.data[i]
                assert item_1[0]>=0
                item_2 = self.data[i+1]
                assert item_1[0] < item_2[0]
                pass
            #assert self.data[-1][0]>=0 not needed.
            pass
        _count_ones = count_ones(self.data[-1][0])
        assert self.max_input_bits>=_count_ones
        pass #end of function
    
    def readable_str(self, addr_as_binary = True, pad_with_zero = True, add_title = True)->str:
        pad_to = 0
        if pad_with_zero:
            pad_to = self.max_input_bits
            pass
        
        result_list:list[str] =  []
        for _addr_result in self.data:
            addr_for_this_iter = readable_binary(_addr_result[0], pad_to)
            str_for_this_iter = f"{addr_for_this_iter}:{int(_addr_result[1])}"            
            result_list.append(str_for_this_iter)
            pass
        result_str = ", ".join(result_list)
        if add_title:
            result_str = "<Dataset>"+result_str
            pass
        return result_str
    
    def log_the_error(self, filename = "wrong case log.txt", comment = ""):
        if filename == "wrong case log.txt":
            fds=543
        
        line_number_1 = sys._getframe(1).f_lineno
        line_number_2 = sys._getframe(2).f_lineno
        with open(filename, mode="+a", encoding="utf-8") as file:
            #old 
            # file.write(f"{comment} (line:{line_number_1} or {line_number_2})\n")
            # input_bits_count_str = f"input_bits_count = {self.max_input_bits}\n"
            # file.write(input_bits_count_str)
            # dataset_str:str = f"dataset = Dataset(max_input_bits, {str(dataset.data)})\n"
            # #dataset_str:str = f"dataset = "+dataset.__str__()
            # file.write(dataset_str)
            # file.write("\n")
            
            file.write(f'''    #{comment} (line:{line_number_1} or {line_number_2})
    input_bits_count = {self.max_input_bits}
    dataset = Dataset(input_bits_count, {str(dataset.data)})
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)
    #a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
    ''')
            pass
        pass
    pass
    
    
    
if "clone, reverse, xor_like_gen, extend, rand_true_xor" and True:
    for i in range(3, 10):
        dataset_1 = Dataset.rand__sorted(i, 0.2, 0.2)
        dataset_2 = dataset_1.clone()
        assert dataset_1.data == dataset_2.data
        pass
    
    dataset_1 = Dataset.from_str("01101010")
    dataset_1.reverse()
    dataset_2 = Dataset.from_str("10010101")
    assert dataset_1.data == dataset_2.data
    dataset_1 = Dataset.from_str("011_10_0")
    dataset_1.reverse()
    dataset_2 = Dataset.from_str("100_01_1")
    assert dataset_1.data == dataset_2.data
    dataset_1 = Dataset.from_str("011_1___")
    dataset_1.reverse()
    dataset_2 = Dataset.from_str("100_0___")
    assert dataset_1.data == dataset_2.data
    
    dataset = Dataset.inst_2_xor_like_dataset("")
    assert dataset.data == [(0b0, False)]
    dataset = Dataset.inst_2_xor_like_dataset("", True)
    assert dataset.data == [(0b0, True)]
    dataset = Dataset.inst_2_xor_like_dataset(".")
    assert dataset.data == [(0b0,False),(0b1,True),]
    dataset = Dataset.inst_2_xor_like_dataset(".", True)
    assert dataset.data == [(0b0,True),(0b1,False),]
    
    dataset = Dataset.inst_2_xor_like_dataset("..")
    assert dataset.data == Dataset.from_str("0110").data
    dataset = Dataset.inst_2_xor_like_dataset("i.")
    assert dataset.data == Dataset.from_str("0101").data
    dataset = Dataset.inst_2_xor_like_dataset("ii")
    assert dataset.data == Dataset.from_str("0000").data
    dataset = Dataset.inst_2_xor_like_dataset("1.")
    assert dataset.data == Dataset.from_str("__01").data
    dataset = Dataset.inst_2_xor_like_dataset("0.")
    assert dataset.data == Dataset.from_str("01__").data
    dataset = Dataset.inst_2_xor_like_dataset("i..")
    assert dataset.data == Dataset.from_str("01100110").data
    dataset = Dataset.inst_2_xor_like_dataset("...")
    assert dataset.data == Dataset.from_str("01101001").data
    
    
    dataset_1 = Dataset.from_str("__11")
    dataset_2 = Dataset.from_str("11__")
    dataset_1.extend(dataset_2)
    assert dataset_1.data == Dataset.from_str("1111").data
    assert dataset_1.max_input_bits == 2
    dataset_1 = Dataset.from_str("__01")
    dataset_2 = Dataset.from_str("10__")
    dataset_1.extend(dataset_2)
    assert dataset_1.data == Dataset.from_str("1001").data
    assert dataset_1.max_input_bits == 2
    dataset_1 = Dataset.from_str("_0_1")
    dataset_2 = Dataset.from_str("1__")
    dataset_1.extend(dataset_2)
    assert dataset_1.data == Dataset.from_str("10_1").data
    assert dataset_1.max_input_bits == 2
    dataset_1 = Dataset.from_str("00_1")
    dataset_2 = Dataset.from_str("_____101")
    dataset_1.extend(dataset_2)
    assert dataset_1.data == Dataset.from_str("00_1_101").data
    assert dataset_1.max_input_bits == 3
    dataset_1 = Dataset.from_str("_____101")
    dataset_2 = Dataset.from_str("00_1")
    dataset_1.extend(dataset_2)
    assert dataset_1.data == Dataset.from_str("00_1_101").data
    assert dataset_1.max_input_bits == 3

    for _ in range(5):
        for bit_count in range(5, 10):
            dataset, instruction = Dataset.rand_true_xor(bit_count)
            assert instruction.__len__() == bit_count
            dataset.safety_check___sort_btw()
            # dataset, instruction = Dataset.rand_true_xor(bit_count, False)
            # assert instruction.__len__() == bit_count
            # dataset.safety_check___sort_btw()
            pass
        pass
    pass
    
if "test log_the_error" and False:
    dataset = Dataset(5, [(2,True), (3,True), (5,False), (8,True), ])
    dataset.log_the_error(comment="Feature test. Not real log.")
    pass
    
if "add_addr" and True:
    a_Dataset = Dataset(2, [(2,True),(1,True),(0,True),])
    #a_Dataset = Dataset(2, [(2,True),(1,True),(2,True),])
    
    a_Dataset = Dataset(3, [(0,True),(1,True),(2,True),])
    a_Dataset.add_addr(4,True)
    #a_Dataset.add_addr(4,True)
    a_Dataset.add_addr(4,True,False)
    #a_Dataset.safety_check___sort_btw()
    
    a_Dataset = Dataset(2, [(0,True),(1,True),(2,True),])
    assert a_Dataset.readable_str(add_title=False) == "00:1, 01:1, 10:1"
    assert a_Dataset.get_input_bits() == 2
    assert a_Dataset.get_output_bits() == 1
    a_Dataset.increase_input_bit_count(3)
    assert a_Dataset.get_input_bits() == 3
    assert a_Dataset.get_output_bits() == 1
    #a_Dataset.increase_input_bit_count(3)
    
    assert a_Dataset.find_addr(0) == (True, 0)
    assert a_Dataset.find_addr(1) == (True, 1)
    assert a_Dataset.find_addr(3)[0] == False
    #assert a_Dataset.find_addr(3) == (False, 2)
    assert a_Dataset.find_addr(33)[0] == False
    #assert a_Dataset.find_addr(33) == (False, 2)
    pass
    
if "find addr" and True:
    a_Dataset = Dataset(4, [(1,True), (3,True), (9,True), ])
    assert a_Dataset.find_addr(0)[0] == False
    assert a_Dataset.find_addr(2)[0] == False
    assert a_Dataset.find_addr(5)[0] == False
    assert a_Dataset.find_addr(1) == (True, 0)
    assert a_Dataset.find_addr(3) == (True, 1)
    assert a_Dataset.find_addr(9) == (True, 2)
    _temp_set = a_Dataset.get_addr_as_set()
    assert _temp_set == {1,3,9}
    pass
    
if "test" and True:
    ds2 = Dataset.rand__sorted(4, 0.1, 0.1, 123)
    print(ds2)
    ds3 = Dataset.rand__sorted(4, 0.1, 0.1, 123)
    print(ds3)
    ds4 = Dataset.rand__sorted(4, 0.2, 0. , 123)
    print(ds4)
    pass

if "test from_str" and True:
    a_Dataset_11 = Dataset.from_str("1______",True)
    assert a_Dataset_11.data           == [(0,True)]
    assert a_Dataset_11.max_input_bits == 0
    a_Dataset_11a = Dataset.from_str("1______",False)
    assert a_Dataset_11a.data           == [(0,True)]
    assert a_Dataset_11a.max_input_bits == 3
    
    a_Dataset_0 = Dataset.from_str("", increase_input_bit_count_to=0)
    a_Dataset_1 = Dataset.from_str("10011")
    a_Dataset_1a = Dataset.from_str("10 0,,11")
    assert a_Dataset_1.data == a_Dataset_1a.data
    assert a_Dataset_1.max_input_bits == a_Dataset_1a.max_input_bits
    a_Dataset_2 = Dataset.from_str("1001")
    a_Dataset_3 = Dataset.from_str("100")
    a_Dataset_4 = Dataset.from_str("101")
    a_Dataset_5 = Dataset.from_str("00000")
    a_Dataset_6 = Dataset.from_str("_", increase_input_bit_count_to = 0)
    assert a_Dataset_6.data == []
    assert a_Dataset_6.max_input_bits == 0
    a_Dataset_7 = Dataset.from_str("__", increase_input_bit_count_to=0)
    a_Dataset_8 = Dataset.from_str("1_")
    a_Dataset_8b = Dataset.from_str("_1")
    a_Dataset_9 = Dataset.from_str("1_01")
    a_Dataset_10 = Dataset.from_str("1_01_")
    a_Dataset_10a = Dataset.from_str("1_  ,01_")
    assert a_Dataset_10.data == a_Dataset_10a.data
    assert a_Dataset_10.max_input_bits == a_Dataset_10a.max_input_bits
    pass

if "test" and True:
    a_Dataset = Dataset.from_str("1001")
    assert 2 == a_Dataset.max_input_bits
    a_Dataset.add_const_bit_into_addr(1,True)
    assert 3 == a_Dataset.max_input_bits
    assert a_Dataset.data[0b00][0] == 0b010
    assert a_Dataset.data[0b01][0] == 0b011
    assert a_Dataset.data[0b10][0] == 0b110
    assert a_Dataset.data[0b11][0] == 0b111
    
    a_Dataset = Dataset.from_str("1001")
    assert 2 == a_Dataset.max_input_bits
    a_Dataset.add_const_bit_into_addr(1,False)
    assert 3 == a_Dataset.max_input_bits
    assert a_Dataset.data[0b00][0] == 0b000
    assert a_Dataset.data[0b01][0] == 0b001
    assert a_Dataset.data[0b10][0] == 0b100
    assert a_Dataset.data[0b11][0] == 0b101
    
    a_Dataset = Dataset.from_str("101")
    assert 2 == a_Dataset.max_input_bits
    a_Dataset.add_const_bit_into_addr(1,True)
    assert 3 == a_Dataset.max_input_bits
    assert a_Dataset.data[0b00][0] == 0b010
    assert a_Dataset.data[0b01][0] == 0b011
    assert a_Dataset.data[0b10][0] == 0b110

    a_Dataset = Dataset.from_str("10_1")
    assert 2 == a_Dataset.max_input_bits
    a_Dataset.add_const_bit_into_addr(1,True)
    assert 3 == a_Dataset.max_input_bits
    assert a_Dataset.data[0b00][0] == 0b010
    assert a_Dataset.data[0b01][0] == 0b011
    assert a_Dataset.data[0b10][0] == 0b111
    
    
    a_Dataset= Dataset(1, [(0,True)])
    a_Dataset.add_const_bit_into_addr(0,True)
    assert a_Dataset.max_input_bits == 2
    assert a_Dataset.data[0][0] == 0b01
    
    a_Dataset= Dataset(1, [(0,True)])
    a_Dataset.add_const_bit_into_addr(1,True)
    assert a_Dataset.max_input_bits == 2
    assert a_Dataset.data[0][0] == 0b10
    
    a_Dataset= Dataset(1, [(0,True)])
    a_Dataset.increase_input_bit_count(3)
    a_Dataset.add_const_bit_into_addr(3,True)
    assert a_Dataset.max_input_bits == 4
    assert a_Dataset.data[0][0] == 0b1000
    
    a_Dataset= Dataset(1, [(0,True)])
    a_Dataset.increase_input_bit_count(3)
    a_Dataset.increase_input_bit_count(5)
    a_Dataset.add_const_bit_into_addr(5,True)
    assert a_Dataset.max_input_bits == 6
    assert a_Dataset.data[0][0] == 0b100000
    
    #new_dataset = add_const_bit_into_addr(5,6,True, [(0,True)])
    
    a_Dataset = Dataset(6, [(0b11111,True)])
    assert a_Dataset.max_input_bits == 6
    a_Dataset.add_const_bit_into_addr(0,False)
    assert a_Dataset.max_input_bits == 7
    assert a_Dataset.data[0][0] == 0b111110
    a_Dataset= Dataset(6, [(0b11111,True)])
    a_Dataset.add_const_bit_into_addr(3,False)
    assert a_Dataset.data[0][0] == 0b110111
    pass

if "test" and True:
    a_Dataset = Dataset(2, [(0,True),(1,True),(2,True),(3,True),])
    dataset_2 = a_Dataset.get_subset(bitmask=0b10, addr=0b00)
    assert dataset_2.data == [(0,True),(1,True),]
    dataset_2 = a_Dataset.get_subset(bitmask=0b10, addr=0b10)
    assert dataset_2.data == [(2,True),(3,True),]
    #dataset_2 = a_Dataset.get_subset(dataset,bitmask=0b10, addr=0b11)
    dataset_2 = a_Dataset.get_subset(bitmask=0b1, addr=0b0)
    assert dataset_2.data == [(0,True),(2,True),]
    dataset_2 = a_Dataset.get_subset(bitmask=0b1, addr=0b1)
    assert dataset_2.data == [(1,True),(3,True),]
    dataset_2 = a_Dataset.get_subset(bitmask=0b11, addr=0b01)
    assert dataset_2.data == [(1,True),]
    dataset_2 = a_Dataset.get_subset(bitmask=0b11, addr=0b11)
    assert dataset_2.data == [(3,True),]
    pass

if "test slow" and False:
    #special cases.
    #empty
    a_dataset = Dataset(1)
    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
    irr_amount = irr_addr_list_tuple[1].__len__()
    assert not irr_addr_list_tuple[0]
    if irr_addr_list_tuple[0]:
        assert 2 == irr_amount
    
    #full
    a_dataset = Dataset(1, [(0,True), (1,True), ])
    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
    irr_amount = irr_addr_list_tuple[1].__len__()
    assert irr_addr_list_tuple[0]
    assert 0 == irr_amount
    
    #the last one is irrelevant.
    a_dataset = Dataset(1, [(0,True), ])
    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
    irr_amount = irr_addr_list_tuple[1].__len__()
    assert irr_addr_list_tuple[0]
    assert 1 == irr_amount
    
    a_dataset = Dataset(2, [(0,True), ])
    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
    irr_amount = irr_addr_list_tuple[1].__len__()
    assert irr_addr_list_tuple[0]
    assert 3 == irr_amount
    
    if "random cases" and True:
        #empty 
        for max_input_bits in range(1,5):
            a_dataset = Dataset(max_input_bits)
            irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
            assert irr_addr_list_tuple[0] == False
            pass
        
        #full
        for max_input_bits in range(1,5):
            for _ in range(max_input_bits*5):
                a_dataset = Dataset.rand__sorted(max_input_bits,0.5,0.501)
                irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
                irr_amount = irr_addr_list_tuple[1].__len__()
                assert irr_addr_list_tuple[0]
                assert 0 == irr_amount
                pass
            pass
        
        for _test_iter in range(333):
            print(_test_iter, end=", ")
            #partly
            for max_input_bits in range(1,15):
                for _ in range(max_input_bits*25):
                    a_dataset = Dataset.rand__sorted(max_input_bits,0.2,0.2)
                    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
                    irr_amount = irr_addr_list_tuple[1].__len__()
                    if irr_addr_list_tuple[0]:
                        assert a_dataset.data.__len__() + irr_amount == (1<<max_input_bits)
                        pass
                    pass
                pass
        
        #one side.
        for max_input_bits in range(1,15):
            for _ in range(max_input_bits*25):
                a_dataset = Dataset.rand__sorted(max_input_bits,0.2,0)
                irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
                irr_amount = irr_addr_list_tuple[1].__len__()
                if irr_addr_list_tuple[0]:
                    assert a_dataset.data.__len__() + irr_amount == (1<<max_input_bits)
                    pass
                pass
            pass
        for max_input_bits in range(1,15):
            for _ in range(max_input_bits*25):
                a_dataset = Dataset.rand__sorted(max_input_bits,0,0.2)
                irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self()
                irr_amount = irr_addr_list_tuple[1].__len__()
                if irr_addr_list_tuple[0]:
                    assert a_dataset.data.__len__() + irr_amount == (1<<max_input_bits)
                    pass
                pass
            pass
        
        pass
    pass



'''DatasetField class:
First, the entire tool does a reverse way to use Karnaugh map. In the normal
way, it starts with the 1x2 circles, and if these 1x2 also combines, it's 1x4
or 2x2. The normal way of simplify a logic boolean algebra expression is from
small circles to big ones. But in this tool, it's the reverse way. The entire 
Karnaugh map(I don't mean you need to print it visibly) is cut into 2 halves, 
based on the correlation of the bit and the Y. Let's say a input is x1 to xn and the 
result of the boolean algebra expression is Y(let's consider only 1 output first),
if x1 and Y are the most likely to be the same(or different), then, the map
will be split by x1. This makes the 2 subfields most likely to have more correlation 
then the entire map(not mathmatically proved yet, but true in some examples I checked.).
Then, repeat this method to get a binary-tree-like structure.
I call the Karnaugh map a dataset in this tool(self.dataset). Only relevant items
are stored. Irrelevant items don't show up in it.

Main jobs of this class.
1, detect a raw dataset.
2, split a sorted dataset and get 2 subfield(if necessary. and the 2 subfields will be 
easier to build.)
3, look up(input the value of x1 to xn, and get the result.)

4, (no plan) dynamically update. It will be a minimum update.
5, (no plan) formula based simplification. Maybe it not gonna work? 
6, (no plan) integer, float point number(ieee754 like) support with acceptable error.


How to detect a raw dataset.
In the __init__ function, if the dataset comes without any specifies(the first object 
of this class, and the dataset is from other source. The class must be able to handle this).
The init func must be able to detect useful info from the dataset, and get it ready for 
future use.
Steps are:
1, is dataset empty. If true, it's a leaf node. [return]
2, is dataset full. If true, it has no irrelevant items.
3, detect the dataset, and get has_1, has_0, and 
(best_index_to_split, best_abs_of_num_of_same).
4, a special case is ***all xor***. Condition is ((not self.has_irr) and 
0 == best_abs_of_num_of_same). If true, 
[[[update]]]
if true, it's a true xor or fake xor.
a true xor is a true xor. The formula is (a@b)@c (where @ means xor) and d,e,f,g. Then check out the flag to 
decide if it needs a not before giving out the result.
But a fake xor is a bit tricky. Let's say, 0110 is xor, 1001 is xnor, but in this code, 
1001 is a xor but needs a not before giving out the result. 0110 with a 1001, they combine
into a 3 bits xor. The formula is (a@b)@c. But 2 0110s don't combine into a 3 bits xor. 
The formula is a@b, no c at all. 2 1001s also do this. This is the fake xor case.
In fake xor case, the method is to figure out which bit is not a xor bit, and split at it.
This splitting gives out 2 half and they are still (either true or fake) xor.
By doing this, it will eventually get 2 bits xor or more-bits-xor, and they are all true xor.
(2 bits xor is always true xor. Fake xor is at least 3 bits.)
Then, a true xor is a leaf node, while a fake xor is a branch.
Both path [return].
5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
6, it's not a leaf node, split it[split][return]

The split function not only creats the dataset for subfield(children if you deem it
as a binary tree), it also creats the suggestions for them. 
Suggestions include:
(is_dataset_empty too easy, not included here.)
is_dataset_full.
has_1, has_0
Also, the dataset is also sorted by *addr*

If an object of this class is created with suggestion, the init func is easier.
But it still has to do the rest of the job.


Lookup:
To lookup in this tool, provide a set of the values of each input. They can be
1, 0 or irrelevant(irr for short)
[WIP] A lookup requist comes with some suggestions, like: allow_irr, better_non_irr, 
better_irr, true_when_irr, false_when_irr.
The lookup starts from the root node. When a node cannot decide the result, it 
calls 1 of its 2 subfield(children) to do the job. This should only happen 
when the node is non-leaf.
The sequence for leaf node: 1 all_irr, 2 all_xor, 3 normal leaf(no 0 or no 1).


Split:
To create subfields(children), iterates its own dataset, paste(or move???) it's own
items into 2 new datasets. While doing this, all the useful flags are created. Then
datasets and suggestions(flags) are sent to new objects, so new objects' init function
is easier.


(no plan) dynamically update:
If any relevant item is removed, theoritically, the tree can stay while provides 
correct result. But if any item is modified(from true to false or viseversa), or added
, it's not guarunteed to provide correct result anymore. This happens in AI. When 
agent adapts to environment, it happens.
A minimum update method is to keep all the split relationship, but to split more from 
the existing leaf nodes. If any leaf node, let's say, it has no 0(false), and gets a 0,
then it needs to split more.
A not minimum but probably better way is to check correlation value. Because the splitting
happens at the most correlative bit, if it's not the best place to split, then the entire 
subtree must all be recreated.
The first way fits the *shortest updating time* demand, while the second fits *shortest 
inference time* demand.

(no plan) formula based simplification. Maybe it not gonna work? 
I'm not sure if this is needed. So yeah, if I progress, I'll write it here.


(no plan) integer, float point number(ieee754 like) support with acceptable error.
This will be implemented after the other parts finish.
The idea is to allow some error, because these are numbers. If the most/more 
significant bits are correct, then the least/less significant bits are much less 
important. If we allow some error, the tree maybe smaller.
In my another (probably failed) approach (digital neural network), I also have similar 
design. Yeah, it's natually to come up with this idea.
A possible way to do this is a progressive way. If a number is 8 bits, and all bits 
matters(it can be 128 to 255, if unsigned.), then it's possible to only consider the 
first 2 or 3 bits in the first round of training(let me use this word). Then, based 
on the reulst, add more bits of these numbers, and do a progressive training. To a 
point, the error is small enough, and the rest of the bits are not needed anymore. 
Or at least, the training progress maybe easier, and if anything goes wrong, it's 
possible to detect it earlier and cheaper.

Optimizable:
If only the leaf nodes need dataset, them it's possible to reuse the original one. 
But let's keep it simple at the moment.
'''
class How_did_I_quit_init_func(Enum):
    IRR = 0,
    XOR = 1,
    LEAF = 2,
    BRANCH = 3,
    BRANCH__FAKE_XOR = 4,
    pass

class DatasetField:
    '''Some details:
    
    Set leaf_keep_dataset to true, in order to get accurate irrelevant lookup at all cases. Otherwise, 
    only all-irr field provides accurate irrelevant lookup.
    And, all the result from 1+ir or 0+ir field are not for-sure-relevant. 
    This might not be a problem, since this tool is only designed to provide suggestion.
    '''
        
    if "class var" and True:
        #core info
        bitmask:int
        addr:int
        input_bits_count:int
        bits_already_in_use:int
        dataset:Optional[Dataset]
        _debug__how_did_I_quit_init_func:How_did_I_quit_init_func
        #lookup tables.
        best_index_to_split_from_right_side:int
        _it_was_a_temp_var__best_abs_of_num_of_same:int
        children:Optional[tuple['DatasetField','DatasetField']]#yeah, Im a tree.
        ready_for_lookup:bool
        #irrelevant_lookup:list[]??? to do 
        
        #some flags
        has_1:bool
        has_0:bool
        has_irr:bool
        is_dataset_sorted:bool
        #is_leaf_node:bool
        all_xor:bool
        #simple flags
        not_after_xor:bool# self.not_after_xor # if all 0s, and not_after_xor is false, result is false. Every one flips it once.
        when_xor__ignore_these_bits:int
        #already_const_without_irr:bool
        all_irr:bool
        pass
    @staticmethod
    def todo_list()->str:
        return '''  
        non-full xor field detection.
        lookup 里面加一个具体的依据来源的地址的返回。
        '''
    
    @staticmethod
    def _new(dataset:Dataset, leaf_keep_dataset:bool = False, _debug__save_sub_dataset_when_xor = False)->'DatasetField':
        '''If you want to keep accurate irrelevant item info, set leaf_keep_dataset=True.
        Otherwise, only irr field is reported as irr. The 1+irr field or 0+irr field is reported 
        as 1 and 0 respectively.'''
        result = DatasetField(0,0,0,dataset, leaf_keep_dataset = leaf_keep_dataset, \
            _debug__save_sub_dataset_when_xor = _debug__save_sub_dataset_when_xor)
        return result
    
    @staticmethod
    def _new__and_valid(dataset:Dataset, leaf_keep_dataset:bool = False, _debug__save_sub_dataset_when_xor = False)->'DatasetField':
        '''If you want to keep accurate irrelevant item info, set leaf_keep_dataset=True.
        Otherwise, only irr field is reported as irr. The 1+irr field or 0+irr field is reported 
        as 1 and 0 respectively.'''
        result = DatasetField(0,0,0,dataset, leaf_keep_dataset = leaf_keep_dataset, \
            _debug__save_sub_dataset_when_xor = _debug__save_sub_dataset_when_xor)
        if leaf_keep_dataset:
            result.valid(dataset, accurate_lookup = True)
            result.valid_irr(dataset, total_amount_irr = 100)
            pass
        else:
            result.valid(dataset, accurate_lookup = False)
            pass
        return result
    
    def __init__(self, bitmask:int, addr:int, #input_bits_count:int, 
                bits_already_in_use:int, dataset:Dataset, 
                
                with_suggest:bool=False, 
                suggest_has_1:bool=False, 
                suggest_has_0:bool=False, 
                
                _debug__check_all_safety:bool = True,
                _debug__save_sub_dataset_when_xor = False,
                
                leaf_keep_dataset:bool = False,
                branch_keep_dataset:bool = False,
                ):
        if __debug__:
            if branch_keep_dataset and (not leaf_keep_dataset):
                raise Exception("I didn't prepare for this combination. Maybe you want to modify the code. Remember to also modify the lookup function.")
            pass#if __debug__:
            if 0 == bitmask and dataset.data.__len__() == 0:
                raise Exception("Root node doesn't accept empty dataset.")
        
        #core info. simple copy paste.
        self.bitmask = bitmask
        self.addr = addr
        self.input_bits_count = dataset.max_input_bits
        self.bits_already_in_use = bits_already_in_use
        self.dataset = dataset
        #self.leaf_keep_dataset = leaf_keep_dataset
        #self.branch_keep_dataset = branch_keep_dataset#not needed.
        #lookup tables
        self.children = None
        
        #safety check
        if _debug__check_all_safety:
            self._init_only__check_all_addr()
            pass
        
        # only_irr 1, is dataset empty. If true, it's a leaf node. [return]
        if dataset.data.__len__() == 0:
            self.ready_for_lookup = True
            self.has_1 = False
            self.has_0 = False
            self.has_irr = True
            self.all_irr = True
            self.is_dataset_sorted = True#only to init. not a real result.
            #self.is_leaf_node = True
            self.all_xor = False
            self.not_after_xor = False#only to init. not a real result.
            self.when_xor__ignore_these_bits = 0
            #self.already_const_without_irr = True#only to init. not a real result.
            self.best_index_to_split_from_right_side = -1
            self._debug__how_did_I_quit_init_func = How_did_I_quit_init_func.IRR
            
            #all_irr has no dataset.
            self.dataset = None
            return
        
        # 2, is dataset full. If true, it has no irrelevant items.
        num_of_irr = self._init_only__get_num_of_irrelevant()
        self.has_irr = 0 != num_of_irr
        
        # 3, detect the dataset, and get has_1, has_0, and 
        # (best_index_to_split, best_abs_of_num_of_same).
        if with_suggest:
            self.has_1 = suggest_has_1
            self.has_0 = suggest_has_0
            self.is_dataset_sorted = True#this maybe a bit dangerous?
            pass
        else:
            self._init_only__sort_dataset()
            #self.is_dataset_sorted = True#keep it a comment, only for check.
            found_true = False
            found_false = False
            for item in dataset.data:
                if item[1]:
                    found_true = True
                    pass
                else:
                    found_false = True
                    pass
                if found_true and found_false:
                    break
                pass
            self.has_1 = found_true
            self.has_0 = found_false
            pass
        self.best_index_to_split_from_right_side, self._it_was_a_temp_var__best_abs_of_num_of_same = self._detect_best_bit_to_split()
        
        # 4, a special case is ***all xor***. Condition is ((not self.has_irr) and 
        # 0 == best_abs_of_num_of_same and has_0 and has_1). If true, it's a leaf node. [return]
        if (0 == self._it_was_a_temp_var__best_abs_of_num_of_same) and (not self.has_irr) and self.has_0 and self.has_1:
            '''Generally, xor is a 2 bits structure. The result is either 1001 or 0110. 
            People call the 1001 xnor, but in this tool, they are both xor.
            In this tool, more than 2 bits can also be treated as xor, to simplify and speed up.
            When input(a) is true, input(bc) is 1001, when input(a) is false, input(bc) is 0110, 
            they can combine as a 3 bits xor. But when 2 1001 combine, it's not a xor. 
            This 3 bits fake xor is to validate this special case.        
            '''
            
            #self.all_xor, self.not_after_xor, self.best_index_to_split_from_right_side = \
            is_true_xor, irr_bitmask, needs_a_not_after_xor, best_index_to_split_from_right_side = \
                self._init_only__detect_xor_info(_debug__save_sub_dataset_when_xor)
                
            '''
            This function return in 2 styles. (- means not related in a given case.)

            >>> 1, (true xor)pure xor/xnor, with and without irr-bit. It's (true,  useful, useful, -).
            >>> 2, fake xor(false, -, -, useful.)

            1 is leaf, while 2 is branch.
            '''
            self.ready_for_lookup = True
            #self.has_1 see uppon
            #self.has_0 see uppon
            #self.has_irr see uppon
            self.all_irr = False
            #self.is_dataset_sorted = True already set before
            self.all_xor = is_true_xor#it depends
            self.not_after_xor = needs_a_not_after_xor
            self.when_xor__ignore_these_bits = irr_bitmask#for 1. in style 2, it's ignored.
            #self.already_const_without_irr = False#only to init. not a real result.
            self.best_index_to_split_from_right_side = best_index_to_split_from_right_side
            if is_true_xor:#case 1
                assert 0 == irr_bitmask&self.bitmask
                
                '''true xor. both pure/unpure xor(with or without irr-bit). leaf. [return]'''
                #self.is_leaf_node = True
                self._debug__how_did_I_quit_init_func = How_did_I_quit_init_func.XOR
                
                #all_xor doesn't need the dataset.
                self.dataset = None
                return 
            else:#case 2
                #it's not a true xor. But splitting it trickily can get smaller true xor.
                #self.is_leaf_node = False
                self._debug__how_did_I_quit_init_func = How_did_I_quit_init_func.BRANCH__FAKE_XOR
                #now split.
                _check_all_safety = _debug__check_all_safety
                self.split(self.best_index_to_split_from_right_side, _debug__check_all_safety_in_split = _check_all_safety,
                        leaf_keep_dataset = leaf_keep_dataset, branch_keep_dataset = branch_keep_dataset)
                #
                if not branch_keep_dataset:
                    self.dataset = None
                    pass
                #assert False, "untested."
                return
        
        # normal leaf. 5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
        if (not self.has_1) or (not self.has_0):
            #at least 1 true. If both false, it's a all_irr case, which returned long ago.
            self.ready_for_lookup = True
            #self.has_1 see uppon
            #self.has_0 see uppon
            #self.has_irr see uppon
            self.all_irr = False
            #self.is_dataset_sorted = True already set before
            #self.is_leaf_node = True
            self.all_xor = False
            self.not_after_xor = False#only to init. not a real result.
            self.when_xor__ignore_these_bits = 0
            #self.already_const_without_irr = not self.has_irr
            self.best_index_to_split_from_right_side = -1
            self._debug__how_did_I_quit_init_func = How_did_I_quit_init_func.LEAF
            if not leaf_keep_dataset:
                self.dataset = None
                pass
            return 
        
        # branch, to split. 6, it's not a leaf node, split it[split][return]
        self.ready_for_lookup = True
        #self.has_1 see uppon
        #self.has_0 see uppon
        #self.has_irr see uppon
        self.all_irr = False
        #self.is_dataset_sorted = True already set before
        #self.is_leaf_node = False
        self.all_xor = False
        self.not_after_xor = False#only to init. not a real result.
        self.when_xor__ignore_these_bits = 0
        #self.already_const_without_irr = False#only to init. not a real result.
        self._debug__how_did_I_quit_init_func = How_did_I_quit_init_func.BRANCH
        
        _check_all_safety = _debug__check_all_safety
        self.split(self.best_index_to_split_from_right_side, _debug__check_all_safety_in_split = _check_all_safety,
                leaf_keep_dataset = leaf_keep_dataset, branch_keep_dataset = branch_keep_dataset)#this line was on top.
        if not branch_keep_dataset:
            self.dataset = None
            pass
        return #end of function
    
    def get_is_leaf(self)->bool:
        return self.children is None
    
    def has_dataset(self)->bool:
        return self.dataset is not None
    
    def _init_only__detect_xor_info(self, __debug__save_sub_dataset_when_xor = False)->tuple[bool, int, bool, int]:
        '''return(is full xor, irr_bitmask, needs a not after xor(if is full xor), best bit to split from_right_side(if not a full xor))
        
        This function return in 2 styles. (- means not related in a given case.)
        >>> 1, (true xor)pure xor/xnor, with and without irr-bit. It's (true,  useful, useful, -).
        >>> 2, fake xor(false, -, -, useful.)
        
        1 is leaf, while 2 is branch.
        
        irr_bitmask:when the bits left to detect are only true-xor bits, or irrelevant bits, all the irr-bits 
        are reported in this bitmask. It's the irr-bit, not the irr-item. irr-item the digital-circuit-concept 
        and you can find it anywhere. But the irr-bit is a concept introduced here by me.
        
        位的优先级。
        非xor有关位>非xor无关位>xor位。
        其中，非xor有关位可以形成branch。
        非xor无关位 和 xor位 共同构成xor leaf。

        在xor检测的时候，非xor有关位的xor分数是大于0，而且小于最大可能分数，非xor无关位的分数为0，xor位的分数是最大可能分数。
        于是当有非xor有关位的时候，按其中的最无关位split（未证明最优）。
        也就是说，有 非xor有关位 的时候，是style2，没有就是style1。
        '''
        assert self.dataset is not None
        if __debug__save_sub_dataset_when_xor:
            self.dataset.log_the_error("xor cases.txt")
            pass
        
        '''this loop is cut into 2 parts. The code is a lil different.'''
        score_of_xor_list_from_right_side:list[int] = []
        score_of_xor = 0
        assert self.dataset is not None
        #part 1
        for index_1 in range(0, self.dataset.data.__len__(), 2):
            '''case 1, last bit.
            0and1, 2and3, ind2-ind1 is 1. step is 1??, chunk is 2.'''
            index_2 = index_1+1
            data_1_tuple:tuple[int, bool] = self.dataset.data[index_1]
            data_2_tuple:tuple[int, bool] = self.dataset.data[index_2]
            #wrong assert data_1_tuple[0]^data_2_tuple[0] == 1 #not needed in real case. 
            if data_1_tuple[1]^data_2_tuple[1]:
                score_of_xor = score_of_xor +1
                pass
            pass#end of part 1
        score_of_xor_list_from_right_side.append(score_of_xor)
        #part 2
        for i in range(1, self.input_bits_count-self.bits_already_in_use):
            '''
            case 2, 
            0and2, 1and3,,, 4and6, 5and7. ind2-ind1 is 2. step is 1, chunk is 4.
            0-4, 1-5, 2-6, 3-7,,, 8-12, 9-13, 10-14, 11-15. ind2-ind1 is 4. step is 1, chunk is 8.
            '''
            score_of_xor = 0
            index_2_minus_index_1 = 1<<i #this is also half chunk size.
            chunk_size = index_2_minus_index_1<<1
            for chunk_start_addr in range(0, self.dataset.data.__len__(), chunk_size):
                chunk_start_addr_plus_half_chunk_size = chunk_start_addr+index_2_minus_index_1#
                for index_1 in range(chunk_start_addr, chunk_start_addr_plus_half_chunk_size):
                    index_2 = index_1 + index_2_minus_index_1
                    data_1_tuple = self.dataset.data[index_1]
                    data_2_tuple = self.dataset.data[index_2]
                    #wrong assert data_1_tuple[0]^data_2_tuple[0] == index_2_minus_index_1 #not needed in real case. 
                    if data_1_tuple[1]^data_2_tuple[1]:
                        score_of_xor = score_of_xor +1
                        pass
                    pass#for index_1
                pass#for chunk_start_addr
            score_of_xor_list_from_right_side.append(score_of_xor)
            pass#end of part 2.
        
        '''score-list into useful info.'''
        CONST_score_if_full_xor = self.dataset.data.__len__()//2#11111111111111111下一个版本里面这个要改
        irr_bit_as_int__from_right_side__squeezed__before_translate = 0
        # min_of_score_list____from_right_side = self.dataset.data.__len__()# the max of this is len/2, so this is big enough.
        # index_of_min_of_score_list____from_right_side = -1
        min_of___non_zero_score___from_right_side = CONST_score_if_full_xor
        index_of_min_of___non_zero_score___from_right_side = -1
        #for i in range(score_of_xor_list_from_right_side.__len__()):
        for i_from_right_side in range(score_of_xor_list_from_right_side.__len__()-1,-1,-1):
            item = score_of_xor_list_from_right_side[i_from_right_side]
            assert item >=0
            '''
            match item:
            case 0: it's a irr-bit.
            case CONST_score_if_full_xor: means 
            case in between: a fake xor.
            '''
            if 0 == item:
                irr_bit_as_int__from_right_side__squeezed__before_translate = \
                    irr_bit_as_int__from_right_side__squeezed__before_translate |(1<<i_from_right_side)
                pass
            # if item<min_of_score_list____from_right_side:
            #     min_of_score_list____from_right_side = item
            #     index_of_min_of_score_list____from_right_side = i
            if (item<min_of___non_zero_score___from_right_side) and (item!=0):
                min_of___non_zero_score___from_right_side = item
                index_of_min_of___non_zero_score___from_right_side = i_from_right_side
                pass
            pass
        
        '''
        if index_of_min_of___non_zero_score___from_right_side is -1, it's untouched. 
        which means no score between 0 and max was found. It's a true xor(may have irr-bit).
        
        min_of___non_zero_score___from_right_side can only be the max or between.
        When it's max, it's a true xor. Otherwise it needs to split.
        
        They give basically the same info.
        '''
        
        '''now all the useful info are ready. Return respectively.'''
        if -1 == index_of_min_of___non_zero_score___from_right_side:
            '''true xor.'''
            
            #maybe optimizable. 11111111111111111
            if 0 == irr_bit_as_int__from_right_side__squeezed__before_translate:
                '''true xor, withOUT irr bits.'''
                '''return(is full xor, irr_bitmask, needs a not after xor(if is full xor), best bit to split from_right_side(if not a full xor))'''
                needs_not_after_xor = self.dataset.data[0][1]#result of the pure not addr.
                '''1, (true xor)pure xor/xnor. It's (true, useful, useful, -). copy pasted from the docstring of function.'''
                return (True, 0, needs_not_after_xor, -2)
                '''-------------------------------------------------------'''
                '''-------------------------------------------------------'''
            else:
                '''true xor with "irr bit". '''
                '''translate the index to actual index.'''
                irr_bit_as_int = 0
                squeezed_bit_count__from_right_side = 0
                for actual_bit_index___from_right_side in range(self.input_bits_count):
                    one_shift_by_i:int = 1<<actual_bit_index___from_right_side
                    bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
                    if bit_of_bitmask_for_this_i != 0:
                        #This bit is in bitmask, ignore this i.
                        continue
                    '''
                    .4      actual index
                    ._ 2    squeezed index
                    001100  bitmask of the obj
                    .1  00  squeezed
                    .....1  the_bit
                    .10000  in actual place.    
                    '''
                    
                    one_shift_by___something___ = 1<<squeezed_bit_count__from_right_side
                    
                    
                    #the_bit__before_shift = irr_bit_as_int__from_right_side__squeezed__before_translate&squeezed_bit_count__from_right_side
                    the_bit__before_shift = irr_bit_as_int__from_right_side__squeezed__before_translate&one_shift_by___something___
                    #the_bit = the_bit__before_shift >>
                    the_bit_in_actual_place = the_bit__before_shift <<(actual_bit_index___from_right_side-squeezed_bit_count__from_right_side)
                    irr_bit_as_int = irr_bit_as_int |the_bit_in_actual_place
                    #tail
                    squeezed_bit_count__from_right_side = squeezed_bit_count__from_right_side +1
                    pass
                #assert False,"the line below."
                needs_not_after_xor = self.dataset.data[0][1]#result of the pure not addr.
                '''2, (true xor)xor/xnor with irr-bits(true, useful, useful, -) copy pasted from the docstring of function.'''
                return (True, irr_bit_as_int, needs_not_after_xor, -3)
                '''return(is full xor, irr_bitmask, needs a not after xor(if is full xor), best bit to split from_right_side(if not a full xor))'''
                pass#if   true xor without irr bit   else   xor with irr bit 
            pass#if true xor
        
        #no need for a else.
        '''fake xor. Split at the least xor bit.'''
        '''translate the index to actual index.'''
        squeezed_index = 0
        for actual_bit_index___from_right_side in range(self.input_bits_count):
            one_shift_by_i = 1<<actual_bit_index___from_right_side
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #This bit is in bitmask, ignore this i.
                continue
            #if squeezed_index == index_of_min_of_score_list____from_right_side:#old.
            if squeezed_index == index_of_min_of___non_zero_score___from_right_side:#new. irr-bit excluded here.
                '''return(is full xor, irr_bitmask, needs a not after xor(if is full xor), best bit to split from_right_side(if not a full xor))'''
                '''2, fake xor(false, -1, -, useful.) copy pasted from the docstring of function.'''
                return(False, -1, False, actual_bit_index___from_right_side)
            #tail
            squeezed_index = squeezed_index + 1
            pass
        assert False, "unreachable"

    if "untested but probably wrong. actually the version 2." and False:
        def _init_only__detect_xor_info(self)->tuple[bool, bool, int]:
            '''return(is full xor, needs a not after xor(if is full xor), best bit to split(if not a full xor))'''
            '''
            The case is, 1001 0000 0000 1001 is wrongly detected as true xor in the old O(log(n)) version.
            I don't have any idea how to make it O(log(n)) anymore, so I decide to do the old good O(n) version.
            new notes uppon
            old notes below.
            assert self.all_xor, "non-all-xor case can not call this function."
            pick anything from self.dataset and detect if all input are Falses(simpler than true), is the output the xor result, or the reversed.
            actually, the dataset should already be sorted, but the performance is probably similar and trivial?
            So, yeah, I don't care, I detect the addr here, instead of check the "sorted" and do the trick.
            maybe it's optimizable. But not for now.
            '''
                
            for i in range(self.input_bits_count-self.bits_already_in_use):
                dist_between_1_and_2 = 1<<i
                dist_to_next = 2<<i
                for index_1 in range(0, self.dataset.__len__(), dist_to_next):
                    index_2:int = index_1 + dist_between_1_and_2
                    data_1_tuple:tuple[int, bool] = self.dataset[index_1]
                    data_2_tuple:tuple[int, bool] = self.dataset[index_2]
                    if __debug__:
                        temp_to_check = data_1_tuple[0] ^ data_2_tuple[0]
                        assert temp_to_check&self.bitmask == 0
                        num_of_ones = 0
                        while temp_to_check>0:
                            if temp_to_check&1:
                                num_of_ones = num_of_ones + 1
                                pass
                            temp_to_check>>1
                        assert 1 == num_of_ones
                        pass
                    if not(data_1_tuple[1] ^ data_2_tuple[1]):
                        split_at_this_bit = data_1_tuple[0] ^ data_2_tuple[0]
                        index_of_this_bit_from_right_side = 0
                        while split_at_this_bit>0:
                            if split_at_this_bit&1 == 1:
                                return (False, False, index_of_this_bit_from_right_side)
                            #tail of iter.
                            index_of_this_bit_from_right_side =index_of_this_bit_from_right_side +1
                            split_at_this_bit = split_at_this_bit >> 1
                            pass#while split_at_this_bit>0:
                        assert False, "unreachable."
                            
    if "the old wrong O(log(n)) version." and False:
        def _init_only__detect_xor_info(self)->tuple[bool, bool, int]:
            '''return(is full xor, needs a not after xor(if is full xor), best bit to split(if not a full xor))'''
            #assert self.all_xor, "non-all-xor case can not call this function."
            #pick anything from self.dataset and detect if all input are Falses(simpler than true), is the output the xor result, or the reversed.
            #actually, the dataset should already be sorted, but the performance is probably similar and trivial?
            # So, yeah, I don't care, I detect the addr here, instead of check the "sorted" and do the trick.
            #maybe it's optimizable. But not for now.
            #unused addr_of_first_item = self.dataset[0][0]
            result_of_first_item = self.dataset[0][1]
            temp_dataset_len_left = self.dataset.__len__()
            data_1_tuple:tuple[int, bool] = self.dataset[0]
            data_2_tuple:tuple[int, bool]
            #is_xor = True
            #skip_first_iteration = True
            for i in range(self.input_bits_count-1,-1,-1):
                one_shift_by_i:int = 1<<i
                bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
                if bit_of_bitmask_for_this_i != 0:
                    #This bit is in bitmask, ignore this i.
                    continue
                # if skip_first_iteration:
                #     skip_first_iteration = False
                #     continue
                
                temp_dataset_len_left = temp_dataset_len_left>>1
                index_for_data_2 = self.dataset.__len__()- temp_dataset_len_left
                data_2_tuple = self.dataset[index_for_data_2]
                
                data_1_addr_with_bit_as_true_at_this_i = data_1_tuple[0] | one_shift_by_i
                assert data_2_tuple[0] == data_1_addr_with_bit_as_true_at_this_i
                
                temp_bool:bool = data_1_tuple[1]^data_2_tuple[1]
                if not temp_bool:
                    return (False, False, i)
                #for the next iteration.
                data_1_tuple = data_2_tuple
                if "collaps" and False:
                
                #bit_in_addr_for_this_i = addr_of_first_item & one_shift_by_i
                #assert 0 == bit_in_addr_for_this_i
                #flip the bit, and check if the result is also flipped.
                #temp_addr = temp_addr|one_shift_by_i
                #index = self.dataset.__len__() - temp_dataset_len_left
                #assert self.dataset[index][0] == temp_addr
                #temp_dataset_len_left = temp_dataset_len_left>>1
                
                
                # if bit_in_addr_for_this_i != 0:
                #     num_of_ones_in_addr = num_of_ones_in_addr + 1
                #     pass
                    pass
                pass
            # self.not_after_xor # if all 0s, and not_after_xor is false, result is false. Every one flips it once.
            return (True, result_of_first_item, -2)
        pass
    
    def _init_only__check_all_addr(self)->None:
        #part 1, bits_already_in_use must equal to 1s in bitmask
        temp_bitmask = self.bitmask
        ones_in_bitmask = 0
        while temp_bitmask!=0:
            if (temp_bitmask&1)!=0:
                ones_in_bitmask = ones_in_bitmask + 1
                pass
            temp_bitmask = temp_bitmask>>1
        assert self.bits_already_in_use == ones_in_bitmask
        #part 2, addr bits outside the bitmask must be 0s.
        reversed_bitmask = ~self.bitmask
        addr_bits_out_of_bitmask = self.addr&reversed_bitmask
        assert 0 == addr_bits_out_of_bitmask, "self.addr is bad. It has bit set to 1 outside the bitmask."
        #part 3, all addr bits under the bitmask must equal.
        masked_addr = self.addr&self.bitmask
        assert self.dataset is not None
        for item in self.dataset.data:
            masked_addr_of_item = item[0]&self.bitmask
            assert masked_addr == masked_addr_of_item
            pass
        pass
    
    def _detect_best_bit_to_split(self)->tuple[int,int]:
        '''return (best_index_to_split, best_abs_of_num_of_same)'''
        actual_index:list[int] = []
        num_of_same:list[int] = []
        for i in range(self.input_bits_count-1,-1,-1):
            one_shift_by_i:int = 1<<i
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #This bit is in bitmask, ignore this i.
                continue
            actual_index.append(i)
            
            dot_product_like = 0
            #Y is same as this bit. If a and result is true, +1. If a' and result is false, +1. Otherwise +0.
            #bit_of_addr_for_this_i = self.addr&one_shift_by_i#always 0????
            assert self.dataset is not None
            for item in self.dataset.data:
                #total = total+1
                bit_of_addr_of_item_raw = item[0] & one_shift_by_i
                bit_of_addr_of_item = bit_of_addr_of_item_raw!=0
                #if not (bit_of_addr_of_item ^ item[1]):
                logic_var_xor_this_result:bool = bit_of_addr_of_item ^ item[1]
                same_minus_this = 1-2*int(logic_var_xor_this_result)
                dot_product_like = dot_product_like + same_minus_this
                pass
            num_of_same.append(dot_product_like)
            pass#for i in range
        
        '''self.has_irr is already set before calling this function.
        All_xor case has no irrelevant.
        if this case has irr, it's not a all_xor, then the split target can NOT be -1.
        While, all scores can be 0. So, if this case has any irr, pick any index for it.
        
        The other way(unchecked) to do this is, when check the argmax, set the start score to -1.
        Because the score is >=0, the first iter will always set a new max info. 
        But in this case, the output index will NEVER be -1.
        In this version, I use this -1 to indicate that this case is a xor(maybe fake xor).
        Maybe I'll change it later.
        '''
        best_index_to_split_from_right_side:int=-1
        if self.has_irr:
            best_index_to_split_from_right_side = actual_index[0]
            pass
        
        best_abs_of_num_of_same:int = 0
        for i in range(actual_index.__len__()):
            index = actual_index[i]
            abs_of_same = abs(num_of_same[i])
            if abs_of_same>best_abs_of_num_of_same:
                best_abs_of_num_of_same = abs_of_same
                best_index_to_split_from_right_side = index
                pass
            pass
        return (best_index_to_split_from_right_side, best_abs_of_num_of_same)
        pass#end of function.
    
    def _init_only__get_num_of_irrelevant(self)->int:
        '''Because only relevant items are stored in "dataset", 
            and the total possible number is 1<<N.'''
            
        assert self.dataset is not None
        length = self.dataset.data.__len__()
        total_possible = 1<<(self.input_bits_count-self.bits_already_in_use)
        result = total_possible-length
        return result
    def _init_only__sort_dataset(self):
        self.is_dataset_sorted = True
        self.dataset.sort()#key=lambda item:item[0])
        pass
        
    def split(self, bit_index:int, _debug__check_all_safety_in_split: bool = False,
            leaf_keep_dataset:bool = False, branch_keep_dataset:bool = False):
        #assert False, "untested"
        #safety ckeck
        split_at_this_bit:int = 1<<bit_index
        if _debug__check_all_safety_in_split:
            assert (self.bitmask&split_at_this_bit) == 0, "bit_index already used in this dataset_field."
            pass
        
        #this is not needed. dataset is sorted in init.
        #sort first.
        # if not self.is_dataset_sorted:
        #     self.__sort_dataset()
        #     pass
        
        new_bitmask = self.bitmask|split_at_this_bit
        false_addr = self.addr
        true_addr = self.addr|split_at_this_bit
        
        #optimizable
        true_dataset:list[tuple[int,bool]] = []
        true_has_0 = False
        true_has_1 = False
        false_dataset:list[tuple[int,bool]] = []
        false_has_0 = False
        false_has_1 = False
        #assert False,"顺便把有什么东西一起找了。"
        
        assert self.dataset is not None
        for item in self.dataset.data:
            the_bit_in_item_addr_with_shift = item[0]&split_at_this_bit
            the_bit_in_item_addr:bool = the_bit_in_item_addr_with_shift!=0
            if the_bit_in_item_addr:
                true_dataset.append(item)
                #optimizable? like:
                # if not (true_has_1) and item[1]???
                # if not (true_has_0) and item[1]???
                if item[1]:
                    true_has_1=True
                    pass
                else:
                    true_has_0=True
                    pass
                pass
            else:
                false_dataset.append(item)
                #also here.
                if item[1]:
                    false_has_1=True
                    pass
                else:
                    false_has_0=True
                    pass
                pass
            pass
        
        #this is not needed. the items are added in sequence.
        #maybe optimizable.
        #false_dataset.sort(key=lambda item:item[0])
        #true_dataset.sort(key=lambda item:item[0])
        
        __check_all_safety = _debug__check_all_safety_in_split
        
        #这是一幅对联
        false_part = DatasetField(bitmask = new_bitmask, addr=false_addr, #input_bits_count=self.input_bits_count,
            bits_already_in_use = self.bits_already_in_use+1, dataset = Dataset(self.input_bits_count, false_dataset, True), 
            with_suggest = True, suggest_has_1 = false_has_1, suggest_has_0 = false_has_0, 
            _debug__check_all_safety = __check_all_safety,
            leaf_keep_dataset = leaf_keep_dataset, branch_keep_dataset = branch_keep_dataset,
            )
        
        true_part = DatasetField(bitmask = new_bitmask, addr=true_addr, #input_bits_count=self.input_bits_count,
            bits_already_in_use = self.bits_already_in_use+1, dataset = Dataset(self.input_bits_count, true_dataset, True), 
            with_suggest = True, suggest_has_1 = true_has_1, suggest_has_0 = true_has_0, 
            _debug__check_all_safety = __check_all_safety,
            leaf_keep_dataset = leaf_keep_dataset, branch_keep_dataset = branch_keep_dataset,
            )

        self.children = (false_part, true_part)
        pass
        
        
    def _lookup_only___is_leaf(self)->bool:
        if self.children is None:
            return True
        else:
            return False
        
    def lookup_version_1___dont_use(self, addr:int, lookup_in_leaf_dataset = False)->tuple[bool,bool]:#[None,None,bool,bool]:
        #to do :suggest, like allow_irr, better_non_irr, better_irr, true_when_irr, false_when_irr.
        
        #the sequence is the same as __init__, except for all the branch cases are priotized to the top.
        '''return (result_is_irr, result_is_true)
        
        Note. This function never look up in the self.dataset of branch.'''
        #to do:return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)'''
        
        # 0, this is different. I guess a lot calling to this function is not the end, 
        # so let's detect non leaf node first.
        if not self.get_is_leaf():
            _temp__mask_of_this_bit = 1<<self.best_index_to_split_from_right_side
            this_bit_of_addr__with_shift = _temp__mask_of_this_bit&addr
            this_bit_of_addr = this_bit_of_addr__with_shift != 0
            the_child = self._get_child(this_bit_of_addr)
            return the_child.lookup_version_1___dont_use(addr, lookup_in_leaf_dataset = lookup_in_leaf_dataset)
        
        # 1, is dataset empty. 
        if self.all_irr:
            '''return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)'''
            return(True, False)#irr.
        
        # 2 and 3 don't return.
        
        # 4, only the true xor here. Fake xor results in a branch node and handled at 0th uppon 
        if self.all_xor:
            result = self._lookup_only__all_xor_only(addr)
            return(False, result)
            
        # 5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
        # the dataset is assumed to be sorted.
        if not lookup_in_leaf_dataset:
            if self.has_0:
                return(False, False)
            else:
                return(False, True)
        else:#with the dataset for leaf node, it's 
            assert self.dataset is not None, "Set leaf_keep_dataset=True when create this object or the root object."
            found, _ = self.dataset.find_addr(addr)
            if found:
                if self.has_0:
                    return(False, False)
                else:
                    return(False, True)
            else:#not in the dataset, then it's irr.
                return(True, False)
            pass#else
            '''
            #assert False,"old code below"
            
            #a binary search pasted from bing.
            left = 0
            right = self.dataset._len() - 1
            #found = False
            while left <= right:
                mid:int = (left + right) // 2
                addr_here = self.dataset.data[mid][0]
                if addr_here == addr:
                    return (False, self.dataset.data[mid][1])  # Target found
                elif addr_here < addr:
                    left = mid + 1  # Search in the right half
                    pass
                else:
                    right = mid - 1  # Search in the left half
                    pass
                pass
            return (True, False)#irr
            '''
        pass#end of function.
        # 6, it's the 0th uppon. So, no 6 here.
    
    def lookup_version_2___dont_use(self, addr:int, lookup_in_leaf_dataset = False, as_xor_as_possible = False)->tuple[bool,bool,bool,bool,bool]:#[None,None,bool,bool]:
        #to do :suggest, like allow_irr, better_non_irr, better_irr, true_when_irr, false_when_irr.
        
        #the sequence is the same as __init__, except for all the branch cases are priotized to the top.
        '''For the outside use, only take the first 2 return value. eg.
        >>> is_irr, is_true, _,_,_ = self.looup(addr)
        
        return (result_is_irr, result_or_suggest_is_true, from_all_irr_field, from_xor_field, actually_irr_according_to_dataset)
                
        The last input param lookup_in_leaf_dataset is a __debug__ feature, and only affects the last output bool. 
        If the input flag is true, and the result is irr according to the dataset(I call it accurate irr), 
        the last return flag is true.
        This case only happens when the leaf is 1+ir or 0+ir, and the addr is ir.
        (maybe in the future, after I implement the non-full-xor-field, maybe this irr also comes from non-full-xor-field.)
                
        This function returns in several styles.
        If this node is a branch, it looks up it's child(according to address.). After get the result from child, 
        it decides how it returns.
        Branch-1, it looks up into a normal child(non all-irr, non xor), it returns what the child returns.
        Branch-2, it's all-irr-field(doesn't apply to accurate irr), it creates a fake address(only one bit different, 
        and the bit is represented by the node.).
        It looks up into the other child(with the fake addr just created), in order to get at least some suggestion.
        The reason is that, this tool first seperate the entire field at the most related non-xor bit in addr. 
        That is the root node. Then, as it digs deeper, and the nodes are farther from the root node, the related-score decreases.
        I mean, ***most related->least related->non related->xor***.
        If the dataset doesn't have any xor-field as any sub field(children), it only has most ***related->least related->non related***
        If a addr is irrelevant item, then this tool tries to provide a suggestion. It firstly changes the non related bits 
        and see if there is any useful info. This is the reason I organize the addrs in fields. 
        A irr in a 1+ir field is suggested as 1, becasuse it's possible to only change the non-related and very little related 
        bits in addr to see a 1 somewhere. 
        If this in-field-reference still doesn't do the job(the only case is, it's a all-irr field/node),then it needs to change 
        a more related bit in addr to see if it can modify the least and least related bits to get any reasonable reference.
        Anyway, I know it's messy, and I didn't prove it in math. But this tool is designed to do this. Trust my testing skill.
        Branch-3, the fake addr goes into a xor-field. I don't know if I should reverse the suggestion or not.
        
        If this node is a leaf, return according to its info stored.(The simple case. 
        I can have some fun writing simple code please? This tool is too hard for me.)
        
        Note. This function never looks up in the self.dataset of branch. If the lookup_in_leaf_dataset is true, it looks up 
        in the dataset in leaves if the type keeps the dataset. It only store the info in the last bool of output, and never 
        changes any other behavior of this function. 
        '''
        #to do:return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)'''
        
        # 0, this is different. I guess a lot calling to this function is not the end, 
        # so let's detect non leaf node first.
        if not self.get_is_leaf():#branch.
            _temp__mask_of_this_bit = 1<<self.best_index_to_split_from_right_side
            this_bit_of_addr__with_shift = _temp__mask_of_this_bit&addr
            this_bit_of_addr = this_bit_of_addr__with_shift != 0
            the_child = self._get_child(this_bit_of_addr)
            result_is_irr, result_or_suggest_is_true, from_all_irr_field, from_xor_field, actually_irr_according_to_dataset \
                = the_child.lookup(addr, lookup_in_leaf_dataset = lookup_in_leaf_dataset)
            if from_all_irr_field:
                '''use the other child to get any suggestion.'''
                
                '''this line(below) automatically uses the fake addr with only 1 bit different. But the fake addr can get faked again 
                inside. My idea is, if it can touch only bits ***less related than a certain level***, it's more reasonable. 
                If you found this hypothesis wrong or you don't like it, maybe you want to modify the code below, or 
                maybe you need to implement some other algo based on containers, but not recursive.'''
                the_other_child = self._get_child(not this_bit_of_addr)
                result_is_irr, result_or_suggest_is_true, from_all_irr_field, from_xor_field, actually_irr_according_to_dataset \
                    = the_other_child.lookup(addr, lookup_in_leaf_dataset = lookup_in_leaf_dataset)
                
                '''Branch-2. If the fake addr is also in all-irr-field, return as a all-irr leaf'''
                if from_all_irr_field:
                    assert False, "unfinished."
                    #return (result_is_irr 1w可能应该是true, result_or_suggest_is_true, True, False, actually_irr_according_to_dataset)
                
                '''Branch-3. If the fake addr is a all-xor-field, reverse '''
                if from_xor_field:
                    if as_xor_as_possible:
                        '''0110____ explained as 0110 1001'''
                        assert False, "unfinished."
                        #return (False 1w可能应该是true, not result_or_suggest_is_true, False, True, actually_irr_according_to_dataset)
                    else:
                        '''0110____ explained as 0110 0110'''
                        assert False, "unfinished."
                        #return (False 1w可能应该是true, result_or_suggest_is_true, False, True, actually_irr_according_to_dataset)
                    pass#
                
                '''Branch-1. The normal case, simply returns'''
                assert False, "unfinished."
                #return (result_is_irr 1w可能应该是true, result_or_suggest_is_true, False, False, actually_irr_according_to_dataset)
            else: #the child is 1/0, 1/0+irr, xor/xnor, or branch. A simple return.
                '''Branch-1. The normal case, simply returns'''
                assert False,"unfinished....... Probably not gonna come back. "
                return (result_is_irr, result_or_suggest_is_true, from_all_irr_field, from_xor_field, \
                    actually_irr_according_to_dataset)
                
        
        # 1, is dataset empty. 
        if self.all_irr:
            '''return (result_is_irr, result_or_suggest_is_true, from_all_irr_field, from_xor_field, 
            actually_irr_according_to_dataset)'''
            return(True, False, True, False, False)#irr.
        
        # 2 and 3 don't return.
        
        # 4, only the true xor here. Fake xor results in a branch node and handled at 0th uppon 
        if self.all_xor:
            result = self._lookup_only__all_xor_only(addr)
            '''return (result_is_irr, result_or_suggest_is_true, from_all_irr_field, from_xor_field, 
            actually_irr_according_to_dataset)'''
            return(False, result, False, True, False)
            
        # 5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
        # the dataset is assumed to be sorted.
        if not lookup_in_leaf_dataset:
            '''return (result_is_irr, result_or_suggest_is_true, from_all_irr_field, from_xor_field, 
            actually_irr_according_to_dataset)'''
            return (False, self.has_1, False, False, False)
            '''old style
                if self.has_0:
                    return(False, False, False, False)
                else:
                    return(False, True, False, False)
            '''
        else:#with the dataset for leaf node, it's 
            assert self.dataset is not None, "Set leaf_keep_dataset=True when create this object or the root object."
            found, _ = self.dataset.find_addr(addr)
            '''return (result_is_irr, result_or_suggest_is_true, from_all_irr_field, from_xor_field, 
            actually_irr_according_to_dataset)'''
            return (not found, self.has_1, False, False, not found)#new return style.
            '''the old style
            if found:
                if self.has_0:
                    return(False, False, False, False)
                else:
                    return(False, True, False, False)
            else:#not in the dataset, then it's irr. But this function recommends the field answer, which doesn't exist in version 1.
                if self.has_0:
                    return(True, False, False, False)
                else:
                    return(True, True, False, False)
            '''
        # 6, it's the 0th uppon. So, no 6 here.
        pass#end of function.
    
    
    def _get_critial_bit(self)->int:
        one_shift = 1<<self.best_index_to_split_from_right_side
        return one_shift
    
    
    def lookup(self, addr:int, lookup_in_leaf_dataset = False, \
                as_xor_as_possible = False, \
                dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned = False\
                    )->tuple[bool,bool,bool,int,'DatasetField']:
        '''
        this is the 3rd version of this function. It's not a recursive function. I don't like recursive.
        docs here maybe outdated.
        
        return(for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node)
        
        >>> for_sure_it_is_irr: If it's true, it's irr. Otherwise, NOT SURE!!!.
        >>> for_sure_it_is_NOT_irr: If it's true, it's NOT irr. Otherwise, NOT SURE!!!. Notice: I'm not 
        sure if I should change this. Only 1,0,xor field set this flag to true, 1+ir and 0+ir doesn't. Thus, 
        it's impossible to distinguish (1+ir/0+ir) from all-irr.
        
        >>> result_or_suggest: If is_original_irr, then this is suggestion. Otherwise this is result.
        If you only care about if this is irrelevant addr, ignore this value.
        
        >>> found_in_addr: If the original addr is a all-irr field, then the addr is modified to find some suggestion. 
        This found_in_addr value indicates the source of suggestion. Basically debug purpose.
        (If the last test in this function is in a 1+ir or 0+ir field, the addr is not modified. So this value is always in 
        a leaf but doesn't guarantee to be relevant. In future version, the xor field may get modified, so the behavior of 
        this function may also change.)
        >>> found_in_node: Similar to found_in_addr, but the node info.
        
        If you only want to check out if an addr is irrelevant, Im planning another function.
        
        input:
        
        >>> addr: Looks up in this addr.
        >>> lookup_in_leaf_dataset: Mostly __debug__ feature. If the addr is in a 1+ir/0+ir node, 
        it looks up in the dataset to see if it's a real irrelevant item.
        If it doesn't have dataset, it raise exception. 
        >>> as_xor_as_possible: A hyperparameter. Have fun.
        
        Docs here maybe outdated below
        >>> Docs here maybe outdated below
        >>> Docs here maybe outdated below
        >>> Docs here maybe outdated below
        
        
        This case only happens when the leaf is 1+ir or 0+ir, and the addr is ir.
        (maybe in the future, after I implement the non-full-xor-field, maybe this irr also comes from non-full-xor-field.)
                
        This function returns in several styles.
        If this node is a branch, it looks up it's child(according to address.). After get the result from child, 
        it decides how it returns.
        Branch-1, it looks up into a normal child(non all-irr, non xor), it returns what the child returns.
        Branch-2, it's all-irr-field(doesn't apply to accurate irr), it creates a fake address(only one bit different, 
        and the bit is represented by the node.).
        It looks up into the other child(with the fake addr just created), in order to get at least some suggestion.
        The reason is that, this tool first seperate the entire field at the most related non-xor bit in addr. 
        That is the root node. Then, as it digs deeper, and the nodes are farther from the root node, the related-score decreases.
        I mean, ***most related->least related->non related->xor***.
        If the dataset doesn't have any xor-field as any sub field(children), it only has most ***related->least related->non related***
        If a addr is irrelevant item, then this tool tries to provide a suggestion. It firstly changes the non related bits 
        and see if there is any useful info. This is the reason I organize the addrs in fields. 
        A irr in a 1+ir field is suggested as 1, becasuse it's possible to only change the non-related and very little related 
        bits in addr to see a 1 somewhere. 
        If this in-field-reference still doesn't do the job(the only case is, it's a all-irr field/node),then it needs to change 
        a more related bit in addr to see if it can modify the least and least related bits to get any reasonable reference.
        Anyway, I know it's messy, and I didn't prove it in math. But this tool is designed to do this. Trust my testing skill.
        Branch-3, the fake addr goes into a xor-field. I don't know if I should reverse the suggestion or not.
        
        If this node is a leaf, return according to its info stored.(The simple case. 
        I can have some fun writing simple code please? This tool is too hard for me.)
        
        Note. This function never looks up in the self.dataset of branch. If the lookup_in_leaf_dataset is true, it looks up 
        in the dataset in leaves if the type keeps the dataset. It only store the info in the last bool of output, and never 
        changes any other behavior of this function. 
        '''
        
        root = self
        if not dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned:
            assert root.bitmask == 0
            pass
        
        class LookupLog:
            child:'DatasetField'
            bitmask:int
            child_addr:int
            is_leaf:bool
            target_addr:int
            fake_addr_into:int
            info:str
            def __init__(self, child:'DatasetField', bitmask:int ,child_addr:int, is_leaf:bool, \
                        target_addr:int, fake_addr_into:int, info:str):
                self.child = child
                self.bitmask = bitmask
                self.child_addr = child_addr
                self.is_leaf = is_leaf
                self.target_addr = target_addr
                self.fake_addr_into = fake_addr_into
                self.info = info
                pass
            pass
        log:list[LookupLog] = []
        log_simple:list[str] = []
        node:'DatasetField' = root
        
        #return(for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node)
        for_sure_it_is_irr = False
        for_sure_it_is_NOT_irr = False
        target_addr = addr
        path:list['DatasetField'] = [root]
        _path_in_bool:list[bool] = []
        
        '''
        return(is_original_irr, result_or_suggest, is_accurate_irr, found_in_addr, found_in_node)
        and, they are actually:
        original_an_irr:bool
        result:bool
        original_an_accurate_irr:bool
        target_addr:int
        node:'DatasetField'
        '''
        #the sequence is the same as __init__, except for all the branch cases are priotized to the top.
        
        # 0, this is different. I guess a lot calling to this function is not the end, 
        # so let's detect non leaf node first.
        
        
        # if addr == 3:
        #     fdsfds = 432
        
        while True:#a while here?
            node = path[-1]#does this look like the call stack?
            is_node_leaf = node.get_is_leaf()
            if not is_node_leaf:#branch.
                #a. some work
                _temp__mask_of_this_bit = 1<<node.best_index_to_split_from_right_side
                this_bit_of_addr__with_shift = _temp__mask_of_this_bit&target_addr
                this_bit_of_addr = this_bit_of_addr__with_shift != 0
                #b. move (or return)
                _path_in_bool.append(this_bit_of_addr)
                the_child = node._get_child(this_bit_of_addr)
                path.append(the_child)
                #c. log:
                _log_str = f"into child branch, new depth:{path.__len__()}"
                _log_obj = LookupLog(node, node.bitmask, node.addr, is_node_leaf, target_addr, -1, _log_str)
                log.append(_log_obj)
                log_simple.append(_log_str)
                continue            
            
            #leaf:
            # 1, is dataset empty. 
            if node.all_irr:#fake a new addr.
                
                if dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned:
                    '''return(for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node)'''
                    return(True, False, False, target_addr, node)
                
                #a. some work
                old_target_addr = target_addr#for log
                
                # two results.
                '''It's either the first time into an all-irr field(set the flag), 
                or from some all-irr field(flag already set and should not get modified).'''
                for_sure_it_is_irr = True
                
                #b. move (or return)
                #fake the addr to get suggestion.
                parent = path[-2]
                the_critical_bit = parent._get_critial_bit()
                target_addr = target_addr ^ the_critical_bit
                path.pop()
                
                #log:
                _log_str = f"all irr leaf, faked addr into {readable_binary(target_addr, node.input_bits_count)}"
                _log_obj = LookupLog(node, node.bitmask, node.addr, is_node_leaf, old_target_addr, target_addr, _log_str)
                log.append(_log_obj)
                log_simple.append(_log_str)
                continue
            # 2 and 3 don't return.
            # 4, only the true xor here. Fake xor results in a branch node and handled at 0th uppon 
            if node.all_xor:
                #a. some work
                result_or_suggest = node._lookup_only__all_xor_only(target_addr)
                
                if not for_sure_it_is_irr:#make sure it's not "fake addr" from a all-irr node.
                    #if lookup_in_leaf_dataset:
                        #In this version, all xor is a full relevant field.
                    #no need to modify it again. for_sure_it_is_irr = False
                    for_sure_it_is_NOT_irr = True
                    #    pass
                    pass
                #b. (move or) return (see below)
                #c. log:
                _log_str = f"all xor leaf, [return:{result_or_suggest}]"
                _log_obj = LookupLog(node, node.bitmask, node.addr, is_node_leaf, target_addr, -1, _log_str)
                log.append(_log_obj)
                log_simple.append(_log_str)
                #return
                '''return(for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node)'''
                return(for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, target_addr, node)

            # 5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
            # the dataset is assumed to be sorted.

            # a normal leaf, return the result.
            #a. some work
            if not for_sure_it_is_irr:#make sure it's not "fake addr" from a all-irr node.
                if lookup_in_leaf_dataset:
                    assert node.dataset is not None
                    found, _ = node.dataset.find_addr(target_addr)
                    #accurate result.
                    for_sure_it_is_irr = not found
                    for_sure_it_is_NOT_irr = found
                    pass
                else:#not a accurate lookup
                    if not node.has_irr:#but it's a 1 or 0 field. Still very sure it's not irr.
                        #no needed. for_sure_it_is_irr = False 
                        for_sure_it_is_NOT_irr = True
                        pass
                    pass
                #else not needed. If it's not accurate lookup, don't touch the 2 bools.
                pass
            
            result_or_suggest = node.has_1
            #old code for_sure_it_is_NOT_irr = not node.has_irr#a field without irr, then the addr is for sure not irr.
            #b. (move or) return (see below)
            #c. log:
            _log_str = f"simple leaf, [return:{result_or_suggest}]"
            _log_obj = LookupLog(node, node.bitmask, node.addr, is_node_leaf, target_addr, -1, _log_str)
            log.append(_log_obj)
            log_simple.append(_log_str)
            #return
            '''return(for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node)'''
            return(for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, target_addr, node)
            # 6, it's the 0th uppon. So, no 6 here.
        pass#end of function.
    
    
    def _lookup_only__all_xor_only(self, addr:int)->bool:
        assert self.all_xor, "不记得昨天写的什么意思了。总之还没测试。non-all-xor case can not call this function."
        #the docs here is copied form the _init_only__get__if_not_after_xor__it_s_already_all_xor and probably wrong.
        #pick anything from self.dataset and detect if all input are Falses(simpler than true), is the output the xor result, or the reversed.
        #actually, the dataset should already be sorted, but the performance is probably similar and trivial?
        # So, yeah, I don't care, I detect the addr here, instead of check the "sorted" and do the trick.
        #maybe it's optimizable. But not for now.
        #addr_of_item = self.dataset[0][0]
        #result_of_item = self.dataset[0][1]
        num_of_ones_in_addr = 0
        for i in range(self.input_bits_count-1,-1,-1):
            one_shift_by_i:int = 1<<i
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #This bit is in bitmask, ignore this i.
                continue
            
            '''ignore the irr-bit'''
            bit_in_irr_bitmask__for_this_i = one_shift_by_i&self.when_xor__ignore_these_bits
            if bit_in_irr_bitmask__for_this_i != 0:
                #This bit is in bitmask, ignore this i.
                continue
            
            bit_in_addr_for_this_i = addr & one_shift_by_i
            if bit_in_addr_for_this_i != 0:
                num_of_ones_in_addr = num_of_ones_in_addr + 1#optimizable. This way simplifies the debugging.
                pass
            pass
        is_num_of_ones_in_addr_odd = (num_of_ones_in_addr&1) != 0
        
        # self.not_after_xor # if all 0s, and not_after_xor is false, result is false. Every one flips it once.
        result = is_num_of_ones_in_addr_odd^self.not_after_xor
        return result
        
    def get_already_const_without_irr(self)->bool:
        has_only_one_side = (self.has_0 ^ self.has_1)
        has_only_one_side_without_irr = has_only_one_side and (not self.has_irr)
        return has_only_one_side_without_irr

    def _get_child(self, true_or_false:bool)->'DatasetField':
        # self.children = (true_part, false_part)
        if self.children is None:
            raise Exception("Is this a leaf node?")
        else:
            if true_or_false:
                return self.children[1]
            else:
                return self.children[0]  
        #end of function
    
    def ____broken____Im_sorry____print_all_info_directly(self):
        temp = f"<DatasetField> bitmask:{self.bitmask}, addr:{self.addr:b}, input_bits_count{self.input_bits_count}, "
        temp += f"bits_already_in_use{self.bits_already_in_use}, "
        temp += f"dataset__len__:{self.dataset.__len__()}, best_index_to_split:{self.best_index_to_split_from_right_side}, "
        if self.children is None:
            temp+="children is None, "
        else:
            temp+="has children, "
            pass
        print(temp)
        temp2 = f"has_irr:{self.has_irr}, all_irr:{self.all_irr}, has_0:{self.has_0}, has_1:{self.has_1}, all_xor:{self.all_xor}, "
        temp2 += f"already_const_without_irr:{self.get_already_const_without_irr()}, "
        temp2 += f"ready_for_lookup:{self.ready_for_lookup}, is_dataset_sorted:{self.is_dataset_sorted}, is_leaf_node:{self.get_is_leaf()}, "
        temp2 += f"not_after_xor:{self.not_after_xor}, "
        print(temp2)
        pass
    
    def readable_as_tree(self, depth:int = -1, use_TF = False, with_irr_bit_for_xor = True)->str:
        '''Behavior:
        leaf node: addr, result.
        branch node:
        depth: -1 to print all depth. 
        depth: 0, has children, print addr(...)
        depth: +num, print addr(child1, child2) 
        '''
        true_char = "1"
        false_char = "0"
        if use_TF:
            true_char = "T"
            false_char = "F"
            pass
            
        addr_str = self.get_readable_addr()+":"
        if self.get_is_leaf():
            if self.all_xor:
                irr_bit_str = ""
                if with_irr_bit_for_xor:
                    irr_bit_str = f"(irr-bits:{self._get_readable_only___irr_bits()})"
                    pass
                if self.not_after_xor:
                    return addr_str+"xnor"+irr_bit_str
                else:
                    return addr_str+"xor"+irr_bit_str
                pass#
            elif self.all_irr:
                return addr_str+"ir"        
            else:
                if self.has_irr:
                    if self.has_0:#0+irr
                        return addr_str+false_char+"+ir"        
                    else:#1+irr
                        return addr_str+true_char+"+ir"        
                else:#no irr
                    if self.has_0:
                        return addr_str+false_char   
                    else:
                        return addr_str+true_char
        else:# self is not leaf_node:
            if 0 == depth:
                return addr_str+"(..)"
            else:
                new_depth = depth-1
                false_child = self._get_child(False)
                true_child = self._get_child(True)
                false_str = false_child.readable_as_tree(new_depth, use_TF)
                true_str = true_child.readable_as_tree(new_depth, use_TF)
                return addr_str+f"({false_str}, {true_str})"
        #end of function
    
    def get_readable_addr(self)->str:
        '''Call this function after check the addr, otherwise the behavior of this function is undefined.
        
        This function at least takes advantage of an assuption, that, if the addr is correct, it only has 1s where the bitmask has 1s.
        Which means, if a bit in bitmask is 0, in addr, it must be 0, '''
        bitmask_in_str = f"{self.bitmask:b}"
        addr_in_str = f"{self.addr:b}"
        bitmask_is_this_longer_than_addr = bitmask_in_str.__len__()-addr_in_str.__len__()
        temp_str = ""
        for i in range(bitmask_in_str.__len__()):
            #i is left to right.
            bit = bitmask_in_str[i]
            if "0" == bit:
                temp_str += "_"
                pass
            else:#"1" == bit:
                offset_in__addr_in_str = i-bitmask_is_this_longer_than_addr
                if offset_in__addr_in_str<0:
                    temp_str += "0"
                    pass
                else:
                    temp_str += addr_in_str[i-bitmask_is_this_longer_than_addr]
                    pass
                pass
            pass
        how_many_underscores_needs_in_the_left = self.input_bits_count-bitmask_in_str.__len__()
        result = "_"*how_many_underscores_needs_in_the_left+temp_str
        return result
    
    def _get_readable_only___irr_bits(self, char_for_1_in_addr = "1", char_for_0_in_addr = "0", \
        char_for_irr_in_xor = "i", char_for_normal_in_xor = ".", )->str:
        str_for_every_bit:list[str] = []
        for i_from_left_side in range(self.input_bits_count-1,-1,-1):
            one_shift_by__i = 1<<i_from_left_side
            bitmask_of_this_i = self.bitmask&one_shift_by__i
            addr_of_this_i = self.addr&one_shift_by__i
            irr_bit_for_xor_of_this_i = self.when_xor__ignore_these_bits&one_shift_by__i
            if bitmask_of_this_i>0:
                if addr_of_this_i>0:
                    str_for_every_bit.append(char_for_1_in_addr)
                    pass
                else:
                    str_for_every_bit.append(char_for_0_in_addr)
                    pass
                pass
            else:
                if irr_bit_for_xor_of_this_i>0:
                    str_for_every_bit.append(char_for_irr_in_xor)
                    pass
                else:
                    str_for_every_bit.append(char_for_normal_in_xor)
                    pass
                pass
            pass#for
        final_result_str = "".join(str_for_every_bit)
        return final_result_str
        
    
    def valid(self, dataset:Dataset, total_amount = -1, accurate_lookup = False, \
            log_the_error_to_file_and_return_immediately = True)->tuple[bool,int,int]:
        '''return (finished_the_check?, check_count, error_count)
        
        >>> finished_the_check: all the reasonable amount of check finishes?
        >>> check_count: actual check amount. For a finished case, it's still possible to be less than the givin number.
        >>> error_count:[literally]
        
        The real check amount is calculated. If meaningful amount is less than the givin number, 
        this function only checks the reasonable amount.
        
        You can call this function directly on a non-root node, but the lookup function may raise some exception.
        '''
        assert total_amount!=0
        assert self.input_bits_count == dataset.max_input_bits
        
        #if the dataset is empty, the field should also be all irr.
        if dataset.data.__len__() == 0:
            assert self.bitmask != 0, "empty dataset + root node is wired."
            #otherwise, this is a debug only case.
            if not self.all_irr: 
                dataset.log_the_error()
                return (False, 1, 1)
            return (True, 1, 0)
        #safety first.
        _last_addr_in_input = dataset.data[-1][0]
        assert _last_addr_in_input < (1<<self.input_bits_count)
        
        error_count = 0
        if total_amount>=dataset.data.__len__():
            total_amount = -1
            pass
        if -1 == total_amount:#valid all.
            for item in dataset.data:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,_,_ = self.lookup(item[0], \
                    lookup_in_leaf_dataset=accurate_lookup)
                if for_sure_it_is_irr:
                    if log_the_error_to_file_and_return_immediately:
                        dataset.log_the_error()
                        return (False, -1, 1)
                    error_count = error_count +1
                    pass
                if accurate_lookup:
                    assert for_sure_it_is_irr^for_sure_it_is_NOT_irr
                    if not for_sure_it_is_NOT_irr:
                        if log_the_error_to_file_and_return_immediately:
                            dataset.log_the_error()
                            return (False, -1, 1)
                        error_count = error_count +1
                        pass
                    pass
                if item[1] != result_or_suggest:#the result.
                    if log_the_error_to_file_and_return_immediately:
                        dataset.log_the_error()
                        return (False, -1, 1)
                    error_count = error_count +1
                    pass
                pass#for
            return (True, dataset.data.__len__(), error_count)
            pass#if total_amount
        else:#valid random but only part of them.
            _total_amount = total_amount
            while _total_amount>0:
                item = dataset.data[random.randint(0, dataset.data.__len__()-1)]
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,_,_ = self.lookup(item[0], \
                    lookup_in_leaf_dataset=accurate_lookup)
                if for_sure_it_is_irr:
                    if log_the_error_to_file_and_return_immediately:
                        dataset.log_the_error()
                        return (False, -1, 1)
                    error_count = error_count +1
                    pass
                if accurate_lookup:
                    assert for_sure_it_is_irr^for_sure_it_is_NOT_irr
                    if not for_sure_it_is_NOT_irr:
                        if log_the_error_to_file_and_return_immediately:
                            dataset.log_the_error()
                            return (False, -1, 1)
                        error_count = error_count +1
                        pass
                    pass
                if item[1] != result_or_suggest:
                    if log_the_error_to_file_and_return_immediately:
                        dataset.log_the_error()
                        return (False, -1, 1)
                    error_count = error_count +1
                    pass
                #tail
                _total_amount = _total_amount -1
                pass#while
            return (True, total_amount, error_count)
            pass#else of if -1 == total_amount
        
        #end of function.
    
    def valid_irr(self, dataset:Dataset, total_amount_irr = -1, 
                log_the_error_to_file_and_return_immediately = True)->tuple[bool,int,int]:
        '''return (finished_the_check?, check_count, error_count)
        
        >>> finished_the_check: all the reasonable amount of check finishes?
        >>> check_count: actual check amount. For a finished case, it's still possible to be less than the givin number.
        >>> error_count:[literally]
        
        The real check amount is calculated. If meaningful amount is less than the givin number, 
        this function only checks the reasonable amount.
        
        You can call this function directly on a non-root node, but the lookup function may raise some exception.
        '''
        assert total_amount_irr!=0
        assert self.input_bits_count == dataset.max_input_bits
        
        #if the dataset is empty, the field should also be all irr.
        if dataset.data.__len__() == 0:
            assert self.bitmask != 0, "empty dataset + root node is wired."
            #otherwise, this is a debug only case.
            if not self.all_irr: 
                dataset.log_the_error()
                return(False, 0, 1)
            return(True, 0, 0)
        #safety first.
        assert dataset.is_sorted
        _last_addr_in_input = dataset.data[-1][0]
        assert _last_addr_in_input < (1<<self.input_bits_count)
        
        total_possible_amount = 1<<self.input_bits_count
        total_possible_amount_of_irr = total_possible_amount - dataset.data.__len__()
        if total_amount_irr>=total_possible_amount_of_irr:
            total_amount_irr = -1
            pass
        error_count = 0
        if -1 == total_amount_irr:
            #valid all irr.
            #valid all irr.
            #valid all irr.
            _, irr_addr_list = dataset.get_irr_addr___sorts_self()
            #dataset already non-empty.
            for irr_addr in irr_addr_list:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,_,_ = self.lookup(irr_addr, lookup_in_leaf_dataset=True)
                if (not for_sure_it_is_irr) or for_sure_it_is_NOT_irr:
                    if log_the_error_to_file_and_return_immediately:
                        dataset.log_the_error()
                        return (False, -1, 1)
                    error_count = error_count +1
                    pass
                pass
            return (True, total_possible_amount_of_irr, error_count)
            pass#-1 == total_amount_irr
        else:#only check random irrelevant:
            if total_possible_amount_of_irr< total_possible_amount/3:
                # not very many irr, so first get the list
                #random pick from irr list
                #random pick from irr list
                #random pick from irr list
                _, irr_addr_list = dataset.get_irr_addr___sorts_self()
                #dataset already non-empty.
                _total_amount_irr = total_amount_irr
                while _total_amount_irr>0:
                    rand_irr_addr:int = irr_addr_list[random.randint(0, irr_addr_list.__len__()-1)]
                    for_sure_it_is_irr, for_sure_it_is_NOT_irr,_,_,_ = self.lookup(rand_irr_addr, lookup_in_leaf_dataset=True)
                    if (not for_sure_it_is_irr) or for_sure_it_is_NOT_irr:
                        if log_the_error_to_file_and_return_immediately:
                            dataset.log_the_error()
                            return (False, -1, 1)
                        error_count = error_count +1
                        pass
                    #tail
                    _total_amount_irr = _total_amount_irr -1
                    pass#while
                return (True, total_amount_irr, error_count)
                pass#if 
                
            else:#a lot irr, random and check if it's a irr in ref set, and then.
                # guess
                # guess
                # guess
                total_trial_amount = total_amount_irr*20
                _total_amount_irr = total_amount_irr
                _total_trial_amount = total_trial_amount
                already_guessed:set[int] = set()
                one_shift__input_bits_count__minus_one = (1<<self.input_bits_count )-1
                #one_shift_field_len = 1<<self.input_bits_count
                #_not_important_incr = 0
                while _total_trial_amount>0 and _total_amount_irr>0:
                    guess_irr_addr = random.randint(0, one_shift__input_bits_count__minus_one)
                    #_not_important_incr = _not_important_incr + int(one_shift_field_len*61/337)
                    #_not_important_incr = _not_important_incr % one_shift_field_len
                    #guess_addr = (guess_addr+_not_important_incr) %one_shift_field_len
                    
                    if guess_irr_addr in already_guessed:
                        #tail
                        _total_trial_amount = _total_trial_amount -1
                        continue
                    else:
                        already_guessed.add(guess_irr_addr)
                        pass
                    #find the item in dataset. But if it finds, it's bad guess. 
                    #binary search # two valid style.
                    #guess_addr is the input. key = lambda item:dataset[item][0]
                    found = False
                    left = 0
                    right = dataset.data.__len__()-1
                    #if any bug occurs here, use the dataset.find_addr instead.
                    while left<=right:
                        mid:int = (left+right)//2
                        temp_addr = dataset.data[mid][0]
                        if guess_irr_addr<temp_addr:
                            right = mid-1# two valid style.
                            continue
                        elif guess_irr_addr>temp_addr:
                            left = mid+1# two valid style.
                            continue
                        else:#guess_addr == dataset[mid][0]:
                            found = True
                            break
                        pass#while of binary search.
                    if found: # the guess_addr is found in dataset, so it's a relevant. Ignore it.
                        #tail
                        _total_trial_amount = _total_trial_amount -1
                        continue
                    
                    #now the guess_addr is a irr according to dataset.
                    for_sure_it_is_irr, for_sure_it_is_NOT_irr,_,_,_ = self.lookup(guess_irr_addr, lookup_in_leaf_dataset=True)
                    if (not for_sure_it_is_irr) or for_sure_it_is_NOT_irr:
                        if log_the_error_to_file_and_return_immediately:
                            dataset.log_the_error()
                            return (False, -1, 1)
                        error_count = error_count +1
                        pass
                    #tail
                    _total_trial_amount = _total_trial_amount -1
                    _total_amount_irr = _total_amount_irr -1
                    pass#while
                if _total_amount_irr>0:#not fully checked.
                    dataset.log_the_error(filename="not totally tested.txt")
                    return (False, total_amount_irr - _total_amount_irr, error_count)
                else:#_total_amount_irr == 0:
                    return (True, total_amount_irr, error_count)
                pass#else
            pass#if -1 == total_amount_irr:#valid all irr.
        pass# end of function.
            
    def get_input_length(self)->int:
        return self.input_bits_count
    def get_output_length(self)->int:
        return 1
            
    #end of class
    
    
    
    
    
    
    
    
if True:
    max_input_bits = 4
    dataset = Dataset.from_str('''0110 1111 1111 0110''')
    assert dataset.data == [(0, False), (1, True), (2, True), (3, False), (4, True), (5, True), (6, True), (7, True), (8, True), 
            (9, True), (10, True), (11, True), (12, False), (13, True), (14, True), (15, False)]
    a_DatasetField:DatasetField = DatasetField._new__and_valid(dataset)
    pass
    
if "valid function" and True:
    if "correct dataset" and True:
        '''
        old code
        input_bits_count = 3
        dataset = Dataset(input_bits_count)
        a_DatasetField = DatasetField._new(dataset)
        assert a_DatasetField.get_input_length() == 3
        assert a_DatasetField.get_output_length() == 1
        
        # empty
        result_tuple = a_DatasetField.valid(dataset)
        assert result_tuple == (True, 1, 0)
        '''
        
        input_bits_count = 2
        dataset = Dataset(input_bits_count, [(0,True), (1,False), (3,True),])
        a_DatasetField = DatasetField._new__and_valid(dataset)
        assert a_DatasetField.get_input_length() == 2
        assert a_DatasetField.get_output_length() == 1
        
        # all relevant
        result_tuple = a_DatasetField.valid(dataset)
        assert result_tuple == (True, 3, 0)
        # random relevant
        result_tuple = a_DatasetField.valid(dataset, total_amount=2)
        assert result_tuple == (True, 2, 0)
        
        input_bits_count = 1
        dataset = Dataset(input_bits_count, [(0,True), ])
        a_DatasetField = DatasetField._new__and_valid(dataset)
        assert a_DatasetField.get_input_length() == 1
        assert a_DatasetField.get_output_length() == 1
        # big number.
        result_tuple = a_DatasetField.valid(dataset, total_amount=111)
        assert result_tuple == (True, 1, 0)
        
        input_bits_count = 2
        dataset = Dataset(input_bits_count, [(0,True), (1,False), (3,True),])
        a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset = True)
        tree = a_DatasetField.readable_as_tree()
        result_tuple = a_DatasetField.valid(dataset)
        assert result_tuple == (True, 3, 0)
        
        # all irr
        result_tuple = a_DatasetField.valid_irr(dataset)
        assert result_tuple == (True, 1, 0)
        
        # random from irr list
        input_bits_count = 3
        dataset = Dataset(input_bits_count, [(0,True), (1,False), (3,True), (4,False), (5,True), (7,False), ])
        a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset = True)
        result_tuple = a_DatasetField.valid_irr(dataset, total_amount_irr=1)
        assert result_tuple == (True, 1, 0)
        
        input_bits_count = 3
        dataset = Dataset(input_bits_count, [(0,True), (1,False), (3,True), (4,False), ])
        a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset = True)
        # guess irr
        result_tuple = a_DatasetField.valid_irr(dataset, total_amount_irr=2)
        #this should be (True, 2, 0). But if it guessed 40 times and less than 2 useful guess, then it's (False, 0 or 1, 0)
        assert result_tuple[2] == 0
        # big number.
        result_tuple = a_DatasetField.valid_irr(dataset, total_amount_irr=111)
        assert result_tuple == (True, 4, 0)
        pass
        
    if "wrong dataset" and True:
        input_bits_count = 2
        dataset = Dataset(input_bits_count, [(0,True), (1,False), ])
        a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)
        # all relevant
        wrong_dataset_1 = Dataset(input_bits_count, [(0,False), ])
        result_tuple = a_DatasetField.valid(wrong_dataset_1,log_the_error_to_file_and_return_immediately=False)
        assert result_tuple == (True, 1, 1)
        
        
        wrong_dataset_1 = Dataset(input_bits_count, [(2,False), ])
        result_tuple = a_DatasetField.valid(wrong_dataset_1,log_the_error_to_file_and_return_immediately=False)
        assert result_tuple == (True, 1, 1)
        # random relevant
        wrong_dataset_2 = Dataset(input_bits_count, [(0,False), (1,True), ])
        result_tuple = a_DatasetField.valid(wrong_dataset_2, total_amount=1,log_the_error_to_file_and_return_immediately=False)
        assert result_tuple == (True, 1, 1)
        wrong_dataset_2 = Dataset(input_bits_count, [(2,False), (3,True), ])
        result_tuple = a_DatasetField.valid(wrong_dataset_2, total_amount=1,log_the_error_to_file_and_return_immediately=False)
        assert result_tuple == (True, 1, 1)
        # all irr
        wrong_dataset_3 = Dataset(input_bits_count, [(1,True), (2,True), (3,False), ])
        wrong_dataset_3.sort()
        result_tuple = a_DatasetField.valid_irr(wrong_dataset_3, log_the_error_to_file_and_return_immediately=False)
        assert result_tuple == (True, 1, 1)
        
        # random from irr list
        input_bits_count = 3
        dataset = Dataset(input_bits_count, [(0,True), (1,False), (2,True), (3,True), (4,False), (5,False), (6,False), (7,False), ])
        a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)
        wrong_dataset_4 = Dataset(input_bits_count, [(0,True), (1,True), (2,True), (3,True), (4,True), (5,True), ])
        wrong_dataset_4.sort()
        result_tuple = a_DatasetField.valid_irr(wrong_dataset_4, total_amount_irr=1,log_the_error_to_file_and_return_immediately=False)
        assert result_tuple == (True, 1, 1)
        # guess irr
        wrong_dataset_5 = Dataset(input_bits_count, [(0,True), ])
        result_tuple = a_DatasetField.valid_irr(wrong_dataset_5, total_amount_irr=1,log_the_error_to_file_and_return_immediately=False)
        assert result_tuple == (True, 1, 1)
        pass
    pass
    
if "readable addr" and True:
    for addr in range(0b10):
        a_DatasetField = DatasetField(bitmask = 0b1, addr = addr, bits_already_in_use=1, \
                        dataset = Dataset(1), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass
    
    for addr in range(0b100):
        a_DatasetField = DatasetField(bitmask = 0b11, addr = addr, bits_already_in_use=2, \
                        dataset = Dataset(2), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass

    for addr in range(0b1000):
        a_DatasetField = DatasetField(bitmask = 0b111, addr = addr, bits_already_in_use=3, \
                        dataset = Dataset(3), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass
    
    for addr in range(0b10000):
        a_DatasetField = DatasetField(bitmask = 0b1111, addr = addr, bits_already_in_use=4, \
                        dataset = Dataset(4), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass
    
    for addr in range(0b100000):
        a_DatasetField = DatasetField(bitmask = 0b11111, addr = addr, bits_already_in_use=5, \
                        dataset = Dataset(5), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass
    
    pass

if "readable tree" and True:
    #111111111111111可以继续
    
    '''
    old code
    a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                        dataset = Dataset(1), with_suggest=False,_debug__check_all_safety = True)
    tree_str = a_DatasetField.readable_as_tree()
    assert tree_str == "_:ir"
    tree_str_TF = a_DatasetField.readable_as_tree(use_TF=True)
    assert tree_str_TF == "_:ir"'''
    
    def str_to_readable_tree(input:str)->tuple[str, str]:
        dataset = Dataset.from_str(input)
        a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        a_DatasetField.valid(dataset)
        tree_str = a_DatasetField.readable_as_tree()
        tree_str_TF = a_DatasetField.readable_as_tree(use_TF=True)
        return (tree_str, tree_str_TF)
    if "already tested" and True:
        tree_str, tree_str_TF = str_to_readable_tree("10")
        assert tree_str == "_:(0:1, 1:0)"
        assert tree_str_TF == "_:(0:T, 1:F)"
        
        tree_str, tree_str_TF = str_to_readable_tree("1_")
        assert tree_str == "_:1+ir"
        assert tree_str_TF == "_:T+ir"
        
        tree_str, tree_str_TF = str_to_readable_tree("_1")
        assert tree_str == "_:1+ir"
        assert tree_str_TF == "_:T+ir"
        
        tree_str, tree_str_TF = str_to_readable_tree("0_")
        assert tree_str == "_:0+ir"
        assert tree_str_TF == "_:F+ir"
        
        tree_str, tree_str_TF = str_to_readable_tree("111_")
        assert tree_str_TF == "__:T+ir"
        
        tree_str, tree_str_TF = str_to_readable_tree("1100")
        assert tree_str_TF == "__:(0_:T, 1_:F)"
        
        tree_str, tree_str_TF = str_to_readable_tree("11_0")
        assert tree_str_TF == "__:(0_:T, 1_:F+ir)"
        
        tree_str, tree_str_TF = str_to_readable_tree("1110")
        assert tree_str_TF == "__:(0_:T, 1_:(10:T, 11:F))"
        
        tree_str, tree_str_TF = str_to_readable_tree("0110")
        assert tree_str_TF == "__:xor(irr-bits:..)"
        
        tree_str, tree_str_TF = str_to_readable_tree("01101001")
        assert tree_str_TF == "___:xor(irr-bits:...)"
        
        tree_str, tree_str_TF = str_to_readable_tree("01100110")
        assert tree_str_TF == "___:xor(irr-bits:i..)"
        #old version assert tree_str_TF == "___:(0__:xor, 1__:xor)"
        
        tree_str, tree_str_TF = str_to_readable_tree("00111100")
        assert tree_str_TF == "___:xor(irr-bits:..i)"
        
        tree_str, tree_str_TF = str_to_readable_tree("01011010")
        assert tree_str_TF == "___:xor(irr-bits:.i.)"
        
        tree_str, tree_str_TF = str_to_readable_tree("0110100110010110")
        assert tree_str_TF == "____:xor(irr-bits:....)"
        
        tree_str, tree_str_TF = str_to_readable_tree("0110100101101001")
        assert tree_str_TF == "____:xor(irr-bits:i...)"
        
        tree_str, tree_str_TF = str_to_readable_tree("0110111111110110")
        #assert tree_str_TF == "____:(___0:(__00:xor(irr-bits:..00), __10:T), ___1:(__01:T, __11:xor(irr-bits:..11)))"
        assert tree_str_TF == "____:(0___:(00__:xor(irr-bits:00..), 01__:T), 1___:(10__:T, 11__:xor(irr-bits:11..)))"
        
        tree_str, tree_str_TF = str_to_readable_tree("0110011011111111")
        assert tree_str_TF == "____:(0___:xor(irr-bits:0i..), 1___:T)"
    
    pass

if "init and split" and True:
    # it's not allowed to have no input. So test starts with 1 input.
    if "1 bit input, 0 free bit" and True:
        if "two wrong addr, they raise" and False and False:
            dataset = Dataset(1, [(0b1,True), ])
            a_DatasetField = DatasetField(bitmask = 1, addr = 0, bits_already_in_use=1, \
                    dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            dataset = Dataset(1, [(0b0,True), ])
            a_DatasetField = DatasetField(bitmask = 1, addr = 1, bits_already_in_use=1, \
                    dataset = dataset,with_suggest=False,_debug__check_all_safety = True)
            pass
        #irr 
        a_DatasetField = DatasetField(bitmask = 1, addr = 0, bits_already_in_use=1, \
                        dataset = Dataset._debug__new_empty(1), with_suggest=False,_debug__check_all_safety = True)
        irr_addr = 0b0
        found_in_addr:int
        for_sure_it_is_irr, for_sure_it_is_NOT_irr, _, found_in_addr, found_in_node = \
            a_DatasetField.lookup(irr_addr, dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned=True)
        assert for_sure_it_is_irr
        assert for_sure_it_is_NOT_irr == False
        assert found_in_addr == irr_addr
        assert found_in_node == a_DatasetField
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "0"
        
        a_DatasetField = DatasetField(bitmask = 1, addr = 1, bits_already_in_use=1, \
                        dataset = Dataset._debug__new_empty(1), with_suggest=False,_debug__check_all_safety = True)
        irr_addr = 0b1
        for_sure_it_is_irr, for_sure_it_is_NOT_irr, _, found_in_addr, found_in_node = \
            a_DatasetField.lookup(irr_addr, dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned=True)
        assert for_sure_it_is_irr
        assert for_sure_it_is_NOT_irr == False
        assert found_in_addr == irr_addr
        assert found_in_node == a_DatasetField
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "1"
        
        #relevant.
        dataset = Dataset(1, [(0b0,True), ])
        a_DatasetField = DatasetField(bitmask = 1, addr = 0, bits_already_in_use=1, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True)#, leaf_keep_dataset=True)
        for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node = \
            a_DatasetField.lookup(dataset.data[0][0], dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned=True)
        assert for_sure_it_is_irr == False
        assert for_sure_it_is_NOT_irr
        assert result_or_suggest == dataset.data[0][1]
        assert found_in_addr == dataset.data[0][0]
        assert found_in_node == a_DatasetField
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "0"
        
        dataset = Dataset(1, [(0b0,False), ])
        a_DatasetField = DatasetField(bitmask = 1, addr = 0, bits_already_in_use=1, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node = \
            a_DatasetField.lookup(dataset.data[0][0], dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned=True)
        assert for_sure_it_is_irr == False
        assert for_sure_it_is_NOT_irr
        assert result_or_suggest == dataset.data[0][1]
        assert found_in_addr == dataset.data[0][0]
        assert found_in_node == a_DatasetField
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "0"
        
        dataset = Dataset(1, [(0b1,True), ])
        a_DatasetField = DatasetField(bitmask = 1, addr = 1, bits_already_in_use=1, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node = \
            a_DatasetField.lookup(dataset.data[0][0], dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned=True)
        assert for_sure_it_is_irr == False
        assert for_sure_it_is_NOT_irr
        assert result_or_suggest == dataset.data[0][1]
        assert found_in_addr == dataset.data[0][0]
        assert found_in_node == a_DatasetField
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "1"
        
        dataset = Dataset(1, [(0b1,False), ])
        a_DatasetField = DatasetField(bitmask = 1, addr = 1, bits_already_in_use=1, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node = \
            a_DatasetField.lookup(dataset.data[0][0], dont_fake_addr_from_all_irr_field__and__no_useful_suggestion_is_returned=True)
        assert for_sure_it_is_irr == False
        assert for_sure_it_is_NOT_irr
        assert result_or_suggest == dataset.data[0][1]
        assert found_in_addr == dataset.data[0][0]
        assert found_in_node == a_DatasetField
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "1"
        pass
    
    if "1 bit input, 1 free bit" and True:
        if "already checked" and True:
            dataset = Dataset(1, [(0b0,True), (0b1,True), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert -1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.input_bits_count
            assert a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert not a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.get_is_leaf()
            assert a_DatasetField.ready_for_lookup
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            for item in dataset.data:
                _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
                for_sure_it_is_irr = _temp_tuple_bbbio[0]
                for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
                result_or_suggest = _temp_tuple_bbbio[2]
                found_in_addr = _temp_tuple_bbbio[3]
                #for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,_ = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert for_sure_it_is_NOT_irr
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                pass
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            
            
            # a smallese non-leaf.
            # optimizable. it's equalavent to a 1bit xor, and possible to do some optimization. But let's keep it simple.
            dataset = Dataset(1, [(0b0,True), (0b1,False), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert 0 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 1 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert not a_DatasetField.get_is_leaf()
            assert a_DatasetField.ready_for_lookup
        
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            addr_1_child = a_DatasetField._get_child(true_or_false=True)
            assert not addr_1_child.all_irr
            assert 1 == addr_1_child.addr
            assert not addr_1_child.all_xor
            assert addr_1_child.when_xor__ignore_these_bits == 0
            assert -1 == addr_1_child.best_index_to_split_from_right_side
            assert 1 == addr_1_child.bitmask
            assert 1 == addr_1_child.bits_already_in_use
            assert addr_1_child.children is None
            assert 1 == addr_1_child.input_bits_count
            assert addr_1_child.get_already_const_without_irr()
            assert not addr_1_child.has_irr
            assert addr_1_child.has_0
            assert not addr_1_child.has_1
            assert addr_1_child.is_dataset_sorted
            assert addr_1_child.get_is_leaf()
            assert addr_1_child.ready_for_lookup
            
            readable_addr = addr_1_child.get_readable_addr()
            assert readable_addr == "1"
            
            addr_0_child = a_DatasetField._get_child(true_or_false=False)
            assert not addr_0_child.all_irr
            assert 0 == addr_0_child.addr
            assert not addr_0_child.all_xor
            assert addr_0_child.when_xor__ignore_these_bits == 0
            assert -1 == addr_0_child.best_index_to_split_from_right_side
            assert 1 == addr_0_child.bitmask
            assert 1 == addr_0_child.bits_already_in_use
            assert addr_0_child.children is None
            assert 1 == addr_0_child.input_bits_count
            assert addr_0_child.get_already_const_without_irr()
            assert not addr_0_child.has_irr
            assert not addr_0_child.has_0
            assert addr_0_child.has_1
            assert addr_0_child.is_dataset_sorted
            assert addr_0_child.get_is_leaf()
            assert addr_0_child.ready_for_lookup
            
            readable_addr = addr_0_child.get_readable_addr()
            assert readable_addr == "0"
            
            for item in dataset.data:
                _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
                for_sure_it_is_irr = _temp_tuple_bbbio[0]
                for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
                result_or_suggest = _temp_tuple_bbbio[2]
                found_in_addr = _temp_tuple_bbbio[3]
                assert for_sure_it_is_irr == False
                assert for_sure_it_is_NOT_irr
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                pass
                pass
            
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 2 == best_abs_of_num_of_same
            
            addr_1_child.dataset = dataset.get_subset(addr_1_child.bitmask, addr_1_child.addr)
            best_index_to_split, best_abs_of_num_of_same = addr_1_child._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            
            addr_0_child.dataset = dataset.get_subset(addr_0_child.bitmask, addr_0_child.addr)
            best_index_to_split, best_abs_of_num_of_same = addr_0_child._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            
            
            dataset = Dataset(1, [(0b0,False), (0b1,False), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert -1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.input_bits_count
            assert a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert not a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.get_is_leaf()
            assert a_DatasetField.ready_for_lookup
            
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            for item in dataset.data:
                _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
                for_sure_it_is_irr = _temp_tuple_bbbio[0]
                for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
                result_or_suggest = _temp_tuple_bbbio[2]
                found_in_addr = _temp_tuple_bbbio[3]
                #for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,_ = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert for_sure_it_is_NOT_irr
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                pass
                pass
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            
            
            
            # the other smallese non-leaf.
            # optimizable. it's equalavent to a 1bit xor, and possible to do some optimization. But let's keep it simple.
            dataset = Dataset(1, [(0b0,False), (0b1,True), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert 0 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 1 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert not a_DatasetField.get_is_leaf()
            assert a_DatasetField.ready_for_lookup
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            addr_1_child = a_DatasetField._get_child(true_or_false=True)
            assert not addr_1_child.all_irr
            assert 1 == addr_1_child.addr
            assert not addr_1_child.all_xor
            assert addr_1_child.when_xor__ignore_these_bits == 0
            assert -1 == addr_1_child.best_index_to_split_from_right_side
            assert 1 == addr_1_child.bitmask
            assert 1 == addr_1_child.bits_already_in_use
            assert addr_1_child.children is None
            assert 1 == addr_1_child.input_bits_count
            assert addr_1_child.get_already_const_without_irr()
            assert not addr_1_child.has_irr
            assert not addr_1_child.has_0
            assert addr_1_child.has_1
            assert addr_1_child.is_dataset_sorted
            assert addr_1_child.get_is_leaf()
            assert addr_1_child.ready_for_lookup
            readable_addr = addr_1_child.get_readable_addr()
            assert readable_addr == "1"
                
            addr_0_child = a_DatasetField._get_child(true_or_false=False)
            assert not addr_0_child.all_irr
            assert 0 == addr_0_child.addr
            assert not addr_0_child.all_xor
            assert addr_0_child.when_xor__ignore_these_bits == 0
            assert -1 == addr_0_child.best_index_to_split_from_right_side
            assert 1 == addr_0_child.bitmask
            assert 1 == addr_0_child.bits_already_in_use
            assert addr_0_child.children is None
            assert 1 == addr_0_child.input_bits_count
            assert addr_0_child.get_already_const_without_irr()
            assert not addr_0_child.has_irr
            assert addr_0_child.has_0
            assert not addr_0_child.has_1
            assert addr_0_child.is_dataset_sorted
            assert addr_0_child.get_is_leaf()
            assert addr_0_child.ready_for_lookup
            readable_addr = addr_0_child.get_readable_addr()
            assert readable_addr == "0"
            
            for item in dataset.data:
                _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
                for_sure_it_is_irr = _temp_tuple_bbbio[0]
                for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
                result_or_suggest = _temp_tuple_bbbio[2]
                found_in_addr = _temp_tuple_bbbio[3]
                #for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,_ = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert for_sure_it_is_NOT_irr
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                pass
                pass
            
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 2 == best_abs_of_num_of_same
            
            addr_1_child.dataset = dataset.get_subset(addr_1_child.bitmask, addr_1_child.addr)
            best_index_to_split, best_abs_of_num_of_same = addr_1_child._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            
            addr_0_child.dataset = dataset.get_subset(addr_0_child.bitmask, addr_0_child.addr)
            best_index_to_split, best_abs_of_num_of_same = addr_0_child._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            
        
            #partly irr
            dataset = Dataset(1, [(0b1,False), ])#both 0b0 and 0b1 pass the test.
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert -1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert not a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.get_is_leaf()
            assert a_DatasetField.ready_for_lookup
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            #relevant items.
            for item in dataset.data:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,found_in_node \
                    = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert for_sure_it_is_NOT_irr == False
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                assert found_in_node == a_DatasetField
                pass
            #plain irr
            _irr_addr = 0b0
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr)
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr == _irr_addr
            assert found_in_node == a_DatasetField
            
            #accurate irr
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True, \
                            leaf_keep_dataset=True)
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr, lookup_in_leaf_dataset=True)
            assert for_sure_it_is_irr == True
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr == _irr_addr
            assert found_in_node == a_DatasetField
            
            
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 1 == best_abs_of_num_of_same
            

            dataset = Dataset(1, [(0b0,True), ])#both 0b0 and 0b1 pass the test.
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert -1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert not a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.get_is_leaf()
            assert a_DatasetField.ready_for_lookup
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            #relevant items.
            for item in dataset.data:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,found_in_node \
                    = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert for_sure_it_is_NOT_irr == False
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                assert found_in_node == a_DatasetField
                pass
            
            #plain irr
            _irr_addr = 0b1
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr)
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr == _irr_addr
            assert found_in_node == a_DatasetField
            #accurate irr
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True, \
                            leaf_keep_dataset=True)
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr, lookup_in_leaf_dataset=True)
            assert for_sure_it_is_irr == True
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr == _irr_addr
            assert found_in_node == a_DatasetField

            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 1 == best_abs_of_num_of_same
            
            '''old code 
            #all irr
            dataset = Dataset(1, [])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert -1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert not a_DatasetField.has_0
            assert not a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.get_is_leaf()
            assert a_DatasetField.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            # for item in dataset:
            #     temp = a_DatasetField.lookup(item[0])
            #     assert item[1] == temp[1]
            #     pass
            for addr in range(0b10):
                temp = a_DatasetField.lookup(addr)
                assert temp[0]
                pass
            '''
            pass
        pass
        
    if "2 bits input, has 1,0 and irr" and True:
        if "already tested" and True:
            dataset = Dataset(2, [(0b00,True), (0b01,True), (0b10,False), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert 1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 2 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert not a_DatasetField.get_is_leaf()
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "__"
            
            addr_1__child = a_DatasetField._get_child(true_or_false=True)
            readable_addr = addr_1__child.get_readable_addr()
            assert readable_addr == "1_"
            assert addr_1__child.has_0
            assert addr_1__child.children is None
            
            addr_0__child = a_DatasetField._get_child(true_or_false=False)
            readable_addr = addr_0__child.get_readable_addr()
            assert readable_addr == "0_"
            assert addr_0__child.has_1
            assert not addr_0__child.has_irr
            assert addr_0__child.children is None
            
            #relevant items.
            for item in dataset.data:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,found_in_node \
                    = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                pass
                
            #plain irr
            _irr_addr = 0b11
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr)
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr == False
                
            #accurate irr
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True, \
                            leaf_keep_dataset=True)
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr, lookup_in_leaf_dataset=True)
            assert for_sure_it_is_irr == True
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr == _irr_addr
            
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 1 == best_index_to_split
            assert 3 == best_abs_of_num_of_same
            
            
            
            dataset = Dataset(2, [(0b00,True), (0b01,False), (0b10,True), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert 0 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 2 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert not a_DatasetField.get_is_leaf()
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "__"
            readable_as_tree = a_DatasetField.readable_as_tree() 
            assert a_DatasetField.readable_as_tree() == "__:(_0:1, _1:0+ir)"
            
            addr_1__child = a_DatasetField._get_child(true_or_false=True)
            readable_addr = addr_1__child.get_readable_addr()
            assert readable_addr == "_1"
            assert addr_1__child.has_0
            assert addr_1__child.children is None
            readable_as_tree = addr_1__child.readable_as_tree() 
            assert addr_1__child.readable_as_tree() == "_1:0+ir"
            
            addr_0__child = a_DatasetField._get_child(true_or_false=False)
            readable_addr = addr_0__child.get_readable_addr()
            assert readable_addr == "_0"
            assert addr_0__child.has_1
            assert not addr_0__child.has_irr
            assert addr_0__child.children is None
            readable_as_tree = addr_0__child.readable_as_tree() 
            assert addr_0__child.readable_as_tree() == "_0:1"
            
            #relevant items.
            for item in dataset.data:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,found_in_node \
                    = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                pass
                
            #plain irr
            _irr_addr = 0b11
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr)
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr == False
                
            #accurate irr
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True, \
                            leaf_keep_dataset=True)
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr, lookup_in_leaf_dataset=True)
            assert for_sure_it_is_irr == True
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr == _irr_addr
            
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 3 == best_abs_of_num_of_same
            
            
            
            #basically a xor, but replaced with 1 irr.
            dataset = Dataset(2, [(0b00,True), (0b01,False), (0b10,False), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert 1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 2 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert not a_DatasetField.get_is_leaf()
            assert a_DatasetField.get_readable_addr() == "__"
            readable_as_tree = a_DatasetField.readable_as_tree() 
            assert a_DatasetField.readable_as_tree() == "__:(0_:(00:1, 01:0), 1_:0+ir)"
            
            #assert not a_DatasetField.not_after_xor not important.
            
            addr_1__child = a_DatasetField._get_child(true_or_false=True)
            readable_addr = addr_1__child.get_readable_addr()
            assert readable_addr == "1_"
            assert addr_1__child.get_is_leaf()
            assert addr_1__child.has_0
            assert addr_1__child.has_irr
            readable_as_tree = addr_1__child.readable_as_tree() 
            assert addr_1__child.readable_as_tree() == "1_:0+ir"
            
            addr_0__child = a_DatasetField._get_child(true_or_false=False)
            readable_addr = addr_0__child.get_readable_addr()
            assert readable_addr == "0_"
            assert not addr_0__child.get_is_leaf()
            assert addr_0__child.has_1
            assert addr_0__child.has_0
            assert not addr_0__child.has_irr
            assert not addr_0__child.all_xor
            readable_as_tree = addr_0__child.readable_as_tree() 
            assert addr_0__child.readable_as_tree() == "0_:(00:1, 01:0)"
            
            addr_00_child = addr_0__child._get_child(true_or_false=False)
            readable_addr = addr_00_child.get_readable_addr()
            assert readable_addr == "00"
            assert addr_00_child.get_is_leaf()
            assert addr_00_child.has_1
            assert not addr_00_child.has_0
            assert not addr_00_child.has_irr
            readable_as_tree = addr_00_child.readable_as_tree() 
            assert addr_00_child.readable_as_tree() == "00:1"
            
            addr_01_child = addr_0__child._get_child(true_or_false=True)
            readable_addr = addr_01_child.get_readable_addr()
            assert readable_addr == "01"
            assert addr_01_child.get_is_leaf()
            assert not addr_01_child.has_1
            assert addr_01_child.has_0
            readable_as_tree = addr_01_child.readable_as_tree() 
            assert addr_01_child.readable_as_tree() == "01:0"
            
            #relevant items.
            for item in dataset.data:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,found_in_node \
                    = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                pass
                
            #plain irr
            _irr_addr = 0b11
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr)
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr == False
                
            #accurate irr
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True, \
                            leaf_keep_dataset=True)
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest, found_in_addr, found_in_node \
                = a_DatasetField.lookup(_irr_addr, lookup_in_leaf_dataset=True)
            assert for_sure_it_is_irr == True
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr == _irr_addr
            
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 1 == best_index_to_split
            assert 1 == best_abs_of_num_of_same
            
            pass
        
        if "2 bits xor(and xnor)" and True:
            #xor.
            dataset = Dataset(2, [(0b00,True), (0b01,False), (0b10,False), (0b11,True), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert a_DatasetField.best_index_to_split_from_right_side <0
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 2 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.get_is_leaf()
            assert a_DatasetField.not_after_xor
            assert a_DatasetField.get_readable_addr() == "__"
            assert a_DatasetField.readable_as_tree() == "__:xnor(irr-bits:..)"
            
            #relevant items.
            for item in dataset.data:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,found_in_node \
                    = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert for_sure_it_is_NOT_irr
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                assert found_in_node == a_DatasetField
                pass
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            
            
            
            #xnor. But in code it's a not after xor.
            dataset = Dataset(2, [(0b00,False), (0b01,True), (0b10,True), (0b11,False), ])
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert a_DatasetField.all_xor
            assert a_DatasetField.when_xor__ignore_these_bits == 0
            assert a_DatasetField.best_index_to_split_from_right_side <0
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 2 == a_DatasetField.input_bits_count
            assert not a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.get_is_leaf()
            assert not a_DatasetField.not_after_xor
            assert a_DatasetField.get_readable_addr() == "__"
            assert a_DatasetField.readable_as_tree() == "__:xor(irr-bits:..)"
            #relevant items.
            for item in dataset.data:
                for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,found_in_node \
                    = a_DatasetField.lookup(item[0])
                assert for_sure_it_is_irr == False
                assert for_sure_it_is_NOT_irr
                assert result_or_suggest == item[1]
                assert found_in_addr == item[0]
                assert found_in_node == a_DatasetField
                pass
            #manually modify the obj and test.
            a_DatasetField.dataset = dataset
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            
            pass
            
    if "3 bits fake xor" and True:
        '''Generally, xor is a 2 bits structure. The result is either 1001 or 0110. 
        People call the 1001 xnor, but in this tool, they are both xor.
        In this tool, more than 2 bits can also be treated as xor, to simplify and speed up.
        When input(a) is true, input(bc) is 1001, when input(a) is false, input(bc) is 0110, 
        they can combine as a 3 bits xor. But when 2 1001 combine, it's not a xor. 
        This 3 bits fake xor is to validate this special case.        
        '''
        max_input_bits = 3
        #dataset is 2 0110 combined.
        
        #dataset = Dataset(max_input_bits, [(0, False), (1, True), (2, True), (3, False), (4, False), (5, True), (6, True), (7, False)], True)
        dataset = Dataset.from_str("01100110")#(max_input_bits, [(0, False), (1, True), (2, True), (3, False), (4, False), (5, True), (6, True), (7, False)], True)
        a_DatasetField = DatasetField._new__and_valid(dataset,_debug__save_sub_dataset_when_xor = True)
        
        assert a_DatasetField.when_xor__ignore_these_bits == 0b100
        assert a_DatasetField.bitmask == 0
        assert a_DatasetField.addr == 0
        assert a_DatasetField.when_xor__ignore_these_bits == 0b100
        assert a_DatasetField.input_bits_count == 3
        assert a_DatasetField.bits_already_in_use == 0
        #assert a_DatasetField.dataset:list[tuple[int,bool]]
        assert a_DatasetField._debug__how_did_I_quit_init_func == How_did_I_quit_init_func.XOR
        assert a_DatasetField.best_index_to_split_from_right_side < 0 #ignored
        assert a_DatasetField._it_was_a_temp_var__best_abs_of_num_of_same == 0
        assert a_DatasetField.children is None
        assert a_DatasetField.ready_for_lookup
        assert a_DatasetField.has_1
        assert a_DatasetField.has_0
        assert a_DatasetField.has_irr == False
        #assert a_DatasetField.is_dataset_sorted 
        assert a_DatasetField.get_is_leaf()
        assert a_DatasetField.all_xor
        assert a_DatasetField.not_after_xor == False
        assert a_DatasetField.all_irr ==False
        assert a_DatasetField.readable_as_tree() == "___:xor(irr-bits:i..)"
        
        #relevant items.
        for item in dataset.data:
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,found_in_node \
                = a_DatasetField.lookup(item[0])
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr
            assert result_or_suggest == item[1]
            assert found_in_addr == item[0]
            assert found_in_node == a_DatasetField
            pass
        
        pass
        
        
        if "old" and True:
            assert a_DatasetField.bitmask == 0
            assert a_DatasetField.addr == 0
            assert a_DatasetField.input_bits_count == 3
            assert a_DatasetField.when_xor__ignore_these_bits == 0b100
            assert a_DatasetField.bits_already_in_use == 0
            #assert a_DatasetField.dataset:list[tuple[int,bool]]
            assert a_DatasetField._debug__how_did_I_quit_init_func == How_did_I_quit_init_func.XOR
            assert a_DatasetField.best_index_to_split_from_right_side <0
            assert a_DatasetField._it_was_a_temp_var__best_abs_of_num_of_same == 0
            assert a_DatasetField.children is None
            assert a_DatasetField.ready_for_lookup
            assert a_DatasetField.has_1
            assert a_DatasetField.has_0
            assert a_DatasetField.has_irr == False
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.get_is_leaf()# == False
            assert a_DatasetField.all_xor# == False
            assert a_DatasetField.not_after_xor == False
            assert a_DatasetField.all_irr ==False
            '''old code
            addr_1_child = a_DatasetField._get_child(true_or_false=True)
            assert addr_1_child.bitmask == 0b100
            assert addr_1_child.addr == 0b100
            assert addr_1_child.input_bits_count == 3
            assert addr_1_child.when_xor__ignore_these_bits == 0
            assert addr_1_child.bits_already_in_use == 1
            #assert addr_1_child.dataset:list[tuple[int,bool]]
            assert addr_1_child._debug__how_did_I_quit_init_func == How_did_I_quit_init_func.XOR
            assert addr_1_child.best_index_to_split_from_right_side == -1
            assert addr_1_child._it_was_a_temp_var__best_abs_of_num_of_same == 0
            assert addr_1_child.children is None
            assert addr_1_child.ready_for_lookup
            assert addr_1_child.has_1
            assert addr_1_child.has_0
            assert addr_1_child.has_irr == False
            assert addr_1_child.is_dataset_sorted
            assert addr_1_child.get_is_leaf() #== False
            assert addr_1_child.all_xor #== False
            assert addr_1_child.not_after_xor == False
            assert addr_1_child.all_irr ==False
            
            addr_0_child = a_DatasetField._get_child(true_or_false=False)
            assert addr_0_child.bitmask == 0b100
            assert addr_0_child.addr == 0b000
            assert addr_0_child.input_bits_count == 3
            assert addr_0_child.when_xor__ignore_these_bits == 0
            assert addr_0_child.bits_already_in_use == 1
            #assert addr_1_child.dataset:list[tuple[int,bool]]
            assert addr_0_child._debug__how_did_I_quit_init_func == How_did_I_quit_init_func.XOR
            assert addr_0_child.best_index_to_split_from_right_side == -1
            assert addr_0_child._it_was_a_temp_var__best_abs_of_num_of_same == 0
            assert addr_0_child.children is None
            assert addr_0_child.ready_for_lookup
            assert addr_0_child.has_1
            assert addr_0_child.has_0
            assert addr_0_child.has_irr == False
            assert addr_0_child.is_dataset_sorted
            assert addr_0_child.get_is_leaf() #== False
            assert addr_0_child.all_xor #== False
            assert addr_0_child.not_after_xor == False
            assert addr_0_child.all_irr ==False
            '''
            
            pass
        
        irr_addr_list_tuple = dataset.get_irr_addr___sorts_self()
        assert irr_addr_list_tuple[0]
        assert irr_addr_list_tuple[1].__len__() == 0
        # if irr_addr_list_tuple[1].__len__() != 0:
        #     for irr_addr in irr_addr_list_tuple[1]:
        #         temp = a_DatasetField.lookup(irr_addr)
        #         assert item[0]
        #         pass
        #     pass
        
    if "a 4 bits fake xor case" and True:
        '''0110 1111 1111 0110'''
        max_input_bits = 4
        dataset = Dataset.from_str('''0110 1111 1111 0110''')
        assert dataset.data == [(0, False), (1, True), (2, True), (3, False), (4, True), (5, True), (6, True), (7, True), (8, True), 
                (9, True), (10, True), (11, True), (12, False), (13, True), (14, True), (15, False)]
        a_DatasetField = DatasetField._new__and_valid(dataset)
        tree_str_TF = a_DatasetField.readable_as_tree(use_TF = True)
        assert tree_str_TF == "____:(0___:(00__:xor(irr-bits:00..), 01__:T), 1___:(10__:T, 11__:xor(irr-bits:11..)))"
        
        #relevant items.
        for item in dataset.data:
            
            
            _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
            for_sure_it_is_irr = _temp_tuple_bbbio[0]
            for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
            result_or_suggest = _temp_tuple_bbbio[2]
            found_in_addr = _temp_tuple_bbbio[3]
            
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr
            assert result_or_suggest == item[1]
            assert found_in_addr == item[0]
            pass
        #full.
        irr_addr_list_tuple = dataset.get_irr_addr___sorts_self()
        assert irr_addr_list_tuple[0]
        assert irr_addr_list_tuple[1].__len__() == 0
        pass
    
    if "insert bits to addr and check" and True:
        dataset = Dataset.from_str("1001,0000,0000,1001")
        dataset.add_const_bit_into_addr(2, True)
        assert dataset.max_input_bits == 5
        a_DatasetField = DatasetField._new__and_valid(dataset)
        tree_str_TF = a_DatasetField.readable_as_tree(use_TF=True)
        assert tree_str_TF == "_____:(__0__:ir, __1__:(0_1__:(001__:xnor(irr-bits:001..), 011__:F), 1_1__:(101__:F, 111__:xnor(irr-bits:111..))))"
        #relevant items.
        for item in dataset.data:
            _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
            for_sure_it_is_irr = _temp_tuple_bbbio[0]
            for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
            result_or_suggest = _temp_tuple_bbbio[2]
            found_in_addr = _temp_tuple_bbbio[3]
            # for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,_ \
            #     = a_DatasetField.lookup(item[0])
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr
            assert result_or_suggest == item[1]
            assert found_in_addr&0b11011 == item[0]&0b11011
            assert found_in_addr == item[0]|0b00100
            pass
        
        #plain irr
        _, irr_dataset = dataset.get_irr_addr___sorts_self()
        for irr_addr in irr_dataset:
            _temp_tuple_bbbio = a_DatasetField.lookup(irr_addr)
            for_sure_it_is_irr = _temp_tuple_bbbio[0]
            for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
            result_or_suggest = _temp_tuple_bbbio[2]
            found_in_addr = _temp_tuple_bbbio[3]
            
            # for_sure_it_is_irr, for_sure_it_is_NOT_irr, _, found_in_addr, _ \
            #     = a_DatasetField.lookup(_irr_addr)
            assert for_sure_it_is_irr #this case is very clear.
            assert for_sure_it_is_NOT_irr == False#a clear case.
            assert found_in_addr&0b11011 == irr_addr&0b11011
            assert found_in_addr == irr_addr|0b00100
            pass
        
        #accurate irr
        a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True, \
                        leaf_keep_dataset=True)
        for irr_addr in irr_dataset:
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, _, found_in_addr, found_in_node \
                = a_DatasetField.lookup(irr_addr)
            assert for_sure_it_is_irr#from the all-irr field
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr&0b11011 == irr_addr&0b11011
            assert found_in_addr == irr_addr|0b00100
            pass
        pass
    
    if "0110____" and True:
        dataset = Dataset.from_str("0110____")
        a_DatasetField = DatasetField._new__and_valid(dataset)
        tree_str_TF = a_DatasetField.readable_as_tree(use_TF=True)
        assert tree_str_TF == "___:(0__:xor(irr-bits:0..), 1__:ir)"
        #relevant items.
        for item in dataset.data:
            
            _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
            for_sure_it_is_irr = _temp_tuple_bbbio[0]
            for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
            result_or_suggest = _temp_tuple_bbbio[2]
            found_in_addr = _temp_tuple_bbbio[3]
            
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr
            assert result_or_suggest == item[1]
            assert found_in_addr&0b011 == item[0]&0b011
            pass
        
        #plain irr
        _,irr_dataset = dataset.get_irr_addr___sorts_self()
        for irr_addr in irr_dataset:
            
            _temp_tuple_bbbio = a_DatasetField.lookup(irr_addr)
            for_sure_it_is_irr = _temp_tuple_bbbio[0]
            for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
            
            found_in_addr = _temp_tuple_bbbio[3]
            
            assert for_sure_it_is_irr #this case is very clear.
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr&0b011 == irr_addr&0b011
            pass
        
        #accurate irr
        a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True, \
                        leaf_keep_dataset=True)
        for irr_addr in irr_dataset:
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, _, found_in_addr, found_in_node \
                = a_DatasetField.lookup(irr_addr)
            assert for_sure_it_is_irr
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr&0b011 == irr_addr&0b011
            pass
        pass
    
    if "10_1_1__" and True:
        dataset = Dataset.from_str("10_1_1__")
        a_DatasetField = DatasetField._new__and_valid(dataset)
        tree_str_TF = a_DatasetField.readable_as_tree(use_TF=True)
        assert tree_str_TF == "___:(0__:(00_:(000:T, 001:F), 01_:T+ir), 1__:T+ir)"
        #relevant items.
        for item in dataset.data:
            
            _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
            for_sure_it_is_irr = _temp_tuple_bbbio[0]
            for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
            result_or_suggest = _temp_tuple_bbbio[2]
            found_in_addr = _temp_tuple_bbbio[3]
            
            
            assert for_sure_it_is_irr == False
            #assert for_sure_it_is_NOT_irr this is not the only pure1 or pure0 case.
            assert result_or_suggest == item[1]
            assert found_in_addr == item[0]
            pass
        #accurate
        a_DatasetField = DatasetField(bitmask = 0, addr = 0, bits_already_in_use=0, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True, \
                        leaf_keep_dataset=True)
        #accurate relevant items.
        for item in dataset.data:
            
            
            _temp_tuple_bbbio = a_DatasetField.lookup(item[0])
            for_sure_it_is_irr = _temp_tuple_bbbio[0]
            for_sure_it_is_NOT_irr = _temp_tuple_bbbio[1]
            result_or_suggest = _temp_tuple_bbbio[2]
            found_in_addr = _temp_tuple_bbbio[3]
            
            
            assert for_sure_it_is_irr == False
            assert for_sure_it_is_NOT_irr
            assert result_or_suggest == item[1]
            assert found_in_addr == item[0]
            pass
        #accurate irr
        _, irr_dataset = dataset.get_irr_addr___sorts_self()
        for irr_addr in irr_dataset:
            for_sure_it_is_irr, for_sure_it_is_NOT_irr, _, found_in_addr, found_in_node \
                = a_DatasetField.lookup(irr_addr, lookup_in_leaf_dataset=True)
            assert for_sure_it_is_irr
            assert for_sure_it_is_NOT_irr == False
            assert found_in_addr == irr_addr
            pass
        pass
    
if "some special case" and True:
    input_bits_count = 7
    addr = 114
    bitmask = 114
    bits_already_in_use = 4
    #dataset_big = Dataset(input_bits_count, [(0, False), (2, True), (4, False), (9, True), (13, False), (17, True), (25, True), (26, False), (28, True), (29, False), (30, False), (31, True), (35, True), (37, True), (38, True), (39, False), (41, True), (42, False), (44, True), (46, False), (48, True), (50, True), (52, True), (53, True), (58, True), (60, True), (61, True), (62, True), (63, True), (64, False), (68, True), (71, False), (72, True), (74, True), (77, True), (78, True), (79, True), (83, True), (85, True), (87, True), (88, False), (92, True), (93, True), (95, True), (97, True), (99, True), (100, True), (103, True), (107, True), (110, True), (113, True), (117, True), (119, True), (121, True), (123, True), (124, True), (126, True), (127, False)])
    dataset = Dataset(input_bits_count, [(0, False), (2, True), (4, False), (9, True), (13, False), (17, True), (25, True), (26, False), (28, True), (29, False), (30, False), (31, True), (35, True), (37, True), (38, True), (39, False), (41, True), (42, False), (44, True), (46, False), (48, True), (50, True), (52, True), (53, True), (58, True), (60, True), (61, True), (62, True), (63, True), (64, False), (68, True), (71, False), (72, True), (74, True), (77, True), (78, True), (79, True), (83, True), (85, True), (87, True), (88, False), (92, True), (93, True), (95, True), (97, True), (99, True), (100, True), (103, True), (107, True), (110, True), (113, True), (117, True), (119, True), (121, True), (123, True), (124, True), (126, True), (127, False)])
    '''
    print(f"{addr:b}") #1110010
    for item in dataset:
        print(f"{item[0]:b}")
        pass
    1110111
    1111011
    1111110
    1111111
    '''
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)
    
    pass

if "special cases" and True:
    input_bits_count = 10
    dataset = Dataset(input_bits_count, [(2, True), (3, True), (4, True), (5, False), (7, False), (8, True), (9, True), (13, False), (15, True), (18, False), (19, True), (20, False), (22, False), (23, False), (24, False), (25, True), (26, True), (27, True), (29, True), (30, False), (31, False), (32, False), (33, False), (35, True), (36, False), (39, False), (40, True), (41, True), (42, False), (44, False), (46, True), (47, False), (48, False), (49, False), (50, True), (51, True), (54, True), (55, True), (59, True), (60, False), (61, True), (62, True), (63, False), (64, False), (65, True), (67, False), (69, True), (71, False), (74, False), (75, True), (80, False), (83, False), (85, False), (86, False), (88, False), (89, False), (90, False), (91, True), (94, False), (95, False), (97, False), (101, False), (102, False), (103, False), (104, True), (105, False), (107, False), (108, True), (109, True), (110, False), (111, True), (113, False), (115, True), (117, False), (118, False), (120, True), (121, True), (122, False), (123, True), (124, False), (125, True), (127, True), (128, False), (129, True), (131, True), (133, False), (134, True), (137, False), (138, True), (139, False), (141, False), (144, True), (145, False), (146, False), (147, True), (148, False), (149, False), (152, False), (154, False), (158, False), (159, True), (160, True), (163, False), (166, False), (167, False), (170, True), (171, True), (172, True), (174, False), (175, False), (183, True), (185, False), (188, True), (189, True), (190, False), (191, False), (193, False), (194, True), (195, True), (196, False), (197, True), (198, False), (199, False), (200, True), (201, True), (202, False), (203, True), (204, False), (205, False), (206, True), (207, True), (209, False), (211, False), (212, True), (213, False), (214, False), (216, True), (217, True), (218, False), (220, True), (221, True), (222, True), (223, True), (228, False), (229, False), (230, False), (231, False), (232, True), (233, False), (234, True), (235, False), (236, False), (240, True), (242, False), (243, False), (244, True), (245, True), (247, False), (255, True), (256, False), (258, True), (259, True), (260, False), (261, True), (263, False), (265, True), (267, False), (269, False), (271, True), (272, False), (274, True), (276, True), (279, False), (280, True), (281, True), (283, True), (284, True), (286, True), (287, True), (288, False), (289, True), (291, True), (292, True), (293, True), (294, False), (295, True), (296, True), (297, False), (300, False), (301, True), (303, False), (304, True), (306, False), (307, False), (308, True), (309, True), (310, True), (311, True), (313, False), (314, False), (316, True), (317, False), (318, True), (319, False), (322, True), (326, True), (327, True), (328, True), (329, False), (330, False), (331, False), (332, False), (333, False), (335, True), (337, False), (338, True), (341, True), (344, True), (346, False), (351, True), (355, False), (356, False), (358, False), (359, False), (360, True), (361, True), (362, False), (364, True), (365, True), (367, False), (369, True), (373, True), (374, False), (377, False), (378, False), (379, False), (380, True), (381, False), (382, True), (384, False), (385, True), (386, False), (387, False), (389, False), (390, True), (392, False), (393, True), (394, True), (395, True), (398, True), (400, True), (403, False), (404, True), (410, False), (412, False), (414, False), (415, True), (417, True), (418, False), (419, False), (421, False), (426, False), (427, False), (428, False), (432, False), (434, True), (436, False), (437, True), (440, False), (441, True), (442, True), (444, True), (446, False), (447, False), (449, True), (450, True), (451, True), (452, False), (453, True), (456, True), (457, False), (458, False), (459, False), (460, True), (462, True), (463, False), (468, False), (469, False), (470, True), (471, False), (472, True), (473, False), (474, False), (477, False), (479, False), (482, True), (483, False), (484, True), (485, False), (488, True), (489, False), (491, False), (493, True), (494, False), (495, True), (497, True), (498, True), (499, True), (501, False), (504, False), (509, False), (510, False), (511, True), (512, True), (513, False), (514, False), (515, True), (518, False), (519, False), (520, False), (521, False), (523, True), (524, False), (527, True), (529, True), (530, False), (532, True), (535, True), (536, True), (537, False), (539, True), (540, False), (543, True), (546, False), (548, False), (550, False), (551, False), (556, True), (557, False), (559, True), (560, False), (561, False), (563, False), (568, True), (569, False), (571, False), (572, True), (574, False), (575, False), (576, False), (577, True), (578, True), (581, True), (582, True), (586, True), (587, True), (588, True), (591, True), (592, True), (593, True), (594, True), (595, True), (596, True), (598, False), (599, True), (601, True), (602, False), (605, False), (606, False), (607, True), (608, True), (609, False), (610, True), (611, False), (615, False), (616, False), (618, False), (619, True), (621, True), (622, True), (623, True), (626, False), (628, False), (629, False), (631, False), (632, False), (636, True), (638, True), (641, True), (642, False), (643, True), (644, False), (645, True), (646, False), (647, False), (648, False), (649, False), (650, True), (652, True), (653, True), (654, False), (657, True), (663, True), (664, True), (665, True), (668, False), (670, True), (672, True), (673, False), (676, True), (678, True), (680, True), (681, True), (692, False), (693, False), (698, True), (699, True), (702, True), (703, True), (705, False), (714, False), (718, False), (721, False), (722, True), (724, False), (726, False), (727, True), (729, True), (730, True), (731, False), (732, True), (733, False), (734, True), (735, True), (739, True), (740, False), (741, False), (743, True), (744, True), (746, False), (747, True), (748, True), (751, False), (752, False), (754, False), (756, True), (757, True), (758, True), (759, False), (760, False), (762, True), (763, False), (765, False), (767, False), (768, False), (769, False), (772, False), (774, False), (775, True), (776, False), (777, True), (780, True), (781, False), (784, True), (786, True), (787, True), (789, False), (792, False), (793, True), (795, True), (796, False), (797, True), (798, True), (799, False), (800, False), (803, False), (804, False), (805, False), (806, True), (810, True), (812, True), (815, False), (816, False), (817, False), (819, True), (822, True), (825, False), (827, True), (829, False), (831, True), (835, True), (836, False), (837, False), (838, True), (839, False), (840, True), (841, True), (842, True), (843, True), (845, False), (848, False), (849, True), (850, True), (851, True), (852, False), (854, True), (855, False), (856, False), (861, True), (862, True), (863, False), (865, False), (869, False), (870, False), (871, True), (873, True), (874, False), (875, True), (879, False), (880, False), (881, True), (884, True), (889, True), (890, True), (896, True), (897, False), (898, True), (901, True), (902, True), (907, False), (916, False), (917, True), (919, True), (920, True), (923, False), (925, True), (926, False), (928, True), (930, True), (933, False), (936, True), (937, False), (939, False), (940, False), (941, True), (942, True), (945, True), (946, True), (949, False), (951, True), (956, True), (957, True), (960, False), (963, True), (967, False), (970, True), (977, True), (978, False), (979, False), (980, False), (981, True), (982, True), (983, False), (985, True), (988, False), (989, False), (994, False), (995, False), (996, False), (997, True), (998, False), (999, False), (1001, False), (1007, True), (1008, True), (1009, False), (1010, False), (1013, True), (1014, False), (1016, False), (1019, False), (1020, True), (1021, False), (1023, False)])
    a_small_set1 = dataset.get_subset(bitmask =1018, addr = 904)
    a_small_set2 = dataset.get_subset(bitmask =986, addr = 904)
    a_DatasetField = DatasetField._new(dataset)
    result_of_904 = a_DatasetField.lookup(904)#1111111111111111111w
    found, index = dataset.find_addr(936)
    assert found
    assert result_of_904[3] == 936
    assert result_of_904[2] == dataset.data[index][1]
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)
    '''
    fffff = readable_binary(1018)
    ffff = readable_binary(986)
    fff = readable_binary(904)
    split_here = []
    split_here.append(a_DatasetField.best_index_to_split_from_right_side)
    child = a_DatasetField._get_child(True)
    children = []
    children.append(child)
    split_here.append(child.best_index_to_split_from_right_side)
    child = child._get_child(True)
    children.append(child)
    split_here.append(child.best_index_to_split_from_right_side)
    child = child._get_child(True)
    children.append(child)
    split_here.append(child.best_index_to_split_from_right_side)
    child = child._get_child(False)
    children.append(child)
    split_here.append(child.best_index_to_split_from_right_side)
    child = child._get_child(False)
    children.append(child)
    split_here.append(child.best_index_to_split_from_right_side)
    child = child._get_child(True)
    children.append(child)
    split_here.append(child.best_index_to_split_from_right_side)
    child = child._get_child(False)
    children.append(child)
    '''
    
    
    input_bits_count = 4
    dataset = Dataset(input_bits_count, [(4, False), (5, True), (6, True), (7, False), (8, False), (9, False), (10, False), (11, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)
        
    input_bits_count = 11
    dataset = Dataset(input_bits_count, [(0, True), (5, True), (9, True), (10, False), (14, True), (18, False), (23, False), (26, False), (29, False), (35, False), (37, True), (38, False), (39, True), (46, True), (48, False), (53, False), (67, True), (80, False), (81, True), (82, False), (84, True), (93, True), (94, True), (96, True), (102, True), (107, True), (110, True), (115, True), (119, False), (123, True), (125, False), (129, True), (132, True), (134, True), (135, True), (146, False), (150, False), (151, True), (152, True), (153, True), (155, True), (163, False), (164, False), (167, True), (171, False), (172, False), (173, True), (177, False), (178, True), (192, True), (193, True), (199, True), (202, False), (204, False), (206, False), (215, False), (216, True), (221, False), (226, True), (228, True), (229, True), (232, True), (235, True), (239, False), (242, True), (249, True), (255, True), (261, False), (262, False), (266, True), (267, True), (268, True), (270, False), (271, True), (274, True), (276, False), (277, True), (289, True), (292, False), (296, True), (297, True), (304, True), (305, True), (306, False), (307, True), (308, True), (310, True), (314, True), (315, False), (334, True), (337, False), (338, True), (341, True), (342, True), (346, False), (352, True), (353, True), (356, True), (357, False), (365, True), (369, True), (370, True), (371, False), (383, True), (384, False), (385, True), (389, False), (393, False), (402, True), (404, False), (420, False), (430, False), (437, True), (445, False), (452, True), (455, True), (457, True), (458, False), (459, False), (464, False), (465, True), (466, True), (468, False), (469, True), (473, True), (475, True), (476, True), (487, False), (488, True), (489, True), (493, True), (494, True), (499, True), (500, True), (501, True), (504, False), (505, True), (510, True), (513, True), (517, True), (519, True), (520, False), (524, True), (526, False), (529, True), (530, True), (534, False), (536, True), (538, False), (544, False), (550, True), (554, True), (557, False), (560, True), (561, True), (563, True), (564, False), (565, False), (567, True), (571, False), (576, True), (578, True), (579, True), (581, False), (588, True), (589, False), (593, True), (595, False), (599, True), (600, True), (601, False), (603, False), (606, True), (609, False), (611, True), (612, True), (617, True), (630, True), (632, False), (641, True), (646, True), (647, True), (648, True), (654, True), (656, False), (660, False), (661, True), (666, True), (674, True), (678, True), (682, True), (687, False), (690, True), (696, False), (697, True), (698, True), (699, True), (701, True), (702, False), (704, True), (706, True), (707, True), (708, True), (709, True), (711, False), (715, True), (717, False), (722, True), (725, True), (734, False), (738, False), (744, True), (745, True), (746, False), (750, True), (753, True), (759, False), (760, False), (762, True), (766, False), (770, False), (773, False), (779, False), (782, True), (787, False), (790, False), (796, False), (807, False), (809, True), (810, True), (813, True), (815, True), (818, True), (822, True), (823, True), (827, False), (830, True), (831, False), (835, False), (837, True), (838, False), (851, False), (854, False), (855, True), (857, False), (861, True), (862, True), (863, True), (869, True), (874, False), (875, True), (876, False), (879, True), (880, True), (885, False), (886, True), (888, True), (891, True), (892, True), (895, True), (896, True), (898, False), (899, True), (903, False), (906, False), (907, False), (909, True), (910, False), (918, True), (922, True), (927, True), (932, True), (933, True), (936, False), (937, True), (938, True), (942, True), (943, True), (949, False), (953, True), (955, True), (957, True), (967, True), (969, False), (974, True), (990, True), (995, False), (1004, True), (1005, False), (1006, True), (1015, False), (1018, False), (1020, True), (1027, True), (1034, False), (1035, True), (1043, False), (1047, True), (1051, True), (1052, False), (1056, False), (1060, False), (1062, True), (1063, True), (1065, True), (1067, True), (1070, False), (1074, False), (1081, False), (1083, True), (1087, False), (1090, False), (1091, True), (1094, True), (1095, False), (1097, True), (1099, True), (1100, True), (1103, True), (1106, True), (1111, False), (1112, False), (1120, False), (1122, True), (1126, True), (1129, False), (1132, True), (1135, True), (1140, True), (1147, False), (1148, True), (1149, False), (1156, False), (1157, False), (1158, True), (1159, True), (1160, False), (1161, True), (1168, True), (1174, True), (1176, True), (1177, False), (1178, False), (1180, True), (1191, False), (1198, False), (1203, False), (1206, True), (1207, True), (1208, True), (1210, False), (1211, True), (1212, True), (1216, True), (1221, True), (1224, True), (1225, True), (1226, True), (1230, True), (1231, True), (1239, False), (1243, False), (1244, True), (1247, True), (1251, False), (1255, True), (1256, True), (1267, False), (1268, True), (1273, True), (1283, True), (1285, True), (1286, False), (1291, False), (1293, True), (1296, True), (1301, True), (1302, True), (1303, True), (1309, False), (1312, True), (1319, True), (1321, False), (1324, False), (1331, True), (1332, True), (1341, True), (1343, True), (1347, True), (1350, True), (1351, False), (1353, False), (1360, False), (1361, False), (1364, True), (1365, False), (1370, True), (1372, True), (1373, True), (1374, True), (1378, True), (1382, True), (1385, False), (1389, False), (1391, True), (1392, True), (1396, True), (1401, True), (1402, False), (1404, True), (1405, True), (1409, False), (1424, True), (1430, False), (1431, True), (1445, True), (1448, True), (1459, True), (1460, False), (1463, False), (1464, True), (1465, True), (1467, True), (1468, True), (1472, True), (1475, True), (1476, True), (1483, True), (1486, True), (1489, False), (1494, True), (1496, False), (1498, True), (1499, True), (1501, True), (1503, True), (1506, True), (1509, False), (1512, True), (1517, True), (1518, False), (1521, True), (1523, True), (1534, True), (1553, True), (1554, True), (1555, True), (1558, True), (1561, True), (1565, False), (1567, True), (1572, False), (1574, True), (1581, False), (1582, False), (1585, True), (1587, False), (1589, True), (1590, True), (1596, True), (1597, True), (1604, True), (1605, False), (1613, False), (1618, True), (1619, False), (1621, True), (1622, True), (1627, True), (1630, False), (1631, False), (1632, True), (1633, False), (1634, False), (1636, True), (1639, True), (1643, False), (1644, True), (1645, False), (1649, False), (1650, False), (1652, True), (1661, True), (1666, False), (1667, True), (1669, False), (1671, True), (1673, True), (1679, True), (1681, True), (1686, True), (1687, True), (1690, True), (1696, False), (1699, True), (1704, True), (1707, False), (1714, False), (1715, False), (1720, True), (1724, True), (1728, True), (1729, True), (1732, False), (1738, False), (1742, True), (1746, True), (1747, True), (1754, False), (1756, True), (1761, True), (1762, True), (1766, True), (1767, True), (1768, False), (1772, False), (1781, False), (1782, False), (1783, False), (1785, False), (1786, False), (1791, False), (1794, False), (1797, True), (1800, True), (1802, False), (1808, True), (1809, True), (1820, True), (1821, True), (1833, False), (1839, False), (1840, True), (1843, True), (1844, False), (1848, True), (1853, False), (1855, True), (1856, True), (1858, True), (1860, True), (1863, True), (1864, True), (1867, True), (1872, True), (1873, True), (1879, False), (1886, True), (1895, True), (1909, True), (1910, True), (1913, True), (1914, True), (1926, True), (1928, True), (1929, False), (1930, True), (1931, True), (1932, True), (1934, True), (1941, False), (1950, False), (1951, True), (1952, True), (1963, True), (1976, True), (1981, True), (1982, False), (1985, True), (1989, True), (1999, True), (2002, True), (2006, False), (2015, True), (2016, False), (2020, False), (2022, True), (2024, True), (2029, True), (2031, False), (2032, True), (2035, False), (2037, True), (2040, False), (2042, True), (2047, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)
    input_bits_count = 9
    dataset = Dataset(input_bits_count, [(0, True), (1, False), (2, False), (3, False), (4, False), (5, True), (6, True), (7, False), (10, False), (11, False), (12, False), (13, False), (14, True), (16, False), (17, False), (19, False), (20, False), (22, False), (23, True), (24, False), (26, False), (27, False), (28, False), (30, True), (31, False), (32, True), (33, False), (34, True), (35, True), (36, False), (37, True), (39, True), (41, False), (43, True), (45, True), (47, False), (48, False), (51, False), (52, True), (54, False), (55, False), (56, False), (57, False), (60, True), (61, False), (62, True), (64, True), (68, True), (69, True), (70, False), (74, False), (75, False), (76, False), (77, False), (80, False), (81, True), (82, False), (83, False), (84, True), (85, False), (86, True), (89, False), (90, False), (92, False), (93, False), (94, False), (95, False), (96, False), (97, True), (103, False), (104, False), (106, False), (107, False), (108, True), (109, False), (110, False), (112, False), (114, True), (116, False), (117, False), (118, False), (119, True), (120, False), (121, True), (122, False), (123, True), (125, False), (126, False), (127, False), (132, False), (133, False), (134, False), (136, False), (138, False), (140, False), (141, False), (142, True), (143, False), (144, False), (146, True), (147, False), (150, False), (151, True), (153, False), (155, False), (156, True), (157, True), (158, False), (159, True), (160, False), (161, False), (162, False), (163, True), (164, True), (165, False), (166, False), (167, False), (168, False), (170, True), (173, True), (174, False), (175, False), (176, False), (177, False), (179, False), (180, False), (182, False), (183, True), (184, False), (185, True), (189, True), (193, False), (194, True), (195, True), (197, False), (198, False), (199, False), (200, False), (202, True), (203, True), (205, False), (206, False), (207, False), (208, False), (209, True), (211, False), (213, False), (215, True), (216, False), (218, False), (221, True), (224, False), (225, True), (226, False), (228, False), (229, False), (231, True), (232, True), (233, False), (235, True), (236, False), (237, False), (238, False), (239, False), (240, False), (242, True), (243, False), (244, False), (245, True), (247, False), (248, False), (249, False), (250, False), (253, True), (255, False), (256, True), (257, True), (259, True), (260, True), (263, False), (266, False), (267, True), (270, False), (273, True), (274, False), (275, False), (276, True), (278, True), (279, True), (280, False), (283, False), (284, False), (285, False), (288, False), (292, True), (294, True), (295, False), (296, True), (297, False), (298, False), (299, True), (300, False), (301, False), (302, False), (304, True), (305, False), (306, False), (308, True), (309, True), (311, False), (312, False), (313, True), (314, True), (315, False), (316, True), (317, True), (320, False), (324, False), (325, True), (327, False), (328, True), (329, True), (330, False), (331, True), (332, True), (333, True), (334, True), (335, False), (337, True), (339, False), (340, False), (341, False), (345, False), (347, False), (348, False), (349, False), (350, True), (351, False), (352, False), (353, False), (354, False), (355, False), (356, True), (357, True), (358, False), (360, True), (362, False), (363, True), (364, False), (365, False), (366, True), (367, True), (369, True), (371, False), (373, False), (378, True), (379, False), (380, False), (382, False), (383, False), (384, False), (385, False), (387, False), (388, False), (389, True), (391, False), (392, False), (393, True), (394, False), (397, False), (398, False), (400, False), (401, True), (404, True), (405, True), (406, False), (409, False), (410, False), (411, False), (413, False), (415, False), (416, True), (418, True), (419, False), (420, False), (422, False), (423, False), (425, False), (426, True), (428, True), (429, False), (431, False), (432, True), (433, False), (435, False), (438, True), (440, True), (442, True), (444, False), (445, False), (448, False), (450, False), (451, False), (453, False), (454, True), (456, False), (458, True), (459, False), (460, False), (462, True), (464, False), (465, False), (466, True), (467, False), (468, True), (469, False), (470, False), (472, True), (473, False), (474, True), (475, False), (478, False), (480, False), (483, False), (484, True), (485, True), (486, True), (487, False), (488, False), (489, False), (490, True), (491, False), (492, True), (493, False), (494, False), (495, False), (496, False), (498, False), (501, False), (502, False), (503, True), (504, False), (507, False), (508, True), (509, True), (510, True), (511, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 11
    dataset = Dataset(input_bits_count, [(0, False), (1, True), (2, True), (4, True), (5, True), (7, False), (8, False), (9, True), (11, True), (12, False), (13, False), (15, False), (16, False), (18, False), (19, True), (21, True), (26, False), (27, True), (29, True), (31, True), (32, True), (33, False), (34, False), (36, False), (37, False), (38, False), (39, True), (40, True), (41, False), (42, False), (43, False), (44, True), (46, True), (48, True), (51, True), (53, True), (55, False), (57, False), (59, True), (60, True), (61, True), (62, True), (63, False), (64, False), (66, False), (67, False), (68, False), (69, False), (70, False), (72, False), (75, False), (76, True), (78, True), (79, False), (81, False), (82, True), (84, True), (85, False), (86, False), (87, False), (89, False), (90, False), (91, False), (92, True), (93, False), (94, True), (96, True), (97, False), (100, False), (104, True), (105, False), (107, True), (110, True), (111, True), (113, False), (114, False), (115, True), (116, True), (117, False), (119, False), (120, True), (121, False), (122, True), (126, False), (127, True), (130, True), (131, False), (132, True), (133, False), (134, False), (136, True), (137, True), (138, False), (139, False), (140, True), (141, False), (144, True), (149, False), (152, True), (153, True), (155, False), (157, True), (159, False), (161, True), (162, False), (163, False), (164, False), (165, False), (167, False), (168, False), (169, False), (170, False), (171, False), (172, True), (173, False), (175, False), (176, True), (177, True), (179, True), (180, False), (184, False), (185, False), (188, True), (189, False), (191, False), (192, True), (193, False), (194, False), (195, True), (196, False), (197, True), (198, False), (199, False), (200, True), (201, False), (202, False), (203, True), (204, False), (205, True), (206, False), (208, True), (209, False), (210, False), (214, True), (215, True), (217, True), (218, True), (219, True), (222, False), (223, False), (225, True), (227, True), (228, False), (229, True), (230, False), (232, True), (233, False), (235, True), (236, False), (237, True), (238, True), (239, False), (240, True), (241, True), (243, False), (244, True), (245, False), (247, True), (248, True), (250, False), (251, True), (252, True), (253, True), (254, True), (255, False), (256, False), (257, True), (258, True), (260, True), (261, False), (262, False), (263, True), (264, False), (265, False), (266, False), (267, True), (268, False), (270, True), (271, True), (272, False), (273, True), (276, True), (279, True), (281, True), (282, True), (283, False), (285, False), (286, False), (287, False), (289, False), (291, True), (293, False), (294, False), (295, False), (297, True), (300, False), (301, False), (302, False), (304, False), (305, False), (306, True), (307, True), (308, False), (309, False), (310, True), (311, False), (313, False), (314, True), (315, False), (316, True), (317, True), (318, False), (319, True), (320, False), (321, False), (322, True), (324, True), (325, True), (327, True), (329, True), (330, False), (331, True), (332, False), (333, True), (334, True), (339, True), (340, False), (341, True), (342, True), (343, True), (345, True), (347, True), (349, False), (351, True), (352, False), (353, False), (355, False), (356, True), (358, False), (360, True), (361, True), (362, True), (364, False), (366, True), (367, True), (369, True), (370, True), (371, False), (372, False), (373, True), (374, False), (376, True), (378, False), (381, True), (382, True), (384, True), (385, False), (386, True), (387, False), (389, True), (390, False), (391, False), (392, True), (393, True), (394, True), (395, False), (396, False), (397, False), (399, False), (400, True), (401, True), (402, True), (403, False), (404, True), (405, False), (406, True), (407, True), (408, True), (409, False), (413, False), (415, True), (416, True), (417, False), (418, True), (419, True), (422, False), (424, True), (425, False), (431, True), (432, True), (434, True), (435, True), (436, False), (437, True), (438, False), (439, False), (442, False), (443, True), (444, True), (448, False), (449, True), (450, True), (452, False), (453, False), (455, True), (457, False), (458, False), (459, False), (460, True), (462, True), (463, True), (466, False), (467, False), (471, False), (472, True), (473, False), (474, False), (475, False), (477, False), (478, False), (479, True), (480, False), (482, True), (484, True), (485, False), (486, False), (488, True), (489, True), (490, True), (493, True), (494, False), (495, False), (497, True), (498, True), (500, False), (502, True), (504, True), (507, True), (508, False), (509, False), (510, False), (512, True), (513, False), (517, True), (519, True), (520, True), (521, False), (522, False), (523, True), (524, False), (525, False), (528, True), (530, False), (531, False), (532, True), (533, False), (535, True), (536, True), (537, True), (538, True), (539, True), (540, False), (543, False), (548, False), (549, True), (550, True), (552, False), (553, True), (556, True), (561, True), (562, False), (563, False), (564, True), (565, False), (568, False), (569, False), (571, False), (573, False), (574, False), (575, True), (577, False), (578, True), (580, True), (581, True), (583, False), (585, True), (586, False), (587, False), (588, False), (591, False), (592, False), (593, False), (594, False), (596, True), (599, True), (600, True), (601, False), (602, False), (603, True), (604, False), (605, True), (606, True), (608, True), (610, False), (612, True), (615, True), (616, False), (617, False), (618, True), (619, True), (620, False), (621, False), (623, False), (624, False), (625, True), (627, True), (629, True), (630, False), (633, True), (634, False), (635, False), (636, True), (638, True), (641, False), (643, False), (644, True), (645, True), (646, True), (647, True), (649, True), (650, True), (651, True), (652, False), (653, False), (655, False), (658, False), (659, False), (660, True), (661, False), (662, True), (663, True), (665, False), (666, False), (667, False), (668, True), (669, False), (671, False), (675, False), (676, False), (677, False), (679, True), (680, False), (681, False), (682, True), (683, False), (684, True), (687, True), (688, True), (689, False), (691, True), (693, True), (695, True), (697, False), (698, False), (699, False), (700, True), (701, True), (702, False), (707, True), (708, True), (709, False), (710, False), (712, False), (713, False), (716, True), (717, False), (719, False), (720, False), (723, True), (724, True), (725, False), (726, False), (728, True), (730, True), (731, False), (732, True), (733, False), (735, True), (736, True), (737, True), (738, True), (740, True), (741, True), (743, False), (744, False), (746, True), (749, False), (752, False), (755, True), (756, False), (757, True), (758, False), (759, True), (760, True), (761, True), (762, True), (764, False), (765, False), (766, False), (767, False), (769, False), (770, True), (772, True), (773, False), (774, True), (775, True), (777, False), (778, True), (780, False), (781, False), (782, False), (783, True), (784, False), (786, True), (788, False), (789, True), (790, False), (791, True), (792, False), (793, True), (795, False), (796, True), (797, True), (799, False), (800, True), (801, False), (802, True), (804, False), (805, False), (806, True), (807, False), (808, True), (810, False), (812, True), (814, False), (815, False), (817, True), (820, False), (824, False), (825, False), (826, False), (827, False), (829, False), (830, False), (831, True), (832, False), (833, True), (834, False), (835, False), (837, False), (839, True), (840, True), (841, False), (842, False), (843, True), (845, True), (846, True), (847, True), (850, False), (851, False), (852, True), (853, True), (854, False), (855, True), (856, False), (857, True), (860, True), (861, False), (862, True), (866, False), (867, False), (869, False), (871, True), (873, True), (874, True), (875, False), (876, False), (877, False), (880, True), (881, True), (883, False), (884, False), (886, False), (891, True), (892, False), (893, True), (894, True), (897, False), (898, True), (899, True), (901, False), (903, True), (905, True), (906, True), (908, False), (909, False), (910, False), (912, True), (913, False), (916, False), (917, True), (918, True), (920, True), (921, False), (927, True), (930, False), (931, True), (933, True), (934, True), (940, True), (941, True), (944, False), (947, False), (948, False), (949, False), (952, False), (954, False), (955, True), (956, True), (957, False), (958, True), (960, False), (962, True), (963, False), (964, True), (965, True), (966, False), (967, True), (968, True), (969, True), (970, False), (972, False), (973, False), (974, True), (975, True), (976, True), (978, False), (980, False), (981, False), (983, False), (985, True), (986, True), (991, True), (993, False), (996, True), (997, False), (999, True), (1000, True), (1001, True), (1002, False), (1003, True), (1004, False), (1006, False), (1007, True), (1009, False), (1011, True), (1012, False), (1013, False), (1015, True), (1017, False), (1018, True), (1020, True), (1022, True), (1023, False), (1024, True), (1025, False), (1027, True), (1028, True), (1029, True), (1030, True), (1031, True), (1033, False), (1034, True), (1035, True), (1036, False), (1038, True), (1039, True), (1041, False), (1042, False), (1043, False), (1044, True), (1045, True), (1047, True), (1049, False), (1050, False), (1052, False), (1053, True), (1054, True), (1055, True), (1056, True), (1057, True), (1058, True), (1060, False), (1061, False), (1062, True), (1063, True), (1064, False), (1065, True), (1066, True), (1067, False), (1070, True), (1071, True), (1072, False), (1073, False), (1074, True), (1075, True), (1077, False), (1078, False), (1080, False), (1082, False), (1083, False), (1085, False), (1086, False), (1087, True), (1088, True), (1089, False), (1093, True), (1094, True), (1095, False), (1096, True), (1097, True), (1100, False), (1101, False), (1103, True), (1104, False), (1107, True), (1108, False), (1110, True), (1111, True), (1112, True), (1113, False), (1114, True), (1115, True), (1120, False), (1121, False), (1125, False), (1128, True), (1129, False), (1130, True), (1132, True), (1133, False), (1137, True), (1138, False), (1139, True), (1140, True), (1142, True), (1143, True), (1144, True), (1146, True), (1148, True), (1149, False), (1150, True), (1151, False), (1152, False), (1153, False), (1154, False), (1155, True), (1157, True), (1158, True), (1160, False), (1161, False), (1162, False), (1163, True), (1165, True), (1166, False), (1168, False), (1169, True), (1170, True), (1171, True), (1172, True), (1173, True), (1174, True), (1175, False), (1176, True), (1177, True), (1178, False), (1180, False), (1181, True), (1183, True), (1184, False), (1185, False), (1186, True), (1187, True), (1188, True), (1190, True), (1191, False), (1193, False), (1194, True), (1196, False), (1197, True), (1198, False), (1199, False), (1200, False), (1201, True), (1202, True), (1204, True), (1205, False), (1206, False), (1209, True), (1210, True), (1213, False), (1215, True), (1216, False), (1218, True), (1219, False), (1220, True), (1222, True), (1223, False), (1224, True), (1226, True), (1227, False), (1228, True), (1229, False), (1230, True), (1232, False), (1233, False), (1234, True), (1237, False), (1238, True), (1240, False), (1242, False), (1246, False), (1247, True), (1249, True), (1250, False), (1252, False), (1254, True), (1257, True), (1261, False), (1262, False), (1266, True), (1268, True), (1269, False), (1270, False), (1271, True), (1272, False), (1275, True), (1276, True), (1277, True), (1278, True), (1279, True), (1281, False), (1282, True), (1283, True), (1285, True), (1286, False), (1287, False), (1288, True), (1289, False), (1290, False), (1291, True), (1292, True), (1294, False), (1297, False), (1299, True), (1300, False), (1302, False), (1303, False), (1304, False), (1305, True), (1307, True), (1308, True), (1309, True), (1310, True), (1311, False), (1312, True), (1313, True), (1314, False), (1316, False), (1317, True), (1320, False), (1321, False), (1322, False), (1323, False), (1324, True), (1325, False), (1326, False), (1327, True), (1328, True), (1330, False), (1331, False), (1332, False), (1337, True), (1341, True), (1342, True), (1344, False), (1345, False), (1347, False), (1349, True), (1353, True), (1354, False), (1355, True), (1356, True), (1360, False), (1361, False), (1362, True), (1363, True), (1364, True), (1366, True), (1368, True), (1369, False), (1371, True), (1373, True), (1374, True), (1375, True), (1376, True), (1378, True), (1380, False), (1381, False), (1383, False), (1384, False), (1385, True), (1387, False), (1391, True), (1394, True), (1395, False), (1396, True), (1397, True), (1400, True), (1401, False), (1403, True), (1405, True), (1406, True), (1408, False), (1409, False), (1410, False), (1413, True), (1415, False), (1416, False), (1417, False), (1418, True), (1419, False), (1420, False), (1423, False), (1424, False), (1425, True), (1426, False), (1429, True), (1430, False), (1432, False), (1434, True), (1435, True), (1436, True), (1437, True), (1438, False), (1439, False), (1440, True), (1441, True), (1445, True), (1447, False), (1448, False), (1449, False), (1450, False), (1451, False), (1452, False), (1455, True), (1457, True), (1458, True), (1459, True), (1460, False), (1461, False), (1463, False), (1464, False), (1465, False), (1467, False), (1468, True), (1470, False), (1471, False), (1475, True), (1476, False), (1477, True), (1478, False), (1480, True), (1481, False), (1482, False), (1488, False), (1489, False), (1490, True), (1491, False), (1496, False), (1497, False), (1498, True), (1499, False), (1500, True), (1501, False), (1505, False), (1506, False), (1507, True), (1508, False), (1509, True), (1510, True), (1511, False), (1512, True), (1513, False), (1514, False), (1517, True), (1518, True), (1519, False), (1520, True), (1521, False), (1522, True), (1523, True), (1524, True), (1525, True), (1527, False), (1529, False), (1530, False), (1532, False), (1533, True), (1535, True), (1537, False), (1539, True), (1540, True), (1541, True), (1546, True), (1547, False), (1550, True), (1552, True), (1555, False), (1556, True), (1557, False), (1560, False), (1561, True), (1564, False), (1565, False), (1566, True), (1567, False), (1568, True), (1570, False), (1571, True), (1572, True), (1573, True), (1576, False), (1578, False), (1579, False), (1580, True), (1581, True), (1583, True), (1584, True), (1585, False), (1586, True), (1589, True), (1590, True), (1591, True), (1593, False), (1595, True), (1596, False), (1597, False), (1598, False), (1602, True), (1603, True), (1604, True), (1605, False), (1606, False), (1607, True), (1608, False), (1609, True), (1610, True), (1611, False), (1612, True), (1614, False), (1615, False), (1617, False), (1618, True), (1619, False), (1620, False), (1622, False), (1623, True), (1624, False), (1626, True), (1627, True), (1628, False), (1629, False), (1634, False), (1635, True), (1638, True), (1639, False), (1640, False), (1644, True), (1645, True), (1648, True), (1649, True), (1650, True), (1651, True), (1652, True), (1656, True), (1658, True), (1659, False), (1660, False), (1663, True), (1664, True), (1665, True), (1666, False), (1667, False), (1668, True), (1669, False), (1670, True), (1671, True), (1674, True), (1676, False), (1679, True), (1680, False), (1682, True), (1683, True), (1684, False), (1685, False), (1686, True), (1687, True), (1689, True), (1690, False), (1691, True), (1694, False), (1695, True), (1696, False), (1697, False), (1698, False), (1700, False), (1701, True), (1702, True), (1703, True), (1706, False), (1707, False), (1709, False), (1711, True), (1712, True), (1715, False), (1716, True), (1717, False), (1718, True), (1719, False), (1720, True), (1721, True), (1723, False), (1724, False), (1728, False), (1729, False), (1730, False), (1732, False), (1734, False), (1735, True), (1736, False), (1737, True), (1738, False), (1739, False), (1741, True), (1742, True), (1745, True), (1746, True), (1747, True), (1748, False), (1750, True), (1755, False), (1757, False), (1758, True), (1759, False), (1760, False), (1761, False), (1764, False), (1765, True), (1766, False), (1767, False), (1768, False), (1769, True), (1772, False), (1773, False), (1774, False), (1776, True), (1777, False), (1778, True), (1780, True), (1781, False), (1782, False), (1783, False), (1785, False), (1786, True), (1787, False), (1788, True), (1789, False), (1793, True), (1795, True), (1796, True), (1799, False), (1800, True), (1802, False), (1804, False), (1805, False), (1806, True), (1807, True), (1808, False), (1810, False), (1811, True), (1814, False), (1815, True), (1816, False), (1818, True), (1819, True), (1821, True), (1822, True), (1823, False), (1824, True), (1826, True), (1827, False), (1828, False), (1829, False), (1831, False), (1833, False), (1835, True), (1836, True), (1837, True), (1838, True), (1839, False), (1840, False), (1841, True), (1842, False), (1846, True), (1847, False), (1848, True), (1849, False), (1850, False), (1851, True), (1852, False), (1853, False), (1857, False), (1859, True), (1864, False), (1866, True), (1867, True), (1868, False), (1869, True), (1870, True), (1872, False), (1873, True), (1874, False), (1875, True), (1876, True), (1879, True), (1880, False), (1881, True), (1884, True), (1886, False), (1887, True), (1891, True), (1894, True), (1895, True), (1898, False), (1899, False), (1900, False), (1902, True), (1903, True), (1904, True), (1906, True), (1909, True), (1910, True), (1911, False), (1912, False), (1913, False), (1914, False), (1915, True), (1916, True), (1917, False), (1918, False), (1921, True), (1923, True), (1924, False), (1925, True), (1929, True), (1930, True), (1933, True), (1934, False), (1935, False), (1937, True), (1938, True), (1940, False), (1941, False), (1942, True), (1943, True), (1945, True), (1946, False), (1947, True), (1948, True), (1950, True), (1951, True), (1953, True), (1954, False), (1955, False), (1957, True), (1960, True), (1961, False), (1963, False), (1964, True), (1965, False), (1966, True), (1968, False), (1969, False), (1970, False), (1971, True), (1972, False), (1973, True), (1975, True), (1976, True), (1978, False), (1979, True), (1980, False), (1981, True), (1982, True), (1985, False), (1986, True), (1987, True), (1988, False), (1989, False), (1990, True), (1991, False), (1993, True), (1994, True), (1995, True), (1996, False), (1997, False), (1998, True), (2000, True), (2001, True), (2002, False), (2006, False), (2007, False), (2008, True), (2010, True), (2011, False), (2012, True), (2013, False), (2014, False), (2015, False), (2018, False), (2019, True), (2020, False), (2023, False), (2026, True), (2027, False), (2028, False), (2030, False), (2032, True), (2033, True), (2034, True), (2036, True), (2037, False), (2043, True), (2045, True), (2047, True)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 10
    dataset = Dataset(input_bits_count, [(3, True), (7, True), (8, True), (9, True), (10, True), (12, True), (17, True), (31, False), (32, True), (42, True), (47, True), (50, True), (53, False), (56, True), (59, False), (60, True), (65, True), (67, True), (76, True), (82, True), (86, True), (92, True), (101, True), (111, True), (113, False), (116, True), (118, True), (119, True), (123, True), (135, True), (137, False), (139, True), (141, False), (142, False), (147, True), (157, False), (161, True), (167, True), (170, False), (173, True), (176, False), (178, True), (183, True), (184, True), (190, True), (197, False), (198, True), (203, False), (204, True), (205, True), (209, True), (213, True), (218, True), (221, True), (223, False), (232, True), (242, True), (247, True), (248, True), (249, True), (253, True), (256, True), (262, False), (264, True), (267, False), (272, False), (275, False), (277, True), (288, True), (295, True), (297, True), (304, False), (306, True), (309, False), (313, False), (317, True), (323, True), (329, False), (334, False), (338, True), (346, True), (350, True), (358, True), (367, True), (371, True), (374, True), (380, True), (389, True), (390, True), (393, True), (395, True), (400, True), (406, True), (408, True), (410, True), (412, True), (416, False), (419, True), (421, True), (424, True), (425, False), (430, False), (434, False), (436, False), (439, False), (441, False), (443, True), (449, True), (452, True), (454, False), (456, True), (457, True), (460, False), (474, False), (478, False), (482, True), (485, True), (486, True), (497, True), (498, False), (500, False), (501, False), (511, True), (512, True), (515, True), (520, False), (527, False), (530, True), (531, True), (532, True), (535, False), (536, True), (541, True), (543, True), (546, True), (551, True), (553, False), (561, True), (567, True), (570, True), (576, True), (578, True), (581, False), (583, True), (585, True), (586, True), (590, True), (591, True), (604, True), (607, False), (620, True), (621, False), (622, False), (623, True), (626, False), (632, True), (636, True), (639, False), (640, True), (643, True), (647, True), (648, True), (649, True), (652, True), (658, True), (661, True), (663, True), (665, True), (670, True), (671, True), (682, True), (683, False), (696, False), (698, True), (700, False), (703, True), (705, False), (715, False), (716, True), (727, True), (728, True), (732, True), (739, True), (742, False), (743, True), (744, False), (752, True), (753, True), (755, True), (762, True), (765, False), (766, True), (767, True), (771, True), (773, True), (774, False), (779, True), (781, True), (786, True), (796, True), (797, False), (799, True), (802, False), (804, True), (806, True), (812, True), (813, True), (820, False), (825, True), (830, True), (831, True), (835, True), (841, True), (845, True), (846, True), (848, True), (850, True), (860, False), (868, True), (869, False), (872, True), (877, True), (885, True), (886, False), (890, True), (896, False), (898, True), (899, True), (900, False), (901, True), (904, True), (908, True), (910, False), (919, True), (921, True), (926, True), (928, True), (930, True), (936, True), (937, True), (944, True), (948, False), (949, False), (952, False), (957, False), (962, True), (966, False), (968, True), (969, False), (972, False), (973, True), (979, True), (984, True), (985, True), (986, False), (995, False), (999, True), (1006, True), (1013, True), (1016, True), (1017, True), (1018, True), (1019, False), (1021, True)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 4
    dataset = Dataset(input_bits_count, [(4, False), (5, True), (6, True), (7, False), (8, False), (9, False), (10, False), (11, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 10
    dataset = Dataset(input_bits_count, [(0, False), (1, False), (2, True), (3, True), (4, True), (5, False), (7, True), (8, True), (10, False), (12, True), (14, True), (17, True), (18, False), (19, False), (21, True), (22, True), (25, False), (26, False), (28, True), (29, False), (30, False), (31, True), (32, True), (33, True), (35, False), (36, True), (37, True), (39, False), (40, True), (45, False), (49, True), (50, True), (53, False), (54, True), (55, True), (57, False), (58, False), (59, True), (60, True), (62, True), (63, True), (64, True), (65, True), (66, True), (67, False), (68, True), (70, True), (74, True), (75, True), (78, False), (80, True), (81, False), (82, True), (83, True), (84, True), (86, False), (87, True), (88, True), (89, False), (90, True), (91, True), (92, False), (93, True), (94, True), (96, False), (97, False), (98, True), (99, True), (101, True), (102, False), (103, False), (104, True), (107, True), (109, False), (111, False), (113, True), (114, False), (115, False), (116, True), (117, True), (119, False), (121, False), (122, False), (124, False), (125, True), (126, False), (127, False), (128, False), (129, True), (130, False), (132, False), (133, True), (135, True), (136, False), (137, True), (139, False), (140, True), (141, False), (142, True), (145, False), (146, True), (147, True), (148, True), (150, True), (155, False), (156, True), (157, False), (159, True), (163, True), (167, True), (169, False), (170, False), (171, False), (172, False), (173, False), (175, False), (178, False), (179, True), (180, False), (181, True), (183, True), (184, False), (185, False), (188, True), (190, True), (192, False), (193, True), (194, False), (195, True), (196, True), (197, False), (199, False), (200, True), (202, False), (203, False), (204, False), (205, True), (206, False), (208, True), (209, True), (210, True), (211, False), (212, True), (213, True), (215, True), (216, False), (217, True), (218, True), (219, True), (220, True), (221, True), (222, False), (223, True), (225, False), (226, False), (228, False), (229, True), (230, False), (232, False), (233, True), (234, True), (235, False), (236, False), (238, True), (239, True), (240, False), (241, True), (242, False), (246, False), (247, True), (248, False), (250, True), (251, True), (252, True), (254, True), (255, True), (256, True), (257, True), (258, True), (259, False), (260, False), (261, True), (263, False), (265, True), (266, True), (267, True), (268, False), (269, False), (270, False), (272, False), (273, False), (274, True), (275, True), (276, False), (277, True), (279, False), (280, True), (281, True), (282, True), (283, False), (284, True), (285, False), (287, True), (288, True), (289, False), (290, True), (291, True), (292, False), (293, True), (294, False), (295, True), (296, True), (298, False), (299, False), (301, False), (302, True), (303, False), (304, True), (305, True), (307, True), (308, False), (312, True), (313, False), (314, False), (315, True), (317, False), (318, True), (320, True), (321, False), (322, True), (323, True), (325, True), (326, True), (327, True), (328, True), (329, True), (331, True), (332, False), (334, True), (335, False), (336, False), (337, True), (338, True), (339, True), (341, True), (342, True), (343, False), (344, True), (345, False), (346, True), (347, True), (348, True), (349, False), (350, False), (351, True), (352, False), (353, True), (354, True), (355, True), (356, False), (358, True), (359, False), (361, False), (362, True), (366, True), (367, True), (368, False), (369, True), (370, False), (371, True), (373, False), (374, False), (375, False), (376, True), (377, False), (378, True), (379, False), (380, False), (382, True), (383, True), (384, False), (385, False), (386, False), (387, False), (388, True), (391, True), (393, False), (394, False), (396, True), (397, False), (400, False), (401, True), (403, True), (404, False), (406, False), (407, True), (408, True), (410, True), (411, False), (412, True), (413, True), (414, True), (415, True), (416, False), (417, True), (418, False), (419, False), (422, True), (423, False), (424, True), (425, False), (427, False), (428, False), (430, True), (431, False), (432, False), (433, False), (434, True), (435, True), (436, False), (437, True), (439, False), (440, True), (441, False), (442, False), (444, False), (445, True), (446, True), (447, True), (448, True), (449, True), (450, True), (451, False), (452, False), (453, True), (454, True), (455, True), (456, False), (457, False), (458, False), (459, True), (461, False), (463, True), (464, False), (466, True), (468, False), (470, False), (471, True), (472, True), (473, True), (474, False), (475, False), (476, True), (477, True), (478, True), (479, False), (480, True), (481, True), (483, True), (484, True), (485, True), (487, False), (488, True), (490, True), (491, False), (492, True), (493, False), (494, True), (495, False), (496, False), (497, False), (499, False), (500, False), (501, True), (503, True), (506, False), (507, True), (510, False), (511, False), (512, False), (514, True), (515, True), (516, True), (518, True), (519, True), (520, True), (521, False), (522, False), (523, False), (525, True), (528, True), (529, True), (530, False), (531, True), (532, True), (533, False), (534, False), (535, False), (536, False), (537, True), (538, False), (540, True), (541, True), (542, True), (543, True), (545, False), (546, True), (547, True), (548, True), (549, True), (550, True), (551, True), (553, True), (556, False), (557, True), (558, False), (559, False), (560, True), (561, False), (562, False), (564, False), (566, True), (567, True), (568, False), (569, True), (572, True), (573, False), (574, False), (577, True), (579, False), (580, False), (581, True), (583, False), (584, False), (586, True), (588, True), (589, False), (590, True), (591, False), (592, False), (593, False), (594, True), (596, True), (597, False), (598, True), (599, False), (600, True), (601, False), (602, False), (603, False), (605, True), (607, True), (608, True), (611, True), (612, False), (613, False), (614, False), (616, True), (617, False), (619, True), (620, True), (622, False), (624, False), (625, True), (626, False), (628, True), (630, False), (633, False), (634, False), (635, False), (637, True), (639, True), (640, True), (642, True), (643, True), (645, False), (646, True), (647, False), (649, False), (650, False), (651, False), (652, False), (653, True), (654, False), (655, False), (656, True), (657, True), (659, True), (660, False), (662, True), (664, True), (665, True), (666, False), (667, True), (668, False), (669, True), (670, True), (671, True), (672, True), (673, False), (674, False), (675, True), (676, False), (677, False), (678, True), (679, False), (680, True), (682, False), (683, True), (684, True), (685, False), (686, True), (687, False), (688, False), (689, True), (691, True), (693, False), (695, True), (696, True), (697, False), (698, True), (699, False), (700, True), (701, False), (702, True), (703, False), (704, False), (705, True), (706, True), (707, False), (708, True), (709, True), (710, True), (711, True), (712, True), (713, True), (714, True), (715, False), (716, False), (717, False), (718, False), (720, True), (721, True), (722, True), (724, False), (725, False), (726, False), (727, False), (728, False), (729, False), (730, False), (733, False), (735, True), (736, True), (737, True), (738, False), (740, True), (741, True), (742, True), (743, True), (744, True), (745, True), (746, False), (747, False), (748, True), (751, False), (752, True), (753, False), (754, False), (755, True), (756, False), (757, True), (758, True), (759, False), (760, False), (761, True), (762, True), (764, False), (766, False), (767, True), (768, False), (769, False), (771, True), (772, True), (773, True), (776, True), (778, True), (779, True), (780, False), (781, False), (782, False), (783, True), (784, False), (786, False), (787, True), (789, True), (790, True), (791, True), (792, True), (793, False), (794, True), (797, False), (798, False), (799, True), (800, True), (802, True), (804, False), (805, True), (806, False), (807, True), (808, True), (809, False), (810, False), (811, False), (812, True), (813, True), (816, True), (817, False), (818, False), (819, False), (821, False), (822, False), (823, False), (824, False), (825, False), (828, False), (829, False), (830, True), (831, True), (832, True), (833, True), (834, False), (835, True), (836, True), (837, False), (838, False), (839, True), (840, False), (841, True), (842, False), (844, True), (845, True), (846, False), (847, True), (848, True), (850, False), (851, False), (852, False), (854, True), (855, True), (856, True), (857, False), (858, True), (859, True), (860, True), (861, False), (863, True), (864, True), (866, True), (868, True), (870, False), (871, False), (872, True), (873, True), (874, True), (875, False), (876, False), (877, True), (878, True), (880, True), (882, False), (883, False), (884, False), (885, False), (886, True), (887, True), (888, True), (889, False), (890, False), (891, False), (892, True), (893, True), (894, True), (895, False), (896, True), (897, True), (898, False), (900, True), (901, True), (902, True), (903, True), (904, False), (905, False), (906, False), (907, True), (908, False), (909, True), (910, False), (912, True), (914, True), (915, True), (919, False), (921, True), (923, False), (924, True), (925, False), (926, False), (927, False), (928, True), (929, False), (930, True), (931, False), (933, True), (934, True), (935, False), (936, True), (937, False), (938, True), (940, True), (941, False), (942, False), (944, False), (946, False), (947, False), (948, True), (949, False), (950, False), (953, False), (955, False), (957, True), (958, False), (959, True), (960, True), (961, False), (962, True), (964, True), (966, False), (967, False), (969, False), (972, False), (974, False), (975, False), (976, False), (977, False), (978, False), (979, False), (981, False), (983, True), (984, False), (985, True), (986, True), (987, True), (988, True), (989, True), (990, False), (991, True), (992, True), (993, False), (994, False), (996, True), (997, False), (998, True), (1001, False), (1002, False), (1003, True), (1004, False), (1006, False), (1007, True), (1008, False), (1016, False), (1017, True), (1019, True), (1021, True), (1023, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 11
    dataset = Dataset(input_bits_count, [(1, True), (2, True), (3, True), (4, True), (5, True), (6, True), (7, True), (8, False), (9, True), (10, True), (11, True), (12, True), (13, True), (14, True), (15, True), (17, True), (18, False), (20, True), (21, True), (22, True), (23, True), (24, False), (25, True), (26, False), (27, True), (29, True), (30, False), (31, True), (35, True), (36, True), (38, False), (39, True), (40, True), (41, True), (46, True), (47, True), (48, True), (50, False), (51, True), (52, False), (54, False), (55, True), (56, True), (57, True), (58, True), (59, True), (60, True), (61, True), (62, True), (63, True), (67, True), (68, True), (69, True), (70, True), (71, True), (72, True), (73, True), (76, True), (77, True), (78, True), (79, True), (80, True), (81, True), (83, True), (84, True), (85, True), (86, True), (87, True), (88, True), (89, True), (91, True), (92, True), (94, True), (95, False), (96, True), (97, False), (98, True), (101, True), (102, True), (103, True), (104, True), (106, True), (107, True), (108, True), (109, True), (110, True), (112, True), (113, True), (114, True), (115, True), (116, True), (117, True), (118, False), (119, True), (121, True), (122, True), (123, True), (125, True), (126, True), (127, True), (129, True), (130, True), (131, True), (132, True), (135, True), (136, False), (138, True), (140, True), (141, True), (142, False), (143, True), (144, True), (145, False), (146, True), (150, True), (151, False), (153, True), (154, False), (156, False), (160, False), (161, True), (162, True), (164, False), (165, True), (166, True), (167, True), (168, True), (169, True), (170, True), (171, True), (172, True), (173, True), (174, True), (175, False), (176, False), (178, True), (179, True), (180, True), (181, True), (183, True), (185, True), (187, True), (188, True), (189, True), (192, True), (194, False), (195, True), (196, False), (199, True), (200, True), (201, True), (203, True), (204, True), (205, True), (207, True), (208, True), (209, True), (211, False), (212, True), (213, False), (216, True), (217, True), (219, True), (220, True), (221, True), (223, False), (226, True), (227, True), (228, True), (229, False), (230, True), (232, True), (233, True), (234, False), (235, True), (236, False), (237, True), (238, True), (241, True), (243, True), (244, True), (246, True), (247, True), (248, True), (249, True), (251, True), (252, True), (255, True), (257, True), (258, False), (262, True), (263, True), (264, True), (265, True), (267, False), (268, True), (269, True), (271, False), (272, True), (273, True), (274, False), (275, False), (276, True), (277, True), (278, True), (282, True), (283, False), (284, False), (286, True), (287, False), (289, True), (290, True), (291, True), (294, True), (295, False), (296, False), (297, True), (298, True), (299, True), (301, True), (302, True), (303, True), (304, True), (305, True), (306, True), (308, True), (309, False), (310, True), (311, True), (312, True), (313, True), (314, True), (315, False), (316, True), (317, True), (318, True), (320, True), (322, True), (323, True), (325, True), (326, False), (327, True), (329, True), (330, True), (331, True), (332, False), (334, True), (335, True), (337, False), (338, True), (339, True), (341, True), (344, True), (346, True), (347, True), (348, True), (350, True), (351, True), (352, True), (353, True), (354, True), (355, True), (356, False), (357, True), (358, True), (359, True), (360, False), (362, False), (364, True), (365, True), (366, False), (367, False), (368, True), (369, True), (370, True), (371, True), (374, True), (375, True), (376, False), (378, True), (379, True), (381, True), (382, False), (383, True), (384, True), (386, True), (387, False), (388, True), (389, True), (390, True), (391, True), (392, True), (393, True), (394, True), (395, False), (396, True), (399, True), (400, False), (401, True), (402, True), (403, False), (404, True), (405, True), (406, True), (407, True), (408, True), (409, True), (410, True), (411, False), (412, False), (413, True), (414, True), (415, True), (416, True), (418, True), (419, True), (420, True), (421, True), (423, True), (424, True), (425, True), (427, False), (428, True), (429, True), (430, True), (432, True), (433, True), (436, True), (437, False), (438, True), (439, True), (440, True), (441, True), (442, True), (444, True), (446, False), (447, False), (448, True), (449, True), (450, True), (451, True), (452, True), (455, False), (456, True), (457, True), (458, False), (459, True), (460, True), (461, True), (463, True), (465, True), (466, True), (468, True), (470, True), (471, True), (474, True), (475, True), (477, False), (479, True), (480, True), (482, True), (483, True), (484, True), (485, True), (487, True), (488, True), (489, True), (490, True), (491, True), (494, True), (495, True), (497, True), (499, True), (500, True), (501, True), (502, True), (503, True), (504, True), (505, False), (506, True), (507, False), (509, True), (510, True), (511, False), (512, True), (513, True), (515, False), (516, False), (518, True), (519, True), (521, True), (522, True), (525, True), (526, False), (527, True), (528, False), (529, True), (531, True), (532, True), (533, True), (534, True), (535, True), (536, True), (537, True), (538, True), (539, True), (541, True), (542, False), (543, True), (544, True), (545, True), (546, False), (547, True), (549, True), (550, True), (551, False), (552, True), (553, True), (554, True), (555, False), (557, False), (558, True), (559, True), (560, True), (562, True), (564, True), (565, True), (566, True), (567, True), (568, True), (570, True), (571, True), (572, True), (573, True), (574, True), (575, True), (576, False), (577, True), (578, True), (579, False), (581, True), (582, True), (583, False), (584, True), (585, True), (586, True), (587, True), (588, False), (590, True), (591, True), (594, True), (596, True), (598, True), (599, True), (601, True), (602, True), (603, True), (604, False), (605, False), (606, True), (608, True), (609, True), (610, True), (612, False), (613, True), (614, True), (615, True), (616, False), (617, True), (620, False), (621, True), (622, True), (624, True), (626, False), (627, True), (630, True), (631, True), (632, True), (636, False), (639, True), (640, True), (641, True), (642, True), (644, True), (645, False), (646, False), (648, False), (652, True), (653, True), (655, False), (656, True), (657, True), (658, True), (659, True), (662, True), (663, True), (664, True), (665, True), (666, False), (668, True), (669, False), (670, True), (674, True), (675, True), (680, True), (682, True), (684, True), (685, True), (687, False), (689, True), (692, True), (693, True), (694, True), (695, True), (696, True), (697, True), (698, True), (701, True), (702, False), (703, True), (704, True), (705, True), (707, True), (708, True), (709, True), (710, True), (711, True), (712, True), (713, False), (714, True), (716, True), (717, True), (718, True), (719, True), (721, True), (723, True), (724, True), (726, True), (727, True), (728, True), (729, True), (730, True), (731, True), (733, True), (734, True), (735, True), (737, False), (738, False), (739, True), (740, True), (741, True), (742, True), (743, True), (744, True), (745, True), (748, False), (749, True), (751, True), (752, True), (753, False), (755, True), (756, True), (758, False), (759, True), (762, True), (764, True), (765, True), (766, False), (768, True), (769, True), (770, True), (775, False), (776, True), (778, True), (780, True), (781, False), (782, False), (783, False), (784, True), (785, True), (786, True), (787, True), (788, True), (791, True), (793, True), (794, True), (795, False), (796, True), (797, True), (798, True), (800, True), (802, False), (803, True), (804, True), (805, True), (807, True), (808, True), (809, False), (811, True), (812, True), (813, True), (814, True), (815, True), (816, True), (819, True), (820, True), (821, False), (823, False), (824, True), (828, True), (830, True), (832, True), (833, True), (834, True), (836, True), (837, True), (838, False), (839, True), (841, False), (842, True), (843, True), (844, True), (845, True), (846, False), (847, True), (849, True), (850, True), (851, True), (854, True), (855, True), (856, True), (857, True), (858, False), (860, False), (861, False), (863, True), (864, False), (867, True), (868, False), (869, True), (870, False), (873, False), (874, True), (875, True), (876, True), (879, True), (880, True), (882, True), (883, True), (884, True), (885, True), (886, True), (888, True), (890, False), (891, True), (892, True), (893, True), (894, True), (897, True), (898, True), (899, True), (900, False), (901, False), (903, True), (904, True), (905, True), (906, False), (907, True), (908, True), (909, True), (910, True), (912, True), (913, True), (914, False), (915, True), (917, False), (918, False), (919, True), (920, False), (921, True), (922, True), (924, True), (925, False), (926, True), (927, True), (928, True), (929, False), (933, True), (935, True), (938, True), (939, True), (940, True), (942, False), (943, False), (944, True), (945, True), (946, False), (947, False), (948, True), (949, False), (950, False), (951, False), (952, True), (953, True), (954, True), (955, True), (957, True), (959, True), (960, True), (962, True), (963, True), (964, False), (967, True), (969, True), (970, True), (971, False), (972, True), (973, True), (974, True), (975, False), (977, True), (978, False), (981, True), (982, True), (985, True), (986, True), (987, True), (989, True), (990, True), (992, False), (995, True), (996, True), (997, False), (998, True), (999, True), (1000, True), (1002, True), (1003, True), (1005, True), (1006, False), (1007, True), (1008, True), (1009, True), (1010, True), (1011, True), (1012, True), (1014, True), (1015, True), (1016, False), (1017, True), (1018, False), (1019, False), (1023, True), (1024, True), (1025, True), (1027, True), (1028, False), (1029, True), (1030, True), (1031, False), (1032, False), (1033, True), (1035, True), (1036, True), (1037, True), (1038, True), (1039, True), (1041, True), (1042, False), (1043, False), (1044, False), (1047, True), (1048, True), (1050, True), (1051, True), (1052, True), (1053, True), (1055, True), (1056, True), (1057, True), (1058, True), (1059, True), (1062, True), (1063, True), (1064, True), (1065, False), (1066, True), (1067, True), (1069, False), (1070, True), (1071, False), (1072, True), (1073, True), (1074, False), (1075, True), (1076, False), (1077, True), (1080, False), (1081, True), (1082, True), (1083, True), (1084, True), (1086, True), (1087, False), (1088, False), (1089, False), (1090, True), (1092, True), (1093, True), (1094, True), (1096, True), (1097, True), (1098, True), (1099, False), (1100, True), (1101, True), (1102, True), (1104, True), (1106, False), (1107, True), (1108, True), (1109, True), (1110, True), (1111, False), (1112, True), (1113, True), (1114, True), (1115, True), (1120, True), (1121, True), (1122, True), (1123, False), (1124, True), (1125, True), (1129, True), (1133, True), (1136, True), (1137, True), (1138, True), (1139, False), (1140, True), (1141, True), (1142, False), (1144, False), (1145, True), (1146, True), (1147, True), (1148, True), (1149, True), (1150, False), (1151, False), (1153, True), (1155, True), (1156, True), (1157, False), (1158, True), (1159, True), (1160, False), (1161, True), (1163, False), (1164, True), (1165, True), (1167, False), (1168, True), (1169, True), (1170, True), (1171, False), (1172, True), (1174, False), (1177, True), (1178, True), (1179, True), (1182, True), (1184, True), (1185, True), (1189, True), (1190, True), (1191, True), (1192, True), (1193, True), (1194, True), (1195, True), (1197, True), (1198, True), (1199, True), (1200, False), (1201, False), (1204, False), (1205, True), (1206, False), (1207, True), (1208, True), (1209, False), (1210, False), (1211, True), (1212, True), (1213, True), (1215, True), (1216, False), (1217, False), (1220, True), (1222, True), (1223, True), (1224, False), (1227, True), (1229, True), (1231, True), (1232, True), (1233, True), (1234, True), (1237, True), (1239, False), (1240, True), (1241, True), (1242, True), (1243, True), (1244, True), (1245, True), (1246, True), (1247, True), (1248, True), (1249, True), (1250, True), (1251, True), (1252, True), (1253, True), (1254, True), (1255, False), (1256, True), (1258, True), (1259, False), (1261, True), (1262, False), (1264, True), (1265, True), (1267, False), (1268, True), (1269, False), (1270, True), (1271, True), (1272, True), (1274, True), (1275, True), (1279, True), (1281, True), (1282, True), (1283, False), (1284, True), (1285, True), (1286, True), (1287, True), (1288, False), (1290, True), (1291, True), (1292, True), (1293, True), (1294, True), (1295, True), (1297, True), (1298, True), (1299, True), (1301, True), (1302, True), (1304, True), (1305, True), (1306, True), (1307, True), (1309, True), (1310, True), (1311, True), (1312, True), (1313, True), (1315, True), (1317, True), (1318, False), (1322, False), (1323, True), (1324, True), (1325, True), (1326, True), (1327, True), (1328, True), (1330, True), (1331, False), (1332, True), (1333, True), (1334, True), (1337, True), (1338, True), (1339, False), (1340, True), (1341, True), (1342, True), (1343, True), (1344, True), (1345, True), (1346, True), (1347, True), (1348, True), (1349, False), (1351, True), (1353, True), (1354, True), (1355, True), (1356, True), (1357, True), (1358, True), (1359, True), (1360, True), (1362, True), (1364, True), (1365, True), (1366, False), (1367, True), (1368, True), (1369, True), (1370, True), (1371, True), (1372, True), (1373, True), (1375, True), (1377, True), (1378, False), (1379, False), (1380, False), (1381, True), (1382, True), (1383, True), (1385, True), (1386, True), (1387, True), (1388, True), (1389, True), (1390, True), (1391, True), (1392, False), (1393, True), (1394, True), (1395, False), (1396, False), (1398, True), (1401, True), (1402, True), (1404, True), (1405, False), (1406, False), (1407, False), (1409, True), (1410, True), (1411, True), (1412, True), (1413, True), (1414, True), (1415, False), (1417, True), (1419, True), (1421, True), (1422, True), (1423, False), (1424, True), (1425, True), (1426, True), (1427, True), (1430, True), (1431, True), (1432, True), (1433, True), (1434, False), (1435, True), (1437, False), (1438, True), (1439, True), (1440, True), (1441, True), (1442, True), (1443, True), (1444, True), (1445, False), (1446, True), (1448, True), (1449, True), (1450, False), (1452, True), (1455, True), (1456, True), (1457, True), (1458, True), (1459, True), (1460, True), (1461, True), (1462, True), (1463, True), (1464, False), (1465, True), (1466, True), (1467, True), (1470, False), (1471, False), (1472, True), (1473, True), (1474, False), (1475, True), (1476, False), (1480, True), (1481, True), (1482, True), (1483, True), (1484, False), (1485, False), (1486, True), (1488, True), (1489, False), (1490, True), (1491, True), (1492, False), (1496, False), (1497, False), (1498, True), (1499, True), (1500, True), (1501, True), (1503, True), (1504, True), (1505, False), (1506, True), (1507, True), (1509, False), (1511, True), (1512, True), (1513, True), (1514, False), (1515, True), (1517, False), (1518, True), (1519, True), (1521, False), (1522, False), (1523, False), (1524, True), (1525, False), (1527, True), (1528, False), (1529, True), (1530, True), (1532, True), (1533, True), (1534, True), (1535, True), (1536, False), (1538, True), (1539, True), (1540, True), (1541, True), (1545, True), (1547, True), (1548, True), (1549, False), (1550, True), (1551, False), (1553, True), (1555, True), (1556, True), (1557, True), (1558, False), (1559, True), (1560, True), (1561, True), (1562, True), (1563, True), (1564, True), (1565, False), (1566, True), (1568, False), (1569, True), (1571, False), (1572, True), (1574, True), (1576, True), (1577, False), (1579, False), (1580, False), (1582, True), (1583, True), (1585, False), (1586, False), (1588, True), (1590, True), (1591, True), (1593, True), (1595, False), (1596, True), (1597, True), (1598, True), (1599, True), (1600, False), (1601, True), (1603, True), (1605, True), (1606, True), (1607, True), (1609, True), (1610, True), (1611, True), (1613, True), (1614, True), (1616, True), (1617, True), (1620, True), (1623, True), (1626, True), (1627, True), (1629, False), (1630, True), (1631, True), (1632, False), (1633, False), (1634, True), (1637, True), (1638, False), (1639, True), (1640, True), (1641, True), (1642, True), (1643, True), (1644, False), (1645, True), (1646, False), (1647, True), (1648, True), (1649, True), (1650, False), (1656, False), (1657, True), (1658, True), (1659, True), (1660, True), (1662, False), (1663, True), (1664, True), (1665, True), (1667, False), (1668, False), (1669, True), (1670, True), (1671, True), (1672, True), (1676, True), (1677, True), (1680, True), (1681, True), (1682, True), (1684, True), (1685, True), (1686, False), (1687, False), (1688, True), (1689, False), (1690, False), (1691, True), (1692, True), (1693, True), (1694, False), (1699, False), (1701, True), (1702, False), (1703, True), (1704, True), (1706, True), (1707, True), (1708, True), (1709, False), (1710, True), (1711, True), (1712, False), (1713, True), (1714, True), (1715, False), (1716, True), (1717, True), (1718, True), (1719, True), (1720, True), (1721, True), (1722, True), (1723, True), (1724, False), (1725, False), (1727, False), (1728, True), (1729, True), (1730, True), (1731, True), (1733, False), (1734, True), (1735, True), (1736, True), (1737, True), (1739, False), (1740, False), (1741, True), (1742, False), (1743, True), (1744, True), (1745, True), (1747, True), (1750, False), (1751, False), (1752, True), (1753, True), (1754, True), (1755, False), (1756, True), (1757, False), (1758, True), (1759, True), (1760, True), (1762, False), (1763, True), (1764, False), (1768, True), (1770, False), (1772, False), (1773, False), (1774, True), (1775, False), (1776, True), (1778, True), (1779, False), (1780, True), (1781, True), (1782, False), (1783, True), (1785, True), (1786, True), (1787, False), (1788, True), (1789, True), (1790, True), (1791, True), (1792, True), (1793, False), (1794, True), (1795, True), (1796, True), (1797, True), (1802, True), (1803, True), (1804, True), (1805, True), (1807, True), (1808, False), (1809, True), (1810, False), (1811, True), (1812, True), (1813, True), (1814, True), (1815, True), (1816, True), (1818, True), (1819, False), (1820, True), (1821, False), (1822, False), (1823, True), (1824, True), (1825, False), (1826, True), (1827, True), (1828, True), (1829, True), (1830, True), (1831, True), (1832, True), (1833, True), (1835, True), (1836, False), (1837, True), (1838, False), (1839, True), (1840, True), (1841, True), (1843, True), (1844, True), (1845, True), (1846, False), (1848, True), (1849, True), (1850, False), (1851, True), (1853, False), (1854, True), (1857, False), (1859, True), (1861, False), (1862, True), (1863, False), (1864, False), (1865, True), (1866, True), (1867, True), (1868, True), (1871, True), (1872, True), (1873, False), (1874, True), (1875, True), (1876, True), (1877, False), (1878, True), (1879, True), (1880, True), (1882, True), (1885, True), (1886, True), (1887, True), (1890, True), (1891, True), (1894, True), (1895, True), (1898, True), (1899, False), (1904, True), (1905, True), (1906, True), (1907, False), (1909, False), (1911, True), (1913, True), (1914, True), (1915, True), (1916, True), (1917, False), (1918, True), (1919, False), (1921, True), (1922, False), (1924, True), (1925, True), (1926, True), (1927, False), (1928, True), (1930, True), (1931, True), (1933, True), (1934, True), (1935, True), (1937, True), (1938, True), (1939, True), (1940, True), (1941, False), (1942, True), (1944, True), (1945, True), (1948, True), (1949, True), (1950, True), (1951, True), (1952, True), (1953, True), (1954, False), (1955, False), (1956, True), (1958, True), (1959, True), (1960, False), (1962, False), (1963, True), (1965, True), (1966, False), (1967, True), (1969, True), (1971, True), (1973, True), (1974, True), (1975, True), (1976, False), (1977, True), (1978, True), (1979, True), (1980, True), (1981, False), (1982, True), (1984, True), (1986, True), (1987, True), (1988, True), (1989, True), (1991, False), (1993, True), (1995, True), (1996, True), (1997, True), (1999, True), (2001, True), (2002, True), (2003, True), (2005, True), (2006, True), (2007, True), (2008, False), (2011, True), (2013, True), (2014, True), (2015, True), (2016, True), (2017, True), (2018, True), (2020, True), (2021, True), (2022, True), (2023, False), (2025, False), (2026, True), (2027, True), (2028, True), (2029, True), (2030, False), (2031, True), (2032, True), (2033, True), (2035, True), (2037, True), (2040, True), (2041, True), (2042, True), (2043, False), (2044, True), (2046, True), (2047, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 10
    dataset = Dataset(input_bits_count, [(0, False), (1, True), (2, True), (3, False), (4, False), (9, False), (11, True), (12, False), (14, False), (18, False), (20, False), (22, False), (23, False), (24, False), (26, True), (27, True), (28, False), (29, False), (30, False), (32, False), (33, False), (35, False), (37, True), (38, False), (40, True), (41, False), (42, False), (44, False), (46, True), (48, False), (49, False), (50, False), (53, False), (55, True), (56, False), (58, False), (59, False), (60, False), (61, False), (65, False), (66, True), (67, True), (68, False), (71, False), (72, True), (73, False), (74, False), (75, False), (78, True), (79, False), (80, False), (81, False), (82, True), (83, False), (84, True), (89, True), (90, False), (92, False), (95, False), (97, False), (100, False), (102, False), (103, True), (104, True), (105, False), (106, False), (107, False), (108, False), (109, False), (110, True), (112, False), (113, True), (116, False), (117, False), (119, False), (122, False), (123, False), (126, False), (127, False), (128, True), (129, False), (130, True), (131, False), (133, False), (135, False), (136, False), (137, False), (142, False), (143, False), (144, False), (146, False), (147, False), (148, False), (149, False), (151, True), (153, True), (154, True), (155, False), (156, False), (160, False), (165, False), (167, False), (170, True), (172, False), (175, True), (177, False), (178, True), (183, False), (184, False), (186, True), (187, False), (188, False), (189, False), (190, False), (192, True), (193, False), (197, False), (198, True), (199, False), (202, False), (203, True), (204, False), (206, True), (207, False), (210, False), (211, False), (212, False), (214, True), (218, False), (219, False), (221, False), (222, False), (226, True), (227, True), (228, False), (229, False), (231, False), (232, False), (234, False), (236, False), (239, True), (240, False), (241, True), (245, False), (246, False), (247, False), (248, True), (250, True), (252, False), (254, False), (255, False), (256, False), (258, False), (259, False), (260, False), (261, False), (262, False), (263, False), (265, False), (269, False), (270, True), (271, False), (273, True), (275, True), (277, False), (279, False), (280, True), (281, True), (282, False), (286, False), (287, False), (289, False), (290, False), (291, True), (293, False), (294, True), (295, False), (296, False), (298, False), (299, False), (306, False), (307, False), (309, True), (310, True), (312, False), (313, False), (315, False), (317, False), (318, False), (320, False), (321, False), (322, False), (325, False), (328, False), (330, False), (331, False), (333, False), (335, False), (336, True), (341, False), (342, False), (343, False), (344, True), (346, False), (347, False), (348, False), (349, False), (350, False), (351, False), (352, False), (353, True), (357, False), (358, True), (360, False), (363, True), (364, False), (366, True), (367, False), (369, False), (371, True), (372, False), (373, False), (374, True), (376, False), (378, False), (379, False), (381, False), (382, True), (383, False), (384, True), (385, False), (387, False), (393, False), (395, False), (396, False), (398, False), (399, False), (401, True), (402, True), (405, False), (407, True), (408, False), (409, True), (411, False), (412, False), (413, False), (414, True), (417, False), (420, True), (423, True), (424, False), (425, True), (426, True), (427, False), (428, False), (429, False), (430, True), (431, False), (432, False), (434, False), (437, False), (438, True), (440, False), (441, False), (443, True), (445, False), (448, False), (449, True), (451, False), (452, True), (453, True), (454, False), (456, False), (458, False), (460, False), (462, False), (463, False), (464, True), (465, True), (466, True), (467, False), (468, False), (470, False), (471, False), (477, False), (478, True), (479, False), (480, False), (481, True), (482, False), (483, False), (484, True), (485, False), (486, False), (487, False), (488, True), (489, True), (490, False), (491, True), (492, False), (494, False), (495, True), (498, False), (499, True), (500, True), (501, True), (502, True), (503, True), (504, False), (506, False), (508, False), (510, True), (511, False), (512, False), (514, False), (515, True), (519, False), (522, False), (523, False), (524, False), (526, True), (528, False), (530, False), (531, False), (532, False), (534, False), (535, False), (536, True), (537, False), (538, False), (539, False), (542, False), (543, False), (544, False), (545, True), (547, True), (549, False), (551, False), (553, False), (555, False), (556, True), (557, False), (558, False), (559, False), (561, False), (565, True), (566, False), (567, False), (568, False), (570, True), (571, False), (572, False), (573, False), (574, False), (575, False), (576, True), (580, False), (581, False), (582, True), (583, False), (584, True), (585, False), (589, False), (590, False), (591, False), (595, False), (596, False), (597, False), (599, False), (602, False), (604, True), (605, False), (607, False), (611, False), (613, False), (614, True), (615, True), (617, False), (618, True), (621, True), (623, False), (624, True), (625, False), (626, False), (629, True), (633, False), (635, False), (636, False), (638, False), (639, False), (641, False), (642, False), (644, False), (645, False), (647, False), (648, True), (649, False), (652, False), (653, False), (654, False), (657, False), (658, False), (659, False), (660, True), (661, False), (662, False), (663, True), (664, True), (665, True), (668, True), (669, False), (670, True), (672, False), (673, False), (679, False), (680, False), (683, False), (684, False), (685, True), (686, False), (687, False), (688, True), (690, True), (691, True), (692, False), (695, False), (696, True), (697, False), (698, False), (700, True), (701, False), (702, True), (703, True), (704, False), (706, True), (707, False), (708, False), (709, True), (710, True), (711, False), (712, False), (713, False), (715, False), (716, True), (717, True), (719, True), (721, False), (722, True), (726, True), (729, True), (730, True), (731, False), (732, False), (733, False), (734, False), (736, False), (737, False), (739, False), (740, True), (744, False), (745, False), (746, True), (748, False), (754, False), (758, False), (759, False), (761, False), (762, False), (763, False), (769, False), (770, False), (771, True), (772, False), (773, False), (774, False), (776, False), (778, True), (779, False), (787, False), (789, False), (795, False), (796, False), (798, False), (799, False), (801, False), (802, False), (803, False), (804, False), (805, False), (806, False), (807, False), (808, True), (810, False), (811, False), (812, False), (815, False), (816, False), (817, False), (818, True), (819, False), (821, True), (825, False), (826, False), (830, False), (832, False), (833, True), (836, False), (838, False), (840, True), (841, True), (844, True), (847, False), (848, True), (849, False), (850, False), (851, False), (853, False), (855, False), (856, False), (858, False), (860, False), (861, False), (863, False), (866, False), (868, False), (870, True), (874, False), (875, False), (876, False), (877, False), (878, False), (880, True), (881, True), (883, True), (884, False), (885, True), (886, False), (890, False), (891, False), (892, False), (893, False), (894, True), (896, True), (897, False), (898, True), (899, True), (903, True), (906, False), (907, False), (908, False), (910, False), (912, False), (913, True), (914, True), (916, False), (918, False), (919, False), (921, False), (924, False), (925, False), (927, True), (929, True), (931, False), (932, True), (933, False), (935, False), (936, False), (938, True), (939, False), (940, False), (941, False), (944, False), (945, False), (946, True), (947, False), (949, False), (950, True), (951, False), (953, False), (954, True), (957, False), (958, False), (959, True), (962, False), (963, True), (965, False), (967, False), (969, True), (970, False), (977, False), (979, True), (981, False), (984, False), (986, True), (987, False), (988, False), (995, True), (996, False), (997, False), (998, False), (999, False), (1000, False), (1001, False), (1002, False), (1003, False), (1004, False), (1005, False), (1006, True), (1007, False), (1008, False), (1012, False), (1014, False), (1016, False), (1018, False), (1021, False), (1022, False), (1023, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 10
    dataset = Dataset(input_bits_count, [(0, True), (1, False), (5, False), (6, True), (7, True), (8, False), (9, False), (10, False), (11, True), (12, False), (13, False), (17, False), (19, False), (20, True), (25, True), (26, True), (27, True), (28, False), (29, True), (30, False), (31, False), (33, False), (37, False), (39, True), (41, False), (42, False), (43, False), (45, True), (46, True), (48, True), (51, False), (53, True), (56, True), (58, True), (59, False), (60, True), (62, True), (63, True), (64, False), (66, False), (67, True), (68, False), (69, True), (70, True), (71, True), (72, True), (73, False), (74, False), (76, False), (78, False), (79, False), (81, False), (82, True), (84, True), (85, True), (86, True), (88, False), (89, True), (90, True), (91, True), (92, False), (95, False), (98, False), (99, True), (100, True), (101, True), (102, False), (105, False), (106, True), (107, True), (109, True), (110, True), (111, True), (112, True), (113, True), (114, True), (115, False), (116, True), (117, True), (118, True), (119, False), (120, False), (122, True), (123, False), (124, False), (128, False), (129, False), (130, True), (131, False), (132, False), (134, True), (139, True), (140, True), (142, True), (143, True), (145, False), (147, True), (149, False), (150, True), (151, True), (153, True), (154, False), (156, False), (157, True), (158, True), (162, True), (163, True), (164, False), (165, False), (166, True), (167, True), (168, False), (169, False), (170, True), (172, True), (173, True), (174, False), (175, False), (176, False), (179, True), (181, False), (184, True), (185, True), (186, True), (187, True), (189, False), (190, True), (191, True), (192, False), (194, False), (195, True), (196, False), (197, True), (200, True), (201, True), (202, True), (204, True), (205, True), (206, True), (207, False), (210, True), (211, False), (214, True), (216, False), (218, True), (219, True), (220, False), (221, False), (223, True), (226, False), (227, True), (229, True), (230, True), (231, False), (232, True), (233, False), (234, True), (235, True), (237, False), (238, True), (239, False), (240, False), (242, True), (246, True), (249, True), (251, True), (252, True), (253, False), (254, False), (255, False), (258, True), (259, True), (260, True), (263, False), (264, True), (265, False), (266, True), (267, True), (268, True), (269, False), (270, True), (271, False), (272, False), (273, True), (275, False), (276, True), (278, True), (279, True), (281, True), (282, False), (285, True), (287, True), (288, True), (289, False), (290, False), (292, False), (293, True), (295, False), (297, True), (298, False), (299, True), (300, True), (301, True), (302, True), (303, True), (306, True), (307, True), (308, True), (310, False), (311, True), (312, False), (314, True), (315, True), (316, True), (317, True), (318, False), (319, True), (320, True), (321, False), (322, True), (323, False), (325, True), (326, False), (328, True), (329, False), (330, True), (333, True), (334, False), (335, False), (336, False), (338, True), (339, False), (342, False), (344, True), (345, False), (347, True), (349, False), (352, False), (353, True), (354, False), (355, True), (356, True), (357, False), (358, False), (359, True), (360, True), (363, True), (364, True), (365, False), (366, False), (367, True), (368, True), (369, True), (370, False), (371, False), (372, False), (373, True), (374, True), (375, True), (377, False), (379, False), (380, True), (382, True), (383, True), (384, False), (385, True), (387, True), (388, False), (390, False), (391, True), (392, True), (397, False), (400, False), (401, False), (403, False), (405, True), (406, True), (407, False), (411, True), (412, True), (413, True), (416, False), (419, True), (420, False), (421, True), (422, True), (423, False), (426, True), (427, False), (428, True), (430, False), (431, True), (434, True), (435, True), (436, True), (437, False), (438, False), (439, True), (441, False), (442, False), (445, True), (446, False), (448, True), (450, True), (453, True), (454, False), (455, True), (456, False), (457, True), (458, True), (460, False), (461, True), (463, False), (464, True), (467, True), (469, True), (471, True), (472, True), (473, True), (475, False), (476, True), (477, True), (479, True), (480, True), (481, True), (482, False), (484, False), (485, True), (486, True), (489, True), (490, False), (492, False), (494, False), (495, True), (496, False), (497, True), (498, False), (499, True), (502, False), (503, True), (504, False), (506, True), (507, True), (508, False), (510, True), (513, True), (514, True), (515, True), (516, True), (517, True), (518, False), (519, False), (520, False), (521, True), (523, True), (524, True), (525, False), (526, True), (527, True), (528, False), (530, True), (533, False), (534, True), (535, False), (536, True), (537, False), (538, False), (540, True), (541, True), (542, True), (543, False), (545, True), (550, True), (552, False), (553, True), (554, True), (555, True), (556, False), (557, False), (559, False), (560, True), (562, False), (563, True), (565, False), (566, False), (567, False), (569, True), (570, True), (572, True), (574, True), (575, False), (576, True), (577, True), (579, True), (580, False), (581, True), (582, False), (585, False), (586, True), (587, True), (588, False), (589, False), (591, False), (592, True), (593, True), (594, False), (597, False), (598, False), (599, True), (601, True), (602, True), (603, False), (608, False), (609, False), (610, True), (611, True), (615, False), (616, False), (618, False), (619, True), (620, False), (622, True), (623, True), (624, True), (626, False), (628, False), (629, False), (630, True), (631, False), (632, False), (633, False), (635, False), (636, True), (637, True), (638, True), (642, True), (643, True), (644, True), (645, False), (647, True), (649, True), (650, True), (651, True), (653, False), (655, False), (656, False), (657, True), (659, True), (660, True), (661, False), (663, True), (664, True), (665, False), (667, True), (668, False), (669, True), (671, True), (673, False), (674, True), (675, False), (676, False), (677, True), (678, True), (679, True), (680, False), (682, False), (683, True), (684, False), (686, True), (688, True), (689, True), (690, True), (692, False), (694, True), (695, True), (696, True), (697, True), (699, True), (701, False), (706, False), (707, True), (708, False), (709, False), (710, False), (711, False), (712, True), (713, True), (714, True), (716, False), (717, True), (719, False), (720, True), (721, True), (723, True), (724, True), (725, False), (726, True), (729, True), (730, False), (731, True), (732, True), (734, True), (735, False), (737, True), (738, True), (739, False), (740, True), (741, False), (743, False), (744, True), (745, False), (747, False), (748, False), (749, True), (750, False), (751, True), (752, False), (753, True), (755, True), (756, True), (757, False), (760, False), (762, True), (763, True), (764, False), (765, True), (766, True), (768, True), (770, False), (772, False), (773, True), (774, True), (776, True), (777, True), (779, True), (780, True), (781, True), (782, True), (784, True), (786, False), (787, False), (789, False), (792, True), (793, False), (794, False), (795, True), (796, True), (798, False), (799, False), (800, True), (801, True), (803, True), (805, False), (806, True), (809, True), (810, True), (811, True), (812, True), (813, True), (814, True), (815, True), (816, True), (817, False), (818, False), (819, True), (820, False), (823, False), (824, True), (825, False), (826, True), (828, True), (829, True), (830, True), (834, True), (836, True), (837, True), (839, False), (840, True), (841, False), (842, False), (843, True), (845, True), (846, False), (847, True), (848, True), (850, False), (851, False), (852, False), (853, True), (856, False), (857, True), (858, False), (859, True), (862, False), (864, True), (865, False), (866, False), (868, False), (869, True), (870, True), (872, True), (873, True), (875, False), (876, True), (877, True), (878, False), (879, False), (880, False), (881, False), (882, True), (883, True), (884, False), (886, False), (888, True), (889, True), (890, False), (891, False), (892, True), (897, False), (899, False), (900, True), (903, False), (905, False), (906, False), (907, True), (911, True), (912, True), (914, True), (916, True), (917, True), (919, False), (920, False), (921, True), (923, False), (924, True), (926, True), (929, True), (931, False), (932, True), (935, True), (937, True), (938, True), (939, True), (940, True), (941, True), (942, False), (944, True), (945, False), (946, True), (947, False), (949, True), (950, False), (951, False), (952, True), (953, True), (954, False), (955, False), (956, False), (957, True), (962, False), (963, True), (964, False), (966, True), (967, False), (968, False), (969, True), (970, False), (971, True), (972, True), (973, False), (974, True), (975, True), (978, True), (979, True), (981, False), (985, False), (986, True), (987, True), (989, True), (991, True), (992, True), (994, True), (995, True), (996, True), (997, True), (998, True), (999, False), (1000, True), (1001, False), (1002, False), (1003, False), (1004, True), (1005, True), (1007, False), (1008, True), (1009, True), (1010, True), (1012, True), (1013, True), (1014, False), (1016, True), (1019, True), (1021, True), (1022, True), (1023, True)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 11
    dataset = Dataset(input_bits_count, [(3, False), (9, True), (10, False), (11, False), (12, False), (13, False), (14, False), (15, False), (16, False), (18, False), (19, False), (20, False), (26, False), (27, False), (29, False), (34, False), (35, False), (36, False), (38, True), (40, False), (42, True), (43, True), (44, False), (47, False), (49, True), (50, False), (51, False), (53, False), (55, False), (56, True), (57, False), (58, False), (59, True), (61, True), (63, True), (64, False), (69, False), (71, True), (72, False), (77, False), (79, True), (81, True), (82, False), (83, False), (87, True), (88, False), (89, False), (90, True), (91, False), (93, False), (95, False), (96, False), (97, False), (98, True), (99, True), (100, False), (101, True), (102, True), (103, True), (104, False), (105, False), (107, True), (109, False), (111, False), (112, True), (113, False), (115, True), (116, True), (118, True), (119, False), (120, False), (121, False), (122, True), (125, True), (127, True), (129, False), (130, False), (131, False), (132, False), (133, False), (134, False), (138, True), (139, False), (140, False), (144, False), (145, False), (146, True), (148, True), (149, False), (151, False), (152, False), (153, False), (155, False), (158, False), (160, False), (161, True), (162, False), (164, False), (165, False), (166, True), (167, False), (168, False), (169, False), (170, False), (175, False), (176, False), (177, False), (178, False), (179, False), (183, False), (184, False), (191, True), (192, False), (193, False), (194, False), (195, False), (197, True), (199, True), (203, True), (204, False), (206, False), (207, False), (208, False), (212, True), (217, False), (219, False), (220, False), (221, False), (222, False), (228, False), (231, False), (233, False), (235, False), (236, True), (240, False), (241, True), (243, False), (244, False), (248, True), (251, True), (252, False), (253, True), (258, False), (259, False), (264, True), (267, False), (268, False), (269, False), (271, True), (272, False), (273, False), (274, False), (275, True), (276, False), (277, True), (280, False), (281, False), (283, False), (286, True), (287, False), (288, False), (289, False), (290, False), (292, False), (293, False), (296, True), (297, False), (299, True), (300, False), (301, False), (302, True), (303, False), (305, False), (306, False), (307, True), (310, True), (311, True), (312, False), (313, False), (314, True), (316, True), (318, True), (320, True), (321, False), (322, False), (324, True), (325, False), (326, False), (327, True), (328, False), (329, True), (331, False), (332, False), (333, False), (334, False), (335, False), (336, True), (337, True), (338, False), (342, True), (343, False), (345, False), (346, True), (350, False), (351, False), (352, True), (353, True), (355, False), (358, False), (361, False), (363, True), (369, False), (370, False), (372, False), (373, False), (375, False), (378, True), (381, False), (382, False), (385, False), (386, True), (387, True), (388, False), (389, False), (391, False), (392, False), (395, True), (396, False), (400, True), (401, True), (402, False), (403, False), (404, False), (405, False), (406, False), (407, True), (408, False), (409, True), (410, False), (412, False), (413, False), (414, False), (415, False), (416, True), (418, False), (422, True), (423, False), (424, False), (425, False), (426, True), (427, False), (428, False), (429, False), (432, True), (434, False), (435, True), (437, False), (439, False), (442, False), (443, False), (444, False), (447, True), (448, True), (449, True), (450, False), (451, False), (452, False), (453, False), (455, False), (456, False), (457, True), (458, False), (459, True), (460, False), (462, False), (464, False), (465, False), (466, False), (467, False), (468, True), (470, True), (471, True), (472, False), (473, False), (475, True), (476, True), (478, False), (481, True), (482, False), (484, False), (485, True), (486, True), (487, False), (490, False), (496, False), (497, False), (498, False), (500, True), (502, False), (503, True), (504, False), (505, False), (507, False), (510, True), (511, True), (515, False), (516, True), (517, False), (518, True), (519, False), (520, False), (521, False), (522, False), (523, False), (525, True), (526, False), (528, False), (531, False), (532, True), (533, False), (538, False), (539, False), (541, False), (542, False), (545, False), (549, False), (550, True), (553, True), (555, False), (556, False), (557, True), (559, False), (565, True), (566, True), (567, False), (568, False), (569, True), (571, False), (572, False), (575, False), (576, False), (577, True), (579, True), (583, False), (584, True), (585, True), (586, False), (587, False), (590, True), (591, True), (595, True), (596, False), (597, True), (598, False), (600, False), (601, True), (602, False), (603, False), (606, True), (607, False), (610, False), (614, True), (615, False), (619, False), (620, False), (622, True), (623, False), (624, True), (625, False), (627, False), (629, True), (630, False), (631, False), (632, False), (633, True), (634, True), (635, False), (636, False), (639, False), (643, True), (645, False), (646, False), (647, False), (649, False), (650, True), (652, False), (653, False), (656, True), (657, True), (658, False), (659, False), (661, False), (662, True), (663, False), (666, True), (668, False), (669, False), (670, True), (671, False), (672, False), (674, False), (675, False), (676, False), (678, False), (679, False), (683, True), (684, False), (685, False), (686, True), (687, False), (689, False), (690, False), (691, True), (692, False), (693, False), (694, False), (696, True), (697, False), (698, True), (699, False), (700, False), (702, False), (703, False), (704, False), (706, False), (708, False), (709, False), (710, True), (712, True), (714, True), (716, True), (717, True), (718, True), (719, False), (722, False), (723, False), (725, True), (726, False), (727, True), (730, False), (731, False), (733, False), (736, False), (737, False), (739, True), (740, False), (741, False), (746, False), (748, True), (750, False), (752, False), (754, False), (755, True), (759, False), (762, True), (763, False), (764, False), (766, True), (767, False), (768, False), (769, True), (770, True), (771, False), (773, False), (774, True), (775, True), (776, False), (779, True), (780, False), (782, False), (784, False), (785, False), (786, False), (787, False), (788, False), (790, False), (792, True), (794, False), (795, True), (796, True), (798, False), (799, True), (807, False), (809, True), (810, False), (811, False), (813, False), (817, False), (818, False), (819, False), (820, True), (824, False), (825, True), (827, False), (828, False), (829, False), (830, False), (831, True), (833, False), (834, False), (835, False), (838, False), (839, False), (841, False), (842, True), (844, True), (846, False), (848, False), (849, False), (850, True), (851, False), (852, True), (853, True), (855, True), (859, False), (860, False), (861, False), (865, True), (866, False), (868, True), (869, False), (870, True), (871, False), (872, False), (877, False), (878, True), (879, True), (882, False), (883, True), (888, False), (889, False), (890, False), (891, False), (892, False), (893, False), (897, False), (899, False), (901, True), (902, False), (903, False), (904, False), (906, False), (909, False), (913, False), (914, False), (915, False), (916, True), (917, True), (919, False), (920, True), (921, False), (923, False), (926, False), (927, False), (928, True), (930, True), (931, False), (932, False), (933, False), (934, False), (937, True), (938, True), (940, True), (941, False), (942, False), (943, False), (945, False), (946, True), (947, False), (948, False), (949, True), (953, False), (956, False), (957, False), (959, False), (960, True), (962, False), (964, False), (966, True), (967, True), (968, False), (969, True), (970, False), (971, True), (972, True), (973, False), (975, True), (977, False), (978, False), (981, False), (983, False), (985, False), (987, False), (990, False), (993, False), (994, False), (995, False), (997, False), (999, False), (1003, True), (1005, False), (1007, True), (1008, False), (1011, False), (1012, False), (1013, True), (1014, True), (1015, False), (1016, False), (1017, False), (1018, False), (1019, False), (1020, True), (1021, False), (1022, False), (1023, False), (1025, False), (1027, False), (1028, True), (1029, False), (1030, True), (1031, False), (1034, False), (1035, True), (1036, False), (1040, True), (1042, False), (1043, True), (1044, False), (1045, False), (1046, False), (1049, True), (1052, False), (1056, False), (1057, False), (1058, True), (1059, False), (1060, False), (1061, True), (1063, False), (1064, False), (1065, False), (1066, False), (1071, False), (1074, False), (1075, True), (1079, True), (1080, False), (1081, True), (1082, True), (1084, False), (1085, False), (1086, True), (1089, False), (1092, False), (1093, False), (1094, False), (1095, False), (1096, False), (1097, False), (1098, False), (1100, False), (1102, False), (1106, False), (1107, True), (1108, False), (1109, False), (1110, False), (1111, False), (1113, True), (1114, False), (1115, True), (1116, False), (1117, True), (1119, False), (1120, False), (1121, False), (1122, True), (1123, False), (1124, True), (1125, False), (1126, False), (1129, False), (1130, False), (1132, True), (1134, False), (1135, False), (1136, False), (1137, False), (1140, False), (1142, False), (1144, False), (1145, True), (1146, False), (1147, False), (1148, False), (1151, False), (1152, False), (1154, False), (1155, False), (1156, False), (1157, False), (1161, True), (1162, False), (1166, True), (1167, False), (1169, True), (1170, False), (1171, False), (1179, True), (1183, False), (1184, False), (1185, False), (1186, True), (1187, True), (1188, False), (1191, False), (1192, False), (1194, False), (1198, False), (1200, False), (1201, False), (1202, True), (1203, False), (1204, True), (1205, False), (1206, False), (1207, True), (1209, False), (1210, False), (1211, True), (1212, True), (1213, False), (1214, False), (1217, False), (1219, False), (1220, True), (1222, True), (1223, False), (1224, True), (1227, False), (1229, False), (1232, False), (1234, False), (1237, True), (1238, False), (1239, False), (1240, False), (1241, True), (1242, False), (1244, True), (1247, False), (1248, False), (1253, False), (1254, False), (1255, False), (1256, True), (1257, False), (1260, True), (1263, True), (1267, False), (1269, True), (1270, False), (1271, True), (1272, False), (1273, False), (1275, False), (1276, False), (1278, False), (1279, False), (1281, False), (1282, True), (1285, False), (1286, True), (1287, False), (1288, False), (1291, True), (1292, False), (1295, True), (1296, True), (1297, False), (1298, False), (1299, False), (1300, False), (1301, False), (1303, True), (1304, False), (1305, False), (1307, True), (1310, True), (1312, False), (1313, False), (1314, False), (1315, True), (1317, False), (1318, True), (1319, False), (1320, False), (1323, True), (1325, False), (1326, False), (1329, False), (1330, False), (1332, False), (1334, False), (1335, True), (1336, True), (1337, False), (1342, True), (1343, False), (1346, False), (1347, True), (1348, True), (1351, False), (1353, False), (1356, False), (1357, True), (1358, False), (1359, False), (1360, False), (1361, True), (1363, False), (1364, True), (1365, False), (1366, False), (1369, False), (1370, False), (1371, False), (1373, False), (1374, True), (1375, True), (1376, False), (1378, False), (1379, False), (1381, False), (1382, True), (1385, False), (1387, False), (1388, False), (1389, True), (1390, False), (1391, False), (1395, False), (1396, False), (1397, True), (1398, False), (1399, True), (1400, True), (1401, False), (1402, False), (1403, True), (1404, False), (1405, False), (1408, True), (1409, False), (1410, True), (1411, False), (1412, False), (1413, False), (1414, True), (1415, False), (1417, True), (1418, False), (1421, False), (1423, True), (1425, True), (1429, True), (1432, False), (1434, False), (1436, True), (1438, False), (1440, True), (1442, False), (1443, False), (1446, True), (1448, False), (1450, False), (1451, False), (1452, False), (1453, False), (1456, False), (1457, False), (1461, True), (1462, True), (1464, False), (1466, False), (1467, False), (1469, False), (1470, False), (1471, True), (1472, False), (1473, False), (1474, False), (1476, False), (1477, False), (1478, True), (1481, False), (1483, False), (1484, False), (1488, False), (1489, False), (1490, False), (1491, True), (1493, False), (1494, True), (1495, True), (1496, False), (1498, True), (1500, False), (1501, False), (1503, False), (1504, False), (1505, False), (1506, True), (1509, False), (1511, True), (1513, True), (1514, True), (1515, True), (1517, False), (1518, False), (1520, False), (1521, False), (1523, True), (1525, False), (1526, False), (1527, False), (1528, True), (1529, False), (1530, True), (1531, False), (1532, False), (1533, False), (1534, False), (1535, False), (1536, False), (1537, False), (1538, True), (1539, False), (1540, True), (1541, False), (1542, True), (1544, True), (1545, False), (1547, False), (1548, False), (1549, False), (1550, False), (1551, False), (1552, False), (1557, False), (1560, True), (1561, True), (1562, True), (1563, False), (1564, True), (1565, False), (1567, True), (1569, False), (1570, False), (1571, False), (1574, False), (1575, True), (1577, False), (1578, True), (1581, False), (1583, False), (1584, False), (1585, False), (1586, True), (1587, False), (1589, False), (1592, True), (1595, False), (1596, False), (1597, False), (1598, False), (1600, True), (1606, False), (1607, False), (1608, False), (1613, False), (1616, True), (1617, False), (1618, False), (1620, False), (1621, False), (1622, True), (1623, False), (1624, False), (1625, False), (1627, False), (1628, False), (1629, False), (1631, False), (1633, False), (1636, True), (1637, True), (1639, False), (1641, True), (1643, False), (1644, False), (1645, False), (1646, True), (1647, False), (1648, True), (1649, False), (1651, False), (1653, True), (1654, True), (1655, False), (1656, True), (1658, False), (1659, True), (1660, True), (1661, False), (1663, False), (1666, False), (1667, False), (1670, False), (1671, True), (1672, False), (1673, True), (1675, False), (1677, True), (1679, False), (1680, False), (1681, False), (1682, False), (1684, False), (1685, False), (1686, False), (1687, False), (1688, False), (1690, True), (1691, False), (1693, False), (1694, False), (1697, False), (1698, True), (1702, False), (1705, False), (1707, True), (1708, True), (1710, False), (1712, True), (1716, False), (1717, False), (1718, False), (1719, True), (1721, False), (1722, False), (1723, True), (1724, False), (1725, False), (1726, False), (1729, False), (1731, False), (1734, True), (1735, False), (1736, False), (1737, False), (1738, True), (1739, False), (1740, False), (1742, False), (1743, True), (1746, True), (1747, False), (1749, True), (1752, False), (1753, False), (1754, False), (1755, True), (1758, False), (1759, False), (1761, False), (1762, False), (1764, True), (1767, False), (1768, False), (1769, False), (1772, False), (1773, False), (1776, False), (1777, False), (1779, False), (1781, False), (1782, True), (1783, False), (1784, False), (1786, True), (1788, True), (1793, True), (1794, False), (1796, False), (1797, False), (1798, False), (1799, False), (1801, False), (1802, True), (1804, False), (1811, True), (1812, False), (1813, False), (1816, False), (1817, False), (1818, False), (1820, False), (1821, False), (1823, True), (1826, False), (1828, True), (1830, True), (1831, False), (1832, True), (1836, True), (1837, False), (1840, False), (1841, False), (1842, False), (1843, False), (1844, False), (1845, False), (1846, False), (1847, False), (1849, False), (1850, False), (1852, False), (1853, False), (1855, False), (1857, False), (1858, False), (1860, False), (1862, True), (1868, True), (1870, True), (1871, False), (1872, True), (1873, False), (1879, False), (1880, True), (1881, True), (1882, False), (1886, False), (1887, False), (1890, False), (1891, True), (1892, False), (1893, True), (1894, False), (1895, True), (1898, False), (1899, False), (1903, False), (1904, False), (1905, True), (1908, True), (1909, False), (1910, False), (1911, False), (1912, True), (1913, False), (1914, False), (1915, False), (1920, True), (1921, False), (1922, False), (1923, False), (1924, True), (1925, True), (1926, True), (1928, True), (1930, True), (1931, False), (1933, False), (1934, False), (1935, False), (1938, False), (1939, True), (1943, True), (1946, False), (1947, True), (1948, False), (1951, False), (1953, True), (1954, False), (1955, False), (1956, False), (1957, False), (1959, False), (1960, False), (1961, True), (1962, False), (1963, True), (1966, False), (1968, False), (1969, False), (1971, False), (1974, True), (1977, True), (1979, False), (1981, False), (1984, True), (1985, False), (1986, False), (1987, False), (1988, True), (1989, False), (1990, True), (1991, False), (1996, True), (1997, False), (1998, False), (2000, True), (2001, False), (2002, False), (2003, False), (2004, False), (2005, False), (2007, False), (2009, True), (2010, False), (2012, True), (2013, True), (2017, False), (2018, False), (2020, True), (2021, True), (2025, False), (2026, True), (2027, False), (2029, True), (2030, False), (2031, False), (2032, True), (2033, True), (2034, False), (2038, True), (2039, False), (2043, False), (2044, False), (2045, False), (2046, True), (2047, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 8
    dataset = Dataset(input_bits_count, [(1, True), (3, False), (6, True), (7, True), (10, True), (13, True), (16, True), (18, True), (19, False), (20, True), (23, True), (25, True), (31, False), (32, False), (33, True), (34, True), (35, True), (38, True), (39, True), (41, True), (43, True), (46, True), (49, True), (51, False), (55, False), (56, True), (57, False), (60, True), (61, True), (65, True), (66, False), (67, True), (72, True), (73, True), (76, True), (77, True), (78, False), (79, True), (83, False), (86, False), (89, True), (90, True), (92, False), (93, False), (95, True), (96, True), (97, True), (99, True), (101, True), (102, True), (104, False), (105, False), (107, True), (108, True), (110, True), (111, True), (112, True), (113, True), (114, True), (116, False), (121, True), (122, False), (123, True), (125, True), (126, False), (130, True), (131, True), (132, True), (135, True), (136, False), (141, True), (142, True), (143, True), (149, True), (150, False), (151, True), (155, False), (156, True), (157, True), (158, False), (163, True), (167, True), (168, True), (169, True), (170, True), (171, True), (176, False), (177, True), (180, False), (182, False), (187, True), (190, True), (191, False), (193, True), (196, True), (197, True), (198, False), (201, False), (202, True), (208, False), (209, True), (210, False), (211, False), (212, True), (213, False), (217, False), (218, True), (224, False), (225, False), (227, True), (228, False), (229, True), (231, False), (234, True), (235, True), (236, False), (238, False), (240, True), (241, True), (242, True), (243, True), (244, True), (247, True), (250, True), (252, True), (254, True), (255, True)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 11
    dataset = Dataset(input_bits_count, [(1, True), (3, True), (4, True), (5, False), (7, False), (8, False), (10, True), (11, False), (13, False), (15, False), (17, False), (19, True), (22, True), (23, True), (24, False), (25, True), (30, False), (31, True), (32, True), (33, True), (34, True), (35, False), (36, True), (37, True), (39, False), (40, False), (42, True), (44, False), (46, False), (49, False), (50, False), (51, False), (53, False), (54, False), (58, False), (59, False), (60, False), (62, False), (64, True), (65, False), (67, True), (68, True), (70, False), (71, False), (73, True), (74, True), (76, False), (78, False), (79, False), (80, True), (81, False), (82, False), (84, False), (86, True), (87, False), (88, False), (89, False), (90, False), (91, False), (93, False), (95, True), (96, True), (98, True), (99, False), (100, True), (101, False), (102, False), (105, True), (106, True), (107, True), (110, True), (111, True), (116, False), (118, False), (119, True), (120, False), (121, False), (124, False), (126, False), (127, False), (128, False), (130, True), (135, False), (136, True), (138, True), (140, False), (141, True), (143, False), (144, True), (146, False), (147, True), (148, True), (149, False), (150, True), (151, False), (152, False), (154, False), (155, True), (156, True), (157, True), (158, True), (159, False), (161, False), (163, True), (165, True), (167, False), (170, False), (174, False), (177, False), (178, False), (180, True), (181, False), (182, False), (183, True), (184, False), (185, False), (186, False), (187, False), (192, False), (196, False), (202, False), (203, False), (205, True), (206, True), (207, True), (208, False), (211, False), (212, True), (213, False), (214, True), (215, False), (216, False), (218, False), (219, False), (220, False), (221, False), (226, True), (227, False), (228, False), (229, False), (230, False), (231, False), (232, False), (233, True), (234, True), (235, True), (236, True), (237, False), (240, False), (243, False), (245, False), (246, True), (247, False), (248, False), (249, False), (250, False), (251, True), (252, False), (254, False), (256, True), (257, False), (260, False), (261, False), (262, False), (263, False), (264, False), (266, True), (269, False), (270, False), (271, False), (272, True), (273, False), (274, False), (276, False), (277, True), (279, False), (280, False), (281, False), (282, True), (283, True), (284, False), (285, True), (286, True), (288, True), (289, False), (290, False), (291, True), (292, True), (293, False), (295, True), (296, False), (299, True), (301, False), (302, True), (304, True), (305, True), (307, False), (308, False), (310, False), (311, False), (313, False), (315, False), (316, True), (317, True), (318, False), (321, True), (322, False), (323, True), (324, False), (326, True), (328, False), (331, False), (332, False), (333, False), (334, True), (335, False), (338, False), (340, False), (341, True), (342, False), (343, True), (346, False), (350, False), (352, False), (353, True), (354, True), (355, True), (357, False), (358, True), (360, False), (361, False), (365, True), (366, False), (367, False), (368, False), (370, False), (372, False), (373, False), (375, False), (378, True), (379, False), (380, False), (382, True), (386, True), (391, False), (393, False), (394, True), (395, True), (396, True), (397, False), (399, True), (400, False), (401, False), (402, False), (405, False), (406, True), (408, True), (410, False), (411, False), (412, False), (413, False), (416, False), (417, False), (418, True), (419, True), (422, False), (424, False), (425, True), (426, True), (428, False), (429, True), (430, True), (431, True), (432, False), (434, False), (436, False), (438, True), (441, False), (442, True), (443, False), (445, False), (446, False), (448, True), (450, False), (452, True), (455, True), (456, False), (459, False), (461, False), (462, False), (465, True), (466, False), (467, False), (470, False), (472, True), (473, True), (474, True), (475, False), (476, False), (477, False), (479, True), (480, True), (481, True), (482, True), (483, False), (484, False), (485, True), (486, True), (487, True), (488, False), (489, False), (490, False), (491, True), (492, False), (493, True), (496, False), (497, False), (499, False), (500, False), (501, False), (503, True), (505, True), (506, False), (507, False), (508, False), (510, False), (512, False), (513, True), (516, False), (517, False), (519, False), (520, True), (522, True), (523, False), (525, False), (526, True), (527, False), (528, True), (530, False), (531, True), (532, True), (533, False), (534, True), (535, False), (536, False), (537, False), (538, False), (542, False), (545, False), (547, False), (551, True), (554, False), (555, False), (556, False), (557, False), (559, True), (560, False), (561, True), (562, False), (565, False), (566, False), (567, True), (568, False), (569, False), (570, False), (575, False), (576, True), (577, False), (578, False), (579, True), (580, False), (581, True), (582, True), (583, True), (587, False), (588, True), (593, False), (594, True), (595, True), (596, True), (601, False), (603, True), (604, True), (606, True), (607, False), (608, False), (610, False), (613, False), (614, False), (618, True), (622, True), (625, True), (626, True), (627, False), (628, True), (629, True), (632, False), (633, True), (634, False), (635, False), (637, False), (638, False), (641, False), (643, True), (645, False), (647, False), (650, False), (651, False), (656, True), (659, True), (662, True), (665, False), (666, False), (668, False), (671, False), (672, False), (674, False), (676, False), (677, True), (679, False), (680, True), (682, False), (683, True), (684, True), (685, True), (686, False), (687, False), (688, False), (689, True), (691, True), (693, True), (694, False), (696, False), (697, False), (698, True), (700, False), (702, False), (703, False), (705, False), (708, True), (710, False), (711, True), (712, False), (713, False), (715, True), (716, True), (719, True), (720, False), (722, True), (723, False), (724, False), (725, False), (728, False), (729, False), (731, False), (732, True), (733, False), (734, False), (736, False), (737, False), (738, False), (740, False), (741, False), (743, False), (747, True), (748, True), (751, False), (752, True), (754, False), (755, False), (756, False), (757, False), (758, True), (760, True), (761, False), (762, False), (763, False), (765, True), (769, True), (771, True), (772, False), (774, True), (777, False), (782, True), (784, True), (785, True), (790, True), (791, False), (797, False), (799, True), (800, True), (802, False), (803, False), (805, True), (806, True), (808, False), (809, False), (810, True), (811, True), (812, False), (814, False), (816, False), (817, False), (819, True), (820, True), (822, False), (823, False), (824, False), (826, True), (827, True), (828, False), (829, True), (830, False), (831, False), (832, False), (833, True), (835, False), (837, True), (838, True), (839, True), (841, True), (842, False), (843, False), (846, True), (849, False), (850, False), (851, True), (854, False), (855, False), (856, False), (859, False), (860, True), (861, False), (862, True), (863, True), (864, True), (865, True), (866, True), (868, False), (871, False), (872, True), (873, False), (878, False), (879, False), (880, True), (881, False), (883, True), (886, False), (887, True), (890, False), (891, False), (892, False), (893, True), (894, False), (895, True), (896, True), (898, False), (899, False), (900, False), (903, False), (904, False), (905, False), (906, False), (907, True), (910, True), (911, True), (914, True), (915, False), (916, False), (917, False), (918, False), (919, True), (922, False), (923, False), (926, True), (927, True), (929, False), (931, False), (932, False), (933, False), (934, False), (935, False), (936, False), (938, False), (939, False), (940, False), (942, True), (944, True), (946, False), (947, False), (948, False), (949, True), (950, False), (951, False), (953, True), (954, True), (956, True), (958, False), (960, False), (962, False), (963, True), (964, True), (965, True), (967, False), (968, False), (969, False), (970, False), (972, False), (974, True), (975, True), (976, False), (977, True), (979, True), (980, True), (981, True), (982, False), (983, False), (984, True), (985, True), (986, True), (987, False), (988, False), (990, True), (991, True), (992, False), (993, True), (994, True), (995, False), (997, False), (998, False), (1002, False), (1003, False), (1004, False), (1006, True), (1009, False), (1011, False), (1012, False), (1013, True), (1016, False), (1019, False), (1020, True), (1023, True), (1024, False), (1025, False), (1026, True), (1027, False), (1028, True), (1030, True), (1031, False), (1032, True), (1033, True), (1036, True), (1037, False), (1038, True), (1040, False), (1042, False), (1043, True), (1044, False), (1046, True), (1047, True), (1048, False), (1050, False), (1052, True), (1054, False), (1056, False), (1057, False), (1059, True), (1060, False), (1061, False), (1062, False), (1065, False), (1067, False), (1068, True), (1070, True), (1072, True), (1073, False), (1077, False), (1078, False), (1079, False), (1081, False), (1083, False), (1084, False), (1085, True), (1086, False), (1087, False), (1088, False), (1090, False), (1091, True), (1092, True), (1093, True), (1094, False), (1095, False), (1097, False), (1098, True), (1099, True), (1103, True), (1104, False), (1105, False), (1106, False), (1107, False), (1108, False), (1109, False), (1111, False), (1112, False), (1114, False), (1115, False), (1116, False), (1117, True), (1119, False), (1120, False), (1121, False), (1122, True), (1123, True), (1124, True), (1125, True), (1126, False), (1129, False), (1130, True), (1131, False), (1132, False), (1133, False), (1134, False), (1135, True), (1136, False), (1137, False), (1138, True), (1139, True), (1140, False), (1141, False), (1143, False), (1144, True), (1146, True), (1147, False), (1149, False), (1150, False), (1153, True), (1154, False), (1155, False), (1156, False), (1157, False), (1159, True), (1160, False), (1162, False), (1163, False), (1164, False), (1165, True), (1166, False), (1167, True), (1168, False), (1171, False), (1173, True), (1174, False), (1175, True), (1176, True), (1178, True), (1179, False), (1180, True), (1182, True), (1183, False), (1184, False), (1185, True), (1187, False), (1189, True), (1191, True), (1192, False), (1193, False), (1194, False), (1195, False), (1197, False), (1198, False), (1199, False), (1200, False), (1202, False), (1204, False), (1205, True), (1206, False), (1207, False), (1208, False), (1209, False), (1210, False), (1211, True), (1212, True), (1214, True), (1215, True), (1216, True), (1218, True), (1221, True), (1222, False), (1223, False), (1225, False), (1226, False), (1227, False), (1228, True), (1229, False), (1232, False), (1233, False), (1234, False), (1235, False), (1236, True), (1239, False), (1240, True), (1241, False), (1245, False), (1247, False), (1249, True), (1250, True), (1254, True), (1255, False), (1260, False), (1263, False), (1264, False), (1266, False), (1267, False), (1269, False), (1271, False), (1272, False), (1277, False), (1278, False), (1279, False), (1280, False), (1282, False), (1283, False), (1284, False), (1285, True), (1286, True), (1287, False), (1288, True), (1289, False), (1290, False), (1291, True), (1292, False), (1294, False), (1296, True), (1297, True), (1299, False), (1301, False), (1302, True), (1303, False), (1304, True), (1306, True), (1308, True), (1309, False), (1310, False), (1311, False), (1313, False), (1315, True), (1317, False), (1318, True), (1320, False), (1322, False), (1323, True), (1325, False), (1326, False), (1327, True), (1328, False), (1329, True), (1331, False), (1332, True), (1334, False), (1335, False), (1336, False), (1337, False), (1339, False), (1340, False), (1341, False), (1342, False), (1343, False), (1344, True), (1346, False), (1347, True), (1348, True), (1350, False), (1351, True), (1352, False), (1353, True), (1354, False), (1355, False), (1356, True), (1358, False), (1359, False), (1360, False), (1361, True), (1362, True), (1363, True), (1365, False), (1371, True), (1372, False), (1373, True), (1375, False), (1377, False), (1379, False), (1380, False), (1383, False), (1386, False), (1388, False), (1391, True), (1394, True), (1395, False), (1396, False), (1398, True), (1399, False), (1400, False), (1401, False), (1404, False), (1405, False), (1406, False), (1408, True), (1409, True), (1411, False), (1414, True), (1415, False), (1416, True), (1417, False), (1418, True), (1420, False), (1421, True), (1422, False), (1424, False), (1426, False), (1429, True), (1430, False), (1432, False), (1433, False), (1437, False), (1440, True), (1442, True), (1443, False), (1444, False), (1445, False), (1446, False), (1447, False), (1448, False), (1449, False), (1451, False), (1452, True), (1453, True), (1454, True), (1455, False), (1456, True), (1457, False), (1458, True), (1461, False), (1462, True), (1463, True), (1464, True), (1465, True), (1466, False), (1468, True), (1469, False), (1470, True), (1471, False), (1472, False), (1473, True), (1475, False), (1478, False), (1480, False), (1483, False), (1484, False), (1486, True), (1489, False), (1491, False), (1492, False), (1494, False), (1496, False), (1497, False), (1500, True), (1502, False), (1503, False), (1504, False), (1506, False), (1507, True), (1508, True), (1510, True), (1512, False), (1517, True), (1518, False), (1521, True), (1523, True), (1524, False), (1525, False), (1526, False), (1527, False), (1528, False), (1529, True), (1530, False), (1531, False), (1532, False), (1535, True), (1537, False), (1539, False), (1540, True), (1541, False), (1544, True), (1549, True), (1550, False), (1551, True), (1556, True), (1557, False), (1558, False), (1561, False), (1565, True), (1566, True), (1567, False), (1569, False), (1570, False), (1572, True), (1574, False), (1575, True), (1576, False), (1577, True), (1578, True), (1581, False), (1583, False), (1584, True), (1586, True), (1589, False), (1590, False), (1591, False), (1592, False), (1593, False), (1594, False), (1595, True), (1597, False), (1600, False), (1601, False), (1602, False), (1603, False), (1605, False), (1606, True), (1608, False), (1609, True), (1611, False), (1612, False), (1613, False), (1614, True), (1615, False), (1620, True), (1621, False), (1622, False), (1623, False), (1624, False), (1629, False), (1630, False), (1633, False), (1634, False), (1636, True), (1637, False), (1639, True), (1640, False), (1642, False), (1643, False), (1646, False), (1648, False), (1650, False), (1653, False), (1654, True), (1655, True), (1656, False), (1657, False), (1660, True), (1665, False), (1667, True), (1668, False), (1670, False), (1672, True), (1673, False), (1674, False), (1675, False), (1676, False), (1681, True), (1682, False), (1683, False), (1684, True), (1685, False), (1689, False), (1690, False), (1691, False), (1692, True), (1693, False), (1695, True), (1696, True), (1697, False), (1698, False), (1701, True), (1702, False), (1704, True), (1706, True), (1707, True), (1708, True), (1709, False), (1711, False), (1712, False), (1713, False), (1714, False), (1717, False), (1718, False), (1719, False), (1722, False), (1724, False), (1727, False), (1728, False), (1729, False), (1730, False), (1733, True), (1734, False), (1735, True), (1739, True), (1740, False), (1743, False), (1744, False), (1746, False), (1747, True), (1749, False), (1751, True), (1756, True), (1757, True), (1758, False), (1764, True), (1765, False), (1766, False), (1768, False), (1769, False), (1770, False), (1772, False), (1774, True), (1775, False), (1776, False), (1779, False), (1781, False), (1785, True), (1788, True), (1790, True), (1795, True), (1796, False), (1797, False), (1798, True), (1799, False), (1801, True), (1802, True), (1804, True), (1805, True), (1807, False), (1808, True), (1809, False), (1810, True), (1811, False), (1812, True), (1813, False), (1814, False), (1815, True), (1816, True), (1817, False), (1818, True), (1819, False), (1821, True), (1823, False), (1824, False), (1825, False), (1827, True), (1828, True), (1829, False), (1833, False), (1834, True), (1835, True), (1836, False), (1838, False), (1839, True), (1841, False), (1844, False), (1845, True), (1846, True), (1848, False), (1849, False), (1851, False), (1852, False), (1854, True), (1856, False), (1857, False), (1858, False), (1859, True), (1861, True), (1863, False), (1864, True), (1865, False), (1867, True), (1868, False), (1870, True), (1871, True), (1872, False), (1874, True), (1875, True), (1876, False), (1877, False), (1878, True), (1881, True), (1883, True), (1884, False), (1885, False), (1886, True), (1888, True), (1890, True), (1891, True), (1892, False), (1893, False), (1894, True), (1895, False), (1897, False), (1899, False), (1901, False), (1902, False), (1903, False), (1904, False), (1905, False), (1909, False), (1911, False), (1912, True), (1914, False), (1915, True), (1916, False), (1917, True), (1919, False), (1920, True), (1921, True), (1924, True), (1926, False), (1929, False), (1930, True), (1931, False), (1932, True), (1937, True), (1938, False), (1941, False), (1942, False), (1944, False), (1945, False), (1948, False), (1949, False), (1950, True), (1953, True), (1955, False), (1958, False), (1959, True), (1960, True), (1961, False), (1962, False), (1963, False), (1965, False), (1967, False), (1968, False), (1969, True), (1971, False), (1972, False), (1973, True), (1974, False), (1975, True), (1976, True), (1977, True), (1978, True), (1982, True), (1983, True), (1984, True), (1985, False), (1988, True), (1990, True), (1992, True), (1993, True), (1995, True), (1997, True), (1998, False), (1999, False), (2000, False), (2001, True), (2004, True), (2006, True), (2007, False), (2008, False), (2009, True), (2010, False), (2011, True), (2012, False), (2013, False), (2014, False), (2016, False), (2019, True), (2021, True), (2022, True), (2023, False), (2025, False), (2026, False), (2027, True), (2029, True), (2032, True), (2033, True), (2034, True), (2037, False), (2038, False), (2040, False), (2043, False), (2044, False), (2047, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 11
    dataset = Dataset(input_bits_count, [(5, True), (6, True), (7, False), (8, True), (9, True), (11, True), (12, False), (16, True), (19, True), (20, True), (22, True), (24, True), (25, False), (28, True), (29, True), (32, True), (33, False), (34, True), (36, True), (37, False), (40, True), (41, True), (43, False), (46, True), (48, False), (49, True), (50, True), (51, False), (53, True), (56, False), (57, True), (59, True), (60, False), (63, True), (65, True), (67, False), (68, False), (69, True), (70, False), (73, True), (76, False), (77, True), (78, True), (80, True), (81, True), (83, True), (85, False), (87, True), (88, True), (89, False), (90, False), (93, False), (94, True), (95, False), (96, True), (98, False), (101, True), (104, True), (105, False), (107, True), (108, True), (109, True), (110, False), (112, False), (113, True), (115, False), (117, True), (119, True), (120, True), (121, False), (122, True), (123, False), (125, False), (126, False), (127, False), (129, True), (130, True), (131, True), (134, True), (136, False), (137, True), (140, True), (141, True), (145, False), (146, True), (147, True), (148, False), (149, True), (151, True), (152, True), (153, True), (154, True), (155, True), (157, True), (158, False), (160, True), (164, True), (165, True), (167, True), (169, True), (172, False), (173, False), (174, True), (175, False), (176, False), (178, True), (182, False), (185, True), (186, True), (187, False), (188, True), (189, True), (191, True), (192, True), (193, True), (194, True), (197, True), (198, True), (200, True), (202, True), (203, False), (204, False), (205, False), (206, True), (207, False), (209, True), (211, False), (214, False), (218, True), (219, False), (220, False), (221, True), (222, False), (223, False), (224, True), (227, False), (228, True), (229, True), (231, False), (232, False), (235, True), (236, True), (237, True), (238, True), (239, True), (241, True), (244, True), (248, True), (249, True), (250, False), (251, True), (253, True), (255, True), (256, True), (257, True), (260, True), (261, True), (262, True), (264, True), (265, True), (267, False), (268, True), (269, True), (271, True), (272, False), (273, True), (274, False), (275, False), (276, True), (277, False), (279, True), (281, True), (282, False), (284, True), (285, True), (291, True), (295, False), (297, True), (300, True), (304, False), (307, True), (309, False), (310, True), (311, True), (314, False), (318, True), (319, True), (320, True), (321, True), (322, True), (324, True), (325, True), (328, True), (329, True), (331, False), (332, False), (336, True), (337, True), (338, True), (340, True), (341, True), (342, False), (344, True), (345, True), (346, True), (347, True), (348, False), (350, False), (353, True), (356, True), (361, True), (363, True), (364, False), (365, True), (369, True), (370, True), (374, False), (376, True), (378, True), (380, True), (381, True), (383, True), (384, True), (385, True), (387, False), (388, True), (391, True), (392, True), (393, True), (396, False), (399, False), (400, True), (403, True), (404, True), (407, True), (408, True), (409, True), (414, True), (416, False), (417, True), (419, True), (420, True), (427, False), (428, False), (431, True), (432, False), (433, True), (437, True), (438, True), (439, True), (440, True), (441, False), (442, False), (443, True), (446, True), (449, False), (450, True), (451, False), (452, False), (453, True), (454, True), (455, True), (456, True), (457, True), (458, False), (459, True), (461, True), (463, True), (464, True), (466, True), (468, False), (469, False), (470, True), (471, True), (475, False), (476, True), (477, True), (480, True), (481, False), (483, True), (484, True), (486, False), (487, True), (489, False), (492, False), (494, True), (496, False), (498, True), (499, True), (500, False), (501, False), (502, True), (506, True), (507, False), (508, False), (509, True), (511, True), (512, True), (513, True), (516, False), (518, False), (519, True), (521, False), (522, False), (525, True), (526, False), (528, False), (529, False), (530, True), (531, True), (536, True), (538, True), (540, True), (541, True), (542, False), (543, True), (545, True), (547, False), (548, True), (549, False), (550, False), (551, True), (552, False), (554, False), (556, True), (558, True), (563, True), (564, False), (565, True), (568, True), (569, True), (570, True), (573, False), (576, True), (578, False), (579, True), (580, False), (581, True), (582, True), (583, True), (584, False), (585, True), (586, False), (589, True), (590, True), (591, False), (592, True), (594, False), (596, True), (597, True), (599, False), (602, True), (611, True), (612, True), (613, True), (614, False), (615, True), (617, True), (619, False), (621, False), (623, True), (625, True), (626, True), (627, True), (628, False), (629, False), (630, False), (631, True), (632, True), (633, True), (634, True), (635, True), (636, True), (638, True), (639, True), (641, True), (643, True), (645, False), (648, True), (649, False), (651, True), (653, True), (654, True), (656, True), (657, False), (658, True), (659, True), (660, True), (661, False), (662, True), (663, True), (665, False), (666, True), (668, True), (670, False), (671, True), (673, True), (675, False), (676, True), (681, True), (683, True), (684, False), (686, True), (687, False), (688, False), (689, True), (692, False), (693, False), (695, False), (696, False), (697, False), (698, False), (700, False), (702, True), (704, False), (705, False), (706, True), (707, True), (708, False), (710, True), (711, True), (717, True), (718, True), (719, False), (720, True), (721, False), (722, True), (723, True), (724, True), (726, True), (729, True), (732, False), (733, True), (734, True), (735, True), (736, True), (738, False), (739, True), (740, True), (742, True), (743, True), (749, True), (750, True), (751, False), (752, False), (753, False), (755, False), (756, True), (759, False), (760, True), (761, False), (763, False), (765, True), (770, False), (771, True), (773, False), (775, True), (776, False), (777, True), (778, True), (780, False), (783, True), (787, True), (790, True), (791, True), (792, True), (793, False), (798, True), (799, True), (800, False), (802, True), (803, True), (806, True), (810, False), (811, True), (812, True), (813, False), (814, True), (815, True), (817, True), (818, True), (819, True), (821, True), (822, False), (823, True), (824, False), (825, False), (826, True), (827, True), (831, True), (832, True), (833, True), (835, False), (836, False), (838, True), (843, True), (845, True), (849, True), (850, False), (852, True), (854, True), (855, True), (856, True), (857, False), (858, True), (859, False), (860, True), (862, True), (863, True), (864, False), (867, False), (868, False), (869, False), (870, True), (876, True), (877, False), (878, True), (884, True), (885, False), (886, False), (887, True), (888, True), (889, True), (891, False), (892, True), (897, True), (898, True), (899, True), (904, False), (911, False), (913, False), (914, False), (915, True), (916, True), (919, True), (921, True), (922, True), (925, False), (927, True), (929, True), (930, False), (932, True), (934, True), (936, True), (937, False), (940, False), (941, True), (943, False), (944, False), (945, False), (947, False), (948, True), (949, True), (951, True), (952, True), (954, False), (956, True), (958, False), (960, False), (968, True), (969, True), (970, True), (973, True), (974, True), (975, False), (977, True), (978, False), (979, True), (980, True), (981, False), (982, False), (983, True), (985, False), (986, True), (987, True), (988, True), (989, False), (992, True), (994, False), (1000, True), (1001, False), (1003, True), (1005, False), (1006, True), (1007, False), (1008, True), (1009, False), (1010, True), (1011, False), (1012, True), (1014, True), (1015, False), (1019, True), (1020, True), (1022, False), (1023, True), (1024, False), (1030, True), (1031, False), (1032, True), (1035, True), (1036, True), (1037, True), (1038, True), (1039, True), (1040, False), (1041, True), (1042, True), (1044, True), (1046, False), (1047, True), (1049, True), (1051, True), (1052, True), (1053, True), (1054, False), (1056, True), (1057, False), (1058, False), (1059, True), (1060, True), (1064, True), (1065, True), (1066, True), (1068, False), (1070, False), (1071, True), (1072, False), (1073, True), (1074, True), (1080, False), (1081, False), (1082, True), (1083, False), (1085, False), (1090, True), (1091, True), (1092, True), (1095, True), (1096, False), (1099, True), (1100, False), (1101, True), (1102, True), (1103, True), (1105, True), (1106, True), (1107, True), (1108, False), (1111, False), (1112, True), (1113, True), (1114, True), (1116, True), (1117, True), (1119, True), (1122, True), (1123, False), (1124, True), (1126, True), (1127, True), (1128, True), (1129, False), (1131, True), (1133, False), (1136, False), (1138, False), (1139, False), (1140, True), (1141, True), (1142, False), (1143, True), (1144, False), (1145, False), (1147, False), (1148, True), (1152, True), (1153, False), (1155, False), (1158, True), (1159, True), (1160, True), (1162, False), (1164, False), (1165, True), (1166, False), (1169, True), (1171, True), (1172, True), (1176, False), (1178, True), (1179, False), (1182, True), (1183, True), (1184, True), (1185, True), (1187, False), (1188, False), (1189, True), (1190, True), (1191, True), (1192, True), (1196, False), (1197, False), (1198, False), (1200, True), (1201, True), (1203, True), (1204, True), (1206, True), (1209, True), (1210, False), (1211, True), (1215, True), (1216, True), (1217, True), (1219, True), (1221, True), (1222, True), (1223, False), (1227, True), (1228, True), (1231, True), (1236, True), (1237, True), (1238, False), (1241, True), (1242, True), (1246, False), (1250, True), (1251, True), (1252, True), (1253, True), (1256, True), (1257, True), (1258, True), (1259, False), (1262, True), (1264, True), (1265, True), (1267, True), (1271, False), (1274, True), (1276, True), (1278, False), (1280, True), (1281, True), (1282, True), (1283, True), (1285, False), (1286, True), (1288, True), (1289, False), (1290, True), (1291, True), (1293, False), (1294, True), (1296, True), (1300, True), (1302, True), (1303, False), (1304, True), (1308, False), (1309, True), (1311, True), (1312, False), (1313, True), (1315, True), (1316, True), (1317, False), (1319, True), (1320, True), (1323, True), (1325, True), (1327, False), (1328, True), (1329, False), (1333, False), (1334, True), (1335, True), (1336, False), (1337, True), (1339, True), (1340, True), (1341, False), (1342, False), (1345, False), (1346, False), (1347, True), (1348, True), (1349, False), (1350, False), (1352, True), (1354, True), (1356, True), (1357, True), (1358, True), (1359, True), (1361, True), (1362, True), (1363, True), (1364, True), (1366, True), (1368, True), (1369, False), (1370, True), (1372, True), (1373, False), (1375, False), (1376, True), (1377, False), (1378, True), (1379, False), (1381, True), (1382, False), (1383, True), (1385, True), (1386, True), (1387, False), (1388, False), (1389, True), (1390, True), (1392, True), (1394, True), (1396, False), (1399, True), (1400, True), (1403, True), (1404, True), (1406, False), (1408, True), (1409, False), (1410, True), (1411, True), (1413, False), (1416, False), (1417, True), (1420, False), (1421, True), (1422, False), (1426, False), (1427, True), (1428, False), (1430, True), (1432, False), (1436, False), (1439, True), (1442, True), (1444, True), (1445, False), (1446, False), (1447, True), (1448, True), (1451, False), (1452, True), (1453, True), (1454, False), (1455, False), (1456, False), (1457, False), (1459, False), (1460, True), (1462, False), (1465, True), (1467, True), (1468, True), (1470, True), (1475, True), (1478, False), (1479, True), (1481, True), (1482, True), (1483, False), (1484, True), (1488, True), (1489, True), (1491, False), (1492, False), (1493, False), (1495, True), (1496, False), (1497, False), (1502, True), (1505, True), (1508, True), (1509, True), (1510, False), (1513, True), (1515, True), (1516, True), (1519, True), (1523, False), (1527, False), (1529, True), (1533, True), (1535, True), (1537, True), (1538, True), (1539, True), (1543, True), (1544, False), (1545, True), (1549, True), (1550, True), (1554, True), (1555, True), (1556, False), (1558, True), (1559, True), (1561, False), (1562, True), (1563, False), (1564, True), (1567, True), (1568, True), (1569, True), (1573, True), (1574, True), (1577, True), (1578, True), (1580, True), (1581, True), (1582, True), (1583, True), (1584, True), (1585, True), (1587, True), (1588, False), (1590, False), (1594, False), (1595, False), (1596, True), (1597, True), (1598, False), (1599, False), (1601, False), (1607, True), (1608, False), (1609, True), (1610, True), (1611, True), (1612, True), (1613, False), (1614, False), (1615, False), (1620, False), (1623, True), (1624, False), (1625, False), (1626, False), (1627, True), (1628, True), (1629, True), (1631, True), (1635, True), (1636, False), (1637, True), (1641, True), (1642, False), (1643, True), (1644, False), (1646, True), (1648, True), (1649, True), (1652, False), (1653, False), (1654, True), (1656, True), (1657, True), (1658, True), (1659, True), (1661, True), (1662, False), (1663, False), (1665, False), (1668, True), (1669, False), (1671, False), (1672, True), (1674, False), (1677, False), (1680, True), (1682, False), (1684, True), (1686, False), (1689, True), (1690, True), (1691, True), (1693, False), (1694, True), (1696, True), (1697, False), (1699, False), (1701, True), (1702, True), (1706, True), (1708, True), (1709, False), (1710, True), (1711, True), (1713, False), (1715, False), (1717, True), (1718, True), (1719, True), (1720, True), (1723, True), (1724, True), (1725, False), (1727, False), (1728, False), (1729, False), (1730, True), (1731, False), (1732, True), (1733, False), (1735, True), (1736, True), (1737, False), (1739, True), (1740, True), (1742, True), (1744, True), (1745, True), (1746, False), (1747, True), (1748, False), (1749, True), (1752, True), (1753, True), (1755, True), (1756, True), (1758, True), (1760, True), (1761, False), (1763, False), (1764, False), (1765, False), (1767, False), (1769, False), (1770, True), (1771, False), (1772, False), (1773, False), (1774, True), (1776, False), (1777, True), (1778, True), (1779, True), (1780, False), (1781, False), (1784, True), (1785, True), (1786, True), (1787, False), (1788, False), (1789, False), (1791, True), (1792, False), (1793, False), (1796, False), (1797, True), (1798, True), (1799, False), (1800, False), (1801, True), (1802, True), (1803, True), (1807, True), (1811, True), (1814, True), (1815, True), (1816, True), (1819, True), (1820, False), (1822, True), (1823, True), (1827, False), (1828, True), (1829, True), (1831, True), (1832, True), (1833, True), (1835, True), (1836, True), (1837, True), (1839, False), (1840, True), (1841, True), (1843, True), (1845, True), (1846, True), (1847, False), (1848, False), (1851, True), (1852, True), (1854, True), (1855, True), (1856, True), (1859, True), (1860, False), (1861, False), (1862, True), (1868, True), (1870, False), (1872, True), (1874, False), (1877, False), (1878, False), (1880, True), (1881, True), (1883, False), (1885, True), (1886, False), (1888, False), (1889, False), (1891, True), (1892, False), (1893, False), (1894, True), (1898, True), (1901, True), (1903, True), (1906, True), (1907, False), (1909, True), (1910, True), (1911, True), (1912, True), (1913, True), (1914, True), (1915, True), (1916, False), (1918, True), (1919, True), (1920, True), (1921, True), (1923, True), (1924, False), (1925, True), (1927, True), (1928, True), (1931, True), (1933, True), (1934, True), (1935, True), (1937, False), (1943, True), (1944, True), (1945, False), (1947, True), (1950, False), (1951, True), (1952, False), (1953, False), (1955, False), (1957, True), (1959, False), (1961, True), (1963, True), (1965, True), (1966, True), (1968, False), (1970, True), (1971, True), (1973, True), (1974, True), (1975, False), (1976, True), (1978, True), (1979, True), (1981, True), (1983, True), (1985, True), (1986, False), (1990, True), (1991, False), (1996, True), (1998, True), (2000, True), (2002, True), (2004, True), (2005, True), (2010, False), (2011, False), (2013, False), (2017, False), (2019, False), (2022, True), (2023, False), (2025, True), (2029, False), (2030, True), (2032, False), (2034, False), (2036, True), (2039, True), (2041, False), (2042, True), (2046, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 11
    dataset = Dataset(input_bits_count, [(0, False), (1, True), (2, True), (4, True), (5, True), (6, False), (7, True), (11, True), (12, False), (13, True), (15, True), (18, True), (19, True), (21, False), (23, True), (24, False), (25, True), (27, True), (28, True), (31, False), (33, True), (34, True), (36, False), (37, False), (39, False), (40, True), (43, False), (44, True), (45, True), (46, False), (47, False), (48, True), (49, False), (51, True), (52, True), (54, False), (57, True), (58, False), (59, False), (60, True), (61, True), (63, True), (64, True), (66, True), (68, False), (70, True), (73, False), (74, True), (75, False), (76, True), (78, False), (79, True), (80, True), (82, True), (85, False), (86, False), (87, False), (88, False), (89, True), (90, False), (91, False), (94, False), (97, True), (98, False), (100, False), (101, False), (102, False), (103, True), (107, True), (111, True), (112, True), (113, False), (114, True), (116, True), (118, True), (119, False), (120, True), (123, True), (124, False), (126, False), (130, False), (131, True), (132, False), (134, True), (135, False), (136, True), (138, True), (139, True), (142, True), (143, False), (146, False), (149, False), (152, False), (155, True), (157, False), (158, False), (159, True), (160, True), (164, True), (169, True), (170, True), (171, False), (173, False), (174, False), (178, True), (181, False), (182, True), (183, False), (184, True), (185, True), (187, True), (188, True), (191, False), (192, True), (195, False), (197, True), (199, False), (200, True), (201, False), (203, False), (204, False), (207, True), (208, False), (212, False), (213, True), (216, False), (218, False), (219, True), (220, False), (222, True), (223, False), (224, False), (227, False), (228, True), (230, False), (232, True), (233, True), (235, True), (236, True), (237, True), (238, False), (239, True), (240, True), (241, True), (243, False), (244, False), (245, False), (246, True), (247, False), (248, True), (249, True), (250, True), (251, False), (252, False), (255, True), (257, False), (263, True), (266, True), (268, False), (270, True), (273, False), (274, True), (279, False), (280, True), (281, False), (282, True), (284, False), (286, False), (287, True), (288, False), (291, False), (294, False), (297, True), (299, False), (300, False), (301, True), (302, False), (303, True), (305, True), (306, False), (307, False), (309, False), (313, True), (314, True), (315, True), (316, False), (317, False), (318, True), (319, False), (322, True), (323, True), (325, True), (326, False), (328, True), (329, True), (331, True), (332, False), (333, False), (334, True), (336, False), (337, False), (338, False), (339, True), (341, False), (345, False), (346, True), (347, True), (350, False), (353, True), (355, True), (356, True), (357, True), (358, True), (359, False), (361, True), (363, False), (364, True), (367, False), (368, False), (369, True), (370, False), (371, True), (372, False), (376, True), (377, True), (381, True), (382, False), (384, False), (386, True), (387, False), (388, False), (392, False), (394, True), (396, True), (397, True), (399, False), (400, False), (401, False), (402, True), (404, True), (405, True), (408, True), (409, False), (413, True), (419, True), (423, False), (425, False), (426, False), (429, True), (431, True), (433, True), (434, False), (437, True), (438, False), (441, True), (442, False), (443, True), (445, True), (451, False), (452, True), (455, False), (456, True), (461, False), (466, True), (468, False), (469, False), (471, False), (473, True), (475, True), (476, False), (477, False), (479, True), (481, True), (482, True), (483, False), (486, False), (488, True), (489, False), (491, True), (492, True), (493, True), (496, False), (498, True), (499, False), (501, False), (504, True), (505, True), (506, False), (510, False), (511, True), (516, False), (517, True), (519, True), (520, False), (521, False), (522, True), (523, True), (524, True), (525, True), (526, False), (527, True), (528, True), (529, True), (532, True), (533, True), (535, False), (536, False), (538, True), (539, False), (540, True), (544, True), (545, True), (546, True), (547, True), (548, True), (551, True), (552, False), (553, False), (557, True), (558, False), (559, True), (560, True), (564, True), (566, True), (567, False), (568, False), (569, True), (570, True), (572, False), (573, True), (574, True), (575, True), (577, True), (579, True), (582, False), (583, False), (585, True), (587, False), (588, True), (589, True), (590, True), (591, False), (593, True), (594, True), (596, False), (598, False), (599, False), (600, True), (602, True), (603, True), (604, False), (606, False), (611, True), (612, True), (617, False), (619, False), (621, False), (622, True), (624, False), (625, False), (628, False), (629, True), (631, True), (634, False), (635, False), (636, True), (639, True), (640, True), (641, True), (642, False), (643, False), (644, True), (647, False), (649, True), (652, False), (653, False), (655, True), (657, False), (658, True), (659, True), (660, True), (662, False), (663, False), (665, False), (667, False), (668, True), (669, True), (673, False), (675, True), (677, False), (680, True), (682, False), (689, False), (690, True), (691, False), (693, False), (694, True), (695, True), (697, False), (699, False), (700, True), (701, True), (702, True), (703, False), (704, False), (705, False), (706, False), (707, True), (708, True), (709, True), (710, True), (714, False), (718, True), (719, False), (720, True), (721, False), (722, True), (723, True), (724, True), (725, False), (730, False), (731, True), (732, False), (733, False), (736, True), (737, True), (738, True), (740, False), (742, False), (743, True), (744, True), (747, True), (748, True), (749, True), (750, True), (753, True), (754, True), (755, True), (756, True), (757, True), (758, False), (761, True), (763, True), (764, False), (765, False), (767, True), (770, False), (771, True), (773, False), (774, True), (775, True), (777, False), (778, True), (780, True), (784, True), (786, False), (789, True), (792, True), (793, True), (794, False), (795, False), (796, True), (797, False), (800, False), (801, True), (802, True), (803, True), (807, True), (808, True), (809, True), (810, True), (811, False), (814, False), (816, False), (818, False), (819, True), (820, False), (821, True), (822, True), (823, True), (824, True), (825, False), (827, True), (829, False), (830, True), (832, True), (833, False), (836, True), (837, True), (838, False), (842, True), (843, True), (844, True), (846, False), (847, False), (848, True), (849, True), (850, True), (851, False), (853, True), (856, True), (857, True), (858, False), (859, False), (862, False), (864, True), (866, False), (867, True), (869, True), (870, True), (872, False), (874, True), (875, True), (878, True), (880, True), (881, True), (882, True), (883, False), (887, True), (888, True), (889, False), (893, False), (894, True), (899, True), (902, True), (904, True), (905, True), (906, True), (908, False), (909, True), (911, True), (912, True), (913, True), (915, True), (916, True), (917, False), (919, True), (922, True), (923, True), (927, True), (931, True), (932, True), (933, False), (935, True), (936, False), (937, False), (938, True), (941, True), (942, True), (945, False), (946, True), (947, False), (948, False), (949, False), (950, True), (952, True), (958, False), (959, False), (963, False), (964, True), (965, False), (967, True), (968, False), (969, False), (971, True), (973, False), (975, False), (976, False), (977, False), (978, False), (979, False), (980, True), (981, False), (982, False), (987, True), (989, False), (992, True), (993, False), (996, True), (998, False), (1000, True), (1002, True), (1003, True), (1004, False), (1008, True), (1009, True), (1010, False), (1011, False), (1012, True), (1013, False), (1015, False), (1016, True), (1019, True), (1020, True), (1021, True), (1022, True), (1023, True), (1025, True), (1027, False), (1031, True), (1032, True), (1034, True), (1037, True), (1038, True), (1039, False), (1042, False), (1046, True), (1049, False), (1051, False), (1052, False), (1053, True), (1054, True), (1056, False), (1057, True), (1058, False), (1059, True), (1062, False), (1063, True), (1069, True), (1071, False), (1072, False), (1073, True), (1075, False), (1076, True), (1079, False), (1083, True), (1084, True), (1085, False), (1087, True), (1088, False), (1089, True), (1090, True), (1091, True), (1092, True), (1093, False), (1097, True), (1098, True), (1099, True), (1100, False), (1101, False), (1102, False), (1105, True), (1106, False), (1108, True), (1109, False), (1110, True), (1111, False), (1112, False), (1116, False), (1118, False), (1121, False), (1123, True), (1124, False), (1125, True), (1126, False), (1127, True), (1128, True), (1130, False), (1131, False), (1132, True), (1133, False), (1135, False), (1138, True), (1139, True), (1141, False), (1142, True), (1144, True), (1145, False), (1147, True), (1148, True), (1149, False), (1150, False), (1151, True), (1153, False), (1155, True), (1156, True), (1157, False), (1159, True), (1161, False), (1163, False), (1166, False), (1167, False), (1168, True), (1169, False), (1170, False), (1171, False), (1172, False), (1173, False), (1175, True), (1177, True), (1178, True), (1182, True), (1183, True), (1187, True), (1188, False), (1191, True), (1193, True), (1194, True), (1195, False), (1196, True), (1198, False), (1199, True), (1200, True), (1203, True), (1206, False), (1207, True), (1210, True), (1213, False), (1216, False), (1218, False), (1219, False), (1220, False), (1222, True), (1224, False), (1225, False), (1226, True), (1227, False), (1231, True), (1232, True), (1235, False), (1236, False), (1237, False), (1238, True), (1241, True), (1243, True), (1244, False), (1246, True), (1248, True), (1250, False), (1251, False), (1253, True), (1254, False), (1256, True), (1262, True), (1263, False), (1264, False), (1265, False), (1268, False), (1272, False), (1273, False), (1274, True), (1275, True), (1276, True), (1277, False), (1278, False), (1279, True), (1283, True), (1285, False), (1286, False), (1288, False), (1289, True), (1290, True), (1295, True), (1296, False), (1297, False), (1300, False), (1303, True), (1306, False), (1310, True), (1312, True), (1313, True), (1316, True), (1318, True), (1319, False), (1320, True), (1321, False), (1322, False), (1323, True), (1324, False), (1325, False), (1326, True), (1328, True), (1330, False), (1331, True), (1333, False), (1334, False), (1335, False), (1338, True), (1341, True), (1342, False), (1343, True), (1344, True), (1345, True), (1346, False), (1349, False), (1350, True), (1351, True), (1352, False), (1356, False), (1357, False), (1360, False), (1361, True), (1362, True), (1363, True), (1364, True), (1368, False), (1369, True), (1372, True), (1373, False), (1375, True), (1379, True), (1380, False), (1385, True), (1386, True), (1387, True), (1388, False), (1389, True), (1390, False), (1394, True), (1395, False), (1396, True), (1397, True), (1402, False), (1404, True), (1406, True), (1407, False), (1408, False), (1411, True), (1412, True), (1414, True), (1415, False), (1417, False), (1419, False), (1421, False), (1422, True), (1423, False), (1424, True), (1425, False), (1428, True), (1429, True), (1431, True), (1434, True), (1436, False), (1437, False), (1440, False), (1444, True), (1445, False), (1449, True), (1452, True), (1455, False), (1457, False), (1459, True), (1462, False), (1465, True), (1466, False), (1471, False), (1474, False), (1475, False), (1478, True), (1481, True), (1482, True), (1483, True), (1484, False), (1485, True), (1489, True), (1492, True), (1494, True), (1495, True), (1496, False), (1497, False), (1498, True), (1500, False), (1502, True), (1503, False), (1504, True), (1506, True), (1507, False), (1511, False), (1512, True), (1514, False), (1516, True), (1517, True), (1519, False), (1521, False), (1522, False), (1523, False), (1524, False), (1527, True), (1528, True), (1531, False), (1532, False), (1535, True), (1538, True), (1541, True), (1546, True), (1547, True), (1548, True), (1549, True), (1550, True), (1551, True), (1552, False), (1553, False), (1554, False), (1556, False), (1557, True), (1558, True), (1559, False), (1560, True), (1562, True), (1567, False), (1568, False), (1569, False), (1572, True), (1573, False), (1574, True), (1576, False), (1577, True), (1580, False), (1581, True), (1582, False), (1583, False), (1584, False), (1586, False), (1589, True), (1590, True), (1591, True), (1592, False), (1594, True), (1598, True), (1599, True), (1602, False), (1605, False), (1607, True), (1608, True), (1609, True), (1610, False), (1611, False), (1613, True), (1614, False), (1616, False), (1617, False), (1618, True), (1619, True), (1620, True), (1621, True), (1623, True), (1625, False), (1626, True), (1627, False), (1630, True), (1631, False), (1632, True), (1636, False), (1639, True), (1640, True), (1643, False), (1644, True), (1647, False), (1648, False), (1650, True), (1654, True), (1655, True), (1657, True), (1660, True), (1662, True), (1663, False), (1664, True), (1666, True), (1667, False), (1669, False), (1670, False), (1672, False), (1673, True), (1675, True), (1676, False), (1677, False), (1678, True), (1679, True), (1681, True), (1682, False), (1685, False), (1686, False), (1687, True), (1688, True), (1689, False), (1690, True), (1691, True), (1692, False), (1693, True), (1697, True), (1698, True), (1699, False), (1700, True), (1702, True), (1703, False), (1705, True), (1706, True), (1709, True), (1711, True), (1712, False), (1713, False), (1715, True), (1716, False), (1717, False), (1720, False), (1722, False), (1723, False), (1726, True), (1729, True), (1730, False), (1731, True), (1732, True), (1733, False), (1735, False), (1736, True), (1737, True), (1739, True), (1740, True), (1741, False), (1744, True), (1745, False), (1747, False), (1748, False), (1749, False), (1750, False), (1751, True), (1752, False), (1754, False), (1756, False), (1757, True), (1758, True), (1759, False), (1761, True), (1762, False), (1765, False), (1769, False), (1776, False), (1777, False), (1779, True), (1780, False), (1782, False), (1783, False), (1784, True), (1786, False), (1787, False), (1789, False), (1791, True), (1792, False), (1794, True), (1795, False), (1797, True), (1798, True), (1799, False), (1801, False), (1802, False), (1806, False), (1807, False), (1809, False), (1811, False), (1812, False), (1814, True), (1815, False), (1816, False), (1818, True), (1821, False), (1822, False), (1823, False), (1824, False), (1825, True), (1829, True), (1830, False), (1832, False), (1833, True), (1836, True), (1837, False), (1838, False), (1840, True), (1841, False), (1842, True), (1843, True), (1845, False), (1849, True), (1850, True), (1852, True), (1859, False), (1860, True), (1861, True), (1862, True), (1868, False), (1870, True), (1871, False), (1873, True), (1875, True), (1877, False), (1878, True), (1879, True), (1880, True), (1881, False), (1884, False), (1886, False), (1887, False), (1888, False), (1891, True), (1895, False), (1896, True), (1897, False), (1898, False), (1899, True), (1902, False), (1904, True), (1906, True), (1907, True), (1909, False), (1910, True), (1912, True), (1913, True), (1914, True), (1917, False), (1918, True), (1919, False), (1920, True), (1921, True), (1922, False), (1923, True), (1924, False), (1927, False), (1928, False), (1930, False), (1931, True), (1933, False), (1934, False), (1936, False), (1939, False), (1941, False), (1942, False), (1943, True), (1944, True), (1946, False), (1947, True), (1949, False), (1950, True), (1952, False), (1953, True), (1954, False), (1955, True), (1957, True), (1958, True), (1959, False), (1960, False), (1961, True), (1962, True), (1963, True), (1966, True), (1967, False), (1971, False), (1972, True), (1973, True), (1974, True), (1975, True), (1978, False), (1979, True), (1981, True), (1982, False), (1983, False), (1984, True), (1985, True), (1986, True), (1989, False), (1990, True), (1991, False), (1993, False), (1995, True), (1998, True), (2002, False), (2003, False), (2005, False), (2006, False), (2008, False), (2009, True), (2015, True), (2019, True), (2022, True), (2024, True), (2026, True), (2027, True), (2031, True), (2032, False), (2033, True), (2035, True), (2037, True), (2038, True), (2040, True), (2041, False), (2044, True), (2045, True), (2047, True)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 8
    dataset = Dataset(input_bits_count, [(0, True), (1, True), (2, False), (3, False), (5, False), (6, True), (9, False), (10, False), (11, False), (14, False), (15, True), (16, True), (17, False), (18, False), (20, False), (22, False), (23, True), (24, False), (25, True), (26, True), (27, False), (28, True), (29, True), (30, True), (31, False), (32, True), (33, True), (34, False), (35, True), (36, False), (37, True), (38, False), (40, False), (41, False), (42, False), (43, False), (44, False), (45, False), (47, False), (48, False), (49, False), (50, False), (51, False), (52, True), (53, True), (54, True), (55, True), (56, False), (57, False), (58, True), (59, True), (60, False), (62, True), (63, False), (64, True), (65, False), (66, True), (67, True), (68, False), (69, False), (71, False), (73, True), (75, False), (76, True), (77, False), (79, False), (80, False), (81, False), (82, True), (83, True), (84, True), (85, True), (86, False), (89, True), (91, False), (92, True), (93, False), (94, True), (95, False), (96, True), (99, True), (100, True), (101, False), (102, True), (103, False), (105, True), (106, False), (107, True), (109, True), (112, True), (113, False), (114, True), (115, True), (116, False), (117, True), (118, True), (119, False), (120, True), (121, False), (122, True), (123, False), (126, False), (127, False), (129, True), (133, False), (135, False), (136, False), (137, False), (138, True), (139, False), (140, True), (141, True), (142, False), (143, True), (145, False), (146, True), (147, True), (148, False), (149, False), (151, False), (152, True), (153, False), (154, True), (155, True), (156, False), (157, False), (158, False), (159, False), (160, True), (161, True), (164, True), (166, False), (167, True), (168, True), (170, False), (171, False), (173, False), (174, True), (175, True), (177, True), (178, False), (179, True), (180, False), (181, True), (182, True), (183, False), (184, False), (185, True), (188, False), (190, False), (192, False), (193, False), (195, False), (196, False), (199, False), (200, False), (201, True), (202, False), (203, True), (204, False), (206, False), (207, True), (208, True), (209, False), (210, False), (211, False), (212, True), (213, True), (214, True), (215, False), (217, False), (218, False), (219, True), (220, True), (221, True), (222, True), (223, False), (224, False), (225, False), (226, False), (227, False), (229, True), (230, True), (231, True), (232, True), (233, True), (234, False), (235, False), (237, True), (238, True), (239, True), (240, True), (241, False), (242, True), (243, True), (244, True), (245, False), (246, True), (247, False), (248, True), (249, False), (251, False), (252, True), (254, True), (255, True)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)

    input_bits_count = 8
    dataset = Dataset(input_bits_count, [(0, True), (4, False), (5, True), (7, False), (8, False), (9, False), (10, False), (12, False), (14, True), (15, True), (18, False), (19, False), (21, False), (22, True), (26, True), (27, False), (28, False), (29, False), (32, False), (33, True), (35, False), (37, False), (39, True), (40, True), (41, True), (42, False), (43, False), (46, True), (47, True), (50, False), (51, False), (52, False), (53, False), (55, True), (56, False), (57, True), (58, False), (59, True), (60, False), (61, False), (63, False), (66, False), (67, True), (69, False), (72, True), (73, True), (74, False), (75, False), (79, False), (80, False), (81, True), (82, True), (83, False), (84, False), (87, False), (89, True), (90, True), (91, False), (92, False), (93, True), (95, False), (96, True), (100, False), (101, False), (109, True), (111, False), (113, True), (116, False), (118, False), (123, False), (126, True), (129, False), (130, False), (132, True), (133, True), (134, False), (135, False), (136, False), (137, True), (139, False), (141, True), (142, False), (143, True), (147, False), (151, False), (155, False), (156, True), (157, False), (159, False), (160, False), (162, True), (163, False), (164, False), (165, False), (166, True), (167, False), (168, True), (169, False), (170, False), (171, True), (172, False), (173, False), (174, False), (175, False), (179, False), (180, False), (181, True), (186, False), (187, True), (188, True), (192, True), (193, False), (194, True), (196, True), (198, False), (200, False), (201, False), (202, False), (203, False), (205, True), (206, False), (207, False), (209, False), (211, False), (212, True), (214, False), (216, True), (217, False), (218, False), (219, False), (220, False), (221, False), (223, False), (228, True), (229, False), (230, True), (231, True), (232, False), (233, False), (234, False), (235, False), (236, False), (237, False), (239, False), (242, False), (243, True), (244, False), (245, True), (246, False), (248, False), (249, True), (251, False), (252, False)])
    a_DatasetField = DatasetField._new__and_valid(dataset)
    a_DatasetField = DatasetField._new__and_valid(dataset, leaf_keep_dataset=True)
    
    pass



if "random dataset test   slow" and True:
    iter_multiplier = 100
    #11111111111111111111w 加上保存xor的flag，然后跑一下。
    '''
    old code
    # empty
    print("empty, line:"+str(_line_()))
    for ____total_iter in range(13):
        if ____total_iter%500 == 0:
            print(____total_iter, end=", ")
            pass
        for input_bits_count in range(1, 62):
            dataset = Dataset.new_empty(input_bits_count)
            a_DatasetField = DatasetField._new__and_valid(dataset)
            
            assert a_DatasetField.all_irr
            assert a_DatasetField.get_is_leaf()
            assert a_DatasetField.children is None
            pass#for input_bits_count in range(1, 12):
        pass#for ____total_iter
    print()
    '''
    
    
    for _while_count in range(1, 1111111111):
        if  (_while_count % 1000) == 0:
            print(_while_count, end=", ")
            pass
        
        # full
        print("full,     line:"+str(_line_()) +"      "+ str(datetime.now().time()))
        for ____total_iter in range(2*iter_multiplier):
            for input_bits_count in range(1, 12):
                temp_rand = random.random()*0.6
                p_False = 0.2+temp_rand
                p_True = 0.8-temp_rand+0.001
                
                dataset = Dataset.rand__sorted(input_bits_count, p_False = p_False, p_True = p_True)
                # if dataset.data.__len__() == 0:
                #     continue
                a_DatasetField = DatasetField._new__and_valid(dataset)
                #a_DatasetField.valid(dataset, total_amount = 100)
                a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
                pass#for input_bits_count
            pass#for ____total_iter

        # dense, non full
        print("dense,    line:"+str(_line_()) +"      "+ str(datetime.now().time()))
        for ____total_iter in range(10*iter_multiplier):
            for input_bits_count in range(1, 12):
                temp_rand = random.random()*0.6
                temp_rand2 = random.random()*0.6+0.2
                p_False = (0.2+temp_rand)*temp_rand2
                p_True = (0.8-temp_rand)*temp_rand2
                
                dataset = Dataset.rand__sorted(input_bits_count, p_False = p_False, p_True = p_True)
                if dataset.data.__len__() == 0:
                    continue
                a_DatasetField = DatasetField._new__and_valid(dataset)
                #a_DatasetField.valid(dataset, total_amount = 100)
                a_DatasetField = DatasetField._new(dataset, leaf_keep_dataset=True)
                a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
                pass#for input_bits_count
            pass#for ____total_iter
        
        # sparse
        print("sparse,   line:"+str(_line_()) +"      "+ str(datetime.now().time()))
        for ____total_iter in range(30*iter_multiplier):
            for input_bits_count in range(1, 12):
                temp_rand = random.random()*0.6
                temp_rand2 = random.random()*0.2+0.05
                p_False = (0.2+temp_rand)*temp_rand2
                p_True = (0.8-temp_rand)*temp_rand2
                
                dataset = Dataset.rand__sorted(input_bits_count, p_False = p_False, p_True = p_True)
                if dataset.data.__len__() == 0:
                    continue
                a_DatasetField = DatasetField._new__and_valid(dataset)
                #a_DatasetField.valid(dataset, total_amount = 100)
                a_DatasetField = DatasetField._new(dataset, leaf_keep_dataset=True)
                a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
                pass#for input_bits_count
            pass#for ____total_iter
        
        # one side
        print("one side, line:"+str(_line_()) +"      "+ str(datetime.now().time()))
        for ____total_iter in range(5*iter_multiplier):
            for input_bits_count in range(1, 12):
                if random.random() <0.5:
                    p_False = random.random()*0.6
                    p_True = 0.
                    pass
                else:
                    p_False = 0.
                    p_True = random.random()*0.6
                    pass
                
                dataset = Dataset.rand__sorted(input_bits_count, p_False = p_False, p_True = p_True)
                if dataset.data.__len__() == 0:
                    continue
                a_DatasetField = DatasetField._new__and_valid(dataset)
                #a_DatasetField.valid(dataset, total_amount = 100)
                a_DatasetField = DatasetField._new(dataset, leaf_keep_dataset=True)
                a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
                assert a_DatasetField.children is None
                pass#for input_bits_count
            pass#for ____total_iter
        pass#while
    

'''
From here on, it's only my curiosity.
The tool passed all the test above. If it's provided with perfect dataset, it fits.
But, how much can it extrapolation?

The test below is, if it's only provided with a proportion of a perfect dataset, can it fit?
How much accurate can it get.
Let's say, a 3 bits full adder, it's a 7bits input, 4 bits output. In total, it can have 2^7(128)
different cases. If the tool only sees part of them, let's say 20%, then how much acc can it get.
And what's the relationship between the feed% and acc%. 

This is probably the first binary learning/logic learning test in half-real-case in the world.
Today is 2025 oct 12.
I'm YagaoDirac, a natural Earth human(without a BCI).
I'm in China, Earth.
'''

    

    
    
class Dataset_Set:
    max_input_bits:int
    dataset_children:list[Dataset]
    is_data_sorted:bool
    def __init__(self, max_input_bits:int, output_bits:int):
        assert max_input_bits>0, "why do you need a 0 bit input case?"
        assert output_bits>0
        self.max_input_bits = max_input_bits
        self.dataset_children = []
        for _ in range(output_bits):
            self.dataset_children.append(Dataset._debug__new_empty(max_input_bits))#meant to use this function.
            pass
        self.is_data_sorted = True
        pass
    
    # def get_max_input_bits(self, check = False)->int:
    #     if check:
    #         self._check()
    #         pass
    #     return self.da
        
    def _check(self):
        if self.get_output_count()>1:
            for i in range(1, self.get_output_count()):
                self.dataset_children[0].data.__len__() == self.dataset_children[i].data.__len__()
                for ii in range(self.dataset_children[0].data.__len__()):
                    _element_0_addr = self.dataset_children[0].data[ii][0]
                    _element_i_addr = self.dataset_children[i].data[ii][0]
                    assert _element_0_addr == _element_i_addr
                    pass#ii
                pass#i
            pass
        pass#end of function
    
    def get_as_int___check_btw(self)->list[tuple[int,int]]:
        '''return list[(addr, bools as int)]'''
        #check
        self._check()
        assert int(True) == 1
        assert int(False) == 0
        
        result_list:list[tuple[int,int]] = []
        for i in range(self.dataset_children[0].data.__len__()):
            addr = self.dataset_children[0].data[i][0]
            result_for_this_i = 0
            for ii_left_to_right in range(self.get_output_count()):
                content = int(self.dataset_children[ii_left_to_right].data[i][1])
                ii_right_to_left = (self.get_output_count() -1) - ii_left_to_right
                shifted_content = content<<(ii_right_to_left)
                result_for_this_i += shifted_content
                pass
            result_list.append((addr, result_for_this_i))
            pass#i
        return result_list
    
    def get_readable___check_btw(self, addr_also_as_binary = True, pad_with_zero = True)->str:
        #@TODO: comma between digits. flex padding.
        _temp_result_list = self.get_as_int___check_btw()
        _temp_str_list:list[str] = []
        
        # if addr_also_as_binary:
        #     if -1 == recommended_FieldLength:
        #         recommended_FieldLength = self.get_recommended_addr_FieldLength()
        #         pass
        #     pass
            
        for _addr_result in _temp_result_list:
            if addr_also_as_binary:
                addr_str__unpadded = f"{_addr_result[0]:b}"
                if pad_with_zero:
                    _needs_zeros = self.max_input_bits-addr_str__unpadded.__len__()
                    addr_str = ("0"*_needs_zeros)+addr_str__unpadded
                else:
                    addr_str = addr_str__unpadded
                    pass
                pass
            else:
                addr_str = f"{_addr_result[0]}"
                pass
            
            content = f"{_addr_result[1]:b}"
            _content_len = content.__len__()
            needs_zeros = self.get_output_count()-_content_len
            _temp_str_list.append(f"{addr_str}:{"0"*needs_zeros}{_addr_result[1]:b}")
            pass
        result_str = ", ".join(_temp_str_list)
        return result_str
        
        #check
        self._check()
        
        result_str = ""
        for i in range(self.dataset_children[0]._len()):
            addr = self.dataset_children[0].data[i][0]
            result_str+= f"{addr}:"
            for ii in range(0, self._len()):
                content = self.dataset_children[ii].data[i][1]
                result_str+= f"{content:b}"
                pass
            result_str+= f", "
            pass#i
        return result_str
    
    def get_mininum_input_count___check_btw(self)->int:
        #check
        self._check()
        
        _temp_last_addr = self.dataset_children[0].data[-1][0]
        bit_count = 0
        while _temp_last_addr>0:
            bit_count = bit_count +1
            #tail
            _temp_last_addr = _temp_last_addr >>1
            pass
        return bit_count
    def get_max_input_bits(self)->int:
        return self.max_input_bits
    def get_output_count(self)->int:
        return self.dataset_children.__len__()
    
    def sort(self):
        for dataset in self.dataset_children:
            dataset.sort()
            pass
        self.is_data_sorted = True
        pass
    
    def add_binary(self, addr:int, the_bits:int):
        assert count_ones(addr)<=self.max_input_bits
        
        the_str = f"{the_bits:b}"
        assert the_str.__len__()<=self.get_output_count()
        bool_list = []
        for _ in range(self.get_output_count()):
            bool_list.append(False)
            pass
        offset = self.get_output_count()-the_str.__len__()
        assert offset >= 0
        for i in range(the_str.__len__()-1,-1,-1):
            char = the_str[i]
            if "1" == char:
                bool_list[i+offset] = True
                pass
            pass
    
        for i in range(self.get_output_count()):
            self.dataset_children[i].data.append((addr, bool_list[i]))
            self.dataset_children[i].is_sorted = False
            pass
        
        self.is_data_sorted = False
        pass#end of function
    def add_binaries(self, data_list:list[tuple[int,int]]):
        for data in data_list:
            self.add_binary(data[0], data[1])
            pass
        pass#end of function.
    
    def get_recommended_addr_FieldLength(self)->int:
        self._check()
        assert self.is_data_sorted
        _greatest_addr = self.dataset_children[0].data[-1][0]
        result = 0
        while _greatest_addr>0:
            result = result +1
            #tail
            _greatest_addr = _greatest_addr >>1
            pass
        return result
    
    
    @staticmethod
    def explain_as_full_adder(input:int, input_bit_amount:int, safety_check = True)->tuple[int,int,int,int]:
        '''return (in1, in2, in_carry, out)'''
        if safety_check:
            assert input_bit_amount>0
            _input = input
            _input_binary_length = 0
            while _input>0:
                _input_binary_length = _input_binary_length +1
                #tail
                _input = _input >>1
                pass
            assert _input_binary_length<=(input_bit_amount*2+1)
            pass#safety
        
        mask_1 = ((1<<input_bit_amount)-1)<<(input_bit_amount+1)
        mask_2 = ((1<<input_bit_amount)-1)<<1
        mask_c = 1
        
        number_1 = (input&mask_1)>>input_bit_amount+1
        number_2 = (input&mask_2)>>1
        number_c = input&mask_c
        sum = number_1+number_2+number_c
        return (number_1, number_2, number_c, sum)
    @staticmethod
    def get_full_adder_testset_partly(input_bit_amount:int, proportion = 0.2, max_amount = 1000000):#->list[list[tuple[int,bool]]]:
        '''return (datasetset, FieldLength)'''
        total_bit_amount = input_bit_amount*2+1
        one_shift_by__total_bit_amount = 1<<total_bit_amount
        assert proportion<0.8, "Don't torture the also. You don't need a 0.8+ proportion. Or get some other also."
        assert proportion>0.
        amount_needed = int(proportion * one_shift_by__total_bit_amount)
        if 0 == amount_needed:
            amount_needed = 1
            pass
        assert amount_needed > 0
        if amount_needed>max_amount:
            amount_needed = max_amount
            pass
        
        addr_set = set()
        while True:
            a_rand_num = random.randint(0, one_shift_by__total_bit_amount-1)
            addr_set.add(a_rand_num)
            if addr_set.__len__() == amount_needed:
                break
            pass#while
        
        addr_list = list(addr_set)
        addr_list.sort()
        
        # mask_1 = ((1<<input_bit_amount)-1)<<(input_bit_amount+1)
        # mask_2 = ((1<<input_bit_amount)-1)<<1
        # mask_c = 1
        
        FieldLength = input_bit_amount*2+1
        datasetset = Dataset_Set(FieldLength, input_bit_amount+1)
        for addr in addr_list:
            # number_1 = (addr&mask_1)>>input_bit_amount+1
            # number_2 = (addr&mask_2)>>1
            # number_c = addr&mask_c
            # sum = number_1+number_2+number_c
            _result_tuple = Dataset_Set.explain_as_full_adder(addr, input_bit_amount, False)
            datasetset.add_binary(addr, _result_tuple[3])
            pass
        
        datasetset.sort()#maybe this is repeating?
        return (datasetset,FieldLength)

    #1111111111111111111111more please.


    #end of class
    
if "test" and True:
    datasetset = Dataset_Set(8, 3)
    datasetset.add_binary(addr=111, the_bits = 0b111)
    datasetset.add_binary(addr=100, the_bits = 0b100)
    datasetset.add_binary(addr=101, the_bits = 0b101)
    datasetset.add_binary(addr=1  , the_bits = 0b001)
    datasetset.sort()
    assert datasetset.get_mininum_input_count___check_btw() == 7
    assert datasetset.max_input_bits == 8
    assert datasetset.get_output_count() == 3
    
    _int_tuple = datasetset.get_as_int___check_btw()
    assert _int_tuple[0][0] == 1
    assert _int_tuple[0][1] == 1
    assert _int_tuple[1][0] == 100
    assert _int_tuple[1][1] == 0b100
    assert _int_tuple[2][0] == 101
    assert _int_tuple[2][1] == 0b101
    assert _int_tuple[3][0] == 111
    assert _int_tuple[3][1] == 0b111
    readable_str = datasetset.get_readable___check_btw(addr_also_as_binary = False)
    assert readable_str == "1:001, 100:100, 101:101, 111:111"
    #assert readable_str == "0000001:001, 1100100:100, 1100101:101, 1101111:111"
    readable_str = datasetset.get_readable___check_btw()
    assert readable_str == "00000001:001, 01100100:100, 01100101:101, 01101111:111"
    # print(datasetset.data[0].data)
    # print(datasetset.data[1].data)
    # print(datasetset.data[2].data)
    assert datasetset.get_recommended_addr_FieldLength() == 7
    pass

if "test" and True:
    _result_tuple_int_int_int_int = Dataset_Set.explain_as_full_adder(0b00001,2)
    assert _result_tuple_int_int_int_int == (0,0,1,1)
    _result_tuple_int_int_int_int = Dataset_Set.explain_as_full_adder(0b00101,2)
    assert _result_tuple_int_int_int_int == (0,2,1,3)
    _result_tuple_int_int_int_int = Dataset_Set.explain_as_full_adder(0b11001,2)
    assert _result_tuple_int_int_int_int == (3,0,1,4)
    _result_tuple_int_int_int_int = Dataset_Set.explain_as_full_adder(0b01010,2)
    assert _result_tuple_int_int_int_int == (1,1,0,2)
    _result_tuple_int_int_int_int = Dataset_Set.explain_as_full_adder(0b11111,2)
    assert _result_tuple_int_int_int_int == (3,3,1,7)
    
    a_Dataset_Set, FieldLength = Dataset_Set.get_full_adder_testset_partly(2,0.2)
    #print(a_Dataset_Set.get_readable___check_btw(True))
    assert a_Dataset_Set.get_output_count() == 3
    assert FieldLength == 5
    pass



class DatasetField_Set:
    leaf_keep_dataset:bool
    fields:list[DatasetField]
    def __init__(self, datasetset:Dataset_Set, leaf_keep_dataset:bool = False):
        assert datasetset.is_data_sorted
        
        self.leaf_keep_dataset = leaf_keep_dataset
        self.fields = []
        for dataset in datasetset.dataset_children:
            _temp_datasetfield = DatasetField._new(dataset, leaf_keep_dataset)
            
            #maybe redundent since subfield doesn't init when dataset is empty
            assert not _temp_datasetfield.all_irr
            
            self.fields.append(_temp_datasetfield)
            pass
        pass
    def _check___maybe_redundent(self):
        for field in self.fields:
            assert not field.all_irr
            pass
        pass
    
    def get_input_count(self)->int:
        return self.fields[0].input_bits_count
    def get_output_count(self)->int:
        return self.fields.__len__()
    
    # def get_addr_FieldLength(self)->int:
    #     return self.fields[0].FieldLength
    
    def lookup(self, addr:int)->int:
        '''return result_in_int'''
        _count_ones = count_ones(addr)
        assert self.get_input_count()>=_count_ones
        
        irr_bit_maskin_int = 0
        result_in_int = 0
        for i_from_the_left in range(self.get_output_count()):
            field = self.fields[i_from_the_left]
            
            _temp_tuple_bbbio = a_DatasetField.lookup(addr)
            result_or_suggest = _temp_tuple_bbbio[2]
            #mypy doesn't like this line. for_sure_it_is_irr, for_sure_it_is_NOT_irr, result_or_suggest,found_in_addr,_ = field.lookup(addr)
            
            i_from_the_right = (self.get_output_count()-1)-i_from_the_left#len-1-index_from_left
            #1111111111111111111w
            #irr_bit_maskin_int = irr_bit_maskin_int|(temp_result_is_irr<<i_from_the_right)
            result_in_int = result_in_int|(result_or_suggest<<i_from_the_right)
            pass
        return result_in_int
    
    if "old and probably wrong. The dimention is probably wrong. lookup function." and False:
        def lookup(self, datasetset:Dataset_Set)->tuple[int,int]:
            '''return (irr_bit_maskin_int, result_in_int)'''
            assert self.get_output_count() == datasetset.get_output_count()
            addr_list_with_something_else = datasetset.get_as_int___check_btw()
            irr_bit_maskin_int = 0
            result_in_int = 0
            for i_from_the_left in range(self.get_output_count()):
                field = self.fields[i_from_the_left]
                addr = addr_list_with_something_else[i_from_the_left][0]        
                field.lookup(addr)
                i_from_the_right = (self.get_output_count()-1)-i_from_the_left
                irr_bit_maskin_int = irr_bit_maskin_int|(1<<i_from_the_right)
                result_in_int = result_in_int|(1<<i_from_the_right)
                pass
            return (irr_bit_maskin_int, result_in_int)
        pass
    
    def valid(self, datasetset:Dataset_Set, total_amount: int = -1)->tuple[int,int,list[tuple[int,int]]]:
        '''return flags_of_sub_check_finishes, flags_of_sub_check_finishes___reversed, list_of_total_and_error
        
        >>> flags_of_sub_check_finishes: if all the check finishes, this is all 1s.
        >>> flags_of_sub_check_finishes___reversed: but this one is more convenient. If all finishes, this is all 0s, 
        and you can check this against int(0).
        >>> list_of_total_and_error: a list of tuples. In the tuple, it's total check count and error count.'''
        
        #safety first
        assert self.get_output_count() == datasetset.get_output_count()
        
        list_of_total_and_error:list[tuple[int,int]] = []
        flags_of_sub_check_finishes = 0
        flags_of_sub_check_finishes___reversed = (1<<datasetset.get_output_count()) -1
        for i_from_left in range(self.get_output_count()):
            dataset = datasetset.dataset_children[i_from_left]
            datasetfield = self.fields[i_from_left]
            
            _temp_tuple = datasetfield.valid(dataset, total_amount, log_the_error_to_file_and_return_immediately = False)
            finished_the_check = _temp_tuple[0]
            check_count = _temp_tuple[1]
            error_count = _temp_tuple[2]
            
            if finished_the_check:
                i_from_right = self.get_output_count()-1-i_from_left
                flags_of_sub_check_finishes ^= 1<<i_from_right
                flags_of_sub_check_finishes___reversed ^= 1<<i_from_right
                pass
            
            list_of_total_and_error.append((check_count, error_count))
            pass
        return flags_of_sub_check_finishes, flags_of_sub_check_finishes___reversed, list_of_total_and_error
    
    def valid_irr(self, datasetset:Dataset_Set, total_amount_irr: int = -1)->tuple[int,int,list[tuple[int,int]]]:
        '''return flags_of_sub_check_finishes, flags_of_sub_check_finishes___reversed, list_of_total_and_error
        
        >>> flags_of_sub_check_finishes: if all the check finishes, this is all 1s.
        >>> flags_of_sub_check_finishes___reversed: but this one is more convenient. If all finishes, this is all 0s, 
        and you can check this against int(0).
        >>> list_of_total_and_error: a list of tuples. In the tuple, it's total check count and error count.'''
        
        #safety first
        assert self.get_output_count() == datasetset.get_output_count()
        
        list_of_total_and_error:list[tuple[int,int]] = []
        flags_of_sub_check_finishes = 0
        flags_of_sub_check_finishes___reversed = (1<<datasetset.get_output_count()) -1
        for i_from_left in range(self.get_output_count()):
            dataset = datasetset.dataset_children[i_from_left]
            datasetfield = self.fields[i_from_left]
            
            _temp_tuple = datasetfield.valid_irr(dataset, total_amount_irr, log_the_error_to_file_and_return_immediately = False)
            finished_the_check = _temp_tuple[0]
            check_count = _temp_tuple[1]
            error_count = _temp_tuple[2]
            
            if finished_the_check:
                i_from_right = self.get_output_count()-1-i_from_left
                flags_of_sub_check_finishes ^= 1<<i_from_right
                flags_of_sub_check_finishes___reversed ^= 1<<i_from_right
                pass
            
            list_of_total_and_error.append((check_count, error_count))
            pass
        return flags_of_sub_check_finishes, flags_of_sub_check_finishes___reversed, list_of_total_and_error
    
    pass#end of class
    
if "test" and True:
    
    1w 继续。
    
    a_Dataset_Set = Dataset_Set(6, 3)
    a_Dataset_Set.add_binary(11, 0b111)
    a_Dataset_Set.add_binary(15, 0b110)
    a_Dataset_Set.add_binary(21, 0b100)
    a_Dataset_Set.add_binary(25, 0b000)
    a_Dataset_Set.sort()
    assert a_Dataset_Set.get_recommended_addr_FieldLength() == 5
    assert a_Dataset_Set.max_input_bits == 6
    assert a_Dataset_Set.get_output_count() == 3
    
    FieldLength = a_Dataset_Set.get_recommended_addr_FieldLength()
    a_DatasetField_Set = DatasetField_Set(a_Dataset_Set)
    
    assert a_DatasetField_Set.get_input_count() == 6
    assert a_Dataset_Set.get_output_count() == a_DatasetField_Set.get_output_count()
    assert a_Dataset_Set.get_output_count() == 3

    _, flags___reversed, list_of_total_and_error = a_DatasetField_Set.valid(a_Dataset_Set)
    assert 0 == flags___reversed
    assert [(4, 0), (4, 0), (4, 0), ] == list_of_total_and_error
    
    
    a_Dataset_Set = Dataset_Set(6, 3)
    a_Dataset_Set.add_binary(11, 0b111)
    a_Dataset_Set.add_binary(15, 0b110)
    a_Dataset_Set.sort()
    assert a_Dataset_Set.get_recommended_addr_FieldLength() == 4
    assert a_Dataset_Set.max_input_bits == 6
    assert a_Dataset_Set.get_output_count() == 3
    
    
    
    _, flags___reversed, list_of_total_and_error = a_DatasetField_Set.valid(a_Dataset_Set)
    assert 0 == flags___reversed
    assert [(2, 0), (2, 0), (2, 0), ] == list_of_total_and_error
    
    
    a_Dataset_Set = Dataset_Set(6, 3)
    a_Dataset_Set.add_binary(11, 0b111)
    a_Dataset_Set.add_binary(15, 0b110)
    a_Dataset_Set.add_binary(21, 0b100)
    a_Dataset_Set.add_binary(25, 0b000)
    a_Dataset_Set.sort()
    assert a_Dataset_Set.get_recommended_addr_FieldLength() == 5
    assert a_Dataset_Set.max_input_bits == 6
    assert a_Dataset_Set.get_output_count() == 3
    
    FieldLength = a_Dataset_Set.get_recommended_addr_FieldLength()
    a_DatasetField_Set = DatasetField_Set(a_Dataset_Set, leaf_keep_dataset = True)
    assert a_DatasetField_Set.get_output_count() == 3
    _, flags___reversed, list_of_total_and_error = a_DatasetField_Set.valid_irr(a_Dataset_Set, 10)
    assert 0 == flags___reversed
    assert [(10, 0), (10, 0), (10, 0), ] == list_of_total_and_error, "this is not stable. Usually, simply retry and the test passes."
    pass
    
if "lookup" and True:
    a_Dataset_Set = Dataset_Set(6, 3)
    a_Dataset_Set.add_binary(11, 0b111)
    a_Dataset_Set.add_binary(15, 0b110)
    a_Dataset_Set.add_binary(21, 0b100)
    a_Dataset_Set.add_binary(25, 0b000)
    a_Dataset_Set.sort()
    assert a_Dataset_Set.max_input_bits == 6
    assert a_Dataset_Set.get_recommended_addr_FieldLength() == 5
    assert a_Dataset_Set.get_output_count() == 3
    
    fdsfds = a_Dataset_Set.get_readable___check_btw()
    a_DatasetField_Set = DatasetField_Set(a_Dataset_Set, leaf_keep_dataset = True)
    assert a_DatasetField_Set.get_input_count() == 6
    assert a_DatasetField_Set.get_output_count() == 3
    
    irr_bit_maskin_int, result_in_int = a_DatasetField_Set.lookup(11)#1111111111111111111w
    assert irr_bit_maskin_int == 0
    assert result_in_int == 0b111
    
    irr_bit_maskin_int, result_in_int = a_DatasetField_Set.lookup(25)#1111111111111111111w
    assert irr_bit_maskin_int == 0
    assert result_in_int == 0b000
    pass
    
if "a real extrapolation test. You know, it's exciting." and True:
    if "a special case" and True:
        #011:10, 111:11
        bits_count = 1
        training_Dataset_Set = Dataset_Set(bits_count*2+1, bits_count+1)
        addr = 0b011
        training_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
        addr = 0b111
        training_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
        training_Dataset_Set.sort()
        _input_str = training_Dataset_Set.get_readable___check_btw()
        a_DatasetField_Set = DatasetField_Set(training_Dataset_Set)
        
        _tree_high = a_DatasetField_Set.fields[0].readable_as_tree()
        '''1+ir means, this bits only output 1.'''
        
        _tree_low = a_DatasetField_Set.fields[1].readable_as_tree()
        ''' 0__:0+ir, 1__:1+ir means this bit is the same as the first bit in input.'''
        
        #000, 011, 100, 111
        test_Dataset_Set = Dataset_Set(bits_count*2+1, bits_count+1)
        addr = 0b000
        test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
        addr = 0b011
        test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
        addr = 0b100
        test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
        addr = 0b111
        test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
        test_Dataset_Set.sort()
        _test_str = test_Dataset_Set.get_readable___check_btw()
        '''the higher bit of output is 1 in 2 cases and 0 in 2 cases. While the lower bit is always the same as the first bit in input.'''
        
        fdsfdsfds = a_DatasetField_Set.valid(test_Dataset_Set)
        #111111111111111111111w
        a_DatasetField_Set.lookup
        pass
    
    if "small scale validation" and True:
        #I read this several times. It's prpbably correct. 
        bits_count = 1
        trainingset_proportion = 0.3
        testset_proportion = 0.5
        #preparing the DatasetField object.
        trainingset_Set, FieldLength = Dataset_Set.get_full_adder_testset_partly(bits_count, trainingset_proportion)
        a_DatasetField_Set = DatasetField_Set(trainingset_Set)
        training_correctness_list = a_DatasetField_Set.valid(trainingset_Set)
        for training_correctness in training_correctness_list:
            assert training_correctness[0] == trainingset_Set.dataset_children[0].data.__len__()
            assert training_correctness[1] == 0
            pass
        
        #now the real excitement.
        testset_Set, _ = Dataset_Set.get_full_adder_testset_partly(bits_count, testset_proportion)
        valid_result_list__total_and_error = a_DatasetField_Set.valid(testset_Set)


        print(f"training set:{trainingset_Set.get_readable___check_btw(True)}")
        print(f"bit0:{a_DatasetField_Set.fields[0].readable_as_tree()}")
        print(f"bit1:{a_DatasetField_Set.fields[1].readable_as_tree()}")
        print(f"test set:{testset_Set.get_readable___check_btw(True)}")
        
        #print the result
        print(f"Amount of total possible cases:{1<<(bits_count*2+1)}, training with {trainingset_proportion}({trainingset_Set.dataset_children[0].data.__len__()}), ", end="")
        print(f"(test with {testset_proportion}({testset_Set.dataset_children[0].data.__len__()})):")
        print("from most significant bit to least. Accuracy is shown below.")
        for i in range(valid_result_list__total_and_error.__len__()):
            total_and_error = valid_result_list__total_and_error[i]
            total = total_and_error[0]
            error = total_and_error[1]
            print(f"bit{i}:{1.-(error/total)}, ")
            pass    
    
    pass    
    
    
    
    
    


def _____unchecked___get_adder_testset_full(input_bit_amount, amount_needed=-1)->list[list[tuple[int,bool]]]:
    total_bit_amount = input_bit_amount*2+1
    total_result_bit_amount = input_bit_amount+1

    big_dataset_from_most_to_least:list[list[tuple[int,bool]]] = []
    for _ in range(total_result_bit_amount):
        big_dataset_from_most_to_least.append([])
        pass
    mask_1 = (1<<input_bit_amount)<<(input_bit_amount+1)
    mask_2 = (1<<input_bit_amount)<<1
    mask_c = 1
    for addr in range(1<<total_bit_amount):
        input_1 = (addr&mask_1)>>(input_bit_amount+1)
        input_2 = (addr&mask_2)>>1
        input_c = addr&mask_c
        add_result = input_1 + input_2 + input_c
        add_result_in_str = f"{add_result:b}"
        for i in range(total_result_bit_amount):
            the_char = add_result_in_str[i]
            result = ("1" == the_char)
            big_dataset_from_most_to_least[i].append((addr,result))
            pass
        pass
    if amount_needed > (1<<total_bit_amount):
        return big_dataset_from_most_to_least
    if amount_needed <= 0:
        return big_dataset_from_most_to_least
    
    part_big_dataset_from_most_to_least:list[list[tuple[int,bool]]] = []
    for _ in range(total_result_bit_amount):
        part_big_dataset_from_most_to_least.append([])
        pass
    for _ in range(amount_needed):
        a_rand_num = random.randint(0, big_dataset_from_most_to_least[0].__len__()-1)
        for i in range(total_result_bit_amount):
            popped = big_dataset_from_most_to_least[i].pop(a_rand_num)
            part_big_dataset_from_most_to_least[i].append(popped)
            pass
        pass
    return part_big_dataset_from_most_to_least

    
    

    

assert False , '''还没解决的，1，xor field要标记一下纯无关的位，不然会是一个2的n次方。而且完事了还需要merge，完全没有必要。
all irr的对面最多只有3种情况，记得assert一下。
分别是，xor，xnor，branch。前两个是同一种情况，可以假装是一个大的field，但是要翻转符号。
最后一种（branch），要做一个假的地址，然后用假的地址进去查，而且最后要把这个假的地址返回出来，甚至要考虑要返回所有的假地址的每一次的变化过程，方便调试。


位的优先级。
非xor有关位>非xor无关位>xor位。
其中，非xor有关位可以形成branch。
非xor无关位 和 xor位 共同构成xor leaf。

在xor检测的时候，非xor有关位的xor分数是大于0，而且小于最大可能分数，非xor无关位的分数为0，xor位的分数是最大可能分数。
于是当有非xor有关位的时候，按其中的最无关位split（未证明最优）。



文档里面加一个，如果一个查询跑到了一个all irr的叶节点，应该怎么办。答，从更根的地方分叉出去。
如果是xor了会怎么办。理论依据是什么，答，位和答案的相关性。
'''

assert False , "要不要做一个专门的fake xor检测？？？"




