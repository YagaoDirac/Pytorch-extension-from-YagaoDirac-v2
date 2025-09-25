'''
Author:YagaoDirac (also on X.com)

content
part1, infrastructures of logic algebra.
part2, logic learning.

a lot untested and unfinished. Simply search these 2 words to find them.

'''

from typing import Optional, Union
from enum import Enum
import random

class TriState(Enum):
    true = 1,
    false = 0,
    tbd = -1,
    pass



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
if "test" and True:
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


class DatasetField:
    #core info
    bitmask:int
    addr:int
    FieldLength:int
    bits_already_in_use:int
    dataset:list[tuple[int,bool]]
    #lookup tables.
    children:Optional[tuple['DatasetField','DatasetField']]#yeah, Im a tree.
    #irrelevant_lookup:list[]??? to do 
    
    #some flags
    has_1:bool
    has_0:bool
    has_irr:bool
    is_dataset_sorted:bool
    splittable:bool
    all_xor:bool
    def __init__(self, bitmask:int, addr:int, FieldLength:int, bits_already_in_use:int, 
                dataset:list[tuple[int,bool]], 
                suggest_about_1:bool=False, suggest_has_1:bool=False, 
                suggest_about_0:bool=False, suggest_has_0:bool=False, 
                suggest_about_irr:bool=False, suggest_has_irr:bool=False, 
                suggest_dataset_is_sorted:bool=False, 
                suggest_about_splittable:bool=False, suggest_is_splittable:bool=False, 
                check_addr = False, continue_splitting = True
                ):
        #safety check
        if check_addr:
            is_addr_valid(bit_in_use_mask=bitmask, fixed_addr=addr)
            pass
        #core info
        self.bitmask = bitmask
        self.addr = addr
        self.FieldLength = FieldLength
        self.bits_already_in_use = bits_already_in_use
        self.dataset = dataset
        #lookup tables
        self.children = None
        
        #flags
        has_1 = OneTimeSetup_Bool()
        has_0 = OneTimeSetup_Bool()
        has_irr = OneTimeSetup_Bool()
        is_dataset_sorted = OneTimeSetup_Bool()
        splittable = OneTimeSetup_Bool()
        all_xor = OneTimeSetup_Bool()
        
        if suggest_about_1:
            has_1.set(suggest_has_1)
            pass
        if suggest_about_0:
            has_0.set(suggest_has_0)
            pass
        if suggest_about_irr:
            has_irr.set(suggest_has_irr)
            pass
        if suggest_dataset_is_sorted:
            is_dataset_sorted.set()
            pass
        if suggest_about_splittable:
            splittable.set(suggest_is_splittable)
            pass
        
        if dataset.__len__() == 0:
            #only irrelevant.
            has_1.set(False)
            has_0.set(False)
            has_irr.set()
            is_dataset_sorted.set() #no need to sort it anymore.
            splittable.set(False) #no need to split it.
            all_xor.set(False)
            pass
        else:
            #At least 1 item in the dataset. (item is true or false. Irrelevant is not stored there.)
            
            if FieldLength == bits_already_in_use:
                #full addr. The dataset has 1 or 0 item, and 0 item case is already checked uppon.
                #so here is full addr+only 1 item. It's a minimun item and it's relevant.
                all_xor.set(False)
                is_dataset_sorted.set() #no need to sort it anymore.
                splittable.set(False) #no need to split it.
                if dataset.__len__() == 1:
                    has_irr.set(False)
                    #the only item is relevant.
                    temp = dataset[0][1]
                    if temp:
                        has_1.set()
                        has_0.set(False)
                        pass
                    else:
                        has_1.set(False)
                        has_0.set()
                        pass
                    pass#if dataset.__len__() == 1:
                else:
                    assert False, "unreachable."
                pass
            else:
                # NON_full addr. The dataset has at least 1 item.
                # The dataset needs to be iterated to check what's in it.
                #but, sort it first.
                
               
                # step 1, if the dataset contains 1 and 0.
                if suggest_dataset_is_sorted:
                    is_dataset_sorted.set()
                    pass
                if is_dataset_sorted.uninit:
                    self.__sort_dataset()
                    pass
                looking_for_true = not suggest_about_1
                looking_for_false = not suggest_about_0
                found_true = suggest_has_1
                found_false = suggest_has_0
                for item in dataset:
                    if looking_for_true and item[1]:
                        looking_for_true = False
                        found_true = True
                        pass
                    if looking_for_false and not item[1]:
                        looking_for_false = False
                        found_false = True
                        pass
                    if (not looking_for_true) and (not looking_for_false):
                        break
                    pass
                if not suggest_about_1:
                    has_1.set(found_true)
                    pass
                if not suggest_about_0:
                    has_0.set(found_false)
                    pass
                
                # step 2, irrelevant item exists?
                if self.get_num_of_irrelevant()>0:
                    has_irr.set()
                    pass
                else:
                    has_irr.set(False)
                    pass
                
                # step 3, splittable. condition is, has both 1 and 0.
                if has_1.get_value_safe() and has_0.get_value_safe():
                    splittable.set()
                    pass
                else:
                    splittable.set(False) #no need to split it.
                    pass
                    
                # step 4, figure out the best bit to split. If it doesn't split any more, this is still needed.
                best_index_to_split, best_abs_of_num_of_same = self.detect_best_bit_to_split()
                
                # step 5, all_xor. Condition is, no irrelevant, all sum_of_same equal 0(best abs is 0).
                temp_no_irr = not has_irr.get_value_safe()
                temp_best_abs_of_num_of_same__are_all_0 = 0 == best_abs_of_num_of_same
                all_xor.set(temp_no_irr&temp_best_abs_of_num_of_same__are_all_0)
                pass#else of else# NON_full addr. The dataset has at least 1 item.
            pass#if FieldLength == bits_already_in_use:
            
            #for all cases. from the safe one-time-bool to a normal bool.
            self.has_1 = has_1.get_value_safe()
            self.has_0 = has_0.get_value_safe()
            self.has_irr = has_irr.get_value_safe()
            self.is_dataset_sorted = is_dataset_sorted.get_value_safe()
            self.splittable = splittable.get_value_safe()
            self.all_xor = all_xor.get_value_safe()
            
            if continue_splitting and self.splittable:
                self.split(best_index_to_split)
                pass
        pass#end of function
                
             
    
    def _check_all_addr(self)->None:
        #part 1, bits_already_in_use must equal to 1s in bitmask
        temp_bitmask = self.bitmask
        ones_in_bitmask = 0
        while temp_bitmask!=0:
            if (temp_bitmask&1)!=0:
                ones_in_bitmask = ones_in_bitmask + 1
                pass
            temp_bitmask>>1
        assert self.bits_already_in_use == ones_in_bitmask
        #part 2, addr bits outside the bitmask must be 0s.
        reversed_bitmask = ~self.bitmask
        addr_bits_out_of_bitmask = self.addr&reversed_bitmask
        assert 0 == addr_bits_out_of_bitmask, "self.addr is bad. It has bit set to 1 outside the bitmask."
        #part 3, all addr bits under the bitmask must equal.
        masked_addr = self.addr&self.bitmask
        for item in self.dataset:
            masked_addr_of_item = item[0]&self.bitmask
            assert masked_addr == masked_addr_of_item
            pass
        pass
    
    def detect_best_bit_to_split(self)->tuple[int,int]:
        '''return (best_index_to_split, best_abs_of_num_of_same)'''
        actual_index:list[int] = []
        num_of_same:list[int] = []
        for i in range(self.FieldLength-1,-1,-1):
            one_shift_by_i:int = 1<<i
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #This bit is in bitmask, ignore this i.
                continue
            #total = 0
            dot_product_like = 0
            #Y is same as this bit. If a and result is true, +1. If a' and result is false, +1. Otherwise +0.
            #bit_of_addr_for_this_i = self.addr&one_shift_by_i#always 0????
            for item in self.dataset:
                #total = total+1
                bit_of_addr_of_item_raw = item[0] & one_shift_by_i
                bit_of_addr_of_item = bit_of_addr_of_item_raw!=0
                #if not (bit_of_addr_of_item ^ item[1]):
                logic_var_xor_this_result:bool = bit_of_addr_of_item ^ item[1]
                same_minus_this = 1-2*int(logic_var_xor_this_result)
                dot_product_like = dot_product_like + same_minus_this
                pass
            actual_index.append(i)
            num_of_same.append(dot_product_like)
            pass#for i in range
        # print(actual_index)
        # print(num_of_same)
        best_index_to_split:int=0
        best_abs_of_num_of_same:int = 0
        for i in range(actual_index.__len__()):
            index = actual_index[i]
            abs_of_same = abs(num_of_same[i])
            if abs_of_same>best_abs_of_num_of_same:
                best_abs_of_num_of_same = abs_of_same
                best_index_to_split = index
                pass
            pass
        return (best_index_to_split, best_abs_of_num_of_same)
        pass#end of function.
    
    def get_num_of_irrelevant(self)->int:
        '''Because only relevant items are stored in "dataset", 
            and the total possible number is 1<<N.'''
        length = self.dataset.__len__()
        total_possible = 1<<self.FieldLength
        result = total_possible-length
        return result
    def __sort_dataset(self):
        self.is_dataset_sorted.set()
        self.dataset.sort(key=lambda item:item[0])
        pass
    def lookup(self, addr:int)->tuple[bool,bool]:
        assert False, "可能写错了。"
        if not self.is_dataset_sorted:
            self.__sort_dataset()
            pass

        #a binary search pasted from bing.
        left, right = 0, self.dataset.__len__() - 1
        while left <= right:
            mid:int = (left + right) // 2

            addr_here = self.dataset[mid][0]
            if addr_here == addr:
                return (True, self.dataset[mid][1])  # Target found
            elif addr_here < addr:
                left = mid + 1  # Search in the right half
            else:
                right = mid - 1  # Search in the left half

        return (False, False)  # Target found
    # def has_only_irr(self)->bool:
    #     result = (TriState.false == self.has_1)and(TriState.false == self.has_0)
    #     if result:
    #         self.has_irr = TriState.true
    #         pass
    #     return result
    
        
    def split(self, bit_index:int):#->tuple['DatasetField','DatasetField']:
        assert False, "untested"
        #safety ckeck
        to_add_this_bit = 1<<bit_index
        assert self.bitmask&to_add_this_bit == 0, "bit_index already used in this dataset_field."
                
        #sort first.
        if not self.is_dataset_sorted:
            self.__sort_dataset()
            pass
        
        new_bitmask = self.bitmask|to_add_this_bit
        false_addr = self.addr
        true_addr = self.addr|to_add_this_bit
        
        #optimizable
        false_dataset:list[tuple[int,bool]] = []
        true_dataset:list[tuple[int,bool]] = []
        for item in self.dataset:
            if item[1]:
                true_dataset.append(item)
                pass
            else:# the false case
                false_dataset.append(item)
                pass
            pass
        
        #maybe optimizable.
        false_dataset.sort(key=lambda item:item[0])
        true_dataset.sort(key=lambda item:item[0])
        
        #这是一幅对联
        false_part = DatasetField(self.N, false_dataset, false_addr, new_bitmask, self.bits_already_in_use+1, True)
        true_part  = DatasetField(self.N, false_dataset,  true_addr, new_bitmask, self.bits_already_in_use+1, True)
        return (false_part, true_part)



if "check_all_addr" and True:
    a_DatasetField = DatasetField(bitmask = 0b11000, addr = 0b01001, FieldLength=5, bits_already_in_use=2, \
                    dataset = [
        (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
        (0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
                    ])
    a_DatasetField._check_all_addr()
    #继续。加上自动推荐，加上自动的检查和让更深的子域自动检查。
    pass    

# if "detect" and True:
# a = test()
# a.FieldLength = 5
# a.bitmask = 0b11000
# a.addr = 0b01000
# a.dataset = [
#     (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,False),
#     (0b01100,True),(0b01101,True),(0b01110,False),#(0b01111,False),
#                 ]
# a._check_all_addr()
# a.detect_best_bit_to_split()
# print()

