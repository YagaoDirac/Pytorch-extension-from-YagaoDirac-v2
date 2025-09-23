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
    def __str__(self):
        if self.uninit:
            return "UnInit"
        else:
            return str(self.value)
    #end of class
if "test" and False:
    print(OneTimeSetup_Bool())
    print("---test get_inited")
    a = OneTimeSetup_Bool()
    print(a.get_inited())
    a.set()
    print(a.get_inited())
    print("---test set and reset")
    a = OneTimeSetup_Bool()
    a.set()
    print(a)
    a = OneTimeSetup_Bool()
    a.reset()
    print(a)
    #only set once.
    a = OneTimeSetup_Bool()
    a.set()
    #a.set()
    a = OneTimeSetup_Bool()
    a.set()
    #a.reset()
    a = OneTimeSetup_Bool()
    a.reset()
    #a.set()
    a = OneTimeSetup_Bool()
    a.reset()
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
    has_1:OneTimeSetup_Bool
    has_0:OneTimeSetup_Bool
    has_irr:OneTimeSetup_Bool
    is_dataset_sorted:OneTimeSetup_Bool
    splittable:OneTimeSetup_Bool
    all_xor:bool
    def __init__(self, bitmask:int, addr:int, FieldLength:int, bits_already_in_use:int, 
                dataset:list[tuple[int,bool]], 
                suggest_about_1:bool=False, suggest_has_1:bool=False, 
                suggest_about_0:bool=False, suggest_has_0:bool=False, 
                suggest_about_irr:bool=False, suggest_has_irr:bool=False, 
                suggest_dataset_is_sorted:bool=False, 
                suggest_about_splittable:bool=False, suggest_is_splittable:bool=False, 
                ):
        #safety check
        is_addr_valid(bit_in_use_mask=bitmask, fixed_addr=addr)
        #core info
        self.bitmask = bitmask
        self.addr = addr
        self.FieldLength = FieldLength
        self.bits_already_in_use = bits_already_in_use
        self.dataset = dataset
        #lookup tables
        self.children = None
        
        #flags
        self.has_1 = OneTimeSetup_Bool()
        self.has_0 = OneTimeSetup_Bool()
        self.has_irr = OneTimeSetup_Bool()
        self.is_dataset_sorted = OneTimeSetup_Bool()
        self.splittable = OneTimeSetup_Bool()
        if suggest_about_1:
            self.has_1.set(suggest_has_1)
            pass
        if suggest_about_0:
            self.has_0.set(suggest_has_0)
            pass
        if suggest_about_irr:
            self.has_irr.set(suggest_has_irr)
            pass
        if suggest_dataset_is_sorted:
            self.is_dataset_sorted.set()
            pass
        if suggest_about_splittable:
            self.splittable.set(suggest_is_splittable)
            pass
        
        
        if dataset.__len__() == 0:
            #only irrelevant.
            self.has_1.set(False)
            self.has_0.set(False)
            self.has_irr.set()
            self.is_dataset_sorted.set() #no need to sort it anymore.
            self.splittable.set(False) #no need to split it.
            self.all_xor = False
            pass
        else:
            #At least 1 item in the dataset. (item is true or false. Irrelevant is not stored there.)
            
            if FieldLength == bits_already_in_use:
                #full addr. The dataset has 1 or 0 item, and 0 item case is already checked uppon.
                #so here is full addr+only 1 item. It's a minimun item and it's relevant.
                self.all_xor = False
                if dataset.__len__() == 1:
                    self.has_irr.set(False)
                    #the only item is relevant.
                    temp = dataset[0][1]
                    if temp:
                        self.has_1.set()
                        self.has_0.set(False)
                        pass
                    else:
                        self.has_1.set(False)
                        self.has_0.set()
                        pass
                    pass#if dataset.__len__() == 1:
                else:
                    assert False, "unreachable."
                pass
            else:
                # NON_full addr. The dataset has at least 1 item.
                # The dataset needs to be iterated to check what's in it.
                #but, sort it first.
                if self.is_dataset_sorted.uninit:
                    self.__sort_dataset()
                    pass
                looking_for_true = not suggest_about_1
                looking_for_false = not suggest_about_0
                found_true = suggest_has_1
                found_false = suggest_has_0
                for item in self.dataset:
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
                    self.has_1.set(found_true)
                    pass
                if not suggest_about_0:
                    self.has_0.set(found_false)
                    pass
                
                aaaaaaaa = self.detect_best_bit_to_split()
                #now, is this object splittable?
                #self.splittable
                #all xor case is self.splittable=true.
                
                    
            
        #总之顺便要把几个flag也更新了。
        #1,检查终点，3个条件。
        # 条件1，只有1位了。同时更新状态。这个在后面做化简的时候再写，现在先跳过。
        # 条件2，完全没有1或者0了，那么输出常数就够了。
        # 条件3，全部都是0.5的相关性，没有无关项，于是就是一个奇偶校验了（全xor）
       
                
        #if 1 == self.FieldLength-self.bits_already_in_use:
            #only 1 bit is unused in addr.
            
        assert False, "unfinished"
        
        pass
    def detect_best_bit_to_split(self):
        temp:list[float]
        # str_bitmask = "{:b}".format(self.bitmask)
        # str_bitmask_len = str_bitmask.__len__()
        # start = str_bitmask_len-self.FieldLength
        for i in range(self.FieldLength):
            # bit = 0
            # if i>=0:
            #     bit = int(str_bitmask[i], 2)
            #     pass
            # if bit=0
            one_shift_by_i:int = 1<<i
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #if this bit is not marked in bitmask. Otherwise ignore this i.
                continue
            total = 0
            same = 0
            bit_of_addr_for_this_i = self.addr&one_shift_by_i
            for item in self.dataset:
                bit_of_addr_of_item = item[0] & one_shift_by_i
                if bit_of_addr_for_this_i == bit_of_addr_of_item:
                    total = total+1
                    if 
                    same = 0
                    

        
    
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
        
    
    
if "test" and False:
            ds = {0:0, 1:0}
            assert Num_of_irrelevant(ds, 1) == 0
            ds = {0:0}
            assert Num_of_irrelevant(ds, 1) == 1
            ds = {}
            assert Num_of_irrelevant(ds, 1) == 2
            ds = {0:0}
            assert Num_of_irrelevant(ds, 2) == 3
            assert Num_of_irrelevant(ds, 3) == 7
            pass