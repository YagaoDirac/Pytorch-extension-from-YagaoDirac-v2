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

#1q明天跟着重新整理的条件重新搞一下。

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
4, a special case is ***all xor***. Condition is (dataset_if_full and 
0 == best_abs_of_num_of_same). If true, it's a leaf node. [return]
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
'''
class DatasetField:
    #core info
    bitmask:int
    addr:int
    FieldLength:int
    bits_already_in_use:int
    dataset:list[tuple[int,bool]]
    #lookup tables.
    index_to_split:int
    children:Optional[tuple['DatasetField','DatasetField']]#yeah, Im a tree.
    ready_for_lookup:bool
    #irrelevant_lookup:list[]??? to do 
    
    #some flags
    has_1:bool
    has_0:bool
    has_irr:bool
    is_dataset_sorted:bool
    is_leaf_node:bool
    all_xor:bool
    #simple flags
    not_after_xor:bool
    already_const:bool
    all_irr:bool
    def __init__(self, bitmask:int, addr:int, FieldLength:int, bits_already_in_use:int, 
                dataset:list[tuple[int,bool]], 
                with_suggest:bool=False, 
                suggest_has_1:bool=False, suggest_has_0:bool=False, #suggest_has_irr:bool=False, 
                #suggest_dataset_is_full = False,
                #suggest_dataset_is_sorted:bool=False, 
                #suggest_already_leaf:bool=False, 
                _debug__check_all_safety:bool = True,# __debug__continue_splitting = True,
                #check_addr = False, #__debug_check_all_safety = False, 
                ):
        
        #core info. simple copy paste.
        self.bitmask = bitmask
        self.addr = addr
        self.FieldLength = FieldLength
        self.bits_already_in_use = bits_already_in_use
        self.dataset = dataset
        #lookup tables
        self.children = None
        
        #safety check
        if _debug__check_all_safety:
            self._init_only__check_all_addr()
            pass
        
        # 1, is dataset empty. If true, it's a leaf node. [return]
        if dataset.__len__() == 0:
            self.ready_for_lookup = True
            self.has_1 = False
            self.has_0 = False
            self.has_irr = True
            self.all_irr = True
            self.is_dataset_sorted = True#only to init. not a real result.
            self.is_leaf_node = True
            self.all_xor = False
            self.not_after_xor = False#only to init. not a real result.
            self.already_const = True#only to init. not a real result.
            return
        
        # 2, is dataset full. If true, it has no irrelevant items.
        num_of_irr = self._init_only__get_num_of_irrelevant()
        dataset_if_full = 0 == num_of_irr
        
        # 3, detect the dataset, and get has_1, has_0, and 
        # (best_index_to_split, best_abs_of_num_of_same).
        if with_suggest:
            self.has_1 = suggest_has_1
            self.has_0 = suggest_has_0
            self.is_dataset_sorted = True#this maybe a bit dangerous?
            pass
        else:
            self._init_only__sort_dataset()
            found_true = False
            found_false = False
            for item in dataset:
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
        self.best_index_to_split, best_abs_of_num_of_same = self._detect_best_bit_to_split()
        
        # 4, a special case is ***all xor***. Condition is (dataset_if_full and 
        # 0 == max_I_forgot_name_of_this_var). If true, it's a leaf node. [return]
        if dataset_if_full and (0 == best_abs_of_num_of_same):
            self.not_after_xor = self._init_only__get__if_not_after_xor__it_s_already_all_xor()
            self.ready_for_lookup = True
            #self.has_1 see uppon
            #self.has_0 see uppon
            self.has_irr = not dataset_if_full
            self.all_irr = False
            self.is_dataset_sorted = True
            self.is_leaf_node = True
            self.all_xor = True
            #self.not_after_xor see uppon
            self.already_const = False#only to init. not a real result.
            return
        
        # 5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
        if (not self.has_1) or (not self.has_0):
            #at least 1 true. If both false, it's a all_irr case, which returned long ago.
            self.ready_for_lookup = True
            #self.has_1 see uppon
            #self.has_0 see uppon
            self.has_irr = not dataset_if_full
            self.all_irr = False
            self.is_dataset_sorted = True
            self.is_leaf_node = True
            self.all_xor = False
            self.not_after_xor = False#only to init. not a real result.
            self.already_const = not self.has_irr
            return 
        
        # 6, it's not a leaf node, split it[split][return]
        _check_all_safety = _debug__check_all_safety
        self.split(self.best_index_to_split, _debug__check_all_safety_in_split = _check_all_safety)
        self.ready_for_lookup = True
        #self.has_1 see uppon
        #self.has_0 see uppon
        self.has_irr = not dataset_if_full
        self.all_irr = False
        self.is_dataset_sorted = True
        self.is_leaf_node = False
        self.all_xor = False
        self.not_after_xor = False#only to init. not a real result.
        self.already_const = False#only to init. not a real result.
        return #end of function

                
    def _init_only__get__if_not_after_xor__it_s_already_all_xor(self)->bool:
        assert self.all_xor, "non-all-xor case can not call this function."
        #pick anything from self.dataset and detect if all input are Falses(simpler than true), is the output the xor result, or the reversed.
        #actually, the dataset should already be sorted, but the performance is probably similar and trivial?
        # So, yeah, I don't care, I detect the addr here, instead of check the "sorted" and do the trick.
        #maybe it's optimizable. But not for now.
        addr_of_item = self.dataset[0][0]
        result_of_item = self.dataset[0][1]
        num_of_ones_in_addr = 0
        for i in range(self.FieldLength-1,-1,-1):
            one_shift_by_i:int = 1<<i
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #This bit is in bitmask, ignore this i.
                continue
            bit_in_addr_for_this_i = addr_of_item & one_shift_by_i
            if bit_in_addr_for_this_i != 0:
                num_of_ones_in_addr = num_of_ones_in_addr + 1
                pass
            pass
        is_num_of_ones_in_addr_even = (num_of_ones_in_addr&1) == 0
        return result_of_item ^ is_num_of_ones_in_addr_even
    
    def _init_only__check_all_addr(self)->None:
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
    
    def _detect_best_bit_to_split(self)->tuple[int,int]:
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
    
    def _init_only__get_num_of_irrelevant(self)->int:
        '''Because only relevant items are stored in "dataset", 
            and the total possible number is 1<<N.'''
        length = self.dataset.__len__()
        total_possible = 1<<self.FieldLength
        result = total_possible-length
        return result
    def _init_only__sort_dataset(self):
        self.is_dataset_sorted = True
        self.dataset.sort(key=lambda item:item[0])
        pass
    def lookup(self, addr:int, )->tuple[None,None,bool,bool]:
        #to do :suggest, like allow_irr, better_non_irr, better_irr, true_when_irr, false_when_irr.
        
        #the sequence is the same as __init__.
        '''return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)'''
        
        # 1, is dataset empty. 
        if self.all_irr:
            '''return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)'''
            return(None, None, True, False)
        
        # 2 and 3 don't return.
        
        # 4, a special case is ***all xor***. 
        if self.all_xor:
            继续。。
        
            
        # 5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
        # 6, it's not a leaf node, split it[split][return]

        
        
        
        
        
        
        
        
        
        
        assert self.ready_for_lookup, "The __init__ stoped this object from detecting the inner structure."
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

        
        return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)
 
        
    def split(self, bit_index:int, _debug__check_all_safety_in_split: bool = False,)->None:#->tuple['DatasetField','DatasetField']:
        #assert False, "untested"
        #safety ckeck
        split_at_this_bit:int = 1<<bit_index
        if _debug__check_all_safety_in_split:
            assert self.bitmask&split_at_this_bit == 0, "bit_index already used in this dataset_field."
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
        for item in self.dataset:
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
        true_part = DatasetField(bitmask = new_bitmask, addr=true_addr, FieldLength=self.FieldLength,
            bits_already_in_use = self.bits_already_in_use+1, dataset=true_dataset,with_suggest = True,
            suggest_has_1 = true_has_1, suggest_has_0 = true_has_0, 
            _debug__check_all_safety = __check_all_safety,)

        false_part = DatasetField(bitmask = new_bitmask, addr=false_addr, FieldLength=self.FieldLength,
            bits_already_in_use = self.bits_already_in_use+1, dataset=false_dataset,with_suggest = True,
            suggest_has_1 = false_has_1, suggest_has_0 = false_has_0, 
            _debug__check_all_safety = __check_all_safety,)
        
        self.children = (true_part, false_part)
        pass

#要测试的。init，split，lookup。
继续。
if "init and split" and True:
    a_DatasetField = DatasetField(bitmask = 0b11000, addr = 0b01001, FieldLength=5, bits_already_in_use=2, \
                    dataset = [
        (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
        (0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
                    ],
                    with_suggest=False,_debug__check_all_safety = True)
    pass




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




a = 123