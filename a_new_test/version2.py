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
3, detect the dataset, and get has_1, has_0, and (max_I_forgot_name_of_this_var,
split_at_this_bit).
4, a special case is ***all xor***. Condition is (no_irrelevant and 
0 == max_I_forgot_name_of_this_var). If true, it's a leaf node. [return]
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
better_irr. 
The lookup starts from the root node. When a node cannot decide the result, it 
calls 1 of its 2 subfield(children) to do the job. This should only happen 
when the node is non-leaf.

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
    index_of_children_split:int
    children:Optional[tuple['DatasetField','DatasetField']]#yeah, Im a tree.
    ready_for_lookup:bool
    #irrelevant_lookup:list[]??? to do 
    
    #some flags
    has_1:bool
    has_0:bool
    has_irr:bool
    is_dataset_sorted:bool
    already_leaf:bool
    all_xor:bool
    #simple flags
    not_after_xor:bool
    already_const:bool
    all_irr:bool
    def __init__(self, bitmask:int, addr:int, FieldLength:int, bits_already_in_use:int, 
                dataset:list[tuple[int,bool]], 
                suggest_about_1_0_irr__already_leaf:bool=False, 
                suggest_has_1:bool=False, suggest_has_0:bool=False, suggest_has_irr:bool=False, 
                suggest_dataset_is_sorted:bool=False, 
                suggest_already_leaf:bool=False, 
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
        if dataset.__len__() == 0:
            self.all_irr = True
            pass
        
        
        #flags
        has_1 = OneTimeSetup_Bool()
        has_0 = OneTimeSetup_Bool()
        has_irr = OneTimeSetup_Bool()
        is_dataset_sorted = OneTimeSetup_Bool()
        already_leaf = OneTimeSetup_Bool()
        all_xor = OneTimeSetup_Bool()
        
        if suggest_about_1_0_irr__already_leaf:
            has_1.set(suggest_has_1)
            has_0.set(suggest_has_0)
            has_irr.set(suggest_has_irr)
            already_leaf.set(suggest_already_leaf)
            pass
        if suggest_dataset_is_sorted:
            is_dataset_sorted.set()
            pass
        
        if dataset.__len__() == 0:
            #only irrelevant.
            has_1.set(False)
            has_0.set(False)
            has_irr.set()
            is_dataset_sorted.set() #no need to sort it anymore.
            already_leaf.set() #no need to split it.
            all_xor.set(False)
        
            pass
        else:
            #At least 1 item in the dataset. (item is true or false. Irrelevant is not stored there.)
            
            if FieldLength == bits_already_in_use:
                #full addr. The dataset has 1 or 0 item, and 0 item case is already checked uppon.
                #so here is full addr+only 1 item. It's a minimun item and it's relevant.
                all_xor.set(False)
                is_dataset_sorted.set() #no need to sort it anymore.
                already_leaf.set() #no need to split it.
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
                if not suggest_about_1_0_irr__already_leaf:
                    ignore_true = False
                    ignore_false = False
                    found_true = suggest_has_1
                    found_false = suggest_has_0
                    for item in dataset:
                        if (not ignore_true) and item[1]:
                            ignore_true = True
                            found_true = True
                            pass
                        if (not ignore_false) and not item[1]:
                            ignore_false = True
                            found_false = True
                            pass
                        if ignore_true and ignore_false:
                            break
                        pass
                    has_1.set(found_true)
                    has_0.set(found_false)
                    pass
                
                # step 2, irrelevant item exists?
                num_of_irrelevant = self._get_num_of_irrelevant()
                if num_of_irrelevant>0:
                    has_irr.set()
                    pass
                else:
                    has_irr.set(False)
                    pass
                
                # step 3, splittable. condition is, has both 1 and 0.
                if (not has_1.get_value_safe()) or (not has_0.get_value_safe()):
                    already_leaf.set() #no need to split it.
                    pass
                else:
                    # case here. Has both 1 and 0. Either not already_leaf, or all xor.
                    # step 4, figure out the best bit to split. If it doesn't split any more, this is still needed.
                    best_index_to_split, best_abs_of_num_of_same = self._detect_best_bit_to_split()
                    self.index_of_children_split = best_index_to_split
                    
                    # step 5, all_xor. Condition is, no irrelevant, all sum_of_same equal 0(best abs is 0).
                    temp_no_irr = not has_irr.get_value_safe()
                    temp_best_abs_of_num_of_same__are_all_0 = 0 == best_abs_of_num_of_same
                    all_xor.set(temp_no_irr&temp_best_abs_of_num_of_same__are_all_0)
                    #if it's all_xor, it's leaf. Otherwise not.
                    already_leaf.set(not all_xor.value)
                    pass
                pass#else of else# NON_full addr. The dataset has at least 1 item.
            pass#if FieldLength == bits_already_in_use:
            
            #for all cases. from the safe one-time-bool to a normal bool.
            self.has_1 = has_1.get_value_safe()
            self.has_0 = has_0.get_value_safe()
            self.has_irr = has_irr.get_value_safe()
            self.is_dataset_sorted = is_dataset_sorted.get_value_safe()
            self.already_leaf = already_leaf.get_value_safe()
            self.all_xor = all_xor.get_value_safe()
            #lastly, 
            #If this obj is the end of splitting, either case happens.
            #1, it's a all-xor-field.
            #2, it doesn't have 1 or 0.
            not_after_xor = OneTimeSetup_Bool()
            all_1 = OneTimeSetup_Bool()
            all_0 = OneTimeSetup_Bool()
            all_1_and_irr = OneTimeSetup_Bool()
            all_0_and_irr = OneTimeSetup_Bool()
            ready_for_lookup = OneTimeSetup_Bool()
            if self.all_xor:
                not_after_xor.set(self._get__if_not_after_xor__it_s_already_all_xor())
                all_1.set(False)
                all_0.set(False)
                already_leaf_node.set(False)
                ready_for_lookup.set()
                pass
            self.not_after_xor = (not not_after_xor.uninit)and not_after_xor.value
                
            if not self.has_1:
                if not self.has_0:
                    
                    all_1.set(False)
                    all_0.set(False)
                    already_leaf_node.set(False)
                    ready_for_lookup.set()
                    
                    already_const.set(False)
                    ready_for_lookup.set()
                    pass
                else:
                    #has 0 but no 1
                    already_const.set()
                    ready_for_lookup.set()
                    pass
                pass
            else:
                #has 1
                if not self.has_0:
                    #has 1 but no 0
                    already_const.set()
                    pass
                pass
            
            
            self.ready_for_lookup = False
            if continue_splitting and self.splittable:
                self.split(best_index_to_split)
                self.ready_for_lookup = True
                pass
        pass#end of function
                

                
    def _get__if_not_after_xor__it_s_already_all_xor(self)->bool:
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
    
    def _get_num_of_irrelevant(self)->int:
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
    def __unfinished_______lookup(self, addr:int)->tuple[bool,bool]:
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

        return (False, False)  # Target found
    # def has_only_irr(self)->bool:
    #     result = (TriState.false == self.has_1)and(TriState.false == self.has_0)
    #     if result:
    #         self.has_irr = TriState.true
    #         pass
    #     return result
    
        
    def split(self, bit_index:int):#->tuple['DatasetField','DatasetField']:
        #assert False, "untested"
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
        assert False,"顺便把有什么东西一起找了。"
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
        false_part = DatasetField( , , , self.bits_already_in_use+1, True)
        
        
        DatasetField(bitmask = new_bitmask, addr = false_addr, FieldLength = self.FieldLength,
                bits_already_in_use = self.bits_already_in_use+1, dataset = false_dataset,
                suggest_about_1: bool = False,
    suggest_has_1: bool = False,
    suggest_about_0: bool = False,
    suggest_has_0: bool = False,
    suggest_about_irr: bool = False,
    suggest_has_irr: bool = False,
    suggest_dataset_is_sorted: bool = False,
    suggest_about_splittable: bool = False,
    suggest_is_splittable: bool = False,
    check_addr: bool = False,
    continue_splitting: bool = True
        
        
        
        
        
        
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




a = 123