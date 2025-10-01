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

# class TriState(Enum):
#     true = 1,
#     false = 0,
#     tbd = -1,
#     pass


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


#not in use. But let's leave it.
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








def rand_dataset_sorted(N:int, p_False:float, p_True:float, seed:int = 0)->list[tuple[int,bool]]:
    '''The result looks like addr:value. The addr is naturally sorted???
    
    For debug purpose.'''
    dataset:list[tuple[int,bool]] = []
    random.seed(seed)
    p_both = p_True + p_False
    for i in range(1<<N):
        r = random.random()
        if r < p_False:
            dataset.append((i, False))
        elif r < p_both:
            dataset.append((i, True))
        else:
            # irrelevant items are not in dataset
            pass
        pass
    return dataset
if "test" and False:
    ds2 = rand_dataset_sorted(4, 0.1, 0.1, 123)
    print(ds2)
    ds3 = rand_dataset_sorted(4, 0.1, 0.1, 123)
    print(ds3)
    ds4 = rand_dataset_sorted(4, 0.2, 0. , 123)
    print(ds4)
    pass
#to do optimization: sort the dataset 











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
    best_index_to_split:int
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
    #already_const_without_irr:bool
    all_irr:bool
    def print_directly(self):
        temp = f"<DatasetField> bitmask:{self.bitmask}, addr:{self.addr:b}, FieldLength{self.FieldLength}, "
        temp += f"bits_already_in_use{self.bits_already_in_use}, "
        temp += f"dataset__len__:{self.dataset.__len__()}, best_index_to_split:{self.best_index_to_split}, "
        if self.children is None:
            temp+="children is None, "
        else:
            temp+="has children, "
            pass
        print(temp)
        temp2 = f"has_irr:{self.has_irr}, all_irr:{self.all_irr}, has_0:{self.has_0}, has_1:{self.has_1}, all_xor:{self.all_xor}, "
        temp2 += f"already_const_without_irr:{self.get_already_const_without_irr()}, "
        temp2 += f"ready_for_lookup:{self.ready_for_lookup}, is_dataset_sorted:{self.is_dataset_sorted}, is_leaf_node:{self.is_leaf_node}, "
        temp2 += f"not_after_xor:{self.not_after_xor}, "
        print(temp2)
        pass
    def get_readable_addr(self)->str:
        '''Call this function after check the addr, otherwise the behavior of this function is undefined.
        
        This function at least takes advantage of an assuption, that, if the addr is correct, it only has 1s where the bitmask has 1s.
        Which means, if a bit in bitmask is 0, in addr, it must be 0, '''
        bitmask_in_str = f"{self.bitmask:b}"
        addr_in_str = f"{self.addr:b}"
        bitmask_is_this_longer_than_addr = bitmask_in_str.__len__()-addr_in_str.__len__()
        temp_str = ""
        for i in range(bitmask_in_str.__len__()):
            bit = bitmask_in_str[i]
            if "0" == bit:
                temp_str += "_"
                pass
            else:#"1" == bit:
                temp_str += addr_in_str[i-bitmask_is_this_longer_than_addr]
                pass
            pass
        how_many_underscores_needs_in_the_left = self.FieldLength-bitmask_in_str.__len__()
        result = "_"*how_many_underscores_needs_in_the_left+temp_str
        return result
        
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
        
        # only_irr 1, is dataset empty. If true, it's a leaf node. [return]
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
            #self.already_const_without_irr = True#only to init. not a real result.
            self.best_index_to_split = -1
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
            #self.is_dataset_sorted = True#keep it a comment, only for check.
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
        # 0 == max_I_forgot_name_of_this_var and has_0 and has_1). If true, it's a leaf node. [return]
        if (0 == best_abs_of_num_of_same) and dataset_if_full and self.has_0 and self.has_1:
            self.not_after_xor = self._init_only__get__whether_not_after_xor__it_s_already_all_xor()
            self.ready_for_lookup = True
            #self.has_1 see uppon
            #self.has_0 see uppon
            self.has_irr = not dataset_if_full
            self.all_irr = False
            #self.is_dataset_sorted = True already set before
            self.is_leaf_node = True
            self.all_xor = True
            #self.not_after_xor see uppon
            #self.already_const_without_irr = False#only to init. not a real result.
            self.best_index_to_split = -1
            return
        
        # 5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
        if (not self.has_1) or (not self.has_0):
            #at least 1 true. If both false, it's a all_irr case, which returned long ago.
            self.ready_for_lookup = True
            #self.has_1 see uppon
            #self.has_0 see uppon
            self.has_irr = not dataset_if_full
            self.all_irr = False
            #self.is_dataset_sorted = True already set before
            self.is_leaf_node = True
            self.all_xor = False
            self.not_after_xor = False#only to init. not a real result.
            #self.already_const_without_irr = not self.has_irr
            self.best_index_to_split = -1
            return 
        
        # 6, it's not a leaf node, split it[split][return]
        _check_all_safety = _debug__check_all_safety
        self.split(self.best_index_to_split, _debug__check_all_safety_in_split = _check_all_safety)
        self.ready_for_lookup = True
        #self.has_1 see uppon
        #self.has_0 see uppon
        self.has_irr = not dataset_if_full
        self.all_irr = False
        #self.is_dataset_sorted = True already set before
        self.is_leaf_node = False
        self.all_xor = False
        self.not_after_xor = False#only to init. not a real result.
        #self.already_const_without_irr = False#only to init. not a real result.
        return #end of function

    def get_already_const_without_irr(self)->bool:
        has_only_one_side = (self.has_0 ^ self.has_1)
        has_only_one_side_without_irr = has_only_one_side and (not self.has_irr)
        return has_only_one_side_without_irr

    def _init_only__get__whether_not_after_xor__it_s_already_all_xor(self)->bool:
        #assert self.all_xor, "non-all-xor case can not call this function."
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
    def _lookup_only__all_xor_only(self, addr:int)->bool:
        assert self.all_xor, "non-all-xor case can not call this function."
        #the docs here is copied form the _init_only__get__if_not_after_xor__it_s_already_all_xor and probably wrong.
        #pick anything from self.dataset and detect if all input are Falses(simpler than true), is the output the xor result, or the reversed.
        #actually, the dataset should already be sorted, but the performance is probably similar and trivial?
        # So, yeah, I don't care, I detect the addr here, instead of check the "sorted" and do the trick.
        #maybe it's optimizable. But not for now.
        #addr_of_item = self.dataset[0][0]
        #result_of_item = self.dataset[0][1]
        num_of_ones_in_addr = 0
        for i in range(self.FieldLength-1,-1,-1):
            one_shift_by_i:int = 1<<i
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #This bit is in bitmask, ignore this i.
                continue
            bit_in_addr_for_this_i = addr & one_shift_by_i
            if bit_in_addr_for_this_i != 0:
                num_of_ones_in_addr = num_of_ones_in_addr + 1
                pass
            pass
        is_num_of_ones_in_addr_even = (num_of_ones_in_addr&1) == 0
        result = is_num_of_ones_in_addr_even^self.not_after_xor
        return result
    
    
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
        best_index_to_split:int=-1
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
        total_possible = 1<<(self.FieldLength-self.bits_already_in_use)
        result = total_possible-length
        return result
    def _init_only__sort_dataset(self):
        self.is_dataset_sorted = True
        self.dataset.sort(key=lambda item:item[0])
        pass
    
    def _get_child(self, true_or_false:bool)->'DatasetField':
        # self.children = (true_part, false_part)
        if self.children is None:
            raise Exception("should be a tuple.")
        else:
            if true_or_false:
                return self.children[0]
            else:
                return self.children[1]  
        #end of function
        
    def lookup(self, addr:int, )->tuple[bool,bool]:#[None,None,bool,bool]:
        #to do :suggest, like allow_irr, better_non_irr, better_irr, true_when_irr, false_when_irr.
        
        #the sequence is the same as __init__.
        '''return (result_is_irr, result_is_true)'''
        #to do:return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)'''
        
        # 0, this is different. I guess a lot calling to this function is not the end, 
        # so let's detect non leaf node first.
        if not self.is_leaf_node:
            _temp__mask_of_this_bit = 1<<self.best_index_to_split
            this_bit_of_addr__with_shift = _temp__mask_of_this_bit&addr
            this_bit_of_addr = this_bit_of_addr__with_shift != 0
            the_child = self._get_child(this_bit_of_addr)
            return the_child.lookup(addr)
        
        # 1, is dataset empty. 
        if self.all_irr:
            '''return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)'''
            return(True, False)#irr.
        
        # 2 and 3 don't return.
        
        # 4, a special case is ***all xor***. 
        if self.all_xor:
            result = self._lookup_only__all_xor_only(addr)
            return(False, result)
            
        # 5, if the dataset doesn't have 1 or doesn't have 0, it's a leaf node. [return]
        # the dataset is assumed to be sorted.
        
        #a binary search pasted from bing.
        left = 0
        right = self.dataset.__len__() - 1
        #found = False
        while left <= right:
            mid:int = (left + right) // 2
            addr_here = self.dataset[mid][0]
            if addr_here == addr:
                return (False, self.dataset[mid][1])  # Target found
            elif addr_here < addr:
                left = mid + 1  # Search in the right half
                pass
            else:
                right = mid - 1  # Search in the left half
                pass
            pass
        
        # the raw way.
        # for item in self.dataset:
        #     if addr == item[0]:
        #         return(False, item[1])
        #     pass
        return(True, False)#irr.
        # Moved to top. The case 0. Case 6 in init, it's not a leaf node, split it[split][return]
        #end of function
        
        
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
if "init and split" and True:
    # it's not allowed to have no input. So test starts with 1 input.
    if "1 bit input, 0 free bit" and False:
        if "two wrong addr, they raise" and False:
            dataset = [(0b1,True), ]
            a_DatasetField = DatasetField(bitmask = 1, addr = 0, FieldLength=1, bits_already_in_use=1, \
                            dataset = dataset,
                            with_suggest=False,_debug__check_all_safety = True)
            dataset = [(0b0,True), ]
            a_DatasetField = DatasetField(bitmask = 1, addr = 1, FieldLength=1, bits_already_in_use=1, \
                            dataset = dataset,
                            with_suggest=False,_debug__check_all_safety = True)
            pass
        #irr 
        a_DatasetField = DatasetField(bitmask = 1, addr = 0, FieldLength=1, bits_already_in_use=1, \
                        dataset = [], with_suggest=False,_debug__check_all_safety = True)
        lookup_result = a_DatasetField.lookup(0)
        assert lookup_result[0]
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "0"
        a_DatasetField = DatasetField(bitmask = 1, addr = 1, FieldLength=1, bits_already_in_use=1, \
                        dataset = [], with_suggest=False,_debug__check_all_safety = True)
        lookup_result = a_DatasetField.lookup(0)
        assert lookup_result[0]
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "1"
        
        #relevant.
        dataset = [(0b0,True), ]
        a_DatasetField = DatasetField(bitmask = 1, addr = 0, FieldLength=1, bits_already_in_use=1, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        lookup_result = a_DatasetField.lookup(dataset[0][0])
        assert lookup_result[1] == dataset[0][1]
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "0"
        
        dataset = [(0b0,False), ]
        a_DatasetField = DatasetField(bitmask = 1, addr = 0, FieldLength=1, bits_already_in_use=1, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        lookup_result = a_DatasetField.lookup(dataset[0][0])
        assert lookup_result[1] == dataset[0][1]
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "0"
        
        dataset = [(0b1,True), ]
        a_DatasetField = DatasetField(bitmask = 1, addr = 1, FieldLength=1, bits_already_in_use=1, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        lookup_result = a_DatasetField.lookup(dataset[0][0])
        assert lookup_result[1] == dataset[0][1]
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "1"
        
        dataset = [(0b1,False), ]
        a_DatasetField = DatasetField(bitmask = 1, addr = 1, FieldLength=1, bits_already_in_use=1, \
                        dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        lookup_result = a_DatasetField.lookup(dataset[0][0])
        assert lookup_result[1] == dataset[0][1]
        readable_addr = a_DatasetField.get_readable_addr()
        assert readable_addr == "1"
        pass
    
    if "1 bit input, 1 free bit" and True:
        if "already checked" and False:
            dataset = [(0b0,True), (0b1,True), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert -1 == a_DatasetField.best_index_to_split
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.FieldLength
            assert a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert not a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.is_leaf_node
            assert a_DatasetField.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
            
            # a smallese non-leaf.
            # optimizable. it's equalavent to a 1bit xor, and possible to do some optimization. But let's keep it simple.
            dataset = [(0b0,True), (0b1,False), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert 0 == a_DatasetField.best_index_to_split
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 1 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert not a_DatasetField.is_leaf_node
            assert a_DatasetField.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 2 == best_abs_of_num_of_same
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            addr_1_child = a_DatasetField._get_child(true_or_false=True)
            assert not addr_1_child.all_irr
            assert 1 == addr_1_child.addr
            assert not addr_1_child.all_xor
            assert -1 == addr_1_child.best_index_to_split
            assert 1 == addr_1_child.bitmask
            assert 1 == addr_1_child.bits_already_in_use
            assert addr_1_child.children is None
            assert 1 == addr_1_child.FieldLength
            assert addr_1_child.get_already_const_without_irr()
            assert not addr_1_child.has_irr
            assert addr_1_child.has_0
            assert not addr_1_child.has_1
            assert addr_1_child.is_dataset_sorted
            assert addr_1_child.is_leaf_node
            assert addr_1_child.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = addr_1_child._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            readable_addr = addr_1_child.get_readable_addr()
            assert readable_addr == "1"
            
            addr_0_child = a_DatasetField._get_child(true_or_false=False)
            assert not addr_0_child.all_irr
            assert 0 == addr_0_child.addr
            assert not addr_0_child.all_xor
            assert -1 == addr_0_child.best_index_to_split
            assert 1 == addr_0_child.bitmask
            assert 1 == addr_0_child.bits_already_in_use
            assert addr_0_child.children is None
            assert 1 == addr_0_child.FieldLength
            assert addr_0_child.get_already_const_without_irr()
            assert not addr_0_child.has_irr
            assert not addr_0_child.has_0
            assert addr_0_child.has_1
            assert addr_0_child.is_dataset_sorted
            assert addr_0_child.is_leaf_node
            assert addr_0_child.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = addr_0_child._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            readable_addr = addr_0_child.get_readable_addr()
            assert readable_addr == "0"
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
            
            
            dataset = [(0b0,False), (0b1,False), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert -1 == a_DatasetField.best_index_to_split
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.FieldLength
            assert a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert not a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.is_leaf_node
            assert a_DatasetField.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
        
            
            # the other smallese non-leaf.
            # optimizable. it's equalavent to a 1bit xor, and possible to do some optimization. But let's keep it simple.
            dataset = [(0b0,False), (0b1,True), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert 0 == a_DatasetField.best_index_to_split
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 1 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert not a_DatasetField.is_leaf_node
            assert a_DatasetField.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 2 == best_abs_of_num_of_same
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            addr_1_child = a_DatasetField._get_child(true_or_false=True)
            assert not addr_1_child.all_irr
            assert 1 == addr_1_child.addr
            assert not addr_1_child.all_xor
            assert -1 == addr_1_child.best_index_to_split
            assert 1 == addr_1_child.bitmask
            assert 1 == addr_1_child.bits_already_in_use
            assert addr_1_child.children is None
            assert 1 == addr_1_child.FieldLength
            assert addr_1_child.get_already_const_without_irr()
            assert not addr_1_child.has_irr
            assert not addr_1_child.has_0
            assert addr_1_child.has_1
            assert addr_1_child.is_dataset_sorted
            assert addr_1_child.is_leaf_node
            assert addr_1_child.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = addr_1_child._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            readable_addr = addr_1_child.get_readable_addr()
            assert readable_addr == "1"
                
            addr_0_child = a_DatasetField._get_child(true_or_false=False)
            assert not addr_0_child.all_irr
            assert 0 == addr_0_child.addr
            assert not addr_0_child.all_xor
            assert -1 == addr_0_child.best_index_to_split
            assert 1 == addr_0_child.bitmask
            assert 1 == addr_0_child.bits_already_in_use
            assert addr_0_child.children is None
            assert 1 == addr_0_child.FieldLength
            assert addr_0_child.get_already_const_without_irr()
            assert not addr_0_child.has_irr
            assert addr_0_child.has_0
            assert not addr_0_child.has_1
            assert addr_0_child.is_dataset_sorted
            assert addr_0_child.is_leaf_node
            assert addr_0_child.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = addr_0_child._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            readable_addr = addr_0_child.get_readable_addr()
            assert readable_addr == "0"
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
            
        
            #partly irr
            dataset = [(0b1,False), ]#both 0b0 and 0b1 pass the test.
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert -1 == a_DatasetField.best_index_to_split
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert not a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.is_leaf_node
            assert a_DatasetField.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 1 == best_abs_of_num_of_same
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
            temp = a_DatasetField.lookup(0b0)
            assert temp[0]#irr
        
            dataset = [(0b0,True), ]#both 0b0 and 0b1 pass the test.
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert -1 == a_DatasetField.best_index_to_split
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert not a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.is_leaf_node
            assert a_DatasetField.ready_for_lookup
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 1 == best_abs_of_num_of_same
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "_"
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
            temp = a_DatasetField.lookup(0b1)
            assert temp[0]#irr
        
            #all irr
            dataset = []
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert -1 == a_DatasetField.best_index_to_split
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 1 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert not a_DatasetField.has_0
            assert not a_DatasetField.has_1
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.is_leaf_node
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
            
            
            pass
        
        
        继续，2bit xor，然后上随机数据集测试。
        做一点针对偏树的测试。
        
        fds=432
        
    
    
    
if "test with random dataset" and False:
    dataset = rand_dataset_sorted(1,0.3,0.3)
    a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                    dataset = dataset,
                    with_suggest=False,_debug__check_all_safety = True)
    a_DatasetField.print_directly()
    
    for item in dataset:
        temp = a_DatasetField.lookup(item[0])
        assert item[1] == temp[1]
        pass
    
    
    
    a_DatasetField = DatasetField(bitmask = 0b11000, addr = 0b01001, FieldLength=5, bits_already_in_use=2, \
                    dataset = [
        (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
        (0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
                    ],
                    with_suggest=False,_debug__check_all_safety = True)
    pass




# if "check_all_addr" and True:
#     a_DatasetField = DatasetField(bitmask = 0b11000, addr = 0b01001, FieldLength=5, bits_already_in_use=2, \
#                     dataset = [
#         (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
#         (0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
#                     ])
#     a_DatasetField._check_all_addr()
#     #继续。加上自动推荐，加上自动的检查和让更深的子域自动检查。
#     pass    

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


