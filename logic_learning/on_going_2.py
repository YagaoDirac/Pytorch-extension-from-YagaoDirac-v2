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
from datetime import datetime


dataset:'Dataset'


def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    return caller_s_line_number#######

def log_the_error(FieldLength:int, dataset:"Dataset", filename = "wrong case log.txt", 
                        comment = ""):
    with open(filename, mode="+a", encoding="utf-8") as file:
        file.write(comment)
        FieldLength_str = f"FieldLength = {FieldLength},  "
        file.write(FieldLength_str)
        dataset_str:str = f"dataset = Dataset({dataset.data}, True)"
        #dataset_str:str = f"dataset = "+dataset.__str__()
        file.write(dataset_str)
        file.write("\n")
        pass
    pass
if "test" and False:
    log_the_error(5, [(2,True), (3,True), (5,False), (8,True), ], comment="Feature test. Not real log.")
    pass

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




class Dataset:
    data:list[tuple[int,bool]]
    is_sorted:bool
    def __init__(self, input:list[tuple[int,bool]] = [], is_sorted:bool = False):
        self.data = input
        if self.data.__len__()<2:
            self.is_sorted = True
            pass
        else:
            self.is_sorted = is_sorted
            pass
        pass
    
    @staticmethod
    def rand__sorted(FieldLength:int, p_False:float, p_True:float, seed:Optional[int]=None)->"Dataset":
        '''The result looks like addr:value. The addr is naturally sorted???
        
        For debug purpose.'''
        dataset:list[tuple[int,bool]] = []
        if seed:
            random.seed(seed)
            pass
        p_both = p_True + p_False
        for i in range(1<<FieldLength):
            r = random.random()
            if r < p_False:
                dataset.append((i, False))
            elif r < p_both:
                dataset.append((i, True))
            else:
                # irrelevant items are not in dataset
                pass
            pass
        return Dataset(dataset)
    
    @staticmethod
    def from_str(input:str, recommend_length_according_to_only_relevent_items = False)->tuple["Dataset",int]:
        '''return (dataset, min of field length)'''
        if input.__len__() == 0:
            return (Dataset(), 0)#the same behavior as "_"
        
        
        result_dataset:list[tuple[int,bool]] = []
        addr = 0
        for value in input:
            match value:
                case "1":
                    result_dataset.append((addr,True))
                    addr = addr +1
                    pass
                case "0":
                    result_dataset.append((addr,False))
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
        
        if result_dataset.__len__() == 0:
            return (Dataset(), 0)
        min_of_field_len = 0
        
        if recommend_length_according_to_only_relevent_items:
            _temp_last_addr = result_dataset[-1][0]
        else:#including irr at tail of the string.
            _temp_last_addr = input.__len__()-1
        while _temp_last_addr!=0:
            min_of_field_len = min_of_field_len +1
            #tail
            _temp_last_addr = _temp_last_addr >>1
            pass
        pass
        return (Dataset(result_dataset, True), min_of_field_len)

    def _len(self)->int:
        return self.data.__len__()
    
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
        right = self._len()-1
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
    
    def get_irr_addr___sorts_self(self, FieldLength:int)->tuple[bool, list[int]]:
        '''return (at least one item in dataset, the irr list). 
        
        If obj is empty, them all addresses are irr. In this case, no list is returned. 
        Manually handle it.'''
        if not self.is_sorted:
            self.sort()
            pass
        
        if self._len() == 0:
            return (False, [])
        index = 0
        irr_addr:list[int] = []
        addr = 0
        one_shift_FieldLength = 1<<FieldLength
        for addr in range(one_shift_FieldLength):
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
        for addr2 in range(addr, one_shift_FieldLength):
            irr_addr.append(addr2)
            pass
        return (True, irr_addr)

    def get_addr_as_set(self)->set:
        result:set[int] = set()
        for item in self.data:
            result.add(item[0])
            pass
        return result
    
    def add_const_bit_into_addr(self, FieldLength:int, index_from_right_side:int, value:bool)->int:
        '''return new FieldLength
        
        xyz, add a T in 1st(from right side), is xyTz'''
        
        '''xyz, 1st, T. 
        xyz into xy0 and z. The mask is 110(1<<3 - 1<<1, 3is the FieldLength) and 1(1<<1 -1).
        xy0 into xy00, shift by 1 bit.
        insert T0(T<<1, 1 is the index)
        
        wxyz, 2nd, T.
        wxyz into wx00 and yz. The mask is 1100(1<<4 - 1<<2, 4is the FieldLength) and 11(1<<2 -1)
        wx00 into wx000, shift by 1 bit.
        insert T00(T<<2, 2 is the index)
        '''
        assert index_from_right_side <= FieldLength
        insert_this = value<<index_from_right_side
        mask_for_high_bits = (1<<FieldLength)-(1<<index_from_right_side)
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
        return FieldLength
    
    def get_subset(self, bitmask:int, addr:int)->"Dataset":
        assert is_addr_valid(bitmask, addr)
        new_data:list[tuple[int,bool]] = []
        for item in self.data:
            masked_addr_of_item = item[0]&bitmask
            if masked_addr_of_item == addr:
                new_data.append(item)
                pass
            pass
        return Dataset(new_data, True)
    
if "find addr" and False:
    a_Dataset = Dataset([(1,True), (3,True), (9,True), ])
    assert a_Dataset.find_addr(0)[0] == False
    assert a_Dataset.find_addr(2)[0] == False
    assert a_Dataset.find_addr(5)[0] == False
    assert a_Dataset.find_addr(1)[0]
    assert a_Dataset.find_addr(3)[0]
    assert a_Dataset.find_addr(9)[0]
    _temp_set = a_Dataset.get_addr_as_set()
    assert _temp_set == {1,3,9}
    pass
    
if "test" and False:
    ds2 = Dataset.rand__sorted(4, 0.1, 0.1, 123)
    print(ds2)
    ds3 = Dataset.rand__sorted(4, 0.1, 0.1, 123)
    print(ds3)
    ds4 = Dataset.rand__sorted(4, 0.2, 0. , 123)
    print(ds4)
    pass

if "test slow" and False:
    #special cases.
    #empty
    a_dataset = Dataset()
    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength = 1)
    irr_amount = irr_addr_list_tuple[1].__len__()
    assert not irr_addr_list_tuple[0]
    if irr_addr_list_tuple[0]:
        assert 2 == irr_amount
    
    #full
    a_dataset = Dataset([(0,True), (1,True), ], True)
    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength = 1)
    irr_amount = irr_addr_list_tuple[1].__len__()
    assert irr_addr_list_tuple[0]
    assert 0 == irr_amount
    
    #the last one is irrelevant.
    a_dataset = Dataset([(0,True), ])
    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength = 1)
    irr_amount = irr_addr_list_tuple[1].__len__()
    assert irr_addr_list_tuple[0]
    assert 1 == irr_amount
    
    a_dataset = Dataset([(0,True), ])
    irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength = 2)
    irr_amount = irr_addr_list_tuple[1].__len__()
    assert irr_addr_list_tuple[0]
    assert 3 == irr_amount
    
    #random cases.
    #empty 
    for FieldLength in range(1,5):
        a_dataset = Dataset([])
        irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength)
        assert not irr_addr_list_tuple[0]
        pass
    
    #full
    for FieldLength in range(1,5):
        for _ in range(FieldLength*5):
            a_dataset = Dataset.rand__sorted(FieldLength,0.5,0.501)
            irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength)
            irr_amount = irr_addr_list_tuple[1].__len__()
            assert irr_addr_list_tuple[0]
            assert 0 == irr_amount
            pass
        pass
    
    for _test_iter in range(333):
        print(_test_iter, end=", ")
        #partly
        for FieldLength in range(1,15):
            for _ in range(FieldLength*25):
                a_dataset = Dataset.rand__sorted(FieldLength,0.2,0.2)
                irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength)
                irr_amount = irr_addr_list_tuple[1].__len__()
                if irr_addr_list_tuple[0]:
                    assert a_dataset._len() + irr_amount == (1<<FieldLength)
                    pass
                pass
            pass
    
        #one side.
        for FieldLength in range(1,15):
            for _ in range(FieldLength*25):
                a_dataset = Dataset.rand__sorted(FieldLength,0.2,0)
                irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength)
                irr_amount = irr_addr_list_tuple[1].__len__()
                if irr_addr_list_tuple[0]:
                    assert a_dataset._len() + irr_amount == (1<<FieldLength)
                    pass
                pass
            pass
        for FieldLength in range(1,15):
            for _ in range(FieldLength*25):
                a_dataset = Dataset.rand__sorted(FieldLength,0,0.2)
                irr_addr_list_tuple = a_dataset.get_irr_addr___sorts_self(FieldLength)
                irr_amount = irr_addr_list_tuple[1].__len__()
                if irr_addr_list_tuple[0]:
                    assert a_dataset._len() + irr_amount == (1<<FieldLength)
                    pass
                pass
            pass
    
    pass

if "test" and False:
    a_Dataset_tuple_11 = Dataset.from_str("1______",True)
    assert a_Dataset_tuple_11[0].data == [(0,True)]
    assert a_Dataset_tuple_11[1] == 0
    
    a_Dataset_tuple_11a = Dataset.from_str("1______",False)
    assert a_Dataset_tuple_11a[0].data == [(0,True)]
    assert a_Dataset_tuple_11a[1] == 3
    
    a_Dataset_tuple_0 = Dataset.from_str("")
    a_Dataset_tuple_1 = Dataset.from_str("10011")
    a_Dataset_tuple_1a = Dataset.from_str("10 0,,11")
    assert a_Dataset_tuple_1[0].data == a_Dataset_tuple_1a[0].data
    assert a_Dataset_tuple_1[1] == a_Dataset_tuple_1a[1]
    a_Dataset_tuple_2 = Dataset.from_str("1001")
    a_Dataset_tuple_3 = Dataset.from_str("100")
    a_Dataset_tuple_4 = Dataset.from_str("101")
    a_Dataset_tuple_5 = Dataset.from_str("00000")
    a_Dataset_tuple_6 = Dataset.from_str("_")
    assert a_Dataset_tuple_6[0].data == []
    assert a_Dataset_tuple_6[1] == 0
    a_Dataset_tuple_7 = Dataset.from_str("__")
    a_Dataset_tuple_8 = Dataset.from_str("1_")
    a_Dataset_tuple_8b = Dataset.from_str("_1")
    a_Dataset_tuple_9 = Dataset.from_str("1_01")
    a_Dataset_tuple_10 = Dataset.from_str("1_01_")
    a_Dataset_tuple_10a = Dataset.from_str("1_  ,01_")
    assert a_Dataset_tuple_10[0].data == a_Dataset_tuple_10a[0].data
    assert a_Dataset_tuple_10[1] == a_Dataset_tuple_10a[1]
    pass

if "test" and False:
    a_Dataset, FieldLength = Dataset.from_str("1001")
    FieldLength = a_Dataset.add_const_bit_into_addr(FieldLength,1,True)
    assert a_Dataset.data[0b00][0] == 0b010
    assert a_Dataset.data[0b01][0] == 0b011
    assert a_Dataset.data[0b10][0] == 0b110
    assert a_Dataset.data[0b11][0] == 0b111
    
    a_Dataset, FieldLength = Dataset.from_str("1001")
    FieldLength = a_Dataset.add_const_bit_into_addr(FieldLength,1,False)
    assert a_Dataset.data[0b00][0] == 0b000
    assert a_Dataset.data[0b01][0] == 0b001
    assert a_Dataset.data[0b10][0] == 0b100
    assert a_Dataset.data[0b11][0] == 0b101
    
    a_Dataset, FieldLength = Dataset.from_str("101")
    FieldLength = a_Dataset.add_const_bit_into_addr(FieldLength,1,True)
    assert a_Dataset.data[0b00][0] == 0b010
    assert a_Dataset.data[0b01][0] == 0b011
    assert a_Dataset.data[0b10][0] == 0b110

    a_Dataset, FieldLength = Dataset.from_str("10_1")
    FieldLength = a_Dataset.add_const_bit_into_addr(FieldLength,1,True)
    assert a_Dataset.data[0b00][0] == 0b010
    assert a_Dataset.data[0b01][0] == 0b011
    assert a_Dataset.data[0b10][0] == 0b111
    
    a_Dataset= Dataset([(0,True)])
    FieldLength = a_Dataset.add_const_bit_into_addr(5,0,True)
    assert a_Dataset.data[0][0] == 0b1
    a_Dataset= Dataset([(0,True)])
    FieldLength = a_Dataset.add_const_bit_into_addr(5,1,True)
    assert a_Dataset.data[0][0] == 0b10
    a_Dataset= Dataset([(0,True)])
    FieldLength = a_Dataset.add_const_bit_into_addr(5,3,True)
    assert a_Dataset.data[0][0] == 0b1000
    a_Dataset= Dataset([(0,True)])
    FieldLength = a_Dataset.add_const_bit_into_addr(5,5,True)
    #new_dataset = add_const_bit_into_addr(5,6,True, [(0,True)])
    
    a_Dataset= Dataset([(0b11111,True)])
    FieldLength = a_Dataset.add_const_bit_into_addr(5,0,False)
    assert a_Dataset.data[0][0] == 0b111110
    a_Dataset= Dataset([(0b11111,True)])
    FieldLength = a_Dataset.add_const_bit_into_addr(5,3,False)
    assert a_Dataset.data[0][0] == 0b110111
    pass

if "test" and False:
    a_Dataset = Dataset([(0,True),(1,True),(2,True),(3,True),])
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
    pass

class DatasetField:
    if "class var" and True:
        #core info
        bitmask:int
        addr:int
        FieldLength:int
        bits_already_in_use:int
        dataset:Optional[Dataset]
        #leaf_keep_dataset:bool 
        #branch_keep_dataset:bool 
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
        is_leaf_node:bool
        all_xor:bool
        #simple flags
        not_after_xor:bool# self.not_after_xor # if all 0s, and not_after_xor is false, result is false. Every one flips it once.
        #already_const_without_irr:bool
        all_irr:bool
        pass
    @staticmethod
    def _new(FieldLength, dataset, leaf_keep_dataset:bool = False)->'DatasetField':
        result = DatasetField(0,0,FieldLength,0,dataset, leaf_keep_dataset = leaf_keep_dataset,)
        return result
    
    def __init__(self, bitmask:int, addr:int, FieldLength:int, bits_already_in_use:int, 
                dataset:Dataset, 
                
                with_suggest:bool=False, 
                suggest_has_1:bool=False, suggest_has_0:bool=False, 
                
                _debug__check_all_safety:bool = True,
                leaf_keep_dataset:bool = False,
                branch_keep_dataset:bool = False,
                ):
        if __debug__:
            if branch_keep_dataset and (not leaf_keep_dataset):
                raise Exception("I didn't prepare for this combination. Maybe you want to modify the code. Remember to also modify the lookup function.")
            pass#if __debug__:
        
        #core info. simple copy paste.
        self.bitmask = bitmask
        self.addr = addr
        self.FieldLength = FieldLength
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
        if dataset._len() == 0:
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
            
            # #第一步，构造第一个addr
            # addr_for_first_result = self.bitmask&self.addr
            
            # #第二部，得到第一个的结果。
            # first_data_tuple = self.dataset[0]
            # assert addr_for_first_result == first_data_tuple[0]
            # temp_result:bool = first_data_tuple[1]
            
            # #3，依次找出翻转每一个的结果，
            # for i in range(self.FieldLength-1,-1,-1):
            #     one_shift_by_i:int = 1<<i
            #     bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            #     if bit_of_bitmask_for_this_i != 0:
            #         #This bit is in bitmask, ignore this i.
            #         continue
                
            # #4， 判断，结束。
            
            
            
            #self.all_xor, self.not_after_xor, self.best_index_to_split_from_right_side = \
            _temp__in_case_4 = self._init_only__detect_xor_info()
            if _temp__in_case_4[0]:#true xor.
                self.not_after_xor = _temp__in_case_4[1]
                self.ready_for_lookup = True
                #self.has_1 see uppon
                #self.has_0 see uppon
                #self.has_irr see uppon
                self.all_irr = False
                #self.is_dataset_sorted = True already set before
                self.is_leaf_node = True
                self.all_xor = _temp__in_case_4[0]
                #self.not_after_xor see uppon
                #self.already_const_without_irr = False#only to init. not a real result.
                self.best_index_to_split_from_right_side = -1
                self._debug__how_did_I_quit_init_func = How_did_I_quit_init_func.XOR
                
                #all_xor doesn't need the dataset.
                self.dataset = None
                return 
            else:
                #it's not a true xor. But splitting it trickily can get smaller true xor.
                self.not_after_xor = _temp__in_case_4[1]
                self.ready_for_lookup = True
                #self.has_1 see uppon
                #self.has_0 see uppon
                #self.has_irr see uppon
                self.all_irr = False
                #self.is_dataset_sorted = True already set before
                self.is_leaf_node = False
                self.all_xor = _temp__in_case_4[0]
                #self.not_after_xor see uppon
                #self.already_const_without_irr = False#only to init. not a real result.
                self.best_index_to_split_from_right_side = _temp__in_case_4[2]
                self._debug__how_did_I_quit_init_func = How_did_I_quit_init_func.BRANCH
                
                _check_all_safety = _debug__check_all_safety
                self.split(self.best_index_to_split_from_right_side, _debug__check_all_safety_in_split = _check_all_safety,
                        leaf_keep_dataset = leaf_keep_dataset, branch_keep_dataset = branch_keep_dataset)
                if not branch_keep_dataset:
                    self.dataset = None
                    pass
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
            self.is_leaf_node = True
            self.all_xor = False
            self.not_after_xor = False#only to init. not a real result.
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
        self.is_leaf_node = False
        self.all_xor = False
        self.not_after_xor = False#only to init. not a real result.
        #self.already_const_without_irr = False#only to init. not a real result.
        self._debug__how_did_I_quit_init_func = How_did_I_quit_init_func.BRANCH
        
        _check_all_safety = _debug__check_all_safety
        self.split(self.best_index_to_split_from_right_side, _debug__check_all_safety_in_split = _check_all_safety,
                leaf_keep_dataset = leaf_keep_dataset, branch_keep_dataset = branch_keep_dataset)#this line was on top.
        if not branch_keep_dataset:
            self.dataset = None
            pass
        return #end of function
    
    def has_dataset(self)->bool:
        return self.dataset is not None
    
    def _init_only__detect_xor_info(self)->tuple[bool, bool, int]:
        '''return(is full xor, needs a not after xor(if is full xor), best bit to split from_right_side(if not a full xor))'''
        score_of_xor_list_from_right_side:list[int] = []
        score_of_xor = 0
        assert self.dataset is not None
        for index_1 in range(0, self.dataset._len(), 2):
            '''case 1, last bit.
            0and1, 2and3, ind2-ind1 is 1. step is 1??, chunk is 2.'''
            index_2 = index_1+1
            data_1_tuple:tuple[int, bool] = self.dataset.data[index_1]
            data_2_tuple:tuple[int, bool] = self.dataset.data[index_2]
            #wrong assert data_1_tuple[0]^data_2_tuple[0] == 1 #not needed in real case. 
            if data_1_tuple[1]^data_2_tuple[1]:
                score_of_xor = score_of_xor +1
                pass
            pass
        score_of_xor_list_from_right_side.append(score_of_xor)
        
        for i in range(1, self.FieldLength-self.bits_already_in_use):
            '''
            case 2, 
            0and2, 1and3,,, 4and6, 5and7. ind2-ind1 is 2. step is 1, chunk is 4.
            0-4, 1-5, 2-6, 3-7,,, 8-12, 9-13, 10-14, 11-15. ind2-ind1 is 4. step is 1, chunk is 8.
            '''
            score_of_xor = 0
            index_2_minus_index_1 = 1<<i #this is also half chunk size.
            chunk_size = index_2_minus_index_1<<1
            for chunk_start_addr in range(0, self.dataset._len(), chunk_size):
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
            pass#for i
        
        min_of_score_list____from_right_side = self.dataset._len()# the max of this is len/2, so this is big enough.
        arg_min_of_score_list____from_right_side = -1
        for i in range(score_of_xor_list_from_right_side.__len__()):
            item = score_of_xor_list_from_right_side[i]
            if item<min_of_score_list____from_right_side:
                min_of_score_list____from_right_side = item
                arg_min_of_score_list____from_right_side = i
                pass
            pass
        
        if self.dataset._len()/2 == min_of_score_list____from_right_side:
            '''true xor'''
            '''return(is full xor, needs a not after xor(if is full xor), best bit to split from_right_side(if not a full xor))'''
            needs_not_after_xor = self.dataset.data[0][1]#result of the pure not addr.
            return (True, needs_not_after_xor, -2)

        '''fake xor. Split at the least xor bit.'''
        '''translate the index to actual index.'''
        _temp = 0
        for actual_bit_index___from_right_side in range(self.FieldLength):
            one_shift_by_i:int = 1<<actual_bit_index___from_right_side
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #This bit is in bitmask, ignore this i.
                continue
            if _temp == arg_min_of_score_list____from_right_side:
                '''return(is full xor, needs a not after xor(if is full xor), best bit to split from_right_side(if not a full xor))'''
                return(False, False, actual_bit_index___from_right_side)
            
            _temp = _temp + 1
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
                
            for i in range(self.FieldLength-self.bits_already_in_use):
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
            for i in range(self.FieldLength-1,-1,-1):
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
        for i in range(self.FieldLength-1,-1,-1):
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
        length = self.dataset._len()
        total_possible = 1<<(self.FieldLength-self.bits_already_in_use)
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
        true_part = DatasetField(bitmask = new_bitmask, addr=true_addr, FieldLength=self.FieldLength,
            bits_already_in_use = self.bits_already_in_use+1, dataset = Dataset(true_dataset, True), 
            with_suggest = True, suggest_has_1 = true_has_1, suggest_has_0 = true_has_0, 
            _debug__check_all_safety = __check_all_safety,
            leaf_keep_dataset = leaf_keep_dataset, branch_keep_dataset = branch_keep_dataset,
            )

        false_part = DatasetField(bitmask = new_bitmask, addr=false_addr, FieldLength=self.FieldLength,
            bits_already_in_use = self.bits_already_in_use+1, dataset = Dataset(false_dataset, True), 
            with_suggest = True, suggest_has_1 = false_has_1, suggest_has_0 = false_has_0, 
            _debug__check_all_safety = __check_all_safety,
            leaf_keep_dataset = leaf_keep_dataset, branch_keep_dataset = branch_keep_dataset,
            )
        self.children = (true_part, false_part)
        pass
        
        
    def _lookup_only___is_leaf(self)->bool:
        if self.children is None:
            return True
        else:
            return False
        
    def lookup(self, addr:int, lookup_in_leaf_dataset = False)->tuple[bool,bool]:#[None,None,bool,bool]:
        #to do :suggest, like allow_irr, better_non_irr, better_irr, true_when_irr, false_when_irr.
        
        #the sequence is the same as __init__, except for all the branch cases are priotized to the top.
        '''return (result_is_irr, result_is_true)
        
        Note. This function never look up in the self.dataset of branch.'''
        #to do:return (result_is_irr, result_is_true, is_irr_raw, is_true_raw)'''
        
        # 0, this is different. I guess a lot calling to this function is not the end, 
        # so let's detect non leaf node first.
        if not self.is_leaf_node:
            _temp__mask_of_this_bit = 1<<self.best_index_to_split_from_right_side
            this_bit_of_addr__with_shift = _temp__mask_of_this_bit&addr
            this_bit_of_addr = this_bit_of_addr__with_shift != 0
            the_child = self._get_child(this_bit_of_addr)
            return the_child.lookup(addr, lookup_in_leaf_dataset = lookup_in_leaf_dataset)
        
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
            assert self.dataset is not None
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
            raise Exception("should be a tuple.")
        else:
            if true_or_false:
                return self.children[0]
            else:
                return self.children[1]  
        #end of function
    
    def ____broken____Im_sorry____print_all_info_directly(self):
        temp = f"<DatasetField> bitmask:{self.bitmask}, addr:{self.addr:b}, FieldLength{self.FieldLength}, "
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
        temp2 += f"ready_for_lookup:{self.ready_for_lookup}, is_dataset_sorted:{self.is_dataset_sorted}, is_leaf_node:{self.is_leaf_node}, "
        temp2 += f"not_after_xor:{self.not_after_xor}, "
        print(temp2)
        pass
    
    def readable_as_tree(self, depth:int = -1, use_TF = False)->str:
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
        if self.is_leaf_node:
            if self.all_xor:
                if self.not_after_xor:
                    return addr_str+"xnor"
                else:
                    return addr_str+"xor"
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
        how_many_underscores_needs_in_the_left = self.FieldLength-bitmask_in_str.__len__()
        result = "_"*how_many_underscores_needs_in_the_left+temp_str
        return result
    
    def valid(self, dataset:Dataset, total_amount = -1):
        #if the dataset is empty, the field should also be all irr.
        if dataset._len() == 0:
            if not self.all_irr: 
                log_the_error(self.FieldLength, dataset)
                pass
            return
        #safety first.
        _last_addr_in_input = dataset.data[-1][0]
        assert _last_addr_in_input < (1<<self.FieldLength)
        
        if total_amount>=dataset._len():
            total_amount = -1
            pass
        if -1 == total_amount:#valid all.
            for item in dataset.data:
                temp_tuple:tuple[bool, bool] = self.lookup(item[0])
                #assert not temp_tuple[0]
                if temp_tuple[0]:
                    log_the_error(self.FieldLength, dataset)
                    return
                #assert item[1] == temp_tuple[1]
                if  item[1] != temp_tuple[1]:
                    log_the_error(self.FieldLength, dataset)
                    return
                pass#for
            pass#if total_amount
        else:#valid random but only part of them.
            while total_amount>0:
                item = dataset.data[random.randint(0, dataset._len()-1)]
                temp_tuple = self.lookup(item[0])
                #assert not temp_tuple[0]
                if temp_tuple[0]:
                    log_the_error(self.FieldLength, dataset)
                    return
                #assert item[1] == temp_tuple[1]
                if  item[1] != temp_tuple[1]:
                    log_the_error(self.FieldLength, dataset)
                    return
                #tail
                total_amount = total_amount -1
                pass#while
            pass#else of if -1 == total_amount
        pass#end of function.
            
            
    def valid_irr(self, dataset:Dataset, total_amount_irr = -1):
        #if the dataset is empty, the field should also be all irr.
        if dataset._len() == 0:
            if not self.all_irr: 
                log_the_error(self.FieldLength, dataset)
                pass
            return
        #safety first.
        assert dataset.is_sorted
        _last_addr_in_input = dataset.data[-1][0]
        assert _last_addr_in_input < (1<<self.FieldLength)
        
        total_possible_amount = 1<<self.FieldLength
        total_possible_amount_of_irr = total_possible_amount - dataset._len()
        if total_amount_irr>=total_possible_amount_of_irr:
            total_amount_irr = -1
            pass
        if -1 == total_amount_irr:
            #valid all irr.
            #valid all irr.
            #valid all irr.
            _, irr_addr_list = dataset.get_irr_addr___sorts_self(self.FieldLength)
            #dataset already non-empty.
            for irr_addr in irr_addr_list:
                temp_tuple = self.lookup(irr_addr, lookup_in_leaf_dataset=True)
                #assert item[0]
                if not temp_tuple[0]:
                    log_the_error(self.FieldLength, dataset,filename="wrong irr case log.txt")
                    return
                pass
            pass#-1 == total_amount_irr
        else:#only check random irrelevant:
            if total_possible_amount_of_irr< total_possible_amount/3:
                # not very many irr, so first get the list
                #random pick from irr list
                #random pick from irr list
                #random pick from irr list
                _, irr_addr_list = dataset.get_irr_addr___sorts_self(self.FieldLength)
                #dataset already non-empty.
                while total_amount_irr>0:
                    rand_addr:int = irr_addr_list[random.randint(0, irr_addr_list.__len__()-1)]
                    temp_tuple = self.lookup(rand_addr, lookup_in_leaf_dataset=True)
                    #assert temp_tuple[0]#irr
                    if not temp_tuple[0]:
                        log_the_error(self.FieldLength, dataset, filename="wrong irr case log.txt")
                        return
                    #tail
                    total_amount_irr = total_amount_irr -1
                    pass#while
                pass#if 
            else:#a lot irr, random and check if it's a irr in ref set, and then.
                # guess
                # guess
                # guess
                total_trial_amount = total_amount_irr*20
                already_guessed:set[int] = set()
                one_shift_field_len_minus_one = (1<<self.FieldLength )-1
                #one_shift_field_len = 1<<self.FieldLength
                #_not_important_incr = 0
                while total_trial_amount>0 and total_amount_irr>0:
                    guess_addr = random.randint(0, one_shift_field_len_minus_one)
                    #_not_important_incr = _not_important_incr + int(one_shift_field_len*61/337)
                    #_not_important_incr = _not_important_incr % one_shift_field_len
                    #guess_addr = (guess_addr+_not_important_incr) %one_shift_field_len
                    
                    if guess_addr in already_guessed:
                        #tail
                        total_trial_amount = total_trial_amount -1
                        continue
                    else:
                        already_guessed.add(guess_addr)
                        pass
                    #find the item in dataset. But if it finds, it's bad guess. 
                    #binary search # two valid style.
                    #guess_addr is the input. key = lambda item:dataset[item][0]
                    found = False
                    left = 0
                    right = dataset._len()-1
                    while left<=right:
                        mid:int = (left+right)//2
                        temp_addr = dataset.data[mid][0]
                        if guess_addr<temp_addr:
                            right = mid-1# two valid style.
                            continue
                        elif guess_addr>temp_addr:
                            left = mid+1# two valid style.
                            continue
                        else:#guess_addr == dataset[mid][0]:
                            found = True
                            break
                        pass#while of binary search.
                    if found: # the guess_addr is found in dataset, so it's a relevant. Ignore it.
                        #tail
                        total_trial_amount = total_trial_amount -1
                        continue
                    
                    #now the guess_addr is a irr according to dataset.
                    temp_tuple = self.lookup(guess_addr, lookup_in_leaf_dataset=True)
                    
                    #assert temp_tuple[0]#irr
                    if not temp_tuple[0]:
                        log_the_error(self.FieldLength, dataset, filename="wrong irr case log.txt")
                        return
                    #tail
                    total_trial_amount = total_trial_amount -1
                    total_amount_irr = total_amount_irr -1
                    pass#while
                if total_amount_irr>0:
                    log_the_error(self.FieldLength, dataset, "not totally tested.txt")
                    return
                pass#else
            pass#if -1 == total_amount_irr:#valid all irr.
        pass# end of function.
            
    #end of class
    
if "some special case" and True:
    if "2025 oct 14":
        FieldLength = 1
        dataset = Dataset([(0, True)])
        a_DatasetField = DatasetField._new(FieldLength, dataset)
        a_DatasetField.valid(dataset)
        
        
        
        pass
    
    if "1" and True:
        FieldLength = 7
        addr = 114
        bitmask = 114
        bits_already_in_use = 4
        dataset_big = Dataset([(0, False), (2, True), (4, False), (9, True), (13, False), (17, True), (25, True), (26, False), (28, True), (29, False), (30, False), (31, True), (35, True), (37, True), (38, True), (39, False), (41, True), (42, False), (44, True), (46, False), (48, True), (50, True), (52, True), (53, True), (58, True), (60, True), (61, True), (62, True), (63, True), (64, False), (68, True), (71, False), (72, True), (74, True), (77, True), (78, True), (79, True), (83, True), (85, True), (87, True), (88, False), (92, True), (93, True), (95, True), (97, True), (99, True), (100, True), (103, True), (107, True), (110, True), (113, True), (117, True), (119, True), (121, True), (123, True), (124, True), (126, True), (127, False)])
        dataset = dataset_big.get_subset(bitmask, addr)
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
        a_DatasetField = DatasetField(bitmask, addr, FieldLength, bits_already_in_use, dataset)
        a_DatasetField.valid(dataset)
        pass
    
    if "10_1_1__ case" and True:
        dataset, FieldLength = Dataset.from_str("10_1_1__")
        a_DatasetField = DatasetField._new(FieldLength, dataset)
        a_DatasetField.valid(dataset)
    pass
        
if "valid function" and True:
    if "correct dataset" and True:
        FieldLength = 3
        dataset = Dataset()
        a_DatasetField = DatasetField._new(FieldLength, dataset)
        # empty
        a_DatasetField.valid(dataset)
        
        FieldLength = 2
        dataset = Dataset([(0,True), (1,False), (3,True),])
        a_DatasetField = DatasetField._new(FieldLength, dataset)
        # all relevant
        a_DatasetField.valid(dataset)
        # random relevant
        a_DatasetField.valid(dataset, total_amount=2)
        
        FieldLength = 1
        dataset = Dataset([(0,True), ])
        a_DatasetField = DatasetField._new(FieldLength, dataset)
        # big number.
        a_DatasetField.valid(dataset, total_amount=111)
        
        FieldLength = 2
        dataset = Dataset([(0,True), (1,False), (3,True),])
        a_DatasetField = DatasetField._new(FieldLength, dataset, leaf_keep_dataset = True)
        tree = a_DatasetField.readable_as_tree()
        # all irr
        a_DatasetField.valid(dataset)
        a_DatasetField.valid_irr(dataset)
        # random from irr list
        a_DatasetField.valid_irr(dataset, total_amount_irr=1)
        
        FieldLength = 3
        dataset = Dataset([(0,True), (1,False), (3,True), (4,False), ])
        a_DatasetField = DatasetField._new(FieldLength, dataset, leaf_keep_dataset = True)
        # guess irr
        a_DatasetField.valid_irr(dataset, total_amount_irr=2)
        # big number.
        a_DatasetField.valid_irr(dataset, total_amount_irr=111)
        pass
        
    if "wrong dataset" and True:
        FieldLength = 2
        dataset = Dataset([(0,True), (1,False), ])
        a_DatasetField = DatasetField._new(FieldLength, dataset)
        # all relevant
        wrong_dataset_1 = Dataset([(0,False), ])
        #a_DatasetField.valid(wrong_dataset_1)
        wrong_dataset_1 = Dataset([(2,False), ])
        #a_DatasetField.valid(wrong_dataset_1)
        # random relevant
        wrong_dataset_2 = Dataset([(0,False), (1,True), ])
        #a_DatasetField.valid(wrong_dataset_2, total_amount=1)
        wrong_dataset_2 = Dataset([(2,False), (3,True), ])
        #a_DatasetField.valid(wrong_dataset_2, total_amount=1)
        # all irr
        wrong_dataset_3 = Dataset([(1,True), (2,True), (3,False), ])
        #a_DatasetField.valid_irr(wrong_dataset_3, total_amount_irr=-1)
        
        # random from irr list
        FieldLength = 3
        dataset = Dataset([(0,True), (1,False), (2,True), (3,True), (4,False), (5,False), (6,False), (7,False), ])
        a_DatasetField = DatasetField._new(FieldLength, dataset)
        wrong_dataset_4 = Dataset([(0,True), (1,True), (2,True), (3,True), (4,True), (5,True), ])
        #a_DatasetField.valid_irr(wrong_dataset_4, total_amount_irr=1)
        # guess irr
        wrong_dataset_5 = Dataset([(0,True), ])
        #a_DatasetField.valid_irr(wrong_dataset_5, total_amount_irr=1)
        pass
    pass
    
if "readable addr" and True:
    for addr in range(0b10):
        a_DatasetField = DatasetField(bitmask = 0b1, addr = addr, FieldLength=1, bits_already_in_use=1, \
                        dataset = Dataset(), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass
    
    for addr in range(0b100):
        a_DatasetField = DatasetField(bitmask = 0b11, addr = addr, FieldLength=2, bits_already_in_use=2, \
                        dataset = Dataset(), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass

    for addr in range(0b1000):
        a_DatasetField = DatasetField(bitmask = 0b111, addr = addr, FieldLength=3, bits_already_in_use=3, \
                        dataset = Dataset(), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass
    
    for addr in range(0b10000):
        a_DatasetField = DatasetField(bitmask = 0b1111, addr = addr, FieldLength=4, bits_already_in_use=4, \
                        dataset = Dataset(), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass
    
    for addr in range(0b100000):
        a_DatasetField = DatasetField(bitmask = 0b11111, addr = addr, FieldLength=5, bits_already_in_use=5, \
                        dataset = Dataset(), with_suggest=False,_debug__check_all_safety = True)
        temp_str = a_DatasetField.get_readable_addr()
        assert addr == int(temp_str,2)
        pass
    
    pass

if "readable tree" and True:
    #111111111111111可以继续
    a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=1, bits_already_in_use=0, \
                        dataset = Dataset(), with_suggest=False,_debug__check_all_safety = True)
    tree_str = a_DatasetField.readable_as_tree()
    assert tree_str == "_:ir"
    tree_str_TF = a_DatasetField.readable_as_tree(use_TF=True)
    assert tree_str_TF == "_:ir"
    
    def str_to_readable_tree(input:str)->tuple[str, str]:
        dataset_tuple = Dataset.from_str(input)
        a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=dataset_tuple[1], bits_already_in_use=0, \
                            dataset = dataset_tuple[0], with_suggest=False,_debug__check_all_safety = True)
        tree_str = a_DatasetField.readable_as_tree()
        tree_str_TF = a_DatasetField.readable_as_tree(use_TF=True)
        return (tree_str, tree_str_TF)
    if "already tested" and False:
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
        assert tree_str_TF == "__:xor"
        
        tree_str, tree_str_TF = str_to_readable_tree("01101001")
        assert tree_str_TF == "___:xor"
        
        tree_str, tree_str_TF = str_to_readable_tree("01100110")
        assert tree_str_TF == "___:(0__:xor, 1__:xor)"
        
        tree_str, tree_str_TF = str_to_readable_tree("0110100110010110")
        assert tree_str_TF == "____:xor"
    
    tree_str, tree_str_TF = str_to_readable_tree("0110111111110110")
    assert tree_str_TF == "____:(___0:(__00:xor, __10:T), ___1:(__01:T, __11:xor))"
    
    pass

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
    
    if "1 bit input, 1 free bit" and False:
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
        pass
        
    if "2 bits input, has 1,0 and irr" and False:
        if "already tested" and False:
            dataset = [(0b00,True), (0b01,True), (0b10,False), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=2, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert 1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 2 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert not a_DatasetField.is_leaf_node
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 1 == best_index_to_split
            assert 3 == best_abs_of_num_of_same
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
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
                
            temp = a_DatasetField.lookup(0b11)
            assert temp[0]#irr
        
            
            dataset = [(0b00,True), (0b01,False), (0b10,True), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=2, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert 0 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 2 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert not a_DatasetField.is_leaf_node
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 0 == best_index_to_split
            assert 3 == best_abs_of_num_of_same
            #assert not a_DatasetField.not_after_xor not important.
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "__"
            
            addr_1__child = a_DatasetField._get_child(true_or_false=True)
            readable_addr = addr_1__child.get_readable_addr()
            assert readable_addr == "_1"
            assert addr_1__child.has_0
            assert addr_1__child.children is None
            
            addr_0__child = a_DatasetField._get_child(true_or_false=False)
            readable_addr = addr_0__child.get_readable_addr()
            assert readable_addr == "_0"
            assert addr_0__child.has_1
            assert not addr_0__child.has_irr
            assert addr_0__child.children is None
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
                
            temp = a_DatasetField.lookup(0b11)
            assert temp[0]#irr
            
            
            
            #basically a xor, but replaced with 1 irr.
            dataset = [(0b00,True), (0b01,False), (0b10,False), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=2, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert not a_DatasetField.all_xor
            assert 1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is not None
            assert 2 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert not a_DatasetField.is_leaf_node
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert 1 == best_index_to_split
            assert 1 == best_abs_of_num_of_same
            #assert not a_DatasetField.not_after_xor not important.
            
            addr_1__child = a_DatasetField._get_child(true_or_false=True)
            readable_addr = addr_1__child.get_readable_addr()
            assert readable_addr == "1_"
            assert addr_1__child.is_leaf_node
            assert addr_1__child.has_0
            assert addr_1__child.has_irr
            
            addr_0__child = a_DatasetField._get_child(true_or_false=False)
            readable_addr = addr_0__child.get_readable_addr()
            assert readable_addr == "0_"
            assert not addr_0__child.is_leaf_node
            assert addr_0__child.has_1
            assert addr_0__child.has_0
            assert not addr_0__child.has_irr
            assert not addr_0__child.all_xor
            
            addr_00_child = addr_0__child._get_child(true_or_false=False)
            readable_addr = addr_00_child.get_readable_addr()
            assert readable_addr == "00"
            assert addr_00_child.is_leaf_node
            assert addr_00_child.has_1
            assert not addr_00_child.has_0
            assert not addr_00_child.has_irr
            
            addr_01_child = addr_0__child._get_child(true_or_false=True)
            readable_addr = addr_01_child.get_readable_addr()
            assert readable_addr == "01"
            assert addr_01_child.is_leaf_node
            assert not addr_01_child.has_1
            assert addr_01_child.has_0
            
            for item in dataset:
                temp = a_DatasetField.lookup(item[0])
                assert item[1] == temp[1]
                pass
                
            temp = a_DatasetField.lookup(0b11)
            assert temp[0]#irr
        
        if "2 bits xor(and xnor)" and False:
            #xor.
            dataset = [(0b00,True), (0b01,False), (0b10,False), (0b11,True), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=2, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert a_DatasetField.all_xor
            assert -1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 2 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_leaf_node
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            assert a_DatasetField.not_after_xor
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "__"
            for item in dataset:
                    temp = a_DatasetField.lookup(item[0])
                    assert item[1] == temp[1]
                    pass
            
            #xnor. But in code it's a not after xor.
            dataset = [(0b00,False), (0b01,True), (0b10,True), (0b11,False), ]
            a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=2, bits_already_in_use=0, \
                            dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
            assert not a_DatasetField.all_irr
            assert 0 == a_DatasetField.addr
            assert a_DatasetField.all_xor
            assert -1 == a_DatasetField.best_index_to_split_from_right_side
            assert 0 == a_DatasetField.bitmask
            assert 0 == a_DatasetField.bits_already_in_use
            assert a_DatasetField.children is None
            assert 2 == a_DatasetField.FieldLength
            assert not a_DatasetField.get_already_const_without_irr()
            assert not a_DatasetField.has_irr
            assert a_DatasetField.has_0
            assert a_DatasetField.has_1
            assert a_DatasetField.is_leaf_node
            best_index_to_split, best_abs_of_num_of_same = a_DatasetField._detect_best_bit_to_split()
            assert -1 == best_index_to_split
            assert 0 == best_abs_of_num_of_same
            assert not a_DatasetField.not_after_xor
            readable_addr = a_DatasetField.get_readable_addr()
            assert readable_addr == "__"
            for item in dataset:
                    temp = a_DatasetField.lookup(item[0])
                    assert item[1] == temp[1]
                    pass

    if "3 bits fake xor" and True:
        '''Generally, xor is a 2 bits structure. The result is either 1001 or 0110. 
        People call the 1001 xnor, but in this tool, they are both xor.
        In this tool, more than 2 bits can also be treated as xor, to simplify and speed up.
        When input(a) is true, input(bc) is 1001, when input(a) is false, input(bc) is 0110, 
        they can combine as a 3 bits xor. But when 2 1001 combine, it's not a xor. 
        This 3 bits fake xor is to validate this special case.        
        '''
        FieldLength = 3
        #dataset is 2 0110 combined.
        
        dataset = Dataset([(0, False), (1, True), (2, True), (3, False), (4, False), (5, True), (6, True), (7, False)], True)
        a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=FieldLength, bits_already_in_use=0, \
                                dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        
        if True:
            assert a_DatasetField.bitmask == 0
            assert a_DatasetField.addr == 0
            assert a_DatasetField.FieldLength == 3
            assert a_DatasetField.bits_already_in_use == 0
            #assert a_DatasetField.dataset:list[tuple[int,bool]]
            assert a_DatasetField._debug__how_did_I_quit_init_func == How_did_I_quit_init_func.BRANCH
            assert a_DatasetField.best_index_to_split_from_right_side == 2
            assert a_DatasetField._it_was_a_temp_var__best_abs_of_num_of_same == 0
            assert a_DatasetField.children is not None
            assert a_DatasetField.ready_for_lookup
            assert a_DatasetField.has_1
            assert a_DatasetField.has_0
            assert a_DatasetField.has_irr == False
            assert a_DatasetField.is_dataset_sorted
            assert a_DatasetField.is_leaf_node == False
            assert a_DatasetField.all_xor == False
            assert a_DatasetField.not_after_xor == False
            assert a_DatasetField.all_irr ==False
            
            addr_1_child = a_DatasetField._get_child(true_or_false=True)
            assert addr_1_child.bitmask == 0b100
            assert addr_1_child.addr == 0b100
            assert addr_1_child.FieldLength == 3
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
            assert addr_1_child.is_leaf_node #== False
            assert addr_1_child.all_xor #== False
            assert addr_1_child.not_after_xor == False
            assert addr_1_child.all_irr ==False
            
            addr_0_child = a_DatasetField._get_child(true_or_false=False)
            assert addr_0_child.bitmask == 0b100
            assert addr_0_child.addr == 0b000
            assert addr_0_child.FieldLength == 3
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
            assert addr_0_child.is_leaf_node #== False
            assert addr_0_child.all_xor #== False
            assert addr_0_child.not_after_xor == False
            assert addr_0_child.all_irr ==False
            pass
        
        for item in dataset.data:
            temp_tuple = a_DatasetField.lookup(item[0])
            assert not temp_tuple[0]
            assert item[1] == temp_tuple[1]
            pass
        irr_addr_list_tuple = dataset.get_irr_addr___sorts_self(FieldLength)
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
        FieldLength = 4
        dataset = Dataset([(0, False), (1, True), (2, True), (3, False), (4, True), (5, True), (6, True), (7, True), (8, True), 
                (9, True), (10, True), (11, True), (12, False), (13, True), (14, True), (15, False)])
        a_DatasetField = DatasetField(bitmask = 0, addr = 0, FieldLength=FieldLength, bits_already_in_use=0, \
                                dataset = dataset, with_suggest=False,_debug__check_all_safety = True)
        for item in dataset.data:
            temp_tuple = a_DatasetField.lookup(item[0])
            assert not temp_tuple[0]
            assert item[1] == temp_tuple[1]
            pass
        irr_addr_list_tuple = dataset.get_irr_addr___sorts_self(FieldLength)
        assert irr_addr_list_tuple[0]
        assert irr_addr_list_tuple[1].__len__() == 0
        
        if irr_addr_list_tuple[1].__len__() != 0:
            for irr_addr in irr_addr_list_tuple[1]:
                temp = a_DatasetField.lookup(irr_addr)
                assert item[0]
                pass
            pass
        
        pass
    
    if "insert bits to addr and check" and True:
        dataset, old_FieldLength = Dataset.from_str("1001,0000,0000,1001")
        FieldLength = dataset.add_const_bit_into_addr(old_FieldLength, 2, True)
        assert FieldLength == 5
        a_DatasetField = DatasetField._new(FieldLength, dataset)
        tree_str_TF = a_DatasetField.readable_as_tree(use_TF=True)
        #111111111111可以继续。
        #1w
        pass


if "random dataset test   slow" and True:
    # empty
    print("empty, line:"+str(_line_()))
    for ____total_iter in range(13):
        if ____total_iter%500 == 0:
            print(____total_iter, end=", ")
            pass
        for FieldLength in range(1, 62):
            dataset = Dataset()
            a_DatasetField = DatasetField._new(FieldLength, dataset)
            
            assert a_DatasetField.all_irr
            assert a_DatasetField.is_leaf_node
            assert a_DatasetField.children is None
            pass#for FieldLength in range(1, 12):
        pass#for ____total_iter
    print()
    
    
    for _while_count in range(1, 1111111111):
        if  (_while_count % 1000) == 0:
            print(_while_count, end=", ")
            pass
        
        # full
        print("full,     line:"+str(_line_()) +"      "+ str(datetime.now().time()))
        for ____total_iter in range(200):
            for FieldLength in range(1, 12):
                temp_rand = random.random()*0.6
                p_False = 0.2+temp_rand
                p_True = 0.8-temp_rand+0.001
                
                dataset = Dataset.rand__sorted(FieldLength, p_False = p_False, p_True = p_True)
                a_DatasetField = DatasetField._new(FieldLength, dataset)
                a_DatasetField.valid(dataset, total_amount = 100)
                a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
                pass#for FieldLength
            pass#for ____total_iter

        # dense, non full
        print("dense,    line:"+str(_line_()) +"      "+ str(datetime.now().time()))
        for ____total_iter in range(1000):
            for FieldLength in range(1, 12):
                temp_rand = random.random()*0.6
                temp_rand2 = random.random()*0.6+0.2
                p_False = (0.2+temp_rand)*temp_rand2
                p_True = (0.8-temp_rand)*temp_rand2
                
                dataset = Dataset.rand__sorted(FieldLength, p_False = p_False, p_True = p_True)
                a_DatasetField = DatasetField._new(FieldLength, dataset)
                a_DatasetField.valid(dataset, total_amount = 100)
                a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
                pass#for FieldLength
            pass#for ____total_iter
        
        # sparse
        print("sparse,   line:"+str(_line_()) +"      "+ str(datetime.now().time()))
        for ____total_iter in range(3000):
            for FieldLength in range(1, 12):
                temp_rand = random.random()*0.6
                temp_rand2 = random.random()*0.2+0.05
                p_False = (0.2+temp_rand)*temp_rand2
                p_True = (0.8-temp_rand)*temp_rand2
                
                dataset = Dataset.rand__sorted(FieldLength, p_False = p_False, p_True = p_True)
                a_DatasetField = DatasetField._new(FieldLength, dataset)
                a_DatasetField.valid(dataset, total_amount = 100)
                a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
                pass#for FieldLength
            pass#for ____total_iter
        
        # one side
        print("one side, line:"+str(_line_()) +"      "+ str(datetime.now().time()))
        for ____total_iter in range(500):
            for FieldLength in range(1, 12):
                if random.random() <0.5:
                    p_False = random.random()*0.6
                    p_True = 0.
                    pass
                else:
                    p_False = 0.
                    p_True = random.random()*0.6
                    pass
                
                dataset = Dataset.rand__sorted(FieldLength, p_False = p_False, p_True = p_True)
                a_DatasetField = DatasetField._new(FieldLength, dataset)
                a_DatasetField.valid(dataset, total_amount = 100)
                a_DatasetField.valid_irr(dataset, total_amount_irr = 100)
                assert a_DatasetField.children is None
                pass#for FieldLength
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
#######
#######
#######
#######
#######
#######
#######




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

    
class Dataset_Set:
    data:list[Dataset]
    is_sorted:bool
    def __init__(self, input:list[Dataset] = [], is_sorted = False):
        self.data = input
        self.is_sorted = is_sorted
        pass
    def _len(self)->int:
        return self.data.__len__()
    def sort(self):
        for dataset in self.data:
            dataset.sort()
            pass
        self.is_sorted = True
        pass
    
    if "old seperated" and False:
        @staticmethod
        def binary_into_boollist(input:int, length:int)->list[bool]:
            assert input >=0
            boollist = []
            for _ in range(length):
                boollist.append(False)
                pass
            the_str = f"{input:b}"
            offset = length-the_str.__len__()
            assert offset >= 0
            for i in range(the_str.__len__()-1,-1,-1):
                char = the_str[i]
                if "1" == char:
                    boollist[i+offset] = True
                    pass
                pass
            return boollist
        
        @staticmethod
        def boollist_into_itemset(boollist:list[bool], addr:int)->list[tuple[int,bool]]:
            result:list[tuple[int,bool]] = []
            for the_bool in boollist:
                result.append((addr, the_bool))
                pass
            return result
            pass
        pass
        
        
    
    def add_binary(self, input:int, length:int, addr:int):
        #safety first
        assert self._len() == length
        _input_bits = 0
        _temp_input = input
        while _temp_input>0:
            _input_bits = _input_bits +1
            _temp_input = _temp_input <<1
            pass
        assert _input_bits <=length
        assert input >=0
        
        bool_list = []
        for _ in range(length):
            bool_list.append(False)
            pass
        the_str = f"{input:b}"
        offset = length-the_str.__len__()
        assert offset >= 0
        for i in range(the_str.__len__()-1,-1,-1):
            char = the_str[i]
            if "1" == char:
                bool_list[i+offset] = True
                pass
            pass
    
        # itemset:list[tuple[int,bool]] = []
        # for the_bool in bool_list:
        #     itemset.append((addr, the_bool))
        #     pass
   
        assert self._len() == itemset.__len__()
        for i in range(self._len()):
            self.data[i].data.append((addr, bool_list[i]))
            pass
        pass
    

    
if "test" and True:
    if "sort" and True:
        datasetset = Dataset_Set([Dataset([(2,True),(1,True),]), Dataset([(33,True),(11,True),]), ])
        datasetset.sort()
        assert datasetset.data[0].data == [(1,True),(2,True),]
        assert datasetset.data[1].data == [(11,True),(33,True),]
        pass
    
    if "binary_into_boollist" and True:
        boollist = Dataset_Set.binary_into_boollist(0b1010,4)
        assert boollist == [True,False,True,False,]
        boollist = Dataset_Set.binary_into_boollist(0b1010,6)
        assert boollist == [False,False,True,False,True,False,]
        boollist = Dataset_Set.binary_into_boollist(0b11010,6)
        assert boollist == [False,True,True,False,True,False,]
        pass
    if "boollist_into_itemset" and True:
        boollist = Dataset_Set.binary_into_boollist(0b1010, 6)
        itemset = Dataset_Set.boollist_into_itemset(boollist, 123)  
        assert itemset == [(123,False),(123,False),(123,True),(123,False),(123,True),(123,False),]
        pass

    
    
    
    pass

def get_adder_testset_partly(input_bit_amount:int, proportion = 0.2):#->list[list[tuple[int,bool]]]:
    total_bit_amount = input_bit_amount*2+1
    total_result_bit_amount = input_bit_amount+1
    one_shift_by__total_bit_amount = 1<<total_bit_amount
    assert proportion<0.8, "don't torture the also. You don't need a 0.8+ proportion. Or get some other also."
    assert proportion>0.
    amount_needed = int(proportion * one_shift_by__total_bit_amount)
    if 0 == amount_needed:
        amount_needed = 1
        pass
    
    assert amount_needed > 0
    temp_set = set()
    while True:
        a_rand_num = random.randint(0, one_shift_by__total_bit_amount-1)
        temp_set.add(a_rand_num)
        if temp_set.__len__() == amount_needed:
            break
        pass#while
    #1111111111111111w 继续。
        
        
    
    
if "test" and True:
    assert False
    get_adder_testset_partly()
    
    
        
    

    


    
    
    
    
    
    
    
    

assert False , "要不要做一个专门的fake xor检测？？？"
assert False , "那加法乘法那些来验证一下。包括外延能力。"



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

