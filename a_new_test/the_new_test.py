# ok, cool. I didn't write py for 8 mon.
# a = int( "1000", 2)
# b = 0b111
# s = "{:b}".format(a)
# ss = s[3]
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

#part1 infrastructures of logic algebra.
#part1 infrastructures of logic algebra.
#part1 infrastructures of logic algebra.


class LogicNode_Var:
    is_const:bool
    index:int
    is_non_not:bool
    value:bool
    name:Optional[str]
    def __init__(self, is_const:bool, index:int, is_non_not:bool, 
                value:bool = False, name:Optional[str]=None):
        self.is_const = is_const
        self.index = index
        self.is_non_not = is_non_not
        self.value = value
        self.name = name
        pass
    def set_value(self, new_value:bool):
        self.value = new_value
        pass
    def get_value(self)->bool:
        result:bool = not (self.value^self.is_non_not)
        return result
    def __get_name_when_const(self)->str:
        if self.is_non_not:
            return "1"
        else:
            return "0"
        #end of function    
    def __get_not_str(self)->str:
        if self.is_non_not:
            return ""
        else:
            return "'"
        #end of function    
    def __str__(self)->str:
        if self.is_const:
            result = self.__get_name_when_const()
            return result
        result = "<{}>".format(self.index)+self.__get_not_str()
        return result
    def _str_name(self)->str:
        if self.is_const:
            result = self.__get_name_when_const()
            return result
        
        name:str
        if self.name is None:
            assert self.index<26, "this can only print out a to z."
            name = chr(self.index+97)
            pass
        else:
            name = self.name
            pass
        result = name+self.__get_not_str()#+" "
        return result
    def _str_binary(self)->str:
        if self.is_const:
            result = self.__get_name_when_const()
            return result
        
        addr:str
        addr = "<{:b}>".format(self.index)
        result = addr+self.__get_not_str()
        return result
    #end of class
if "test get_value" and False:
    print(LogicNode_Var(True, 0, False).get_value())
    print(LogicNode_Var(True, 0, True).get_value())
    print(LogicNode_Var(True, 0, False, False).get_value())
    print(LogicNode_Var(True, 0, False, True).get_value())
    print(LogicNode_Var(True, 0, True, False).get_value())
    print(LogicNode_Var(True, 0, True, True).get_value())
    print(LogicNode_Var(False, 0, False, False).get_value())
    print(LogicNode_Var(False, 0, False, True).get_value())
    print(LogicNode_Var(False, 0, True, False).get_value())
    print(LogicNode_Var(False, 0, True, True).get_value())
    pass
if "test _str_" and False:
    print(LogicNode_Var(True, 0, False))
    print(LogicNode_Var(True, 0, True))
    print(LogicNode_Var(False, 0, True))
    print(LogicNode_Var(False, 1, True))
    print(LogicNode_Var(False, 1, False))
    print(LogicNode_Var(False, 2, False))
    print(LogicNode_Var(False, 0, True )._str_name())
    print(LogicNode_Var(False, 1, True )._str_name())
    print(LogicNode_Var(False, 1, False)._str_name())
    print(LogicNode_Var(False, 2, False)._str_name())
    print(LogicNode_Var(False, 0, True )._str_binary())
    print(LogicNode_Var(False, 1, True )._str_binary())
    print(LogicNode_Var(False, 1, False)._str_binary())
    print(LogicNode_Var(False, 2, False)._str_binary())
    pass





class Logic_Op(Enum):
    AND = 1
    OR = 2
    XOR = 3
    pass
class LogicNode_Op:
    op:Logic_Op
    not_the_output:bool
    children:list[Union[LogicNode_Var, 'LogicNode_Op']]
    def __init__(self, op:Logic_Op, not_the_output:bool = False,
                children:list[Union[LogicNode_Var, 'LogicNode_Op']] = []):
        self.op = op
        self.not_the_output = not_the_output
        self.children = children
        pass
    def get_value(self)->bool:
        zeroth_childUnion:Union[LogicNode_Var, 'LogicNode_Op'] = self.children[0]
        result:bool = zeroth_childUnion.get_value()#maybe recursive.
        # if zeroth_childUnion is LogicNode_Op:#optimizable
        #     result = zeroth_childUnion.get_value()
        # elif zeroth_childUnion is LogicNode_Var:
        #     result = zeroth_childUnion.get_value()
        #pass
        #return True
        match self.op:
            case Logic_Op.AND:
                for i in range(1,self.children.__len__()):
                    item = self.children[i]
                    result = result & item.get_value()
                    pass
                pass
            case Logic_Op.OR:
                for i in range(1,self.children.__len__()):
                    item = self.children[i]
                    result = result | item.get_value()
                    pass
                pass
            case Logic_Op.XOR:
                for i in range(1,self.children.__len__()):
                    item = self.children[i]
                    result = result ^ item.get_value()
                    pass
                pass
            case _:
                assert False
        return result
    
    #to do 等价变换。估计好几个。
    # def apply_not_to_children(self):
    #     if not self.not_the_output:
    #         return
    #def simplify(self):
        #这个函数是不是应该写在容器里面？
        #估计要写的还有点多。
        #至少要写的，如果子项有常数，有一些可以直接变成整个的常数。
    
    def __get_not_str(self)->tuple[str,str]:
        if self.not_the_output:
            return ("","")
        else:
            return ("(",")'")
        #end of function    
    def __get_link_symbol(self, explicit_and:bool=False)->str:
        match self.op:
            case Logic_Op.AND:
                if explicit_and:
                    return"&"
                else:
                    return""
            case Logic_Op.OR:
                return"+"
            case Logic_Op.XOR:
                return"^"
            case _:
                assert False
        #end of function    
        
    def __str__(self)->str:
        children:list[str] = []
        for child in self.children:
            children.append(child.__str__())
            pass
        link_symbol = self.__get_link_symbol()
        main_str = link_symbol.join(children)
        start_str, end_str = self.__get_not_str()
        result = start_str+main_str+end_str
        return result
    def _str_name(self)->str:
        children:list[str] = []
        for child in self.children:
            children.append(child._str_name())
            pass
        link_symbol = self.__get_link_symbol()
        main_str = link_symbol.join(children)
        start_str, end_str = self.__get_not_str()
        result = start_str+main_str+end_str
        return result
    def _str_binary(self)->str:
        children:list[str] = []
        for child in self.children:
            children.append(child._str_binary())
            pass
        link_symbol = self.__get_link_symbol()
        main_str = link_symbol.join(children)
        start_str, end_str = self.__get_not_str()
        result = start_str+main_str+end_str
        return result
    #end of class
if "test" and True:
    children:list[LogicNode_Var | LogicNode_Op] = []
    children.append(LogicNode_Var(True,0,True,True))
    children.append(LogicNode_Var(True,0,True,False))
    children.append(LogicNode_Var(False,0,True,True))
    children.append(LogicNode_Var(False,1,True,False))
    children.append(LogicNode_Var(False,2,False,True))
    children.append(LogicNode_Var(False,3,False,False))
    print(LogicNode_Op(Logic_Op.AND,False,children))
    print(LogicNode_Op(Logic_Op.AND,True,children))
    print(LogicNode_Op(Logic_Op.OR,False,children))
    print(LogicNode_Op(Logic_Op.XOR,False,children))
    print()
    print(LogicNode_Op(Logic_Op.AND,False,children)._str_name())
    print(LogicNode_Op(Logic_Op.AND,True,children) ._str_name())
    print(LogicNode_Op(Logic_Op.OR,False,children) ._str_name())
    print(LogicNode_Op(Logic_Op.XOR,False,children)._str_name())
    print()
    print(LogicNode_Op(Logic_Op.AND,False,children)._str_binary())
    print(LogicNode_Op(Logic_Op.AND,True,children) ._str_binary())
    print(LogicNode_Op(Logic_Op.OR,False,children) ._str_binary())
    print(LogicNode_Op(Logic_Op.XOR,False,children)._str_binary())
    pass




继续。后面要写容器，和最最基本的化简。化简写在容器里面。



















def str_as_bin(input:int, N:Optional[int]=None)->str:
    '''Int to str. 
    
    With auto filling 0s to the left is the second param is specified.'''
    temp = "{:b}".format(input)
    if not N:
        return temp
    add_0_to_left:int = N-temp.__len__()
    result = ("0"*add_0_to_left)+temp
    return result
if "test" and False:
    a = str_as_bin(0,1)
    b = str_as_bin(0,2)
    c = str_as_bin(1,1)
    d = str_as_bin(1,2)
    e = str_as_bin(5,6)
    aa = str_as_bin(0)
    bb = str_as_bin(1)
    cc = str_as_bin(5)
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

def addr_fits(to_test:int, bit_in_use_mask:int, fixed_addr:int)->bool:
    '''Get item from dataset, the formation is like addr:1or0.
    example: addr(left) is 0b111, right is 1 or 0.
    to_test is the addr. 
        
    This func is a bit redundant, if you are sure the fixed_addr is valid, you don't have to bit_and it.
    Anyway, let's get it to work first.'''
    useful_part_from_to_test = to_test&bit_in_use_mask
    useful_part_from_fixed_addr = fixed_addr&bit_in_use_mask
    #test_this = fixed_addr&useful_part
    result = useful_part_from_to_test == useful_part_from_fixed_addr
    return result
if "test" and False:
    #0bit
    assert         is_addr_valid(int("0",2), int("0",2))
    assert addr_fits(int("0",2), int("0",2), int("0",2))
    assert addr_fits(int("1",2), int("0",2), int("0",2))
    assert addr_fits(int("10",2), int("0",2), int("0",2))
    assert addr_fits(int("11",2), int("0",2), int("0",2))
    # assert       is_addr_valid(int("0",2), int("1",2))
        
    #1bit
    assert         is_addr_valid(int("1",2), int("0",2))
    assert addr_fits(int("0",2), int("1",2), int("0",2))
    #assert addr_fits(int("1",2), int("1",2), int("0",2))
    assert         is_addr_valid(int("1",2), int("1",2))
    #assert addr_fits(int("0",2), int("1",2), int("1",2))
    assert addr_fits(int("1",2), int("1",2), int("1",2))
    
    assert          is_addr_valid(int("1",2), int("0",2))
    assert addr_fits(int("10",2), int("1",2), int("0",2))
    #assert addr_fits(int("11",2), int("1",2), int("0",2))
    assert          is_addr_valid(int("1",2), int("1",2))
    #assert addr_fits(int("10",2), int("1",2), int("1",2))
    assert addr_fits(int("11",2), int("1",2), int("1",2))
        
    #2bit
    assert           is_addr_valid(int("10",2), int("0",2))
    assert   addr_fits(int("0",2), int("10",2), int("0",2))
    assert   addr_fits(int("1",2), int("10",2), int("0",2))
    #assert  addr_fits(int("10",2), int("10",2), int("0",2))
    #assert  addr_fits(int("11",2), int("10",2), int("0",2))
    assert addr_fits(int("100",2), int("10",2), int("0",2))
    assert addr_fits(int("101",2), int("10",2), int("0",2))
    #assert addr_fits(int("110",2), int("10",2), int("0",2))
    #assert addr_fits(int("111",2), int("10",2), int("0",2))
    #assert          is_addr_valid(int("10",2), int("1",2))
    assert          is_addr_valid(int("10",2), int("10",2))
    #assert  addr_fits(int("0",2), int("10",2), int("10",2))
    #assert  addr_fits(int("1",2), int("10",2), int("10",2))
    assert addr_fits(int("10",2), int("10",2), int("10",2))
    assert addr_fits(int("11",2), int("10",2), int("10",2))
    assert          is_addr_valid(int("11",2), int("10",2))
    #assert  addr_fits(int("0",2), int("11",2), int("10",2))
    #assert  addr_fits(int("1",2), int("11",2), int("10",2))
    assert addr_fits(int("10",2), int("11",2), int("10",2))
    #assert addr_fits(int("11",2), int("11",2), int("10",2))
    assert          is_addr_valid(int("11",2), int("11",2))
    #assert  addr_fits(int("0",2), int("11",2), int("11",2))
    #assert  addr_fits(int("1",2), int("11",2), int("11",2))
    #assert addr_fits(int("10",2), int("11",2), int("11",2))
    assert addr_fits(int("11",2), int("11",2), int("11",2))
    assert          is_addr_valid(int("11",2), int("1",2))
    #assert  addr_fits(int("0",2), int("11",2), int("1",2))
    assert  addr_fits(int("1",2), int("11",2), int("1",2))
    #assert addr_fits(int("10",2), int("11",2), int("1",2))
    #assert addr_fits(int("11",2), int("11",2), int("1",2))
    
    pass
def addr_fits_safe(to_test:int, bit_in_use_mask:int, fixed_addr:int)->bool:
    '''Automatically calls is_addr_valid for you.
    
    See func of addr_fits.'''
    assert      is_addr_valid(bit_in_use_mask, fixed_addr)
    return addr_fits(to_test, bit_in_use_mask, fixed_addr)
if "test" and False:
    addr_fits_safe(int("0",2), int("0",2), int("0",2))
    #addr_fits_safe(int("0",2), int("0",2), int("1",2))
    addr_fits_safe(int("0",2), int("1",2), int("0",2))
    addr_fits_safe(int("0",2), int("1",2), int("1",2))
    #addr_fits_safe(int("0",2), int("1",2), int("10",2))
    addr_fits_safe(int("0",2), int("11",2), int("0",2))
    addr_fits_safe(int("0",2), int("11",2), int("1",2))
    addr_fits_safe(int("0",2), int("11",2), int("10",2))
    #addr_fits_safe(int("0",2), int("11",2), int("100",2))
    pass


if "useless" and False:
    #def combine_addr(already_used:int, fixed_addr:int, index:int)
    N=5
    already_used:int = int("1100",2)
    str_already_used = str_as_bin(already_used ,N)
    already_used_len:int = 2
    fixed_addr:int = int("1000",2)
    str_fixed_addr = str_as_bin(fixed_addr,N)
    index:int = int("000",2)#notice the len
    str_index = str_as_bin(index,N-already_used_len)#notice the len
    #most_left = N
    starts:list[int] = []
    ends:list[int] = []
    lengths:list[int] = []
    start = 0
    end = 0
    while(True):
        start = str_already_used.find("0",end)
        if start == N:
            break
        end = str_already_used.find("0",start)
        starts.append(start)
        ends.append(end)
        lengths.append(end-start)
        pass

    #for i in range(starts.__len__()):
    #   准备遍历的时候，发现根本不现实。
    raise Exception()

def Num_of_irrelevant(dataset:dict[int,int], N:int)->int:
    '''Because only relevant items are stored in "dataset", 
    and the total possible number is 1<<N.'''
    length = dataset.__len__()
    total_possible = 1<<N
    result = total_possible-length
    return result
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


# a = [(3,True),(2,False),(1,True)]
# print(a)
# a.sort(key=lambda item:item[0])
# print(a)





















#这个是有用的，要加入到类里面区的。但是现在先屏蔽了。

# def rand_dataset(N:int, p_True:float, p_False:float, seed:int = 0)->dict[int,int]:
#     '''The result looks like addr:value. The addr is naturally sorted???
    
#     For debug purpose.'''
#     dataset:dict[int,int] = []
#     random.seed(seed)
#     p_both = p_True + p_False
#     for i in range(1<<N):
#         r = random.random()
#         if r < p_True:
#             dataset[i] = 1
#         elif r < p_both:
#             dataset[i] = 0
#         else:
#             # irrelevant items are not in dataset
#             pass
#         pass
#     return dataset
# if "test" and False:
#     ds1 = rand_dataset(2, 0.4, 0.4, 123)
#     ds2 = rand_dataset(2, 0.4, 0.4, 123)
#     ds3 = rand_dataset(3, 0.3, 0.3, 123)
#     ds4 = rand_dataset(3, 0.6, 0. , 123)
#     pass
# to do optimization: sort the dataset 


class DatasetField:
    N:int
    dataset:list[tuple[int,bool]]
    addr:int
    bitmask:int
    bits_already_in_use:int
    is_dataset_sorted:bool
    def __init__(self, N:int, dataset:list[tuple[int,bool]], addr:int, 
                bitmask:int,bits_already_in_use:int, is_dataset_sorted:bool,):
        #safety check
        is_addr_valid(bit_in_use_mask=bitmask, fixed_addr=addr)
        self.N = N
        self.dataset = dataset
        self.addr = addr
        self.bitmask = bitmask
        self.bits_already_in_use = bits_already_in_use
        self.is_dataset_sorted = is_dataset_sorted
        pass
    def __sort_dataset(self):
        self.dataset.sort(key=lambda item:item[0])
        self.is_dataset_sorted = True
        pass
    def lookup(self, addr:int)->tuple[bool,bool]:
        assert False, "untested"
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
    def detect_end(self):
        assert False, "unfinished"
        # 还没想好，反正3个条件。
        # 条件1，只有1位了
        # 条件2，完全没有1或者0了，那么输出常数就够了。特殊情况是，只有无关项，看看怎么处理。。
        # 条件3，全部都是0.5的相关性，没有无关项，于是就是一个奇偶校验了（全xor）
        #if 1 == self.N-self.bits_already_in_use:
        
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
        
        
        
        
#但是感觉这个好像用不到了？？？？
def is_splittable(bit_in_use_mask:int, N:int)->bool:
    assert False, "untested，写了测试在后面。"
    to_str = str_as_bin(bit_in_use_mask)
    if to_str.__len__()<=N-2:
        return True
    if to_str.__len__() == N-1:
        add1:int = bit_in_use_mask+1
        bitwise_and = add1&bit_in_use_mask
        result:bool = 0 != bitwise_and
        return result
    #now, it's full length. If at least 2 0s, true.
    pos = to_str.find("0")    
    if -1 == pos:
        return False
    pos2 = to_str.find("0",pos)   
    if -1 == pos2:
        return False 
    return True
if "untested" and False:
#    继续。
    assert is_splittable(0b0, 1)
    assert is_splittable(0b1, 1)
    assert is_splittable(0b0, 2)
    assert is_splittable(0b1, 2)
    assert is_splittable(0b10, 2)
    assert is_splittable(0b11, 2)
    assert is_splittable(0b1, 3)
    assert is_splittable(0b11, 3)
    assert is_splittable(0b111, 3)
    pass
    

#def split_addr()
#bit_in_use_mask:int = 0b10, fixed_addr:int





#############
#############
#############
#############

dataset = {0: 1, 1: 1, 2: 0, 3: 1}

N_here = 6
bit_in_use_mask = int("001000",2)
already_determined_len = 1
addr_in = int("001000",2)
addr_here = int("000100",2)
if (bit_in_use_mask&addr_here)!=0:
    raise Exception("addr_here is already used in old addr_in")
already_determined_here = bit_in_use_mask|addr_here
already_determined_len_here = already_determined_len+1
c = ~addr_here
d = addr_in|addr_here
ds = str_as_bin(d)
e = addr_in&c
es = str_as_bin(e)
#for i in range(1<<(N_here-already_determined_len_here)):
    #有点麻烦。

raise Exception()


#now dataset is ready.
#for i in 
addr_F_from_previous = (1<<N)-1
addr_T_from_previous = 0

bit=0
#now make False half addr
addr_F = (1<<N)-1
addr_F = addr_F&addr_F_from_previous
(1<<N)>>bit

#~ 

addr_T = 0
#|
addr = 1<<N-1
#set_addr = 


raise Exception()

#dataset:dict[int,int] = {:,}


raise Exception()







