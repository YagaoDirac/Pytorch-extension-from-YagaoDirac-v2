# ok, cool. I didn't write py for 8 mon.
# a = int( "1000", 2)
# b = 0b111
# s = "{:b}".format(a)
# ss = s[3]

from typing import Optional
import random

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

def rand_dataset(N:int, p_True:float, p_False:float, seed:int = 0)->dict[int,int]:
    '''The result looks like addr:value. The addr is naturally sorted???
    
    For debug purpose.'''
    dataset:dict[int,int] = {}
    random.seed(seed)
    p_both = p_True + p_False
    for i in range(1<<N):
        r = random.random()
        if r < p_True:
            dataset[i] = 1
        elif r < p_both:
            dataset[i] = 0
        else:
            # irrelevant items are not in dataset
            pass
        pass
    return dataset
if "test" and False:
    ds1 = rand_dataset(2, 0.4, 0.4, 123)
    ds2 = rand_dataset(2, 0.4, 0.4, 123)
    ds3 = rand_dataset(3, 0.3, 0.3, 123)
    ds4 = rand_dataset(3, 0.6, 0. , 123)
    pass
# to do optimization: sort the dataset 

def is_splittable(bit_in_use_mask:int, N:int)->bool:
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
if "test" and True:
    继续。
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
bit_in_use_mask:int = 0b10, fixed_addr:int





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






















from bitsets import bitset

# Define a pool of objects
PYTHONS = ('Chapman', 'Cleese', 'Gilliam', 'Idle', 'Jones', 'Palin')

# Create a bitset class for the pool
Pythons = bitset('Pythons', PYTHONS)

# Create instances of the bitset
all_pythons = Pythons.supremum # All members
no_pythons = Pythons.infimum # No members

# Access members
print(all_pythons.members()) # ('Chapman', 'Cleese', 'Gilliam', 'Idle', 'Jones', 'Palin')
print(no_pythons.members()) # ()

# Perform set operations
subset = Pythons(['Chapman', 'Gilliam'])
print(subset.bits()) # '101000'
print(Pythons.frombits('101000').members()) # ('Chapman', 'Gilliam')