from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _bool_equal___0_as_false
#         iota, str_the_list
#from pytorch_yagaodirac_v2.Random import rand_sign
import random
import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######














def pos_of_first_one(input:int)->int:
    '''0b10000 into 5'''
    assert input>=0
    result = 0
    while input>0:
        result = result +1
        #tail
        input = input >>1
        pass
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____pos_of_first_one():
        assert pos_of_first_one(0b0) == 0
        assert pos_of_first_one(0b1) == 1
        assert pos_of_first_one(0b10) == 2
        assert pos_of_first_one(0b11) == 2
        assert pos_of_first_one(0b100) == 3
        assert pos_of_first_one(0b10000) == 5
        assert pos_of_first_one(0b10010) == 5
        assert pos_of_first_one(0b10110) == 5

        import random
        for _ in range(5):
            the_number = random.randint(0b100, 0b111)
            assert pos_of_first_one(the_number) == 3
            pass

        for _ in range(11):
            the_number = random.randint(0b1000, 0b1111)
            assert pos_of_first_one(the_number) == 4
            pass

        for nn in [0,1,2,3,5,7,10,15,22]:
            for _ in range(nn*10+1):
                the_number = random.randint((1<<nn), (1<<(nn+1))-1)
                assert pos_of_first_one(the_number) == nn +1
                pass
            pass

        return
    ____test____pos_of_first_one()
    pass

def how_many_ones(input:int)->int:
    '''0b10000 into 1'''
    assert input>=0
    result = 0
    while input>0:
        if input& 0b1 != 0:
            result = result +1
            pass
        #tail
        input = input >>1
        pass
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____how_many_ones():
        assert how_many_ones(0b0) == 0
        assert how_many_ones(0b1) == 1
        assert how_many_ones(0b10) == 1
        assert how_many_ones(0b11) == 2
        assert how_many_ones(0b100) == 1
        assert how_many_ones(0b10000) == 1
        assert how_many_ones(0b10010) == 2
        assert how_many_ones(0b10110) == 3


        MAX_of_something = 5
        for answer in [0,1,2,3,5,7,10,15,22]:
            for _ in range(answer*10+5):
                test_this = 0
                some_pos = -1
                for _ in range(answer):
                    some_pos+= random.randint(1,MAX_of_something)
                    test_this |= (1<<some_pos)
                    assert type(test_this) == int
                    assert isinstance(test_this, int)
                    pass
                assert how_many_ones(test_this) == answer
                pass
            pass

        return
    ____test____how_many_ones()
    pass

def readable_binary(input:int, pad_to_length = 0)->str:
    raw_str = f"{input:b}"
    needs_zeros = pad_to_length - raw_str.__len__()
    if needs_zeros>0:
        return "0"*needs_zeros+raw_str
    else:
        return raw_str
    pass#end of function
if "test" and __DEBUG_ME__() and True:
    def ____test____readable_binary():
        assert readable_binary(0b1) == "1"
        assert readable_binary(0b0) == "0"
        #assert readable_binary(0b0, 3) == "000"
        assert readable_binary(0b10) == "10"
        assert readable_binary(0b1, 3) == "001"
        assert readable_binary(0b0, 3) == "000"
        assert readable_binary(0b10, 3) == "010"
        assert readable_binary(0b10, 5) == "00010"

        for _ in range(55):
            the_number = random.randint(0,255)
            pad_to = random.randint(0,10)
            the_str = readable_binary(input = the_number, pad_to_length=pad_to)
            assert int(the_str, 2) == the_number
            pass

        for _ in range(355):
            the_number = random.randint(1<<6, 1<<12)
            pad_to = random.randint(0,22)
            the_str = readable_binary(input = the_number, pad_to_length=pad_to)
            assert int(the_str, 2) == the_number
            pass

        return 
    ____test____readable_binary()
    pass






#still useful, but not used in the class.
def assert_all_unused_bits_in_addr_are_zeros(bits_in_use_mask:int, fixed_addr:int)->bool:
    '''Checks if fixed_addr has no 1s outside the bits marked by bit_in_use_mask
    
    Mainly for debug.'''
    bits_NOT_in_use_mask = ~bits_in_use_mask
    bitwise_and = fixed_addr&bits_NOT_in_use_mask
    result = 0 == bitwise_and
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____assert_all_unused_bits_in_addr_are_zeros():
        assert type(assert_all_unused_bits_in_addr_are_zeros(int("0",2), int("0",2))) == bool
        assert assert_all_unused_bits_in_addr_are_zeros(int("0" ,2), int("0" ,2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros(int("0" ,2), int("1" ,2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros(int("1" ,2), int("0" ,2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros(int("1" ,2), int("1" ,2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros(int("1" ,2), int("10",2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros(int("10",2), int("10",2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros(int("10",2), int("11",2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros(int("10",2), int("1" ,2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros(int("10",2), int("0" ,2)) == True

        for LENGTH in [2,4,8,16,50]:
            for _ in range(LENGTH*5+20):
                # init
                bits_in_use_mask__list = ["0"]*LENGTH
                fixed_addr__list       = ["0"]*LENGTH
                # random it a bit.
                for __ in range(random.randint(LENGTH//5, LENGTH*2)):
                    rand_pos = random.randint(0,LENGTH-1)
                    bits_in_use_mask__list[rand_pos] = "1"
                    if random.random()<0.5:
                        fixed_addr__list[rand_pos] = "1"
                        pass
                    else:
                        fixed_addr__list[rand_pos] = "0"
                        pass
                # convert
                bits_in_use_mask__str = "".join(bits_in_use_mask__list)
                fixed_addr__str       = "".join(fixed_addr__list)
                bits_in_use_mask__number = int(bits_in_use_mask__str, 2)
                fixed_addr__number       = int(fixed_addr__str, 2)
                # assert
                assert assert_all_unused_bits_in_addr_are_zeros( \
                        bits_in_use_mask=bits_in_use_mask__number, fixed_addr=fixed_addr__number) == True
                pass#for _
            pass#for LENGTH

        return
    ____test____assert_all_unused_bits_in_addr_are_zeros()
    pass






















def int_into_bool_tensor(input:int, length:int)->torch.Tensor:
    format_str = "{:0"+str(length)+"b}"
    bits = format_str.format(input)
    raw_result_list = [bit =="1" for bit in bits]
    result = torch.tensor(raw_result_list)
    return result
    # old code below
    #bits = list(f"{input:b}")
    #raw_result_list = [bit =="1" for bit in bits]
    # result = torch.zeros(size=[length], dtype=torch.bool)
    # result[length-raw_result_list.__len__():] = torch.tensor(raw_result_list, dtype=torch.bool)
    # return result
if "test" and __DEBUG_ME__() and True:
    def ____test____int_into_bool_tensor():
        aaa = int_into_bool_tensor(input=0b1, length=3)
        assert aaa.dtype == torch.bool
        assert _bool_equal___0_as_false(aaa, [0,0,1])
        aaa = int_into_bool_tensor(input=0b11, length=3)
        assert _bool_equal___0_as_false(aaa, [0,1,1])
        aaa = int_into_bool_tensor(input=0b101, length=3)
        assert _bool_equal___0_as_false(aaa, [1,0,1])
        aaa = int_into_bool_tensor(input=0b11, length=5)
        assert _bool_equal___0_as_false(aaa, [0,0,0,1,1])
        aaa = int_into_bool_tensor(input=0b101, length=5)
        assert _bool_equal___0_as_false(aaa, [0,0,1,0,1])
        aaa = int_into_bool_tensor(input=0b1010, length=5)
        assert _bool_equal___0_as_false(aaa, [0,1,0,1,0])

        return 
    ____test____int_into_bool_tensor()
    pass


def int_list_into_bool_tensor(input_list:list[int], length:int)->torch.Tensor:
    format_str = "{:0"+str(length)+"b}"

    bits_list = []
    for input in input_list:
        bits_list.append(list(format_str.format(input)))
        pass

    raw_result_list = []
    for bits in bits_list:
        raw_result_list.append([bit =="1" for bit in bits])
        pass

    result = torch.tensor(raw_result_list)
    return result
if "test" and __DEBUG_ME__() and True:
    def ____test____int_list_into_bool_tensor():

        aaa = int_list_into_bool_tensor(input_list=[0b1, 0b11], length=3)
        assert aaa.dtype == torch.bool
        assert _bool_equal___0_as_false(aaa, [[0,0,1],[0,1,1]])

        aaa = int_list_into_bool_tensor(input_list=[0b11,0b101,0b1010], length=5)
        assert _bool_equal___0_as_false(aaa, [  [0,0,0,1,1],
                                                [0,0,1,0,1],
                                                [0,1,0,1,0]])
        
        return
    ____test____int_list_into_bool_tensor()
    pass




1w
1w
1w
1w然后把前面的工具改到torch上。








