from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _bool_equal___0_as_false, all_False, _tensor_equal, _tensor_shape_check
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













'tools   bool list in int'
'tools   bool list in int'
'tools   bool list in int'
def pos_of_first_one__int(input:int)->int:
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
    def ____test____pos_of_first_one__int():
        assert pos_of_first_one__int(0b0) == 0
        assert pos_of_first_one__int(0b1) == 1
        assert pos_of_first_one__int(0b10) == 2
        assert pos_of_first_one__int(0b11) == 2
        assert pos_of_first_one__int(0b100) == 3
        assert pos_of_first_one__int(0b10000) == 5
        assert pos_of_first_one__int(0b10010) == 5
        assert pos_of_first_one__int(0b10110) == 5
        assert pos_of_first_one__int(0b10101000001010100000101010000010101000001010100000101010000010101000001010100000) == 80

        for _ in range(5):
            the_number = random.randint(0b100, 0b111)
            assert pos_of_first_one__int(the_number) == 3
            pass

        for _ in range(11):
            the_number = random.randint(0b1000, 0b1111)
            assert pos_of_first_one__int(the_number) == 4
            pass

        for nn in [0,1,2,3,5,7,10,15,22]:
            for _ in range(nn*10+1):
                the_number = random.randint((1<<nn), (1<<(nn+1))-1)
                assert pos_of_first_one__int(the_number) == nn +1
                pass
            pass

        return
    ____test____pos_of_first_one__int()
    pass

def how_many_ones__int(input:int)->int:
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
    def ____test____how_many_ones__int():
        assert how_many_ones__int(0b0) == 0
        assert how_many_ones__int(0b1) == 1
        assert how_many_ones__int(0b10) == 1
        assert how_many_ones__int(0b11) == 2
        assert how_many_ones__int(0b100) == 1
        assert how_many_ones__int(0b10000) == 1
        assert how_many_ones__int(0b10010) == 2
        assert how_many_ones__int(0b10110) == 3
        assert how_many_ones__int(0b10101000001010100000101010000010101000001010100000101010000010101000001010100000) == 24

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
                assert how_many_ones__int(test_this) == answer
                pass
            pass

        return
    ____test____how_many_ones__int()
    pass

def int_into_str_as_if_its_binary(input:int, pad_to_length = 0)->str:
    assert type(input) == int
    raw_str = bin(input)[2:]
    needs_zeros = pad_to_length - raw_str.__len__()
    if needs_zeros>0:
        return "0"*needs_zeros+raw_str
    else:
        return raw_str
    pass#end of function
if "test" and __DEBUG_ME__() and False:
    def ____test____int_into_str_as_if_its_binary():
        assert int_into_str_as_if_its_binary(0b1) == "1"
        assert int_into_str_as_if_its_binary(0b0) == "0"
        #assert int_into_str_as_if_its_binary(0b0, 3) == "000"
        assert int_into_str_as_if_its_binary(0b10) == "10"
        assert int_into_str_as_if_its_binary(0b1, 3) == "001"
        assert int_into_str_as_if_its_binary(0b0, 3) == "000"
        assert int_into_str_as_if_its_binary(0b10, 3) == "010"
        assert int_into_str_as_if_its_binary(0b10, 5) == "00010"
        assert int_into_str_as_if_its_binary(0b10101000001010100000101010000010101000001010100000101010000010101000001010100000, 80) ==  \
                "10101000001010100000101010000010101000001010100000101010000010101000001010100000"
        assert int_into_str_as_if_its_binary(0b10101000001010100000101010000010101000001010100000101010000010101000001010100000, 90) ==  \
                "000000000010101000001010100000101010000010101000001010100000101010000010101000001010100000"

        for _ in range(55):
            the_number = random.randint(0,255)
            pad_to = random.randint(0,10)
            the_str = int_into_str_as_if_its_binary(input = the_number, pad_to_length=pad_to)
            assert int(the_str, 2) == the_number
            pass

        for _ in range(355):
            the_number = random.randint(1<<6, 1<<12)
            pad_to = random.randint(0,22)
            the_str = int_into_str_as_if_its_binary(input = the_number, pad_to_length=pad_to)
            assert int(the_str, 2) == the_number
            pass

        return 
    ____test____int_into_str_as_if_its_binary()
    pass

#still useful, but not used in the class.
def assert_all_unused_bits_in_addr_are_zeros__int(bits_in_use_mask:int, fixed_addr:int)->bool:
    '''Checks if fixed_addr has no 1s outside the bits marked by bit_in_use_mask
    
    Mainly for debug.'''
    bits_NOT_in_use_mask = ~bits_in_use_mask
    bitwise_and = fixed_addr&bits_NOT_in_use_mask
    result = 0 == bitwise_and
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____assert_all_unused_bits_in_addr_are_zeros__int():
        assert type(assert_all_unused_bits_in_addr_are_zeros__int(int("0",2), int("0",2))) == bool
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("0" ,2), int("0" ,2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("0" ,2), int("1" ,2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("1" ,2), int("0" ,2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("1" ,2), int("1" ,2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("1" ,2), int("10",2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("10",2), int("10",2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("10",2), int("11",2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("10",2), int("1" ,2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros__int(int("10",2), int("0" ,2)) == True


        assert assert_all_unused_bits_in_addr_are_zeros__int(   0b10101000001010100000101010000010101000001010100000101010000010101000001010100000, \
                                                                0b10101000001010100000101010000010101000001010100000101010000010101000001010100000) == True
        assert assert_all_unused_bits_in_addr_are_zeros__int(   0b10101011101010100000101010111110101000001010100100101010110010101000001010100000, \
                                                                0b10101000001010100000101010000010101000001010100000101010000010101000001010100000) == True
        

        for LENGTH in [2,4,8,16,50]:
            for _ in range(LENGTH*5+20):
                # init
                bits_in_use_mask__list = ["0"]*LENGTH
                fixed_addr__list       = ["0"]*LENGTH
                # random it a bit.
                # the mask can only init into 1 if touched. But the addr can convert into both 0 or 1. 
                # but the untouched bits are always 0 for both var.
                for __ in range(random.randint(LENGTH//5, LENGTH*2)):
                    rand_pos = random.randint(0,LENGTH-1)
                    bits_in_use_mask__list[rand_pos] = "1"
                    if random.random()<0.5:
                        fixed_addr__list[rand_pos] = "1"
                        pass
                    else:
                        fixed_addr__list[rand_pos] = "0"
                        pass
                # convert      list into str into int
                bits_in_use_mask__str = "".join(bits_in_use_mask__list)
                fixed_addr__str       = "".join(fixed_addr__list)
                bits_in_use_mask__number = int(bits_in_use_mask__str, 2)
                fixed_addr__number       = int(fixed_addr__str, 2)
                # assert
                assert assert_all_unused_bits_in_addr_are_zeros__int( \
                        bits_in_use_mask=bits_in_use_mask__number, fixed_addr=fixed_addr__number) == True
                pass#for _
            pass#for LENGTH

        return
    ____test____assert_all_unused_bits_in_addr_are_zeros__int()
    pass














'bool list in int   convert into   tensor.'
'bool list in int   convert into   tensor.'
'bool list in int   convert into   tensor.'
"param:length doesn't affect the result. This detail is going into the function docstring."

def int_into_bool_tensor(input:int, length:int)->torch.Tensor:
    raw_str = bin(input)[2:]
    bits = raw_str.rjust(length, "0")
    assert bits.__len__() == length, f"{bin(input)} is provided to get converted, but it needs longer length than the provided param:length, which is ({length}\
        ). Or if the param:length is 0, this function doesn't work."
    raw_result_list = [bit =="1" for bit in bits]
    result = torch.tensor(raw_result_list)
    return result
if "test" and __DEBUG_ME__() and False:
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

        # int_into_bool_tensor(input=0b11, length=1)#should not work
        # int_into_bool_tensor(input=0b101, length=1)#should not work
        # int_into_bool_tensor(input=0b101, length=2)#should not work

        aaa = int_into_bool_tensor(input=0b10101000001010100000101010000010101000001010100000101010000010101000001010100000, length=80)
        assert _bool_equal___0_as_false(aaa, [  
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                                                ])
        
        aaa = int_into_bool_tensor(input=0b10101000001010100000101010000010101000001010100000101010000010101000001010100000, length=90)
        assert _bool_equal___0_as_false(aaa, [  
                                                    0,0,0,0,0, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                    1,0,1,0,1, 0,0,0,0,0, 
                                                                                ])

        return 
    ____test____int_into_bool_tensor()
    pass

def int_list_into_bool_tensor(input_list:list[int], length:int)->torch.Tensor:
    temp_list_of_bits_list = []
    for input in input_list:
        _temp___raw_str = bin(input)[2:]
        bits = _temp___raw_str.rjust(length, "0")
        assert bits.__len__() == length, f"{bin(input)} is provided to get converted, but it needs longer length than the provided param:length, which is ({length})"
        temp_list_of_bits_list.append(list(bits))
        pass

    raw_result_list = []
    for list_of_bits in temp_list_of_bits_list:
        raw_result_list.append([bit =="1" for bit in list_of_bits])
        pass

    result = torch.tensor(raw_result_list)
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____int_list_into_bool_tensor():

        aaa = int_list_into_bool_tensor(input_list=[0b1, 0b11], length=3)
        assert aaa.dtype == torch.bool
        assert _bool_equal___0_as_false(aaa, [[0,0,1],[0,1,1]])

        aaa = int_list_into_bool_tensor(input_list=[0b11,0b101,0b1010], length=5)
        assert _bool_equal___0_as_false(aaa, [  [0,0,0,1,1],
                                                [0,0,1,0,1],
                                                [0,1,0,1,0]])

        # int_list_into_bool_tensor(input_list=[0b1, 0b11], length=1)#should not work
        # int_list_into_bool_tensor(input_list=[0b1, 0b11, 0b101], length=1)#should not work
        # int_list_into_bool_tensor(input_list=[0b1, 0b11, 0b101], length=2)#should not work



        if "bit int" and True:
            aaa = int_list_into_bool_tensor(input_list=[0b10101000001010100000101010000010101000001010100000101010000010101000001010100000], length=80)
            assert aaa.shape.__len__() == 2
            assert aaa.shape[0] == 1
            assert _bool_equal___0_as_false(aaa[0], [  
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                                                    ])
            
            aaa = int_list_into_bool_tensor(input_list=[0b10101000001010100000101010000010101000001010100000101010000010101000001010100000], length=90)
            assert aaa.shape.__len__() == 2
            assert aaa.shape[0] == 1
            assert _bool_equal___0_as_false(aaa[0], [  
                                                        0,0,0,0,0, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                                                    ])

            aaa = int_list_into_bool_tensor(input_list=[0b10101000001010100000101010000010101000001010100000101010000010101000001010100000, 
                                                        0b10101011101010101110101010111010101011101010101110101010111010101011101010101110], length=80)
            assert aaa.shape.__len__() == 2
            assert aaa.shape[0] == 2
            assert _bool_equal___0_as_false(aaa, [  [
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                        1,0,1,0,1, 0,0,0,0,0, 
                                                    ],[
                                                        1,0,1,0,1, 0,1,1,1,0, 
                                                        1,0,1,0,1, 0,1,1,1,0, 
                                                        1,0,1,0,1, 0,1,1,1,0, 
                                                        1,0,1,0,1, 0,1,1,1,0, 
                                                        1,0,1,0,1, 0,1,1,1,0, 
                                                        1,0,1,0,1, 0,1,1,1,0, 
                                                        1,0,1,0,1, 0,1,1,1,0, 
                                                        1,0,1,0,1, 0,1,1,1,0, 
                                                                                                    ],  ])
            pass#/test

        return
    ____test____int_list_into_bool_tensor()
    pass














'tools   bool list in tensor'
'tools   bool list in tensor'
'tools   bool list in tensor'
def pos_of_first_one(input:torch.Tensor)->torch.Tensor:
    '''tensor([T,F,F,F,F]) into 5'''
    assert input.dtype == torch.bool, "if the input is int, call pos_of_first_one__int"
    assert input.shape.__len__() == 1
    are_all_False = input.any().logical_not()
    if are_all_False:
        return torch.tensor(0)
    
    the_argmax = input.to(torch.int8).argmax()
    result = input.shape[0]-the_argmax
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____pos_of_first_one():

        result = pos_of_first_one(int_into_bool_tensor(0b1, length = 3))
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.int64

        assert pos_of_first_one(int_into_bool_tensor(0b0, length = 1)) == 0
        assert pos_of_first_one(int_into_bool_tensor(0b0, length = 2)) == 0
        assert pos_of_first_one(int_into_bool_tensor(0b0, length = 3)) == 0
        assert pos_of_first_one(int_into_bool_tensor(0b1, length = 1)) == 1
        assert pos_of_first_one(int_into_bool_tensor(0b1, length = 2)) == 1
        assert pos_of_first_one(int_into_bool_tensor(0b1, length = 3)) == 1
        #assert pos_of_first_one(int_into_bool_tensor(0b10, length = 1)) == 2#length is too small
        assert pos_of_first_one(int_into_bool_tensor(0b10, length = 2)) == 2
        assert pos_of_first_one(int_into_bool_tensor(0b10, length = 3)) == 2
        assert pos_of_first_one(int_into_bool_tensor(0b10, length = 4)) == 2
        assert pos_of_first_one(int_into_bool_tensor(0b11, length = 2)) == 2
        assert pos_of_first_one(int_into_bool_tensor(0b11, length = 4)) == 2
        assert pos_of_first_one(int_into_bool_tensor(0b100, length = 3)) == 3
        assert pos_of_first_one(int_into_bool_tensor(0b100, length = 10)) == 3
        assert pos_of_first_one(int_into_bool_tensor(0b10000, length = 5))== 5
        assert pos_of_first_one(int_into_bool_tensor(0b10000, length = 10)) == 5
        assert pos_of_first_one(int_into_bool_tensor(0b10010, length = 10)) == 5
        assert pos_of_first_one(int_into_bool_tensor(0b10110, length = 10)) == 5

        for _ in range(5):
            the_number = random.randint(0b100, 0b111)
            assert pos_of_first_one(int_into_bool_tensor(the_number, length=random.randint(3,13))) == 3
            pass

        for _ in range(11):
            the_number = random.randint(0b1000, 0b1111)
            assert pos_of_first_one(int_into_bool_tensor(the_number, length=random.randint(4,13))) == 4
            pass

        for length_minus_1 in [0,1,2,3,5,7,10,15,22]:
            for _ in range(length_minus_1*10+1):
                the_number = random.randint((1<<length_minus_1), (1<<(length_minus_1+1))-1)
                the_number_in_tensor = int_into_bool_tensor(the_number, length=random.randint(length_minus_1+1, length_minus_1+7))
                assert pos_of_first_one(the_number_in_tensor) == length_minus_1+1
                pass
            pass

        return
    ____test____pos_of_first_one()
    pass

def how_many_ones(input:torch.Tensor)->torch.Tensor:
    '''tensor([T,F,F,F,F]) into 1'''
    assert input.dtype == torch.bool
    assert input.shape.__len__() == 1
    if input.shape[0]> 1<<30:
        result = input.to(torch.int64).sum()
        pass
    else:
        result = input.to(torch.int32).sum()
        pass
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____how_many_ones():
        assert how_many_ones(int_into_bool_tensor(0b0, length=1)) == 0
        assert how_many_ones(int_into_bool_tensor(0b0, length=3)) == 0
        assert how_many_ones(int_into_bool_tensor(0b1, length=1)) == 1
        assert how_many_ones(int_into_bool_tensor(0b1, length=3)) == 1
        assert how_many_ones(int_into_bool_tensor(0b10, length=10)) == 1
        assert how_many_ones(int_into_bool_tensor(0b11, length=10)) == 2
        assert how_many_ones(int_into_bool_tensor(0b100, length=10)) == 1
        assert how_many_ones(int_into_bool_tensor(0b10000, length=10)) == 1
        assert how_many_ones(int_into_bool_tensor(0b10010, length=10)) == 2
        assert how_many_ones(int_into_bool_tensor(0b10110, length=10)) == 3

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
                assert how_many_ones(int_into_bool_tensor(test_this, length=answer*6+11)) == answer
                pass
            pass

        return
    ____test____how_many_ones()
    pass

def str_1d_bool_tensor_as_if_its_binary(input:torch.Tensor, pad_to_length = 0)->str:
    assert input.shape.__len__() == 1
    assert isinstance(input, torch.Tensor)
    bool_py_list = input.tolist()

    char_py_list = []
    for the_bool in bool_py_list:
        if the_bool:
            char_py_list.append("1")
            pass
        else:
            char_py_list.append("0")
            pass
        pass#for the_bool
    raw_str = "".join(char_py_list)

    needs_zeros = pad_to_length - raw_str.__len__()
    if needs_zeros>0:
        return "0"*needs_zeros+raw_str
    else:
        return raw_str
    pass#end of function
if "test" and __DEBUG_ME__() and False:
    def ____test____str_1d_bool_tensor_as_if_its_binary():
        assert str_1d_bool_tensor_as_if_its_binary(int_into_bool_tensor(0b1, length=1)) == "1"
        assert str_1d_bool_tensor_as_if_its_binary(int_into_bool_tensor(0b0, length=1)) == "0"
        #assert str_as_binary(0b0, 3) == "000"
        assert str_1d_bool_tensor_as_if_its_binary(int_into_bool_tensor(0b10, length=2)) == "10"
        assert str_1d_bool_tensor_as_if_its_binary(int_into_bool_tensor(0b1, length=1), 3) == "001"
        assert str_1d_bool_tensor_as_if_its_binary(int_into_bool_tensor(0b0, length=1), 3) == "000"
        assert str_1d_bool_tensor_as_if_its_binary(int_into_bool_tensor(0b10, length=2), 3) == "010"
        assert str_1d_bool_tensor_as_if_its_binary(int_into_bool_tensor(0b10, length=2), 5) == "00010"

        for _ in range(55):
            the_number = random.randint(0,255)# 0 to 8 bits
            length = random.randint(8, 22)
            pad_to = random.randint(0, 28)

            the_str = str_1d_bool_tensor_as_if_its_binary(input = int_into_bool_tensor(the_number, length=length), pad_to_length=pad_to)
            assert int(the_str, 2) == the_number
            pass

        for _ in range(355):
            the_number = random.randint(1<<6, 1<<12-1)# 6 to 12 bits
            length = random.randint(12, 22)
            pad_to = random.randint( 0, 33)

            the_str = str_1d_bool_tensor_as_if_its_binary(input = int_into_bool_tensor(the_number, length=length), pad_to_length=pad_to)
            assert int(the_str, 2) == the_number
            pass

        return 
    ____test____str_1d_bool_tensor_as_if_its_binary()
    pass

#still useful, but not used in the class.
def assert_all_unused_bits_in_addr_are_zeros(bits_in_use_mask:torch.Tensor, fixed_addr:torch.Tensor)->torch.Tensor:
    '''Checks if fixed_addr has no 1s outside the bits marked by bit_in_use_mask
    
    Mainly for debug.'''
    assert bits_in_use_mask.dtype == torch.bool
    assert bits_in_use_mask.shape.__len__() == 1
    assert fixed_addr.dtype == torch.bool
    assert fixed_addr.shape.__len__() == 1

    bits_NOT_in_use_mask = bits_in_use_mask.logical_not()
    unmasked_bits = fixed_addr[bits_NOT_in_use_mask]
    result = unmasked_bits.any().logical_not()
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____assert_all_unused_bits_in_addr_are_zeros():
        result = assert_all_unused_bits_in_addr_are_zeros(torch.tensor([False]), torch.tensor([False]))
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.bool


        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b0,1), int_into_bool_tensor(0b0,1)) == True
        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b0,1), int_into_bool_tensor(0b1,1)) == False
        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b1,1), int_into_bool_tensor(0b0,1)) == True
        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b1,1), int_into_bool_tensor(0b1,1)) == True
        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b01,2), int_into_bool_tensor(0b10,2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b10,2), int_into_bool_tensor(0b10,2)) == True
        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b10,2), int_into_bool_tensor(0b11,2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b10,2), int_into_bool_tensor(0b01,2)) == False
        assert assert_all_unused_bits_in_addr_are_zeros( \
            int_into_bool_tensor(0b10,2), int_into_bool_tensor(0b00,2)) == True



        for LENGTH in [2,4,8,16,50]:
            for _ in range(LENGTH*5+20):
                # init
                bits_in_use_mask = torch.zeros(size=[LENGTH], dtype=torch.bool)
                fixed_addr       = torch.zeros(size=[LENGTH], dtype=torch.bool)
                # random it a bit.
                # the mask can only init into 1 if touched. But the addr can convert into both 0 or 1. 
                # but the untouched bits are always 0 for both var.
                for __ in range(random.randint(LENGTH//5, LENGTH*2)):
                    rand_pos = random.randint(0,LENGTH-1)
                    bits_in_use_mask[rand_pos] = True
                    if random.random()<0.5:
                        fixed_addr[rand_pos] = True
                        pass
                    else:
                        fixed_addr[rand_pos] = False
                        pass

                # assert
                assert assert_all_unused_bits_in_addr_are_zeros( \
                        bits_in_use_mask=bits_in_use_mask, fixed_addr=fixed_addr) == True
                pass#for _
            pass#for LENGTH

        return
    ____test____assert_all_unused_bits_in_addr_are_zeros()
    pass










'bool tensor      back into     int'
'bool tensor      back into     int'
'bool tensor      back into     int'
def bool_tensor_into_int(input:torch.Tensor, dtype = torch.int64)->int:
    '''notice, int("") doesn't work. So is input is empty, this function doesn't work.'''
    temp_str = str_1d_bool_tensor_as_if_its_binary(input)
    result = int(temp_str, base=2)# notice, int("") doesn't work. So is input is empty, this function doesn't work.
    return result
    #  ver 1 uppon
    #  ver 2 below      v2 doesn't handle big int.   So let's go back to v1.
    #safety
    # assert input.shape.__len__() == 1
    # the_iota_power_of_2 = torch.vander(x=torch.tensor([2], dtype=dtype), N=input.shape[0], increasing=False) #https://docs.pytorch.org/docs/2.14/generated/torch.vander.html
    # the_iota_power_of_2 = the_iota_power_of_2.reshape([-1])#flatten
    # picked = the_iota_power_of_2[input]
    # result = picked.sum()
    # return result
if "test" and __DEBUG_ME__() and False:
    def ____test____bool_tensor_into_int():

        if "randomly test" and True:
            for length in [1,2,3,6,11,18]:
                for _ in range(length*5+11):
                    answer = random.randint(0, 1<<(length)-1)
                    the_tensor = int_into_bool_tensor(input = answer, length=random.randint(length, length+11))
                    back_into_int = bool_tensor_into_int(the_tensor)
                    assert back_into_int == answer
                    pass#for _ 
                pass#for length
            pass#/test

        if "big int" and True:
            answer = (1<<31) -1
            assert answer == 0b1111111111111111111111111111111
            #                  1234567890123456789012345678901   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<31)
            assert answer == 0b10000000000000000000000000000000
            #                   1234567890123456789012345678901   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<32) -1
            assert answer == 0b11111111111111111111111111111111
            #                  12345678901234567890123456789012   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<32)
            assert answer == 0b100000000000000000000000000000000
            #                   12345678901234567890123456789012   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<33) -1
            assert answer == 0b111111111111111111111111111111111
            #                  123456789012345678901234567890123   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<33)
            assert answer == 0b1000000000000000000000000000000000
            #                   123456789012345678901234567890123   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<63) -1
            assert answer == 0b111111111111111111111111111111111111111111111111111111111111111
            #                  123456789012345678901234567890123456789012345678901234567890123   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<63)
            assert answer == 0b1000000000000000000000000000000000000000000000000000000000000000
            #                   123456789012345678901234567890123456789012345678901234567890123   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<64) -1
            assert answer == 0b1111111111111111111111111111111111111111111111111111111111111111
            #                  1234567890123456789012345678901234567890123456789012345678901234   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<64)
            assert answer == 0b10000000000000000000000000000000000000000000000000000000000000000
            #                   1234567890123456789012345678901234567890123456789012345678901234   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<65) -1
            assert answer == 0b11111111111111111111111111111111111111111111111111111111111111111
            #                  12345678901234567890123456789012345678901234567890123456789012345   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<65)
            assert answer == 0b100000000000000000000000000000000000000000000000000000000000000000
            #                   12345678901234567890123456789012345678901234567890123456789012345   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer
            pass#/test

        return 
    ____test____bool_tensor_into_int()
    pass

def bool_tensor_into_int_list(input:torch.Tensor, dtype = torch.int64)->list[int]:#torch.Tensor this is v2, but it doesn't handle bit int:
    '''notice, int("") doesn't work. So is input is empty, this function doesn't work.'''
    assert input.shape.__len__() == 2
    result_list:list[int] = []
    for the_slice in input:
        temp_str = str_1d_bool_tensor_as_if_its_binary(the_slice)
        result_list.append(int(temp_str, base=2))# notice, int("") doesn't work. So is input is empty, this function doesn't work.
        pass
    if "to fold v2" and False:
        #  ver 2 below
        #safety
        # assert input.shape.__len__() == 2
        # the_iota_power_of_2___1_l = torch.vander(x=torch.tensor([2], dtype=dtype), N=input.shape[1], increasing=False) #https://docs.pytorch.org/docs/2.14/generated/torch.vander.html
        # the_iota_power_of_2___EXPANDb_l = the_iota_power_of_2___1_l.expand_as(input)
        # #assert the_iota_power_of_2___EXPANDb_l.shape == input.shape#debug         111111111111111111111111111111111111111111
        # picked = torch.where(condition=input, input=the_iota_power_of_2___EXPANDb_l, other=torch.zeros(size=(), dtype=dtype))
        # result = picked.sum(dim=1)
        # #assert _tensor_shape_check(result, input.shape[0])#debug         111111111111111111111111111111111111111111
        # return result
        pass
    return result_list
if "test" and __DEBUG_ME__() and False:
    def ____test____bool_tensor_into_int_list():
        for batch in [1,2,3,5,14]:
            for length in [1,2,3,6,11,18]:
                for _ in range((batch+3)*(length+3)+11):

                    answer_list = []
                    for __ in range(batch):
                        answer_list.append(random.randint(0, 1<<(length)-1))
                        pass
                    the_tensor = int_list_into_bool_tensor(input_list = answer_list, length=random.randint(length, length+11))

                    back_into_int_list = bool_tensor_into_int_list(the_tensor)

                    assert back_into_int_list.__len__() == batch
                    assert back_into_int_list.__len__() == answer_list.__len__()
                    for ii in range(back_into_int_list.__len__()):
                        assert back_into_int_list[ii] == answer_list[ii]
                        pass

                    pass#for _ 
                pass#for length
            pass#for batch

        if "big int" and True:
            answer = (1<<31) -1
            assert answer == 0b1111111111111111111111111111111
            #                  1234567890123456789012345678901   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<31)
            assert answer == 0b10000000000000000000000000000000
            #                   1234567890123456789012345678901   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<32) -1
            assert answer == 0b11111111111111111111111111111111
            #                  12345678901234567890123456789012   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<32)
            assert answer == 0b100000000000000000000000000000000
            #                   12345678901234567890123456789012   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<33) -1
            assert answer == 0b111111111111111111111111111111111
            #                  123456789012345678901234567890123   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<33)
            assert answer == 0b1000000000000000000000000000000000
            #                   123456789012345678901234567890123   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<63) -1
            assert answer == 0b111111111111111111111111111111111111111111111111111111111111111
            #                  123456789012345678901234567890123456789012345678901234567890123   count it a bit.
            the_tensor = int_into_bool_tensor(input = answer, length=100)
            back_into_int = bool_tensor_into_int(the_tensor)
            assert back_into_int == answer

            answer = (1<<63)
            assert answer == 0b1000000000000000000000000000000000000000000000000000000000000000
            #                   123456789012345678901234567890123456789012345678901234567890123   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<64) -1
            assert answer == 0b1111111111111111111111111111111111111111111111111111111111111111
            #                  1234567890123456789012345678901234567890123456789012345678901234   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<64)
            assert answer == 0b10000000000000000000000000000000000000000000000000000000000000000
            #                   1234567890123456789012345678901234567890123456789012345678901234   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<65) -1
            assert answer == 0b11111111111111111111111111111111111111111111111111111111111111111
            #                  12345678901234567890123456789012345678901234567890123456789012345   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            answer = (1<<65)
            assert answer == 0b100000000000000000000000000000000000000000000000000000000000000000
            #                   12345678901234567890123456789012345678901234567890123456789012345   count it a bit.
            the_tensor = int_list_into_bool_tensor(input_list = [answer], length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == 1
            assert back_into_int_list[0] == answer

            #  all in one
            answer = [  (1<<31)  , (1<<32)  , (1<<33)  , (1<<63)  , (1<<64)  , (1<<65)  , 
                        (1<<31)-1, (1<<32)-1, (1<<33)-1, (1<<63)-1, (1<<64)-1, (1<<65)-1, ]
            the_tensor = int_list_into_bool_tensor(input_list = answer, length=100)
            back_into_int_list = bool_tensor_into_int_list(the_tensor)
            assert back_into_int_list.__len__() == answer.__len__()
            for ii in range(back_into_int_list.__len__()):
                assert back_into_int_list[ii] == answer[ii]
                pass
            pass#/test

        return 
    ____test____bool_tensor_into_int_list()
    pass










'bool tensor       into       pos neg 1'
'bool tensor       into       pos neg 1'
'bool tensor       into       pos neg 1'

def bool_tensor_into_posneg1(input:torch.Tensor, dtype:torch.dtype)->torch.Tensor:
    if dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
        result = torch.where(condition=input, input=torch.ones(size=[1], dtype=dtype), other=torch.ones(size=[1], dtype=dtype)*-1)
        return result
    if dtype in [torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        result = torch.where(condition=input, input=torch.ones(size=[1], dtype=dtype), other=torch.ones(size=[1], dtype=dtype)*-1.)
        return result
    assert False, "other dtypes are not adapted in this function. No plan for now."
    pass# end of function
if "test" and __DEBUG_ME__() and False:
    def ____test____bool_tensor_into_posneg1():
        if "dtype" and True:
            result = bool_tensor_into_posneg1(int_into_bool_tensor(0b01, length=2), dtype=torch.int32)
            assert result.dtype == torch.int32
            assert _tensor_equal(result, [-1, 1])

            result = bool_tensor_into_posneg1(int_into_bool_tensor(0b01, length=2), dtype=torch.float32)
            assert result.dtype == torch.float32
            assert _tensor_equal(result, [-1., 1.])

            result = bool_tensor_into_posneg1(int_list_into_bool_tensor(input_list=[0b101, 0b1011, 0b1], length=5), dtype=torch.int32)
            assert result.dtype == torch.int32
            assert _tensor_equal(result, [  [-1, -1,  1, -1,  1],
                                            [-1,  1, -1,  1,  1],
                                            [-1, -1, -1, -1,  1],])

            result = bool_tensor_into_posneg1(int_list_into_bool_tensor(input_list=[0b101, 0b1011, 0b1], length=5), dtype=torch.float32)
            assert result.dtype == torch.float32
            assert _tensor_equal(result, [  [-1., -1.,  1., -1.,  1.],
                                            [-1.,  1., -1.,  1.,  1.],
                                            [-1., -1., -1., -1.,  1.],])
            pass# /test

        if "randomly test" and True:
            for batch in [1,2,3,4,5, 11, 26]:
                for length in [1,2,3,4,5, 16, 31]:
                    for _ in range(66):
                        # init
                        input_list = []
                        for __ in range(batch):
                            input_list.append(random.randint(0, (1<<length) -1))
                            pass
                        # how many ones in input_list in total
                        sum_of_ori = 0
                        for the_number in input_list:
                            sum_of_ori += how_many_ones__int(the_number)
                            pass

                        # int as binary, convert into boolean, back to -1 and 1, sum.
                        the_bool_tensor = int_list_into_bool_tensor(input_list=input_list, length=length)
                        assert _tensor_shape_check(the_bool_tensor, batch, length)
                        posneg1_tensor = bool_tensor_into_posneg1(the_bool_tensor, dtype=torch.int32)
                        sum_of_posneg1_tensor = posneg1_tensor.sum()
                        nelement = the_bool_tensor.nelement()
                        #assert 
                        assert sum_of_ori >= 0
                        assert sum_of_ori == nelement -   (nelement - sum_of_posneg1_tensor) /2
                        #                               ^^^^^ this part is number of falses ^^^^^
                        pass#for _ 
                    pass#for length 
                pass#for batch 

            for length in [1,2,3,4,5, 16, 31]:
                for _ in range(66):
                    # init
                    input = random.randint(0, (1<<length) -1)
                    # how many ones in input in total
                    sum_of_ori = how_many_ones__int(input)
                    # int as binary, convert into boolean, back to -1 and 1, sum.
                    the_bool_tensor = int_into_bool_tensor(input=input, length=length)
                    assert _tensor_shape_check(the_bool_tensor, length)
                    posneg1_tensor = bool_tensor_into_posneg1(the_bool_tensor, dtype=torch.int32)
                    sum_of_posneg1_tensor = posneg1_tensor.sum()
                    nelement = the_bool_tensor.nelement()
                    #assert 
                    assert sum_of_ori >= 0
                    assert sum_of_ori == nelement -   (nelement - sum_of_posneg1_tensor) /2
                    #                               ^^^^^ this part is number of falses ^^^^^
                    pass#for _ 
                pass#for length 

            pass# /test
        
        return 
    ____test____bool_tensor_into_posneg1()
    pass












