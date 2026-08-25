import time
from typing import List, Tuple, Optional, Literal
import torch
import math, random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
# if "test" and True:
#     assert __DEBUG_ME__()
#     pass

import sys
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######
if "test" and False:
    a = _line_()
    b = _line_()
    c = _line_()
    pass








'''assertions'''
'''assertions'''
'''assertions'''

def _float_equal(a:float|torch.Tensor, b:float, epsilon:float = 0.0001)->bool:
    if isinstance(a, torch.Tensor):
        a = a.item()
        pass
    if isinstance(b, torch.Tensor):
        b = b.item()
        pass
    assert epsilon>0.
    return abs(a-b)<epsilon
if "test" and __DEBUG_ME__() and False:
    assert _float_equal(1., 1.)
    assert _float_equal(1., 1.0000001)
    assert _float_equal(1., 1.01) == False
    assert _float_equal(1., 1.01, 0.1) 
    def ____test_____float_equal():
        a = torch.tensor(1.)
        assert _float_equal(a, 1.) 
        assert isinstance(a, torch.Tensor)
        return 
    ____test_____float_equal()
    pass



# def my_function(*args):
#     for arg in args:
#         print(arg)
# my_function(1, 2, 3, "a", "b", "c")


def _tensor_shape_check(the_tensor:torch.Tensor, *args)->bool:
    #assert isinstance(args, list)
    if (args.__len__() == 0) or (args.__len__() == 1 and args[0] == 1):
        if the_tensor.shape == torch.Size([1]):
            return True
        if the_tensor.shape == torch.Size([]):
            return True
        else:
            return False
        pass# if (args.__len__() == 0) or (args.__len__() == 1 and args[0] == 1):
    _temp_shape = torch.Size(args)
    return the_tensor.shape == _temp_shape

if "test" and __DEBUG_ME__() and True:
    def ____test______tensor_shape_check():
        assert _tensor_shape_check(torch.rand(size=[]))
        assert _tensor_shape_check(torch.rand(size=[]), 1)
        assert _tensor_shape_check(torch.rand(size=[1]))
        assert _tensor_shape_check(torch.rand(size=[1]), 1)
        assert _tensor_shape_check(torch.rand(size=[]), 0) == False
        assert _tensor_shape_check(torch.rand(size=[1]), 0) == False

        assert _tensor_shape_check(torch.rand(size=[2,3,4]), 2,3,4)

        import random
        for _ in range(155):
            a = random.randint(3,20)
            b = random.randint(3,20)
            assert _tensor_shape_check(torch.rand(size=[a,b]), a,b)
            pass

        for _ in range(155):
            a = random.randint(3,20)
            b = random.randint(3,20)
            c = random.randint(3,20)
            assert _tensor_shape_check(torch.rand(size=[a,b,c]), a,b,c)
            pass

        for _ in range(155):
            a = random.randint(3,20)
            b = random.randint(3,20)
            c = random.randint(3,20)
            d = random.randint(3,20)
            assert _tensor_shape_check(torch.rand(size=[a,b,c,d]), a,b,c,d)
            pass

        for _ in range(155):
            a = random.randint(3,20)
            b = random.randint(3,20)
            if a == b:
                continue
            assert _tensor_shape_check(torch.rand(size=[b,a]), a,b) == False
            pass
        
        for _ in range(155):
            a = random.randint(3,20)
            b = random.randint(3,20)
            assert _tensor_shape_check(torch.rand(size=[a,b]), a+1,b) == False
            pass

        return
    ____test______tensor_shape_check()
    pass






def _tensor_equal(  a:torch.Tensor|list[float]|list[list[float]], \
                    b:torch.Tensor|list[float]|list[list[float]], \
                        epsilon:float = 0.0001)->bool:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        pass
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
        pass
    #check the shape.
    if a.shape == torch.Size([]):
        assert b.shape == torch.Size([]) or b.shape == torch.Size([1])
        pass
    elif b.shape == torch.Size([]):#a is not Size([])
        assert a.shape == torch.Size([1])
        pass
    else:#no Size([]), a normal check.
        assert a.shape == b.shape
        pass
    
    # maybe I should not do this???
    # if a.device.type!=b.device.type or a.device.index!=b.device.index:
    #     proxy_of_a = a.detach().clone().to(b.device)
    with torch.inference_mode():
        diff = a-b
        abs_of_diff = diff.abs()
        less_than = abs_of_diff.lt(epsilon)
        after_all = less_than.all()
        assert after_all.dtype == torch.bool
        the_item = after_all.item()
        assert isinstance(the_item, bool)
        return the_item
    pass#end of function
if "test" and __DEBUG_ME__() and False:
    def ____test_____tensor_equal():
        assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.]))
        assert _tensor_equal(torch.tensor([1.,2.]), [1.,2.])
        #assert _tensor_equal(torch.tensor([1.]), torch.tensor([[1.]]))
        assert _tensor_equal(torch.tensor([[1.]]), torch.tensor([[1.]]))
        assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.000001]))
        assert _tensor_equal(torch.tensor([1.]), torch.tensor([0.99999]))
        assert _tensor_equal(torch.tensor([1.]), torch.tensor([1.001])) == False
        
        #shape
        assert _tensor_equal(torch.tensor([0.]), torch.tensor([0.]))
        assert _tensor_equal(torch.tensor([0.]), torch.tensor(0.))
        assert _tensor_equal(torch.tensor(0.), torch.tensor([0.]))
        assert _tensor_equal(torch.tensor(0.), torch.tensor(0.))

        return
    ____test_____tensor_equal()
    pass






def _bool_equal___0_as_false(  a:torch.Tensor|list[int]|list[list[int]], \
                    b:torch.Tensor|list[int]|list[list[int]], )->bool:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype = torch.bool)
        pass
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype = torch.bool)
        pass
    #check the shape.
    if a.shape == torch.Size([]):
        assert b.shape == torch.Size([]) or b.shape == torch.Size([1])
        pass
    elif b.shape == torch.Size([]):#a is not Size([])
        assert a.shape == torch.Size([1])
        pass
    else:#no Size([]), a normal check.
        assert a.shape == b.shape
        pass
    
    # maybe I should not do this???
    # if a.device.type!=b.device.type or a.device.index!=b.device.index:
    #     proxy_of_a = a.detach().clone().to(b.device)
    with torch.inference_mode():
        result = a.eq(b).all()
        return result
    pass#end of function
if "test" and __DEBUG_ME__() and False:
    def ____test____bool_equal____():
        if "dtype test" and True:
            int_test_1 = torch.tensor([1,  -1,  0], dtype=torch.int32)
            int_test_1 = int_test_1.to(torch.bool)
            assert int_test_1.dtype == torch.bool
            assert int_test_1[0] == True
            assert int_test_1[1] == True
            assert int_test_1[2] == False
            int_test_2 = torch.tensor([1,  -1,  0], dtype=torch.int32)
            int_test_2 = int_test_2.bool()
            assert int_test_2.dtype == torch.bool
            assert int_test_2[0] == True
            assert int_test_2[1] == True
            assert int_test_2[2] == False

            float_test_1 = torch.tensor([1., -1., 0.], dtype=torch.float)
            float_test_1 = float_test_1.to(torch.bool)
            assert float_test_1.dtype == torch.bool
            assert float_test_1[0] == True
            assert float_test_1[1] == True
            assert float_test_1[2] == False
            float_test_2 = torch.tensor([1., -1., 0.], dtype=torch.float)
            float_test_2 = float_test_2.bool()
            assert float_test_2.dtype == torch.bool
            assert float_test_2[0] == True
            assert float_test_2[1] == True
            assert float_test_2[2] == False
            pass#/ test

        if "function behavior" and True:
            assert _tensor_equal(torch.tensor([1]), torch.tensor([1]))
            assert _tensor_equal(torch.tensor([1, 1]), [1, 1])
            assert _tensor_equal(torch.tensor(1), [1])

            assert _tensor_equal(torch.tensor([1]), torch.tensor([0])) == False
            assert _tensor_equal(torch.tensor([1, 1]), [1, 0]) == False
            assert _tensor_equal(torch.tensor([1, 1]), [0, 0]) == False
            assert _tensor_equal(torch.tensor(1), [0]) == False
            pass#/ test

        return  
    ____test____bool_equal____()
    pass

def _either_1_or_neg1(input:torch.Tensor)->torch.Tensor:
    flag__is_1 = input.eq(1)
    flag__is_neg1 = input.eq(-1)
    flag__either = flag__is_1.logical_or(flag__is_neg1)
    return flag__either.all()
if "test" and __DEBUG_ME__() and False:
    def ____test____either_1_or_neg1():
        assert _either_1_or_neg1(torch.tensor([1, -1, 1]))
        assert _either_1_or_neg1(torch.tensor([[1, -1, 1]]))
        assert _either_1_or_neg1(torch.tensor([[1, -1, 1], [1, -1, 1]]))
        assert _either_1_or_neg1(torch.tensor([[1, -1, 1], [1, -1, 0]])) == False
        assert _either_1_or_neg1(torch.tensor([[1, -1, 1], [1, -1, torch.nan]])) == False
        assert _either_1_or_neg1(torch.tensor([[1, -1, 1], [1, -1, torch.inf]])) == False
        assert _either_1_or_neg1(torch.tensor([[1, -1, 1], [1, -1, -torch.inf]])) == False

        return 
    ____test____either_1_or_neg1()
    pass

















"stringify the number list."


def print_table(data:list[list], precision = 3, separator = ", ", transpose = False)->list[list[str]]:
    '''This function help with printing a table with a more readable style.'''
    if transpose: 
        #<  safety 
        # is this needed?
        for ii in range(1, data.__len__()):
            row = data[ii]
            assert row.__len__() == data[0].__len__(), \
                    f"not all rows in param:data have the same amount of elements. " + \
                    f"data[{ii}] and data[0] have different amount of elements."
            pass
        
        #<  init   empty
        buffer:list[list[str]] = []
        for row in data[0]:
            buffer.append([])
            pass
        #<  real payload
        for data_item in data:
            max_length:int = 0
            _temp___to_print:list[str] = []
            for what_to_stringify in data_item:
                if type(what_to_stringify) == float:
                    format_string = "{:."+str(precision)+"f}"
                    _temp___str = format_string.format(what_to_stringify)
                    del format_string
                    pass
                else:#not a float
                    _temp___str = str(what_to_stringify)
                    pass

                _temp___to_print.append(_temp___str)
                if _temp___str.__len__()> max_length:
                    max_length = _temp___str.__len__()
                    pass
                pass#for ii_row
            assert _temp___to_print.__len__() == data[0].__len__()
            assert type(max_length) == int
            #<  align the length

            format_string = "{:>"+str(max_length)+"}"
            for ii in range(_temp___to_print.__len__()):
                item:str = _temp___to_print[ii]
                buffer[ii].append(format_string.format(item))
                pass
            pass#for ii_colomn

        for buffer_item in buffer:
            print(separator.join(buffer_item))
            pass

        return buffer
    
    else:# not transpose
        #<  safety 
        # is this needed?
        for ii in range(1, data.__len__()):
            row = data[ii]
            assert row.__len__() == data[0].__len__(), \
                    f"not all rows in param:data have the same amount of elements. " + \
                    f"data[{ii}] and data[0] have different amount of elements."
            pass
        
        #<  init   empty
        buffer:list[list[str]] = []
        for row in data:
            buffer.append([])
            pass
        #<  real payload
        for ii_colomn in range(data[0].__len__()):
            max_length:int = 0
            _temp___to_print:list[str] = []
            for ii_row in range(data.__len__()):
                what_to_stringify = data[ii_row][ii_colomn]
                if type(what_to_stringify) == float:
                    format_string = "{:."+str(precision)+"f}"
                    _temp___str = format_string.format(what_to_stringify)
                    del format_string
                    pass
                else:#not a float
                    _temp___str = str(what_to_stringify)
                    pass

                _temp___to_print.append(_temp___str)
                if _temp___str.__len__()> max_length:
                    max_length = _temp___str.__len__()
                    pass
                pass#for ii_row
            assert _temp___to_print.__len__() == data.__len__()
            assert type(max_length) == int
            #<  align the length

            format_string = "{:>"+str(max_length)+"}"
            for ii in range(_temp___to_print.__len__()):
                item:str = _temp___to_print[ii]
                buffer[ii].append(format_string.format(item))
                pass
            pass#for ii_colomn

        for buffer_item in buffer:
            print(separator.join(buffer_item))
            pass

        return buffer
    #end of function
if "test" and __DEBUG_ME__() and False:
    def ____test____print_table():
        if "no transpose" and False:
            buffer = print_table([[111.12, "aa", 443.3],
                                    ["a", 1.1,      "bb"]])
            assert buffer == [  ["111.120", "   aa", "443.300"],
                                ["      a", "1.100", "     bb"]]

            buffer = print_table([
                    ["scan param", 0.01,  0.1,   1.,   1000.],
                    ["mean",      1.111, 1.11, 1.100, 1.000],
                    ["max",           1,   12,   123,  1234],])
            
            assert buffer == [  ["scan param", "0.010", "0.100", "1.000", "1000.000"],
                                ["      mean", "1.111", "1.110", "1.100", "   1.000"],
                                ["       max", "    1", "   12", "  123", "    1234"],]

            # assert buffer.__len__() == buffer1.__len__()
            # for ii in range(buffer.__len__()):
            #     aaaaa = buffer[ii]
            #     bbbbb = buffer1[ii]
            #     for jj in range(aaaaa.__len__()):
            #         assert aaaaa[jj] == bbbbb[jj]
            #         pass
            #     pass
            pass#/ test

        if "with transpose" and True:
            buffer = print_table([[111.12, "aa", 443.3],
                                    ["a", 1.1,      "bb"]], transpose=True)
            assert buffer == [  ["111.120", "    a"], 
                                ["     aa", "1.100"],
                                ["443.300", "   bb"],]


            buffer = print_table([
                    ["scan param", 0.01,  0.1,   1.,   1000.],
                    ["mean",      1.111, 1.11, 1.100, 1.000],
                    ["max",           1,   12,   123,  1234],], transpose=True)
            
            assert buffer == [  ["scan param", " mean", " max"],        
                                ["     0.010", "1.111", "   1"],
                                ["     0.100", "1.110", "  12"],
                                ["     1.000", "1.100", " 123"],
                                ["  1000.000", "1.000", "1234"],]
            pass#/ test
        return
    ____test____print_table()
    pass


def str_the_list(the_list:list, precision = 3, separator = ", ")->str:
    format_string = "{: ."+str(precision)+"f}"
    all_the_sub_strings = []
    for the_number in the_list:
        all_the_sub_strings.append(format_string.format(the_number))
        pass
    # for ii in range(all_the_sub_strings.__len__()):
    #     sub_str = all_the_sub_strings[ii]
    #     if sub_str[0]!='-':
    #         all_the_sub_strings[ii] = " "+all_the_sub_strings[ii]
    #         pass
    #     pass
            
    mid_str = separator.join(all_the_sub_strings)
    result = f"[{mid_str}]"
    return result

if "test" and __DEBUG_ME__() and True:
    def ____test____str_the_list():
        the_str = str_the_list([1.23467,-2.23467], 3)
        assert the_str == "[ 1.235, -2.235]"
        
        the_str = str_the_list([1.23467,-2.23467], 3, separator="...")
        assert the_str == "[ 1.235...-2.235]"
        
        return
    ____test____str_the_list()
    pass

def str_the_list__probability(the_list:list, precision = 3,
                good_prefix = "  ", bad_prefix  = "XX", separator = ", ", white_space_in_mid = 1,
                flag__offset_by50 = False, flag__mul_2_after_offset = False, )->str:
    assert good_prefix.__len__() == bad_prefix.__len__()
    assert white_space_in_mid>=1
    if not flag__offset_by50:
        assert not flag__mul_2_after_offset, "when flag__offset_by50=False, this flag__mul_2_after_offset doesn't do anything."
    
    _format_str = "{:."+str(precision)+"f}"
    _perfect_str = " v"+" "*(precision+white_space_in_mid+2)
    _worst_str   = "XX"+" "*(precision+white_space_in_mid+2)
    str_white_space_in_mid = " "*white_space_in_mid
    str_white_space_in_mid__shorter = " "*(white_space_in_mid-1)
    
    all_the_sub_strings:list[str] = []
    for item in the_list:
        if item == 1.:
            all_the_sub_strings.append(_perfect_str)
            pass
        elif item == 0.:
            all_the_sub_strings.append(_worst_str)
            pass
        else:
            #<  number_str
            print_this_number = item
            if flag__offset_by50:
                print_this_number -= 0.5
                if flag__mul_2_after_offset:
                    print_this_number*=2.
                    pass
                pass
            number_str = _format_str.format(print_this_number)
            
            #<  prefix
            if item>0.5:
                prefix = good_prefix
                pass
            else:
                prefix = bad_prefix
                pass
            
            if number_str[0] == '-':
                all_the_sub_strings.append(prefix+str_white_space_in_mid__shorter+number_str)
                pass
            else:
                all_the_sub_strings.append(prefix+str_white_space_in_mid+number_str)
                pass
            pass
        
        pass
    
    #debug test. Turn off after test.
    # assert all_the_sub_strings.__len__() == the_list.__len__()
    # for sub_str in all_the_sub_strings:
    #     assert sub_str.__len__() == all_the_sub_strings[0].__len__()
    #     pass
    
    mid_str = separator.join(all_the_sub_strings)
    result = f"[{mid_str}]"
    return result

if "test" and __DEBUG_ME__() and False:
    def ____test____str_the_list__probability():
        the_str = str_the_list__probability([0.,0.1,0.9,1.], 2)
        assert the_str == "[XX     , XX 0.10,    0.90,  v     ]"
        
        the_str = str_the_list__probability([0.,0.1,0.9,1.], 3)
        assert the_str == "[XX      , XX 0.100,    0.900,  v      ]"
        
        the_str = str_the_list__probability([0.,0.1,0.9,1.], 2, good_prefix = "gg")
        assert the_str == "[XX     , XX 0.10, gg 0.90,  v     ]"
        
        the_str = str_the_list__probability([0.,0.1,0.9,1.], 2, bad_prefix = "gg")
        assert the_str == "[XX     , gg 0.10,    0.90,  v     ]"
            
        the_str = str_the_list__probability([0.,0.1,0.9,1.], 2, white_space_in_mid = 2)
        assert the_str == "[XX      , XX  0.10,     0.90,  v      ]"
        
        the_str = str_the_list__probability([0.,0.1,0.9,1.], 2, flag__offset_by50 = True)
        assert the_str == "[XX     , XX-0.40,    0.40,  v     ]"
        
        the_str = str_the_list__probability([0.,0.1,0.9,1.], 2, flag__offset_by50 = True, flag__mul_2_after_offset=True)
        assert the_str == "[XX     , XX-0.80,    0.80,  v     ]"
        
        the_str = str_the_list__probability([0.,0.1,0.9,1.], 2, separator="..")
        assert the_str == "[XX     ..XX 0.10..   0.90.. v     ]"
        
        return
    ____test____str_the_list__probability()
    pass








if "adaptor func    if_number_then_to_tensor     maybe later??" and False:
    # def if_number_then_to_tensor(input:int|float|torch.Tensor)->torch.Tensor:
    #     if isinstance(input, int):
    #         return torch.tensor(input, dtype=torch.int64)
    #     if isinstance(input, float):
    #         return torch.tensor(input, dtype=torch.float32)
    #     return input.detach().clone()
    # if "test" and __DEBUG_ME__() and True:
    #     def ____test____if_number_then_to_tensor():
    #         a = if_number_then_to_tensor(123)
    #         assert isinstance(a, torch.Tensor)
    #         assert a == 123
    #         assert a.dtype == torch.int64
    #         assert a.device.type == 'cpu'
            
    #         b = if_number_then_to_tensor(1234.)
    #         assert isinstance(b, torch.Tensor)
    #         assert b == 1234.
    #         assert b.dtype == torch.float32
    #         assert b.device.type == 'cpu'
            
    #         return 
    #     ____test____if_number_then_to_tensor()
    
    pass










"smart expand."

def expand_vec_to_matrix(input:torch.Tensor, each_element_to:Literal["row", "col", "column"])->torch.Tensor:
    assert input.shape.__len__() == 1, "batch feature not implemented yet."
    dim = input.shape[0]
    if each_element_to.lower() == "row":
        return input.reshape([-1,1]).expand([-1,dim])
    if each_element_to.lower() in ["col", "column"]:
        return input.reshape([1,-1]).expand([dim,-1])
    assert False, "bad param: each_element_to. Only "
    #end of function.

if "test" and __DEBUG_ME__() and False:
    def ____test____expand_vec_to_matrix():
        the_vec = torch.tensor([2.,3])
        the_matrix = expand_vec_to_matrix(the_vec, each_element_to="row")
        assert the_matrix.eq(torch.tensor([[ 2., 2],
                                            [3,  3]])).all()
        
        the_vec = torch.tensor([2.,3])
        the_matrix = expand_vec_to_matrix(the_vec, each_element_to="col")
        assert the_matrix.eq(torch.tensor([[ 2., 3],
                                            [2,  3]])).all()
        
        #<  device adaption.
        
        the_vec = torch.tensor([2.,3], device='cuda')
        the_matrix = expand_vec_to_matrix(the_vec, each_element_to="col")
        assert the_matrix.device.type == 'cuda'        
        
        return
    
    ____test____expand_vec_to_matrix()
    pass




"check the matrix"

def have_same_elements(tensor_1:torch.Tensor, tensor_2:torch.Tensor)->bool:
    assert tensor_1.shape == tensor_2.shape

    tensor_1_elements = tensor_1.reshape([-1]).sort().values
    tensor_2_elements = tensor_2.reshape([-1]).sort().values
    return tensor_1_elements.eq(tensor_2_elements).all()
if "test" and __DEBUG_ME__() and False:
    def ____test____have_same_elements():
        t1 = torch.tensor([1,2,3])
        t2 = torch.tensor([3,2,1])
        assert have_same_elements(t1,t2)
        
        t1 = torch.tensor([[1,2,3]])
        t2 = torch.tensor([[3,2,1]])
        assert have_same_elements(t1,t2)
        
        t1 = torch.tensor([[1,2,1111]])
        t2 = torch.tensor([[3,2,1]])
        assert have_same_elements(t1,t2) == False
        return
    ____test____have_same_elements()
    pass

def is_square_matrix(matrix:torch.Tensor)->bool:
    if matrix.shape.__len__() != 2:
        return False
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return True
if "test" and __DEBUG_ME__() and False:
    assert is_square_matrix(torch.randn(size=(2,2)))
    assert is_square_matrix(torch.randn(size=(3,3)))
    assert is_square_matrix(torch.randn(size=(2,3))) == False
    assert is_square_matrix(torch.randn(size=(2,))) == False
    assert is_square_matrix(torch.randn(size=(2,3,3))) == False
    pass

from typing import TypeAlias,Literal
DeviceLikeType: TypeAlias = str|torch.device|int
def iota(how_many:int, dtype_is_int64=False,\
            device: DeviceLikeType|None = None)->torch.Tensor:
    dtype = torch.int64
    if not dtype_is_int64 and how_many<(1<<31):
        dtype = torch.int32
        pass
    return torch.linspace(start=0,end=how_many-1,steps=how_many ,dtype=dtype, device=device)

if "torch linspace dtype test" and __DEBUG_ME__() and False:
    for device in ["cpu", "cuda"]:
        for dtype in [torch.int8,torch.int16,torch.int32,torch.int64,torch.int,torch.uint8,torch.long]:
                                            #but uint16,uint32,uint64 are not allowed
            _temp = torch.linspace(start=0,end=7,steps=8 ,dtype=dtype, device=device)
            pass
        pass
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int8)
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int16)
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int32)
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int64)
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint8)
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint16) not working
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint32) not working
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint64) not working
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int)#int32
    assert _temp_tensor.dtype == torch.int32
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.long)#int64
    assert _temp_tensor.dtype == torch.int64
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int8,device='cuda')
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int16,device='cuda')
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int32,device='cuda')
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int64,device='cuda')
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint8,device='cuda')
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint16,device='cuda') not working
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint32,device='cuda') not working
    #_temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.uint64,device='cuda') not working
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.int,device='cuda')#int32
    assert _temp_tensor.dtype == torch.int32
    _temp_tensor = torch.linspace(start=0,end=7,steps=8 ,dtype=torch.long,device='cuda')#int64
    assert _temp_tensor.dtype == torch.int64
    pass    
if "test" and __DEBUG_ME__() and False:
    _1_leftshift_31_minus_1 = (1<<31)-1
    _temp = iota(_1_leftshift_31_minus_1)
    assert _temp.__len__() == _1_leftshift_31_minus_1
    assert _temp[-1] == _1_leftshift_31_minus_1-1
    assert _temp.dtype == torch.int32
    
    _1_leftshift_31 = 1<<31
    _temp = iota(_1_leftshift_31)
    assert _temp.__len__() == _1_leftshift_31
    assert _temp[-1] == _1_leftshift_31-1
    assert _temp.dtype == torch.int64
    
    _temp = iota(3, dtype_is_int64=True)
    assert _tensor_equal(_temp, [0.,1,2])
    assert _temp.dtype == torch.int64
    pass
if "can it be index?" and __DEBUG_ME__() and False:
    "pytorch only allows int32 or int64 as index."
    _data = torch.linspace(0,99,100, dtype=torch.float32).reshape([10,10])
    
    iota_of_data = iota(4)
    assert iota_of_data.dtype == torch.int32
    _part_of_data = _data[iota_of_data, iota_of_data]
    assert _tensor_equal(_part_of_data, [0.,11,22,33])
    
    iota_of_data = iota(4, dtype_is_int64=True)
    assert iota_of_data.dtype == torch.int64
    _part_of_data = _data[iota_of_data, iota_of_data]
    assert _tensor_equal(_part_of_data, [0.,11,22,33])
    
    if "the following don't work. They raise." and False and False and False:
        _data = torch.linspace(0,99,100, dtype=torch.float32).reshape([10,10])
        iota_of_data = iota(4)
        _part_of_data = _data[iota_of_data.to(torch.int8), iota_of_data]
        _part_of_data = _data[iota_of_data.to(torch.int16), iota_of_data]
        _part_of_data = _data[iota_of_data.to(torch.uint8), iota_of_data]
        pass
    pass






'''measuren a M@(M.T). 再想想'''
def info_of_abs_of_triu(input:torch.Tensor, including_diagonal = False, 
                                needs_max = False)->tuple[torch.Tensor, torch.Tensor|None]:
    assert False, "再想想"
    '''return the_avg, the_max
    
    This function was designed to measuren a M@(M.T).'''
    assert input.shape.__len__() == 2
    dim = input.shape[-1]
    
    if including_diagonal:#with diag
        before_sum__d_d = input.triu(diagonal=0).abs()#with diag
        the_sum = before_sum__d_d.sum()
        the_avg = the_sum/((dim+1)*dim/2.)
        pass
    else:#no diag
        before_sum__d_d = input.triu(diagonal=1).abs()#no diag
        the_sum = before_sum__d_d.sum()
        the_avg = the_sum/((dim-1)*dim/2.)
        pass

    the_max = None
    if needs_max:
        the_max = before_sum__d_d.max()
        assert the_max.shape == torch.Size([])
        pass

    return the_avg, the_max
if "test" and __DEBUG_ME__() and False:
    def ____debug____info_of_abs_of_triu():
        if "basic" and True:

            input = torch.tensor([[  11., 12, 13],
                                    [51., 52, 53],
                                    [61., 62, 63],])

            the_avg, the_max = info_of_abs_of_triu(input, needs_max=True)
            assert _tensor_equal(input, torch.tensor([[  11., 12, 13],
                                                        [51., 52, 53],
                                                        [61., 62, 63],]))
            assert the_avg == (12+13+53)/3.
            assert the_max == 53

            input = torch.tensor([[  11., 12, 13],
                                    [51., 52, 53],
                                    [61., 62, 63],])
            the_avg, the_max = info_of_abs_of_triu(input, including_diagonal = True, needs_max=True)
            assert _tensor_equal(input, torch.tensor([[  11., 12, 13],
                                                        [51., 52, 53],
                                                        [61., 62, 63],]))
            assert the_avg == (11+12+13+52+53+63)/6
            assert the_max == 63
            
            
            input = torch.tensor([[  11., 12, 13],
                                    [51., 52, 53],
                                    [61., 62, 63],])
            the_avg, the_max = info_of_abs_of_triu(input)
            assert _tensor_equal(input, torch.tensor([[  11., 12, 13],
                                                        [51., 52, 53],
                                                        [61., 62, 63],]))
            assert the_avg == (12+13+53)/3.
            assert the_max is None
            pass#/ test

        if "not including diag" and False:
            for dim in [3,5,10]:
                for ii_test in range(33):
                    #<  ori
                    input = torch.randn(size=[dim,dim])
                    ori_input = input.detach().clone()
                    before_avg, before_max = info_of_abs_of_triu(ori_input, needs_max=True)
                    #<  modified
                    for _ in range(dim*dim//2):
                        sum_of_2_index = random.randint(0, dim-1)
                        index_0 = random.randint(0, sum_of_2_index)
                        index_1 = sum_of_2_index - index_0
                        assert index_0 <= dim-1
                        assert index_1 <= dim-1
                        index_0 = (dim-1)-index_0
                        assert index_0 <= dim-1 and index_0 >= 0
                        input[index_0, index_1] = torch.randn(size=[])
                        pass
                    #<  calc new
                    # prin(ori_input)
                    # prin(input)
                    after_avg, after_max = info_of_abs_of_triu(input, needs_max=True)
                    #<  assert
                    assert _tensor_equal(before_avg, after_avg)
                    assert _tensor_equal(before_max, after_max)
                    pass# for ii_test
                pass# for dim
            pass#/ test

        if "including diag" and True:
            for dim in [3, 5, 10]:
                for ii_test in range(33):
                    #<  ori
                    input = torch.randn(size=[dim,dim])
                    ori_input = input.detach().clone()
                    before_avg, before_max = info_of_abs_of_triu(ori_input, including_diagonal=True, needs_max=True)
                    #<  modified
                    for _ in range(dim*dim//2):
                        sum_of_2_index = random.randint(0, dim-2)
                        index_0 = random.randint(0, sum_of_2_index)
                        index_1 = sum_of_2_index - index_0
                        assert index_0 <= dim-1
                        assert index_1 <= dim-1
                        index_0 = (dim-1)-index_0
                        assert index_0 <= dim-1 and index_0 >= 0

                        #input[index_0, index_1] = torch.nan
                        input[index_0, index_1] = torch.randn(size=[])
                        pass
                    #<  calc new
                    print(ori_input)
                    print(input)

                    after_avg, after_max = info_of_abs_of_triu(input, including_diagonal=True, needs_max=True)
                    #<  assert
                    assert _tensor_equal(before_avg, after_avg)
                    assert _tensor_equal(before_max, after_max)
                    pass# for ii_test
                pass# for dim
            pass#/ test

        return 
    ____debug____info_of_abs_of_triu()
    pass







"vector length"

def vector_length_norm(input:torch.Tensor, epsilon:float|torch.Tensor = 0.001, dtype_inner = torch.float64)->torch.Tensor:
    r'''The shape must be [batch, dim]
    
    I don't remember, but the dtype_inner feels like also the return dtype.'''
    if input.shape.__len__()!=2:
        raise Exception("The shape must be [batch, dim]")
    with torch.no_grad():
        
        if isinstance(epsilon, float):
            epi_tensor = torch.tensor([epsilon], device=input.device, dtype=dtype_inner)
            pass
        else:
            epi_tensor = epsilon
            pass
        
        length_of_input_b_1 = input.mul(input).sum(dim=1,keepdim=True,dtype=dtype_inner).sqrt()
        length_of_input_safe__b = length_of_input_b_1.maximum(epi_tensor)
        
        if input.device.type == 'cpu':
            if input.shape[-1]<300:#div version
                # div version
                length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b.expand([-1,input.shape[1]])
                length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b_1EXPANDdim.to(dtype=input.dtype)
                result = input.div(length_of_input_safe__b_1EXPANDdim)
                return result
                
            # mul version
            mul_me_before_expand__b = 1./length_of_input_safe__b
            mul_me_before_expand__b = mul_me_before_expand__b.to(dtype=input.dtype)
            mul_me___b_1EXPANDdim = mul_me_before_expand__b.expand([-1,input.shape[1]])
            result = input*mul_me___b_1EXPANDdim
            return result
        
        if input.device.type == 'cuda':
            if input.nelement()<=1000_000:
                # div version
                length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b.expand([-1,input.shape[1]])
                length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b_1EXPANDdim.to(dtype=input.dtype)
                result = input.div(length_of_input_safe__b_1EXPANDdim)
                return result
                
            # mul version
            mul_me_before_expand__b = 1./length_of_input_safe__b
            mul_me_before_expand__b = mul_me_before_expand__b.to(dtype=input.dtype)
            mul_me___b_1EXPANDdim = mul_me_before_expand__b.expand([-1,input.shape[1]])
            result = input*mul_me___b_1EXPANDdim
            return result
        
        assert False, "unknown device. implement it or choose any of my version. The performance test is below."
        # mul version
        # mul_me_before_expand__b = 1./length_of_input_safe__b
        # mul_me_before_expand__b = mul_me_before_expand__b.to(dtype=input.dtype)
        # mul_me___b_1EXPANDdim = mul_me_before_expand__b.expand([-1,input.shape[1]])
        # result = input*mul_me___b_1EXPANDdim
        
        # div version
        # length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b.expand([-1,input.shape[1]])
        # length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b_1EXPANDdim.to(dtype=input.dtype)
        # result = input.div(length_of_input_safe__b_1EXPANDdim)
        
    #end of function.
if "performance test of two versions" and False:
    def ____test____performance_test_of_two_versions():
        from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
        if True:
            # cpu, dim<300, div
            # gpu, n <= 1e6, div
            # 10
            # mul__ver  = [ 0.01038,  0.00968,  0.01900,  0.06659,  0.39719]
            # div__ver  = [ 0.00796,  0.00811,  0.01620,  0.06072,  0.38007]
            # mul____gpu= [ 0.04148,  0.03992,  0.03992,  0.03997,  0.05870]
            # div____gpu= [ 0.03190,  0.03880,  0.03408,  0.03114,  0.05448]
            # how_many_ = [ 10,      100,      1000,      10000,      100000]
            # 100
            # mul__ver  = [ 0.00966,  0.01127,  0.04011,  0.17684,  2.06436]
            # div__ver  = [ 0.00728,  0.01081,  0.04097,  0.21540,  1.94373]
            # mul____gpu= [ 0.04303,  0.04036,  0.04051,  0.05185,  0.26671]
            # div____gpu= [ 0.03606,  0.03435,  0.03272,  0.04809,  0.31185]
            # how_many_ = [ 10,      100,      1000,      10000,      100000]
            # 1000
            # mul__ver  = [ 0.00270,  0.00643,  0.02093,  1.01336]
            # div__ver  = [ 0.00270,  0.00755,  0.02905,  1.27344]
            # mul____gpu= [ 0.01474,  0.01578,  0.03572,  0.24063]
            # div____gpu= [ 0.01103,  0.01267,  0.03782,  0.28381]
            # how_many_ = [ 10,      100,      1000,      10000]
            # 10000
            # mul__ver  = [ 0.00642,  0.02090,  0.99586]
            # div__ver  = [ 0.00751,  0.03018,  1.26331]
            # mul____gpu= [ 0.01555,  0.03419,  0.23736]
            # div____gpu= [ 0.01280,  0.03611,  0.28248]
            # how_many_ = [ 10,      100,      1000]
            # 100000
            # mul__ver  = [ 0.02126,  0.98957]
            # div__ver  = [ 0.03096,  1.28091]
            # mul____gpu= [ 0.03366,  0.23825]
            # div____gpu= [ 0.03579,  0.28197]
            # how_many_ = [ 10,      100]
                        
            
            
            
            
            
            
            
            #----------------#----------------#----------------
            loop_time = 100
            time_at_most = 2.
            
            dim_list = [  10, 100, 1000,10000, 100000]
            for dim in dim_list:
                print(dim)
            #----------------#----------------#----------------
                
                mul__ver = []#don't modify this.
                div__ver = []#don't modify this.
                mul__ver__gpu = []#don't modify this.
                div__ver__gpu = []#don't modify this.
                
                #----------------#----------------#----------------
                how_many_vec_list = []
                for _raw_how_many in [  10, 100, 1000,10000, 100000]:
                    if _raw_how_many*dim<=10_000_000:
                        how_many_vec_list.append(_raw_how_many)
                        pass
                    pass
                    
                #for outter_param_count in range(how_many_vec_list.__len__()):
                    #how_many_vec = how_many_vec_list[outter_param_count]
                for how_many_vec in how_many_vec_list:
                #----------------#----------------#----------------
                    
                #----------------#----------------#----------------
                    input = torch.randn(size=[how_many_vec,dim], device='cpu')
                    epsilon = torch.tensor(0.001)
                    dtype_inner = torch.float64
                    
                    def _timeit_null():
                        for _ in range(loop_time):
                            if input.shape.__len__()!=2:
                                raise Exception("The shape must be [batch, dim]")
                            with torch.no_grad():
                                pass
                            pass
                        return 
                    null_time = timeit(_timeit_null, time_at_most=time_at_most)[0]
                    del _timeit_null
                    
                    def _timeit_mul():
                        for _ in range(loop_time):
                            
                            if input.shape.__len__()!=2:
                                raise Exception("The shape must be [batch, dim]")
                            with torch.no_grad():
                                
                                length_of_input_b_1 = input.mul(input).sum(dim=1,keepdim=True,dtype=dtype_inner).sqrt()
                                
                                epi_tensor = torch.tensor([epsilon], device=length_of_input_b_1.device, dtype=dtype_inner)
                                length_of_input_safe__b = length_of_input_b_1.maximum(epi_tensor)
                                
                                #this is the new version but I didn't test the performance.
                                mul_me_before_expand__b = 1./length_of_input_safe__b
                                mul_me_before_expand__b = mul_me_before_expand__b.to(dtype=input.dtype)
                                mul_me___b_1EXPANDdim = mul_me_before_expand__b.expand([-1,input.shape[1]])
                                result = input*mul_me___b_1EXPANDdim
                            
                            pass
                        return 
                    _timeit_mul()
                    raw_mul_time = timeit(_timeit_mul, time_at_most=time_at_most)[0]
                    mul_time = raw_mul_time-null_time
                    mul__ver.append(mul_time)
                    del _timeit_mul, raw_mul_time, mul_time
                    
                    def _timeit_div():
                        for _ in range(loop_time):
                            
                            if input.shape.__len__()!=2:
                                raise Exception("The shape must be [batch, dim]")
                            with torch.no_grad():
                                
                                length_of_input_b_1 = input.mul(input).sum(dim=1,keepdim=True,dtype=dtype_inner).sqrt()
                                
                                epi_tensor = torch.tensor([epsilon], device=length_of_input_b_1.device, dtype=dtype_inner)
                                length_of_input_safe__b = length_of_input_b_1.maximum(epi_tensor)
                                
                                #this is the new version but I didn't test the performance.
                                # mul_me_before_expand__b = 1./length_of_input_safe__b
                                # mul_me_before_expand__b = mul_me_before_expand__b.to(dtype=input.dtype)
                                # mul_me___b_1EXPANDdim = mul_me_before_expand__b.expand([-1,input.shape[1]])
                                # result = input*mul_me___b_1EXPANDdim
                                
                                #old code
                                length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b.expand([-1,input.shape[1]])
                                length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b_1EXPANDdim.to(dtype=input.dtype)
                                result = input.div(length_of_input_safe__b_1EXPANDdim)
                            
                            pass
                        return 
                    _timeit_div()
                    raw_div_time = timeit(_timeit_div, time_at_most=time_at_most)[0]
                    div_time = raw_div_time-null_time
                    div__ver.append(div_time)
                    del _timeit_div, raw_div_time, div_time
                    del null_time
                    
                    # ^^^^ cpu ^^^^
                    # vvvv gpu vvvv
                    
                    input = torch.randn(size=[how_many_vec,dim], device='cuda')
                    epsilon = torch.tensor(0.001, device='cuda')
                    dtype_inner = torch.float64
                    
                    def _timeit_null():
                        for _ in range(loop_time):
                            if input.shape.__len__()!=2:
                                raise Exception("The shape must be [batch, dim]")
                            with torch.no_grad():
                                pass
                            pass
                        return 
                    null_time = timeit(_timeit_null, time_at_most=time_at_most)[0]
                    del _timeit_null
                    
                    def _timeit_mul():
                        for _ in range(loop_time):
                            
                            if input.shape.__len__()!=2:
                                raise Exception("The shape must be [batch, dim]")
                            with torch.no_grad():
                                
                                length_of_input_b_1 = input.mul(input).sum(dim=1,keepdim=True,dtype=dtype_inner).sqrt()
                                
                                epi_tensor = torch.tensor([epsilon], device=length_of_input_b_1.device, dtype=dtype_inner)
                                length_of_input_safe__b = length_of_input_b_1.maximum(epi_tensor)
                                
                                #this is the new version but I didn't test the performance.
                                mul_me_before_expand__b = 1./length_of_input_safe__b
                                mul_me_before_expand__b = mul_me_before_expand__b.to(dtype=input.dtype)
                                mul_me___b_1EXPANDdim = mul_me_before_expand__b.expand([-1,input.shape[1]])
                                result = input*mul_me___b_1EXPANDdim
                            
                            pass
                        return 
                    _timeit_mul()
                    raw_mul_time = timeit(_timeit_mul, time_at_most=time_at_most)[0]
                    mul_time = raw_mul_time-null_time
                    mul__ver__gpu.append(mul_time)
                    del _timeit_mul, raw_mul_time, mul_time
                    
                    def _timeit_div():
                        for _ in range(loop_time):
                            
                            if input.shape.__len__()!=2:
                                raise Exception("The shape must be [batch, dim]")
                            with torch.no_grad():
                                
                                length_of_input_b_1 = input.mul(input).sum(dim=1,keepdim=True,dtype=dtype_inner).sqrt()
                                
                                epi_tensor = torch.tensor([epsilon], device=length_of_input_b_1.device, dtype=dtype_inner)
                                length_of_input_safe__b = length_of_input_b_1.maximum(epi_tensor)
                                
                                #this is the new version but I didn't test the performance.
                                # mul_me_before_expand__b = 1./length_of_input_safe__b
                                # mul_me_before_expand__b = mul_me_before_expand__b.to(dtype=input.dtype)
                                # mul_me___b_1EXPANDdim = mul_me_before_expand__b.expand([-1,input.shape[1]])
                                # result = input*mul_me___b_1EXPANDdim
                                
                                #old code
                                length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b.expand([-1,input.shape[1]])
                                length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b_1EXPANDdim.to(dtype=input.dtype)
                                result = input.div(length_of_input_safe__b_1EXPANDdim)
                            
                            pass
                        return 
                    _timeit_div()
                    raw_div_time = timeit(_timeit_div, time_at_most=time_at_most)[0]
                    div_time = raw_div_time-null_time
                    div__ver__gpu.append(div_time)
                    del _timeit_div, raw_div_time, div_time
                    del null_time
                    
                    pass#for outter_param_count
                print(f"dim {dim}")
                print(f"mul__ver  = {str_the_list(mul__ver  , 5)}")
                print(f"div__ver  = {str_the_list(div__ver  , 5)}")
                print(f"mul____gpu= {str_the_list(mul__ver__gpu  , 5)}")
                print(f"div____gpu= {str_the_list(div__ver__gpu  , 5)}")
                print(f"how_many_ = {str_the_list(how_many_vec_list     , 0, ",     ")}")
                pass# for dim
            pass#/test
        
        return 
    
    ____test____performance_test_of_two_versions()
    pass
    
if '''some basic test.''' and __DEBUG_ME__() and True:
    def ____test____vector_length_norm():
        input = torch.tensor([[0.,0.],[0.,1.],[1.,1.]])
        output = vector_length_norm(input)
        assert _tensor_equal(output, [[0.,0.],[0.,1.],[0.7,0.7]], epsilon = 0.05)
        assert output.dtype == torch.float32
        _vector_len = output.mul(output).sum(dim=1)
        assert _tensor_equal(_vector_len[0], torch.zeros_like(_vector_len[0]), epsilon = 0.05)
        assert _tensor_equal(_vector_len[1:], torch.ones_like(_vector_len[1:]), epsilon = 0.05)
        
        #transform
        input = torch.tensor([[  1.,   1],
                                [0.1,  0.1]])
        output = vector_length_norm(input.T).T
        assert _tensor_equal(output,   [[0.9950, 0.9950],
                                        [0.0995, 0.0995]], epsilon=0.001)

        input = torch.tensor([                     [0.],[0.1],[0.01],[0.001],[0.0001],[10],[100],[1000],[10000]])
        output = vector_length_norm(input, epsilon=0.01)
        assert _tensor_equal(output, torch.tensor([[0.],[1.], [1.],  [0.1],  [0.01],  [1.],[1.], [1.],  [1.],]))

        return 
    ____test____vector_length_norm()
    pass

def get_vector_length(input:torch.Tensor, result_dtype = torch.float64)->torch.Tensor:
    _temp = input*input
    _temp = _temp.sum(dim=-1, dtype=result_dtype)
    _temp.sqrt_()
    return _temp
if "test get_vector_length" and __DEBUG_ME__() and False:
    def ____test____get_vector_length():
        input = torch.tensor([1.,1])
        output = get_vector_length(input)
        assert output.shape == torch.Size([])
        assert output.dtype == torch.float64
        assert _tensor_equal(output, [1.4142])
        
        input = torch.tensor([[1.,1],[1,2]])
        output = get_vector_length(input)
        assert output.shape == torch.Size([2])
        assert _tensor_equal(output, [1.4142,2.2361])
        
        input = torch.tensor([[0.71, 0.71],[1,0]])
        output = get_vector_length(input)
        assert output.shape == torch.Size([2])
        assert _tensor_equal(output, [1., 1], epsilon=0.01)
        
        input = torch.tensor([[  0.71, 1],
                                [0.71, 0]])
        output = get_vector_length(input.T)
        assert output.shape == torch.Size([2])
        assert _tensor_equal(output, [1., 1], epsilon=0.01)
        
        input = torch.tensor([[[1.,1],[1,2]],[[2,1],[2,2]],[[3,1],[3,2]]])
        output = get_vector_length(input)
        assert output.shape == torch.Size([3,2])
        assert _tensor_equal(output, [[1.4142,2.2361],[2.2361, 2.8284],[3.1623,3.6056]])
        
        "dtype"
        input = torch.tensor([1.,1])
        output = get_vector_length(input, result_dtype=torch.float16)
        assert output.dtype == torch.float16
        
        return 
    ____test____get_vector_length()
    pass

def get_full_info_of_vector_length__1d(input:torch.Tensor, epi = 0.000001)-> \
                                                    tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''return normalized_vector, length_of_input, sqr_length_of_input'''
    assert input.shape.__len__() == 1# [dim]
    
    sqr_length_of_input__s = input.mul(input).sum()
    length_of_input__s = sqr_length_of_input__s.sqrt()
    epi_tensor = torch.tensor([epi], device=length_of_input__s.device)
    length_of_input_safe__s = length_of_input__s.maximum(epi_tensor)
    normalized_vector = input.div(length_of_input_safe__s)

    return normalized_vector, length_of_input__s.squeeze(dim=-1), sqr_length_of_input__s.squeeze(dim=-1)
if '''some basic test.''' and __DEBUG_ME__() and False:
    def ____test____get_full_info_of_vector_length__1d():
        #this test func is a raw combination of the 2 above.
        input = torch.tensor([0.,0.])
        normalized_vector, length_of_input, sqr_length_of_input = get_full_info_of_vector_length__1d(input)
        assert _tensor_equal(normalized_vector, [0.,0.])
        assert normalized_vector.dtype == torch.float32
        assert _tensor_equal(get_vector_length(normalized_vector), [0.])
        assert _tensor_equal(length_of_input, [0.])
        assert length_of_input.dtype == torch.float32
        assert _tensor_equal(sqr_length_of_input, [0.])
        assert sqr_length_of_input.dtype == torch.float32
        
        input = torch.tensor([1.,0.])
        normalized_vector, length_of_input, sqr_length_of_input = get_full_info_of_vector_length__1d(input)
        assert _tensor_equal(normalized_vector, [1.,0.])
        assert _tensor_equal(get_vector_length(normalized_vector), [1.])
        assert _tensor_equal(length_of_input, [1.])
        assert _tensor_equal(sqr_length_of_input, [1.])
        
        input = torch.tensor([1.,1.])
        normalized_vector, length_of_input, sqr_length_of_input = get_full_info_of_vector_length__1d(input)
        assert _tensor_equal(normalized_vector, [0.7071, 0.7071])
        assert _tensor_equal(get_vector_length(normalized_vector), [1.])
        assert _tensor_equal(length_of_input, [1.4142])
        assert _tensor_equal(sqr_length_of_input, [2.])
        
        return 
    
    ____test____get_full_info_of_vector_length__1d()
    pass

def get_full_info_of_vector_length__2d(input:torch.Tensor, epi = 0.000001, #dtype_inner = torch.float64,
                                length_result_dtype = torch.float64, keepdim_for_length = False)-> \
                                                            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''return normalized_vector, length_of_input, sqr_length_of_input'''
    assert input.shape.__len__() == 2, "Manully reshape please."# [..., dim]
    
    # __bat = input.shape[0]
    # __dim = input.shape[1]
    
    sqr_length_of_input__b_1 = input.mul(input).sum(dim=-1,keepdim=True,dtype=length_result_dtype)
    #assert sqr_length_of_input__b_1.shape == torch.Size([__bat,1])
    length_of_input__b_1 = sqr_length_of_input__b_1.sqrt()
    #assert sqr_length_of_input__b_1.shape == torch.Size([__bat,1])
    epi_tensor = torch.tensor([epi], device=length_of_input__b_1.device, dtype=length_result_dtype)
    #assert epi_tensor.shape == torch.Size([1])
    length_of_input_safe__b_1 = length_of_input__b_1.maximum(epi_tensor)
    #assert length_of_input_safe__b_1.shape == torch.Size([__bat, 1])
    length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b_1.expand([-1,input.shape[-1]])
    length_of_input_safe__b_1EXPANDdim = length_of_input_safe__b_1EXPANDdim.to(dtype=input.dtype)
    #assert length_of_input_safe__b_1EXPANDdim.shape == torch.Size([__bat, __dim])
    normalized_vector__b_d = input.div(length_of_input_safe__b_1EXPANDdim)
    #assert length_of_input_safe__b_1EXPANDdim.shape == torch.Size([__bat, __dim])
    if keepdim_for_length:
        return normalized_vector__b_d, length_of_input__b_1, sqr_length_of_input__b_1
    return     normalized_vector__b_d, length_of_input__b_1.squeeze(dim=-1), sqr_length_of_input__b_1.squeeze(dim=-1)
    #end of function
if '''some basic test.''' and __DEBUG_ME__() and False:
    def ____test____get_full_info_of_vector_length__2d():
        
        #this test func is a raw combination of the 2 above.
        input = torch.tensor([[0.,0.],[0.,1.],[1.,1.]])
        normalized_vector, length_of_input, sqr_length_of_input = get_full_info_of_vector_length__2d(input)
        assert _tensor_equal(normalized_vector, [[0.,0.],[0.,1.],[0.7,0.7]], 0.05)
        assert normalized_vector.dtype == torch.float32
        assert _tensor_equal(get_vector_length(normalized_vector), [0., 1., 1.], 0.001)
        assert _tensor_equal(length_of_input, [0., 1., 1.414], 0.001)
        assert length_of_input.dtype == torch.float64
        assert _tensor_equal(sqr_length_of_input, [0., 1., 2.], 0.001)
        assert sqr_length_of_input.dtype == torch.float64

        _vector_len = normalized_vector.mul(normalized_vector).sum(dim=1)
        assert _tensor_equal(_vector_len[0 ], torch.zeros_like(_vector_len[0 ]), 0.05)
        assert _tensor_equal(_vector_len[1:], torch.ones_like (_vector_len[1:]), 0.05)
        
        input = torch.tensor([[  1.,   1],
                                [0.1,  0.1]])
        transpost_of_normalized_vector, length_of_input, sqr_length_of_input = get_full_info_of_vector_length__2d(input.T)
        normalized_vector = transpost_of_normalized_vector.T
        assert _tensor_equal(normalized_vector,[[0.9950, 0.9950],
                                                [0.0995, 0.0995]], epsilon=0.001)
        
        input = torch.tensor([[1.,1],[1,2]])
        normalized_vector, length_of_input, sqr_length_of_input = get_full_info_of_vector_length__2d(input)
        assert length_of_input.shape == torch.Size([2])
        assert _tensor_equal(length_of_input, [1.4142,2.2361])
        assert sqr_length_of_input.shape == torch.Size([2])
        assert _tensor_equal(sqr_length_of_input, [2., 5.])


        input = torch.tensor([[0.71, 0.71],[1,0]])
        normalized_vector, length_of_input, sqr_length_of_input = get_full_info_of_vector_length__2d(input)
        assert length_of_input.shape == torch.Size([2])
        assert _tensor_equal(length_of_input, [1., 1], epsilon=0.01)
        assert sqr_length_of_input.shape == torch.Size([2])
        assert _tensor_equal(sqr_length_of_input, [1., 1], epsilon=0.01)

        input = torch.tensor([[  0.71, 1],
                                [0.71, 0]])
        normalized_vector, length_of_input, sqr_length_of_input = get_full_info_of_vector_length__2d(input.T)
        assert length_of_input.shape == torch.Size([2])
        assert _tensor_equal(length_of_input, [1., 1], epsilon=0.01)
        assert sqr_length_of_input.shape == torch.Size([2])
        assert _tensor_equal(sqr_length_of_input, [1., 1], epsilon=0.01)

        "dtype"
        input = torch.tensor([[1.],[1]])
        normalized_vector, length_of_input, sqr_length_of_input = \
                get_full_info_of_vector_length__2d(input, length_result_dtype=torch.float16)
        assert length_of_input.dtype == torch.float16
        assert sqr_length_of_input.dtype == torch.float16

        "keep dim"
        input = torch.tensor([[1.],[1]])
        normalized_vector, length_of_input, sqr_length_of_input = \
                get_full_info_of_vector_length__2d(input, keepdim_for_length=True)
        assert length_of_input.shape.__len__() == 2
        assert length_of_input.shape == torch.Size([2,1])
        assert sqr_length_of_input.shape.__len__() == 2
        assert sqr_length_of_input.shape == torch.Size([2,1])

        return 
    
    ____test____get_full_info_of_vector_length__2d()
    pass



"vector angle"

def are_parallel(tensor1:torch.Tensor, tensor2:torch.Tensor, 
                epsilon: float | torch.Tensor = 0.0001, dtype_inner: torch.dtype = torch.float64)->torch.Tensor:
    '''Notice, the 0 vector is not parallel to any vector in this function.'''
    tensor1_standardized = vector_length_norm(tensor1, epsilon = epsilon, dtype_inner = dtype_inner)
    tensor2_standardized = vector_length_norm(tensor2, epsilon = epsilon, dtype_inner = dtype_inner)
    the_dot_prod = tensor1_standardized.mul(tensor2_standardized).sum(dim=-1)
    #this is a _tensor_equal, but a bit different.
    with torch.inference_mode():
        diff = the_dot_prod.abs()-1.
        abs_of_diff = diff.abs()
        result = abs_of_diff.lt(epsilon)
        return result
    #end of function
if "test" and __DEBUG_ME__() and False:
    def ____basic_behavior____are_parallel():
        if "no batch" and True:
            for dim in [2,5,10,100,1000]:
                for _ in range (321):
                    tensor1 = torch.randn(size=[1,dim])
                    tensor2 = tensor1*(random.random()+0.1)*3.
                    if random.random()>0.5:
                        tensor2 = -tensor2
                        pass
                    result_tensor = are_parallel(tensor1, tensor2)
                    assert isinstance(result_tensor, torch.Tensor)
                    assert result_tensor.shape == torch.Size([1])
                    assert result_tensor.all()  
                    pass
                pass#for dim
            pass#/ test

        if "batch behavior" and True:
            for batch in [2, 10]:
                for dim in [3, 7]:
                    #init
                    tensor1 = torch.randn(size=[batch, dim])
                    expected_answer__b_1 = torch.randint(low=0, high=2, size=[batch, 1])
                    expected_answer__b_1 = expected_answer__b_1.to(torch.bool)
                    expected_answer__b_d = expected_answer__b_1.expand(size=[-1, dim])
                    assert expected_answer__b_d.shape == tensor1.shape
                    _temp__tensor1_randomly_scaled = tensor1*(torch.randn(size=[batch, 1]).expand(size=[-1, dim]))
                    tensor2 = torch.where(expected_answer__b_d, _temp__tensor1_randomly_scaled, torch.randn(size=[batch, dim]))
                    #call
                    answer = are_parallel(tensor1, tensor2)
                    #assert
                    assert answer.eq(expected_answer__b_1.reshape([-1])).all()
                    for ii_batch in range(batch):
                        this_tensor1 = tensor1[ii_batch].reshape([1, dim])
                        this_tensor2 = tensor2[ii_batch].reshape([1, dim])
                        this_answer = are_parallel(this_tensor1, this_tensor2)
                        assert this_answer == answer[ii_batch]
                        pass#for ii
                    pass#for dim
                pass#for batch
            pass#/ test
        return
    ____basic_behavior____are_parallel()
    pass

def are_orthogonal(tensor1:torch.Tensor, tensor2:torch.Tensor, 
                epsilon: float | torch.Tensor | None = None)->torch.Tensor:
    
    if epsilon is None:
        # The formula was from a test. See below. 
        epsilon = torch.tensor(1e-5* math.sqrt(tensor1.shape[-1]), device=tensor1.device, dtype=tensor1.dtype)
        pass

    the_dot_prod = tensor1.mul(tensor2).sum(dim=-1)
    #this is a _tensor_equal, but a bit different.
    with torch.inference_mode():
        result = the_dot_prod.abs().lt(epsilon)
        return result
    #end of function
if "algo test" and __DEBUG_ME__() and False:
    '''
    conclusion. The normal case is always safer than the extreme case.
    '''
    def ____algo_test____are_orthogonal():
        # err_max_log = [-5.623, -5.146, -4.489, -3.816]
        if "are_orthogonal algo test  the n-1 style" and False:
            if True:
                #result
                # err_max_log = [-5.623, -5.146, -4.489, -3.816]
                # let's use        -4.5    -4      -3.5    -3
                # ref_max_log = [-4.500, -4.000, -3.500, -3.000]
                #                   1       2       3       4
                
                pass
            
            print(f"{_line_()}    are_orthogonal algo test by dim")
            
            error_max = []#don't modify this
            error_avg = []#don't modify this
            error_max_log10 = []#don't modify this
            error_avg_log10 = []#don't modify this
            epsilon_max = []#don't modify this
            epsilon_max_log10 = []#don't modify this
            #------------------#------------------#------------------
            dim_list =                          [   10,    100,  1000, 10000]
            number_of_tests_list = torch.tensor([100000, 50000, 20000, 10000])
            number_of_tests_list = number_of_tests_list.mul(0.1).to(torch.int32)
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                epsilon_max.append(1e-5* math.sqrt(dim))
                epsilon_max_log10.append(math.log10(1e-5* math.sqrt(dim)))
                # iota_of_dim = iota(dim)
                number_of_tests = number_of_tests_list[outter_param_set]
                device = 'cpu'
                if dim>100:
                    device = 'cuda'
                    pass
                print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------

                _raw_result__error = torch.empty(size=[number_of_tests])
                _when_start = time.perf_counter()
                
                for ii__test in range(number_of_tests):
                    
                    #------------------#------------------#------------------
                    #<  init
                    tensor1 = torch.randn(size=[1,dim])
                    tensor1[0,0] = -1.
                    tensor2 = torch.randn(size=[1,dim])
                    tensor2[0,0] = 0.
                    _temp__dot_prod___1 = tensor1.mul(tensor2).sum(dim=-1)
                    tensor2[0,0] = _temp__dot_prod___1[0]
                    the_dot_prod___1 = tensor1.mul(tensor2).sum(dim=-1)

                    #<  measure
                    _raw_result__error[ii__test] = the_dot_prod___1.abs().item()
                    #------------------#------------------#------------------
                    pass#for ii__test
                _when_end = time.perf_counter()
                
                error_max.append(_raw_result__error.max().item())
                error_avg.append(_raw_result__error.mean().item())
                error_max_log10.append(_raw_result__error.max().log10().item())
                error_avg_log10.append(_raw_result__error.mean().log10().item())
                pass#for outter_param_set
            
            print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
            #print(f"error_max   = {str_the_list(error_max  , 6)}")
            #print(f"error_avg   = {str_the_list(error_avg  , 6)}")
            print(f"err_max_log = {str_the_list(error_max_log10  , 3)}")
            print(f"err_avg_log = {str_the_list(error_avg_log10  , 3)}")
            print(f"epsi_max    = {str_the_list(epsilon_max        , 3)}")
            print(f"epsi_max_log = {str_the_list(epsilon_max_log10  , 3)}")
            print(f"dim        = {str_the_list(dim_list, 0, ",    ")}")
            
            pass#/ test
        
        # err_max_log = [-6.225, -5.543, -5.176, -4.589]
        # err_avg_log = [-7.256, -6.366, -5.653, -5.037]
        if "are_orthogonal algo test  the swap style" and True:
            if True:
                #result
                # err_max_log = [-6.225, -5.543, -5.176, -4.589]
                # err_avg_log = [-7.256, -6.366, -5.653, -5.037]
                # epsi_max    = [ 0.000,  0.000,  0.000,  0.001]
                # epsi_max_log = [-4.500, -4.000, -3.500, -3.000]
                # dim        = [ 10,     100,     1000,     10000]
                pass
            
            print(f"{_line_()}    are_orthogonal algo test by dim")
            
            error_max = []#don't modify this
            error_avg = []#don't modify this
            error_max_log10 = []#don't modify this
            error_avg_log10 = []#don't modify this
            epsilon_max = []#don't modify this
            epsilon_max_log10 = []#don't modify this
            #------------------#------------------#------------------
            dim_list =                          [  10,  100, 1000, 10000]
            number_of_tests_list = torch.tensor([2000, 1000,  100,   50])
            number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                epsilon_max.append(1e-5* math.sqrt(dim))
                epsilon_max_log10.append(math.log10(1e-5* math.sqrt(dim)))
                # iota_of_dim = iota(dim)
                number_of_tests = number_of_tests_list[outter_param_set]
                device = 'cpu'
                # if dim>100:
                #     device = 'cuda'
                #     pass
                print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------

                _raw_result__error = torch.empty(size=[number_of_tests])
                _when_start = time.perf_counter()
                
                for ii__test in range(number_of_tests):
                    
                    #------------------#------------------#------------------
                    #<  init
                    tensor1 = torch.randn(size=[1,dim])
                    tensor2 = tensor1.detach().clone()
                    #swap elements
                    _buffer = list(range(dim//2*2))
                    assert _buffer.__len__() >= dim - 1 and _buffer.__len__() <= dim
                    random.shuffle(_buffer)

                    for ii in range(0, _buffer.__len__(), 2):
                        rand_index_1 = _buffer[ii]    
                        rand_index_2 = _buffer[ii + 1]
                        tensor2[:, rand_index_1] =  tensor1[:, rand_index_2].detach().clone()
                        tensor2[:, rand_index_2] = -tensor1[:, rand_index_1].detach().clone()
                        pass
                    
                    #<  calc
                    the_dot_prod___1 = tensor1.mul(tensor2).sum(dim=-1)
                    #<  measure
                    _raw_result__error[ii__test] = the_dot_prod___1.abs().item()
                    #------------------#------------------#------------------
                    pass#for ii__test
                _when_end = time.perf_counter()
                
                error_max.append(_raw_result__error.max().item())
                error_avg.append(_raw_result__error.mean().item())
                error_max_log10.append(_raw_result__error.max().log10().item())
                error_avg_log10.append(_raw_result__error.mean().log10().item())
                pass#for outter_param_set
            
            print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
            #print(f"error_max   = {str_the_list(error_max  , 6)}")
            #print(f"error_avg   = {str_the_list(error_avg  , 6)}")
            print(f"err_max_log = {str_the_list(error_max_log10  , 3)}")
            print(f"err_avg_log = {str_the_list(error_avg_log10  , 3)}")
            print(f"epsi_max    = {str_the_list(epsilon_max        , 3)}")
            print(f"epsi_max_log = {str_the_list(epsilon_max_log10  , 3)}")
            print(f"dim        = {str_the_list(dim_list, 0, ",    ")}")
            
            pass#/ test
        
        return
    ____algo_test____are_orthogonal()
    pass

if "test" and __DEBUG_ME__() and False:
    def ____basic_behavior____are_orthogonal():
        if "basic" and True:
            '''[?, 0] and [0, ?] are orthogonal.'''
            for _ in range(33):
                tensor1 = torch.tensor([[random.random(), 0]])
                tensor2 = torch.tensor([[0.,                random.random()]])
                result_tensor = are_orthogonal(tensor1, tensor2)
                assert isinstance(result_tensor, torch.Tensor)
                assert result_tensor.shape == torch.Size([1])
                assert result_tensor.all()
                pass#for _
            
            '''[a, b] and [-b, a] are orthogonal.'''
            for _ in range(33):
                tensor1 = torch.tensor([[1., 0]])
                tensor2 = torch.empty_like(tensor1)
                tensor2[0,0] =  tensor1[0,1]
                tensor2[0,1] = -tensor1[0,0]
                result_tensor = are_orthogonal(tensor1, tensor2)
                assert result_tensor.all()
                pass#for _
            pass#/ test

        if "no batch, the n-1 style." and True:
            for dim in [2,5,10,100,1000]:
                for _ in range (321):
                    tensor1 = torch.randn(size=[1,dim])
                    tensor1[0,0] = -1.
                    tensor2 = torch.randn(size=[1,dim])
                    tensor2[0,0] = 0.
                    _temp__dot_prod___1 = tensor1.mul(tensor2).sum(dim=-1)
                    tensor2[0,0] = _temp__dot_prod___1[0]
                    #the prod now should be 0.
                    assert _tensor_equal(tensor1.mul(tensor2).sum(dim=-1), [0.], epsilon=0.00001*math.sqrt(dim))

                    _scale_factor = random.random()*3.+0.1
                    if random.random()>0.5:
                        _scale_factor = -_scale_factor#maybe different direction.
                        pass
                    tensor1 *= _scale_factor
                    _scale_factor = random.random()*3.+0.1
                    if random.random()>0.5:
                        _scale_factor = -_scale_factor#maybe different direction.
                        pass
                    tensor2 *= _scale_factor
                    #the prod now should still be 0.
                    assert _tensor_equal(tensor1.mul(tensor2).sum(dim=-1), [0.], epsilon=0.00001*math.sqrt(dim))

                    result_tensor = are_orthogonal(tensor1, tensor2)#should always be true .
                    assert isinstance(result_tensor, torch.Tensor)
                    assert result_tensor.shape == torch.Size([1])
                    assert result_tensor.all()  
                    pass
                pass#for dim
            pass#/ test

        if "batch behavior" and True:
            for batch in [2, 10]:
                for dim in [4, 12]:
                    for _ in range (33):
                        #init
                        tensor1 = torch.randn(size=[batch, dim])
                        _temp__tensor2_if_all_true = tensor1.detach().clone()
                        expected_answer__b_1 = torch.randint(low=0, high=2, size=[batch, 1])
                        expected_answer__b_1 = expected_answer__b_1.to(torch.bool)
                        expected_answer__b_d = expected_answer__b_1.expand(size=[-1, dim])
                        assert expected_answer__b_d.shape == tensor1.shape


                        #swap elements
                        assert dim%2 == 0
                        _buffer = list(range(dim//2*2))
                        assert _buffer.__len__() >= dim - 1 and _buffer.__len__() <= dim
                        random.shuffle(_buffer)

                        for ii in range(0, _buffer.__len__(), 2):
                            rand_index_1 = _buffer[ii]    
                            rand_index_2 = _buffer[ii + 1]
                            _temp__tensor2_if_all_true[:, rand_index_1] =  tensor1[:, rand_index_2].detach().clone()
                            _temp__tensor2_if_all_true[:, rand_index_2] = -tensor1[:, rand_index_1].detach().clone()
                            pass

                        tensor2 = torch.where(expected_answer__b_d, _temp__tensor2_if_all_true, torch.randn(size=[batch, dim]))

                        #scale a bit
                        _scale_factor = random.random()*3.+0.1
                        if random.random()>0.5:
                            _scale_factor = -_scale_factor#maybe different direction.
                            pass
                        tensor1 *= _scale_factor
                        _scale_factor = random.random()*3.+0.1
                        if random.random()>0.5:
                            _scale_factor = -_scale_factor#maybe different direction.
                            pass
                        tensor2 *= _scale_factor

                        #call
                        answer = are_orthogonal(tensor1, tensor2)
                        #assert
                        if not answer.eq(expected_answer__b_1.reshape([-1])).all():
                            print(f"tensor1 = torch.{tensor1}")
                            print(f"tensor2 = torch.{tensor2}")
                            print(f"answer = {answer}")
                            print(f"expected_answer__b_1 = {expected_answer__b_1.reshape([-1])}")
                            pass

                        assert answer.eq(expected_answer__b_1.reshape([-1])).all()
                        for ii_batch in range(batch):#every data inside this batch.
                            this_tensor1 = tensor1[ii_batch].reshape([1, dim])
                            this_tensor2 = tensor2[ii_batch].reshape([1, dim])
                            this_answer = are_orthogonal(this_tensor1, this_tensor2)
                            assert this_answer == answer[ii_batch]
                            pass#for ii
                        pass#for _
                    pass#for dim
                pass#for batch
            pass#/ test

        return
    ____basic_behavior____are_orthogonal()
    pass



"vector projection"

def standardized_vector_proj(input:torch.Tensor, proj_to:torch.Tensor, 
                needs_error:bool, 
                )->tuple[torch.Tensor, torch.Tensor|None]:
    '''return standard_proj_vec, standard_error_vec'''
    the_cos = input.mul(proj_to).sum(dim=-1, keepdim=True)
    standard_proj_vec = proj_to.mul(the_cos.expand_as(proj_to))
    standard_error_vec = None
    if needs_error:
        standard_error_vec = input - standard_proj_vec
        pass
    return standard_proj_vec, standard_error_vec
if "test" and __DEBUG_ME__() and False:
    def ____basic_behavior_test____standardized_vector_proj():
        if "basic" and True:
            input = vector_length_norm(torch.tensor([[1., 1]]))
            proj_to = vector_length_norm(torch.tensor([[1., 0]]))
            result_tuple = standardized_vector_proj(input, proj_to, needs_error=True)
            assert _tensor_equal(result_tuple[0], torch.tensor([[0.7071, 0]]))
            assert _tensor_equal(result_tuple[1], torch.tensor([[0,      0.7071]]))
            
            input = vector_length_norm(torch.tensor([[1., 0]]))
            proj_to = vector_length_norm(torch.tensor([[1., 1]]))
            result_tuple = standardized_vector_proj(input, proj_to, needs_error=True)
            assert _tensor_equal(result_tuple[0], torch.tensor([[0.5,  0.5]]))
            assert _tensor_equal(result_tuple[1], torch.tensor([[0.5, -0.5]]))

            '''some geometric check.'''
            for dim in [2, 10, 100]:
                for _ in range(33):
                    input = vector_length_norm(torch.randn(size=[1, dim]))
                    proj_to = vector_length_norm(torch.randn(size=[1, dim]))
                    standard_proj_vec, standard_error_vec = standardized_vector_proj(input, proj_to, needs_error=True)
                    
                    #<  is projected_vector the same direction with proj_to??
                    _temp_result = are_parallel(standard_proj_vec, proj_to)
                    assert _temp_result.all()
                    #<  is error_vector orthogonal to the proj_to
                    _temp_result = are_orthogonal(standard_error_vec, proj_to)
                    assert _temp_result.all()
                    pass# for _
                pass#for dim

            pass#/ test
        
        if "does batch works the same as non-batched?" and True:
            for dim in [2, 10, 100]:
                for batch in [3, 15, 88]:
                    for _ in range(33):
                        input = vector_length_norm(torch.randn(size=[batch, dim]))
                        proj_to = vector_length_norm(torch.randn(size=[batch, dim]))
                        standard_proj_vec, standard_error_vec = standardized_vector_proj(input, proj_to, needs_error=True)
                        
                        standard_proj_vec__manually = torch.empty(size=[batch, dim])
                        standard_error_vec__manually = torch.empty(size=[batch, dim])
                        for ii in range(batch):
                            result_tuple = standardized_vector_proj(input[ii], proj_to[ii], needs_error=True)
                            standard_proj_vec__manually[ii] = result_tuple[0]
                            standard_error_vec__manually[ii] = result_tuple[1]
                            pass
                        assert _tensor_equal(standard_proj_vec, standard_proj_vec__manually)
                        assert _tensor_equal(standard_error_vec, standard_error_vec__manually)
                        pass# for _
                    pass#for batch
                pass#for dim

            pass#/ test
        

        #if "visualization" and True:





        return 
    
    ____basic_behavior_test____standardized_vector_proj()
    pass

'''the formula is a bit weird. Although it works.'''
def vector_proj__full_info(input:torch.Tensor, proj_to:torch.Tensor, 
                input__already_scaled_to_1 = False,
                proj_to__already_scaled_to_1 = False,
                needs_proj = True,
                needs_error = False, 
                needs_standard_error = False, 
                )->tuple[torch.Tensor|None, torch.Tensor|None, tuple[torch.Tensor, torch.Tensor|None]]:
    '''return proj_vec, error_vec, (standardized_proj_vec, standardized_error_vec)
    
    But if you don't need some* of them, then some** of them are not calculated, then they are None.'''
    if input__already_scaled_to_1:
        input__standardized = input
        length_of_input = torch.ones(size=[input.shape[0], 1])
        pass
    else:
        input__standardized, length_of_input = \
                get_full_info_of_vector_length__2d(input, keepdim_for_length=True)
        #assert length_of_input.shape.__len__() == 2
        pass
    if proj_to__already_scaled_to_1:
        proj_to__standardized = proj_to
        pass
    else:
        proj_to__standardized = vector_length_norm(proj_to)
        pass
    
    standard_result_tuple = standardized_vector_proj(input__standardized, proj_to__standardized, 
                                        needs_error=needs_error or needs_standard_error)    
    # results:
    if needs_proj:
        proj_vec = standard_result_tuple[0].mul(length_of_input)
        pass
    else:
        proj_vec = None
        pass
    if needs_error:
        error_vec = standard_result_tuple[1].mul(length_of_input)
        pass
    else:
        error_vec = None
        pass
    # return
    return proj_vec, error_vec, standard_result_tuple
    # end of function

if "test" and __DEBUG_ME__() and False:  
    #assert False, "change to the real math formula."
    def ____basic_behavior_test____vector_proj__full_info():
        if "basic" and True:
            input = torch.tensor([[1., 1]])
            proj_to = torch.tensor([[1., 0]])
            result_tuple = vector_proj__full_info(input, proj_to, needs_error=True)
            assert _tensor_equal(result_tuple[0], torch.tensor([[1., 0]]))
            assert _tensor_equal(result_tuple[1], torch.tensor([[0,  1.]]))
            assert _tensor_equal(get_vector_length(result_tuple[2][0]), torch.tensor([0.7071]))
            assert _tensor_equal(get_vector_length(result_tuple[2][1]), torch.tensor([0.7071]))
            assert _tensor_equal(result_tuple[2][0], torch.tensor([[0.7071, 0]]))
            assert _tensor_equal(result_tuple[2][1], torch.tensor([[0,      0.7071]]))
            
            input = torch.tensor([[1., 0]])
            proj_to = torch.tensor([[1., 1]])
            result_tuple = vector_proj__full_info(input, proj_to, needs_error=True)
            assert _tensor_equal(result_tuple[0], torch.tensor([[0.5,  0.5]]))
            assert _tensor_equal(result_tuple[1], torch.tensor([[0.5, -0.5]]))
            assert _tensor_equal(get_vector_length(result_tuple[2][0]), torch.tensor([0.7071]))
            assert _tensor_equal(get_vector_length(result_tuple[2][1]), torch.tensor([0.7071]))
            assert _tensor_equal(result_tuple[2][0], torch.tensor([[0.5,  0.5]]))
            assert _tensor_equal(result_tuple[2][1], torch.tensor([[0.5, -0.5]]))
            
            input = torch.tensor([[2., 0]])
            proj_to = torch.tensor([[1., 1]])
            result_tuple = vector_proj__full_info(input, proj_to, needs_error=True)
            assert _tensor_equal(result_tuple[0], torch.tensor([[1.,  1]]))
            assert _tensor_equal(result_tuple[1], torch.tensor([[1., -1]]))
            assert _tensor_equal(get_vector_length(result_tuple[2][0]), torch.tensor([0.7071]))
            assert _tensor_equal(get_vector_length(result_tuple[2][1]), torch.tensor([0.7071]))
            assert _tensor_equal(result_tuple[2][0], torch.tensor([[0.5,  0.5]]))
            assert _tensor_equal(result_tuple[2][1], torch.tensor([[0.5, -0.5]]))

            '''some geometric check.    direction???'''
            for dim in [2, 10, 100]:
                for _ in range(33):
                    input = torch.randn(size=[1, dim])
                    proj_to = torch.randn(size=[1, dim])
                    proj_vec, error_vec, (standardized_proj_vec, standardized_error_vec) = vector_proj__full_info(input, proj_to, needs_error=True)
                    
                    #<  is projected_vector the same direction with proj_to??
                    _temp_result = are_parallel(proj_vec, proj_to)
                    assert _temp_result.all()
                    _temp_result = are_parallel(standardized_proj_vec, proj_to)
                    assert _temp_result.all()

                    #<  is error_vector orthogonal to the proj_to
                    _temp_result = are_orthogonal(error_vec, proj_to)
                    assert _temp_result.all()
                    _temp_result = are_orthogonal(standardized_error_vec, proj_to)
                    assert _temp_result.all()
                    pass# for _
                pass#for dim

            '''some geometric check.    when the length of input changes???'''
            for dim in [2, 10, 100]:
                for _ in range(33):
                    proj_to = torch.randn(size=[1, dim])

                    ori_input = torch.randn(size=[1, dim])
                    ori_proj_vec, ori_error_vec, (ori_standardized_proj_vec, ori_standardized_error_vec)= \
                            vector_proj__full_info(ori_input, proj_to, needs_error=True)

                    length_factor = (random.random()+0.1)*3
                    new_input = ori_input*length_factor#scale the input a bit.
                    new_proj_vec, new_error_vec, (new_standardized_proj_vec, new_standardized_error_vec) = \
                            vector_proj__full_info(new_input, proj_to, needs_error=True)

                    assert _tensor_equal(ori_proj_vec*length_factor,  new_proj_vec)
                    assert _tensor_equal(ori_error_vec*length_factor, new_error_vec)
                    assert _tensor_equal(ori_standardized_proj_vec, new_standardized_proj_vec)
                    assert _tensor_equal(ori_standardized_error_vec, new_standardized_error_vec)

                    pass# for _
                pass# for dim
            pass#/ test
        
        if "does batch works the same as non-batched?" and True:
            for dim in [2, 10, 30]:
                for batch in [3, 15, 44]:
                    for _ in range(33):
                        input = torch.randn(size=[batch, dim])
                        proj_to = torch.randn(size=[batch, dim])
                        proj_vec, error_vec, (standard_proj_vec, standard_error_vec) = \
                                vector_proj__full_info(input, proj_to, needs_error=True)
                        
                        proj_vec__manually = torch.empty(size=[batch, dim])
                        error_vec__manually = torch.empty(size=[batch, dim])
                        standard_proj_vec__manually = torch.empty(size=[batch, dim])
                        standard_error_vec__manually = torch.empty(size=[batch, dim])
                        for ii in range(batch):
                            result_tuple = vector_proj__full_info(input[ii].reshape([-1, dim]), proj_to[ii].reshape([-1, dim]), needs_error=True)
                            proj_vec__manually[ii] = result_tuple[0]
                            error_vec__manually[ii] = result_tuple[1]
                            standard_proj_vec__manually[ii] = result_tuple[2][0]
                            standard_error_vec__manually[ii] = result_tuple[2][1]
                            pass
                        assert _tensor_equal(proj_vec,  proj_vec__manually)
                        assert _tensor_equal(error_vec, error_vec__manually)
                        assert _tensor_equal(standard_proj_vec,  standard_proj_vec__manually)
                        assert _tensor_equal(standard_error_vec, standard_error_vec__manually)
                        pass# for _
                    pass#for batch
                pass#for dim

            pass#/ test

        if "return style" and True:
            input = torch.tensor([[1., 1]])
            proj_to = torch.tensor([[1., 0]])
            result_tuple = vector_proj__full_info(input, proj_to, needs_proj=True,  needs_error=True,  needs_standard_error=True)
            assert result_tuple[0]    is not None # proj
            assert result_tuple[1]    is not None # error
            assert result_tuple[2][0] is not None # std proj (always there)
            assert result_tuple[2][1] is not None # std error
            result_tuple = vector_proj__full_info(input, proj_to, needs_proj=False, needs_error=True,  needs_standard_error=True)
            assert result_tuple[0]    is     None # proj
            assert result_tuple[1]    is not None # error
            assert result_tuple[2][0] is not None # std proj (always there)
            assert result_tuple[2][1] is not None # std error
            result_tuple = vector_proj__full_info(input, proj_to, needs_proj=True,  needs_error=False, needs_standard_error=True)
            assert result_tuple[0]    is not None # proj
            assert result_tuple[1]    is     None # error
            assert result_tuple[2][0] is not None # std proj (always there)
            assert result_tuple[2][1] is not None # std error
            result_tuple = vector_proj__full_info(input, proj_to, needs_proj=False, needs_error=False, needs_standard_error=True)
            assert result_tuple[0]    is     None # proj
            assert result_tuple[1]    is     None # error
            assert result_tuple[2][0] is not None # std proj (always there)
            assert result_tuple[2][1] is not None # std error
            result_tuple = vector_proj__full_info(input, proj_to, needs_proj=True,  needs_error=True,  needs_standard_error=False)
            assert result_tuple[0]    is not None # proj
            assert result_tuple[1]    is not None # error
            assert result_tuple[2][0] is not None # std proj (always there)
            assert result_tuple[2][1] is not None # std error
            result_tuple = vector_proj__full_info(input, proj_to, needs_proj=False, needs_error=True,  needs_standard_error=False)
            assert result_tuple[0]    is     None # proj
            assert result_tuple[1]    is not None # error
            assert result_tuple[2][0] is not None # std proj (always there)
            assert result_tuple[2][1] is not None # std error
            result_tuple = vector_proj__full_info(input, proj_to, needs_proj=True,  needs_error=False, needs_standard_error=False)
            assert result_tuple[0]    is not None # proj
            assert result_tuple[1]    is     None # error
            assert result_tuple[2][0] is not None # std proj (always there)
            assert result_tuple[2][1] is     None # std error
            result_tuple = vector_proj__full_info(input, proj_to, needs_proj=False, needs_error=False, needs_standard_error=False)
            assert result_tuple[0]    is     None # proj
            assert result_tuple[1]    is     None # error
            assert result_tuple[2][0] is not None # std proj (always there)
            assert result_tuple[2][1] is     None # std error
            pass #/ test


        if "what happens if the already_scaled flags are set wrongly?" and True:
            input = torch.tensor([[2., 0]])
            proj_to = torch.tensor([[1., 1]])
            proj_vec, error_vec, (standardized_proj_vec, standardized_error_vec) = \
                vector_proj__full_info(input, proj_to, needs_error=True)
            assert _tensor_equal(proj_vec, torch.tensor([[1.,  1]]))
            assert _tensor_equal(error_vec, torch.tensor([[1., -1]]))
            assert _tensor_equal(get_vector_length(standardized_proj_vec), torch.tensor([0.7071]))
            assert _tensor_equal(                  standardized_proj_vec,  torch.tensor([[0.5,  0.5]]))
            assert _tensor_equal(get_vector_length(standardized_error_vec), torch.tensor([0.7071]))
            assert _tensor_equal(                  standardized_error_vec,  torch.tensor([[0.5, -0.5]]))

            fake_input__result_tuple = vector_proj__full_info(input, proj_to, needs_error=True, \
                        input__already_scaled_to_1 = True)
            assert _tensor_equal(fake_input__result_tuple[0], proj_vec)
            assert _tensor_equal(fake_input__result_tuple[1], error_vec)
            assert not _tensor_equal(fake_input__result_tuple[2][0], standardized_proj_vec)
            assert not _tensor_equal(fake_input__result_tuple[2][1], standardized_error_vec)    

            fake_proj_to__result_tuple = vector_proj__full_info(input, proj_to, needs_error=True, \
                        proj_to__already_scaled_to_1 = True)
            assert not _tensor_equal(fake_proj_to__result_tuple[0], proj_vec)
            assert not _tensor_equal(fake_proj_to__result_tuple[1], error_vec)
            assert not _tensor_equal(fake_proj_to__result_tuple[2][0], standardized_proj_vec)
            assert not _tensor_equal(fake_proj_to__result_tuple[2][1], standardized_error_vec)

            fake_all__result_tuple = vector_proj__full_info(input, proj_to, needs_error=True, \
                        input__already_scaled_to_1 = True, \
                        proj_to__already_scaled_to_1 = True)
            assert not _tensor_equal(fake_all__result_tuple[0], proj_vec)
            assert not _tensor_equal(fake_all__result_tuple[1], error_vec)
            assert not _tensor_equal(fake_all__result_tuple[2][0], standardized_proj_vec)
            assert not _tensor_equal(fake_all__result_tuple[2][1], standardized_error_vec)


            pass

        return 
    
    ____basic_behavior_test____vector_proj__full_info()
    pass


'''this is the proj == (a.b)/(b.b) * b version.'''
def vector_proj(input:torch.Tensor, proj_to:torch.Tensor, 
                needs_error = False, proj_to__already_normalized = False
                )->tuple[torch.Tensor, torch.Tensor|None]:
    '''return proj_vec, error_vec

    if needs_error is False, then error_vec is None.
    
    This is the proj == (a.b)/(b.b) * b version. LINEAR ALGEBRA!!!'''

    #input is b_1, proj_to is b_1.
    a_dot_b___b_1 = input.mul(proj_to).sum(dim=-1, keepdim=True)
    if proj_to__already_normalized :
        proj_vec = proj_to.mul(a_dot_b___b_1.expand_as(proj_to))
        pass
    else:
        b_dot_b___b_1 = proj_to.mul(proj_to).sum(dim=-1, keepdim=True)
        t___b_1 = a_dot_b___b_1.div(b_dot_b___b_1)
        proj_vec = proj_to.mul(t___b_1.expand_as(proj_to))
        pass
    if needs_error:
        error_vec = input - proj_vec
        pass
    else:
        error_vec = None
        pass

    return proj_vec, error_vec
if "test" and __DEBUG_ME__() and True:
    def ____basic_behavior_test____vector_proj():
        if "basic" and True:
            input = torch.tensor([[1., 1]])
            proj_to = torch.tensor([[1., 0]])
            proj_vec, error_vec = vector_proj(input, proj_to, needs_error=True)
            assert _tensor_equal(proj_vec, torch.tensor([[1., 0]]))
            assert _tensor_equal(error_vec, torch.tensor([[0,  1.]]))
            
            input = torch.tensor([[1., 0]])
            proj_to = torch.tensor([[1., 1]])
            proj_vec, error_vec = vector_proj(input, proj_to, needs_error=True)
            assert _tensor_equal(proj_vec, torch.tensor([[0.5,  0.5]]))
            assert _tensor_equal(error_vec, torch.tensor([[0.5, -0.5]]))
            
            input = torch.tensor([[2., 0]])
            proj_to = torch.tensor([[1., 1]])
            proj_vec, error_vec = vector_proj(input, proj_to, needs_error=True)
            assert _tensor_equal(proj_vec, torch.tensor([[1.,  1]]))
            assert _tensor_equal(error_vec, torch.tensor([[1., -1]]))

            input = torch.tensor([[2., 0]])
            proj_to = vector_length_norm(torch.tensor([[1., 1]]))
            proj_vec, error_vec = vector_proj(input, proj_to, needs_error=True, proj_to__already_normalized=True)
            assert _tensor_equal(proj_vec, torch.tensor([[1.,  1]]))
            assert _tensor_equal(error_vec, torch.tensor([[1., -1]]))


            '''proj_to__already_normalized'''
            for batch in [3,11,101]:
                for dim in [2, 10, 100]:
                    for _ in range(33):

                        input = torch.randn(size=[batch, dim])
                        proj_to = torch.randn(size=[batch, dim])
                        without_the_flag__proj_vec, without_the_flag__error_vec = vector_proj(input,                    proj_to,  needs_error=True)
                        with_the_flag__proj_vec,    with_the_flag__error_vec    = vector_proj(input, vector_length_norm(proj_to), needs_error=True, \
                                                                                        proj_to__already_normalized=True)
                        assert _tensor_equal(without_the_flag__proj_vec,  with_the_flag__proj_vec)
                        assert _tensor_equal(without_the_flag__error_vec, with_the_flag__error_vec)
                        pass# for _
                    pass#for dim
                pass#for batch


            '''some geometric check.    direction???'''
            for dim in [2, 10, 100]:
                for _ in range(33):
                    input = torch.randn(size=[1, dim])
                    proj_to = torch.randn(size=[1, dim])
                    proj_vec, error_vec = vector_proj(input, proj_to, needs_error=True)
                    
                    #<  is projected_vector the same direction with proj_to??
                    _temp_result = are_parallel(proj_vec, proj_to)
                    assert _temp_result.all()

                    #<  is error_vector orthogonal to the proj_to
                    _temp_result = are_orthogonal(error_vec, proj_to)
                    assert _temp_result.all()
                    pass# for _
                pass#for dim

            '''some geometric check.    when the length of input changes???'''
            for dim in [2, 10, 100]:
                for _ in range(33):
                    proj_to = torch.randn(size=[1, dim])

                    ori_input = torch.randn(size=[1, dim])
                    ori_proj_vec, ori_error_vec = vector_proj(ori_input, proj_to, needs_error=True)

                    length_factor = (random.random()+0.1)*3
                    new_input = ori_input*length_factor#scale the input a bit.
                    new_proj_vec, new_error_vec = vector_proj(new_input, proj_to, needs_error=True)

                    assert _tensor_equal(ori_proj_vec*length_factor,  new_proj_vec)
                    assert _tensor_equal(ori_error_vec*length_factor, new_error_vec)
                    pass# for _
                pass# for dim
            pass#/ test
        
        if "does batch works the same as non-batched?" and True:
            for dim in [2, 10, 30]:
                for batch in [3, 15, 44]:
                    for _ in range(33):
                        input = torch.randn(size=[batch, dim])
                        proj_to = torch.randn(size=[batch, dim])
                        proj_vec, error_vec = vector_proj(input, proj_to, needs_error=True)
                        
                        proj_vec__manually = torch.empty(size=[batch, dim])
                        error_vec__manually = torch.empty(size=[batch, dim])
                        for ii in range(batch):
                            result_tuple = vector_proj(input[ii].reshape([-1, dim]), proj_to[ii].reshape([-1, dim]), needs_error=True)
                            proj_vec__manually[ii] = result_tuple[0]
                            error_vec__manually[ii] = result_tuple[1]
                            pass
                        assert _tensor_equal(proj_vec,  proj_vec__manually)
                        assert _tensor_equal(error_vec, error_vec__manually)
                        pass# for _
                    pass#for batch
                pass#for dim
            pass#/ test
        
        if "return style" and True:
            input = torch.tensor([[1., 1]])
            proj_to = torch.tensor([[1., 0]])
            proj_vec, error_vec = vector_proj(input, proj_to, needs_error=True)
            assert proj_vec     is not None # proj
            assert error_vec    is not None # error
            proj_vec, error_vec = vector_proj(input, proj_to, needs_error=False)
            assert proj_vec     is not None # proj
            assert error_vec    is     None # error
            
            pass #/ test

        return 
    
    ____basic_behavior_test____vector_proj()
    pass










if "old rotation neural net related         留着" and False:


    # def protect_rotation_matrix(input:torch.Tensor, epi = 0.000001):#->torch.Tensor:
    #     if len(input.shape)!=2:
    #         raise Exception("send matrix here.")
    #     dim = input.shape[0]
    #     if dim!=input.shape[1]:
    #         raise Exception("It must be square.")
        
    #     with torch.no_grad():
    #         # two_triagles = (input-input.T)*0.5
    #         # diagonal = input.mul(torch.eye(dim))
    #         # output_raw = two_triagles+diagonal
            
    #         length_of_output_raw_b = input.mul(input).sum(dim=1,keepdim=False).sqrt()
    #         epi_tensor = torch.tensor([epi], device=length_of_output_raw_b.device, dtype=length_of_output_raw_b.dtype)
    #         length_of_output_raw_safe_b = length_of_output_raw_b.maximum(epi_tensor)
    #         sqrt_of_length_b = length_of_output_raw_safe_b.sqrt()
    #         #result = input/length_of_input_safe_b#.unsqueeze(dim=1)
    #         output = input/sqrt_of_length_b.unsqueeze(dim=1)/sqrt_of_length_b.unsqueeze(dim=0)
            
    #         raise Exception("test not passed..")
    #         fds=432
        
    #     #output = vector_length_norm(output_raw)#shape is intentional.
        
    #     return output
    # raw_from_randn = torch.tensor([[0.5,2],[3.,4]])#randn([2,2])
    # rotation_matrix = protect_rotation_matrix(raw_from_randn)
    # print(rotation_matrix[0].mul(rotation_matrix[0]).sum())
    # print(rotation_matrix[1].mul(rotation_matrix[1]).sum())
    # print(rotation_matrix.T[0].mul(rotation_matrix.T[0]).sum())
    # print(rotation_matrix.T[1].mul(rotation_matrix.T[1]).sum())
    # unit_length_vec = vector_length_norm(torch.randn([1,2])).unsqueeze(dim=2)
    # print(unit_length_vec.mul(unit_length_vec).sum(), "unit_length_vec")
    # after_rotation = rotation_matrix.matmul(unit_length_vec).squeeze(dim=2)
    # print(after_rotation.mul(after_rotation).sum())
    # length_after_rotation = after_rotation.mul(after_rotation).sum(dim=1)

    # fds=432
        
        
            



    # def float_to_spherical(input:torch.Tensor, mix = False)->torch.Tensor:
    #     '''Basically, the mix flag only helps with debug. It may be slower a bit.'''
    #     if len(input.shape)!=2:
    #         raise Exception("The shape must be [batch, dim]")
    #     if input.amax()>1. or input.amin()<0.:
    #         raise Exception("Value must be inside [0., 1.] (both included.)")
    #     input_in_rad =  input*torch.pi/2.
    #     the_cos = input_in_rad.cos()
    #     the_sin = input_in_rad.sin()
    #     if not mix:
    #         result = torch.concat([the_cos, the_sin], dim=1)
    #         return result
    #     the_cos = the_cos.unsqueeze(dim=2)
    #     the_sin = the_sin.unsqueeze(dim=2)
    #     result = torch.concat([the_cos, the_sin], dim=2)
    #     result = result.view([input.shape[0], -1])
    #     return result
    # '''some basic test.'''
    # input = torch.tensor([[0., 0.33333, 0.5], [0.6, 0.7, 0.8]])
    # print(float_to_spherical(input))
    # print(float_to_spherical(input, True))
    # fds=432
            

    # def spherical_to_float(input:torch.Tensor, mix = False, rigorous = False)->torch.Tensor:
    #     if len(input.shape)!=2:
    #         raise Exception("The shape must be [batch, dim]")
    #     if input.shape[1]%2 == 1:
    #         raise Exception("The dim must be 2x something. They are pairs of cos and sin.")
    #     if rigorous and (input.amax()>1. or input.amin()<0.):
    #         raise Exception("Value must be inside [0., 1.] (both included.). Or set the param:rigorous to False.")
    #     if not mix:
    #         reshaped_input = input.view([input.shape[0], 2, -1])
    #         the_cos = reshaped_input[:,0,:]
    #         the_sin = reshaped_input[:,1,:]
    #         result_in_rad = torch.atan2(the_sin, the_cos)
    #         result = result_in_rad*2./torch.pi
    #         return result
    #     # mixed.
    #     reshaped_input = input.view([input.shape[0], -1, 2])
    #     the_cos = reshaped_input[:,:,0]
    #     the_sin = reshaped_input[:,:,1]
    #     result_in_rad = torch.atan2(the_sin, the_cos)
    #     result = result_in_rad*2./torch.pi
    #     return result
    # '''some basic test.'''
    # temp = torch.tensor([[0., 0.33333, 0.5], [0.6, 0.7, 0.8]])
    # input = float_to_spherical(temp)
    # print(spherical_to_float(input))
    # input = float_to_spherical(temp, mix=True)
    # print(spherical_to_float(input, mix=True))
    # fds=432
    pass

if "grad balancer,          not in use at the moment        留着" and False:

    # 写法是v1的写法。而且应该是多输出的。
    # 需要额外写一个function.set_materialize什么什么函数的实例。
    # class Grad_Balancer_2out_Function(torch.autograd.Function):
    #     r'''This class is not designed to be used directly.
    #     A critical safety check is in the wrapper class.    
    #     '''
    #     @staticmethod
    #     def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
    #         x:torch.Tensor = args[0]
    #         factor_for_path_1 = args[1]
    #         factor_for_path_2 = args[2]
    #         ctx.save_for_backward(factor_for_path_1, factor_for_path_2)
            
    #         x = torch.stack([x, x], dim=0)
    #         x = x.requires_grad_()
    #         return x

    #     @staticmethod
    #     def backward(ctx, g):
    #         #super().backward()
    #         # factor_for_path_1:torch.Tensor
    #         # factor_for_path_2:torch.Tensor
    #         factor_for_path_1, factor_for_path_2 = ctx.saved_tensors
            
    #         return g[0]*factor_for_path_1+g[1]*factor_for_path_2, None, None

    #     pass  # class
    # if '''some basic test.''' and __DEBUG_ME__() and True:
    #     input = torch.tensor([1., 2., 3.], requires_grad=True)
    #     factor_for_path_1 = torch.tensor([0.1])
    #     factor_for_path_2 = torch.tensor([0.01])
    #     output = Grad_Balancer_2out_Function.apply(input, factor_for_path_1, factor_for_path_2)
    #     print(output, "output")
    #     g_in = torch.ones_like(output)
    #     torch.autograd.backward(output, g_in,inputs= input)
    #     print(input.grad, "grad")
    #     pass




    # class Grad_Balancer_2out(torch.nn.Module):
    #     r"""This is a wrapper class. It helps you use the inner functional properly.
        
    #     It duplicates the forward path, 
    #     and multiplies the gradient from different backward path with a given weight.
    #     """
    #     def __init__(self, factor1:float, factor2:float, \
    #                     device=None, dtype=None) -> None:
    #         # factory_kwargs = {'device': device, 'dtype': dtype}
    #         super().__init__()
            
    #         if factor1<=0.:
    #             raise Exception("Param:factor1 must > 0.")
    #         if factor2<=0.:
    #             raise Exception("Param:factor2 must > 0.")
            
    #         self.factor_for_path_1 = torch.Tensor([factor1])
    #         self.factor_for_path_2 = torch.Tensor([factor2])
    #         pass
    #     def forward(self, x:torch.Tensor)->torch.Tensor:
    #         # If you know how pytorch works, you can comment this checking out.
    #         if self.training and (not x.requires_grad):
    #             raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

    #         #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
    #         #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
    #         return Grad_Balancer_2out_Function.apply(x, self.factor_for_path_1, self.factor_for_path_2)

    #     pass # class
    # if '''some basic test.''' and __DEBUG_ME__() and True:
    #     layer = Grad_Balancer_2out(0.1, 0.02)
    #     input = torch.tensor([1., 2., 3.], requires_grad=True)
    #     output = layer(input)
    #     print(output, "output")
    #     g_in = torch.ones_like(output)
    #     torch.autograd.backward(output, g_in,inputs= input)
    #     print(input.grad, "grad")
    #     pass



    # class Grad_Balancer_Function(torch.autograd.Function):
    #     r'''This class is not designed to be used directly.
    #     A critical safety check is in the wrapper class.    
    #     '''
    #     @staticmethod
    #     def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
    #         x:torch.Tensor = args[0]
    #         factor = args[1]
    #         x = x.unsqueeze(dim=0)
    #         result = x
            
    #         for _ in range(1, len(factor)):
    #             result = torch.concat([result,x], dim=0)
            
    #         ctx.save_for_backward(factor)
            
    #         result = result.requires_grad_()
    #         return result

    #     @staticmethod
    #     def backward(ctx, g):
    #         #super().backward()
    #         (factor,) = ctx.saved_tensors#this gives a TUPLE!!!
    #         g_out = torch.zeros_like(g[0])
            
    #         for i in range(len(factor)):
    #             g_out += g[i]*(factor[i].item())
                
    #         return g_out, None

    #     pass  # class
    # if '''some basic test.''' and __DEBUG_ME__() and True:
    #     input = torch.tensor([1., 2.], requires_grad=True)
    #     factor = torch.tensor([0.1, 0.02, 0.003])
    #     output = Grad_Balancer_Function.apply(input, factor)
    #     print(output, "output")
    #     g_in = torch.ones_like(output)
    #     torch.autograd.backward(output, g_in,inputs= input)
    #     print(input.grad, "grad")

    #     input = torch.tensor([[1., 2.], [3., 4.], ], requires_grad=True)
    #     factor = torch.tensor([0.1, 0.02, 0.003])
    #     output = Grad_Balancer_Function.apply(input, factor)
    #     print(output, "output")
    #     g_in = torch.ones_like(output)
    #     torch.autograd.backward(output, g_in,inputs= input)
    #     print(input.grad, "grad")
    #     pass




    # class Grad_Balancer(torch.nn.Module):
    #     r"""This is a wrapper class. It helps you use the inner functional properly.
        
    #     It duplicates the forward path, 
    #     and multiplies the gradient from different backward path with a given weight.
    #     """
    #     def __init__(self, weight_tensor_for_grad:torch.Tensor = torch.Tensor([1., 1.]), \
    #                     device=None, dtype=None) -> None:
    #         # factory_kwargs = {'device': device, 'dtype': dtype}
    #         super().__init__()
    #         if len(weight_tensor_for_grad.shape)!=1:
    #             raise Exception("Param:weight_tensor_for_grad should be a vector.")
    #         for i in range(len(weight_tensor_for_grad)):
    #             if weight_tensor_for_grad[i]<=0.:
    #                 raise Exception(f'The [{i}] element in the factor tensor is <=0.. It must be >0..')
                
    #         self.weight_tensor_for_grad = weight_tensor_for_grad
    #         pass
    #     def forward(self, x:torch.Tensor)->torch.Tensor:
    #         # If you know how pytorch works, you can comment this checking out.
    #         if self.training and (not x.requires_grad):
    #             raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")

    #         #forward(ctx, x:torch.Tensor, scaling_ratio:torch.Tensor, epi=torch.Tensor, \
    #         #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
    #         return Grad_Balancer_Function.apply(x, self.weight_tensor_for_grad)
    # if '''some basic test.''' and __DEBUG_ME__() and True:
    #     factor = torch.tensor([0.1, 0.02, 0.003])
    #     layer = Grad_Balancer(factor)
    #     input = torch.tensor([1., 2.], requires_grad=True)
    #     output = layer(input)
    #     print(output, "output")
    #     g_in = torch.ones_like(output)
    #     torch.autograd.backward(output, g_in,inputs= input)
    #     print(input.grad, "grad")
    #     pass
    pass













"grad tool"

def debug_zero_grad_ratio(parameter:torch.nn.parameter.Parameter, \
    print_out:float = False)->float:
    if parameter.grad is None:
        if print_out:
            print(f"{0.}, inside debug_zero_grad_ratio function __line {_line_()}")
            pass
        return 0.
    with torch.no_grad():
        result = 0.
        if not parameter.grad is None:
            flags = parameter.grad.eq(0.)
            total_amount = flags.sum().item()
            result = float(total_amount)/parameter.nelement()
        if print_out:
            print("get_zero_grad_ratio:", result)
        return result
#where is the test???
def debug_strong_grad_ratio(parameter:torch.nn.parameter.Parameter, log10_diff = 0., \
            epi_for_w = 0.01, epi_for_g = 0.01, print_out = False)->float:
    r'''the log10_diff should be approximately calculated like, 
    >>> log10(planned_epoch * learning_rate)
    I my test, I usually plan <3k epoch, and use 0.001 as lr, 
    so the default value for log10_diff  is 0.'''
    #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
    if parameter.grad is None:
        if print_out:
            print(0., "inside debug_strong_grad_ratio function __line 1082")
            pass
        return 0.

    the_device=parameter.device
    epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
    raw_weight_abs = parameter.abs()
    flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

    epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
    raw_weight_grad_abs = parameter.grad.abs()
    flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

    ten = torch.tensor([10.], device=the_device)
    log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
    corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
    flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

    flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
    result = (flag_useful_g.sum().to(torch.float32)/parameter.nelement()).item()
    if print_out:
        print(result, "inside debug_micro_grad_ratio function __line 1082")
        pass
    return result
#where is the test???
def make_grad_noisy(model:torch.nn.Module, noise_base:float = 1.5):
    for p in model.parameters():
        if p.requires_grad and (not p.grad is None):
            temp = torch.randn_like(p.grad)
            noise_factor = torch.pow(noise_base, temp)
            with torch.no_grad():
                #p.grad = p.grad.detach().clone().mul(noise_factor)
                p.grad = p.grad.detach().mul(noise_factor)
                pass
            pass
        pass
    pass

# p = torch.nn.Parameter(torch.tensor([42.]))
# p.grad = torch.tensor([1.])
# p.grad = p.grad.detach().clone().mul(torch.tensor([1.23]))
# print(p.grad)
# fds=432


import sys
# def __line__int():
#     return sys._getframe(1).f_lineno
def __line__str():
    return "    Line number: "+str(sys._getframe(1).f_lineno)
#print('This is line', __line__())







class Debug__LinearTeacher(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, num_layers = 2, mid_width =Optional[int], \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.layers = torch.nn.ParameterList()
        if 1 == num_layers:
            self.layers.append(torch.nn.Linear(in_features, out_features,bias))
        else:
            self.layers.append(torch.nn.Linear(in_features, mid_width,bias))
            for _ in range(num_layers-2):
                self.layers.append(torch.nn.Linear(mid_width, mid_width,bias))
                pass
            self.layers.append(torch.nn.Linear(mid_width, out_features, bias))
            pass
        pass 
    #end of function
    def forward(self, input_b_i:torch.Tensor) -> torch.Tensor:
        x = input_b_i
        layer:torch.nn.Linear
        for layer in self.layers:
            x = layer(x)
            pass
        return x
    #end of function
    pass





class Print_Timing:
    r'''
    >>> pt = Print_Timing(max_gap = 100, start_with = 0, first = 3, density:float = 4.)
    >>> for i in range(501):
    >>>     if pt.check(i):
    >>>         print(i, end = ", ")
    >>>         pass
    >>>     pass
    The result is 0, 1, 2, 5, 10, 19, 34, 62, 100, 200, 300, 400, 500, 
    '''
    def __init__(self, max_gap = 100, start_with = 0, first = 1, density:float = 1.):
        self.start_with = start_with
        self.first = first
        self.max_gap = max_gap
        
        self.return_true_when:List[float] = []
        the_exp = 0.
        if first-start_with-1>0:
            the_exp = math.log10(first-start_with-1)
            pass
        end_log = math.log10(max_gap)
        invert_of_density = 1/float(density)
        while the_exp<end_log:
            self.return_true_when.append(int(math.pow(10, the_exp)))
            the_exp += invert_of_density
            pass
        pass
    #end of function
    
    def check(self, epoch:int)->bool:
        if epoch>=self.max_gap and epoch%self.max_gap==0:
            return True
        
        calibrated_epoch = epoch-self.start_with+1
        if calibrated_epoch<=self.first:
            return True
        if calibrated_epoch in self.return_true_when:
            return True
        return False
    #end of function
            
    pass# end of class
if "test" and __DEBUG_ME__() and True:
    assert False, "格式还没改好。"
# pt = Print_Timing()
# for i in range(501):
#     if pt.check(i):
#         print(i, end = ", ")
#         pass
#     pass


    
    
    
def softmax_dim_1_from_yagaodirac(the_tensor:torch.Tensor, epi:Optional[torch.Tensor]=None)->torch.Tensor:
    if the_tensor.shape.__len__()!=2:
        raise Exception("According to my convention, the shape should be [batch, dim].")
    top_raw_element_of_each_row_b_d = the_tensor.amax(dim=1, keepdim=True)
    offset_input_b_d = the_tensor-top_raw_element_of_each_row_b_d
    the_exp_b_d = offset_input_b_d.exp()
    #only positive values.
    sum_of_each_row_b_1 = the_exp_b_d.sum(dim=1, keepdim=True)
    if epi is None:
        if torch.float16 == the_tensor.dtype:
            epi = torch.tensor(1e-3,dtype=torch.float16,device=the_tensor.device)
            pass
        elif torch.float32 == the_tensor.dtype:
            epi = torch.tensor(1e-6,dtype=torch.float32,device=the_tensor.device)
            pass
        else:
            raise Exception("dtype is weird. No implemented for fp64 now.")
    sum_of_each_row__safe__b_1 = sum_of_each_row_b_1.maximum(epi)
    result = the_exp_b_d/sum_of_each_row__safe__b_1
    return result
if "test" and __DEBUG_ME__() and True:
    input = torch.tensor([[0.,1]],dtype=torch.float16)
    print(softmax_dim_1_from_yagaodirac(input))
    print(input.to(torch.float32).softmax(dim=1))
    pass
if "test" and __DEBUG_ME__() and True:
    dummy = torch.tensor([[0,1]],dtype=torch.int64)
    import random
    input = torch.randn((random.randint(2,5),random.randint(2,5)),dtype=torch.float16)
    print(softmax_dim_1_from_yagaodirac(input))
    print(input.to(torch.float32).softmax(dim=1))
    pass

