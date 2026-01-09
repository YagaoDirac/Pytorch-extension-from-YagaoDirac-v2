import math
import numpy
import torch

if "fp64" and False:
    for to_me in range(1,1000):
        the_number = 10**(-to_me)
        if to_me<324:
            assert the_number < 1. and the_number > 0.
            pass
        else:
            assert the_number == 0.
            pass
        
        number_to_number = the_number**the_number
        math_number_to_number = math.pow(the_number,the_number)
        assert number_to_number>0.79 and number_to_number<=1.
        assert math_number_to_number>0.79 and math_number_to_number<=1.
        assert number_to_number == math_number_to_number
        if to_me>=18:
            assert number_to_number == 1.
            assert math_number_to_number == 1.
            pass
        pass
    
    pass

if "fp32 in numpy" and True:
    for to_me in range(1,100):
        the_number = numpy.float32(10**(-to_me))
        if to_me<46:
            assert the_number < 1. and the_number > 0.
            pass
        else:
            assert the_number == 0.
            pass
        
        math_number_to_number = math.pow(the_number,the_number)
        numpy_number_to_number = numpy.pow(the_number, the_number)
        assert math_number_to_number>0.79 and math_number_to_number<=1.
        assert numpy_number_to_number>0.79 and numpy_number_to_number<=1.
        assert math_number_to_number == numpy_number_to_number
        if to_me>=18:
            assert math_number_to_number == 1.
            assert numpy_number_to_number == 1.
            pass
        pass
    
    pass