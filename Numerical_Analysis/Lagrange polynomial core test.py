from typing import Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import accumulative_mul, _bool_equal___0_as_false, all_False, \
        _tensor_equal, _tensor_shape_check, iota, is_list_int, _float_equal
#         iota, str_the_list


import random
import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######






'''conclusion: Don't use it.
On the edge of the distribution, this core is very unstable.
F5 and have fun. 
ANCHOR_NUMBER doesn't affect the conclusion. Don't use it.'''











def remove_single_element__1d(input:torch.Tensor, remove_where:int|torch.Tensor)->torch.Tensor:
    assert input.shape.__len__() == 1
    
    left_part = input[0:remove_where]
    right_part = input[remove_where+1:]

    result = torch.concat((left_part, right_part), dim=0)
    return result

if "test   不是很彻底。" and __DEBUG_ME__() and False:
    def ____test____remove_single_element__1d():
        if "manual value " and True:
            input = torch.tensor([1., 2, 3, 4, 5])
            output = remove_single_element__1d(input, 1)
            assert _tensor_equal(output, [1,3,4,5])

            input = torch.tensor([1., 2, 3, 4, 5])
            output = remove_single_element__1d(input, 3)
            assert _tensor_equal(output, [1,2,3,5])

            pass#/test

        return
    ____test____remove_single_element__1d()
    pass






from matplotlib import pyplot as plt
if "the behavior of the core" and __DEBUG_ME__() and False:
    def ____the_behavior_of_the_core():

        if "VISUAL      attribution from a single point." and False:

            '''dim names: a: anchor, s:scan'''

            ANCHOR_NUMBER = 10         # this is the only line you need to change. 

            other_x___a = torch.rand(size=[ANCHOR_NUMBER])
            the_min = other_x___a.min().item()
            the_max = other_x___a.max().item()

            special_x___1 = torch.rand(size=[1])*(the_max-the_min)+the_min
            denominator__before_mul___a = special_x___1-other_x___a
            denominator__before_mul___1_a = denominator__before_mul___a.reshape(shape=[1, -1])

            #denominator___1 = accumulative_multiplication(denominator__before_mul___1_a)


            scan_this_x___s = torch.arange(the_min, end=the_max, step=0.01)
            assert scan_this_x___s.shape.__len__() == 1
            SCAN_STEPS = scan_this_x___s.nelement()

            scan_this_x___s_EXPANDa = scan_this_x___s.reshape(shape=[SCAN_STEPS, 1]).expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
            assert _tensor_shape_check(scan_this_x___s_EXPANDa, SCAN_STEPS, ANCHOR_NUMBER)
            other_x___EXPANDs_a = other_x___a.reshape(shape=[1, ANCHOR_NUMBER]).expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
            assert _tensor_shape_check(other_x___EXPANDs_a, SCAN_STEPS, ANCHOR_NUMBER)
            numerator__before_mul___s_a = (scan_this_x___s_EXPANDa - other_x___EXPANDs_a)
            assert _tensor_shape_check(numerator__before_mul___s_a, SCAN_STEPS, ANCHOR_NUMBER)

            # numerator___s = accumulative_multiplication(numerator__before_mul___s_a)
            # assert _tensor_shape_check(numerator___s, SCAN_STEPS)
            denominator__before_mul___EXPANDs_a = denominator__before_mul___1_a.expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
            result___s = accumulative_mul(numerator__before_mul___s_a/denominator__before_mul___EXPANDs_a)
            assert _tensor_shape_check(result___s, SCAN_STEPS)

            #to_plot___s = numerator___s/denominator___1
            to_plot___s = result___s
            plt.plot(scan_this_x___s, to_plot___s, c = "#00000099")

            the_max_of_result = result___s.max().item()
            the_min_of_result = result___s.min().item()

            #  plot marks
            x_of_the_mark = [special_x___1.item(), special_x___1.item()]
            y_of_the_mark = [the_max_of_result, the_min_of_result]
            plt.plot(x_of_the_mark, y_of_the_mark, c = "#FF0000FF", linewidth = 2)
            _temp_mark_width = (the_max-the_min)*0.1
            x_of_the_mark = [special_x___1.item()-_temp_mark_width/2, special_x___1.item()+_temp_mark_width/2]
            y_of_the_mark = [1., 1.]
            plt.plot(x_of_the_mark, y_of_the_mark, c = "#FF0000FF", linewidth = 2)

            

            for ii_ANCHOR_NUMBER in range(ANCHOR_NUMBER):
                x_of_the_mark = [other_x___a[ii_ANCHOR_NUMBER].item(), other_x___a[ii_ANCHOR_NUMBER].item()]
                y_of_the_mark = [the_max_of_result*0.2, the_min_of_result*0.2]
                plt.plot(x_of_the_mark, y_of_the_mark, c="#00000057")
                pass

            #plt.legend()
            plt.show()
            pass#/test

        if "simplification.   before into a function" and False and False:
            # param
            # other_x___a : torch.Tensor
            # special_x___1 : torch.Tensor
            # doc string
            '''dim names: a: anchor, s:scan'''
            # safety
            assert other_x___a.shape.__len__() == 1
            assert special_x___1.nelement() == 1

            # denominator
            denominator__before_mul___a = special_x___1-other_x___a
            denominator__before_mul___1_a = denominator__before_mul___a.reshape(shape=[1, -1])
            denominator__before_mul___EXPANDs_a = denominator__before_mul___1_a.expand(size=[SCAN_STEPS, ANCHOR_NUMBER])

            # scan range
            the_min = other_x___a.min().item()
            the_max = other_x___a.max().item()

            scan_this_x___s = torch.arange(the_min, end=the_max, step=0.01)
            SCAN_STEPS = scan_this_x___s.nelement()

            # numerator
            scan_this_x___s_EXPANDa = scan_this_x___s.reshape(shape=[SCAN_STEPS, 1]).expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
            other_x___EXPANDs_a = other_x___a.reshape(shape=[1, ANCHOR_NUMBER]).expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
            numerator__before_mul___s_a = (scan_this_x___s_EXPANDa - other_x___EXPANDs_a)

            # result
            result___s = accumulative_mul(numerator__before_mul___s_a/denominator__before_mul___EXPANDs_a)
            pass#/ simplification

        return

    ____the_behavior_of_the_core()

def _only_for__Lagrange_polynomial__to_use____calc_core(other_x___a : torch.Tensor, 
                special_x___1 : torch.Tensor, step:float, 
                #the_min:float|None = None, the_max:float|None = None
                the_min_of_x:float, the_max_of_x:float
                )->torch.Tensor:
    '''dim names: a: anchor, s:scan'''
    # safety
    assert other_x___a.shape.__len__() == 1
    assert special_x___1.nelement() == 1
    ANCHOR_NUMBER = other_x___a.shape[0]

    # # scan range
    # if the_min_of_x is None:
    #     the_min_of_x = other_x___a.min().item()
    #     pass
    # if the_max_of_x is None:
    #     the_max_of_x = other_x___a.max().item()
    #     pass

    scan_this_x___s = torch.arange(the_min_of_x, end=the_max_of_x, step=step)
    SCAN_STEPS = scan_this_x___s.nelement()

    # denominator
    denominator__before_mul___a = special_x___1-other_x___a
    denominator__before_mul___1_a = denominator__before_mul___a.reshape(shape=[1, -1])
    denominator__before_mul___EXPANDs_a = denominator__before_mul___1_a.expand(size=[SCAN_STEPS, ANCHOR_NUMBER])

    # numerator
    scan_this_x___s_EXPANDa = scan_this_x___s.reshape(shape=[SCAN_STEPS, 1]).expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
    other_x___EXPANDs_a = other_x___a.reshape(shape=[1, ANCHOR_NUMBER]).expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
    numerator__before_mul___s_a = (scan_this_x___s_EXPANDa - other_x___EXPANDs_a)

    # result
    result___s = accumulative_mul(numerator__before_mul___s_a/denominator__before_mul___EXPANDs_a)
    return result___s

if "test" and __DEBUG_ME__() and True:
    def ____test_____only_for__Lagrange_polynomial__to_use____calc_core():

        if "equivalance" and True:

            for ANCHOR_NUMBER in [3,5,10,21]:
                for _ii in range(11):

                    other_x___a = torch.rand(size=[ANCHOR_NUMBER])
                    the_min = other_x___a.min().item()
                    the_max = other_x___a.max().item()

                    special_x___1 = torch.rand(size=[1])*(the_max-the_min)+the_min
                    denominator__before_mul___a = special_x___1-other_x___a
                    denominator__before_mul___1_a = denominator__before_mul___a.reshape(shape=[1, -1])

                    scan_this_x___s = torch.arange(the_min, end=the_max, step=0.01)
                    assert scan_this_x___s.shape.__len__() == 1
                    SCAN_STEPS = scan_this_x___s.nelement()

                    scan_this_x___s_EXPANDa = scan_this_x___s.reshape(shape=[SCAN_STEPS, 1]).expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
                    assert _tensor_shape_check(scan_this_x___s_EXPANDa, SCAN_STEPS, ANCHOR_NUMBER)
                    other_x___EXPANDs_a = other_x___a.reshape(shape=[1, ANCHOR_NUMBER]).expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
                    assert _tensor_shape_check(other_x___EXPANDs_a, SCAN_STEPS, ANCHOR_NUMBER)
                    numerator__before_mul___s_a = (scan_this_x___s_EXPANDa - other_x___EXPANDs_a)
                    assert _tensor_shape_check(numerator__before_mul___s_a, SCAN_STEPS, ANCHOR_NUMBER)

                    denominator__before_mul___EXPANDs_a = denominator__before_mul___1_a.expand(size=[SCAN_STEPS, ANCHOR_NUMBER])
                    result___s = accumulative_mul(numerator__before_mul___s_a/denominator__before_mul___EXPANDs_a)
                    assert _tensor_shape_check(result___s, SCAN_STEPS)

                    #  func ver
                    #  func ver
                    the_min_of_x = other_x___a.min().item()
                    the_max_of_x = other_x___a.max().item()

                    func_result = _only_for__Lagrange_polynomial__to_use____calc_core(other_x___a=other_x___a, special_x___1 = special_x___1, 
                                                                        step= 0.01, the_min_of_x=the_min_of_x, the_max_of_x=the_max_of_x)
                    assert _tensor_equal(result___s, func_result)

                    pass#for _ii
                pass#for ANCHOR_NUMBER

            pass#/test


        if "VISUAL       a series of cores" and False:
            dim = 11
            data_x = torch.rand(size=[dim])
            data_x = data_x.sort().values
            #data_y = torch.rand(size=[dim])

            for core_index_for_this_iter in range(dim):
                other_x___a=remove_single_element__1d(data_x, remove_where=core_index_for_this_iter)
                special_x___1 = data_x[core_index_for_this_iter]

                the_min_of_x = other_x___a.min().item()
                the_max_of_x = other_x___a.max().item()
                the_core_for_this_iter = _only_for__Lagrange_polynomial__to_use____calc_core(other_x___a = other_x___a, special_x___1=special_x___1, 
                                                                        step= 0.01, the_min_of_x=the_min_of_x, the_max_of_x=the_max_of_x)

                #contribution_from_this_sample = the_core_for_this_iter*data_y[core_index_for_this_iter]

    

                scan_this_x___s = torch.arange(the_min_of_x, end=the_max_of_x, step=0.01)
                assert scan_this_x___s.shape.__len__() == 1


                #to_plot___s = contribution_from_this_sample
                to_plot___s = the_core_for_this_iter
                
                plt.plot(scan_this_x___s, to_plot___s, c = "#00000099")

                the_max_of_result = result___s.max().item()
                the_min_of_result = result___s.min().item()

                #  plot marks
                x_of_the_mark = [special_x___1.item(), special_x___1.item()]
                y_of_the_mark = [the_max_of_result, the_min_of_result]
                plt.plot(x_of_the_mark, y_of_the_mark, c = "#FF0000FF", linewidth = 2)
                _temp_mark_width = (the_max-the_min)*0.1
                x_of_the_mark = [special_x___1.item()-_temp_mark_width/2, special_x___1.item()+_temp_mark_width/2]
                y_of_the_mark = [1., 1.]
                plt.plot(x_of_the_mark, y_of_the_mark, c = "#FF0000FF", linewidth = 2)

                ANCHOR_NUMBER = dim-1
                for ii_ANCHOR_NUMBER in range(ANCHOR_NUMBER):
                    x_of_the_mark = [other_x___a[ii_ANCHOR_NUMBER].item(), other_x___a[ii_ANCHOR_NUMBER].item()]
                    y_of_the_mark = [the_max_of_result*0.2, the_min_of_result*0.2]
                    plt.plot(x_of_the_mark, y_of_the_mark, c="#00000057")
                    pass

                plt.show()
                pass#/core_index_for_this_iter

            pass#/test





        run this test and have fun.
        if "VISUAL       final result" and True:
            dim = 11
            data_x = torch.rand(size=[dim])
            data_x = data_x.sort().values
            data_y = torch.rand(size=[dim])

            result:torch.Tensor|None = None


            the_min_of_x = other_x___a.min().item()
            the_max_of_x = other_x___a.max().item()
            for core_index_for_this_iter in range(dim):
                other_x___a=remove_single_element__1d(data_x, remove_where=core_index_for_this_iter)
                special_x___1 = data_x[core_index_for_this_iter]
                the_core_for_this_iter = _only_for__Lagrange_polynomial__to_use____calc_core(other_x___a = other_x___a, special_x___1=special_x___1, 
                                                                        step= 0.01, the_min_of_x=the_min_of_x, the_max_of_x=the_max_of_x)

                contribution_from_this_sample = the_core_for_this_iter*data_y[core_index_for_this_iter]

                if result is None:
                    result = contribution_from_this_sample
                    pass
                else:
                    result += contribution_from_this_sample
                    pass
                pass#/core_index_for_this_iter




            scan_this_x___s = torch.arange(the_min_of_x, end=the_max_of_x, step=0.01)
            assert scan_this_x___s.shape.__len__() == 1



            to_plot___s = result
            #to_plot___s = the_core_for_this_iter
            
            plt.plot(scan_this_x___s, to_plot___s, c = "#00000099")

            the_max_of_result = result___s.max().item()
            the_min_of_result = result___s.min().item()

            #  plot marks
            x_of_the_mark = [special_x___1.item(), special_x___1.item()]
            y_of_the_mark = [the_max_of_result, the_min_of_result]
            plt.plot(x_of_the_mark, y_of_the_mark, c = "#FF0000FF", linewidth = 2)
            _temp_mark_width = (the_max-the_min)*0.1
            x_of_the_mark = [special_x___1.item()-_temp_mark_width/2, special_x___1.item()+_temp_mark_width/2]
            y_of_the_mark = [1., 1.]
            plt.plot(x_of_the_mark, y_of_the_mark, c = "#FF0000FF", linewidth = 2)

            ANCHOR_NUMBER = dim-1
            for ii_ANCHOR_NUMBER in range(ANCHOR_NUMBER):
                x_of_the_mark = [other_x___a[ii_ANCHOR_NUMBER].item(), other_x___a[ii_ANCHOR_NUMBER].item()]
                y_of_the_mark = [the_max_of_result*0.2, the_min_of_result*0.2]
                plt.plot(x_of_the_mark, y_of_the_mark, c="#00000057")
                pass

            plt.show()

            pass#/test











    ____test_____only_for__Lagrange_polynomial__to_use____calc_core()
    pass
















