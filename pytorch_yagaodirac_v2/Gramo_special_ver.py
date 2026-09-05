'''
2026年，在做新的数字神经网络的时候，因为要用最早的gramo，然后发现当时的gramo的制作并不符合我现在的规范。
于是决定单独对这一个进行重新制作。

其他的以后如果制作了，也会加入到这个文件。如果没有加入，就是还在原来的地方。
'''


from typing import Any, Optional, Literal
#import math
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _tensor_shape_check, _float_equal, \
        vector_length_norm, get_vector_length, \
        iota, \
        print_table
from pytorch_yagaodirac_v2.Util_log10_related import log10_avg_safe


def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
if "test" and False:
    assert __DEBUG_ME__()
    pass

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



if "to fold old code" and False:
    # def dtype_upgrade(input:torch.dtype)->torch.dtype:
    #     '''this function is adapted to pytorch. When shift to other framework, recheck it.'''
    #     if input == torch.float64:
    #         return torch.float64
    #     else:
    #         return torch.float32



    # def get_vector_length(input:torch.Tensor, result_dtype = torch.float64)->torch.Tensor:
    #     _temp = input*input
    #     _temp = _temp.sum(dim=-1, dtype=result_dtype)
    #     _temp.sqrt_()
    #     return _temp
    # #copied from the Util.py file. No validation here.
    pass











if "algo prototype test before the function ver." and __DEBUG_ME__() and False:
    def ____algo_prototype____gramo():

        if "prototype" and False:
            batch = 5
            out_dim = 3

            epsilon = torch.tensor(0.001)
            scaling_factor = torch.tensor(3.)
            mul_me__when_g_too_small___s = torch.tensor(64.)
            protect_accuracy = True

            g_in___b_o = torch.tensor([ 
                                    [0., 0, 0], 
                                    [3., 4, 0], 
                                    [100, 100, 100], 
                                    [0.01, 0.01, 0.01], 
                                    [0.0001, 0.0001, 0.0001], 
                                    ])
            assert _tensor_shape_check(g_in___b_o, batch, out_dim)

            #<  length?
            _g_in__sqr___b_o = g_in___b_o*g_in___b_o
            assert _tensor_shape_check(_g_in__sqr___b_o, batch, out_dim)
            assert _tensor_equal(_g_in__sqr___b_o, [ 
                                    [0, 0,  0], 
                                    [9, 16, 0],
                                    [10000, 10000, 10000],
                                    [0.0001, 0.0001, 0.0001],
                                    [0.00000001, 0.00000001, 0.00000001], 
                                                    ])
            #del g_in___b_o

            _length_sqr___b_1 = _g_in__sqr___b_o.sum(dim=1, keepdim=True)
            assert _tensor_shape_check(_length_sqr___b_1, batch, 1)
            assert _tensor_equal(_length_sqr___b_1, [   [0      ], 
                                                        [25     ], 
                                                        [30000  ], 
                                                        [0.0003 ], 
                                                        [0.00000003 ], 
                                                        ])
            del _g_in__sqr___b_o
            
            length___b_1 = _length_sqr___b_1.sqrt_()
            assert _tensor_shape_check(length___b_1, batch, 1)
            assert _tensor_equal(length___b_1[:4], [[0    ], 
                                                    [5    ],
                                                    [173.2051 ],
                                                    [0.01732  ],])
            assert _tensor_equal(length___b_1[4],   [0.0001732  ], epsilon=0.0000001)
            del _length_sqr___b_1
            #</ length
            #<  flag
            flag__length_too_small___b_1 = length___b_1.lt(epsilon)#*dim__s)
            assert flag__length_too_small___b_1.eq(torch.tensor([[True], [False], [False], [False], [True]])).all()

            #<  mul_me>
            mul_me___when_g_is_ok__raw___b_1 = scaling_factor/length___b_1
            assert _tensor_shape_check(mul_me___when_g_is_ok__raw___b_1, batch, 1)
            assert mul_me___when_g_is_ok__raw___b_1[0].isinf() == True
            assert _tensor_equal(mul_me___when_g_is_ok__raw___b_1[1:4], [#[inf     ], # sqrt of 1.5
                                                                        [0.6   ],
                                                                        [0.0173  ],
                                                                        [173.2051 ],])
            assert _tensor_equal(mul_me___when_g_is_ok__raw___b_1[4],   [17320.51  ], epsilon=0.01)

            mul_me___when_g_is_ok__raw___b_1.nan_to_num_(posinf = 1., neginf = 1.)
            assert mul_me___when_g_is_ok__raw___b_1.eq(0.).any() == False


            # mul_me__b_1=flag__length_too_small__b_1.logical_not() * mul_me___when_g_is_ok__raw__b_1 + \
            #             flag__length_too_small__b_1               * mul_me__when_g_too_small__s#this is input, 1e-3 by default
            mul_me___b_1 = torch.where(flag__length_too_small___b_1, mul_me__when_g_too_small___s, mul_me___when_g_is_ok__raw___b_1)
            assert _tensor_shape_check(mul_me___b_1, batch, 1)
            assert _tensor_equal(mul_me___b_1, [[64 ],
                                                [0.6   ],
                                                [0.01732  ],
                                                [173.2051 ],
                                                [64 ],
                                                ])
            assert mul_me___b_1.dtype == g_in___b_o.dtype
            
            
            if protect_accuracy:
                mul_me___b_1.log2_().add_(0.5).floor_()# nearest power of 2. This floor func returns fp. It works. No need to convert it to integer.
                assert _tensor_equal(mul_me___b_1, mul_me___b_1.floor())
                mul_me___b_1.exp2_()
                assert _tensor_equal(mul_me___b_1, [[64.], # sqrt of 1.5
                                                    [0.5],
                                                    [0.0156],
                                                    [128],
                                                    [64 ],
                                                    ])
                pass

            mul_me___b_o = mul_me___b_1.expand(size=[-1, g_in___b_o.shape[1]])
            assert _tensor_shape_check(mul_me___b_o, batch, out_dim)
            del mul_me___b_1
            grad_for_x__b_o = g_in___b_o*mul_me___b_o
            assert _tensor_shape_check(grad_for_x__b_o, batch, out_dim)
            assert _tensor_equal(grad_for_x__b_o, [ [0., 0,  0], 
                                                    [1.5, 2, 0],
                                                    [1.5625, 1.5625, 1.5625],
                                                    [1.28, 1.28, 1.28],
                                                    [0.0064, 0.0064, 0.0064],
                                                    ])
            _assert_only___length_of_result = get_vector_length(grad_for_x__b_o)
            assert _tensor_equal(_assert_only___length_of_result[:4], [ 0, 
                                                                    2.5, 
                                                                    2.706, 
                                                                    2.217, ], epsilon=0.01)
            assert _tensor_equal(_assert_only___length_of_result[4], [ 0.01108  ], epsilon=0.00001)

            _assert_only___length_of_g_in = get_vector_length(g_in___b_o)
            assert _tensor_equal(_assert_only___length_of_g_in[:4], [0,  5,  173.2051, 0.01732], epsilon=0.01)
            assert _tensor_equal(_assert_only___length_of_g_in[4], [ 0.0001732  ], epsilon=0.0000001)

            pass#/ test

        if "Simplified.     Not test purpose." and False:
            #<  param
            batch = 5
            out_dim = 3

            epsilon = torch.tensor(0.001)
            scaling_factor = torch.tensor(3.)
            mul_me__when_g_too_small___s = torch.tensor(64.)
            protect_accuracy = True

            g_in___b_o = torch.tensor([ 
                                    [0., 0, 0], 
                                    [3., 4, 0], 
                                    [100, 100, 100], 
                                    [0.01, 0.01, 0.01], 
                                    [0.0001, 0.0001, 0.0001], 
                                    ])
            #</ param

            #<  length
            _g_in__sqr___b_o = g_in___b_o*g_in___b_o
            #del g_in___b_o
            _length_sqr___b_1 = _g_in__sqr___b_o.sum(dim=1, keepdim=True)
            del _g_in__sqr___b_o
            length___b_1 = _length_sqr___b_1.sqrt_()
            del _length_sqr___b_1
            #</ length
            #<  flag
            flag__length_too_small___b_1 = length___b_1.lt(epsilon)#*dim__s)

            #<  mul_me>
            mul_me___when_g_is_ok__raw___b_1 = scaling_factor/length___b_1
            mul_me___when_g_is_ok__raw___b_1.nan_to_num_(posinf = 1., neginf = 1.)#protection. The 1. is not important. It can be any number.
            assert mul_me___when_g_is_ok__raw___b_1.eq(0.).any() == False

            #<  torch where
            mul_me___b_1 = torch.where(flag__length_too_small___b_1, mul_me__when_g_too_small___s, mul_me___when_g_is_ok__raw___b_1)
            
            #<  value accuracy.       not the result acc.
            if protect_accuracy:
                mul_me___b_1.log2_().add_(0.5).floor_()# nearest power of 2. This floor func returns fp. It works. No need to convert it to integer.
                mul_me___b_1.exp2_()
                pass

            mul_me___b_o = mul_me___b_1.expand(size=[-1, g_in___b_o.shape[1]])
            assert _tensor_shape_check(mul_me___b_o, batch, out_dim)
            del mul_me___b_1
            grad_for_x__b_o = g_in___b_o*mul_me___b_o
            # pass#if input_needs_grad
            # return grad_for_x__b_o

            pass#/ test

        return 
    ____algo_prototype____gramo()
    pass

def _gramo_algo_test(g_in___b_o:torch.Tensor, scaling_factor = torch.tensor(3.), epsilon = torch.tensor(0.001), 
            mul_me__when_g_too_small___s = torch.tensor(64.), protect_accuracy = True)->torch.Tensor:

    #<  length
    _g_in__sqr___b_o = g_in___b_o*g_in___b_o
    #del g_in___b_o
    _length_sqr___b_1 = _g_in__sqr___b_o.sum(dim=1, keepdim=True)
    del _g_in__sqr___b_o
    length___b_1 = _length_sqr___b_1.sqrt_()
    del _length_sqr___b_1
    #</ length
    #<  flag
    flag__length_too_small___b_1 = length___b_1.lt(epsilon)#*dim__s)

    #<  mul_me>
    mul_me___when_g_is_ok__raw___b_1 = scaling_factor/length___b_1
    mul_me___when_g_is_ok__raw___b_1.nan_to_num_(posinf = 1., neginf = 1.)#protection. The 1. is not important. It can be any number.
    assert mul_me___when_g_is_ok__raw___b_1.eq(0.).any() == False

    #<  torch where
    mul_me___b_1 = torch.where(flag__length_too_small___b_1, mul_me__when_g_too_small___s, mul_me___when_g_is_ok__raw___b_1)
    
    #<  value accuracy.       not the result acc.
    if protect_accuracy:
        mul_me___b_1.log2_().add_(0.5).floor_()# nearest power of 2. This floor func returns fp. It works. No need to convert it to integer.
        mul_me___b_1.exp2_()
        pass

    mul_me___b_o = mul_me___b_1.expand(size=[-1, g_in___b_o.shape[1]])
    del mul_me___b_1
    grad_for_x__b_o = g_in___b_o*mul_me___b_o
    # pass#if input_needs_grad
    return grad_for_x__b_o

if "algo prototype test           with function." and __DEBUG_ME__() and False:
    def ____test____the_function_____gramo_algo_test():

        if "equivalence" and False:
            batch = 5
            out_dim = 3

            epsilon = torch.tensor(0.001)
            scaling_factor = torch.tensor(3.)
            mul_me__when_g_too_small___s = torch.tensor(64.)
            protect_accuracy = True

            g_in___b_o = torch.tensor([ 
                                    [0., 0, 0], 
                                    [3., 4, 0], 
                                    [100, 100, 100], 
                                    [0.01, 0.01, 0.01], 
                                    [0.0001, 0.0001, 0.0001], 
                                    ])
            assert _tensor_shape_check(g_in___b_o, batch, out_dim)

            #<  length?
            _g_in__sqr___b_o = g_in___b_o*g_in___b_o
            assert _tensor_shape_check(_g_in__sqr___b_o, batch, out_dim)
            assert _tensor_equal(_g_in__sqr___b_o, [ 
                                    [0, 0,  0], 
                                    [9, 16, 0],
                                    [10000, 10000, 10000],
                                    [0.0001, 0.0001, 0.0001],
                                    [0.00000001, 0.00000001, 0.00000001], 
                                                    ])
            #del g_in___b_o

            _length_sqr___b_1 = _g_in__sqr___b_o.sum(dim=1, keepdim=True)
            assert _tensor_shape_check(_length_sqr___b_1, batch, 1)
            assert _tensor_equal(_length_sqr___b_1, [   [0      ], 
                                                        [25     ], 
                                                        [30000  ], 
                                                        [0.0003 ], 
                                                        [0.00000003 ], 
                                                        ])
            del _g_in__sqr___b_o
            
            length___b_1 = _length_sqr___b_1.sqrt_()
            assert _tensor_shape_check(length___b_1, batch, 1)
            assert _tensor_equal(length___b_1[:4], [[0    ], 
                                                    [5    ],
                                                    [173.2051 ],
                                                    [0.01732  ],])
            assert _tensor_equal(length___b_1[4],   [0.0001732  ], epsilon=0.0000001)
            del _length_sqr___b_1
            #</ length
            #<  flag
            flag__length_too_small___b_1 = length___b_1.lt(epsilon)#*dim__s)
            assert flag__length_too_small___b_1.eq(torch.tensor([[True], [False], [False], [False], [True]])).all()

            #<  mul_me>
            mul_me___when_g_is_ok__raw___b_1 = scaling_factor/length___b_1
            assert _tensor_shape_check(mul_me___when_g_is_ok__raw___b_1, batch, 1)
            assert mul_me___when_g_is_ok__raw___b_1[0].isinf() == True
            assert _tensor_equal(mul_me___when_g_is_ok__raw___b_1[1:4], [#[inf     ], # sqrt of 1.5
                                                                        [0.6   ],
                                                                        [0.0173  ],
                                                                        [173.2051 ],])
            assert _tensor_equal(mul_me___when_g_is_ok__raw___b_1[4],   [17320.51  ], epsilon=0.01)

            mul_me___when_g_is_ok__raw___b_1.nan_to_num_(posinf = 1., neginf = 1.)
            assert mul_me___when_g_is_ok__raw___b_1.eq(0.).any() == False


            # mul_me__b_1=flag__length_too_small__b_1.logical_not() * mul_me___when_g_is_ok__raw__b_1 + \
            #             flag__length_too_small__b_1               * mul_me__when_g_too_small__s#this is input, 1e-3 by default
            mul_me___b_1 = torch.where(flag__length_too_small___b_1, mul_me__when_g_too_small___s, mul_me___when_g_is_ok__raw___b_1)
            assert _tensor_shape_check(mul_me___b_1, batch, 1)
            assert _tensor_equal(mul_me___b_1, [[64 ],
                                                [0.6   ],
                                                [0.01732  ],
                                                [173.2051 ],
                                                [64 ],
                                                ])
            assert mul_me___b_1.dtype == g_in___b_o.dtype
            
            
            if protect_accuracy:
                mul_me___b_1.log2_().add_(0.5).floor_()# nearest power of 2. This floor func returns fp. It works. No need to convert it to integer.
                assert _tensor_equal(mul_me___b_1, mul_me___b_1.floor())
                mul_me___b_1.exp2_()
                assert _tensor_equal(mul_me___b_1, [[64.], # sqrt of 1.5
                                                    [0.5],
                                                    [0.0156],
                                                    [128],
                                                    [64 ],
                                                    ])
                pass

            mul_me___b_o = mul_me___b_1.expand(size=[-1, g_in___b_o.shape[1]])
            assert _tensor_shape_check(mul_me___b_o, batch, out_dim)
            del mul_me___b_1
            grad_for_x__b_o = g_in___b_o*mul_me___b_o
            assert _tensor_shape_check(grad_for_x__b_o, batch, out_dim)
            assert _tensor_equal(grad_for_x__b_o, [ [0., 0,  0], 
                                                    [1.5, 2, 0],
                                                    [1.5625, 1.5625, 1.5625],
                                                    [1.28, 1.28, 1.28],
                                                    [0.0064, 0.0064, 0.0064],
                                                    ])



            #<  function equilavence
            function_result = _gramo_algo_test(g_in___b_o = torch.tensor([ 
                                                [0., 0, 0], 
                                                [3., 4, 0], 
                                                [100, 100, 100], 
                                                [0.01, 0.01, 0.01], 
                                                [0.0001, 0.0001, 0.0001], 
                                                ]),
                        scaling_factor = torch.tensor(3.), epsilon = torch.tensor(0.001), 
                        mul_me__when_g_too_small___s = torch.tensor(64.), protect_accuracy = True)
            assert _tensor_equal(grad_for_x__b_o, function_result)
            pass#/ test

        import random

        if "shuffle before or after function call" and False:
            for batch in [2,6,13,27]:
                for out_dim in [3,9,18,31]:
                    for protect_accuracy in [True, False]:
                        for _ in range(66):

                            iota_of_batch = iota(batch)
                            random.shuffle(iota_of_batch)
                            index_to_shuffle = iota_of_batch
                            del iota_of_batch

                            g_in___b_o = torch.randn(size=[batch, out_dim])

                            scaling_factor = torch.rand(size=[])*3. + 0.5
                            assert scaling_factor.nelement() == 1
                            assert scaling_factor>=0.5

                            epsilon = torch.pow(0.1, torch.rand(size=[])*2. + 2.)
                            assert epsilon.nelement() == 1
                            assert epsilon >= 0.0001
                            assert epsilon <= 0.01

                            mul_me__when_g_too_small___s = torch.pow(10, torch.rand(size=[])*1. + 2.)
                            assert mul_me__when_g_too_small___s.nelement() == 1
                            assert mul_me__when_g_too_small___s >= 100
                            assert mul_me__when_g_too_small___s <= 1000

                            #<  func_then_shuffle
                            _temp_normal_result___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o, 
                                        scaling_factor = scaling_factor, epsilon = epsilon, 
                                        mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, protect_accuracy = protect_accuracy)
                            assert _tensor_shape_check(_temp_normal_result___b_o, batch, out_dim)
                            result___func_then_shuffle___b_o = _temp_normal_result___b_o[index_to_shuffle]
                            assert _tensor_shape_check(result___func_then_shuffle___b_o, batch, out_dim)
                            #<  shuffle_then_func
                            g_in___shuffle_then_func___b_o = g_in___b_o[index_to_shuffle]
                            assert _tensor_shape_check(g_in___shuffle_then_func___b_o, batch, out_dim)
                            result___shuffle_then_func___b_o  = _gramo_algo_test(g_in___b_o = g_in___shuffle_then_func___b_o, 
                                        scaling_factor = scaling_factor, epsilon = epsilon, 
                                        mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, protect_accuracy = protect_accuracy)
                            assert _tensor_shape_check(result___shuffle_then_func___b_o, batch, out_dim)

                            #<  assert 
                            assert _tensor_equal(result___func_then_shuffle___b_o, result___shuffle_then_func___b_o)

                            pass#for _
                        pass#for protect_accuracy
                    pass#for out_dim
                pass#for batch

            pass#/ test

        if "batch independence" and False:
            for batch in [2,6,13,27]:
                for out_dim in [3,9,18,31]:
                    for protect_accuracy in [True, False]:
                        for _ in range(66):

                            g_in___b_o = torch.randn(size=[batch, out_dim])

                            scaling_factor = torch.rand(size=[])*3. + 0.5
                            assert scaling_factor.nelement() == 1
                            assert scaling_factor>=0.5

                            epsilon = torch.pow(0.1, torch.rand(size=[])*2. + 2.)
                            assert epsilon.nelement() == 1
                            assert epsilon >= 0.0001
                            assert epsilon <= 0.01

                            mul_me__when_g_too_small___s = torch.pow(10, torch.rand(size=[])*1. + 2.)
                            assert mul_me__when_g_too_small___s.nelement() == 1
                            assert mul_me__when_g_too_small___s >= 100
                            assert mul_me__when_g_too_small___s <= 1000

                            #<  func_then_shuffle
                            result_with_batch___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o, 
                                        scaling_factor = scaling_factor, epsilon = epsilon, 
                                        mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, protect_accuracy = protect_accuracy)
                            assert _tensor_shape_check(result_with_batch___b_o, batch, out_dim)

                            for ii_batch in range(batch):
                                the_row_of_input___1_o = g_in___b_o[ii_batch].reshape(shape=[1, -1])
                                assert _tensor_shape_check(the_row_of_input___1_o, 1, out_dim)

                                result_of_the_row___1_o  = _gramo_algo_test(g_in___b_o = the_row_of_input___1_o, 
                                            scaling_factor = scaling_factor, epsilon = epsilon, 
                                            mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, protect_accuracy = protect_accuracy)
                                assert _tensor_shape_check(result_of_the_row___1_o, 1, out_dim)
                                #<  assert 
                                assert _tensor_equal(result_with_batch___b_o[ii_batch].reshape(shape=[1, -1]), result_of_the_row___1_o)
                                pass#for ii_batch

                            pass#for _
                        pass#for protect_accuracy
                    pass#for out_dim
                pass#for batch

            pass#/ test

        '''the avg of abs of elements of the output (with protect_accuracy = False) is roughly  0.82 * sqrt( 1 / out_dim)'''
        if "avg of elements      by shape" and False:
            if "results" and True:

                # batch 10     test_time 100    
                # out_dim__list,    10,   100,  1000
                # avg_of_elements, 0.259, 0.080, 0.025

                # batch 100     test_time 100  
                # out_dim__list,    10,   100,  1000
                # avg_of_elements, 0.259, 0.080, 0.025

                # batch 1000     test_time 100   
                # out_dim__list,    10,   100,  1000
                # avg_of_elements, 0.259, 0.080, 0.025

                pass



            print(f"__LINE__ {_line_()}      avg of elements      by shape")
                        
            #------------------#------------------#------------------
            dim_list =                          [ 10,100, 1000]
            number_of_tests_list = torch.tensor([100,100, 100])
            number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
            for ii_outter_param_set in range(dim_list.__len__()):
                batch = dim_list[ii_outter_param_set]
                # iota_of_dim = iota(dim)
                number_of_tests = int(number_of_tests_list[ii_outter_param_set].item())
                device = 'cpu'
                # if dim>100:
                #     device = 'cuda'
                #     pass
                print(f"batch {batch}     test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------
                avg_of_elements = ["avg_of_elements"]#don't modify this.
                
                out_dim__list = [10, 100, 1000]      ################################
                #_when_start = time.perf_counter()
                
                for out_dim in out_dim__list:
                    _raw_result___avg_of_elements = torch.empty(size=[number_of_tests])
                    for ii__test in range(number_of_tests):
                        
                        #------------------#------------------#------------------
                        #<  init           
                        g_in___b_o = torch.randn(size=[batch, out_dim])

                        scaling_factor = torch.tensor(1.)
                        epsilon = torch.tensor(0.001)
                        mul_me__when_g_too_small___s = torch.tensor(100.)

                        #<  func_then_shuffle
                        result___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o, 
                                    scaling_factor = scaling_factor, epsilon = epsilon, 
                                    mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, 
                                    protect_accuracy = False) #######################    only for this test
                        assert _tensor_shape_check(result___b_o, batch, out_dim)
                        avg_of_elements___s = result___b_o.abs().mean()
                        assert _tensor_shape_check(avg_of_elements___s)
                        
                        #<  measure
                        _this_result = 123
                        #------------------#------------------#------------------
                        
                        _raw_result___avg_of_elements[ii__test] = avg_of_elements___s
                        pass#for ii__test

                    avg_of_elements.append(_raw_result___avg_of_elements.mean().item())
                    
                    pass#for scanned_param
                #_when_end = time.perf_counter()
                #print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")

                print(f"batch {batch}")
                out_dim__list.insert(0, "out_dim__list")
                print_table([out_dim__list, avg_of_elements])
                pass#for ii_outter_param_set


            pass#/ test

        if "scaling_factor" and False:
            for batch in [2,6,13,27]:
                for out_dim in [3,9,18,31]:
                    for _ in range(66):
                        
                        g_in___b_o = torch.randn(size=[batch, out_dim])

                        scaling_factor_1 = torch.rand(size=[])*3. + 0.5
                        assert scaling_factor_1.nelement() == 1
                        assert scaling_factor_1>=0.5
                        scaling_factor_2 = torch.rand(size=[])*3. + 0.5
                        scaling_factor_2_div_1 = scaling_factor_2 / scaling_factor_1

                        epsilon = torch.pow(0.1, torch.rand(size=[])*2. + 2.)
                        assert epsilon.nelement() == 1
                        assert epsilon >= 0.0001
                        assert epsilon <= 0.01

                        mul_me__when_g_too_small___s = torch.pow(10, torch.rand(size=[])*1. + 2.)
                        assert mul_me__when_g_too_small___s.nelement() == 1
                        assert mul_me__when_g_too_small___s >= 100
                        assert mul_me__when_g_too_small___s <= 1000

                        #<  func_then_shuffle
                        result_1___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o, 
                                    scaling_factor = scaling_factor_1, epsilon = epsilon, 
                                    mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, 
                                    protect_accuracy = False) #######################    only for this test
                        assert _tensor_shape_check(result_1___b_o, batch, out_dim)
                        result_2___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o, 
                                    scaling_factor = scaling_factor_2, epsilon = epsilon, 
                                    mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, 
                                    protect_accuracy = False) #######################    only for this test
                        assert _tensor_shape_check(result_2___b_o, batch, out_dim)

                        maybe_same_as_2___b_o = result_1___b_o * scaling_factor_2_div_1
                        assert _tensor_shape_check(maybe_same_as_2___b_o, batch, out_dim)

                        assert _tensor_equal(result_2___b_o, maybe_same_as_2___b_o)
                        pass#for _
                    pass#for out_dim
                pass#for batch
            pass#/ test

        if "VISUAL       all the 3 trivial params" and False:
            from matplotlib import pyplot as plt

            batch = 1000
            out_dim = 1

            for scaling_factor__float in [1, 3, 10]:
                scaling_factor = torch.tensor(scaling_factor__float)
                    
                for epsilon__float in [0.03, 0.1, 0.25]:
                    epsilon = torch.tensor(epsilon__float)
    
                    for mul_me__when_g_too_small___s__float in [10, 20, 30]:
                        mul_me__when_g_too_small___s = torch.tensor(mul_me__when_g_too_small___s__float)

                        log_of_g_in___b_o = torch.linspace(start=-2., end=1., steps = batch).reshape(shape=[batch, 1])
                        g_in___b_o = torch.pow(10., log_of_g_in___b_o)

                        # plt.plot(g_in___b_o, g_in___b_o)
                        # plt.xscale('log')
                        # plt.show()
                        
                        result___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o, 
                                    scaling_factor = scaling_factor, epsilon = epsilon, 
                                    mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, protect_accuracy = False)
                        assert _tensor_shape_check(result___b_o, batch, out_dim)

                        plt.plot(g_in___b_o, result___b_o, label = "result")
                        plt.plot(g_in___b_o, result___b_o/g_in___b_o, label = "change")
                        plt.plot(g_in___b_o, g_in___b_o, ":", lw = 1, label = "ori", color="#aaaaaa")
                        plt.plot([0.3, 3],    [scaling_factor__float, scaling_factor__float], ":", lw = 6, color="#ff0000")
                        plt.plot([epsilon__float, epsilon__float], [0.1, 10],":", lw = 3, color="#aaaaaa")
                        plt.plot([0.01, 0.1], [mul_me__when_g_too_small___s__float, mul_me__when_g_too_small___s__float],":", lw = 3, color="#aaaaaa")
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.legend()
                        plt.title(f"sf {scaling_factor__float}    ep {epsilon__float}    mul me {mul_me__when_g_too_small___s__float}")
                        plt.show()
                        
                        pass#for mul_me__when_g_too_small___s__float
                    pass#for epsilon__float
                pass#for scaling_factor__float
            pass#/ test

        '''to get a smooth curve, the scaling_factor == epsilon * mul_me__when_g_too_small___s'''
        if "VISUAL       smooth ???" and False:
            from matplotlib import pyplot as plt

            batch = 1000
            out_dim = 1

            for _ in range(1011):

                scaling_factor = torch.pow(10, torch.rand([])*1.5)
                assert _tensor_shape_check(scaling_factor)
                epsilon = torch.pow(10, torch.rand([])*1.5-3)
                assert _tensor_shape_check(epsilon)
                mul_me__when_g_too_small___s = scaling_factor/epsilon
    
                log_of_g_in___b_o = torch.linspace(start=-3., end=1., steps = batch).reshape(shape=[batch, 1])
                g_in___b_o = torch.pow(10., log_of_g_in___b_o)

                
                result___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o, 
                            scaling_factor = scaling_factor, epsilon = epsilon, 
                            mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, protect_accuracy = False)
                assert _tensor_shape_check(result___b_o, batch, out_dim)

                plt.plot(g_in___b_o, result___b_o, label = "result")
                plt.plot(g_in___b_o, result___b_o/g_in___b_o, label = "change")
                plt.plot(g_in___b_o, g_in___b_o, ":", lw = 1, label = "ori", color="#aaaaaa")
                plt.plot([0.3, 3],    [scaling_factor.item(), scaling_factor.item()], ":", lw = 6, color="#ff0000")
                plt.plot([epsilon.item(), epsilon.item()], [0.1, 10],":", lw = 3, color="#aaaaaa")
                plt.plot([0.01, 0.1], [mul_me__when_g_too_small___s.item(), mul_me__when_g_too_small___s.item()],":", lw = 3, color="#aaaaaa")
                plt.xscale('log')
                plt.yscale('log')
                plt.legend()
                plt.title(f"sf {scaling_factor.item():.3f}    ep {epsilon.item():.3f}    mul me {mul_me__when_g_too_small___s.item():.3f}")
                plt.show()
                pass#for _
            pass#/ test

        if "protect_accuracy" and False:
            for batch in [2,7,15,27]:
                for out_dim in [5,11,27,42]:

                    for _ in range(11):

                        scaling_factor = torch.pow(10, torch.rand([])*4. - 2.)
                        assert scaling_factor>0.
                        assert _tensor_shape_check(scaling_factor)
                        epsilon = torch.tensor(0.001)
                        assert _tensor_shape_check(epsilon)
                        mul_me__when_g_too_small___s = torch.tensor(100.)

                        g_in___b_o = torch.randn(size=[batch, out_dim])
                        assert _tensor_shape_check(g_in___b_o, batch, out_dim)
                        g_in__but_bigger___b_o  = g_in___b_o.abs() * 1.1
                        g_in__but_smaller___b_o = g_in___b_o.abs() * 0.9

                        result___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o, 
                                    scaling_factor = scaling_factor, epsilon = epsilon, 
                                    mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, protect_accuracy = True)
                        assert _tensor_shape_check(result___b_o, batch, out_dim)

                        assert result___b_o.sign().eq(g_in___b_o.sign()).all()

                        while True:
                            break_flag_1 = False
                            _flag___bigger___b_1 = result___b_o[:, 0].abs().gt(g_in__but_bigger___b_o[:, 0]).reshape(shape=[-1, 1])
                            assert _tensor_shape_check(_flag___bigger___b_1, batch, 1)
                            if _flag___bigger___b_1.any():
                                _flag___bigger___b_o = _flag___bigger___b_1.expand(size=[-1, out_dim])
                                result___b_o = torch.where(_flag___bigger___b_o, result___b_o*0.5, result___b_o)
                                assert _tensor_shape_check(result___b_o, batch, out_dim)
                                pass
                            else:
                                break_flag_1 = True
                                pass

                            break_flag_2 = False
                            _flag___smaller___b_1 = result___b_o[:, 0].abs().lt(g_in__but_smaller___b_o[:, 0]).reshape(shape=[-1, 1])
                            assert _tensor_shape_check(_flag___smaller___b_1, batch, 1)
                            if _flag___smaller___b_1.any():
                                _flag___smaller___b_o = _flag___smaller___b_1.expand(size=[-1, out_dim])
                                result___b_o = torch.where(_flag___smaller___b_o, result___b_o*2., result___b_o)
                                assert _tensor_shape_check(result___b_o, batch, out_dim)
                                pass
                            else:
                                break_flag_2 = True
                                pass

                            if break_flag_1 and break_flag_2:
                                break
                            pass# while True:

                        #<  assert
                        assert _tensor_equal(g_in___b_o, result___b_o)
                        pass#for _
                    pass#for out_dim
                pass#for batch
            pass#/ test

        return
    ____test____the_function_____gramo_algo_test()
    pass



class _only_for___Gramo_vec_len_to_scaling_factor___to_use___Function_class(torch.autograd.Function):
    r'''a special version in 2026. 
    
    input param:
    >>> x:torch.Tensor (must be set as require_grad = True)
    >>> scaling_factor = torch.tensor([1.])
    >>> epsilon = torch.tensor([1e-5])
    >>> mul_me__when_g_too_small__s = torch.tensor([1e-3])
    >>> protect_accuracy = torch.tensor(True)

    retur type: torch.Tensor
    '''
    @staticmethod
    #def forward(*args: Any, **kwargs: Any)->Any:
    def forward(x:torch.Tensor, scaling_factor___s:torch.Tensor, epsilon___s:torch.Tensor, \
                mul_me__when_g_too_small___s:torch.Tensor, protect_accuracy___1_bool:torch.Tensor, \
                            *args: Any, **kwargs: Any)->Any:
        if x.requires_grad:
            assert x.shape.__len__() == 2, "Only accept rank-2 tensor. The shape should be[batch, something]"
            assert scaling_factor___s.nelement() == 1
            assert epsilon___s.nelement() == 1
            assert mul_me__when_g_too_small___s.nelement() == 1
            assert protect_accuracy___1_bool.nelement() == 1
            pass
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        #x:torch.Tensor
        x= inputs[0]
        if x.requires_grad:
            scaling_factor___s:torch.Tensor = inputs[1]
            epsilon___s:torch.Tensor = inputs[2]
            mul_me__when_g_too_small___s:torch.Tensor = inputs[3]
            protect_accuracy___1_bool:torch.Tensor = inputs[4]

            ctx.save_for_backward(scaling_factor___s, epsilon___s, mul_me__when_g_too_small___s, 
                        protect_accuracy___1_bool)
            return
        else:
            #nothing is saved for backward. Return directly.
            return
        #end of function

    @staticmethod
    def backward(ctx, g_in___b_o):#->tuple[Optional[torch.Tensor], None, None, None]:


        #g_in___b_o:torch.Tensor
        # scaling_factor___s:torch.Tensor
        # epsilon___s:torch.Tensor
        # mul_me__when_g_too_small___s:torch.Tensor
        # protect_accuracy___1_bool:torch.Tensor

        if ctx.saved_tensors.__len__() == 0:
            return None, None, None, None, None

        (scaling_factor___s, epsilon___s, mul_me__when_g_too_small___s, protect_accuracy___1_bool) = ctx.saved_tensors

        #展开 

        #<  length
        _g_in__sqr___b_o = g_in___b_o*g_in___b_o
        #del g_in___b_o
        _length_sqr___b_1 = _g_in__sqr___b_o.sum(dim=1, keepdim=True)
        del _g_in__sqr___b_o
        length___b_1 = _length_sqr___b_1.sqrt_()
        del _length_sqr___b_1
        #</ length
        #<  flag
        flag__length_too_small___b_1 = length___b_1.lt(epsilon___s)#*dim__s)# dim相关的事情移到外面去.

        #<  mul_me>
        mul_me___when_g_is_ok__raw___b_1 = scaling_factor___s/length___b_1
        mul_me___when_g_is_ok__raw___b_1.nan_to_num_(posinf = 1., neginf = 1.)#protection. The 1. is not important. It can be any number.
        assert mul_me___when_g_is_ok__raw___b_1.eq(0.).any() == False

        #<  torch where
        mul_me___b_1 = torch.where(flag__length_too_small___b_1, mul_me__when_g_too_small___s, mul_me___when_g_is_ok__raw___b_1)
        
        #<  value accuracy.       not the result acc.
        if protect_accuracy___1_bool:
            mul_me___b_1.log2_().add_(0.5).floor_()# nearest power of 2. This floor func returns fp. It works. No need to convert it to integer.
            mul_me___b_1.exp2_()
            pass

        mul_me___b_o = mul_me___b_1.expand(size=[-1, g_in___b_o.shape[1]])
        del mul_me___b_1
        grad_for_x__b_o = g_in___b_o*mul_me___b_o
        # pass#if input_needs_grad
        return grad_for_x__b_o, None, None, None, None

    pass  # class

if "equivalence" and False:
    def ____test____quivalence____nn_function_subclass():
        if "quivalence" and True:
            for batch in [2,8,15,32]:
                for out_dim in [5,11,19,52]:
                    for protect_accuracy in [True, False]:
                        for _ in range(66):
                            
                            g_in___b_o = torch.randn(size=[batch, out_dim])

                            scaling_factor = torch.rand(size=[])*3. + 0.5
                            assert scaling_factor.nelement() == 1
                            assert scaling_factor>=0.5

                            epsilon = torch.pow(0.1, torch.rand(size=[])*2. + 2.)
                            assert epsilon.nelement() == 1
                            assert epsilon >= 0.0001
                            assert epsilon <= 0.01

                            mul_me__when_g_too_small___s = torch.pow(10, torch.rand(size=[])*1. + 2.)
                            assert mul_me__when_g_too_small___s.nelement() == 1
                            assert mul_me__when_g_too_small___s >= 100
                            assert mul_me__when_g_too_small___s <= 1000

                            #<  function ver
                            function_result___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o.detach().clone(), 
                                        scaling_factor = scaling_factor, epsilon = epsilon, 
                                        mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, 
                                        protect_accuracy = protect_accuracy) #######################    only for this test
                            assert _tensor_shape_check(function_result___b_o, batch, out_dim)

                            #<  subclass ver
                            _temp__input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True)
                            output_1___b_o:torch.Tensor = _only_for___Gramo_vec_len_to_scaling_factor___to_use___Function_class.apply( \
                                        _temp__input___b_o, scaling_factor, epsilon, 
                                        mul_me__when_g_too_small___s, torch.tensor(protect_accuracy))
                            assert _tensor_shape_check(output_1___b_o, batch, out_dim)
                            assert _tensor_equal(_temp__input___b_o, output_1___b_o)
                            output_1___b_o.backward(gradient=g_in___b_o.detach().clone(), inputs=[_temp__input___b_o])
                            assert _temp__input___b_o.grad is not None
                            #<  assert
                            assert _temp__input___b_o.grad.eq(function_result___b_o).all()

                            pass#for _
                        pass#for protect_accuracy
                    pass#for out_dim
                pass#for batch
            pass#/ test

        if "device adaptive" and False:
            batch = 2
            out_dim = 3
            device='cpu'
            input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True, device=device)

            g_in___b_o = torch.randn(size=[batch, out_dim], device=device)

            scaling_factor___s = torch.tensor(1., device=device)
            epsilon___s = torch.tensor(0.01, device=device)
            mul_me__when_g_too_small___s = torch.tensor(100., device=device)
            protect_accuracy___s = torch.tensor(True, device=device)


            output___b_o:torch.Tensor = _only_for___Gramo_vec_len_to_scaling_factor___to_use___Function_class.apply( \
                        input___b_o, scaling_factor___s, epsilon___s, mul_me__when_g_too_small___s, protect_accuracy___s)
            assert output___b_o.device.type == device
            assert _tensor_shape_check(output___b_o, batch, out_dim)

            output___b_o.backward(gradient=g_in___b_o, inputs=[input___b_o])
            assert input___b_o.grad is not None


            device='cuda'
            input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True, device=device)

            g_in___b_o = torch.randn(size=[batch, out_dim], device=device)

            scaling_factor___s = torch.tensor(1., device=device)
            epsilon___s = torch.tensor(0.01, device=device)
            mul_me__when_g_too_small___s = torch.tensor(100., device=device)
            protect_accuracy___s = torch.tensor(True, device=device)


            output___b_o:torch.Tensor = _only_for___Gramo_vec_len_to_scaling_factor___to_use___Function_class.apply( \
                        input___b_o, scaling_factor___s, epsilon___s, mul_me__when_g_too_small___s, protect_accuracy___s)
            assert output___b_o.device.type == device
            assert _tensor_shape_check(output___b_o, batch, out_dim)

            output___b_o.backward(gradient=g_in___b_o, inputs=[input___b_o])
            assert input___b_o.grad is not None
            pass#/ test

        if "dtype adaptive" and True:
            batch = 2
            out_dim = 3

            for dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                for g_in_dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                    for scaling_factor___s__dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                        for epsilon___s__dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                            for mul_me__when_g_too_small___s__dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                                #for protect_accuracy___s__dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:

                                input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True, dtype=dtype)
                                g_in___b_o = torch.randn(size=[batch, out_dim], dtype=g_in_dtype)

                                scaling_factor___s           = torch.tensor(1.,   dtype=scaling_factor___s__dtype)
                                epsilon___s                  = torch.tensor(0.01, dtype=epsilon___s__dtype)
                                mul_me__when_g_too_small___s = torch.tensor(100., dtype=mul_me__when_g_too_small___s__dtype)
                                protect_accuracy___s         = torch.tensor(True)#, dtype=protect_accuracy___s__dtype)
                                #<  payload
                                output___b_o:torch.Tensor = _only_for___Gramo_vec_len_to_scaling_factor___to_use___Function_class.apply( \
                                            input___b_o, scaling_factor___s, epsilon___s, mul_me__when_g_too_small___s, protect_accuracy___s)
                                assert output___b_o.dtype == dtype

                                output___b_o.backward(gradient=g_in___b_o, inputs=[input___b_o])
                                assert input___b_o.grad is not None
                                assert input___b_o.grad.dtype == dtype
                                #pass#for protect_accuracy___s__dtype
                                pass#for mul_me__when_g_too_small___s__dtype
                            pass#for epsilon___s__dtype
                        pass#for scaling_factor___s__dtype
                    pass#for g_in_dtype
                pass#for dtype
            pass#/ test

        return
    ____test____quivalence____nn_function_subclass()
    pass


















class Gramo_vec_len_to_scaling_factor(torch.nn.Module):
    r"""this is a special version of the very first gramo. Basically the same. 

    """
    scaling_factor___s :torch.nn.Parameter
    epsilon___s        :torch.nn.Parameter
    mul_me__when_g_too_small___s :torch.nn.Parameter
    protect_accuracy___1_flag       :torch.nn.Parameter

    def __init__(self, scaling_factor = 1., epsilon = 1e-3, mul_me__when_g_too_small = 200., protect_accuracy = True, device = None, \
                _debug__allow_spike = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #<  safety
        assert scaling_factor>0.
        assert epsilon>0.
        assert mul_me__when_g_too_small>0.
        if not _debug__allow_spike:
            assert scaling_factor>= epsilon*mul_me__when_g_too_small, "To make sure there's no spike in the curve. See test above. { \
                    ''}If you know what you are doing, you can comment this assertion out."
            pass
        #<  payload
        self.scaling_factor___s = torch.nn.Parameter(torch.tensor(scaling_factor, device=device), requires_grad=False)
        self.epsilon___s        = torch.nn.Parameter(torch.tensor(epsilon,        device=device), requires_grad=False)
        self.mul_me__when_g_too_small___s        = torch.nn.Parameter(torch.tensor(mul_me__when_g_too_small,        device=device), requires_grad=False)
        self.protect_accuracy___1_flag        = torch.nn.Parameter(torch.tensor(protect_accuracy,        device=device), requires_grad=False)

        assert self.scaling_factor___s.requires_grad == False
        assert self.epsilon___s.requires_grad == False
        assert self.mul_me__when_g_too_small___s.requires_grad == False
        assert self.protect_accuracy___1_flag.requires_grad == False
        return

    def forward(self, x:torch.Tensor)->torch.Tensor:
        
        #assert x.shape.__len__() == 2, "Only accept rank-2 tensor. The shape should be[batch, something]"      
        #  this check is now inside the function subclass
        
        return _only_for___Gramo_vec_len_to_scaling_factor___to_use___Function_class.apply(
                x, self.scaling_factor___s, self.epsilon___s, self.mul_me__when_g_too_small___s, self.protect_accuracy___1_flag)

    def set_mode___vec_len_to_1(self)->None:
        self.scaling_factor___s.data = torch.tensor(1., device=self.scaling_factor___s.device)
        return
    def set_mode___vec_len_to_sqrt_of_dim(self, out_dim:int)->None:
        self.scaling_factor___s.data = torch.sqrt(torch.tensor(out_dim, dtype=self.scaling_factor___s.dtype, device=self.scaling_factor___s.device))
        return
    def set_smooth(self, scaling_factor:float|None = None, epsilon:float|None = None, mul_me__when_g_too_small:float|None = None, \
                smooth_when_1 = 1., dtype = None)->None:
        '''Formula: scaling_factor == epsilon * mul_me__when_g_too_small * smooth_when_1. 

        Exact 2 of the first 3 params should be provided. The param:smooth_when_1 is always provided.
        
        See the VISUAL test above'''

        assert smooth_when_1 >= 1.
        if dtype is None:
            dtype = self.scaling_factor___s.dtype
            pass

        #_temp__given_param_count_must_be_2 = 0

        if scaling_factor is not None:
            #_temp__given_param_count_must_be_2 += 1
            assert scaling_factor>=1. , "Or if you know what you are doing, do anything to this line. Make sure it's at least > 0."

            if epsilon is not None:
                # scaling_factor, epsilon are given.
                assert epsilon > 0.
                assert mul_me__when_g_too_small is None , "All 3 params are all provided. Exact 2 of them should be provided."
                mul_me__when_g_too_small = scaling_factor / (epsilon * smooth_when_1)
                pass
            else: #   epsilon is None
                # scaling_factor, mul_me__when_g_too_small are given.
                assert mul_me__when_g_too_small is not None , "Only param:epsilon is provided. Exact 2 of them should be provided."
                epsilon = scaling_factor / (mul_me__when_g_too_small * smooth_when_1)
                pass
            pass
        else: #   scaling_factor is  None
                # epsilon, mul_me__when_g_too_small are given.
            assert epsilon is not None , "Not enough params are provided. Exact 2 of them should be provided."
            assert epsilon > 0.
            assert mul_me__when_g_too_small is not None , "Only param:epsilon is provided. Exact 2 of them should be provided."
            assert mul_me__when_g_too_small > 0.

            scaling_factor = epsilon * mul_me__when_g_too_small * smooth_when_1
            assert scaling_factor>=1. , "Or if you know what you are doing, do anything to this line. Make sure it's at least > 0."
            pass
        pass#if scaling_factor

        self.scaling_factor___s          .data = torch.tensor(scaling_factor,           device=self.scaling_factor___s.device, dtype=dtype)
        self.epsilon___s                 .data = torch.tensor(epsilon,                  device=self.scaling_factor___s.device, dtype=dtype)
        self.mul_me__when_g_too_small___s.data = torch.tensor(mul_me__when_g_too_small, device=self.scaling_factor___s.device, dtype=dtype)

        assert self.scaling_factor___s          .requires_grad == False
        assert self.epsilon___s                 .requires_grad == False
        assert self.mul_me__when_g_too_small___s.requires_grad == False

        return

    def set_smooth___changes_only__mul_me__when_g_too_small(self)->None:
        self.set_smooth(scaling_factor=self.scaling_factor___s.item(), epsilon=self.epsilon___s.item())
        return
    
    def set_scaling_factor(self, scaling_factor:float)->None:
        assert scaling_factor > 1., "or at least > 0."
        the_device = self.scaling_factor___s.device
        the_dtype = self.scaling_factor___s.dtype
        self.scaling_factor___s.data = torch.tensor(scaling_factor, device=the_device, dtype=the_dtype)
        assert self.scaling_factor___s.requires_grad == False
        pass
    def scale_scaling_factor(self, by:float)->None:
        assert by > 0.
        self.scaling_factor___s.data = self.scaling_factor___s * by
        assert self.scaling_factor___s.requires_grad == False
        pass

    def set_epsilon(self, epsilon:float)->None:
        assert epsilon > 0.
        the_device = self.epsilon___s.device
        the_dtype = self.epsilon___s.dtype
        self.epsilon___s.data = torch.tensor(epsilon, device=the_device, dtype=the_dtype, requires_grad=False)
        assert self.epsilon___s.requires_grad == False
        pass
    
    def set_mul_me__when_g_too_small(self, mul_me__when_g_too_small___sepsilon:float)->None:
        assert mul_me__when_g_too_small___sepsilon > 0.
        the_device = self.mul_me__when_g_too_small___s.device
        the_dtype = self.mul_me__when_g_too_small___s.dtype
        self.mul_me__when_g_too_small___s.data = torch.tensor(mul_me__when_g_too_small___sepsilon, device=the_device, dtype=the_dtype, requires_grad=False)
        assert self.mul_me__when_g_too_small___s.requires_grad == False
        pass
    def set_protect_accuracy(self, protect_accuracy = True)->None:
        the_device = self.protect_accuracy___1_flag.device
        self.protect_accuracy___1_flag.data = torch.tensor(protect_accuracy, device=the_device, requires_grad=False)
        assert self.protect_accuracy___1_flag.requires_grad == False
        pass

    #old code. I wrote this probably before the dim adaption. 
    # def scale_scaling_factor(self, by:float)->None:
    #     self.set_scaling_factor((self.scaling_factor*by).item())
    #     pass
    # def set_epsilon(self, epsilon:float)->None:
    #     the_device = self.epsilon.device
    #     the_dtype = self.epsilon.dtype
    #     self.epsilon.data = torch.tensor(epsilon, device=the_device, dtype=the_dtype, requires_grad=False)
    #     pass
    # def set_mul_me_when_g_too_small(self, mul_me_when_g_too_small:float)->None:
    #     the_device = self.mul_me_when_g_too_small.device
    #     the_dtype = self.mul_me_when_g_too_small.dtype
    #     self.mul_me_when_g_too_small.data = torch.tensor(mul_me_when_g_too_small, device=the_device, dtype=the_dtype, requires_grad=False)
    #     pass
    

    
    # def extra_repr(self) -> str:
    #     return f'scaling_factor={self.scaling_factor.item():.4e}, epsilon={self.epsilon.item():.4e}, mul_me_when_g_too_small={self.mul_me_when_g_too_small.item():.4e}'
    pass#end of class

if "equivalence" and False:
    def ____equivalence____Gramo_vec_len_to_scaling_factor():

        if "quivalence" and True:
            for batch in [2,8,15,32]:
                for out_dim in [5,11,19,52]:
                    for protect_accuracy in [True, False]:
                        for _ in range(66):
                            
                            g_in___b_o = torch.randn(size=[batch, out_dim])

                            scaling_factor = torch.rand(size=[])*3. + 0.5
                            assert scaling_factor.nelement() == 1
                            assert scaling_factor>=0.5

                            epsilon = torch.pow(0.1, torch.rand(size=[])*2. + 2.)
                            assert epsilon.nelement() == 1
                            assert epsilon >= 0.0001
                            assert epsilon <= 0.01

                            mul_me__when_g_too_small___s = torch.pow(10, torch.rand(size=[])*1. + 2.)
                            assert mul_me__when_g_too_small___s.nelement() == 1
                            assert mul_me__when_g_too_small___s >= 100
                            assert mul_me__when_g_too_small___s <= 1000

                            #<  function ver
                            function_result___b_o = _gramo_algo_test(g_in___b_o = g_in___b_o.detach().clone(), 
                                        scaling_factor = scaling_factor, epsilon = epsilon, 
                                        mul_me__when_g_too_small___s = mul_me__when_g_too_small___s, 
                                        protect_accuracy = protect_accuracy) #######################    only for this test
                            assert _tensor_shape_check(function_result___b_o, batch, out_dim)

                            #<  subclass ver
                            _temp__input_1___b_o = torch.randn(size=[batch, out_dim], requires_grad=True)
                            output_1___b_o:torch.Tensor = _only_for___Gramo_vec_len_to_scaling_factor___to_use___Function_class.apply( \
                                        _temp__input_1___b_o, scaling_factor, epsilon, 
                                        mul_me__when_g_too_small___s, torch.tensor(protect_accuracy))
                            assert _tensor_shape_check(output_1___b_o, batch, out_dim)
                            assert _tensor_equal(_temp__input_1___b_o, output_1___b_o)
                            output_1___b_o.backward(gradient=g_in___b_o.detach().clone(), inputs=[_temp__input_1___b_o])
                            assert _temp__input_1___b_o.grad is not None
                            #<  assert
                            assert _temp__input_1___b_o.grad.eq(function_result___b_o).all()


                            #<  full gramo ver
                            gramo = Gramo_vec_len_to_scaling_factor(scaling_factor = scaling_factor.item(), epsilon = epsilon.item(), 
                                        mul_me__when_g_too_small = mul_me__when_g_too_small___s.item(), protect_accuracy = protect_accuracy, 
                                        _debug__allow_spike = True)#debug purpose.

                            _temp__input_2___b_o = torch.randn(size=[batch, out_dim], requires_grad=True)

                            output_2___b_o:torch.Tensor = gramo(_temp__input_2___b_o)
                            assert _tensor_shape_check(output_2___b_o, batch, out_dim)
                            assert _tensor_equal(_temp__input_2___b_o, output_2___b_o)
                            output_2___b_o.backward(gradient=g_in___b_o.detach().clone(), inputs=[_temp__input_2___b_o])
                            assert _temp__input_2___b_o.grad is not None
                            #<  assert
                            assert _temp__input_2___b_o.grad.eq(function_result___b_o).all()

                            pass#for _
                        pass#for protect_accuracy
                    pass#for out_dim
                pass#for batch
            pass#/ test

        return
    ____equivalence____Gramo_vec_len_to_scaling_factor()
    pass

if "device/dtype adaption" and False:
    def ____adaption____Gramo_vec_len_to_scaling_factor():

        if "quivalence" and True:
            batch = 2
            out_dim = 3

                            
            gramo = Gramo_vec_len_to_scaling_factor()
            device = 'cpu'
            assert gramo.scaling_factor___s.device.type == device

            input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True, device=device)
            g_in___b_o = torch.randn(size=[batch, out_dim], device=device)

            output___b_o:torch.Tensor = gramo(input___b_o)
            assert _tensor_shape_check(output___b_o, batch, out_dim)
            assert _tensor_equal(input___b_o, output___b_o)
            assert output___b_o.device.type == device
            output___b_o.backward(gradient=g_in___b_o, inputs=[input___b_o])
            assert input___b_o.grad is not None


            gramo.cuda()
            device = 'cuda'
            assert gramo.scaling_factor___s.device.type == device
            
            input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True, device=device)
            g_in___b_o = torch.randn(size=[batch, out_dim], device=device)

            output___b_o:torch.Tensor = gramo(input___b_o)
            assert _tensor_shape_check(output___b_o, batch, out_dim)
            assert _tensor_equal(input___b_o, output___b_o)
            assert output___b_o.device.type == device
            output___b_o.backward(gradient=g_in___b_o, inputs=[input___b_o])
            assert input___b_o.grad is not None




            gramo = Gramo_vec_len_to_scaling_factor(device='cuda')
            device = 'cuda'
            assert gramo.scaling_factor___s.device.type == device

            input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True, device=device)
            g_in___b_o = torch.randn(size=[batch, out_dim], device=device)

            output___b_o:torch.Tensor = gramo(input___b_o)
            assert _tensor_shape_check(output___b_o, batch, out_dim)
            assert _tensor_equal(input___b_o, output___b_o)
            assert output___b_o.device.type == device
            output___b_o.backward(gradient=g_in___b_o, inputs=[input___b_o])
            assert input___b_o.grad is not None


            gramo.cpu()
            device = 'cpu'
            assert gramo.scaling_factor___s.device.type == device
            
            input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True, device=device)
            g_in___b_o = torch.randn(size=[batch, out_dim], device=device)

            output___b_o:torch.Tensor = gramo(input___b_o)
            assert _tensor_shape_check(output___b_o, batch, out_dim)
            assert _tensor_equal(input___b_o, output___b_o)
            assert output___b_o.device.type == device
            output___b_o.backward(gradient=g_in___b_o, inputs=[input___b_o])
            assert input___b_o.grad is not None

            pass#/ test

        if "dtype adaptive" and True:
            batch = 2
            out_dim = 3

            for dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                for g_in_dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                    for scaling_factor___s__dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                        for epsilon___s__dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:
                            for mul_me__when_g_too_small___s__dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16, ]:

                                input___b_o = torch.randn(size=[batch, out_dim], requires_grad=True, dtype=dtype)
                                g_in___b_o = torch.randn(size=[batch, out_dim], dtype=g_in_dtype)

                                gramo = Gramo_vec_len_to_scaling_factor()
                                gramo.scaling_factor___s.to(scaling_factor___s__dtype)
                                gramo.epsilon___s.to(epsilon___s__dtype)
                                gramo.mul_me__when_g_too_small___s.to(mul_me__when_g_too_small___s__dtype)

                                #<  payload
                                output___b_o:torch.Tensor = gramo(input___b_o)
                                assert output___b_o.dtype == dtype

                                output___b_o.backward(gradient=g_in___b_o, inputs=[input___b_o])
                                assert input___b_o.grad is not None
                                assert input___b_o.grad.dtype == dtype
                                pass#for mul_me__when_g_too_small___s__dtype
                            pass#for epsilon___s__dtype
                        pass#for scaling_factor___s__dtype
                    pass#for g_in_dtype
                pass#for dtype
            pass#/ test

        return
    ____adaption____Gramo_vec_len_to_scaling_factor()
    pass

if "all the setters" and False:
    def ____test____all_the_setters_of___Gramo_vec_len_to_scaling_factor():
        if "init         and         direct setter" and True:
            gramo = Gramo_vec_len_to_scaling_factor()
            assert gramo.scaling_factor___s == 1.
            assert gramo.epsilon___s == 0.001
            assert gramo.mul_me__when_g_too_small___s == 200.

            gramo.set_scaling_factor(1.234)
            assert gramo.scaling_factor___s == 1.234
            gramo.set_epsilon(0.002)
            assert gramo.epsilon___s == 0.002
            gramo.set_mul_me__when_g_too_small(112.233)
            assert gramo.mul_me__when_g_too_small___s == 112.233

            gramo = Gramo_vec_len_to_scaling_factor()
            gramo.set_mode___vec_len_to_sqrt_of_dim(10)
            assert gramo.scaling_factor___s == torch.sqrt(torch.tensor(10.))
            gramo.set_mode___vec_len_to_1()
            assert gramo.scaling_factor___s == 1.
            pass#/ test

        if "smooth setter" and True:
            gramo.set_smooth(scaling_factor=100., epsilon=1.)
            assert gramo.scaling_factor___s == 100.
            assert gramo.epsilon___s == 1.
            assert gramo.mul_me__when_g_too_small___s == 100.

            gramo.set_smooth(scaling_factor=100., epsilon=1., smooth_when_1=5.)
            assert gramo.scaling_factor___s == 100.
            assert gramo.epsilon___s == 1.
            assert gramo.mul_me__when_g_too_small___s == 20.


            gramo.set_smooth(scaling_factor=100., mul_me__when_g_too_small=5.)
            assert gramo.scaling_factor___s == 100.
            assert gramo.epsilon___s == 20.
            assert gramo.mul_me__when_g_too_small___s == 5.
            
            gramo.set_smooth(scaling_factor=100., mul_me__when_g_too_small=5., smooth_when_1=5)
            assert gramo.scaling_factor___s == 100.
            assert gramo.epsilon___s == 4.
            assert gramo.mul_me__when_g_too_small___s == 5.


            gramo.set_smooth(epsilon=2., mul_me__when_g_too_small=5.)
            assert gramo.scaling_factor___s == 10.
            assert gramo.epsilon___s == 2.
            assert gramo.mul_me__when_g_too_small___s == 5.
            
            gramo.set_smooth(epsilon=2., mul_me__when_g_too_small=5., smooth_when_1=7.)
            assert gramo.scaling_factor___s == 70.
            assert gramo.epsilon___s == 2.
            assert gramo.mul_me__when_g_too_small___s == 5.

            gramo.set_smooth___changes_only__mul_me__when_g_too_small()
            assert gramo.scaling_factor___s == 70.
            assert gramo.epsilon___s == 2.
            assert gramo.mul_me__when_g_too_small___s == 35.



            for _ in range(100):
                ori___scaling_factor = torch.rand([]).item()+1.5
                ori___epsilon = torch.rand([]).item()+0.5
                gramo.set_smooth(scaling_factor=ori___scaling_factor, epsilon=ori___epsilon)
                ori___mul_me__when_g_too_small = gramo.mul_me__when_g_too_small___s.item()
                gramo.set_smooth___changes_only__mul_me__when_g_too_small()

                assert gramo.scaling_factor___s == ori___scaling_factor
                assert gramo.epsilon___s == ori___epsilon
                assert _tensor_equal(gramo.mul_me__when_g_too_small___s, [ori___mul_me__when_g_too_small])


                gramo.set_smooth(scaling_factor=ori___scaling_factor, epsilon=ori___epsilon, smooth_when_1=2.)
                ori___mul_me__when_g_too_small = gramo.mul_me__when_g_too_small___s.item()
                gramo.set_smooth___changes_only__mul_me__when_g_too_small()

                assert gramo.scaling_factor___s == ori___scaling_factor
                assert gramo.epsilon___s == ori___epsilon
                assert _tensor_equal(gramo.mul_me__when_g_too_small___s, [ori___mul_me__when_g_too_small *2.])

                pass#for _ 
            pass#/ test

        if "set_protect_accuracy" and True:
            gramo = Gramo_vec_len_to_scaling_factor()
            gramo.set_protect_accuracy()
            assert gramo.protect_accuracy___1_flag == True
            gramo.set_protect_accuracy(False)
            assert gramo.protect_accuracy___1_flag == False    
            gramo.set_protect_accuracy()
            assert gramo.protect_accuracy___1_flag == True
            pass#/ test
        
        if "set_protect_accuracy" and True:
            gramo = Gramo_vec_len_to_scaling_factor()
            assert gramo.scaling_factor___s == 1.
            gramo.scale_scaling_factor(2.)
            assert gramo.scaling_factor___s == 2.
            gramo.scale_scaling_factor(3.)
            assert gramo.scaling_factor___s == 6.
            pass#/ test

        if "mode" and True:
            gramo = Gramo_vec_len_to_scaling_factor(protect_accuracy=False)#debug purpose.
            assert gramo.scaling_factor___s == 1.
            assert gramo.epsilon___s == 0.001
            assert gramo.mul_me__when_g_too_small___s == 200.
            #<  Set to vec len to 1 mode.     It's already in this mode. The next line is a no-op here.
            gramo.set_mode___vec_len_to_1()
            assert gramo.scaling_factor___s == 1.
            assert gramo.epsilon___s == 0.001
            assert gramo.mul_me__when_g_too_small___s == 200.

            input = torch.randn(size=[2,10], requires_grad=True)
            output:torch.Tensor = gramo(input)
            output.backward(gradient=torch.ones(size=[2,10]), inputs=[input])
            assert isinstance(input.grad, torch.Tensor)
            _temp__vec_len_of_output = get_vector_length(input.grad)
            assert _tensor_shape_check(_temp__vec_len_of_output, 2)
            assert _tensor_equal(_temp__vec_len_of_output, [1., 1])

            input = torch.randn(size=[2,15], requires_grad=True)
            output:torch.Tensor = gramo(input)
            output.backward(gradient=torch.ones(size=[2,15]), inputs=[input])
            assert isinstance(input.grad, torch.Tensor)
            _temp__vec_len_of_output = get_vector_length(input.grad)
            assert _tensor_shape_check(_temp__vec_len_of_output, 2)
            assert _tensor_equal(_temp__vec_len_of_output, [1., 1])

            #<  Set to avg len to 1 mode.
            gramo.set_mode___vec_len_to_sqrt_of_dim(10)
            assert gramo.scaling_factor___s == torch.sqrt(torch.tensor(10.))
            assert gramo.scaling_factor___s>3.
            assert gramo.scaling_factor___s<3.2
            assert gramo.epsilon___s == 0.001
            assert gramo.mul_me__when_g_too_small___s == 200.

            input = torch.randn(size=[2,10], requires_grad=True)
            output:torch.Tensor = gramo(input)
            output.backward(gradient=torch.ones(size=[2,10])*23.456, inputs=[input])
            assert isinstance(input.grad, torch.Tensor)
            _temp__vec_len_of_output = get_vector_length(input.grad)
            assert _tensor_equal(input.grad, torch.ones(size=[2,10]))

            #<  Set to vec len to 1 mode.
            gramo.set_mode___vec_len_to_1()
            assert gramo.scaling_factor___s == 1.
            assert gramo.epsilon___s == 0.001
            assert gramo.mul_me__when_g_too_small___s == 200.
            pass#/ test

        return
    ____test____all_the_setters_of___Gramo_vec_len_to_scaling_factor()
    pass

















class _only_for___Grad_inspector___to_use___Function_class(torch.autograd.Function):
    r'''a special version in 2026. 
    
    input param:
    >>> x:torch.Tensor (must be set as require_grad = True)
    >>> fake_data:torch.Tensor (must be set as require_grad = True, and the same shape as x)

    retur type: torch.Tensor
    '''
    @staticmethod
    #def forward(*args: Any, **kwargs: Any)->Any:
    def forward(x:torch.Tensor, fake_data:torch.Tensor, *args: Any, **kwargs: Any)->Any:
        if x.requires_grad:
            assert fake_data.requires_grad == True
            assert x.shape == fake_data.shape
            assert x.dtype == fake_data.dtype
            pass
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        #x:torch.Tensor
        x = inputs[0]
        x_requires_grad = torch.tensor(x.requires_grad, device=x.device)

        ctx.save_for_backward(x_requires_grad)
        return

    @staticmethod
    def backward(ctx, grad):#->tuple[Optional[torch.Tensor], None, None, None]:


        #g_in___b_o:torch.Tensor
        # scaling_factor___s:torch.Tensor
        # epsilon___s:torch.Tensor
        # mul_me__when_g_too_small___s:torch.Tensor
        # protect_accuracy___1_bool:torch.Tensor


        (x_requires_grad,) = ctx.saved_tensors
        assert isinstance(x_requires_grad, torch.Tensor)
        if x_requires_grad:
            return grad, grad
        else:
            return None, None
        #end of function
    pass  # class

if "basic behavior" and False:
    def ____test____basic_behavior_____only_for___Grad_inspector___to_use___Function_class():
        if "basic behavior" and True:
            import random
            for _ in range(5):
                size = [random.randint(2,100)]
                input = torch.randn(size=size,      requires_grad=True)
                fake_data = torch.randn_like(input, requires_grad=True)
                grad_in = torch.randn_like(input)
                output:torch.Tensor = _only_for___Grad_inspector___to_use___Function_class.apply(input, fake_data)
                output.backward(gradient = grad_in, inputs=[input, fake_data])
                assert input.grad is not None
                assert input.grad.eq(grad_in).all()
                assert fake_data.grad is not None
                assert fake_data.grad.eq(grad_in).all()
                pass#for _

            for _ in range(5):
                size = [random.randint(2,30), random.randint(2,30)]
                input = torch.randn(size=size,      requires_grad=True)
                fake_data = torch.randn_like(input, requires_grad=True)
                grad_in = torch.randn_like(input)
                output:torch.Tensor = _only_for___Grad_inspector___to_use___Function_class.apply(input, fake_data)
                output.backward(gradient = grad_in, inputs=[input, fake_data])
                assert input.grad is not None
                assert input.grad.eq(grad_in).all()
                assert fake_data.grad is not None
                assert fake_data.grad.eq(grad_in).all()
                pass#for _

            for _ in range(5):
                size = [random.randint(2,20), random.randint(2,20), random.randint(2,20)]
                input = torch.randn(size=size,      requires_grad=True)
                fake_data = torch.randn_like(input, requires_grad=True)
                grad_in = torch.randn_like(input)
                output:torch.Tensor = _only_for___Grad_inspector___to_use___Function_class.apply(input, fake_data)
                output.backward(gradient = grad_in, inputs=[input, fake_data])
                assert input.grad is not None
                assert input.grad.eq(grad_in).all()
                assert fake_data.grad is not None
                assert fake_data.grad.eq(grad_in).all()
                pass#for _

            pass#/ test

        if "device adaptive" and True:
            device = 'cuda'
            input = torch.randn(size=size,      requires_grad=True, device=device)
            fake_data = torch.randn_like(input, requires_grad=True)
            grad_in = torch.randn_like(input)
            output:torch.Tensor = _only_for___Grad_inspector___to_use___Function_class.apply(input, fake_data)
            output.backward(gradient = grad_in, inputs=[input, fake_data])
            assert input.grad is not None
            assert input.grad.eq(grad_in).all()
            assert input.grad.device.type == device
            assert fake_data.grad is not None
            assert fake_data.grad.eq(grad_in).all()
            assert fake_data.grad.device.type == device
            pass#/ test

        '''this tool doesn't provide dtype adaption.'''
        '''this tool doesn't provide dtype adaption.'''
        '''this tool doesn't provide dtype adaption.'''
        #if "dtype adaptive" and False:

        return
    ____test____basic_behavior_____only_for___Grad_inspector___to_use___Function_class()
    pass















class Grad_inspector(torch.nn.Module):
    r"""to help you check the grad.

    """
    fake_data:torch.nn.Parameter
    clear_grad_in___step:bool

    def __init__(self, clear_grad_in___step = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #<  safety
        self.fake_data = torch.nn.Parameter(torch.empty(size=[]))
        self.clear_grad_in___step = clear_grad_in___step
        return

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if x.requires_grad == False:
            return x

        _flag___needs_a_new_param = False
        if self.fake_data.nelement() != x.nelement():
            _flag___needs_a_new_param = True
            pass
        elif self.fake_data.dtype != x.dtype:
            _flag___needs_a_new_param = True
            pass
        elif self.fake_data.device != x.device:
            _flag___needs_a_new_param = True
            pass
        else:
            #nothing here. keep the flag as false
            pass

        if _flag___needs_a_new_param:
            self.fake_data = torch.nn.Parameter(torch.empty_like(x))
            pass
        else:
            self.fake_data.reshape_as(x)
            self.fake_data.grad = None
            pass

        assert self.fake_data.shape == x.shape
        assert self.fake_data.dtype == x.dtype
        assert self.fake_data.device == x.device
        assert self.fake_data.grad is None
        
        return _only_for___Grad_inspector___to_use___Function_class.apply(x, self.fake_data)

    def get_grad(self)->torch.Tensor:
        assert isinstance(self.fake_data.grad)
        return self.fake_data.grad
    
    def step(self)->None:
        '''If the flag:clear_grad_in___step is True, then the gradient is set to None in this function.'''
        if self.clear_grad_in___step:
            self.fake_data.grad = None
            pass
        return 


    def abs_mean(self)->float:
        return self.fake_data.grad.abs().mean().item()
    
    def log10_avg_safe(self)->float:
        return log10_avg_safe(self.fake_data.grad).item()

    
    # def extra_repr(self) -> str:
    #     return f'scaling_factor={self.scaling_factor.item():.4e}, epsilon={self.epsilon.item():.4e}, mul_me_when_g_too_small={self.mul_me_when_g_too_small.item():.4e}'
    pass#end of class

if "basic behavior" and False:
    def ____equivalence____Grad_inspector():

        if "basic behavior" and True:
            import random
            for ori___dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16]:
                for dtype in [torch.float32, torch.float16, torch.float64, torch.bfloat16]:
                    for ori___device in ['cpu', 'cuda']:
                        for device in ['cpu', 'cuda']:
                            for _ in range(16):

                                if random.random()< 0.3:
                                    size = [random.randint(2,100)]
                                    pass
                                if random.random()< 0.5:
                                    size = [random.randint(2,30), random.randint(2,30)]
                                    pass
                                else:
                                    size = [random.randint(2,20), random.randint(2,20), random.randint(2,20)]
                                    pass

                                input = torch.randn(size=size, dtype=dtype, device=device, requires_grad=True)
                                the_inspector = Grad_inspector()
                                the_inspector = the_inspector.to(ori___device).to(ori___dtype)
                                assert the_inspector.fake_data.device.type == ori___device
                                assert the_inspector.fake_data.dtype == ori___dtype

                                output:torch.Tensor = the_inspector(input)
                                grad_in = torch.randn_like(output)
                                output.backward(gradient=grad_in, inputs=[input, the_inspector.fake_data])

                                assert isinstance(input.grad, torch.Tensor)
                                assert            input.grad.eq(grad_in).all()
                                assert isinstance(the_inspector.fake_data.grad, torch.Tensor)
                                assert            the_inspector.fake_data.grad.eq(grad_in).all()

                                pass#for _ 
                            pass#for device 
                        pass#for ori___device 
                    pass#for dtype
                pass#for ori___dtype
            pass#/ test

        if "clear_grad_in___step" and True:
            #step
            input = torch.randn(size=size, dtype=dtype, device=device, requires_grad=True)
            the_inspector = Grad_inspector(clear_grad_in___step=True)#default

            output:torch.Tensor = the_inspector(input)
            grad_in = torch.randn_like(output)
            output.backward(gradient=grad_in, inputs=[input, the_inspector.fake_data])
            assert isinstance(the_inspector.fake_data.grad, torch.Tensor)

            the_inspector.step()
            assert the_inspector.fake_data.grad is None  # cleared


            #step       but not to clear it.
            input = torch.randn(size=size, dtype=dtype, device=device, requires_grad=True)
            the_inspector = Grad_inspector(clear_grad_in___step=False)

            output:torch.Tensor = the_inspector(input)
            grad_in = torch.randn_like(output)
            output.backward(gradient=grad_in, inputs=[input, the_inspector.fake_data])
            assert isinstance(the_inspector.fake_data.grad, torch.Tensor)

            the_inspector.step()
            assert isinstance(the_inspector.fake_data.grad, torch.Tensor)  # NOT cleared


            #forward
            input = torch.randn(size=size, dtype=dtype, device=device, requires_grad=True)
            the_inspector = Grad_inspector(clear_grad_in___step=True)#default

            output = the_inspector(input)
            grad_in = torch.randn_like(output)
            output.backward(gradient=grad_in, inputs=[input, the_inspector.fake_data])
            assert isinstance(the_inspector.fake_data.grad, torch.Tensor)

            output = the_inspector(input)
            assert the_inspector.fake_data.grad is None  # cleared

            pass#/ test

        return
    ____equivalence____Grad_inspector()
    pass

if "measure functions" and True:
    def ____measure_functions____Grad_inspector():

        if "abs mean" and True:
            import random
            for _ in range(16):

                if random.random()< 0.3:
                    size = [random.randint(2,100)]
                    pass
                if random.random()< 0.5:
                    size = [random.randint(2,30), random.randint(2,30)]
                    pass
                else:
                    size = [random.randint(2,20), random.randint(2,20), random.randint(2,20)]
                    pass

                input = torch.randn(size=size, requires_grad=True)
                the_inspector = Grad_inspector()
                the_inspector = the_inspector

                output:torch.Tensor = the_inspector(input)
                grad_in = torch.randn_like(output)
                output.backward(gradient=grad_in, inputs=[input, the_inspector.fake_data])

                assert grad_in.abs().mean() == the_inspector.abs_mean()
                assert log10_avg_safe(grad_in) == the_inspector.log10_avg_safe()

                pass#for _ 
            pass#/ test

        return
    ____measure_functions____Grad_inspector()
    pass



