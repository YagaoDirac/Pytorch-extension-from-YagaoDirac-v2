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




def dtype_upgrade(input:torch.dtype)->torch.dtype:
    '''this function is adapted to pytorch. When shift to other framework, recheck it.'''
    if input == torch.float64:
        return torch.float64
    else:
        return torch.float32




# def get_vector_length(input:torch.Tensor, result_dtype = torch.float64)->torch.Tensor:
#     _temp = input*input
#     _temp = _temp.sum(dim=-1, dtype=result_dtype)
#     _temp.sqrt_()
#     return _temp
# #copied from the Util.py file. No validation here.













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
































1w
1w
1w
1w





assert False, "继续"
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
    def forward(x:torch.Tensor,scaling_factor:torch.Tensor,epsilon:torch.Tensor,\
                    mul_me__when_g_too_small__s:torch.Tensor, protect_accuracy:torch.Tensor,\
                        mode__mean_true_sum_false:torch.Tensor,\
                            *args: Any, **kwargs: Any)->Any:
        assert x.shape.__len__() == 2, "Only accept rank-2 tensor. The shape should be[batch, something]"
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        #x:torch.Tensor
        x = inputs[0]
        scaling_factor = inputs[1]
        epsilon = inputs[2]
        mul_me__when_g_too_small__s = inputs[3]
        protect_accuracy = inputs[4]
        mode__mean_true_sum_false = inputs[5]
        x_needs_grad = torch.tensor([x.requires_grad])
        ctx.save_for_backward(scaling_factor, epsilon, mul_me__when_g_too_small__s, protect_accuracy,
                                x_needs_grad, mode__mean_true_sum_false)
        return
        #return super().setup_context(ctx, inputs, output)

    @staticmethod
    def backward(ctx, g_in__b_o):#->tuple[Optional[torch.Tensor], None, None, None]:
        #super().backward()
        # scaling_factor:torch.Tensor
        # epsilon:torch.Tensor
        # mul_me_when_g_too_small:torch.Tensor
        # protect_accuracy:torch.Tensor
        (scaling_factor, epsilon, mul_me__when_g_too_small__s, protect_accuracy,
                                x_needs_grad, mode__mean_true_sum_false) = ctx.saved_tensors
        
        #grad_for_x_b_o:Optional[torch.Tensor] = None
        grad_for_x__b_o = None
        
        if x_needs_grad:
            #dim__s:torch.Tensor
            #dim__s = torch.tensor([g_in__b_o.shape[-1]], dtype=torch.float64, device=g_in__b_o.device)
            #dim__s = g_in__b_o.shape[1]
            #<  infra>
            dim__s = torch.tensor([g_in__b_o.shape[-1]], dtype=torch.int64, device=g_in__b_o.device)
            #</ infra>
            # old code mul_me_when_g_too_small = mul_me_when_g_too_small_per_element#*dim__s

            #avg_length_per_element_b_1 = (g_in_b_o.mul(g_in_b_o).sum(dim=1,keepdim = True)/dim__s).sqrt()
            
            #<  avg_length_per_element__b_1>
            #avg_length_per_element__b_1:torch.Tensor
            avg_length_per_element__or_depends_on_mode__b_1 = g_in__b_o*g_in__b_o
            if mode__mean_true_sum_false:#"mean"
                avg_length_per_element__or_depends_on_mode__b_1 = avg_length_per_element__or_depends_on_mode__b_1.mean(
                                                                                                dim=1,keepdim = True)
                #real_epsilon = epsilon
                #real_mul_me__when_g_too_small__s = mul_me__when_g_too_small__s
                pass
            else:#"sum"
                avg_length_per_element__or_depends_on_mode__b_1 = avg_length_per_element__or_depends_on_mode__b_1.sum(
                                                                                                dim=1,keepdim = True)
                # ____temp_sqrt_of_dim = dim__s.detach().clone()
                # ____temp_sqrt_of_dim = ____temp_sqrt_of_dim.to(torch.float64)
                # ____temp_sqrt_of_dim.sqrt_()
                #real_epsilon = epsilon#                                        *____temp_sqrt_of_dim
                #real_mul_me__when_g_too_small__s = mul_me__when_g_too_small__s#/____temp_sqrt_of_dim
                pass
            avg_length_per_element__or_depends_on_mode__b_1.sqrt_()
            #</ avg_length_per_element__b_1>
            
            #<  mul_me>
            mul_me__when_g_is_ok__raw__b_1 = scaling_factor/avg_length_per_element__or_depends_on_mode__b_1
            # ^^^^^ optimizable ^^^^^
            #                  torch.Tensor.nan_to_num_(posinf=,neginf=)
            mul_me__when_g_is_ok__raw__b_1.nan_to_num_( posinf = 42, 
                                                        neginf = 42)
            #div doesn't provide nan. This protection is overwritten below when calc mul_me__b_1.
            
            #old code. The g_in should not be too big. And even if the g_in is too big, it's still ok to shrink it back 
            # also, the avg_length_per_element__or_depends_on_mode__b_1.le(real_epsilon) works for basically the same purpose.
            # flag__too_big__b_1 = mul_me__when_g_is_ok__raw__b_1.gt(real_mul_me__when_g_too_small__s*1024.)#is this needed?
            
            # old code.
            # mul_me__when_g_is_ok__b_1 = flag__not_too_big__b_1              *mul_me__when_g_is_ok__raw__b_1 + \
            #                             flag__not_too_big__b_1.logical_not()*real_mul_me__when_g_too_small__s#this is input, 1e-3 by default
            
            # too_small_b_1:torch.Tensor
            flag__length_too_small__b_1 = avg_length_per_element__or_depends_on_mode__b_1.lt(epsilon)#*dim__s)
            
            #flag__needs_default_value__b_1 = flag__too_big__b_1.logical_and(flag__too_small__b_1)
            
            
            mul_me__b_1=flag__length_too_small__b_1.logical_not() * mul_me__when_g_is_ok__raw__b_1 + \
                        flag__length_too_small__b_1               * mul_me__when_g_too_small__s#this is input, 1e-3 by default
            mul_me__b_1 = mul_me__b_1.to(g_in__b_o.dtype)
            #</ mul_me>
            
            #mul_me__b_1:torch.Tensor
            
            if protect_accuracy:
                mul_me__b_1.log2_().add_(0.5).floor_()# nearest power of 2. This floor func returns fp. It works. No need to convert it to integer.
                mul_me__b_1.exp2_()
                pass
            
            # grad_for_x_b_o:torch.Tensor
            #grad_for_x__b_o = g_in__b_o*mul_me__b_1
            grad_for_x__b_o = g_in__b_o*(mul_me__b_1.expand([-1,dim__s.item()]))
            pass

        return grad_for_x__b_o, None,None,None,None,None

    pass  # class

if '''dim adaptive gramo''' and __DEBUG_ME__() and True:
    def dim_adaptive_gramo___GradientModificationFunction__mean_len_of_element_to_1():
        import math
        scaling_factor = torch.tensor([1.])
        epsilon=torch.tensor([1e-3])
        mul_me_when_g_too_small = torch.tensor([10.])#the new default is 100. this is only for test.
        protect_accuracy = torch.tensor(False)
        mode__mean = torch.tensor(True)
        
        a = torch.zeros([5,2], requires_grad=True, dtype=torch.float16)
        b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                        mul_me_when_g_too_small,protect_accuracy,mode__mean)
        g_in = torch.tensor([[0.1,0.2],[0.01,0.02,],[0.001,0.002],[1e-4,2e-4],[1e-5,2e-5]], dtype=torch.float16)
        torch.autograd.backward(b, g_in,inputs= a)
        assert a.grad is not None
        assert _tensor_equal(a.grad[:3], torch.tensor( [[6.3232e-01, 1.2646e+00],
                                                        [6.3232e-01, 1.2646e+00],
                                                        [6.3232e-01, 1.2646e+00]], dtype=torch.float16), epsilon=1e-3)
        assert _tensor_equal(a.grad[3:], torch.tensor( [[1.0004e-03, 2.0008e-03],
                                                        [1.0014e-04, 2.0027e-04]], dtype=torch.float16), epsilon=1e-7)
        
        
        a = torch.zeros([5,1], requires_grad=True)
        b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                        mul_me_when_g_too_small,protect_accuracy,mode__mean)
        g_in = torch.tensor([[0.1],[0.01],[0.001],[1e-4],[1e-5]], dtype=torch.float16)
        torch.autograd.backward(b, g_in,inputs= a)
        assert a.grad is not None
        assert _tensor_equal(a.grad[:3], torch.tensor( [[1.],
                                                        [1.],
                                                        [1.]], dtype=torch.float16), epsilon=1e-3)
        assert _tensor_equal(a.grad[3:], torch.tensor( [[1.0002e-03],
                                                        [1.0014e-04]], dtype=torch.float16), epsilon=1e-6)
        
        
        
        
        "mean mode has nothing to do with dimention."
        scaling_factor = torch.tensor([1.])
        epsilon=torch.tensor([1e-3])
        mul_me_when_g_too_small = torch.tensor([10.])#the new default is 100. this is only for test.
        protect_accuracy = torch.tensor(False)
        mode__mean = torch.tensor(True)
        
        for dim in [2,5,11,5555]:#1 doesn't work.
            a = torch.empty([3,dim], requires_grad=True)
            b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                            mul_me_when_g_too_small,protect_accuracy,mode__mean)
            g_in = torch.tensor([[0.0011],[0.001],[0.0009]]).repeat([1,dim])
            torch.autograd.backward(b, g_in,inputs= a)
            assert a.grad is not None
            _first_row_of_grad = a.grad[:,1]
            _ref_grad = torch.tensor([1.,1.,0.0009*10])
            assert _tensor_equal(_first_row_of_grad, _ref_grad, epsilon=1e-4)
            pass
        
        "sum mode is the per-vector mode."
        mode__sum = torch.tensor(False)
        for dim in [2,5,11,5555]:#1 doesn't work.
            a = torch.empty([3,dim], requires_grad=True, dtype=torch.float32)
            b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                            mul_me_when_g_too_small,protect_accuracy,mode__sum)
            g_in = (torch.tensor([[0.0011],[0.0010],[0.0009]])/math.sqrt(dim)).repeat([1,dim])
            
            torch.autograd.backward(b, g_in,inputs= a)
            assert a.grad is not None 
            _first_row_of_grad = a.grad[:,1]
            _ref_grad = torch.tensor([1., 1, -42424242])/math.sqrt(dim)
            _ref_grad[2] = 0.0009/math.sqrt(dim) * 10
            assert _ref_grad[0] == _ref_grad[1]
            assert _ref_grad[0] > _ref_grad[2]#discontinuous.
            assert _tensor_equal(_first_row_of_grad, _ref_grad, epsilon=1e-4)
            pass
            
            
        "when the g_in maybe too small."
        import math, random
        for _ in range(116):
            dim = int(math.pow(10, random.random()*3.+0.9))
            assert dim >=7 and dim <10000
            a = torch.empty(size=[1,dim], requires_grad=True)
            b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                            mul_me_when_g_too_small,protect_accuracy,mode__sum)
            g_in = torch.randn_like(b)*0.0001
            torch.autograd.backward(b, g_in,inputs= a)
            assert a.grad is not None
            grad_length = (a.grad*a.grad).sum(dim=1).sqrt()
            assert grad_length.shape == torch.Size([1])
            assert _tensor_equal(grad_length, [1.]) or grad_length.lt(0.011)
            pass
        
        return
    dim_adaptive_gramo___GradientModificationFunction__mean_len_of_element_to_1()
    pass

if '''dtype adaption.''' and __DEBUG_ME__() and True:
    def dtype_adaption___GradientModificationFunction__mean_len_of_element_to_1():
        scaling_factor = torch.tensor([1.], dtype=torch.float64)
        epsilon=torch.tensor([1e-5], dtype=torch.float32)
        mul_me_when_g_too_small = torch.tensor([1e3], dtype=torch.float16)
        protect_accuracy = torch.tensor(False)
        mode__mean_true_sum_false = torch.tensor(True)
        
        a = torch.tensor([[0.]], requires_grad=True, dtype=torch.float16)
        original_dtype = a.dtype
        b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                        mul_me_when_g_too_small,protect_accuracy,mode__mean_true_sum_false)
        ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
        g_in = torch.tensor([[1.]], dtype=torch.float16)
        torch.autograd.backward(b, g_in,inputs= a)
        assert a.grad
        assert a.grad.dtype == original_dtype
        
        return 
    dtype_adaption___GradientModificationFunction__mean_len_of_element_to_1()
    pass

if '''device adaption''' and __DEBUG_ME__() and True:
    def device_adaption___GradientModificationFunction__mean_len_of_element_to_1():
        scaling_factor = torch.tensor([1.]).cuda()
        epsilon=torch.tensor([1e-5]).cuda()
        mul_me_when_g_too_small = torch.tensor([1e3]).cuda()
        protect_accuracy = torch.tensor(False)
        mode__mean_true_sum_false = torch.tensor(True)
        
        a = torch.tensor([[0.]], requires_grad=True).cuda()
        protect_accuracy = torch.tensor(False)
        b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                    mul_me_when_g_too_small,protect_accuracy,mode__mean_true_sum_false)
        g_in = torch.tensor([[1.]]).cuda()
        torch.autograd.backward(b, g_in,inputs= a)
        assert a.grad is not None
        assert a.grad.device == torch.device('cuda', index=0)
        assert a.grad.device.type == 'cuda'
        
        return 
    device_adaption___GradientModificationFunction__mean_len_of_element_to_1()
    pass

if "acc protection feature." and __DEBUG_ME__() and True:
    def acc_protection_feature___GradientModificationFunction__mean_len_of_element_to_1():
        scaling_factor = torch.tensor([1.], dtype=torch.float64)
        epsilon=torch.tensor([1e-3], dtype=torch.float32)
        mul_me_when_g_too_small = torch.tensor([10], dtype=torch.float16)
        protect_accuracy = torch.tensor(True)
        mode__mean_true_sum_false = torch.tensor(True)
        
        a = torch.zeros([5,2], requires_grad=True, dtype=torch.float16)
        b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                        mul_me_when_g_too_small,protect_accuracy,mode__mean_true_sum_false)
        g_in = torch.tensor([[0.1,0.2],[0.01,0.02,],[0.001,0.002],[1e-4,2e-4],[1e-5,2e-5]], dtype=torch.float16)
        torch.autograd.backward(b, g_in,inputs= a)
        assert a.grad is not None
        assert _tensor_equal(a.grad[:3], torch.tensor( [[0.79980, 1.5996],
                                                        [0.64014, 1.2803],
                                                        [0.51221, 1.0244]], dtype=torch.float16), epsilon=1e-4)
        assert _tensor_equal(a.grad[3:], torch.tensor( [[8.0013e-04, 1.6003e-03],
                                                        [8.0109e-05, 1.6022e-04]], dtype=torch.float16), epsilon=1e-7)
        
        _the_real_modification:torch.Tensor = a.grad.div(g_in)
        assert _tensor_equal(_the_real_modification, torch.tensor( [[  8.,   8.],
                                                                    [ 64.,  64.],
                                                                    [512., 512.],
                                                                    [  8.,   8.],
                                                                    [  8.,   8.]], dtype=torch.float16))
        
        return 
    acc_protection_feature___GradientModificationFunction__mean_len_of_element_to_1()
    pass

if "protection behavior" and __DEBUG_ME__() and True:
    def protection_behavior____GradientModificationFunction__mean_len_of_element_to_1():
        #0.1, 10
        scaling_factor = torch.tensor([1.])
        epsilon=torch.tensor([0.1])
        mul_me_when_g_too_small = torch.tensor([10.])
        protect_accuracy = torch.tensor(False)
        mode__mean = torch.tensor(True)
        
        a = torch.empty(size=[5,1], requires_grad=True)
        b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                        mul_me_when_g_too_small,protect_accuracy,mode__mean)
        g_in = torch.tensor([[0.12],[0.11],[0.1],[0.09],[0.08]])
        torch.autograd.backward(b, g_in,inputs= a)
        assert a.grad is not None
        assert _tensor_equal(a.grad, torch.tensor([[1.],[1.],[1.],[0.9],[0.8]]))
        
        #0.1, 2
        scaling_factor = torch.tensor([1.])
        epsilon=torch.tensor([0.1])
        mul_me_when_g_too_small = torch.tensor([2.])
        protect_accuracy = torch.tensor(False)
        mode__mean = torch.tensor(True)
        
        a = torch.empty(size=[5,1], requires_grad=True)
        b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                        mul_me_when_g_too_small,protect_accuracy,mode__mean)
        g_in = torch.tensor([[0.12],[0.11],[0.1],[0.09],[0.08]])
        torch.autograd.backward(b, g_in,inputs= a)
        assert a.grad is not None
        assert _tensor_equal(a.grad, torch.tensor([[1.],[1.],[1.],[0.18],[0.16]]))
        #                                      discontinuous  ^^^^ ^^^^^
        
        
        #0.1, 2
        scaling_factor = torch.tensor([1.])
        epsilon=torch.tensor([0.1*3])
        mul_me_when_g_too_small = torch.tensor([2.])
        protect_accuracy = torch.tensor(False)
        mode__sum = torch.tensor(False)
        
        dim = 3*3
        a = torch.empty(size=[5,1], requires_grad=True).repeat([1,dim])
        b = GradientModificationFunction__mean_len_of_element_to_1.apply(a,scaling_factor,epsilon,
                                        mul_me_when_g_too_small,protect_accuracy,mode__sum)
        g_in = torch.tensor([[0.12],[0.11],[0.1],[0.09],[0.08]]).repeat([1,dim])
        torch.autograd.backward(b, g_in,inputs= a)
        assert a.grad is not None
        grad_length = (a.grad*a.grad).sum(dim=1).sqrt()
        assert _tensor_equal(grad_length, torch.tensor([1., 1, 1, 0.54,0.48]))
        #                                     discontinuous    ^  ^^^^
        #ref
        ori_grad_length = (g_in*g_in).sum(dim=1).sqrt()
        assert _tensor_equal(ori_grad_length, [0.12*3,0.11*3,0.1*3,0.09*3,0.08*3])
        
        return 
    protection_behavior____GradientModificationFunction__mean_len_of_element_to_1()
    pass




class GradientModification__mean_len_of_something_to_1(torch.nn.Module):
    r"""Oh, yeah, first thing first. I call this layer the gramo. This tool is the beginning of my ai career.
    
    This autograd function scale grad to have a sqrt([mode](square)) to specified number(1 by default).
    The [mode] is either mean or sum.
    
    It's designed mainly to help analogy signal handling with error propagation.
    
    Remember to use adaptive learning rate. Mse doesn't help auto shrink the updating strength when gramo is used.
    To access the learning rate, you usually need some thing like:
    >>> lr:float = optimizer.param_groups[0]["lr"]
    It's much easier to deal with such cases with a breakpoint. To learn how to setup breakpoints, search online.
    
    The protect_binary_accuracy and mode:
    
    If this layer is in halfway in a gradient chain, especially the main grad chain, protect_binary_accuracy = True, mode="mean". 
    
    If this layer is in right in front of a learning param, protect_binary_accuracy can be either, mode="mean". 
    
    If this layer is in right in front of a learning param which is meant to have a vector-length of one, 
    protect_binary_accuracy = False, mode="sum", if the param is a single vector, scaling_factor = 1, 
    if the param is something similar to standard orthogonal matrix of size [dim,dim](each row vector or column
    vector are all of length 1.), scaling_factor = dim.
    """
    protect_binary_accuracy:torch.nn.Parameter
    per_what:Literal["element", "vector"]
    
    scaling_factor:torch.nn.Parameter
    epsilon:torch.nn.Parameter 
    mul_me_when_g_too_small:torch.nn.Parameter
    def __init__(self, protect_binary_accuracy:bool, per_what:Literal["element", "vector"] = "element",\
                    scaling_factor:float = 1., 
                    epsilon=1e-3, #it was 1e-5
                    mul_me_when_g_too_small = 1e2, #it was 1e3
                        device=None,dtype=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(protect_binary_accuracy, bool), "the api is updated."
        self.protect_binary_accuracy = torch.nn.Parameter(torch.tensor(protect_binary_accuracy, device=device, dtype=torch.bool), requires_grad=False)
        assert per_what in ["element", "vector"]
        self.per_what = per_what
        
        dtype = dtype_upgrade(dtype)
        self.scaling_factor          = torch.nn.Parameter(torch.tensor(scaling_factor,          device=device, dtype=dtype), requires_grad=False)
        self.epsilon                     = torch.nn.Parameter(torch.tensor(epsilon,                     device=device, dtype=dtype), requires_grad=False)
        self.mul_me_when_g_too_small = torch.nn.Parameter(torch.tensor(mul_me_when_g_too_small, device=device, dtype=dtype), requires_grad=False)
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        assert x.shape.__len__() == 2, "Only accept rank-2 tensor. The shape should be[batch, something]"

        
        _mode__mean_true_sum_false  = torch.tensor(self.per_what == "element", device=x.device)
        
        #forward(ctx, x:torch.Tensor, scaling_factor:torch.Tensor, epsilon=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction__mean_len_of_element_to_1.apply(x, self.scaling_factor,\
                self.epsilon, self.mul_me_when_g_too_small, self.protect_binary_accuracy, _mode__mean_true_sum_false)
    
    def set_scaling_factor(self, scaling_factor:float)->None:
        the_device = self.scaling_factor.device
        the_dtype = self.scaling_factor.dtype
        self.scaling_factor.data = torch.tensor(scaling_factor, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    #old code. I wrote this probably before the dim adaption. 
    # def scale_scaling_factor(self, by:float)->None:
    #     self.set_scaling_factor((self.scaling_factor*by).item())
    #     pass
    def set_epsilon(self, epsilon:float)->None:
        the_device = self.epsilon.device
        the_dtype = self.epsilon.dtype
        self.epsilon.data = torch.tensor(epsilon, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def set_mul_me_when_g_too_small(self, mul_me_when_g_too_small:float)->None:
        the_device = self.mul_me_when_g_too_small.device
        the_dtype = self.mul_me_when_g_too_small.dtype
        self.mul_me_when_g_too_small.data = torch.tensor(mul_me_when_g_too_small, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def set_protect_accuracy(self, protect_accuracy:bool)->None:
        the_device = self.protect_binary_accuracy.device
        the_dtype = self.protect_binary_accuracy.dtype
        self.protect_binary_accuracy.data = torch.tensor(protect_accuracy, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def set_mode(self, per_what:Literal["element", "vector"])->None:
        assert per_what in ["element", "vector"]
        self.per_what = per_what
        pass
    
    def extra_repr(self) -> str:
        return f'scaling_factor={self.scaling_factor.item():.4e}, epsilon={self.epsilon.item():.4e}, mul_me_when_g_too_small={self.mul_me_when_g_too_small.item():.4e}'

if '''all the setters''' and __DEBUG_ME__() and True:
    def all_the_setters____GradientModification__mean_len_of_element_to_1():
        model_GradientModification_v2_mean_abs_to_1 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False)
        assert model_GradientModification_v2_mean_abs_to_1.scaling_factor.requires_grad == False
        assert model_GradientModification_v2_mean_abs_to_1.epsilon.requires_grad == False
        assert model_GradientModification_v2_mean_abs_to_1.mul_me_when_g_too_small.requires_grad == False
        assert model_GradientModification_v2_mean_abs_to_1.protect_binary_accuracy == False
        assert model_GradientModification_v2_mean_abs_to_1.per_what == 'element'
        model_GradientModification_v2_mean_abs_to_1.set_scaling_factor(0.123)
        #1w 为什么不对啊？64吗？
        assert _float_equal(model_GradientModification_v2_mean_abs_to_1.scaling_factor.item(), 0.123)
        assert model_GradientModification_v2_mean_abs_to_1.scaling_factor.requires_grad == False
        model_GradientModification_v2_mean_abs_to_1.set_epsilon(0.234)
        assert _float_equal(model_GradientModification_v2_mean_abs_to_1.epsilon.item(), 0.234)
        assert model_GradientModification_v2_mean_abs_to_1.epsilon.requires_grad == False
        model_GradientModification_v2_mean_abs_to_1.set_mul_me_when_g_too_small(0.345)
        assert _float_equal(model_GradientModification_v2_mean_abs_to_1.mul_me_when_g_too_small.item(), 0.345)
        assert model_GradientModification_v2_mean_abs_to_1.mul_me_when_g_too_small.requires_grad == False
        model_GradientModification_v2_mean_abs_to_1.set_protect_accuracy(False)
        assert model_GradientModification_v2_mean_abs_to_1.protect_binary_accuracy == False
        assert model_GradientModification_v2_mean_abs_to_1.protect_binary_accuracy.requires_grad == False
        model_GradientModification_v2_mean_abs_to_1.set_mode('vector')
        assert model_GradientModification_v2_mean_abs_to_1.per_what == 'vector'
        return 
    all_the_setters____GradientModification__mean_len_of_element_to_1()
    pass

if '''dtype adaption.''' and __DEBUG_ME__() and True:
    def dtype_adaption____GradientModification__mean_len_of_element_to_1():
        input = torch.tensor([[1.]], requires_grad=True)
        target = torch.tensor([[0.]])
        model_GradientModification_v2_mean_abs_to_1 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False)
        model_GradientModification_v2_mean_abs_to_1.to(torch.float64)
        #model.to(torch.float16)

        loss_function = torch.nn.L1Loss()# the L1Loss function only provides the direction. It's the dirivitive of abs.
        optimizer = torch.optim.SGD([input], lr=0.1)
        for epoch in range(1):
            model_GradientModification_v2_mean_abs_to_1.train()
            pred = model_GradientModification_v2_mean_abs_to_1(input)
            assert pred.dtype == torch.float32
            loss = loss_function(pred, target)
            assert loss.dtype == torch.float32
            optimizer.zero_grad()
            loss.backward()#inputs = ?
            #optimizer.param_groups[0]["lr"] = 0.01
            assert input.grad is not None
            assert input.grad.item() == 1.
            assert input.grad.dtype == torch.float32

            optimizer.step()
            assert input.item() <0.9+0.00001
            assert input.item() >0.9-0.00001
            
            model_GradientModification_v2_mean_abs_to_1.eval()
            pass
        
        return 
    dtype_adaption____GradientModification__mean_len_of_element_to_1()
    pass

if '''init test''' and __DEBUG_ME__() and True:
    def _init_test____GradientModification__mean_len_of_element_to_1():
        layer_GradientModification_v2_mean_abs_to_1 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False, device='cuda')
        assert layer_GradientModification_v2_mean_abs_to_1.scaling_factor.device == torch.device('cuda', index=0)
        assert layer_GradientModification_v2_mean_abs_to_1.scaling_factor.dtype == torch.float32
        layer_GradientModification_v2_mean_abs_to_1 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False, dtype=torch.float64)
        assert layer_GradientModification_v2_mean_abs_to_1.scaling_factor.dtype == torch.float64
        layer_GradientModification_v2_mean_abs_to_1 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False, dtype=torch.float32)
        assert layer_GradientModification_v2_mean_abs_to_1.scaling_factor.dtype == torch.float32
        layer_GradientModification_v2_mean_abs_to_1 = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy=False, dtype=torch.float16)
        assert layer_GradientModification_v2_mean_abs_to_1.scaling_factor.dtype == torch.float32
        
        return
    _init_test____GradientModification__mean_len_of_element_to_1()
    pass

if "some extra use case test" and __DEBUG_ME__() and True:
    def _some_extra_use_case_test____GradientModification__mean_len_of_element_to_1():
        mat = torch.empty(size=[2,3], requires_grad=True)
        gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = True)
        mat_gramo = gramo(mat.reshape([1,-1])).reshape([2,3])
        assert mat.shape == mat_gramo.shape
        mat_gramo.backward(inputs = mat, gradient = torch.ones_like(mat_gramo)*2.)
        assert mat.grad is not None
        assert _tensor_equal(mat.grad, torch.ones_like(mat_gramo))
        
        #<  protect_binary_accuracy>
        mat = torch.empty(size=[2,3], requires_grad=True)
        gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = True)
        mat_gramo = gramo(mat.reshape([1,-1])).reshape([2,3])
        assert mat.shape == mat_gramo.shape
        _grad = torch.tensor([[1.,1,1],[0,0,0]])
        mat_gramo.backward(inputs = mat, gradient = _grad)
        assert mat.grad is not None
        assert _tensor_equal(mat.grad, torch.tensor([[2.,2,2],
                                                    [ 0 ,0,0]]), epsilon=1e-3)
        
        mat = torch.empty(size=[2,3], requires_grad=True)
        gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = False)
        mat_gramo = gramo(mat.reshape([1,-1])).reshape([2,3])
        assert mat.shape == mat_gramo.shape
        _grad = torch.tensor([[1.,1,1],[0,0,0]])
        mat_gramo.backward(inputs = mat, gradient = _grad)
        assert mat.grad is not None
        assert _tensor_equal(mat.grad, torch.tensor([[1.4142, 1.4142, 1.4142],
                                                        [0,0,0]]), epsilon=1e-3)
        #</ protect_binary_accuracy>
        
        "transpose? ofc."
        mat = torch.empty(size=[2,2], requires_grad=True)
        gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = False, per_what="vector")
        mat_gramo_by_column = gramo(mat.T).T
        assert mat.shape == mat_gramo_by_column.shape
        _grad = torch.tensor([[1.,1],[2,3]])
        mat_gramo_by_column.backward(inputs = mat, gradient = _grad)
        assert mat.grad is not None
        assert _tensor_equal(mat.grad, torch.tensor(   [[0.4472, 0.3162],
                                                        [0.8944, 0.9487]]), epsilon=1e-4)
        "_ref"
        mat = torch.empty(size=[2,2], requires_grad=True)
        gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = False, per_what="vector")
        mat_gramo_by_row = gramo(mat)
        assert mat.shape == mat_gramo_by_row.shape
        _grad = torch.tensor([[1.,2],[1,3]])
        mat_gramo_by_row.backward(inputs = mat, gradient = _grad)
        assert mat.grad is not None
        assert _tensor_equal(mat.grad, torch.tensor(   [[0.4472, 0.8944],
                                                        [0.3162, 0.9487]]), epsilon=1e-4)
        
        
        if "old test" and False:
            "big dim"
            import math, random
            #dim1, scale 0.0001 works.
            #dim10000, scale 0.0001 works.
            
            "per element, when grad is not too small."
            for _ in range(8):
                dim = int(math.pow(10,random.random()*3+0.9))#basically 10 to 10000
                assert dim>=7 and dim <10000
                grad_scale = math.pow(10,random.random()*-2.-1.)
                assert grad_scale<=0.1 and grad_scale>=0.001
                
                mat = torch.empty(size=[dim,dim], requires_grad=True)
                gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = False,
                                                                            per_what="element")
                assert grad_scale>=gramo.epsilon
                
                mat_gramo = gramo(mat.reshape([1,-1])).reshape([dim,dim])
                assert mat.shape == mat_gramo.shape
                mat_gramo.backward(inputs = mat, gradient = torch.ones_like(mat_gramo)*grad_scale)
                assert mat.grad is not None
                assert _tensor_equal(mat.grad, torch.ones_like(mat_gramo))
                pass
            
            
            
            "per vector"
            for _ in range(118):
                dim = int(math.pow(10,random.random()*3+0.9))#basically 10 to 10000
                assert dim>7 and dim <10000
                grad_scale = math.pow(10,random.random()*-2.-2.)
                assert grad_scale<=0.01 and grad_scale>=0.0001
            
                mat = torch.empty(size=[1,dim], requires_grad=True)
                gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = False,
                                                                        per_what = "vector")################
                mat_gramo = gramo(mat)
                assert mat.shape == mat_gramo.shape
                mat_gramo.backward(inputs = mat, gradient = torch.ones_like(mat_gramo)*grad_scale)
                assert mat.grad is not None
                length_of_grad = (mat.grad*mat.grad).sum(dim=1).sqrt()
                assert _tensor_equal(length_of_grad, [1.], epsilon=1e-4)
                pass
            
            pass
        
        #old code.
        # dim = 10000
        # grad_scale = 0.0001
        # mat = torch.empty(size=[dim,dim], requires_grad=True)
        # gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = False,per_what="vector")
        # mat_gramo = gramo(mat.reshape([1,-1])).reshape([dim,dim])
        # assert mat.shape == mat_gramo.shape
        # mat_gramo.backward(inputs = mat, gradient = torch.ones_like(mat_gramo)*grad_scale)
        # assert mat.grad is not None
        # assert _tensor_equal(mat.grad, torch.ones_like(mat_gramo)/torch.tensor(dim, dtype=torch.float64))

        # dim = 10000
        # grad_scale = 0.0001
        # mat = torch.empty(size=[dim,dim], requires_grad=True)
        # gramo = GradientModification__mean_len_of_something_to_1(protect_binary_accuracy = False,per_what="vector")
        # mat_gramo_by_row = gramo(mat)
        # assert mat.shape == mat_gramo_by_row.shape
        # mat_gramo_by_row.backward(inputs = mat, gradient = torch.ones_like(mat_gramo_by_row)*grad_scale)
        # assert mat.grad is not None
        # assert _tensor_equal(mat.grad, torch.ones_like(mat_gramo_by_row)/torch.sqrt(torch.tensor(dim, dtype=torch.float64)))
        
        
        
        return 
    _some_extra_use_case_test____GradientModification__mean_len_of_element_to_1()
    pass


































