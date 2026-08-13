from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, _tensor_shape_check, \
        _either_1_or_neg1, \
        iota
from pytorch_yagaodirac_v2.Random import rand_sign
from DNN2026.DNN_util import Index_container
from DNN2026.digitalmapping_layer___prototype_test import DigitalMapping_layer__2026

import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######





'''some customized function to help you calc the shape of the entire model.'''
def _only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape( \
        in_feature:int, out_feature:int, layer_count:int)->torch.Tensor:
    assert in_feature>out_feature, "暂时是这么设计的."

    result = torch.linspace(start = in_feature, end = out_feature, steps = layer_count, dtype=torch.float16)
    result[1:-2] = result[1:-2]+ 0.5
    result = result.to(torch.int32)
    #<  safety
    assert result[0]  ==  in_feature
    assert result[-1] == out_feature
    assert result.__len__() == layer_count
    return result
if "test" and __DEBUG_ME__() and False:
    from matplotlib import pyplot as plt
    def ____test_____only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape():
        if "VISUAL" and True:
            for in_dim in [3,11,18]:
                for out_dim in [5,17,31]:
                    if in_dim<= out_dim:
                        continue

                    for layer_count in [4,14,29]:
                        the_list = _only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape(
                                in_feature=in_dim, out_feature=out_dim,layer_count=layer_count)
                        x_axis = torch.linspace(0,layer_count-1,layer_count,dtype=torch.int32)
                        _, ax = plt.subplots()
                        ax.plot(x_axis, the_list.tolist())
                        plt.title(f"in {10}   out {5}   layer{3}")
                        plt.show()
                        pass#for layer_count
                    pass#for out_dim
                pass#for in_dim
            pass#/ test
        return
    ____test_____only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape()
    pass



class dry_stack_test__DNN_model__2026(torch.nn.Module):
    _in_dim:int
    _out_dim:int
    _layer_count:int
    _layers:torch.nn.ParameterList
    #customized function
    _calc_shape_function:function
    def __init__(self, in_features:int, out_features:int, layer_count:int, \
                some_hyper_param: float = 1, init_to_nan: bool = True, 
                _dtype_for_raw_weight = torch.float32, _always_check_input_is_posneg1__in_forward: bool = True, 
                device = None):
        super().__init__()
        self. _in_dim =  in_features
        self._out_dim = out_features
        self._layer_count = layer_count

        self._calc_shape_function = _only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape
        _temp__shape:list[int] = self._calc_shape_function(
                in_feature = in_features, out_feature = out_features, layer_count = layer_count)

        _temp_layer_list = []
        for ii in range(_temp__shape.__len__()-1):
            in_dim = _temp__shape[ii]
            out_dim = _temp__shape[ii+1]
            _temp_layer_list.append(DigitalMapping_layer__2026(in_features = in_dim, out_features = out_dim, 
                some_hyper_param = some_hyper_param, init_to_nan = init_to_nan, 
                _dtype_for_raw_weight = _dtype_for_raw_weight, 
                _always_check_input_is_posneg1__in_forward = _always_check_input_is_posneg1__in_forward, 
                device = device))
            pass
        #the last layer is square.
        _temp_layer_list.append(DigitalMapping_layer__2026(
                in_features = _temp__shape[-1], 
                init_capacity__for_in = _temp__shape[-1], 

                out_features = _temp__shape[-1], 
                init_capacity__for_out = _temp__shape[-1], 

                some_hyper_param = some_hyper_param, init_to_nan = init_to_nan, 
                _dtype_for_raw_weight = _dtype_for_raw_weight, 
                _always_check_input_is_posneg1__in_forward = _always_check_input_is_posneg1__in_forward, 
                device = device))

        self._layers = torch.nn.ParameterList(_temp_layer_list)
        assert self._layers.__len__() == self._layer_count
        return

    def forward(self, input___b_i:torch.Tensor)->torch.Tensor:
        '''return output___b_o'''
        x = input___b_i
        for digitalmapping_layer in self._layers:
            assert isinstance(digitalmapping_layer, DigitalMapping_layer__2026)
            x = digitalmapping_layer.forward(x)
            pass
        output___b_o = x
        return output___b_o

    def add_input_slot(self, how_many_new_input:int)->None:
        '''The out_dim of the last layer is untouched in this function.
        '''
        new__shape_list:list[int] = self._calc_shape_function(
                in_feature = self._in_dim, out_feature = self._out_dim, layer_count = self._layer_count)

        for ii in range(new__shape_list.__len__()-1):
            new__in_dim = new__shape_list[ii]
            new__out_dim = new__shape_list[ii+1]
            layer = self._layers[ii]
            assert isinstance(layer, DigitalMapping_layer__2026)

            how_many__new_in_dim = new__in_dim - layer.in_dim
            layer.add_input_slot__to_the_tail(how_many = how_many__new_in_dim)

            how_many__new_out_dim = new__out_dim - layer.out_dim
            layer.add_output_slot__to_the_tail(how_many = how_many__new_out_dim)
            pass
        del layer

        the_last_layer = self._layers[-1]
        assert isinstance(the_last_layer, DigitalMapping_layer__2026)
        new__in_dim = new__shape_list[-1]
        how_many__new_in_dim = new__in_dim - the_last_layer.in_dim
        the_last_layer.add_input_slot__to_the_tail(how_many = how_many__new_in_dim)
        return 
    
    def keep_output_slot(self, keep_which: torch.Tensor)->None:
        '''only the out_dim of the last layer is touched in this function
        '''
        the_last_layer = self._layers[-1]
        assert isinstance(the_last_layer, DigitalMapping_layer__2026)
        the_last_layer.keep_output_slot(keep_which = keep_which)
        return 


    # extra_repr, str, repr
    pass#end of class.

if "test" and __DEBUG_ME__() and False:
    def ____basic_behavior____dry_stack_test__DNN_model__2026():

        if "shape scan" and False:
            for in_dim in [3,11,18]:
                for out_dim in [5,17,31]:
                    if in_dim<= out_dim:
                        continue

                    for batch in [2,9,17]:
                        for layer_count in [4,14,29]:

                            the_model = dry_stack_test__DNN_model__2026(in_features = in_dim, 
                                        out_features = out_dim, layer_count = layer_count)
                            input___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                            output___b_o = the_model(input___b_i)
                            assert _tensor_shape_check(output___b_o, batch, out_dim)
                            pass#for layer_count
                        pass#for batch
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "add input_slot" and False:
            for ori___in_dim in [3,11,18]:
                for out_dim in [5,17,31]:
                    if ori___in_dim<= out_dim:
                        continue

                    for batch in [2,9,17]:
                        for layer_count in [4,14,29]:
                            for extra__input_dim in [7,33,41]:
                                #<  ori model
                                the_model = dry_stack_test__DNN_model__2026(in_features = ori___in_dim, 
                                            out_features = out_dim, layer_count = layer_count)
                                ori___input___b_i = rand_sign(size=[batch, ori___in_dim], dtype=torch.float32)
                                output___b_o = the_model(ori___input___b_i)
                                assert _tensor_shape_check(output___b_o, batch, out_dim)
                                #<  modify the shape
                                the_model.add_input_slot(how_many_new_input=extra__input_dim)
                                new___input___b_i = rand_sign(size=[batch, ori___in_dim+extra__input_dim], dtype=torch.float32)
                                output___b_o = the_model(new___input___b_i)
                                assert _tensor_shape_check(output___b_o, batch, out_dim)
                            pass#for layer_count
                        pass#for batch
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "keep/remove output_slot" and True:
            for in_dim in [3,11,18]:
                for ori___out_dim in [5,17,31]:
                    if in_dim<= ori___out_dim:
                        continue

                    for batch in [2,9,17]:
                        for layer_count in [4,14,29]:
                            #<  answer
                            flag__keep_these = torch.rand(size=[ori___out_dim]).gt(0.5)
                            assert flag__keep_these.dtype == torch.bool
                            new___out_dim = int(flag__keep_these.to(torch.int32).sum().item())
                            #<  ori model
                            the_model = dry_stack_test__DNN_model__2026(in_features = in_dim, 
                                        out_features = ori___out_dim, layer_count = layer_count)
                            input___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                            ori___output___b_o = the_model(input___b_i.detach().clone())
                            assert _tensor_shape_check(ori___output___b_o, batch, ori___out_dim)
                            #<  modify the shape
                            the_model.keep_output_slot(flag__keep_these)
                            new___output___b_o = the_model(input___b_i.detach().clone())
                            assert _tensor_shape_check(new___output___b_o, batch, new___out_dim)
                            pass#for layer_count
                        pass#for batch
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        return
    ____basic_behavior____dry_stack_test__DNN_model__2026()
    pass

if "basic behavior of dry stack test" and __DEBUG_ME__() and True:
1w
1w
1w
1w

        



        





