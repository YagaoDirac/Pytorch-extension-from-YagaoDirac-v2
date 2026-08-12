from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, _tensor_shape_check, \
        _either_1_or_neg1, \
        iota
from pytorch_yagaodirac_v2.Random import rand_sign
from DNN2026.DNN_util import Index_container
from DNN2026.Digital_mapping_layer_for_DNN_2026 import DigitalMapper_layer__2026

import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######





又是两种reshape。。。

自动计算形状
def _only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape( \
        in_feature:int, out_feature:int, layer_count:int)->list[int]:



    assert result.__len__() == layer_count
    return result

    记得可视化一下plt.plot



class dry_stack_test__DNN_model__2026(torch.nn.Module):
    _in_dim:int
    _out_dim:int
    _layer_count:int
    layers:torch.nn.ParameterList
    #customized function
    _calc_shape_function:function
    def _1_init__(self, in_feature:int, out_feature:int, layer_count:int, \
                some_hyper_param: float = 1, init_to_nan: bool = True, 
                _dtype_for_raw_weight = torch.float32, _always_check_input_is_posneg1__in_forward: bool = True, 
                device = None):
        super().__init__()
        self. _in_dim =  in_feature
        self._out_dim = out_feature
        self._layer_count = layer_count

        self._calc_shape_function = _only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape
        _temp__shape:list[int] = self._calc_shape_function(
                in_feature = in_feature, out_feature = out_feature, layer_count = layer_count)

        _temp_layer_list = []
        for ii in range(_temp__shape.__len__()-1):
            in_dim = _temp__shape[ii]
            out_dim = _temp__shape[ii+1]
            _temp_layer_list.append(DigitalMapper_layer__2026(in_feature = in_dim, out_feature = out_dim, 
                some_hyper_param = some_hyper_param, init_to_nan = init_to_nan, 
                _dtype_for_raw_weight = _dtype_for_raw_weight, 
                _always_check_input_is_posneg1__in_forward = _always_check_input_is_posneg1__in_forward, 
                device = device))
            pass
        #the last layer is square.
        _temp_layer_list.append(DigitalMapper_layer__2026(in_feature = _temp__shape[-1], out_feature = _temp__shape[-1], 
                some_hyper_param = some_hyper_param, init_to_nan = init_to_nan, 
                _dtype_for_raw_weight = _dtype_for_raw_weight, 
                _always_check_input_is_posneg1__in_forward = _always_check_input_is_posneg1__in_forward, 
                device = device))

        self._layer_type = torch.nn.ParameterList(_temp_layer_list)
        assert self._layer_type.__len__() == self._layer_count
        assert False, "untested"
        return

    def forward(self, input___b_i:torch.Tensor)->torch.Tensor:
        '''return output___b_o'''
        x = input___b_i
        for digitalmapper_layer in self.layers:
            assert isinstance(digitalmapper_layer, DigitalMapper_layer__2026)
            x = digitalmapper_layer.forward(x)
            pass
        output___b_o = x
        assert False, "untested"
        return output___b_o

    def add_input_slot(self, how_many_new_input:int)->None:
        #the last layer is untouched in this function
        _temp__shape:list[int] = self._calc_shape_function(
                in_feature = self._in_dim, out_feature = self._out_dim, layer_count = self._layer_count)

        for ii in range(_temp__shape.__len__()-1):
            new__in_dim = _temp__shape[ii]
            new__out_dim = _temp__shape[ii+1]
            layer = self.layers[ii]
            assert isinstance(layer, DigitalMapper_layer__2026)
            how_many__new_in_dim = new__in_dim - layer.in_dim
            if how_many__new_in_dim > 0:这个判断要不要移进去？？
                layer.add_input_slot__to_the_tail(how_many = how_many__new_in_dim)
                pass
            how_many__new_out_dim = new__out_dim - layer.out_dim
            if how_many__new_out_dim > 0:
                layer.add_output_slot__to_the_tail(how_many = how_many__new_out_dim)
                pass
            pass
        del layer

        the_last_layer = self.layers[-1]
        assert isinstance(the_last_layer, DigitalMapper_layer__2026)
        new__in_dim = _temp__shape[-1]
        how_many__new_in_dim = new__in_dim - the_last_layer.in_dim
        if how_many__new_in_dim > 0:
            the_last_layer.add_input_slot__to_the_tail(how_many = how_many__new_in_dim)
            pass
        assert False, "untested"
        return 
    
    def keep_output_slot(self, keep_which: Tensor)->None:
        #only the last layer is touched in this function
        the_last_layer = self.layers[-1]
        assert isinstance(the_last_layer, DigitalMapper_layer__2026)
        the_last_layer.keep_output_slot(keep_which = keep_which)
        assert False, "untested"
        return 


    # extra_repr, str, repr
    pass#end of class.


        



        





