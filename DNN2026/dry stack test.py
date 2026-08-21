from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import iota, \
        _tensor_equal, _bool_equal___0_as_false, _tensor_shape_check, _either_1_or_neg1, \
        str_the_list
from pytorch_yagaodirac_v2.Random import rand_sign
from DNN2026.DNN_util import Index_container, partly_reasonable_label_from_input, \
        _test___binary_accuracy___full_safety
from DNN2026.digitalmapping_layer___prototype_test import DigitalMapping_layer__2026#, optim_for___Digital
from DNN2026.data_container_for_DNN_2026___prototype_test import DNN_input_container_2026, DNN_label_container_2026

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
        in_feature:int, out_feature:int, layer_count:int)->list[int]:#torch.Tensor:
    assert in_feature>out_feature, "暂时是这么设计的."
    assert layer_count>1, "1层的另外研究。"

    result = torch.linspace(start = in_feature, end = out_feature, steps = layer_count, dtype=torch.float16)
    result[1:-2] = result[1:-2] + 0.5#or 0.499?
    result = result.to(torch.int32)
    result[ 0] =  in_feature
    result[-1] = out_feature
    #<  safety
    assert result.__len__() == layer_count
    return result.tolist()#py list.
if "test" and __DEBUG_ME__() and False:
    from matplotlib import pyplot as plt
    def ____test_____only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape():
        # if "hand write param combination" and True:
        #     the_list = _only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape(
        #             in_feature=5, out_feature=3,layer_count=1)
        #     assert the_list.__len__() == 1
        #     assert the_list[0] == 


        #     pass#/ test



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
                        ax.plot(x_axis, the_list)
                        plt.title(f"in {10}   out {5}   layer{3}")
                        plt.show()
                        pass#for layer_count
                    pass#for out_dim
                pass#for in_dim
            pass#/ test
        return
    ____test_____only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape()
    pass




'''dry stack             model'''
'''dry stack             model'''
class dry_stack_test__DNN_model__2026(torch.nn.Module):
    #in_dim:int
    _original__out_dim:int
    _layer_count:int
    _layers:torch.nn.ParameterList
    #customized function
    _calc_shape_function:function
    def __init__(self, in_features:int, out_features:int, layer_count:int, \
                some_hyper_param: float = 1, init_to_nan: bool = True, 
                _dtype_for_raw_weight = torch.float32, _always_check_input_is_posneg1__in_forward: bool = True, 
                device = None):
        super().__init__()
        #self. in_dim =  in_features
        self._original__out_dim = out_features
        self._layer_count = layer_count

        self._calc_shape_function = _only_for_dry_stack_test__DNN_model__2026_to_use____calc_shape

        if layer_count == 1:
            _temp__the_only_layer = DigitalMapping_layer__2026(
                    in_features = in_features, 
                    init_capacity__for_in = in_features, 

                    out_features = out_features, 
                    init_capacity__for_out = out_features, 

                    some_hyper_param = some_hyper_param, init_to_nan = init_to_nan, 
                    _dtype_for_raw_weight = _dtype_for_raw_weight, 
                    _always_check_input_is_posneg1__in_forward = _always_check_input_is_posneg1__in_forward, 
                    device = device)
            self._layers = torch.nn.ParameterList([_temp__the_only_layer])
            pass#if layer_count == 1:
        else:# layer_count >= 2
            _temp__shape:list[int] = self._calc_shape_function(
                    in_feature = in_features, out_feature = out_features, layer_count = layer_count)
            assert type(_temp__shape) == list
            assert type(_temp__shape[0]) == int

            _temp_layer_list:list[DigitalMapping_layer__2026] = []
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
            pass#else:# layer_count >= 2

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
    @staticmethod
    def backward(the_model:dry_stack_test__DNN_model__2026, output___b_o:torch.Tensor, label___b_o:torch.Tensor)->None:
        '''This function helps you handle the "inputs=" inside the backward function call.'''
        #backward_to_this_list = []
        #backward_to_this_list.extend(the_model.parameters(for_backward=True))

        parameter_list:list[torch.nn.Parameter] = []
        for digitalmapping_layer in the_model._layers:
            assert isinstance(digitalmapping_layer, DigitalMapping_layer__2026)
            parameter_list.extend(digitalmapping_layer.parameters())
            pass

        # for param in parameter_list:
        #     assert isinstance(param, torch.Tensor)
        #     if hasattr(param, "aaaaa"):
        #         print(param)
        #     else:
        #         param.aaaaa = "debug aaaaa"
        #         param.data.aaaaaaaaa = "debug aaaaaaaaa"
        #         pass
        #     pass#for 
        
        output___b_o.backward(gradient=label___b_o, inputs=parameter_list)
        return

    def zero_grad(self, set_to_none: bool = True) -> None:
        for layer in self._layers:
            assert isinstance(layer, DigitalMapping_layer__2026)
            layer._raw_weight___oCAP_iCAP[0].grad = None
            pass

    @torch.no_grad() # Important: disable gradient tracking within the optimizer step
    def step(self, learning_rate___s:float|torch.Tensor, epsilon___s:float|torch.Tensor = 0.01, safety_check = False)->None:
        #https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-optimizers
        '''Bc I don't use this closure style, and I have no idea how it works.
        Just in case, let me turn it off. 
        Fyi, https://apxml.com/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-optimizers'''

        #<  safety
        assert epsilon___s > 0., "Bad param"
        assert learning_rate___s> 0., "Bad param"

        #<  type
        _the_device = self._layers[0]._raw_weight___oCAP_iCAP[0].device
        _the_dtype = self._layers[0]._raw_weight___oCAP_iCAP[0].dtype
        if type(learning_rate___s) == float:
            learning_rate___s = torch.tensor(learning_rate___s, device=_the_device, dtype=_the_dtype)
            pass
        assert isinstance(learning_rate___s, torch.Tensor)
        if type(epsilon___s) == float:
            epsilon___s = torch.tensor(epsilon___s, device=_the_device, dtype=_the_dtype)
            pass
        assert isinstance(epsilon___s, torch.Tensor)


        #<  real payload
        for layer in self._layers:
            assert isinstance(layer, DigitalMapping_layer__2026)

            if layer._raw_weight___oCAP_iCAP[0].grad is None:
                assert False, "according to the design, this must NOT be None."
                continue # or maybe skip it? Skip parameters without gradients

            grad_like_for_raw_weight___o_i = layer._get_useful_part_of_raw_weight_grad()
            if grad_like_for_raw_weight___o_i is None:
                continue
            
            # old code      new_data_for_parameter = only_for_DigitalMapping_layer__2026_to_use___optim_step( \
            #                     raw_weight___o_i = digitalmapping_layer.get_useful_part_of_raw_weight(),  
            #                     grad_like_for_raw_weight___o_i = grad_like, 
            #                     learning_rate___s = self.learning_rate___s, 
            #                     epsilon=self.epsilon)#展开
            #<  展开
            raw_weight___o_i:torch.Tensor = layer.get_useful_part_of_raw_weight()

            #<  real payload
            _temp___max___o:torch.Tensor = grad_like_for_raw_weight___o_i.max(dim=1).values
            _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
            del _temp___max___o, _temp___max___o_EXPANDi
            if safety_check:
                assert inner___grad_like_for_raw_weight___o_i.le(0.).all()#################
                pass

            _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
            _temp___mean_of_abs___o = _temp___mean_of_abs___o.max(epsilon___s)
            #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
            if safety_check:
                assert _temp___mean_of_abs___o.ge(epsilon___s).all()
                pass

            _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
            del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi

            new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * learning_rate___s
            if safety_check:
                assert new___raw_weight___before_tanh___o_i.le(0.).all()
                pass
            new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
            #</  展开


            layer.set_useful_part_of_raw_weight(new___raw_weight___o_i)
            pass#for for layer 
        return #end of function

    def parameters(self, recurse = True, for_optim = False, for_backward = False):
        assert False, "emmmm 还没想好，但是肯定不用这个函数了。"
        if for_optim:
            return self._layers
            pass
        elif for_backward:
            #return super().parameters(recurse) the original code of pytorch.
            parameter_list:list[torch.nn.Parameter] = []
            for digitalmapping_layer in self._layers:
                assert isinstance(digitalmapping_layer, DigitalMapping_layer__2026)
                parameter_list.append(digitalmapping_layer._raw_weight___oCAP_iCAP[0])#1 11w1w1 这个原版的尺寸不对。。要给最终用的尺寸
                pass
            return parameter_list
        else:
            assert False, "unreachable. One of the for_??? param must be True."
        #end of function.
    '''all the shape and reshape'''
    '''all the shape and reshape'''
    def in_dim(self)->int:
        '''get'''
        the_first_layer = self._layers[0]
        assert isinstance(the_first_layer, DigitalMapping_layer__2026)
        return the_first_layer.in_dim
    def out_dim(self)->int:
        '''get'''
        the_last_layer = self._layers[-1]
        assert isinstance(the_last_layer, DigitalMapping_layer__2026)
        return the_last_layer.out_dim

    def add_input_slot(self, how_many_new_input:int)->None:
        '''The out_dim of the last layer is untouched in this function.
        '''
        #<  safety
        assert self._layers.__len__() > 1, "this is only a dry stack test."
        #<  real payload
        new__shape_list:list[int] = self._calc_shape_function(
                in_feature = self.in_dim()+how_many_new_input, out_feature = self._original__out_dim, layer_count = self._layer_count)

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

    def remove_output_slot(self, remove_which: torch.Tensor)->None:
        '''only the out_dim of the last layer is touched in this function
        '''
        assert remove_which.dtype == torch.bool#duplicated???
        self.keep_output_slot(remove_which.logical_not())
        return 
    def _report_shape(self)->list[list[int]]:
        '''return a list of [in_dim, out_dim, in_capacity, out_capacity]'''
        result:list[list[int]] = []
        for layer in self._layers:
            assert isinstance(layer, DigitalMapping_layer__2026)
            result.append([layer.in_dim, layer.out_dim, layer._raw_weight___oCAP_iCAP[0].shape[1], layer._raw_weight___oCAP_iCAP[0].shape[0]])
            assert type(layer.in_dim) == int
            assert type(layer.out_dim) == int
            assert type(layer._raw_weight___oCAP_iCAP[0].shape[1]) == int
            assert type(layer._raw_weight___oCAP_iCAP[0].shape[0]) == int
            pass
        return result



    # extra_repr, str, repr
    pass#end of class.

if "test" and __DEBUG_ME__() and False:
    def ____basic_behavior____dry_stack_test__DNN_model__2026():

        if "report shape       while changing the shapes       , but read only" and True:
            ori_out_dim = 1000
            the_model = dry_stack_test__DNN_model__2026(3333,ori_out_dim,6)
            result_0 = the_model._report_shape()
            assert result_0[0][0] == 3333
            assert result_0[-1][1] == ori_out_dim
            assert result_0[-2][1] == ori_out_dim
            assert result_0[-1][2] == ori_out_dim

            _keep_which = torch.zeros(size=[ori_out_dim], dtype=torch.bool)
            _keep_which[:ori_out_dim//2] = True
            the_model.keep_output_slot(keep_which = _keep_which)
            result_1 = the_model._report_shape()
            assert result_1[0][0] == 3333
            assert result_1[-1][1] == ori_out_dim//2
            assert result_1[-2][1] == ori_out_dim
            assert result_1[-1][2] == ori_out_dim
            
            the_model.add_input_slot(1111)
            result_2 = the_model._report_shape()
            assert result_2[0][0] == 4444
            assert result_2[-1][1] == ori_out_dim//2
            assert result_2[-2][1] == ori_out_dim
            assert result_2[-1][2] == ori_out_dim

            _keep_which = torch.zeros(size=[ori_out_dim//2], dtype=torch.bool)
            _keep_which[:ori_out_dim//4] = True
            the_model.keep_output_slot(keep_which = _keep_which)
            result_3 = the_model._report_shape()
            assert result_3[0][0] == 4444
            assert result_3[-1][1] == ori_out_dim//4
            assert result_3[-2][1] == ori_out_dim
            assert result_3[-1][2] == ori_out_dim

            the_model.add_input_slot(1111)
            result_4 = the_model._report_shape()
            assert result_4[0][0] == 5555
            assert result_4[-1][1] == ori_out_dim//4
            assert result_4[-2][1] == ori_out_dim
            assert result_4[-1][2] == ori_out_dim

            _keep_which = torch.zeros(size=[ori_out_dim//4], dtype=torch.bool)
            _keep_which[:ori_out_dim//10] = True
            the_model.keep_output_slot(keep_which = _keep_which)
            result_5 = the_model._report_shape()
            assert result_5[0][0] == 5555
            assert result_5[-1][1] == ori_out_dim//10
            assert result_5[-2][1] == ori_out_dim
            assert result_5[-1][2] == ori_out_dim
            pass#/ test

        if "shape scan" and True:
            for in_dim in [3,11,18]:
                for out_dim in [5,17,31]:
                    if in_dim<= out_dim:
                        continue

                    for batch in [2,9,17]:
                        for layer_count in [4,14,29]:
                            for _ in range(14):


                                the_model = dry_stack_test__DNN_model__2026(in_features = in_dim, 
                                            out_features = out_dim, layer_count = layer_count)
                                assert the_model._layers.__len__() == layer_count
                                input___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                                output___b_o = the_model(input___b_i)
                                assert _tensor_shape_check(output___b_o, batch, out_dim)
                                pass#for _
                            pass#for layer_count
                        pass#for batch
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        if "add input_slot" and True:
            for ori___in_dim in [3,11,18]:
                for out_dim in [5,17,31]:
                    if ori___in_dim<= out_dim:
                        continue

                    for batch in [2,9,17]:
                        for layer_count in [4,14,29]:
                            for extra__input_dim in [7,33,41]:
                                for _ in range(14):
                                    
                                    #<  ori model
                                    the_model = dry_stack_test__DNN_model__2026(in_features = ori___in_dim, 
                                                out_features = out_dim, layer_count = layer_count)
                                    assert the_model._layers.__len__() == layer_count
                                    ori___input___b_i = rand_sign(size=[batch, ori___in_dim], dtype=torch.float32)
                                    output___b_o = the_model(ori___input___b_i)
                                    assert _tensor_shape_check(output___b_o, batch, out_dim)
                                    #<  modify the shape
                                    the_model.add_input_slot(how_many_new_input=extra__input_dim)
                                    assert the_model.in_dim() == ori___in_dim + extra__input_dim
                                    new___input___b_i = rand_sign(size=[batch, ori___in_dim+extra__input_dim], dtype=torch.float32)
                                    output___b_o = the_model(new___input___b_i)
                                    assert _tensor_shape_check(output___b_o, batch, out_dim)
                                pass#for _
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
                            for _ in range(14):
                                
                                #<  answer
                                flag__keep_these = torch.rand(size=[ori___out_dim]).gt(0.5)
                                assert flag__keep_these.dtype == torch.bool
                                new___out_dim = int(flag__keep_these.to(torch.int32).sum().item())
                                #<  ori model
                                the_model = dry_stack_test__DNN_model__2026(in_features = in_dim, 
                                            out_features = ori___out_dim, layer_count = layer_count)
                                assert the_model._layers.__len__() == layer_count
                                input___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                                ori___output___b_o = the_model(input___b_i.detach().clone())
                                assert _tensor_shape_check(ori___output___b_o, batch, ori___out_dim)
                                #<  modify the shape
                                the_model.keep_output_slot(flag__keep_these)
                                new___output___b_o = the_model(input___b_i.detach().clone())
                                assert the_model.out_dim() == new___out_dim
                                assert _tensor_shape_check(new___output___b_o, batch, new___out_dim)
                                pass#for _
                            pass#for layer_count
                        pass#for batch
                    pass#for out_dim
                pass#for in_dim
            pass#/ test

        return
    ____basic_behavior____dry_stack_test__DNN_model__2026()
    pass

if "the optim part" and __DEBUG_ME__() and False:
    def ____test____optim_part_of_DigitalMapping_layer__2026()->None:
        if "zero grad function.      scan" and True:
            for batch in [2,5,10]:
                for out_dim in [3,7,11]:
                    for in_dim in [6,9,13]:
                        if in_dim<=out_dim:
                            continue

                        the_model = dry_stack_test__DNN_model__2026(in_features=in_dim, out_features=out_dim, layer_count=2)
                        for ii in range(2):
                            the_layer = the_model._layers[ii]
                            assert isinstance(the_layer, DigitalMapping_layer__2026)
                            assert the_layer._raw_weight___oCAP_iCAP[0].requires_grad == True
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            assert the_layer.some_hyper_param.requires_grad == False
                            assert the_layer.some_hyper_param.grad is None
                            pass
                        
                        the_model.zero_grad()
                        for ii in range(2):
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            pass


                        the_model = dry_stack_test__DNN_model__2026(in_features=in_dim, out_features=out_dim, layer_count=2)
                        for ii in range(2):
                            the_layer = the_model._layers[ii]
                            assert isinstance(the_layer, DigitalMapping_layer__2026)
                            assert the_layer._raw_weight___oCAP_iCAP[0].requires_grad == True
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            assert the_layer.some_hyper_param.requires_grad == False
                            assert the_layer.some_hyper_param.grad is None
                            pass


                        input___b_i = rand_sign(size=[batch, in_dim])
                        input___b_i.requires_grad_()
                        output___b_o:torch.Tensor = the_model(input___b_i)
                        _temp_inputs = [input___b_i]
                        for ii in range(2):
                            the_layer = the_model._layers[ii]
                            assert isinstance(the_layer, DigitalMapping_layer__2026)
                            _temp_inputs.append(the_layer._raw_weight___oCAP_iCAP[0])
                            pass
                        output___b_o.backward(gradient=torch.randn_like(output___b_o), inputs = _temp_inputs)
                        del _temp_inputs
                        assert input___b_i.grad is not None

                        for ii in range(2):
                            the_layer = the_model._layers[ii]
                            assert isinstance(the_layer, DigitalMapping_layer__2026)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                            assert the_layer.some_hyper_param.grad is None
                            pass

                        the_model.zero_grad()
                        for ii in range(2):
                            layer = the_model._layers[ii]
                            assert isinstance(the_layer, DigitalMapping_layer__2026)
                            assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                            pass
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test

        if "class equivalence         no scan" and True:
            out_dim = 2
            in_dim = 3

            the_model = dry_stack_test__DNN_model__2026(in_features=in_dim, out_features=out_dim, layer_count=1)
            learning_rate___s = 1.1


            #<  data
            the_layer = the_model._layers[0]
            assert isinstance(the_layer, DigitalMapping_layer__2026)
            the_layer._raw_weight___oCAP_iCAP[0].grad = torch.empty_like(the_layer._raw_weight___oCAP_iCAP[0])
            the_layer._raw_weight___oCAP_iCAP[0].grad.fill_(torch.nan)
            raw_weight___o_i = torch.tensor([   [-10., -11, 0], 
                                                [-100., -11, 0]]) 
            #raw_weight___o_i = torch.rand(size=(out_dim, in_dim))*-1.
            assert raw_weight___o_i.le(0.).all()#bc of the design. No other reason.
            assert _tensor_shape_check(raw_weight___o_i, out_dim, in_dim)
            the_layer.set_useful_part_of_raw_weight(raw_weight___o_i.detach().clone())
            grad_like_for_raw_weight___o_i = torch.tensor([ [1231.,  1232, 1233], 
                                                            [3211.,  3213, 3215], ])
            assert _tensor_shape_check(grad_like_for_raw_weight___o_i, out_dim, in_dim)
            the_layer._raw_weight___oCAP_iCAP[0].grad[:out_dim, :in_dim] = grad_like_for_raw_weight___o_i.detach().clone()

            #<  real payload
            _temp___max___o:torch.Tensor  = grad_like_for_raw_weight___o_i.max(dim=1).values
            assert _tensor_shape_check(_temp___max___o, out_dim)
            assert _tensor_equal(_temp___max___o, torch.tensor([1233, 3215]))
            _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            assert _tensor_shape_check(_temp___max___o_EXPANDi, out_dim, in_dim)
            inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
            del _temp___max___o, _temp___max___o_EXPANDi
            assert _tensor_equal(inner___grad_like_for_raw_weight___o_i, [  [-2., -1, 0], 
                                                                            [-4., -2, 0], ])

            _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
            #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
            assert _tensor_shape_check(_temp___mean_of_abs___o, out_dim)
            assert _tensor_equal(_temp___mean_of_abs___o, [1., 2])

            _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
            assert _tensor_shape_check(_temp___temp___mean_of_abs___o_EXPANDi, out_dim, in_dim)
            inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
            del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi
            assert _tensor_equal(inner___grad_like_for_raw_weight___o_i, torch.tensor([ [-2., -1, 0], 
                                                                                        [-2., -1, 0], ]))

            new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * learning_rate___s

            assert _tensor_equal(new___raw_weight___before_tanh___o_i, torch.tensor([   [-12.2,  -12.1, 0], 
                                                                                        [-102.2, -12.1, 0]]))
            assert new___raw_weight___before_tanh___o_i.le(-0.).all()##############
            new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
            assert new___raw_weight___o_i.ge(-1.).all()##############
            assert new___raw_weight___o_i.le(-0.).all()##############

            #<  step the layer???
            the_model.step(learning_rate___s = learning_rate___s)
            assert the_layer.get_useful_part_of_raw_weight().eq(new___raw_weight___o_i).all()

            pass#/ test

        if "class equivalence         scan" and True:
            for out_dim in [3,7,11]:
                for in_dim in [6,9,13]:
                    for _ in range(11):
                        #<  neural net infra
                        the_model = dry_stack_test__DNN_model__2026(in_features=in_dim, out_features=out_dim, layer_count=1)

                        learning_rate___s = 1.1


                        #<  data
                        the_layer = the_model._layers[0]
                        assert isinstance(the_layer, DigitalMapping_layer__2026)
                        the_layer._raw_weight___oCAP_iCAP[0].grad = torch.empty_like(the_layer._raw_weight___oCAP_iCAP[0])
                        the_layer._raw_weight___oCAP_iCAP[0].grad.fill_(torch.nan)
                        raw_weight___o_i = torch.rand(size=[out_dim, in_dim ])*-1.
                        assert raw_weight___o_i.le(0.).all()#bc of the design. No other reason.
                        assert _tensor_shape_check(raw_weight___o_i, out_dim, in_dim)
                        the_layer.set_useful_part_of_raw_weight(raw_weight___o_i.detach().clone())
                        grad_like_for_raw_weight___o_i = torch.randn(size=[out_dim, in_dim ])
                        assert _tensor_shape_check(grad_like_for_raw_weight___o_i, out_dim, in_dim)
                        the_layer._raw_weight___oCAP_iCAP[0].grad[:out_dim, :in_dim] = grad_like_for_raw_weight___o_i.detach().clone()

                        #<  real payload     function version.
                        _temp___max___o:torch.Tensor  = grad_like_for_raw_weight___o_i.max(dim=1).values
                        assert _tensor_shape_check(_temp___max___o, out_dim)
                        _temp___max___o_EXPANDi = _temp___max___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
                        assert _tensor_shape_check(_temp___max___o_EXPANDi, out_dim, in_dim)
                        inner___grad_like_for_raw_weight___o_i:torch.Tensor = grad_like_for_raw_weight___o_i-_temp___max___o_EXPANDi
                        del _temp___max___o, _temp___max___o_EXPANDi

                        _temp___mean_of_abs___o = inner___grad_like_for_raw_weight___o_i.mean(dim=1).abs()# notice.  
                        #In some of the previous test, there was a *0.5 in the tail of the line above. But maybe it's ok without it.
                        assert _tensor_shape_check(_temp___mean_of_abs___o, out_dim)

                        _temp___temp___mean_of_abs___o_EXPANDi = _temp___mean_of_abs___o.reshape([-1, 1]).expand([-1, raw_weight___o_i.shape[1]])
                        assert _tensor_shape_check(_temp___temp___mean_of_abs___o_EXPANDi, out_dim, in_dim)
                        inner___grad_like_for_raw_weight___o_i /= _temp___temp___mean_of_abs___o_EXPANDi
                        del _temp___mean_of_abs___o, _temp___temp___mean_of_abs___o_EXPANDi

                        new___raw_weight___before_tanh___o_i = raw_weight___o_i + inner___grad_like_for_raw_weight___o_i * learning_rate___s

                        assert new___raw_weight___before_tanh___o_i.le(-0.).all()##############
                        new___raw_weight___o_i = new___raw_weight___before_tanh___o_i.tanh()
                        assert new___raw_weight___o_i.ge(-1.).all()##############
                        assert new___raw_weight___o_i.le(-0.).all()##############

                        #<  layer version.
                        the_model.step(learning_rate___s=learning_rate___s)
                        assert the_layer.get_useful_part_of_raw_weight().eq(new___raw_weight___o_i).all()
                        pass#for _ 
                    pass#for out_dim
                pass#for batch
            pass#/ test

        return
    ____test____optim_part_of_DigitalMapping_layer__2026()
    pass

if "integrated test" and __DEBUG_ME__() and False:
    def ____test____integrated_test()->None:
        '''modified from the backward algo test.'''

        if "xxxxxxxxxxxxxxx  不用了        prototype.    scan" and False:
            
            #------------------#------------------#------------------
            number_of_tests = 3
            random_ratio_list = [0.]
            for ii_random_ratio in range(random_ratio_list.__len__()):
                random_ratio = random_ratio_list[ii_random_ratio]
                #print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------
                result_acc     :list = []#don't modify this.
                result_acc_gain:list = []#don't modify this.
                learning_rate_list = [333.]################################################
                #_when_start = time.perf_counter()
                
                for learning_rate in learning_rate_list:
                    _raw_result__accuracy = torch.empty(size=[number_of_tests])
                    _raw_result__accuracy_gain = torch.empty(size=[number_of_tests])
                    for ii__test in range(number_of_tests):

                        batch = 3#1000
                        in_dim = 4#500
                        out_dim = 2#100
                        #<  dataset
                        #input_posneg1___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                        input_posneg1___b_i = torch.tensor([
                            [1, -1, -1, -1],
                            [1,  1, -1, -1],
                            [1,  1,  1, -1],
                                                            ], dtype=torch.float32, requires_grad=True)#input_posneg1___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                        assert _either_1_or_neg1(input_posneg1___b_i)
                        assert _tensor_shape_check(input_posneg1___b_i, batch, in_dim)

                        # target_posneg1___b_o = partly_reasonable_label_from_input(input___b_i=input_posneg1___b_i, out_dim = out_dim,
                        #             random_ratio=random_ratio, input_is_already_posneg1 = True)
                        target_posneg1___b_o = torch.tensor([
                            [1, -1],
                            [1,  1],
                            [1,  1],
                                                            ], dtype=torch.float32, )
                        assert _tensor_shape_check(target_posneg1___b_o, batch, out_dim)
                        
                        assert _either_1_or_neg1(target_posneg1___b_o)#debug purpose
                        #<  model param       neural net infra
                        # old code     the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)

                        the_model = dry_stack_test__DNN_model__2026(in_features=in_dim, out_features=out_dim, layer_count=1)
                        the_layer = the_model._layers[0]
                        assert isinstance(the_layer, DigitalMapping_layer__2026)
                        the_layer._raw_weight___oCAP_iCAP[0] = torch.nn.Parameter(torch.tensor([
                            [-5., -4., -3., -1., torch.nan],
                            [-5., -4., -3., -1., torch.nan],
                            [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
                            [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
                            [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
                            ]))
                        #torch.nn.Parameter.requires_grad
                        assert the_layer._raw_weight___oCAP_iCAP[0].requires_grad == True
                        assert the_layer. in_dim == 4
                        assert the_layer.out_dim == 2

                        # backward_to_them = []
                        # backward_to_them.extend(the_layer.parameters(改过了))
                        #backward_to_them.pop()#####

                        #<  calc          forward
                        the_model.zero_grad()
                        ori__raw_weight___o_i:torch.Tensor = the_layer(input_posneg1___b_i)
                        assert _tensor_shape_check(ori__raw_weight___o_i, batch, out_dim)
                        #old code    ori__raw_weight___o_i.backward(gradient=target_posneg1___b_o, inputs=backward_to_them)
                        the_layer._raw_weight___oCAP_iCAP[0].requires_grad_() 
                        assert the_layer._raw_weight___oCAP_iCAP[0].grad is None
                        assert the_layer._raw_weight___oCAP_iCAP[0].requires_grad == True
                        
                        dry_stack_test__DNN_model__2026.backward(the_model=the_model, output___b_o=ori__raw_weight___o_i, 
                                    label___b_o = target_posneg1___b_o)
                        assert the_layer._raw_weight___oCAP_iCAP[0].grad is not None
                        assert _tensor_equal(the_layer._raw_weight___oCAP_iCAP[0].grad[:2, :4], [  [ 3,  1, -1, -3],
                                                                                                [ 1,  3,  1, -1],])




                        #<  ori   accuracy
                        ori__accuracy___s, recommended_result_value_name = \
                                _test___binary_accuracy___full_safety(target___b_o = target_posneg1___b_o, 
                                        output_posneg1___b_o = ori__raw_weight___o_i, mean_per =  'for_all', target_is_already_posneg1=True)
                        assert recommended_result_value_name == "accuracy___s"

                        
                        #<  step
                        print(the_layer._raw_weight___oCAP_iCAP[0])
                        the_model.step(learning_rate___s=learning_rate, epsilon___s=0.01)
                        print(the_layer._raw_weight___oCAP_iCAP[0])

                        #<  new   accuracy
                        new__raw_weight___o_i:torch.Tensor = the_layer(input_posneg1___b_i)
                        new__accuracy___s, recommended_result_value_name = \
                                _test___binary_accuracy___full_safety(target___b_o = target_posneg1___b_o, 
                                        output_posneg1___b_o = new__raw_weight___o_i, mean_per =  'for_all', target_is_already_posneg1=True)
                        assert recommended_result_value_name == "accuracy___s"
                        print(f" acc   {ori__accuracy___s.item():.3f}  ori//new {new__accuracy___s.item():.3f}, ")

                        #<  save the result.
                        _raw_result__accuracy[ii__test] = new__accuracy___s
                        _raw_result__accuracy_gain[ii__test] = new__accuracy___s - ori__accuracy___s
                        pass#for ii__test
                                            
                    result_acc     .append(_raw_result__accuracy.     mean().item())
                    result_acc_gain.append(_raw_result__accuracy_gain.mean().item())
                    
                    pass#for scanned_param
                #_when_end = time.perf_counter()
                #print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                

                print(f"random rate {random_ratio}")
                print(f"learning_rate_list = {str_the_list(learning_rate_list, 3)}")#########################
                print(f"acc              = {str_the_list(result_acc, 3)}")#########################
                print(f"acc gain         = {str_the_list(result_acc_gain, 3)}")#########################
                ################################
                pass#for ii_outter_param_set
            pass#/ test

        if "prototype.    scan" and False:
            if "result" and False:
                '''the same as previous tests'''
                # random rate 0.0
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.502,  0.503,  0.509,  0.521,  0.561,  0.661,  0.981,  1.000]
                # acc gain         = [ 0.001,  0.002,  0.009,  0.021,  0.060,  0.160,  0.481,  0.498]
                # random rate 0.1
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.501,  0.503,  0.508,  0.522,  0.555,  0.639,  0.927,  0.950]
                # acc gain         = [ 0.001,  0.002,  0.006,  0.021,  0.054,  0.138,  0.426,  0.450]
                # random rate 0.2
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.502,  0.502,  0.508,  0.516,  0.553,  0.632,  0.878,  0.900]
                # acc gain         = [ 0.001,  0.001,  0.007,  0.015,  0.052,  0.131,  0.377,  0.399]
                # random rate 0.3
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.500,  0.502,  0.507,  0.518,  0.547,  0.616,  0.827,  0.850]
                # acc gain         = [ 0.000,  0.002,  0.007,  0.017,  0.046,  0.115,  0.326,  0.348]
                # random rate 0.5
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.502,  0.502,  0.508,  0.518,  0.541,  0.591,  0.729,  0.750]
                # acc gain         = [ 0.001,  0.002,  0.007,  0.017,  0.041,  0.091,  0.228,  0.249]
                # random rate 0.7
                # learning_rate_list = [ 0.001,  0.003,  0.010,  0.030,  0.100,  0.300,  1.000,  3.000]
                # acc              = [ 0.502,  0.503,  0.508,  0.517,  0.533,  0.560,  0.629,  0.650]
                # acc gain         = [ 0.001,  0.003,  0.008,  0.018,  0.032,  0.060,  0.129,  0.149]
                pass



            #------------------#------------------#------------------
            number_of_tests = 20
            random_ratio_list = [0., 0.1, 0.2, 0.3, 0.5, 0.7]
            for ii_random_ratio in range(random_ratio_list.__len__()):
                random_ratio = random_ratio_list[ii_random_ratio]
                #print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------
                result_acc     :list = []#don't modify this.
                result_acc_gain:list = []#don't modify this.
                learning_rate_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.]################################################
                #_when_start = time.perf_counter()
                
                for learning_rate in learning_rate_list:
                    _raw_result__accuracy = torch.empty(size=[number_of_tests])
                    _raw_result__accuracy_gain = torch.empty(size=[number_of_tests])
                    for ii__test in range(number_of_tests):

                        batch = 1000
                        in_dim = 500
                        out_dim = 100
                        #<  dataset
                        input_posneg1___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)
                        assert _either_1_or_neg1(input_posneg1___b_i)
                        assert input_posneg1___b_i.dtype == torch.float32

                        target_posneg1___b_o = partly_reasonable_label_from_input(input___b_i=input_posneg1___b_i, out_dim = out_dim,
                                    random_ratio=random_ratio, input_is_already_posneg1 = True)
                        assert _either_1_or_neg1(target_posneg1___b_o)#debug purpose
                        assert target_posneg1___b_o.dtype == torch.float32
                        #<  infra
                        the_model = dry_stack_test__DNN_model__2026(in_features=in_dim, out_features=out_dim, layer_count=1)
                        the_layer = the_model._layers[0]
                        assert isinstance(the_layer, DigitalMapping_layer__2026)
                        #<  calc          forward
                        ori__raw_weight___o_i:torch.Tensor = the_layer(input_posneg1___b_i)
                        assert _tensor_shape_check(ori__raw_weight___o_i, batch, out_dim)
                        dry_stack_test__DNN_model__2026.backward(the_model=the_model, output___b_o=ori__raw_weight___o_i,label___b_o=target_posneg1___b_o)
                        #<  ori   accuracy
                        ori__accuracy___s, recommended_result_value_name = \
                                _test___binary_accuracy___full_safety(target___b_o = target_posneg1___b_o, 
                                        output_posneg1___b_o = ori__raw_weight___o_i, mean_per =  'for_all', target_is_already_posneg1=True)
                        assert recommended_result_value_name == "accuracy___s"
                        #<  step
                        the_model.step(learning_rate___s=learning_rate)
                        #<  new   accuracy
                        new__raw_weight___o_i:torch.Tensor = the_layer(input_posneg1___b_i)
                        new__accuracy___s, recommended_result_value_name = \
                                _test___binary_accuracy___full_safety(target___b_o = target_posneg1___b_o, 
                                        output_posneg1___b_o = new__raw_weight___o_i, mean_per =  'for_all', target_is_already_posneg1=True)
                        assert recommended_result_value_name == "accuracy___s"

                        #<  store the result.
                        _raw_result__accuracy[ii__test] = new__accuracy___s
                        _raw_result__accuracy_gain[ii__test] = new__accuracy___s - ori__accuracy___s
                        pass#for ii__test
                                            
                    result_acc     .append(_raw_result__accuracy.     mean().item())
                    result_acc_gain.append(_raw_result__accuracy_gain.mean().item())
                    
                    pass#for scanned_param
                #_when_end = time.perf_counter()
                #print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                
                print(f"random rate {random_ratio}")
                print(f"learning_rate_list = {str_the_list(learning_rate_list, 3)}")#########################
                print(f"acc              = {str_the_list(result_acc, 3)}")#########################
                print(f"acc gain         = {str_the_list(result_acc_gain, 3)}")#########################
                ################################
                pass#for ii_outter_param_set
            pass#/ test

        return

    ____test____integrated_test()
    pass














'''with optim_for___DigitalMapping_layer__2026
但是现在融合进Module subclass了。所以就先不用了'''
if "先不用了      basic behavior of dry stack test" and __DEBUG_ME__() and False:
    def ____basic_behavior_of_dry_stack_test________with_optim_for___DigitalMapping_layer__2026():

        if "layer and 1 layer model equivalence               no in_cont    no out_cont" and True:
            from DNN2026.digitalmapping_layer___prototype_test import optim_for___DigitalMapping_layer__2026
            for batch in [2,11,25]:
                for in_dim in [3,15,31,66]:
                    for out_dim in [5, 19, 37]:
                        if in_dim<=out_dim:
                            continue

                        for learning_rate___s in [0.01, 0.123, 0.321]:
                            for _ in range(14):

                                #<  dataset
                                _model_param = torch.empty([])#a temp store. To make two model params the same.
                                input___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)#or fp16
                                assert _either_1_or_neg1(input___b_i)
                                label___b_o = partly_reasonable_label_from_input(input___b_i = input___b_i, out_dim = out_dim,
                                        random_ratio = 0., input_is_already_posneg1 = True)
                                assert _either_1_or_neg1(label___b_o)
                                #<  layer      infra
                                the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                                _model_param = the_layer.get_useful_part_of_raw_weight().detach().clone()
                                optim_for_layer = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=[the_layer], 
                                                                                    learning_rate___s = learning_rate___s)
                                optim_for_layer.zero_grad()
                                #<  layer before
                                layer__before__output___b_o:torch.Tensor = the_layer(input___b_i)
                                #<  layer backward
                                the_parameters_list = []
                                the_parameters_list.extend(the_layer.parameters())
                                layer__before__output___b_o.backward(gradient=label___b_o.detach().clone(), inputs=the_parameters_list)
                                del the_parameters_list
                                optim_for_layer.step()
                                #<  layer after
                                layer__after__output___b_o:torch.Tensor = the_layer(input___b_i)


                                #<  model      infra
                                the_model = dry_stack_test__DNN_model__2026(in_features=in_dim, out_features=out_dim, layer_count=1)
                                #to make the inner model param the same as the previous test.
                                the_only_layer_in_model = the_model._layers[0]
                                assert isinstance(the_only_layer_in_model, DigitalMapping_layer__2026)
                                with torch.no_grad():
                                    the_only_layer_in_model._raw_weight___oCAP_iCAP[0][:out_dim, :in_dim] = _model_param
                                    pass
                                del _model_param

                                assert the_model._layers.__len__() == 1
                                the_model._layers[0]

                                #optim
                                optim_for_model = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=the_model.parameters(for_optim=True), 
                                                                                    learning_rate___s = learning_rate___s)
                                optim_for_model.zero_grad()
                                #<  model before
                                model__before__output___b_o:torch.Tensor = the_model(input___b_i)
                                #<  model backward
                                the_parameters_list = []
                                the_parameters_list.extend(the_model.parameters(for_backward=True))
                                model__before__output___b_o.backward(gradient=label___b_o.detach().clone(), inputs=the_parameters_list)
                                del the_parameters_list
                                optim_for_model.step()
                                #<  model after
                                model__after__output___b_o:torch.Tensor = the_model(input___b_i)
                                
                                #<  assert
                                
                                assert _tensor_equal(the_layer.get_useful_part_of_raw_weight(), the_only_layer_in_model.get_useful_part_of_raw_weight())
                                assert _tensor_equal(layer__before__output___b_o, model__before__output___b_o)
                                assert _tensor_equal(layer__after__output___b_o,  model__after__output___b_o)
                                pass#for _
                            pass#for learning_rate___s
                        pass#for out_dim7
                    pass#for in_dim
                pass#for batch
                del optim_for___DigitalMapping_layer__2026
            pass#/ test

        if "layer and 1 layer model equivalence               with in_cont    with out_cont" and False:
            from DNN2026.digitalmapping_layer___prototype_test import optim_for___DigitalMapping_layer__2026
            for batch in [2,11,25]:
                for in_dim in [3,15,31,66]:
                    for out_dim in [5, 19, 37]:
                        if in_dim<=out_dim:
                            continue

                        for learning_rate___s in [0.01, 0.123, 0.321]:
                            for _ in range(14):
                                #<  dataset
                                _model_param = torch.empty([])#a temp store. To make two model params the same.
                                input___b_i = rand_sign(size=[batch, in_dim], dtype=torch.float32)#or fp16
                                assert _either_1_or_neg1(input___b_i)
                                label___b_o = partly_reasonable_label_from_input(input___b_i = input___b_i, out_dim = out_dim,
                                        random_ratio = 0., input_is_already_posneg1 = True)
                                assert _either_1_or_neg1(label___b_o)
                                #<  layer      infra
                                the_layer = DigitalMapping_layer__2026(in_features=in_dim, out_features=out_dim)
                                _model_param = the_layer.get_useful_part_of_raw_weight().detach().clone()
                                optim_for_layer = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=[the_layer], 
                                                                                    learning_rate___s = learning_rate___s)
                                optim_for_layer.zero_grad()
                                #<  layer before
                                layer__before__output___b_o:torch.Tensor = the_layer(input___b_i)
                                #<  layer backward
                                the_parameters_list = []
                                the_parameters_list.extend(the_layer.parameters())
                                layer__before__output___b_o.backward(gradient=label___b_o.detach().clone(), inputs=the_parameters_list)
                                del the_parameters_list
                                optim_for_layer.step()
                                #<  layer after
                                layer__after__output___b_o:torch.Tensor = the_layer(input___b_i)


                                #<  data but in containers
                                in_cont = DNN_input_container_2026(batch=batch)
                                in_cont.extend(input___b_i.detach().clone())
                                label_cont = DNN_label_container_2026(data=label___b_o, data_is_already_posneg1=True)
                                #<  model      infra
                                the_model = dry_stack_test__DNN_model__2026(in_features=in_dim, out_features=out_dim, layer_count=1)
                                #to make the inner model param the same as the previous test.
                                the_only_layer_in_model = the_model._layers[0]
                                assert isinstance(the_only_layer_in_model, DigitalMapping_layer__2026)
                                with torch.no_grad():
                                    the_only_layer_in_model._raw_weight___oCAP_iCAP[0][:out_dim, :in_dim] = _model_param
                                    pass
                                del _model_param

                                assert the_model._layers.__len__() == 1
                                the_model._layers[0]

                                #optim
                                optim_for_model = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=the_model.parameters(for_optim=True), 
                                                                                    learning_rate___s = learning_rate___s)
                                optim_for_model.zero_grad()
                                #<  model before
                                model__before__output___b_o:torch.Tensor = the_model(in_cont.get_useful())
                                #<  model backward
                                the_parameters_list = []
                                the_parameters_list.extend(the_model.parameters(for_backward=True))
                                model__before__output___b_o.backward(gradient=label_cont.get_useful(), inputs=the_parameters_list)
                                del the_parameters_list
                                optim_for_model.step()
                                #<  model after
                                model__after__output___b_o:torch.Tensor = the_model(input___b_i)
                                
                                #<  assert
                                
                                assert _tensor_equal(the_layer.get_useful_part_of_raw_weight(), the_only_layer_in_model.get_useful_part_of_raw_weight())
                                assert _tensor_equal(layer__before__output___b_o, model__before__output___b_o)
                                assert _tensor_equal(layer__after__output___b_o,  model__after__output___b_o)

                                pass#for _
                            pass#for learning_rate___s
                        pass#for out_dim7
                    pass#for in_dim
                pass#for batch
            del optim_for___DigitalMapping_layer__2026
            pass#/ test

        if "layer and 2 layer model equivalence               with in_cont    with out_cont" and False:
            from DNN2026.digitalmapping_layer___prototype_test import optim_for___DigitalMapping_layer__2026
            for batch in [2,11,25]:
                for dim_0 in [11,33,64,97,135,167]:
                    for dim_2 in [3,15,31,66,88]:
                        if dim_0<=dim_2:
                            continue
                        for learning_rate___s in [0.01, 0.123, 0.321]:
                            for _ in range(16):

                                #<  dataset
                                _model_param_0 = torch.empty([])#a temp store. To make two model params the same.
                                _model_param_1 = torch.empty([])#a temp store. To make two model params the same.
                                input___b_i = rand_sign(size=[batch, dim_0], dtype=torch.float32)#or fp16
                                assert _either_1_or_neg1(input___b_i)
                                label___b_o = partly_reasonable_label_from_input(input___b_i = input___b_i, out_dim = dim_2,
                                        random_ratio = 0., input_is_already_posneg1 = True)
                                assert _either_1_or_neg1(label___b_o)

                                #<  data but in containers
                                in_cont = DNN_input_container_2026(batch=batch)
                                in_cont.extend(input___b_i.detach().clone())
                                label_cont = DNN_label_container_2026(data=label___b_o, data_is_already_posneg1=True)
                                #<  model      infra
                                the_model = dry_stack_test__DNN_model__2026(in_features=dim_0, out_features=dim_2, layer_count=2)

                                #to make the inner model param the same as the previous test.
                                _temp___layer_in_model_0 = the_model._layers[0]
                                assert isinstance(_temp___layer_in_model_0, DigitalMapping_layer__2026)
                                _model_param_0 = _temp___layer_in_model_0.get_useful_part_of_raw_weight().detach().clone()
                                dim_1 = _model_param_0.shape[0]
                                assert dim_0>=dim_1 
                                assert dim_1>=dim_2
                                _temp___layer_in_model_1 = the_model._layers[1]
                                assert isinstance(_temp___layer_in_model_1, DigitalMapping_layer__2026)
                                _model_param_1 = _temp___layer_in_model_1.get_useful_part_of_raw_weight().detach().clone()


                                #optim
                                optim_for_model = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=the_model.parameters(for_optim=True), 
                                                                                    learning_rate___s = learning_rate___s)
                                optim_for_model.zero_grad()
                                #<  model before
                                model__before__output___b_o:torch.Tensor = the_model(in_cont.get_useful())
                                #<  model backward
                                the_parameters_list = []
                                the_parameters_list.extend(the_model.parameters(for_backward=True))
                                model__before__output___b_o.backward(gradient=label_cont.get_useful(), inputs=the_parameters_list)
                                del the_parameters_list
                                optim_for_model.step()
                                #<  model after
                                model__after__output___b_o:torch.Tensor = the_model(input___b_i)
                                


                                #<  layer      infra
                                the_layer_0 = DigitalMapping_layer__2026(in_features=dim_0, out_features=dim_1)
                                the_layer_1 = DigitalMapping_layer__2026(in_features=dim_1, out_features=dim_2)
                                the_layer_0.set_useful_part_of_raw_weight(input=_model_param_0.detach().clone())
                                the_layer_1.set_useful_part_of_raw_weight(input=_model_param_1.detach().clone())

                                optim_for_layer = optim_for___DigitalMapping_layer__2026(
                                        DigitalMapping_layers=[the_layer_0, the_layer_1], learning_rate___s = learning_rate___s)
                                optim_for_layer.zero_grad()
                                #<  layer before
                                x:torch.Tensor = the_layer_0(input___b_i)
                                layer__before__output___b_o:torch.Tensor = the_layer_1(x)
                                #<  layer backward
                                the_parameters_list = []
                                the_parameters_list.extend(the_layer_0.parameters())
                                the_parameters_list.extend(the_layer_1.parameters())
                                layer__before__output___b_o.backward(gradient=label___b_o.detach().clone(), inputs=the_parameters_list)
                                del the_parameters_list
                                optim_for_layer.step()
                                #<  layer after
                                x = the_layer_0(input___b_i)
                                layer__after__output___b_o:torch.Tensor = the_layer_1(x)


                                #<  assert
                                assert _tensor_equal(the_layer_0.get_useful_part_of_raw_weight(), _temp___layer_in_model_0.get_useful_part_of_raw_weight())
                                assert _tensor_equal(the_layer_1.get_useful_part_of_raw_weight(), _temp___layer_in_model_1.get_useful_part_of_raw_weight())
                                assert _tensor_equal(layer__before__output___b_o, model__before__output___b_o)
                                assert _tensor_equal(layer__after__output___b_o,  model__after__output___b_o)

                                pass#for _
                            pass#for learning_rate___s
                        pass#for dim_2
                    pass#for dim_0
                pass#for batch
            del optim_for___DigitalMapping_layer__2026
            pass#/ test
        return
    ____basic_behavior_of_dry_stack_test________with_optim_for___DigitalMapping_layer__2026()
    pass



'''with the optim part inside model'''
if "basic behavior of dry stack test" and __DEBUG_ME__() and True:
    def ____basic_behavior_of_dry_stack_test____():

        if "remove some output slot" and True:
            for batch in [2]:
                for dim_0 in [11,]:
                    for dim_2 in [3,]:
                        if dim_0<=dim_2:
                            continue
                        for learning_rate___s in [0.6]:#lr 0.3, the acc goes from 0.5 to 0.65 in 1 epoch, according to previous tests.
                            for _ in range(1):

                                #<  dataset
                                input___b_i = rand_sign(size=[batch, dim_0], dtype=torch.float32)#or fp16
                                assert _either_1_or_neg1(input___b_i)
                                assert input___b_i.is_floating_point()

                                _label___b_o = partly_reasonable_label_from_input(input___b_i = input___b_i, out_dim = dim_2,
                                        random_ratio = 0., input_is_already_posneg1 = True)
                                assert _either_1_or_neg1(_label___b_o)
                                assert _label___b_o.is_floating_point()
                                #<  data but in containers
                                in_cont = DNN_input_container_2026(batch=batch)
                                in_cont.extend(input___b_i.detach().clone())
                                #<  model      infra
                                the_model = dry_stack_test__DNN_model__2026(in_features=dim_0, out_features=dim_2, layer_count=2)




                                _temp__layer = the_model._layers[0]
                                assert isinstance(_temp__layer, DigitalMapping_layer__2026)
                                aaaaaaaa0 = _temp__layer._raw_weight___oCAP_iCAP[0].shape
                                assert _temp__layer.in_dim == 11
                                assert _temp__layer.out_dim == 3

                                _temp__layer = the_model._layers[1]
                                assert isinstance(_temp__layer, DigitalMapping_layer__2026)
                                aaaaaaaa1 = _temp__layer._raw_weight___oCAP_iCAP[0].shape
                                assert _temp__layer.in_dim == 3
                                assert _temp__layer.out_dim == 3

                                #epoch 0
                                the_model.zero_grad()
                                output___b_o:torch.Tensor = the_model(in_cont.get_useful())
                                assert _either_1_or_neg1(output___b_o)
                                #

                                #assert label_cont.out_dim() == the_model.out_dim()
                                #_temp___grad_input = label_cont.get_useful()
                                assert _tensor_shape_check(output___b_o, 2, 3)
                                #assert _tensor_shape_check(_temp___grad_input, 2, 3)
                                #assert output___b_o.shape == _temp___grad_input.shape
                                dry_stack_test__DNN_model__2026.backward(the_model = the_model, output___b_o = output___b_o,
                                                                                    label___b_o=_label___b_o)
                                the_model.step(learning_rate___s=learning_rate___s)

                                #<    manually remove some.        doesn't have to be any special slots. Only  debug purpose.
                                test__remove_this = torch.tensor([True, True, False])
                                the_model.remove_output_slot(test__remove_this)
                                #label_cont.remove_output_slot(test__remove_this)
                                del test__remove_this



                                _temp__layer = the_model._layers[0]
                                aaaaaaaa0_after = _temp__layer._raw_weight___oCAP_iCAP[0].shape
                                assert isinstance(_temp__layer, DigitalMapping_layer__2026)
                                assert _temp__layer.in_dim == 11
                                assert _temp__layer.out_dim == 3
                                _temp__layer = the_model._layers[1]
                                assert isinstance(_temp__layer, DigitalMapping_layer__2026)
                                aaaaaaaa1_after = _temp__layer._raw_weight___oCAP_iCAP[0].shape
                                assert _temp__layer.in_dim == 3
                                assert _temp__layer.out_dim == 1
                                #epoch 1
                                the_model.zero_grad()
                                output___b_o:torch.Tensor = the_model(in_cont.get_useful())
                                assert _either_1_or_neg1(output___b_o)
                                assert _tensor_shape_check(output___b_o, batch, 1)
                                #flag_perfect___o, _ = label_cont.detect_good_output___by_position(output___b_o,output_is_already_posneg1=True)

                                #assert label_cont.out_dim() == the_model.out_dim()
                                #_temp___grad_input = label_cont.get_useful()
                                assert _tensor_shape_check(output___b_o, 2, 1)
                                #assert _tensor_shape_check(_temp___grad_input, 2, 1)
                                #assert output___b_o.shape == _temp___grad_input.shape
                                #_temp___grad_input = _label___b_o[:, 2]
                                _temp___grad_input = rand_sign(size=[2,1])
                                dry_stack_test__DNN_model__2026.backward(the_model = the_model, output___b_o = output___b_o, 
                                                                                        label___b_o=_temp___grad_input)
                                the_model.step(learning_rate___s=learning_rate___s)

                                pass#for _
                            pass#for learning_rate___s
                        pass#for dim_2
                    pass#for dim_0
                pass#for batch
            pass#/ test

        '''???????????????????????'''
        if "remove some output slot" and False:
            for batch in [2]:
                for dim_0 in [11,]:
                    for dim_2 in [3,]:
                        if dim_0<=dim_2:
                            continue
                        for learning_rate___s in [0.6]:#lr 0.3, the acc goes from 0.5 to 0.65 in 1 epoch, according to previous tests.
                            for _ in range(1):

                                #<  dataset
                                input___b_i = rand_sign(size=[batch, dim_0], dtype=torch.float32)#or fp16
                                assert _either_1_or_neg1(input___b_i)
                                _label___b_o = partly_reasonable_label_from_input(input___b_i = input___b_i, out_dim = dim_2,
                                        random_ratio = 0., input_is_already_posneg1 = True)
                                assert _either_1_or_neg1(_label___b_o)

                                #<  data but in containers
                                in_cont = DNN_input_container_2026(batch=batch)
                                in_cont.extend(input___b_i.detach().clone())
                                label_cont = DNN_label_container_2026(data=_label___b_o, data_is_already_posneg1=True)
                                del _label___b_o
                                #<  model      infra
                                the_model = dry_stack_test__DNN_model__2026(in_features=dim_0, out_features=dim_2, layer_count=2)
                                optim_for_model = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=the_model.parameters(for_optim=True), 
                                                                                    learning_rate___s = learning_rate___s)




                                _temp__layer = the_model._layers[0]
                                assert isinstance(_temp__layer, DigitalMapping_layer__2026)
                                assert _temp__layer.in_dim == 11
                                assert _temp__layer.out_dim == 3
                                _temp__layer = the_model._layers[1]
                                assert isinstance(_temp__layer, DigitalMapping_layer__2026)
                                aaaaaaaa1 = _temp__layer._raw_weight___oCAP_iCAP[0].shape
                                assert _temp__layer.in_dim == 3
                                assert _temp__layer.out_dim == 3

                                #epoch 0
                                optim_for_model.zero_grad()
                                output___b_o:torch.Tensor = the_model(in_cont.get_useful())
                                assert _either_1_or_neg1(output___b_o)
                                flag_perfect___o, _ = label_cont.detect_good_output___by_position(output___b_o,output_is_already_posneg1=True)

                                assert label_cont.out_dim() == the_model.out_dim()
                                _temp___grad_input = label_cont.get_useful()
                                assert _tensor_shape_check(output___b_o, 2, 3)
                                assert _tensor_shape_check(_temp___grad_input, 2, 3)
                                assert output___b_o.shape == _temp___grad_input.shape
                                backward_to_this_list = []
                                backward_to_this_list.extend(the_model.parameters(for_backward=True))
                                output___b_o.backward(gradient=_temp___grad_input, inputs=backward_to_this_list)
                                optim_for_model.step()


                                test__remove_this = torch.tensor([True, True, False])
                                the_model.remove_output_slot(test__remove_this)
                                label_cont.remove_output_slot(test__remove_this)
                                del test__remove_this



                                _temp__layer = the_model._layers[0]
                                assert isinstance(_temp__layer, DigitalMapping_layer__2026)
                                assert _temp__layer.in_dim == 11
                                assert _temp__layer.out_dim == 3
                                _temp__layer = the_model._layers[1]
                                assert isinstance(_temp__layer, DigitalMapping_layer__2026)
                                aaaaaaaa1 = _temp__layer._raw_weight___oCAP_iCAP[0].shape
                                assert _temp__layer.in_dim == 3
                                assert _temp__layer.out_dim == 1
                                #epoch 1
                                optim_for_model.zero_grad()
                                output___b_o:torch.Tensor = the_model(in_cont.get_useful())
                                assert _either_1_or_neg1(output___b_o)
                                flag_perfect___o, _ = label_cont.detect_good_output___by_position(output___b_o,output_is_already_posneg1=True)

                                assert label_cont.out_dim() == the_model.out_dim()
                                _temp___grad_input = label_cont.get_useful()
                                assert _tensor_shape_check(output___b_o, 2, 1)
                                assert _tensor_shape_check(_temp___grad_input, 2, 1)
                                assert output___b_o.shape == _temp___grad_input.shape
                                backward_to_this_list = []
                                backward_to_this_list.extend(the_model.parameters(for_backward=True))
                                output___b_o.backward(gradient=_temp___grad_input, inputs=backward_to_this_list)
                                optim_for_model.step()

                                pass#for _
                            pass#for learning_rate___s
                        pass#for dim_2
                    pass#for dim_0
                pass#for batch
            pass#/ test







        if "remove some output slot" and True:
            for batch in [2,11,25]:
                for dim_0 in [11,33,64,97,135,167]:
                    for dim_2 in [3,15,31,66,88]:
                        if dim_0<=dim_2:
                            continue
                        for learning_rate___s in [0.3]:#lr 0.3, the acc goes from 0.5 to 0.65 in 1 epoch, according to previous tests.
                            for _ in range(16):

                                #<  dataset
                                input___b_i = rand_sign(size=[batch, dim_0], dtype=torch.float32)#or fp16
                                assert _either_1_or_neg1(input___b_i)
                                _label___b_o = partly_reasonable_label_from_input(input___b_i = input___b_i, out_dim = dim_2,
                                        random_ratio = 0., input_is_already_posneg1 = True)
                                assert _either_1_or_neg1(_label___b_o)

                                #<  data but in containers
                                in_cont = DNN_input_container_2026(batch=batch)
                                in_cont.extend(input___b_i.detach().clone())
                                label_cont = DNN_label_container_2026(data=_label___b_o, data_is_already_posneg1=True)
                                del _label___b_o
                                #<  model      infra
                                the_model = dry_stack_test__DNN_model__2026(in_features=dim_0, out_features=dim_2, layer_count=2)
                                #optim_for_model = optim_for___DigitalMapping_layer__2026(DigitalMapping_layers=the_model.parameters(for_optim=True), 
                                #                                                    learning_rate___s = learning_rate___s)
                                
                                perfect_slots_counts___along_epoch:list[float] = []
                                
                                for epoch in range(5):
                                    #optim_for_model.zero_grad() #优化器也要是新的。。。每一次要用新的。。
                                    the_model.zero_grad() #优化器也要是新的。。。每一次要用新的。。
                                    output___b_o:torch.Tensor = the_model(in_cont.get_useful())
                                    assert _either_1_or_neg1(output___b_o)
                                    flag_perfect___o, _ = label_cont.detect_good_output___by_position(output___b_o,output_is_already_posneg1=True)

                                    assert label_cont.out_dim() == the_model.out_dim()
                                    _temp___grad_input = label_cont.get_useful()
                                    #assert _tensor_shape_check(output___b_o, batch, dim_2)
                                    #assert _tensor_shape_check(_temp___grad_input, batch, dim_2)
                                    assert output___b_o.shape == _temp___grad_input.shape
                                    #print(the_model._report_shape())
                                    ########################
                                    #backward_to_this_list = []
                                    #backward_to_this_list.extend(the_model.parameters(for_backward=True))
                                    
                                    #output___b_o.backward(gradient=_temp___grad_input, inputs=backward_to_this_list)
                                    dry_stack_test__DNN_model__2026.backward(the_model=the_model, output___b_o=output___b_o, label___b_o=label_cont.get_useful())
                                    #optim_for_model.step()
                                    the_model.step(learning_rate___s = learning_rate___s, safety_check=True)

                                    perfect_slots_counts___along_epoch.append(flag_perfect___o.sum().item())
                                    the_model.remove_output_slot(flag_perfect___o)
                                    label_cont.remove_output_slot(flag_perfect___o)
                                    if label_cont.out_dim() == 0:
                                        break
                                    pass
                                #<  model after
                                print(perfect_slots_counts___along_epoch)

                                1w 这个要怎么表达一下？？？读一下。

                                pass#for _
                            pass#for learning_rate___s
                        pass#for dim_2
                    pass#for dim_0
                pass#for batch
            pass#/ test

        






        #device adaption.

        return
    ____basic_behavior_of_dry_stack_test____()
    pass
        
assert False, "反向查index, 利用这个index来输出结果，然后准备集成测试"
assert False, "device adaption.   不用显卡就不写了。"



        





