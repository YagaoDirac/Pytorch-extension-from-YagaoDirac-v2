from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, \
        iota

import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######






if "core idea" and __DEBUG_ME__() and False:
    def ____core_idea____():
        if "example 1" and True:
            out_dim = 2
            in_dim = 3
            batch = 5
            #<  input
            input___b_i = torch.tensor([  [11,   22,   33], 
                                        [111,  122,  133], 
                                        [211,  222,  233], 
                                        [311,  322,  333], 
                                        [411,  422,  433], ])
            assert input___b_i.shape == torch.Size([batch, in_dim])
            #<  model param
            training_buffer___o_i = torch.tensor([   [0.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3]])
            assert training_buffer___o_i.shape == torch.Size([out_dim, in_dim])
            #<  calc
            _temp_one_hot___o = training_buffer___o_i.argmax(dim=1)
            output___b_o = input___b_i[:, _temp_one_hot___o]
            assert output___b_o.shape == torch.Size([batch, out_dim])
            #<  assert
            assert _tensor_equal(output___b_o, [ [ 33,  22], 
                                                [133, 122], 
                                                [233, 222], 
                                                [333, 322], 
                                                [433, 422]])
            pass


        if "example 2" and True:
            out_dim = 3
            in_dim = 4
            batch = 5
            #<  input
            input___b_i = torch.tensor([ [ 11,   22,   33,   44], 
                                        [111,  122,  133,  144], 
                                        [211,  222,  233,  244], 
                                        [311,  322,  333,  344], 
                                        [411,  422,  433,  444], ])
            assert input___b_i.shape == torch.Size([batch, in_dim])
            #<  model param
            training_buffer___o_i = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                            [0.1, 0.2, 1.3, 0.4],
                                            [0.1, 1.2, 0.3, 0.4],])
            assert training_buffer___o_i.shape == torch.Size([out_dim, in_dim])
            #<  calc
            _temp_one_hot___o = training_buffer___o_i.argmax(dim=1)
            output___b_o = input___b_i[:, _temp_one_hot___o]
            assert output___b_o.shape == torch.Size([batch, out_dim])
            #<  assert
            assert _tensor_equal(output___b_o, [ [ 44,  33,  22], 
                                                [144, 133, 122],
                                                [244, 233, 222],
                                                [344, 333, 322],
                                                [444, 433, 422],])
            pass

        return 
    ____core_idea____()
    pass

if "how to update output    example.    not the final version." and __DEBUG_ME__() and False:
    def ____how_to_update_example____():
        if "basic updating    explicityly show the shape." and True:
            out_dim = 2
            in_dim = 3
            batch = 1
            training_buffer___o_i = torch.tensor([[0.1, 0.2, 0.3],
                                            [0.1, 1.2, 0.3]])
            assert training_buffer___o_i.shape == torch.Size([out_dim, in_dim])

            training_target___b_o = torch.tensor([[1., -1.],])
            assert training_target___b_o.shape == torch.Size([ batch, out_dim])
            training_target__b_o_EXPANDi = training_target___b_o.reshape(shape=[batch, out_dim, 1]).expand(size=[-1, -1, in_dim])
            assert training_target__b_o_EXPANDi.shape == torch.Size([batch, out_dim, in_dim])

            input___b_i = torch.tensor([  [1.,  1.,  1.], ])
            assert input___b_i.shape == torch.Size([batch, in_dim])
            input___b_EXPANDo_i = input___b_i.reshape(shape=[batch, 1, in_dim]).expand(size=[-1, out_dim, -1])
            assert input___b_EXPANDo_i.shape == torch.Size([batch, out_dim, in_dim])

            what_to_update__before_sum__b_o_i = training_target__b_o_EXPANDi.mul(input___b_EXPANDo_i)
            assert what_to_update__before_sum__b_o_i.shape == torch.Size([batch, out_dim, in_dim])

            what_to_update___o_i = what_to_update__before_sum__b_o_i.sum(dim=0)
            assert what_to_update___o_i.shape == torch.Size([out_dim, in_dim])

            training_buffer___o_i += what_to_update___o_i
            pass#/ test

        if "basic updating" and True:
            batch = 5
            out_dim = 2
            in_dim = 3
            #<  dataset
            input___b_i = torch.tensor([ [1.,  1.,  1.],
                                        [1.,  1.,  1.],
                                        [1.,  1.,  1.],
                                        [1.,  1.,  1.],
                                        [1.,  1.,  1.],])
            label___b_o = torch.tensor([ [1.,  1.],
                                        [1.,  1.],
                                        [1.,  1.],
                                        [1.,  1.],
                                        [1.,  1.],])
            #<  optimizer param
            lr = 0.2

            #<  model param
            training_buffer___o_i = torch.tensor([   [0.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3]])
            #training_target___b_o____or_empty = torch.empty(size=[0])  not for now.

            #<  forward path
            the_input_for_this_layer___b_i = input___b_i.detach().clone()
            #this layer doesn't need to know what it output ed.

            #<  backward path
            training_target___b_o = label___b_o.detach().clone()

            #<  update the training buffer
            training_target___b_o_EXPANDi = training_target___b_o.reshape(shape=[batch, out_dim, 1]).expand(size=[-1, -1, in_dim])
            input___b_EXPANDo_i = the_input_for_this_layer___b_i.reshape(shape=[batch, 1, in_dim]).expand(size=[-1, out_dim, -1])

            what_to_update__before_sum___b_o_i = training_target___b_o_EXPANDi.mul(input___b_EXPANDo_i)
            what_to_update___o_i = what_to_update__before_sum___b_o_i.sum(dim=0)

            training_buffer___o_i += what_to_update___o_i * lr
            pass#/ test






        return 
    ____how_to_update_example____()
    pass


1w 测一下squeeze
class DNN_input_container_2026(torch.nn.Module):
    '''According to the entire design, this container only provides 1 api.
    
    Call extend function to add extra data points.'''
    _data:torch.nn.parameter.Parameter
    _size:int
    _init_to_nan:bool
    def __init__(self, batch:int, 
                dtype:torch.dtype|None = None, device:torch.device|str|None = "cpu", 
                init_capacity = 16, init_to_nan = False):
        super().__init__()
        self._data = torch.nn.Parameter(torch.empty(size=[batch, init_capacity], 
                    dtype=dtype, device=device, requires_grad=False), requires_grad=False)
        assert self._data.requires_grad == False
        self._size = 0
        self._init_to_nan = init_to_nan
        if init_to_nan:
            self._data.fill_(torch.nan)
            pass
        self._calc_bigger_capacity = lambda a:a*2
        return
    def batch(self)->int:
        '''get'''
        return self._data.shape[0]
    def capacity(self)->int:
        '''get'''
        return self._data.shape[1]
    def get_size(self)->int:
        '''get'''
        return self._size
    def squeeze(self):
        self._data.data = self.get_useful()
        assert False, "untested"
        return
    def extend(self, other:torch.Tensor)->None:
        assert other.shape.__len__() == 2
        assert other.shape[0] == self._data.shape[0]
        with torch.no_grad():
                
            _temp__how_many_to_add = other.shape[1]
            _size_after = self._size + _temp__how_many_to_add
            if _size_after > self.capacity():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity(self.capacity())
                _temp___new_container = torch.empty(size=[self.batch(), _temp___new_capacity],
                        dtype=self._data.dtype, device=self._data.device)
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:, 0:self._size] = self.get_useful()
                self._data.data = _temp___new_container
                pass

            self._data[:, self._size:self._size + _temp__how_many_to_add] = other
            self._size = _size_after
            return
        pass#end of function

    def get_useful(self)->torch.Tensor:
        result = self._data[:,:self._size]
        return result
    
    def __repr__(self):
        return f"{self.get_useful().__repr__()}, size:{self._size}, DNN input container 2026"
    def __str__(self):
        return f"{self.get_useful().__str__()}, size:{self._size}, DNN input container 2026"

    pass

if "how to add input." and __DEBUG_ME__() and True:
    def ____input_container_idea____():
        if "basic idea" and True:
            batch = 2
            #<  the container
            the_container = DNN_input_container_2026(batch=batch,init_capacity=6, init_to_nan=True)
            assert the_container._size == 0
            assert the_container.capacity() == 6
            assert the_container._init_to_nan == True
            the_container.extend(torch.tensor([ [ 11,  22,  33],
                                                [111, 122, 133],]))
            assert the_container._size == 3
            assert the_container.capacity() == 6
            the_container.extend(torch.tensor([ [ 77,  88],
                                                [177, 188],]))
            assert the_container._size == 5
            assert the_container.capacity() == 6
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 
                                                [ 11,  22,  33,  77,  88],
                                                [111, 122, 133, 177, 188],]))
            assert torch.isnan(the_container._data[:,5]).all()

            the_container.extend(torch.tensor([ [ 111,  222],
                                                [1111, 1222],]))
            assert the_container._size == 7
            assert the_container.capacity() == 12
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 
                                                [ 11,  22,  33,  77,  88,  111,  222],
                                                [111, 122, 133, 177, 188, 1111, 1222],]))
            assert torch.isnan(the_container._data[:,7:]).all()
            pass

        if "device adaption" and True:
            the_container = DNN_input_container_2026(batch=2,init_capacity=6 )
            assert the_container._data.device.type == "cpu"
            the_container.cuda()
            assert the_container._data.device.type == "cuda"
            the_container = DNN_input_container_2026(batch=2,init_capacity=6, device="cuda")
            assert the_container._data.device.type == "cuda"
            the_container.cpu()
            assert the_container._data.device.type == "cpu"
            pass

        return
    ____input_container_idea____()
    pass




class _only_for_output_container_to_use____DNN_container_2026(torch.nn.Module):
    '''The only difference from DNN_input_container_2026 is, this class doesn't have batch, and dtype is always int(not uint).'''
    _data:torch.nn.parameter.Parameter
    _size:int
    init_to_neg1:bool
    def __init__(self, dtype:torch.dtype|None = None, device:torch.device|str|None = "cpu", 
                init_capacity = 16, init_to_neg1 = False):
        
        super().__init__()
        if dtype is None:
            dtype = torch.int32
            pass
        self._data = torch.nn.Parameter(torch.empty(size=[init_capacity], 
                    dtype=dtype, device=device, requires_grad=False), requires_grad=False)
        assert self._data.requires_grad == False
        assert self._data.dtype in [torch.int, torch.int32, torch.int64, torch.int16]
        self._size = 0
        self.init_to_neg1 = init_to_neg1
        if init_to_neg1:
            self._data.fill_(-1)
            pass
        self._calc_bigger_capacity = lambda a:a*2
        return
    def capacity(self)->int:
        '''get'''
        return self._data.shape[0]
    def get_size(self)->int:
        '''get'''
        return self._size
    def squeeze(self):
        self._data.data = self.get_useful()
        assert False, "untested"
        return

    def extend(self, other:torch.Tensor)->None:
        assert other.shape.__len__() == 1
        with torch.no_grad():
                
            _temp__how_many_to_add = other.shape[0]
            _size_after = self._size + _temp__how_many_to_add
            if _size_after > self.capacity():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity(self.capacity())
                _temp___new_container = torch.empty(size=[_temp___new_capacity], 
                        dtype=self._data.dtype, device=self._data.device)
                if self.init_to_neg1:
                    _temp___new_container.fill_(-1)
                    pass
                _temp___new_container[0:self._size] = self.get_useful()
                self._data.data = _temp___new_container
                pass

            self._data[self._size:self._size + _temp__how_many_to_add] = other
            self._size = _size_after
            return
        pass#end of function

    def get_useful(self)->torch.Tensor:
        result = self._data[:self._size]
        return result
    
    def __repr__(self):
        return f"{self.get_useful().__repr__()}, size:{self._size}, _only_for_output_container_to_use____DNN_container_2026"
    def __str__(self):
        return f"{self.get_useful().__str__()}, size:{self._size}, _only_for_output_container_to_use____DNN_container_2026"

    pass
if "how to add input." and __DEBUG_ME__() and True:
    def ____test_____only_for_output_container_to_use____DNN_container_2026():
        if "basic idea" and True:
            #<  the container
            the_container = _only_for_output_container_to_use____DNN_container_2026(init_capacity=6, init_to_neg1=True)
            assert the_container._size == 0
            assert the_container.capacity() == 6
            assert the_container.init_to_neg1 == True
            the_container.extend(torch.tensor([ 11,  22,  33]))
            assert the_container._size == 3
            assert the_container.capacity() == 6
            the_container.extend(torch.tensor([ 77,  88]))
            assert the_container._size == 5
            assert the_container.capacity() == 6
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 11,  22,  33,  77,  88]))
            assert _tensor_equal(the_container._data,        torch.tensor([ 11,  22,  33,  77,  88, -1]))

            the_container.extend(torch.tensor([ 111,  222]))
            assert the_container._size == 7
            assert the_container.capacity() == 12
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 11,  22,  33,  77,  88,  111,  222]))
            assert _tensor_equal(the_container._data,        torch.tensor([ 11,  22,  33,  77,  88,  111,  222, -1, -1, -1, -1, -1]))
            pass

        if "device adaption" and True:
            the_container = DNN_input_container_2026(batch=2,init_capacity=6 )
            assert the_container._data.device.type == "cpu"
            the_container.cuda()
            assert the_container._data.device.type == "cuda"
            the_container = DNN_input_container_2026(batch=2,init_capacity=6, device="cuda")
            assert the_container._data.device.type == "cuda"
            the_container.cpu()
            assert the_container._data.device.type == "cpu"
            pass

        return
    ____test_____only_for_output_container_to_use____DNN_container_2026()
    pass



class DNN_output_container_2026(torch.nn.Module):
    data:torch.nn.parameter.Parameter
    #_flag:torch.nn.parameter.Parameter
    def __init__(self, init_data:torch.Tensor):
        '''init_data.shape is [batch, _ ]'''
        super().__init__()
        assert init_data.shape.__len__() == 2

        self.data = torch.nn.Parameter(init_data.detach().clone(), requires_grad=False)
        assert self.data.requires_grad == False

        # self._flag = torch.nn.Parameter(
        #         torch.empty(size=[init_data.shape[1]], dtype=torch.bool, device = init_data.device, requires_grad=False)
        #         , requires_grad=False)
        # assert self._flag.requires_grad == False
        # self._flag.fill_(True)
        return 
    def batch(self)->int:
        '''get'''
        return self.data.shape[0]
    def get_size(self)->int:
        '''get'''
        return self.data.shape[1]
    def keep(self, keep_which:torch.Tensor):
        '''keep_which.shape is [ _ ], dtype is bool or torch.bool'''
        assert keep_which.dtype == torch.bool
        assert keep_which.shape.__len__() == 1
        self.data.data = self.data[:,keep_which]
        return
    def remove(self, remove_what:torch.Tensor):
        '''remove_what.shape is [ _ ], dtype is bool or torch.bool'''
        self.keep(remove_what.logical_not())
        return 
    def detect_perfect_output___by_position(self, the_output:torch.Tensor)->torch.Tensor:
        '''return is the suggestion of which to remove.'''
        self_data = self.get_useful()
        assert self_data.shape == the_output.shape
        #<  calc
        flag_eq__before_all___b_o = self_data.eq(the_output)
        flag_eq___o = flag_eq__before_all___b_o.all(dim=0)
        del flag_eq__before_all___b_o
        assert flag_eq___o.shape == torch.Size([the_output.shape[1]])#debug code
        assert False, "untested"
        return flag_eq___o
    def detect_perfect_output___all_to_all(self, the_output:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        '''return list_of_label, list_of_output
        
        return is the suggestion of which to remove.'''

        batch = self.batch()
        label_dim = self.get_size()
        out_dim = the_output.shape[1]
        #<  data 
        label___b_label = self.data
        assert label___b_label.shape == torch.Size([batch, label_dim])#debug code.
        output___b_o = the_output
        assert output___b_o.shape == torch.Size([batch, out_dim])#debug code.
        #<  calc step 1,     2 datasets to bool matrix.

        #host is 111222333, or 1122
        HOST__label___T___label_EXPANDo_b = label___b_label.T \
                .reshape([label___b_label.shape[1], 1, label___b_label.shape[0]]) \
                .expand([-1, output___b_o.shape[1], -1])
        assert HOST__label___T___label_EXPANDo_b.shape == torch.Size([label_dim, out_dim, batch])#debug code.
        #guest is 123123123, or 1212
        GUEST__output___T___EXPANDlabel_o_b = output___b_o.T \
                .reshape([1, output___b_o.shape[1],  output___b_o.shape[0]]) \
                .expand([label___b_label.shape[1], -1, -1])
        assert GUEST__output___T___EXPANDlabel_o_b.shape == torch.Size([label_dim, out_dim, batch])#debug code.

        flag_eq__before_all___label_o_b = HOST__label___T___label_EXPANDo_b.eq(GUEST__output___T___EXPANDlabel_o_b)

        flag_eq___label_o = flag_eq__before_all___label_o_b.all(dim=2)
        assert flag_eq___label_o.shape == torch.Size([label_dim, out_dim])#debug code.
        assert flag_eq___label_o.dtype == torch.bool#debug code.
        #<  calc step 2,     2d bool to index list.
        list_of_label  = _only_for_output_container_to_use____DNN_container_2026()
        list_of_output = _only_for_output_container_to_use____DNN_container_2026()
        
        iota_of_output_dim = iota(out_dim)
        
        flag__if_this_row_has_something___label = flag_eq___label_o.any(dim=1)
        assert flag__if_this_row_has_something___label.shape == torch.Size([label_dim])
        while True:
            if not flag__if_this_row_has_something___label.any():
                break
            #loop body
            flag_in_int___if_this_row_has_something___label = flag__if_this_row_has_something___label.to(torch.int8)
            ii_row = flag_in_int___if_this_row_has_something___label.argmax()
            this_row___o = flag_eq___label_o[ii_row]
            assert this_row___o.any()#debug code
        
            _temp_what_to_extend = iota_of_output_dim[this_row___o]
            list_of_output.extend(_temp_what_to_extend)
            ii_row_repeated = torch.empty_like(_temp_what_to_extend)
            ii_row_repeated.fill_(ii_row)
            list_of_label.extend(ii_row_repeated)
        
            #tail 
            flag__if_this_row_has_something___label[ii_row] = False
            pass#while true
        return list_of_label.get_useful(), list_of_output.get_useful()
        #end of function.

    def __repr__(self):
        return f"{self.data.__repr__()}, size:{self._size}, DNN output container 2026"
    def __str__(self):
        return f"{self.data.__str__() }, size:{self._size}, DNN output container 2026"
    pass#end of class

if "detect perfect output" and __DEBUG_ME__() and True:
    def ____detect_perfect_output____():
        if "keep and remove test." and True:
            cont = DNN_output_container_2026(torch.tensor([[3,4,5],[13,14,15]]))
            assert _tensor_equal(cont.data, [[3,4,5],[13,14,15]])
            cont.remove(torch.tensor([0,1,0], dtype=torch.bool))
            assert _tensor_equal(cont.data, [[3,5],[13,15]])

            cont = DNN_output_container_2026(torch.tensor([[3,4,5],[13,14,15]]))
            assert _tensor_equal(cont.data, [[3,4,5],[13,14,15]])
            cont.keep(torch.tensor([0,1,0], dtype=torch.bool))
            assert _tensor_equal(cont.data, [[4],[14]])

            pass#/ test

        if "detect perfect output        by position" and False:

            batch = 3
            out_dim = 5
            #<  data 
            label___b_o = torch.tensor([    [0, 0, 0, 0, 1],
                                            [0, 0, 1, 1, 0],
                                            [1, 0, 1, 0, 1],] , dtype=torch.bool)
            assert label___b_o.shape == torch.Size([batch, out_dim])
            output___b_o = torch.tensor([   [1, 1, 1, 0, 1],
                                            [0, 1, 1, 1, 1],
                                            [1, 0, 1, 0, 1],] , dtype=torch.bool)
            assert output___b_o.shape == torch.Size([batch, out_dim])
            assert output___b_o.shape == label___b_o.shape
            #<  calc
            flag_eq__before_all___b_o = label___b_o.eq(output___b_o)
            flag_eq___o = flag_eq__before_all___b_o.all(dim=0)
            assert flag_eq___o.shape == torch.Size([out_dim])
            #<  assert 
            assert flag_eq___o.eq(torch.tensor([0, 0, 0, 1, 0], dtype=torch.bool)).all()
            pass#/ test

        if "detect perfect output        all to all      small ver with int" and True:


            batch = 2
            label_dim = 5
            output_dim = 7
            #<  data 
            label___b_label = torch.tensor([    
                    [1, 2, 3, 4, 5, ],
                    [0, 0, 0, 0, 0, ],])
            assert label___b_label.shape == torch.Size([batch, label_dim])
            output___b_o = torch.tensor([    
                    [2, 2, 5, 9, 5, 2, 7, ],
                    [0, 1, 0, 0, 1, 0, 0, ],])
            assert output___b_o.shape == torch.Size([batch, output_dim])
            #<  calc

            #host is 111222333, or 1122
            HOST__label___T___label_EXPANDo_b = label___b_label.T \
                    .reshape([label___b_label.shape[1], 1, label___b_label.shape[0]]) \
                    .expand([-1, output___b_o.shape[1], -1])
            assert HOST__label___T___label_EXPANDo_b.shape == torch.Size([label_dim, output_dim, batch])
            #guest is 123123123, or 1212
            GUEST__output___T___EXPANDlabel_o_b = output___b_o.T \
                    .reshape([1, output___b_o.shape[1],  output___b_o.shape[0]]) \
                    .expand([label___b_label.shape[1], -1, -1])
            assert GUEST__output___T___EXPANDlabel_o_b.shape == torch.Size([label_dim, output_dim, batch])

            flag_eq__before_all___label_o_b = HOST__label___T___label_EXPANDo_b.eq(GUEST__output___T___EXPANDlabel_o_b)

            flag_eq___label_o = flag_eq__before_all___label_o_b.all(dim=2)
            assert flag_eq___label_o.shape == torch.Size([label_dim, output_dim])
            #<  assert 
            assert _bool_equal___0_as_false(flag_eq___label_o,[
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],])
            
            pass#/ test

        if "2d bool tensor to index list" and True:
            label_dim = 5
            out_dim = 7
            #<  from what 
            flag__the_2d_bool_tensor___label_o = torch.tensor([
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0],], dtype=torch.bool)
            assert flag__the_2d_bool_tensor___label_o.dtype == torch.bool
            assert flag__the_2d_bool_tensor___label_o.shape == torch.Size([label_dim, out_dim])
            #<  calc
            list_of_label  = _only_for_output_container_to_use____DNN_container_2026()
            list_of_output = _only_for_output_container_to_use____DNN_container_2026()

            iota_of_output_dim = iota(out_dim)

            flag__if_this_row_has_something___label = flag__the_2d_bool_tensor___label_o.any(dim=1)
            assert flag__if_this_row_has_something___label.shape == torch.Size([label_dim])
            assert _bool_equal___0_as_false(flag__if_this_row_has_something___label, 
                                                        [0, 1, 0, 1, 1])
            while True:
                if not flag__if_this_row_has_something___label.any():
                    break
                #loop body
                flag_in_int___if_this_row_has_something___label = flag__if_this_row_has_something___label.to(torch.int8)
                ii_row = flag_in_int___if_this_row_has_something___label.argmax()
                this_row___o = flag__the_2d_bool_tensor___label_o[ii_row]
                assert this_row___o.any()#debug code

                _temp_what_to_extend = iota_of_output_dim[this_row___o]
                list_of_output.extend(_temp_what_to_extend)
                ii_row_repeated = torch.empty_like(_temp_what_to_extend)
                ii_row_repeated.fill_(ii_row)
                list_of_label.extend(ii_row_repeated)
                
                #tail 
                flag__if_this_row_has_something___label[ii_row] = False
                pass#while true
            #<  assert

            assert _tensor_equal(list_of_label .get_useful(), [1,1,3,4])
            assert _tensor_equal(list_of_output.get_useful(), [0,5,6,2])
            _temp_assert___set_index_to_false___lable_o = flag__the_2d_bool_tensor___label_o.detach().clone()
            _temp_assert___set_index_to_false___lable_o[list_of_label.get_useful(), list_of_output.get_useful()] = False
            assert _temp_assert___set_index_to_false___lable_o.any() == False

            pass#/ test

        if "combine of previous 2 tests with the new class" and True:
            output_cont = DNN_output_container_2026(torch.tensor([    
                                [1, 2, 3, 4, 5, ],
                                [0, 0, 0, 0, 0, ],]))
            output___b_o = torch.tensor([    
                    [2, 2, 5, 9, 5, 2, 7, ],
                    [0, 1, 0, 0, 1, 0, 0, ],])
            list_of_label, list_of_output = output_cont.detect_perfect_output___all_to_all(output___b_o)

            assert _tensor_equal(list_of_label .data, [1,1,4])
            assert _tensor_equal(list_of_output.data, [0,5,2])
            assert _tensor_equal(output_cont .data[:,1], output___b_o[:,0])
            assert _tensor_equal(output_cont .data[:,1], output___b_o[:,5])
            assert _tensor_equal(output_cont .data[:,4], output___b_o[:,2])
            assert _tensor_equal(output___b_o[:,0],      output___b_o[:,5])


            output_cont = DNN_output_container_2026(torch.tensor([    
                                [1, 2, 3, 4, 5, ],
                                [0, 0, 0, 0, 0, ],]))
            output___b_o = torch.tensor([    
                    [2, 2, 5, 9, 5, 2, 4, ],
                    [0, 1, 0, 0, 1, 0, 0, ],])
            list_of_label, list_of_output = output_cont.detect_perfect_output___all_to_all(output___b_o)
            
            assert _tensor_equal(list_of_label .data, [1,1,3,4])
            assert _tensor_equal(list_of_output.data, [0,5,6,2])
            pass#/ test

        if "detect perfect output        all to all" and True:

            batch = 4
            #<  data 
            output_cont = DNN_output_container_2026(torch.tensor([    
                                [0, 0, 0, 0, 0, 0, 0, 0, ],
                                [0, 0, 0, 0, 1, 1, 1, 1, ],
                                [0, 0, 1, 1, 0, 0, 1, 1, ],
                                [0, 1, 0, 1, 0, 1, 0, 1, ],]))
            output___b_o = torch.tensor([    
                                [1, 0, 1, 0, 1],
                                [0, 1, 1, 1, 1],
                                [1, 0, 1, 0, 1],
                                [1, 0, 1, 1, 1],])
            #<  calc
            list_of_label, list_of_output = output_cont.detect_perfect_output___all_to_all(output___b_o)
            #<  assert
            assert _tensor_equal(list_of_label .data, [4,5])
            assert _tensor_equal(list_of_output.data, [1,3])
            pass#/ test

        return 
    ____detect_perfect_output____()
    pass


