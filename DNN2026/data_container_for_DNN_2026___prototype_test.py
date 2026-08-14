from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _bool_equal___0_as_false, _tensor_shape_check, \
        _either_1_or_neg1, \
        iota
from pytorch_yagaodirac_v2.Random import rand_sign
from DNN2026.DNN_util import Index_container

import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######






# 数据容器，模型容器
#trace back 需要容器的支持。 从整体的class里面得到新的输入数据。
# 重新做干堆测试。





'''input_container'''
'''input_container'''
'''input_container'''

'''申请内存的函数单独拿出来，方便以后调整。'''
def _only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in(
        extra_len:int, 
        len_now:int, batch:int, recommended_min = 16)->int:
    '''return new_in_dim'''
    total_in_dim_needed = extra_len+len_now
    min_new_nelement = total_in_dim_needed*batch
    ONE_M = 1<<20
    if min_new_nelement<ONE_M:
        assert recommended_min>0
        result = total_in_dim_needed*2+recommended_min
        return result
    ONE_G = 1<<30
    if min_new_nelement<ONE_M:
        return int(total_in_dim_needed*1.25)
    return int(total_in_dim_needed*1.1)
    #end of function
if " test" and __DEBUG_ME__() and False:
    "感觉不用很严格？"
    def ____test______only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in():
        if "result must be greater than input combined" and True:

            extra_len   = 0
            len_now     = 0
            batch       = 10

            new_in_dim = _only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now, batch = batch)
            #<  assert
            assert new_in_dim >= extra_len + len_now
            assert new_in_dim < 50

            
            extra_len   = 10
            len_now     = 10
            batch       = 10

            new_in_dim = _only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now, batch = batch)
            #<  assert
            assert new_in_dim >= extra_len + len_now
            assert new_in_dim < 100


            extra_len   = 100
            len_now     = 100
            batch       = 100

            new_in_dim = _only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now, batch = batch)
            #<  assert
            assert new_in_dim >= extra_len + len_now
            assert new_in_dim < 500

            
            extra_len   = 1000
            len_now     = 1000
            batch       = 1000

            new_in_dim = _only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now, batch = batch)
            #<  assert
            assert new_in_dim >= extra_len + len_now
            assert new_in_dim < 3000


            extra_len   = 10000
            len_now     = 10000
            batch       = 10000

            new_in_dim = _only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now, batch = batch)
            #<  assert
            assert new_in_dim >= extra_len + len_now
            assert new_in_dim < 30000

            
            extra_len   = 100000
            len_now     = 100000
            batch       = 100000

            new_in_dim = _only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in(
                    extra_len = extra_len, len_now = len_now, batch = batch)
            #<  assert
            assert new_in_dim >= extra_len + len_now
            assert new_in_dim < 300000

        return
    ____test______only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in()
    pass

'''a container to help you manage the input for DNN_2026'''
class DNN_input_container_2026(torch.nn.Module):
    '''According to the entire design, this container only provides 1 api.
    
    Call extend function to add extra data points.
    
    This container doesn't provide any feature to help you modify the batch dimention.
    It's because it's designed without this feature.'''
    _raw_data___b_CAPi:torch.nn.parameter.Parameter
    _in_dim:int
    _init_to_nan:bool

    #customizable functions.
    _calc_bigger_capacity__for_in   :function

    def __init__(self, batch:int, init_capacity = 16, 
                dtype:torch.dtype|None = None, device:torch.device|str|None = "cpu", 
                init_to_nan = False):
        super().__init__()
        #<  real payload
        self._raw_data___b_CAPi = torch.nn.Parameter(torch.empty(size=[batch, init_capacity], 
                    dtype=dtype, device=device, requires_grad=False), requires_grad=False)
        assert self._raw_data___b_CAPi.dtype in [torch.float, torch.float32, torch.float16, torch.float64, torch.bfloat16]
        assert self._raw_data___b_CAPi.requires_grad == False
        self._in_dim = 0
        self._init_to_nan = init_to_nan
        if init_to_nan:
            self._raw_data___b_CAPi.fill_(torch.nan)
            pass

        self._calc_bigger_capacity__for_in = _only_for_DNN_input_container_2026_to_use____calc_bigger_capacity__for_in

        return
    
    '''shape related getter        getter'''
    def in_dim(self)->int:
        '''get'''
        return self._in_dim
    def capacity(self)->int:
        '''get'''
        return self._raw_data___b_CAPi.shape[1]
    def batch(self)->int:
        '''get'''
        return self._raw_data___b_CAPi.shape[0]
    '''shape modifier             shaper'''
    def get_useful(self)->torch.Tensor:
        result = self._raw_data___b_CAPi[:, :self._in_dim]
        return result
    def squeeze(self):
        with torch.no_grad():
            self._raw_data___b_CAPi.data = self.get_useful()
            return
        #end of function
    def extend(self, other:torch.Tensor)->None:
        #<  safety
        assert other.shape.__len__() == 2
        #assert other.shape[0] == self._raw_data___b_CAPi.shape[0]#they must share the same batch. repeated.
        #<  real payload
        with torch.no_grad():
            how_many_to_add = other.shape[1]
            _size_after = self._in_dim + how_many_to_add
            if _size_after > self.capacity():# get a bigger new capacity first.
                _temp___new_capacity = self._calc_bigger_capacity__for_in(extra_len = how_many_to_add,
                            len_now = self._in_dim, batch = self.batch())
                _temp___new_container = torch.empty(size=[self.batch(), _temp___new_capacity],
                            dtype=self._raw_data___b_CAPi.dtype, device=self._raw_data___b_CAPi.device)
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:, :self._in_dim] = self.get_useful()
                self._raw_data___b_CAPi.data = _temp___new_container
                pass

            self._raw_data___b_CAPi[:, self._in_dim:_size_after] = other
            self._in_dim = _size_after
            return
            assert False, "untested"
        #end of function

    # def extra_repr(self)->str:

    # def __repr__(self):
    #     return f"{self.get_useful().__repr__()}, size:{self._size}, DNN input container 2026"
    # def __str__(self):
    #     return f"{self.get_useful().__str__()}, size:{self._size}, DNN input container 2026"

    pass

if "basic test" and __DEBUG_ME__() and False:
    def ____test____DNN_input_container_2026():
        if "basic idea" and True:
            batch = 2
            #<  the container
            the_container = DNN_input_container_2026(batch=batch,init_capacity=6, init_to_nan=True)
            assert the_container.in_dim() == 0
            assert the_container.capacity() == 6
            assert the_container._init_to_nan == True
            the_container.extend(torch.tensor([ [ 11,  22,  33],
                                                [111, 122, 133],]))
            assert the_container.in_dim() == 3
            assert the_container.capacity() == 6
            the_container.extend(torch.tensor([ [ 77,  88],
                                                [177, 188],]))
            assert the_container.in_dim() == 5
            assert the_container.capacity() == 6
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 
                                                [ 11,  22,  33,  77,  88],
                                                [111, 122, 133, 177, 188],]))
            assert torch.isnan(the_container._raw_data___b_CAPi[:,5]).all()

            the_container.extend(torch.tensor([ [ 111,  222],
                                                [1111, 1222],]))
            assert the_container.in_dim() == 7
            assert the_container.capacity() >= 7
            assert the_container.capacity() <= 50#7*2
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 
                                                [ 11,  22,  33,  77,  88,  111,  222],
                                                [111, 122, 133, 177, 188, 1111, 1222],]))
            assert torch.isnan(the_container._raw_data___b_CAPi[:,7:]).all()
            pass

        if "extend function        scan" and True:
            for batch in [2,8,23]:
                for init_capacity in [3,11,29]:
                    for ori___in_dim in [5,16,41]:
                        for extra___in_dim in [9,19,48]:
                            #<  the container
                            the_container = DNN_input_container_2026(batch=batch,init_capacity=init_capacity, init_to_nan=True)
                            assert the_container.in_dim() == 0
                            _temp_capacity = the_container.capacity()
                            assert _temp_capacity == init_capacity
                            assert the_container._init_to_nan == True
                            del _temp_capacity
                            #data 1
                            the_container.extend(torch.randn(size=[batch, ori___in_dim]))
                            assert the_container.in_dim() == ori___in_dim
                            _temp_capacity = the_container.capacity()
                            _temp_flag___1 = _temp_capacity == init_capacity 
                            _temp_flag___2 = (  _temp_capacity >= ori___in_dim and \
                                                _temp_capacity <= ori___in_dim*2+50)
                            assert _temp_flag___1 or _temp_flag___2
                            del _temp_capacity
                            #data 2
                            the_container.extend(torch.randn(size=[batch, extra___in_dim]))
                            assert the_container.in_dim() == ori___in_dim + extra___in_dim
                            _temp_capacity = the_container.capacity()
                            _temp_flag___1 = _temp_capacity == init_capacity 
                            _temp_flag___2 = (  _temp_capacity >= ori___in_dim + extra___in_dim and \
                                                _temp_capacity <= (ori___in_dim + extra___in_dim)*4)#2)
                            assert _temp_flag___1 or _temp_flag___2
                            del _temp_capacity
                            pass#for batch
                        pass#for init_capacity
                    pass#for ori___in_dim
                pass#for extra___in_dim
            pass#/ test

        if "extend function       get_useful function       scan         with sum" and True:
            for batch in [2,8,23]:
                for init_capacity in [3,11,29]:
                    for ori___in_dim in [5,16,41]:
                        for extra___in_dim in [9,19,48]:
                            #<  the container
                            the_container = DNN_input_container_2026(batch=batch,init_capacity=init_capacity, init_to_nan=True)
                            assert the_container.in_dim() == 0
                            _temp_capacity = the_container.capacity()
                            assert _temp_capacity == init_capacity
                            assert the_container._init_to_nan == True
                            del _temp_capacity
                            #data 1
                            data_1 = torch.randn(size=[batch, ori___in_dim])
                            sum_of_data_1___by_dim_1 = data_1.sum(dim=1)
                            the_container.extend(data_1)

                            assert the_container.in_dim() == ori___in_dim
                            _temp_capacity = the_container.capacity()
                            _temp_flag___1 = _temp_capacity == init_capacity 
                            _temp_flag___2 = (  _temp_capacity >= ori___in_dim and \
                                                _temp_capacity <= ori___in_dim*2+50)
                            assert _temp_flag___1 or _temp_flag___2
                            del _temp_capacity
                            #data 2
                            data_2 = torch.randn(size=[batch, extra___in_dim])
                            sum_of_data_2___by_dim_1 = data_2.sum(dim=1)
                            the_container.extend(data_2)

                            assert the_container.in_dim() == ori___in_dim + extra___in_dim
                            _temp_capacity = the_container.capacity()
                            _temp_flag___1 = _temp_capacity == init_capacity 
                            _temp_flag___2 = (  _temp_capacity >= ori___in_dim + extra___in_dim and \
                                                _temp_capacity <= (ori___in_dim + extra___in_dim)*2+50)
                            assert _temp_flag___1 or _temp_flag___2
                            del _temp_capacity

                            #<  assert
                            _temp_useful_part = the_container.get_useful() 
                            assert _tensor_shape_check(_temp_useful_part, batch, ori___in_dim + extra___in_dim)
                            assert _tensor_equal(sum_of_data_1___by_dim_1+sum_of_data_2___by_dim_1, _temp_useful_part.sum(dim=1))
                            pass#for batch
                        pass#for init_capacity
                    pass#for ori___in_dim
                pass#for extra___in_dim
            pass#/ test

        if "dtype adaption" and True:
            for dtype in [torch.float, torch.float32, torch.float16, torch.float64, torch.bfloat16]:
                the_container = DNN_input_container_2026(batch=2,init_capacity=3, init_to_nan=True, dtype=dtype)
                assert the_container._raw_data___b_CAPi.dtype == dtype
                pass#for dtype

            the_container = DNN_input_container_2026(batch=2,init_capacity=3, init_to_nan=True, dtype=torch.float32)
            assert the_container._raw_data___b_CAPi.dtype == torch.float32
            the_container.to(torch.float16)
            assert the_container._raw_data___b_CAPi.dtype == torch.float16

            pass#/ test

        if "device adaption" and True:
            the_container = DNN_input_container_2026(batch=2,init_capacity=6 )
            assert the_container._raw_data___b_CAPi.device.type == "cpu"
            the_container.cuda()
            assert the_container._raw_data___b_CAPi.device.type == "cuda"
            the_container = DNN_input_container_2026(batch=2,init_capacity=6, device="cuda")
            assert the_container._raw_data___b_CAPi.device.type == "cuda"
            the_container.cpu()
            assert the_container._raw_data___b_CAPi.device.type == "cpu"
            pass

        if "squeeze" and True:
            for batch in [2,8,23]:
                for init_capacity in [3,11,29]:
                    for ori___in_dim in [5,16,41]:
                        for extra___in_dim in [9,19,48]:
                            #<  the container
                            the_container = DNN_input_container_2026(batch=batch,init_capacity=init_capacity, init_to_nan=True)
                            assert the_container.in_dim() == 0
                            _temp_capacity = the_container.capacity()
                            assert _temp_capacity == init_capacity
                            assert the_container._init_to_nan == True
                            del _temp_capacity
                            #data 1
                            the_container.extend(torch.randn(size=[batch, ori___in_dim]))
                            assert the_container.in_dim() == ori___in_dim
                            _temp_capacity = the_container.capacity()
                            _temp_flag___1 = _temp_capacity == init_capacity 
                            _temp_flag___2 = (  _temp_capacity >= ori___in_dim and \
                                                _temp_capacity <= ori___in_dim*2+50)
                            assert _temp_flag___1 or _temp_flag___2
                            del _temp_capacity

                            #squeeze
                            the_container.squeeze()
                            _temp_capacity = the_container.capacity()
                            assert _temp_capacity == ori___in_dim

                            #data 2
                            the_container.extend(torch.randn(size=[batch, extra___in_dim]))
                            assert the_container.in_dim() == ori___in_dim + extra___in_dim
                            _temp_capacity = the_container.capacity()
                            _temp_flag___1 = _temp_capacity == init_capacity 
                            _temp_flag___2 = (  _temp_capacity >= ori___in_dim + extra___in_dim and \
                                                _temp_capacity <= (ori___in_dim + extra___in_dim)*2+50)
                            assert _temp_flag___1 or _temp_flag___2
                            del _temp_capacity

                            #squeeze
                            the_container.squeeze()
                            _temp_capacity = the_container.capacity()
                            assert _temp_capacity == ori___in_dim + extra___in_dim
                            pass#for batch
                        pass#for init_capacity
                    pass#for ori___in_dim
                pass#for extra___in_dim
            pass#/ test

        return
    ____test____DNN_input_container_2026()
    pass










'''label_container'''
'''label_container'''
'''label_container'''
class DNN_label_container_2026(torch.nn.Module):
    '''According to the entire design, this container only provides 1 api.
    
    Call extend function to add extra data points.
    
    This container doesn't provide any feature to help you modify the batch dimention.
    It's because it's designed without this feature.'''
    data___b_o:torch.nn.parameter.Parameter

    def __init__(self, data: torch.Tensor, 
                    data_is_already_posneg1:bool,
                    detach_clone_the_data = True, 
                    ):#, 
                #dtype:torch.dtype|None = None, device:torch.device|str|None = "cpu", 
        super().__init__()
        #<  safety
        assert data.dtype in [torch.float, torch.float32, torch.float16, torch.float64, torch.bfloat16]
        assert data_is_already_posneg1, "还没想好。"
        #<  real payload
        if detach_clone_the_data:
            self.data___b_o = torch.nn.Parameter(data.detach().clone(), requires_grad=False)
            pass
        else:
            self.data___b_o = torch.nn.Parameter(data, requires_grad=False)
            pass
        assert self.data___b_o.dtype in [torch.float, torch.float32, torch.float16, torch.float64, torch.bfloat16]
        assert self.data___b_o.requires_grad == False
        return

    def get_useful(self)->torch.Tensor:
        '''Probably all the other classes have this function. To make your life easier, this class should also have this function.'''
        return self.data___b_o
    '''shape related getter        getter'''
    def out_dim(self)->int:
        '''get'''
        return self.data___b_o.shape[1]
    def batch(self)->int:
        '''get'''
        return self.data___b_o.shape[0]
    '''shape modifier             shaper'''
    
    def keep_output_slot(self, keep_which:torch.Tensor)->None:
        '''This function also squeeze the memory to minimum.'''
        assert keep_which.shape.__len__() == 1
        assert keep_which.dtype == torch.bool
        #<  real payload
        with torch.no_grad():
            _temp___to_keep = self.data___b_o[:, keep_which]
            self.data___b_o.data = _temp___to_keep
            return 
        #end of function
    def remove_output_slot(self, remove_which:torch.Tensor)->None:
            self.keep_output_slot(remove_which.logical_not())
            return
    
    def detect_good_output___by_position(self, output_posneg1___b_o:torch.Tensor, output_is_already_posneg1:bool,
            good_threshold:torch.Tensor|float, perfect_threshold = 0.999, 
            inner_calc_dtype = torch.float32, safety = False  )->tuple[torch.Tensor|torch.Tensor]:
        '''return flag_perfect___o, flag_good_enough___o'''
        #<  safety
        assert output_is_already_posneg1, "还没想好。"
        if safety:
            assert perfect_threshold > 0.5
            assert good_threshold > 0.5
            # label_posneg1___b_o is self.data___b_o in class.
            assert _either_1_or_neg1(self.data___b_o)
            assert _either_1_or_neg1(output_posneg1___b_o)
            #assert output_posneg1___b_o.shape == self.data___b_o.shape#duplicated.
            pass
        #<  data 

        #<  calc
        flag_eq__before_mean___b_o = self.data___b_o.eq(output_posneg1___b_o)
        mean_acc___o = flag_eq__before_mean___b_o.to(inner_calc_dtype).mean(dim=0)
        #<  perfect_threshold
        flag_perfect___o = mean_acc___o.gt(perfect_threshold)
        #<  good_threshold
        flag_good_enough___o = mean_acc___o.gt(good_threshold)
        flag_good_enough___o = flag_good_enough___o.logical_and(flag_perfect___o.logical_not())

        return flag_perfect___o, flag_good_enough___o
    
    if "no plan for now" and False:
        def detect_perfect_output___all_to_all(self, the_output:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
            '''return list_of_label, list_of_output
            
            return is the suggestion of which to remove.'''
            #<  debug      shape
            batch = self.batch()#debug code.
            label_dim = self.out_dim()#debug code.
            out_dim = the_output.shape[1]#debug code.
            #<  data 
            label___b_label = self.data___b_o
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
            list_of_label  = Index_container()
            list_of_output = Index_container()
            
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
            assert False, "untested"  
            return list_of_label.get_useful(), list_of_output.get_useful()
            #end of function.
        pass


    '''stringify'''

    # def extra_repr(self)->str:

    # def __repr__(self):
    #     return f"{self.get_useful().__repr__()}, size:{self._size}, DNN input container 2026"
    # def __str__(self):
    #     return f"{self.get_useful().__str__()}, size:{self._size}, DNN input container 2026"

    pass

if "basic test" and __DEBUG_ME__() and False:
    def ____test____DNN_label_container_2026():
        if "basic idea" and True:
            batch = 2
            #<  the container
            the_container = DNN_label_container_2026(data= \
                        torch.tensor([  [ 11.,  22,  33],
                                        [111, 122, 133],]))
            assert the_container.out_dim() == 3
            assert the_container.batch() == 2
            #assert the_container.capacity() == 6
            #assert the_container._init_to_nan == True

            the_container.keep_output_slot(torch.tensor([1, 0, 1 ], dtype=torch.bool))
            assert the_container.out_dim() == 2
            assert the_container.batch() == 2
            assert the_container.data___b_o.eq(torch.tensor([   [ 11,  33],
                                                                [111, 133],])).all()

            the_container.keep_output_slot(torch.tensor([0, 1 ], dtype=torch.bool))
            assert the_container.out_dim() == 1
            assert the_container.batch() == 2
            assert the_container.data___b_o.eq(torch.tensor([   [ 33],
                                                                [133],])).all()
            pass#/ test

        if "keep function        scan" and True:
            for batch in [2,8,23]:
                for ori___out_dim in [3,11,29]:
                    #<  the container
                    the_container = DNN_label_container_2026(data = torch.randn(size=[batch, ori___out_dim]))
                    assert the_container.out_dim() == ori___out_dim
                    assert the_container.batch() == batch
                    #<  the answer
                    the_answer___o = torch.rand(size=[ori___out_dim]).gt(0.5)
                    number_of_answer = the_answer___o.sum()
                    #<  calc
                    the_container.keep_output_slot(the_answer___o)
                    assert the_container.out_dim() == number_of_answer
                    assert the_container.batch() == batch
                    pass#for ori___out_dim
                pass#for batch
            pass#/ test

        if "keep function        scan" and True:
            for batch in [2,8,23]:
                for ori___out_dim in [3,11,29]:
                    #<  the container
                    the_container_keep = DNN_label_container_2026(data = torch.randn(size=[batch, ori___out_dim]))
                    the_container_remove = DNN_label_container_2026(data = the_container_keep.data___b_o, detach_clone_the_data=True)
                    #<  the answer
                    the_answer___o = torch.rand(size=[ori___out_dim]).gt(0.5)
                    #<  calc
                    the_container_keep.keep_output_slot(the_answer___o)
                    the_container_remove.remove_output_slot(the_answer___o.logical_not())

                    assert the_container_keep.data___b_o.eq(the_container_remove.data___b_o).all()
                    pass#for ori___out_dim
                pass#for batch
            pass#/ test

        if "dtype adaption" and True:
            for dtype in [torch.float, torch.float32, torch.float16, torch.float64, torch.bfloat16]:
                the_container = DNN_label_container_2026(data = torch.randn(size=[2, 3], dtype=dtype))
                assert the_container.data___b_o.dtype == dtype
                pass#for dtype

            the_container = DNN_label_container_2026(data = torch.randn(size=[2, 3], dtype=torch.float32))
            assert the_container.data___b_o.dtype == torch.float32
            the_container.to(torch.float16)
            assert the_container.data___b_o.dtype == torch.float16
            pass#/ test

        if "device adaption" and True:
            the_container = DNN_label_container_2026(data = torch.randn(size=[2, 3], device="cpu"))
            assert the_container.data___b_o.device.type == "cpu"
            the_container.cuda()
            assert the_container.data___b_o.device.type == "cuda"
            the_container = DNN_label_container_2026(data = torch.randn(size=[2, 3], device="cuda"))
            assert the_container.data___b_o.device.type == "cuda"
            the_container.cpu()
            assert the_container.data___b_o.device.type == "cpu"
            pass

        return
    ____test____DNN_label_container_2026()
    pass

if "detect perfect output         only the by position version" and __DEBUG_ME__() and True:
    def ____detect_perfect_output___by_position____():

        if "detect good output        by position       also test" and False:

            batch = 3
            out_dim = 5
            perfect_threshold = 0.999
            assert perfect_threshold > 0.5
            good_threshold = 0.65
            assert good_threshold > 0.5
            inner_calc_dtype = torch.float32
            #<  data 
            label_posneg1___b_o = torch.tensor([    [-1, -1, -1, -1,  1],
                                                    [-1, -1,  1,  1, -1],
                                                    [ 1, -1,  1, -1,  1],])
            assert _either_1_or_neg1(label_posneg1___b_o)
            assert label_posneg1___b_o.shape == torch.Size([batch, out_dim])

            output_posneg1___b_o = torch.tensor([   [ 1,  1,  1, -1,  1],
                                                    [-1,  1,  1,  1,  1],
                                                    [ 1, -1,  1, -1, -1],])
            assert _either_1_or_neg1(output_posneg1___b_o)
            assert output_posneg1___b_o.shape == torch.Size([batch, out_dim])
            assert output_posneg1___b_o.shape == label_posneg1___b_o.shape
            #<  calc
            flag_eq__before_mean___b_o = label_posneg1___b_o.eq(output_posneg1___b_o)
            assert flag_eq__before_mean___b_o.eq(torch.tensor([ [0, 0, 0, 1, 1],
                                                                [1, 0, 1, 1, 0],
                                                                [1, 1, 1, 1, 0],], dtype=torch.bool)).all()
            mean_acc___o = flag_eq__before_mean___b_o.to(inner_calc_dtype).mean(dim=0)
            assert _tensor_equal(mean_acc___o, [0.6, 0.3, 0.6, 1., 0.3], epsilon=0.1)
            #<  perfect_threshold
            flag_perfect___o = mean_acc___o.gt(perfect_threshold)
            assert flag_perfect___o.shape == torch.Size([out_dim])
            assert flag_perfect___o.eq(torch.tensor([0, 0, 0, 1, 0], dtype=torch.bool)).all()
            #<  good_threshold
            flag_good_enough___o = mean_acc___o.gt(good_threshold)
            assert flag_good_enough___o.shape == torch.Size([out_dim])
            assert flag_good_enough___o.eq(torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)).all()
            flag_good_enough___o = flag_good_enough___o.logical_and(flag_perfect___o.logical_not())
            assert flag_good_enough___o.shape == torch.Size([out_dim])
            assert flag_good_enough___o.eq(torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)).all()

            pass#/ test

        if "simplified version. before into a function. not a test" and False:
            #<  param
            output_posneg1___b_o
            # batch = 3
            # out_dim = 5
            perfect_threshold = 0.999
            good_threshold = 0.65
            inner_calc_dtype = torch.float32
            #<  safety
            assert perfect_threshold > 0.5
            assert good_threshold > 0.5
            # label_posneg1___b_o is self.data in class.
            assert _either_1_or_neg1(label_posneg1___b_o)
            assert _either_1_or_neg1(output_posneg1___b_o)
            assert output_posneg1___b_o.shape == label_posneg1___b_o.shape
            #<  data 

            #<  calc
            flag_eq__before_mean___b_o = label_posneg1___b_o.eq(output_posneg1___b_o)
            mean_acc___o = flag_eq__before_mean___b_o.to(inner_calc_dtype).mean(dim=0)
            #<  perfect_threshold
            flag_perfect___o = mean_acc___o.gt(perfect_threshold)
            #<  good_threshold
            flag_good_enough___o = mean_acc___o.gt(good_threshold)
            flag_good_enough___o = flag_good_enough___o.logical_and(flag_perfect___o.logical_not())

            #return flag_perfect___o, flag_good_enough___o
            pass#/ test

        if "class function equivalence" and False:
            perfect_threshold = 0.999
            assert perfect_threshold > 0.5
            good_threshold = 0.65
            assert good_threshold > 0.5
            inner_calc_dtype = torch.float32

            for batch in [3,11,35]:
                for out_dim in [5,18,51]:
                    #<  data 
                    the_out_container = DNN_label_container_2026(
                            data=rand_sign(size=[batch, out_dim], dtype=torch.float32))

                    label_posneg1___b_o = the_out_container.data___b_o.detach().clone()
                    assert _either_1_or_neg1(label_posneg1___b_o)

                    output_posneg1___b_o = rand_sign(size=[batch, out_dim], dtype=torch.float32)
                    assert _either_1_or_neg1(output_posneg1___b_o)

                    assert output_posneg1___b_o.shape == label_posneg1___b_o.shape
                    #<  calc
                    flag_eq__before_mean___b_o = label_posneg1___b_o.eq(output_posneg1___b_o.detach().clone())
                    mean_acc___o = flag_eq__before_mean___b_o.to(inner_calc_dtype).mean(dim=0)

                    #<  perfect_threshold
                    flag_perfect___o = mean_acc___o.gt(perfect_threshold)
                    assert flag_perfect___o.shape == torch.Size([out_dim])
                    #<  good_threshold
                    flag_good_enough___o = mean_acc___o.gt(good_threshold)
                    flag_good_enough___o = flag_good_enough___o.logical_and(flag_perfect___o.logical_not())
                    assert flag_good_enough___o.shape == torch.Size([out_dim])

                    #<  class version
                    class_ver___flag_perfect___o, class_ver___flag_good_enough___o = \
                            the_out_container.detect_good_output___by_position( \
                                    output_posneg1___b_o = output_posneg1___b_o.detach().clone(),
                                    good_threshold = good_threshold, perfect_threshold = perfect_threshold)
                    #<  assert
                    assert flag_perfect___o.eq(class_ver___flag_perfect___o).all()
                    assert flag_good_enough___o.eq(class_ver___flag_good_enough___o).all()
                    pass#for out_dim 
                pass#for batch
            pass#/ test

        if "threshold" and False:
            #<  data
            the_out_container = DNN_label_container_2026(data=torch.tensor([ 
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1.],
                            ]), data_is_already_posneg1 = True)
            output_posneg1___b_o = torch.tensor([   
                    [1,-1,-1,-1,-1,-1],
                    [1, 1,-1,-1,-1,-1],
                    [1, 1, 1,-1,-1,-1],
                    [1, 1, 1, 1,-1,-1],
                    [1, 1, 1, 1, 1,-1.],
                            ])
            assert _either_1_or_neg1(output_posneg1___b_o)
            assert output_posneg1___b_o.shape == the_out_container.data___b_o.shape
            #<  calc
            flag_perfect___o, flag_good_enough___o = \
                    the_out_container.detect_good_output___by_position( \
                            output_posneg1___b_o = output_posneg1___b_o,output_is_already_posneg1=True,
                            good_threshold = 0.55)
            #<  assert
            assert flag_perfect___o.    eq(torch.tensor([1,0,0,0,0,0], dtype=torch.bool)).all()
            assert flag_good_enough___o.eq(torch.tensor([0,1,1,0,0,0], dtype=torch.bool)).all()


            #<  data
            the_out_container = DNN_label_container_2026(data=torch.tensor([ 
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1.],
                            ]), data_is_already_posneg1 = True)
            output_posneg1___b_o = torch.tensor([   
                    [1,-1,-1,-1,-1,-1],
                    [1, 1,-1,-1,-1,-1],
                    [1, 1, 1,-1,-1,-1],
                    [1, 1, 1, 1,-1,-1],
                    [1, 1, 1, 1, 1,-1.],
                            ])
            assert _either_1_or_neg1(output_posneg1___b_o)
            assert output_posneg1___b_o.shape == the_out_container.data___b_o.shape
            #<  calc
            flag_perfect___o, flag_good_enough___o = \
                                the_out_container.detect_good_output___by_position( \
                                        output_posneg1___b_o = output_posneg1___b_o,output_is_already_posneg1=True,
                                        good_threshold = 0.75)
            #<  assert
            assert flag_perfect___o.    eq(torch.tensor([1,0,0,0,0,0], dtype=torch.bool)).all()
            assert flag_good_enough___o.eq(torch.tensor([0,1,0,0,0,0], dtype=torch.bool)).all()

            pass#/ test










        if "no scan         what if the perfect/good output slots are removed?          measured by the acc(is 0 or not 0)" and True:

            #<  data
            the_out_container = DNN_label_container_2026(data=torch.tensor([ 
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1.],
                            ]), data_is_already_posneg1 = True)
            output_posneg1___b_o = torch.tensor([   
                    [1,-1,-1,-1,-1,-1],
                    [1, 1,-1,-1,-1,-1],
                    [1, 1, 1,-1,-1,-1],
                    [1, 1, 1, 1,-1,-1],
                    [1, 1, 1, 1, 1,-1.],
                            ])
            assert _either_1_or_neg1(output_posneg1___b_o)
            assert output_posneg1___b_o.shape == the_out_container.data___b_o.shape
            #<  calc
            flag_perfect___o, flag_good_enough___o = \
                    the_out_container.detect_good_output___by_position( \
                            output_posneg1___b_o = output_posneg1___b_o,output_is_already_posneg1=True,
                            good_threshold = 0.55)
            #<  assert
            assert flag_perfect___o.    eq(torch.tensor([1,0,0,0,0,0], dtype=torch.bool)).all()
            assert flag_good_enough___o.eq(torch.tensor([0,1,1,0,0,0], dtype=torch.bool)).all()
            #<  remove perfect
            the_out_container.remove_output_slot(flag_perfect___o)
            no_perfect___flag_perfect___o, no_perfect___flag_good_enough___o = \
                    the_out_container.detect_good_output___by_position( \
                            output_posneg1___b_o = output_posneg1___b_o,output_is_already_posneg1=True,
                            good_threshold = 0.55)
            assert no_perfect___flag_perfect___o.any() == False# no perfect
            assert no_perfect___flag_perfect___o.    eq(torch.tensor([0,0,0,0,0], dtype=torch.bool)).all()
            assert no_perfect___flag_good_enough___o.eq(torch.tensor([1,1,0,0,0], dtype=torch.bool)).all()

            #<  remove good
            the_out_container.remove_output_slot(no_perfect___flag_good_enough___o)
            no_perfect_no_good___flag_perfect___o, no_perfect_no_good___flag_good_enough___o = \
                    the_out_container.detect_good_output___by_position( \
                            output_posneg1___b_o = output_posneg1___b_o,output_is_already_posneg1=True,
                            good_threshold = 0.55)
            assert no_perfect_no_good___flag_perfect___o.any() == False# no perfect
            assert no_perfect_no_good___flag_good_enough___o.any() == False# no perfect
            assert no_perfect_no_good___flag_perfect___o.    eq(torch.tensor([0,0,0], dtype=torch.bool)).all()
            assert no_perfect_no_good___flag_good_enough___o.eq(torch.tensor([0,0,0], dtype=torch.bool)).all()



            #<  data
            the_out_container = DNN_label_container_2026(data=torch.tensor([ 
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1.],
                            ]), data_is_already_posneg1 = True)
            output_posneg1___b_o = torch.tensor([   
                    [1,-1,-1,-1,-1,-1],
                    [1, 1,-1,-1,-1,-1],
                    [1, 1, 1,-1,-1,-1],
                    [1, 1, 1, 1,-1,-1],
                    [1, 1, 1, 1, 1,-1.],
                            ])
            assert _either_1_or_neg1(output_posneg1___b_o)
            assert output_posneg1___b_o.shape == the_out_container.data___b_o.shape
            #<  calc
            flag_perfect___o, flag_good_enough___o = \
                    the_out_container.detect_good_output___by_position( \
                            output_posneg1___b_o = output_posneg1___b_o,output_is_already_posneg1=True,
                            good_threshold = 0.55)
            #<  assert
            assert flag_perfect___o.    eq(torch.tensor([1,0,0,0,0,0], dtype=torch.bool)).all()
            assert flag_good_enough___o.eq(torch.tensor([0,1,1,0,0,0], dtype=torch.bool)).all()
            #<  remove perfect
            the_out_container.remove_output_slot(flag_good_enough___o)
            no_good___flag_perfect___o, no_good___flag_good_enough___o = \
                    the_out_container.detect_good_output___by_position( \
                            output_posneg1___b_o = output_posneg1___b_o,output_is_already_posneg1=True,
                            good_threshold = 0.55)
            assert no_perfect___flag_perfect___o.    eq(torch.tensor([1,0,0,0], dtype=torch.bool)).all()
            assert no_perfect___flag_good_enough___o.any() == False# no perfect
            assert no_perfect___flag_good_enough___o.eq(torch.tensor([0,0,0,0], dtype=torch.bool)).all()

            pass#/ test









        return 
    ____detect_perfect_output___by_position____()
    pass








1w
1w
1w
1w测试里面还有没跑通的   878行






# if "detect perfect output         the all to all version, but let me leave it for now." and False:
#     def ____no_plan_for_now():
#         if "xxxxxxxxxxxxxxx 多半写错了，先不用。 only perfect detection        all to all      small ver with int" and True:

#             batch = 2
#             label_dim = 5
#             output_dim = 7
#             #<  data 
#             label___b_label = torch.tensor([    
#                     [1, 2, 3, 4, 5, ],
#                     [0, 0, 0, 0, 0, ],])
#             assert label___b_label.shape == torch.Size([batch, label_dim])
#             output___b_o = torch.tensor([    
#                     [2, 2, 5, 9, 5, 2, 7, ],
#                     [0, 1, 0, 0, 1, 0, 0, ],])
#             assert output___b_o.shape == torch.Size([batch, output_dim])
#             #<  calc

#             #host is 111222333, or 1122
#             HOST__label___T___label_EXPANDo_b = label___b_label.T \
#                     .reshape([label___b_label.shape[1], 1, label___b_label.shape[0]]) \
#                     .expand([-1, output___b_o.shape[1], -1])
#             assert HOST__label___T___label_EXPANDo_b.shape == torch.Size([label_dim, output_dim, batch])
#             #guest is 123123123, or 1212
#             GUEST__output___T___EXPANDlabel_o_b = output___b_o.T \
#                     .reshape([1, output___b_o.shape[1],  output___b_o.shape[0]]) \
#                     .expand([label___b_label.shape[1], -1, -1])
#             assert GUEST__output___T___EXPANDlabel_o_b.shape == torch.Size([label_dim, output_dim, batch])

#             flag_eq__before_all___label_o_b = HOST__label___T___label_EXPANDo_b.eq(GUEST__output___T___EXPANDlabel_o_b)

#             flag_eq___label_o = flag_eq__before_all___label_o_b.all(dim=2)
#             assert flag_eq___label_o.shape == torch.Size([label_dim, output_dim])
#             #<  assert 
#             assert _bool_equal___0_as_false(flag_eq___label_o,[
#                     [0, 0, 0, 0, 0, 0, 0],
#                     [1, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0, 0],])

#             1w
#             pass#/ test






#         if "detect perfect output        all to all      small ver with int" and True:

#             batch = 3
#             label_dim = 5
#             output_dim = 7
#             threshold = 0.65
#             assert threshold > 0.5
#             #<  data 
#             label___b_label = torch.tensor([    
#                     [ 1,  2,  3,  4,  5, ],
#                     [ 0,  0,  0,  0,  0, ],
#                     [ 0,  0,  0,  0,  0, ],])
#             assert label___b_label.shape == torch.Size([batch, label_dim])
#             output___b_o = torch.tensor([    
#                     [ 1, -1,  2,  3,  4, -1,  5, ],
#                     [-1, -1,  0,  0,  0, -1, -1, ],
#                     [-1, -1,  0, -1, -1, -1, -1, ],])
#             assert output___b_o.shape == torch.Size([batch, output_dim])
#             #<  calc

#             #host is 111222333, or 1122
#             1w 维度关系重新看看。 先transpose会更快。单独写。记得在T之前chunk
#             HOST__label___T___label_EXPANDo_b = label___b_label.T \
#                     .reshape([label___b_label.shape[1], 1, label___b_label.shape[0]]) \
#                     .expand([-1, output___b_o.shape[1], -1])
#             assert HOST__label___T___label_EXPANDo_b.shape == torch.Size([label_dim, output_dim, batch])
#             #guest is 123123123, or 1212
#             GUEST__output___T___EXPANDlabel_o_b = output___b_o.T \
#                     .reshape([1, output___b_o.shape[1],  output___b_o.shape[0]]) \
#                     .expand([label___b_label.shape[1], -1, -1])
#             assert GUEST__output___T___EXPANDlabel_o_b.shape == torch.Size([label_dim, output_dim, batch])

#             flag_eq__before_calc___label_o_b = HOST__label___T___label_EXPANDo_b.eq(GUEST__output___T___EXPANDlabel_o_b)
#             mean_acc___label_o = flag_eq__before_calc___label_o_b.mean(dim=2 )1w



#             #<  perfect
#             flag_perfect___label_o = mean_acc___label_o.gt(0.9999)
#             assert flag_perfect___label_o.shape == torch.Size([label_dim, out_dim])
#             assert flag_perfect___label_o.eq(torch.tensor([ [0, 0, 0, 0, 0, 0, 0],
#                                                             [0, 0, 1, 0, 0, 0, 0],
#                                                             [0, 0, 0, 0, 0, 0, 0],
#                                                             [0, 0, 0, 0, 0, 0, 0],
#                                                             [0, 0, 0, 0, 0, 0, 0],] dtype=torch.bool)).all()
#             #<  threshold
#             flag_good_enough___label_o = mean_acc___label_o.gt(threshold)
#             flag_good_enough___label_o = flag_good_enough___label_o.logical_and(flag_perfect___label_o.logical_not())
#             assert flag_good_enough___label_o.shape == torch.Size([out_dim])
#             assert flag_good_enough___label_o.eq(torch.tensor([ [0, 0, 0, 0, 0, 0, 0],
#                                                                 [0, 0, 0, 0, 0, 0, 0],
#                                                                 [0, 0, 0, 1, 0, 0, 0],
#                                                                 [0, 0, 0, 0, 1, 0, 0],
#                                                                 [0, 0, 0, 0, 0, 0, 0],] dtype=torch.bool)).all()



#             1w
#             pass#/ test







#         if "2d bool tensor to index list" and True:
#             label_dim = 5
#             out_dim = 7
#             #<  from what 
#             flag__the_2d_bool_tensor___label_o = torch.tensor([
#                     [0, 0, 0, 0, 0, 0, 0],
#                     [1, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 1],
#                     [0, 0, 1, 0, 0, 0, 0],], dtype=torch.bool)
#             assert flag__the_2d_bool_tensor___label_o.dtype == torch.bool
#             assert flag__the_2d_bool_tensor___label_o.shape == torch.Size([label_dim, out_dim])
#             #<  calc
#             list_of_label  = Index_container()
#             list_of_output = Index_container()

#             iota_of_output_dim = iota(out_dim)

#             flag__if_this_row_has_something___label = flag__the_2d_bool_tensor___label_o.any(dim=1)
#             assert flag__if_this_row_has_something___label.shape == torch.Size([label_dim])
#             assert _bool_equal___0_as_false(flag__if_this_row_has_something___label, 
#                                                         [0, 1, 0, 1, 1])
#             while True:
#                 if not flag__if_this_row_has_something___label.any():
#                     break
#                 #loop body
#                 flag_in_int___if_this_row_has_something___label = flag__if_this_row_has_something___label.to(torch.int8)
#                 ii_row = flag_in_int___if_this_row_has_something___label.argmax()
#                 this_row___o = flag__the_2d_bool_tensor___label_o[ii_row]
#                 assert this_row___o.any()#debug code

#                 _temp_what_to_extend = iota_of_output_dim[this_row___o]
#                 list_of_output.extend(_temp_what_to_extend)
#                 ii_row_repeated = torch.empty_like(_temp_what_to_extend)
#                 ii_row_repeated.fill_(ii_row)
#                 list_of_label.extend(ii_row_repeated)
                
#                 #tail 
#                 flag__if_this_row_has_something___label[ii_row] = False
#                 pass#while true
#             #<  assert

#             assert _tensor_equal(list_of_label .get_useful(), [1,1,3,4])
#             assert _tensor_equal(list_of_output.get_useful(), [0,5,6,2])
#             _temp_assert___set_index_to_false___lable_o = flag__the_2d_bool_tensor___label_o.detach().clone()
#             _temp_assert___set_index_to_false___lable_o[list_of_label.get_useful(), list_of_output.get_useful()] = False
#             assert _temp_assert___set_index_to_false___lable_o.any() == False

#             pass#/ test

#         if "combine of previous 2 tests with the new class" and True:
#             output_cont = DNN_label_container_2026(torch.tensor([    
#                                 [1, 2, 3, 4, 5, ],
#                                 [0, 0, 0, 0, 0, ],]))
#             output___b_o = torch.tensor([    
#                     [2, 2, 5, 9, 5, 2, 7, ],
#                     [0, 1, 0, 0, 1, 0, 0, ],])
#             list_of_label, list_of_output = output_cont.detect_perfect_output___all_to_all(output___b_o)

#             assert _tensor_equal(list_of_label .data, [1,1,4])
#             assert _tensor_equal(list_of_output.data, [0,5,2])
#             assert _tensor_equal(output_cont .data[:,1], output___b_o[:,0])
#             assert _tensor_equal(output_cont .data[:,1], output___b_o[:,5])
#             assert _tensor_equal(output_cont .data[:,4], output___b_o[:,2])
#             assert _tensor_equal(output___b_o[:,0],      output___b_o[:,5])


#             output_cont = DNN_label_container_2026(torch.tensor([    
#                                 [1, 2, 3, 4, 5, ],
#                                 [0, 0, 0, 0, 0, ],]))
#             output___b_o = torch.tensor([    
#                     [2, 2, 5, 9, 5, 2, 4, ],
#                     [0, 1, 0, 0, 1, 0, 0, ],])
#             list_of_label, list_of_output = output_cont.detect_perfect_output___all_to_all(output___b_o)
            
#             assert _tensor_equal(list_of_label .data, [1,1,3,4])
#             assert _tensor_equal(list_of_output.data, [0,5,6,2])
#             pass#/ test

#         if "detect perfect output        all to all" and True:

#             batch = 4
#             #<  data 
#             output_cont = DNN_label_container_2026(torch.tensor([    
#                                 [0, 0, 0, 0, 0, 0, 0, 0, ],
#                                 [0, 0, 0, 0, 1, 1, 1, 1, ],
#                                 [0, 0, 1, 1, 0, 0, 1, 1, ],
#                                 [0, 1, 0, 1, 0, 1, 0, 1, ],]))
#             output___b_o = torch.tensor([    
#                                 [1, 0, 1, 0, 1],
#                                 [0, 1, 1, 1, 1],
#                                 [1, 0, 1, 0, 1],
#                                 [1, 0, 1, 1, 1],])
#             #<  calc
#             list_of_label, list_of_output = output_cont.detect_perfect_output___all_to_all(output___b_o)
#             #<  assert
#             assert _tensor_equal(list_of_label .data, [4,5])
#             assert _tensor_equal(list_of_output.data, [1,3])
#             pass#/ test

#         return 
#     ____no_plan_for_now()
#     pass












# old code      maybe still useful.
    # def detect_perfect_output___by_position(self, the_output:torch.Tensor)->torch.Tensor:
    #     '''return is the suggestion of which to remove.'''
    #     self_data = self.get_useful()
    #     assert self_data.shape == the_output.shape
    #     #<  calc
    #     flag_eq__before_all___b_o = self_data.eq(the_output)
    #     flag_eq___o = flag_eq__before_all___b_o.all(dim=0)
    #     del flag_eq__before_all___b_o
    #     assert flag_eq___o.shape == torch.Size([the_output.shape[1]])#debug code
    #     assert False, "untested"
    #     return flag_eq___o
    # def detect_perfect_output___all_to_all(self, the_output:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    #     '''return list_of_label, list_of_output
        
    #     return is the suggestion of which to remove.'''

    #     batch = self.batch()
    #     label_dim = self.get_size()
    #     out_dim = the_output.shape[1]
    #     #<  data 
    #     label___b_label = self.data
    #     assert label___b_label.shape == torch.Size([batch, label_dim])#debug code.
    #     output___b_o = the_output
    #     assert output___b_o.shape == torch.Size([batch, out_dim])#debug code.
    #     #<  calc step 1,     2 datasets to bool matrix.

    #     #host is 111222333, or 1122
    #     HOST__label___T___label_EXPANDo_b = label___b_label.T \
    #             .reshape([label___b_label.shape[1], 1, label___b_label.shape[0]]) \
    #             .expand([-1, output___b_o.shape[1], -1])
    #     assert HOST__label___T___label_EXPANDo_b.shape == torch.Size([label_dim, out_dim, batch])#debug code.
    #     #guest is 123123123, or 1212
    #     GUEST__output___T___EXPANDlabel_o_b = output___b_o.T \
    #             .reshape([1, output___b_o.shape[1],  output___b_o.shape[0]]) \
    #             .expand([label___b_label.shape[1], -1, -1])
    #     assert GUEST__output___T___EXPANDlabel_o_b.shape == torch.Size([label_dim, out_dim, batch])#debug code.

    #     flag_eq__before_all___label_o_b = HOST__label___T___label_EXPANDo_b.eq(GUEST__output___T___EXPANDlabel_o_b)

    #     flag_eq___label_o = flag_eq__before_all___label_o_b.all(dim=2)
    #     assert flag_eq___label_o.shape == torch.Size([label_dim, out_dim])#debug code.
    #     assert flag_eq___label_o.dtype == torch.bool#debug code.
    #     #<  calc step 2,     2d bool to index list.
    #     list_of_label  = _only_for_output_container_to_use____DNN_container_2026()
    #     list_of_output = _only_for_output_container_to_use____DNN_container_2026()
        
    #     iota_of_output_dim = iota(out_dim)
        
    #     flag__if_this_row_has_something___label = flag_eq___label_o.any(dim=1)
    #     assert flag__if_this_row_has_something___label.shape == torch.Size([label_dim])
    #     while True:
    #         if not flag__if_this_row_has_something___label.any():
    #             break
    #         #loop body
    #         flag_in_int___if_this_row_has_something___label = flag__if_this_row_has_something___label.to(torch.int8)
    #         ii_row = flag_in_int___if_this_row_has_something___label.argmax()
    #         this_row___o = flag_eq___label_o[ii_row]
    #         assert this_row___o.any()#debug code
        
    #         _temp_what_to_extend = iota_of_output_dim[this_row___o]
    #         list_of_output.extend(_temp_what_to_extend)
    #         ii_row_repeated = torch.empty_like(_temp_what_to_extend)
    #         ii_row_repeated.fill_(ii_row)
    #         list_of_label.extend(ii_row_repeated)
        
    #         #tail 
    #         flag__if_this_row_has_something___label[ii_row] = False
    #         pass#while true
    #     return list_of_label.get_useful(), list_of_output.get_useful()
    #     #end of function.

    # def __repr__(self):
    #     return f"{self.data.__repr__()}, size:{self._size}, DNN output container 2026"
    # def __str__(self):
    #     return f"{self.data.__str__() }, size:{self._size}, DNN output container 2026"
    # pass#end of class




