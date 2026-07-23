from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal

import torch

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######



'''新的数字神经网络需要的东西比较多。暂时还没想好。总之不是以前的单体直接一次训完的思路了。

我怂了。'''










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
            #training_target___b_o____or_empty = torch.empty(size=[])  not for now.

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


class DNN_input_container_2026(torch.nn.Module):
    data:torch.nn.parameter.Parameter
    size:int
    _init_to_nan:bool
    def __init__(self, batch:int, 
                dtype:torch.dtype|None = None, device:torch.device|str|None = "cpu", 
                init_capacity = 16, init_to_nan = True):
        super().__init__()
        self.data = torch.nn.Parameter(torch.empty(size=[batch, init_capacity], 
                    dtype=dtype, device=device))
        self.data.requires_grad_(False)
        self.size = 0
        self._init_to_nan = init_to_nan
        if init_to_nan:
            self.data.fill_(torch.nan)
            pass
        pass
    def batch(self)->int:
        '''get'''
        return self.data.shape[0]
    def capacity(self)->int:
        '''get'''
        return self.data.shape[1]
    def size(self)->int:
        '''get'''
        return self.size
    def squeeze(self):
        assert False, "unfinished"

    def __calc_bigger_capacity(self)->int:
        return int(self.capacity()*2)

    def extend(self, other:torch.Tensor)->None:
        assert other.shape.__len__() == 2
        assert other.shape[0] == self.data.shape[0]
        with torch.no_grad():
                
            _temp__how_many_to_add = other.shape[1]
            _size_after = self.size + _temp__how_many_to_add
            if _size_after > self.capacity():# get a bigger new capacity first.
                _temp___new_capacity = self.__calc_bigger_capacity()
                _temp___new_container = torch.empty(size=[self.batch(), _temp___new_capacity])
                if self._init_to_nan:
                    _temp___new_container.fill_(torch.nan)
                    pass
                _temp___new_container[:, 0:self.size] = self.get_useful()
                self.data.data = _temp___new_container
                pass

            self.data[:, self.size:self.size + _temp__how_many_to_add] = other
            self.size = _size_after
            return
        pass#end of function

    def get_useful(self)->torch.Tensor:
        result = self.data[:,:self.size]
        return result

    pass

if "how to add input." and __DEBUG_ME__() and True:
    def ____input_container_idea____():
        if "basic idea" and True:
            batch = 2
            #<  the container
            the_container = DNN_input_container_2026(batch=batch,init_capacity=6, )
            assert the_container.size == 0
            assert the_container.capacity() == 6
            assert the_container._init_to_nan == True
            the_container.extend(torch.tensor([ [ 11,  22,  33],
                                                [111, 122, 133],]))
            assert the_container.size == 3
            assert the_container.capacity() == 6
            the_container.extend(torch.tensor([ [ 77,  88],
                                                [177, 188],]))
            assert the_container.size == 5
            assert the_container.capacity() == 6
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 
                                                [ 11,  22,  33,  77,  88],
                                                [111, 122, 133, 177, 188],]))
            assert torch.isnan(the_container.data[:,5]).all()

            the_container.extend(torch.tensor([ [ 111,  222],
                                                [1111, 1222],]))
            assert the_container.size == 7
            assert the_container.capacity() == 12
            assert _tensor_equal(the_container.get_useful(), torch.tensor([ 
                                                [ 11,  22,  33,  77,  88,  111,  222],
                                                [111, 122, 133, 177, 188, 1111, 1222],]))
            assert torch.isnan(the_container.data[:,7:]).all()

            pass

        if "device adaption" and True:
            the_container = DNN_input_container_2026(batch=2,init_capacity=6 )
            assert the_container.data.device.type == "cpu"
            the_container.cuda()
            assert the_container.data.device.type == "cuda"
            the_container = DNN_input_container_2026(batch=2,init_capacity=6, device="cuda")
            assert the_container.data.device.type == "cuda"
            the_container.cpu()
            assert the_container.data.device.type == "cpu"
            pass


        return
    ____input_container_idea____()



# 改层的形状，删结果和加输入
# 1是缓冲区的行为，更新当中的更新动力学？？？可能单独写一个gramo？或者单独的优化器？
# 一个整体的容器
#trace back 需要容器的支持。 从整体的class里面得到新的输入数据。
# 重新做干堆测试。

if "delete output" and __DEBUG_ME__() and True:
    def ____delete_output____():
        if "delete output" and True:

            batch = 2
            out_dim = 5
            in_dim = 3
            #<  the answer
            keep_these_output = torch.tensor([1,1,1,0,1], dtype=torch.bool)
            new_out_dim = int(keep_these_output.sum().item())
            #assert isinstance(new_out_dim, int)

            #<  dataset
            input___b_i = torch.tensor([[1.,  1.,  1.],
                                        [1.,  1.,  1.],])
            label___b_o = torch.tensor([[1.,  1.,  1.,  1.,  1.],
                                        [1.,  1.,  1.,  1.,  1.],])
            #<  model param
            ori___training_buffer___o_i = torch.tensor([  
                                                    [0.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3],
                                                    [0.1, 0.2, 0.3],
                                                    [0.1, 1.2, 0.3],
                                                    [1.1, 0.2, 0.3],])#32321
            #<  original    forward path
            _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
            ori___output___b_o = input___b_i[:, _temp_one_hot___o]
            del _temp_one_hot___o
            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
            #<  the new shape
            new___training_buffer___o_i = ori___training_buffer___o_i[keep_these_output,:]
            assert new___training_buffer___o_i.shape == torch.Size([new_out_dim, in_dim])
            _temp___new____one_hot___o = new___training_buffer___o_i.argmax(dim=1)
            new___output___b_o = input___b_i[:, _temp___new____one_hot___o]
            del _temp___new____one_hot___o
            #<  assert 
            assert _tensor_equal(new___output___b_o, ori___output___b_o[:, keep_these_output])

            pass#/ test

        if "delete output" and True:
            for batch in [2,5,10]:
                for out_dim in [3,7,11]:
                    for in_dim in [6,9,13]:
                        for _ in range(5):
                            #<  the answer
                            keep_these_output = torch.rand(size=[out_dim])
                            keep_these_output = keep_these_output.gt(0.5)

                            new_out_dim = int(keep_these_output.sum().item())
                            #assert isinstance(new_out_dim, int)

                            #<  dataset
                            input___b_i = torch.randn(size=[batch, in_dim])
                            label___b_o = torch.randn(size=[batch, out_dim])
                            #<  model param
                            ori___training_buffer___o_i = torch.randn(size=[out_dim, in_dim])
                            #<  original    forward path
                            _temp_one_hot___o = ori___training_buffer___o_i.argmax(dim=1)
                            ori___output___b_o = input___b_i[:, _temp_one_hot___o]
                            del _temp_one_hot___o
                            assert ori___output___b_o.shape == torch.Size([batch, out_dim])
                            #<  the new shape
                            new___training_buffer___o_i = ori___training_buffer___o_i[keep_these_output,:]
                            assert new___training_buffer___o_i.shape == torch.Size([new_out_dim, in_dim])
                            _temp___new____one_hot___o = new___training_buffer___o_i.argmax(dim=1)
                            new___output___b_o = input___b_i[:, _temp___new____one_hot___o]
                            del _temp___new____one_hot___o
                            #<  assert 
                            assert _tensor_equal(new___output___b_o, ori___output___b_o[:, keep_these_output])
                            pass#for _
                        pass#for in_dim
                    pass#for out_dim
                pass#for batch
            pass#/ test

        return 
    ____delete_output____()
















class DigitalMapper_full__2026(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, \
                    device=None, dtype=None) -> None:  
        
        #this dtype is only for a inner memory in training. It must be float point number.
        assert dtype in [torch.float, torch.float16, torch.float32, torch.float64, torch.bfloat16]

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if in_features<2:
            raise Exception("emmmm")

        self.in_features = in_features
        self.out_features = out_features
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        #self.__reset_parameters__the_plain_rand01_style()
        #self.grad_protection 1w 继续

        pass

    # def __reset_parameters__the_plain_rand01_style(self) -> None:
    #     '''copied from torch.nn.Linear'''
    #     self.raw_weight.data = torch.rand_like(self.raw_weight)*-1.# they should be <0.
    #     pass
    
    if "idk if it's still useful" and False:

        def get_one_hot_format(self)->torch.Tensor:
            with torch.no_grad():
                #raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
                out_features_s = self.raw_weight.shape[0]
                out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
                #print(out_features_iota, "out_features_iota")
                index_of_max_o = self.raw_weight.max(dim=1).indices
                #print(index_of_max_o, "index_of_max_o")

                one_hot_o_i = torch.zeros_like(self.raw_weight)
                one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
                return one_hot_o_i

        def debug_get_zero_grad_ratio(self, directly_print_out:float = False)->float:
            with torch.no_grad():
                result = 0.
                if not self.raw_weight.grad is None:
                    flags = self.raw_weight.grad.eq(0.)
                    total_amount = flags.sum().item()
                    result = float(total_amount)/self.raw_weight.nelement()
                if directly_print_out:
                    print("get_zero_grad_ratio:", result)
                return result


        def debug_strong_grad_ratio(self, log10_diff = -2., epi_for_w = 0.01, epi_for_g = 0.01, \
                                    print_out = False)->float:
            #epi_for_w/epi_for_g<math.pow(10, log10_diff)*0.999??????
            if self.raw_weight.grad is None:
                if print_out:
                    print(0., "inside debug_micro_grad_ratio function __line 1082")
                    pass
                return 0.

            the_device=self.raw_weight.device
            epi_for_w_tensor = torch.tensor([epi_for_w], device=the_device)
            raw_weight_abs = self.raw_weight.abs()
            flag_w_big_enough = raw_weight_abs.gt(epi_for_w_tensor)

            epi_for_g_tensor = torch.tensor([epi_for_g], device=the_device)
            raw_weight_grad_abs = self.raw_weight.grad.abs()
            flag_g_big_enough = raw_weight_grad_abs.gt(epi_for_g_tensor)

            ten = torch.tensor([10.], device=the_device)
            log10_diff_tensor = torch.tensor([log10_diff], device=the_device)
            corresponding_g = raw_weight_grad_abs*torch.pow(ten, log10_diff_tensor)
            flag_w_lt_corresponding_g = raw_weight_abs.lt(corresponding_g)

            flag_useful_g = flag_w_big_enough.logical_and(flag_g_big_enough).logical_and(flag_w_lt_corresponding_g)
            result = (flag_useful_g.sum().to(torch.float32)/self.raw_weight.nelement()).item()
            if print_out:
                print(result, "inside debug_micro_grad_ratio function __line 1082")
                pass
            return result
        def debug_print_param_overlap_ratio(self):
            with torch.no_grad():
                the_max_index = self.get_index_format()
                the_dtype = torch.int32
                if self.out_features<=1:
                    print("Too few output, The overlapping ratio doesn't mean anything. __line__903")
                else:
                    total_overlap_count = 0
                    total_possible_count = self.in_features*(self.in_features-1)//2
                    for i in range(self.in_features-1):
                        host_index = torch.tensor([i], dtype=the_dtype)
                        guest_index = torch.linspace(i+1, self.in_features-1,
                                                self.in_features-i-1, dtype=the_dtype)
                        flag_overlapped = the_max_index[guest_index].eq(the_max_index[host_index])
                        #print(host_index, guest_index, flag_first_input_eq, flag_second_input_eq,flag_overlapped)
                        total_overlap_count += int(flag_overlapped.sum().item())
                        pass
                    overlap_ratio = float(total_overlap_count)/total_possible_count
                    print("overlap_ratio:",
                            f'{overlap_ratio:.4f}',", ", total_overlap_count,
                            "/", total_possible_count)
                    pass#if self.SIG_gate_count>0:
                pass
            return
        pass



    def forward(self, input:torch.Tensor)->torch.Tensor:
        
        raise Exception("unfinished")
    #end of function.


    def get_max_index(self)->torch.Tensor:
        with torch.no_grad():
            the_max_index = self.raw_weight.max(dim=1,keepdim=False).indices
            return the_max_index


    def extra_repr(self) -> str:
        return f'Output is standard binary range. In_features={self.in_features}, out_features={self.out_features}'



    pass





