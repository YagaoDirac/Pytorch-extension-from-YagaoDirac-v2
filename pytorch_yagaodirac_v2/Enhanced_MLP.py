from typing import Any, List, Tuple, Optional, Self
import math
import torch

if "__main__" == __name__:
    import sys
    __temp__package_path__:str = sys.path[0]
    pos = __temp__package_path__.rfind("\\")
    ____package_path__ = __temp__package_path__[:pos]
    sys.path.insert(0, ____package_path__)
    #print("adding sys path:", ____package_path__)
    pass
from pytorch_yagaodirac_v2.ParamMo import GradientModification_v2, ReLU_with_offset
from pytorch_yagaodirac_v2.Util import debug_avg_log, data_gen_from_random_teacher, Print_Timing



class FCL_from_yagaodirac(torch.nn.Module):
    r''' The way to use this class is the same as to use torch.nn.Linear.
    
    You can choose from ReLU and my ReLU_one.
    
    to do: The out gramo and the bias gramo basically does the same thing.
    Now it's for simplicity.
    To get a better performance, remove the bias gramo. 
    But the bias accumulates grad based on batch. And it needs compensition against it. 
    The way is to set the scale factor of out gramo based on batch size in forward func.'''
    def __init__(self, in_features: int, out_features: int, \
                        bias: bool = True, \
                            #expect_avg_log:Optional[float] = None,\
                            device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_o_i = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias_o = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias_o', None)
        self.__reset_parameters()

        # if expect_avg_log is None:
        
        #self.param_avg_log_expectation = math.log10(self.in_features)*-0.5-0.43
        #print(self.param_avg_log_expectation, "param_avg_log_expectation")
        
        #self.grad_avg_log_expectation = math.log10(self.out_features)*-0.5+0.77
        #self.grad_avg_log_expectation = 123123
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        
        #print(self.grad_avg_log_expectation, "grad_avg_log_expectation")
        # +0.77 when batch is 1000. in_features is 100, out_features is 100.
        # batch incr 10x, this +0.03
        # in_features incr 10x, this -0.04
        # out_features doesn't affect this.
        
        # self.gramo_for_weight_extra_factor = math.pow(10., self.param_avg_log_expectation-self.grad_avg_log_expectation)
        # self.drag_avg_log_of_scaling_factor_for_out_gramo_to = torch.nn.Parameter(torch.tensor([self.param_avg_log_expectation]), requires_grad=False)
        
        self.gramo_for_weight = GradientModification_v2()
        self.gramo_for_bias = GradientModification_v2()
        self.out_gramo = GradientModification_v2()
        #self.reset_scaling_factor_for_weight()
        pass
    #end of function.

    def __reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight_o_i, a=math.sqrt(5))
        if self.bias_o is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_o_i)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_o, -bound, bound)
            pass
        pass
    #end of function.
    
    def __XXXXXXXXXXXX__reset_scaling_factor_for_weight(self):
        '''simply sets the inner'''
        sqrt_out_features = math.sqrt(self.out_features)
        self.gramo_for_weight.set_scaling_factor(math.pow(10., -0.2)*sqrt_out_features)
        
        #temp2 = temp.sqrt()
        #temp3 = temp2*10.#2.
        #temp4 = temp3/self.out_gramo.scaling_factor
        #self.gramo_for_weight.set_scaling_factor(temp3.item())
        #self.gramo_for_weight.set_scaling_factor(temp4.item())
        pass
    def scale_the_scaling_factor_for_weight(self, by:float):
        '''simply sets the inner'''
        self.gramo_for_weight.scale_scaling_factor(by)
        pass
    
    def __XXXXXXXXX__set_scaling_factor_to_adapt_kaiming_he_init(self):
        #in_features, out_features, avg_log weight, avg_log bias
        #100        100             -1.4            -1.4
        #10000      100             -2.4            -2.4
        #100        10000           -1.4            -1.4
        self.gramo_for_weight.set_scaling_factor(0.37/math.sqrt(self.in_features))
        self.gramo_for_bias.set_scaling_factor(0.37/math.sqrt(self.in_features))
        #self.gramo_for_bias.set_scaling_factor(0.037)
        pass
    
    def set_scaling_factor_for_weight(self, scaling_factor:float):
        '''simply sets the inner'''
        self.gramo_for_weight.set_scaling_factor(scaling_factor)
        pass

    def forward(self, input_b_i:torch.Tensor)->torch.Tensor:
        w_after_gramo_o_i:torch.Tensor = self.gramo_for_weight(self.weight_o_i.view(1, -1)).view(self.out_features, self.in_features)
        b_after_gramo_o:torch.Tensor = self.gramo_for_bias(self.bias_o.view(1, -1)).view(self.out_features)
        x_after_linear_b_o = torch.nn.functional.linear(input_b_i,w_after_gramo_o_i,b_after_gramo_o)
        result_b_o = self.out_gramo(x_after_linear_b_o)
        return result_b_o
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_o is not None}'

    def debug_get_all_avg_log(self)->Tuple[List[float], str]:
        result:List[float] = []
        result.append(debug_avg_log(self.weight_o_i))
        result.append(debug_avg_log(self.bias_o))
        docs_str = "weight_o_i, bias_o"
        if not self.weight_o_i.grad is None:
            result.append(debug_avg_log(self.weight_o_i.grad))
            docs_str+=", weight_o_i.grad"
            pass
        if not self.bias_o.grad is None:
            result.append(debug_avg_log(self.bias_o.grad))
            docs_str+=", bias_o.grad"
            pass
        return (result, docs_str)

    def convert_to_plain_fcnn(self)->torch.nn.Module:
        has_bias = False
        if not self.bias_o is None:
            has_bias = True
            pass
        result = torch.nn.Linear(self.in_features, self.out_features,has_bias)
        result.weight.data = self.weight_o_i.data.detach().clone()
        if has_bias:
            result.bias.data = self.bias_o.data.detach().clone()
            pass
        return result

    pass
    

if '''basic avg log10 test.(with set numbers)''' and False:
    batch = 1
    in_features = 1
    out_features = 10000
    print(batch, "batch", in_features, "in_features", out_features, "out_features")
    layer = FCL_from_yagaodirac(in_features, out_features, True)
    layer.weight_o_i = torch.nn.Parameter(torch.ones_like(layer.weight_o_i))
    layer.bias_o = torch.nn.Parameter(torch.ones_like(layer.bias_o))
    print(layer.debug_get_all_avg_log())
    input = torch.ones([batch, in_features])
    pred:torch.Tensor = layer(input)
    pred.backward(gradient=torch.ones([batch, out_features]))
    print(layer.debug_get_all_avg_log())
    print(layer.weight_o_i[:2, :7].abs().log10())
    print(layer.bias_o[:7].abs().log10())
    print(layer.weight_o_i.grad[:4, :7].abs().log10())
    print(layer.bias_o.grad[:7].abs().log10())
    #batch, in_features, out_features don't affect the output. they are all 1(log1 is 0.)
    pass

if 'kaiming_he_init avg log test.(with set numbers)' and False:
    in_features = 100
    out_features = 10000
    print(in_features, "in_features", out_features, "out_features")
    layer = FCL_from_yagaodirac(in_features, out_features, True)
    print(layer.debug_get_all_avg_log())
    pass
    #100   100 -1.4 -1.4
    #10000 100 -2.4 -2.4
    #100 10000 -1.4 -1.4

if 'kaiming_he_init adaption test.(with set numbers)' and False:
    batch = 100
    in_features = 100
    out_features = 10000
    print(batch, "batch", in_features, "in_features", out_features, "out_features")
    layer = FCL_from_yagaodirac(in_features, out_features, True)
    layer.set_scaling_factor_to_adapt_kaiming_he_init()
    #layer.weight_o_i = torch.nn.Parameter(torch.ones_like(layer.weight_o_i))
    #layer.bias_o = torch.nn.Parameter(torch.ones_like(layer.bias_o))
    print(layer.debug_get_all_avg_log())
    input = torch.ones([batch, in_features])
    pred:torch.Tensor = layer(input)
    pred.backward(gradient=torch.ones([batch, out_features]))
    print(layer.debug_get_all_avg_log())
    print(layer.weight_o_i[:2, :7].abs().log10())
    print(layer.bias_o[:7].abs().log10())
    print(layer.weight_o_i.grad[:4, :7].abs().log10())
    print(layer.bias_o.grad[:7].abs().log10())
    #batch, in_features, out_features don't affect the output. they are all 1(log1 is 0.)
    pass

if 'kaiming_he_init adaption test.' and False:
    batch = 100
    in_features = 100
    out_features = 100
    print(batch, "batch", in_features, "in_features", out_features, "out_features")
    layer = FCL_from_yagaodirac(in_features, out_features, True)
    layer.set_scaling_factor_to_adapt_kaiming_he_init()
    print(layer.debug_get_all_avg_log())
    input = torch.rand([batch, in_features])+0.01#, requires_grad=True)
    #target = torch.rand([batch, out_features])+0.01
    pred:torch.Tensor = layer(input)
    #loss_function = torch.nn.MSELoss()
    #loss = loss_function(pred, target)
    #loss.backward()
    pred.backward(gradient=torch.rand([batch, out_features])+0.01)
    print(layer.debug_get_all_avg_log())
    print(layer.weight_o_i[:2, :7].abs().log10())
    print(layer.bias_o[:7].abs().log10())
    print(layer.weight_o_i.grad[:4, :7].abs().log10())
    print(layer.bias_o.grad[:7].abs().log10())
    # print(debug_avg_log(layer.weight_o_i), "weight_o_i") the first 3 are already aligned.
    # print(debug_avg_log(layer.bias_o), "bias_o")
    # print(debug_avg_log(layer.weight_o_i.grad), "weight_o_i.grad")
    #print(debug_avg_log(layer.bias_o.grad), "bias_o.grad")
    #print(debug_avg_log(input), "input")
    fds=432

if 'single layer random teacher test' and False:
    batch = 100
    in_features = 200
    #mid_width = 50
    out_features = 100
    teacher = torch.nn.Linear(in_features, out_features, True).eval()
    student = FCL_from_yagaodirac(in_features, out_features, True)
    #student.set_scaling_factor_to_adapt_kaiming_he_init()
    input = torch.rand([batch, in_features])+0.01
    target = data_gen_from_random_teacher(teacher,input)
    loss_function = torch.nn.MSELoss()
    optim = torch.optim.SGD(student.parameters(), lr=0.0001)
    last_loss = 999999
    for epoch in range(999999):
        pred = student(input)
        loss:torch.Tensor = loss_function(pred, target)
        if epoch < 3:
            print(loss, "  loss")
            pass
        if last_loss<loss.item():#+0.0001:
            print(loss, "  loss / epoch  ", epoch)
            break
        last_loss = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pass
    pass

if 'dry stack test.(withOUT activition func)' and False:
    batch = 10000
    in_features = 50
    mid_width = 100
    num_layers = 100
    out_features = 10
    class MLP_dry_stack_teacher(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, 
                     bias: bool, num_layers, mid_width, \
                    device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.num_layers = num_layers
            
            self.layers = torch.nn.ParameterList()
            if 1 == num_layers:
                self.layers.append(torch.nn.Linear(in_features, out_features,bias))
            else:
                self.layers.append(torch.nn.Linear(in_features, mid_width,bias))
                for _ in range(num_layers-2):
                    self.layers.append(torch.nn.Linear(mid_width, mid_width,bias))
                    pass
                self.layers.append(torch.nn.Linear(mid_width, out_features, bias))
                pass
            pass 
        #end of function
        def forward(self, input_b_i:torch.Tensor) -> torch.Tensor:
            x = input_b_i
            for i in range(self.num_layers):
                layer:torch.nn.Linear = self.layers[i]
                x = layer(x)
                pass
            return x
        #end of function
        pass
    teacher = MLP_dry_stack_teacher(in_features, out_features, True, num_layers, mid_width).eval()

    class MLP_dry_stack_student(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, 
                     bias: bool, num_layers, mid_width, \
                    device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.num_layers = num_layers
            
            self.layers = torch.nn.ParameterList()
            if 1 == num_layers:
                self.layers.append(FCL_from_yagaodirac(in_features, out_features,bias))
            else:
                self.layers.append(FCL_from_yagaodirac(in_features, mid_width,bias))
                for _ in range(num_layers-2):
                    self.layers.append(FCL_from_yagaodirac(mid_width, mid_width,bias))
                    pass
                self.layers.append(FCL_from_yagaodirac(mid_width, out_features, bias))
                pass
            pass 
        #end of function
        def forward(self, input_b_i:torch.Tensor) -> torch.Tensor:
            x = input_b_i
            for i in range(self.num_layers):
                layer:FCL_from_yagaodirac = self.layers[i]
                x = layer(x)
                pass
            return x
        #end of function
        pass
    student = MLP_dry_stack_student(in_features, out_features, True, num_layers, mid_width)
    if 'adapt to kaiming he init' and False:
        layer:FCL_from_yagaodirac
        for layer in student.layers:
            layer.__XXXXXXXXX__set_scaling_factor_to_adapt_kaiming_he_init()
            pass
        pass
    input = torch.rand([batch, in_features])+0.01
    target = data_gen_from_random_teacher(teacher,input)
    loss_function = torch.nn.MSELoss()
    optim = torch.optim.SGD(student.parameters(), lr=0.000001)
    last_loss = 999999
    
    input = input.cuda()
    target = target.cuda()
    student.cuda()
    for epoch in range(999999):
        pred = student(input)
        loss:torch.Tensor = loss_function(pred, target)
        if epoch ==0 or epoch%100 == 0:
            print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
            pass
        if last_loss<loss.item():#+0.0001:
            print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
            break
        last_loss = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pass
    pass





class MLP_from_yagaodirac(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_width: int, \
                bias = True,num_layers = 2, relu_offset = 0.,
                device=None, dtype=None) -> None:
        # the default value of relu offset may need to be 0.05 or something.
        # but anyway, the new relu with offset doesn't seem helpful.
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = torch.nn.ParameterList()
        if 1 == num_layers:
            self.layers.append(FCL_from_yagaodirac(in_features, out_features,bias))
        else:
            self.layers.append(FCL_from_yagaodirac(in_features, mid_width,bias))
            for _ in range(num_layers-2):
                self.layers.append(FCL_from_yagaodirac(mid_width, mid_width,bias))
                pass
            self.layers.append(FCL_from_yagaodirac(mid_width, out_features, bias))
            pass
        
        self.some_relus = torch.nn.ParameterList([ReLU_with_offset(relu_offset) for _ in range(num_layers-1)])
        
        pass 
    #end of function
    def forward(self, input_b_i:torch.Tensor) -> torch.Tensor:
        x = input_b_i
        
        for i in range(self.num_layers -1):
            layer:FCL_from_yagaodirac = self.layers[i]
            x = layer(x)
            the_relu = self.some_relus[i]
            x = the_relu(x)
            pass
        
        last_fcl = self.layers[-1]
        x = last_fcl(x)
        return x
    #end of function
    pass

if 'dry stack test.(relu offset)' and False:
    # batch = 1000, in_features = 50, mid_width = 100, num_layers = 2, out_features = 10, relu_offset = ?
    #lr=0.000001(k,e-)
    # test the relu offset
    #            ing0.01(50k,3.3e-4), ?0.03(47k,3e-4), 0.05(23~49k,1e-3~3e-4), 0.07(20~60k 1.1e-3~2.1e-4), 0.1(22k 1.3e-3, 3times),  0.3(13k 1.6e-3, 3times),  1.1(8k 3.2e-4, 2times), 11.1(9k 6e-7)
    #negetive.  -0.01(50k 3.2e-4), -0.1(51k 2.8e-4, 2times), -0.3(35k 2.5e-4, 2times), -1.1(37k 1.5e-6, 2times), -11.1(34k 1.2e-6)
    
    # batch = 1000, in_features = 50, mid_width = 100, num_layers = 3!!!!!!!!, out_features = 10, relu_offset = ?
    #lr=0.00001(k,e-)
    # test the relu offset. all 3times
    # 0.001(2k6 1.2e-4), 0.01(2k 1.3e-4) 0.1(3k 8e-5), 0.3(1k4 3e-5), 1.1(1k 9e-6), 
    
    
    # batch = 1000, in_features = 500, mid_width = 1000, num_layers = 4!!!!!!!!, out_features = 100, relu_offset = ?
    #lr=0.00001(k,e-)
    # test the relu offset. all 3times
    # 0.0001 and 0.001(200 1e-4), 0.01(200 8e-5), 
    
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    # maybe it's still not very clear????????????
    
    relu_offset = -0.1
    print(relu_offset)
    
    lr = 0.00001
    batch = 1000
    in_features = 500
    mid_width = 1000
    num_layers = 4
    out_features = 100
    teacher_relu_offset = relu_offset
    print_timing = Print_Timing(max_gap=5000,density=0.5,first=1)
    
    class MLP_dry_stack_teacher_with_relu_offset(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, 
                     bias: bool, num_layers, mid_width, relu_offset,\
                    device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.num_layers = num_layers
            
            self.layers = torch.nn.ParameterList()
            if 1 == num_layers:
                self.layers.append(torch.nn.Linear(in_features, out_features,bias))
            else:
                self.layers.append(torch.nn.Linear(in_features, mid_width,bias))
                for _ in range(num_layers-2):
                    self.layers.append(torch.nn.Linear(mid_width, mid_width,bias))
                    pass
                self.layers.append(torch.nn.Linear(mid_width, out_features, bias))
                pass
            
            self.some_relus = torch.nn.ParameterList([ReLU_with_offset(relu_offset) for _ in range(num_layers-1)])
            
            pass 
        #end of function
        def forward(self, input_b_i:torch.Tensor) -> torch.Tensor:
            x = input_b_i
            
            for i in range(self.num_layers -1):
                layer:torch.nn.Linear = self.layers[i]
                x = layer(x)
                the_relu = self.some_relus[i]
                x = the_relu(x)
                pass
            
            last_fcl = self.layers[-1]
            x = last_fcl(x)
            return x
        #end of function
        pass
    teacher = MLP_dry_stack_teacher_with_relu_offset(in_features, out_features, True, num_layers, mid_width, teacher_relu_offset).eval()
    
    student = MLP_from_yagaodirac(in_features, out_features, True, num_layers, mid_width, relu_offset)
    if 'adapt to kaiming he init' and False:
        layer:FCL_from_yagaodirac
        for layer in student.layers:
            layer.__XXXXXXXXX__set_scaling_factor_to_adapt_kaiming_he_init()
            pass
        pass
    input = torch.rand([batch, in_features])+0.01
    target = data_gen_from_random_teacher(teacher,input)
    loss_function = torch.nn.MSELoss()
    optim = torch.optim.SGD(student.parameters(), lr=lr)
    
    input = input.cuda()
    target = target.cuda()
    student.cuda()
    
    last_loss = 999999
    oscillation_count = 0
    for epoch in range(999999):
        pred = student(input)
        loss:torch.Tensor = loss_function(pred, target)
        if print_timing.check(epoch):
            print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
            pass
        #print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
        
        if last_loss<loss.item():#+0.0001:
            oscillation_count+=1
            if oscillation_count>=20:
                print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
                break
            pass
        last_loss = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pass
    pass

if 'final test' and False:
    # 2 layers. test relu offset
    # 0(32k 2.1e-2), lr e-6
    # lr 5e-6    0(8k 2.5e-2) 0.01(8k 2.5e-2)
    # 3 layers.
    # lr 5e-6    0(6k1 4.2e-2) 0.01(10k 3.4e-2)
    # 5 layers.
    
    relu_offset = 0.
    print(relu_offset)
    lr = 0.00001
    batch = 10000
    in_features = 50
    mid_width = 100
    num_layers = 10
    out_features = 10
    
    input = torch.rand([batch, in_features])+0.01
    target = torch.rand([batch, out_features])+0.01
    
    print_timing = Print_Timing(max_gap=5000,density=0.2,first=1)
    
    model = MLP_from_yagaodirac(in_features, out_features, mid_width, True, num_layers, relu_offset)
    if 'adapt to kaiming he init' and False:
        layer:FCL_from_yagaodirac
        for layer in model.layers:
            layer.__XXXXXXXXX__set_scaling_factor_to_adapt_kaiming_he_init()
            pass
        pass
    
    class The_normal_way(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True, \
                    num_layers = 2, mid_width =Optional[int], \
                    device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.num_layers = num_layers
            
            self.layers = torch.nn.ParameterList()
            if 1 == num_layers:
                self.layers.append(torch.nn.Linear(in_features, out_features,bias))
            else:
                self.layers.append(torch.nn.Linear(in_features, mid_width,bias))
                for _ in range(num_layers-2):
                    self.layers.append(torch.nn.Linear(mid_width, mid_width,bias))
                    pass
                self.layers.append(torch.nn.Linear(mid_width, out_features, bias))
                pass
            pass 
        #end of function
        def forward(self, input_b_i:torch.Tensor) -> torch.Tensor:
            x = input_b_i
            for i in range(self.num_layers):
                layer:torch.nn.Linear = self.layers[i]
                x = layer(x)
                pass
            return x
        #end of function
        pass
    model_ref = The_normal_way(in_features, out_features, True, num_layers, mid_width)
    
    loss_function = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    
    input = input.cuda()
    target = target.cuda()
    model.cuda()
    model_ref.cuda()
    
    last_loss = 999999
    oscillation_count = 0
    for epoch in range(999999):
        pred = model(input)
        loss:torch.Tensor = loss_function(pred, target)
        if print_timing.check(epoch):
            print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
            pass
        #print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
        
        if last_loss<loss.item():#+0.0001:
            oscillation_count+=1
            if oscillation_count>=20:
                print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
                break
            pass
        last_loss = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pass
    
    
    print("----------now the ref----------")
    # this tests the normal way to fit the data with conventional fully connected layers.
    optim = torch.optim.SGD(model_ref.parameters(), lr=lr)
    
    last_loss = 999999
    oscillation_count = 0
    for epoch in range(999999):
        pred = model_ref(input)
        loss:torch.Tensor = loss_function(pred, target)
        if print_timing.check(epoch):
            print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
            pass
        #print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
        
        if last_loss<loss.item():#+0.0001:
            oscillation_count+=1
            if oscillation_count>=20:
                print(f"{loss.item():.3e}", "  loss / epoch  ", epoch)
                break
            pass
        last_loss = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pass
    
    pass

#非假老师测试。



# some old test. 
 
# batch = 2
# in_features = 3
# out_features = 5
# mid_width = 20
# num_layers = 3
# lr = 0.001
# iter_per_print = 1#111
# print_count = 155555
# input = torch.randn([batch, in_features])
# target = torch.randn([batch, out_features])
# model = MLP_from_yagaodirac(in_features, out_features, True, num_layers=num_layers, mid_width=mid_width, relu_offset=0.1)
    
# # if True and "print parameters":
# #     if True and "only the training params":
# #         for p in model.parameters():
# #             if p.requires_grad:
# #                 print(p)
# #                 pass
# #             pass
# #         pass
# #     else:# prints all the params.
# #         for p in model.parameters():
# #             print(p)
# #             pass
# #         pass
    
    
# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# model.cuda().half()
# input = input.cuda().to(torch.float16)
# target = target.cuda().to(torch.float16)

# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     #print(pred, "pred", __line__str())
#     if False and "shape":
#         print(pred.shape, "pred.shape")
#         print(target.shape, "target.shape")
#         fds=423
#     if False and "print pred":
#         if epoch%iter_per_print == iter_per_print-1:
#             print(pred[:5], "pred")
#             print(target[:5], "target")
#             pass
#         pass
#     loss = loss_function(pred, target)
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "print the grad":
#         if epoch%iter_per_print == iter_per_print-1:
#             layer:FCL_from_yagaodirac = model.layers[0]
#             print(layer.debug_get_all_avg_log(), "<<<<<<<   0 layer")
#             layer:FCL_from_yagaodirac = model.layers[1]
#             print(layer.debug_get_all_avg_log(), "<<<<<<<   1 layer")
#             layer:FCL_from_yagaodirac = model.layers[-1]
#             print(layer.debug_get_all_avg_log(), "<<<<<<<   -1 layer")
#             pass
#         pass
#     if True and "print the weight":
#         if epoch%iter_per_print == iter_per_print-1:
#             #layer = model.out_mapper
#             layer = model.first_layer.in_mapper
#             print(layer.raw_weight, "first_layer.in_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "first_layer.in_mapper   after update")
            
#             layer = model.model.second_to_last_layers[0]
#             print(layer.raw_weight, "second_to_last_layers[0]   before update")
#             optimizer.step()
#             print(layer.raw_weight, "second_to_last_layers[0]   after update")
            
#             layer = model.out_mapper
#             print(layer.raw_weight, "out_mapper   before update")
#             optimizer.step()
#             print(layer.raw_weight, "out_mapper   after update")
            
#             pass    
#         pass    
#     if True and "print zero grad ratio":
#         if epoch%iter_per_print == iter_per_print-1:
#             result = model.get_zero_grad_ratio()
#             print("print zero grad ratio: ", result)
#             pass
#         pass
#     if True and "print strong grad ratio":
#         if epoch%iter_per_print == iter_per_print-1:
#             result = model.get_strong_grad_ratio()
#             print("print strong grad ratio: ", result)
#             pass
#         pass
#     #optimizer.param_groups[0]["lr"] = 0.01
#     optimizer.step()
#     if True and "print param overlap":
#         every = 100
#         if epoch%every == every-1:
#             model.print_param_overlap_ratio()
#             pass
#         pass
#     if True and "print acc":
#         if epoch%iter_per_print == iter_per_print-1:
#             with torch.inference_mode():
#                 model.eval()
#                 pred = model(input)
#                 #print(pred, "pred", __line__str())
#                 #print(target, "target")
#                 acc = DigitalMapper_V1_1.bitwise_acc(pred, target)
#                 model.set_acc(acc)
#                 if 1. != acc:
#                     print(epoch+1, "    ep/acc    ", acc)
#                 else:
#                     #print(epoch+1, "    ep/acc    ", acc)
#                     finished = model.can_convert_into_eval_only_mode()
#                     print(finished, "is param hard enough __line 1273")
#                     if finished[0]:
#                         print(pred[:5].T, "pred", __line__str())
#                         print(target[:5].T, "target")
#                         break
#                         pass
#                     pass
#                 pass
#             pass
#         pass

# fds=432




