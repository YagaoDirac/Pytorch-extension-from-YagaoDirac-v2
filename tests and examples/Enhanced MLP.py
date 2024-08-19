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
from pytorch_yagaodirac_v2.ParamMo import GradientModification, ReLU_with_offset
from pytorch_yagaodirac_v2.util import debug_avg_log, data_gen_from_random_teacher

#继续， linear, gmo xmo. 直接堆。

# This is actually test code. But it looks very useful... 
# I should say, don't use this one. A better one is at the end. You'll love it.
class FCL_from_yagaodirac(torch.nn.Module):
    r''' The way to use this class is the same as to use torch.nn.Linear.
    
    You can choose from ReLU and my ReLU_one.'''
    def __init__(self, in_features: int, out_features: int, \
                        #batch_size:int, \
                        bias: bool = True, \
                            expect_avg_log:Optional[float] = None,\
                            device: Any | None = None, dtype: Any | None = None, *args, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.batch_size = torch.nn.Parameter(torch.tensor([batch_size]), requires_grad=False)
        self.weight_o_i = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias_o = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias_o', None)
        self.__reset_parameters()

        # if expect_avg_log is None:
        
        self.param_avg_log_expectation = math.log10(self.in_features)*-0.5-0.43
        #print(self.param_avg_log_expectation, "param_avg_log_expectation")
        
        #self.grad_avg_log_expectation = math.log10(self.out_features)*-0.5+0.77
        self.grad_avg_log_expectation = 123123
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        #继续#继续#继续#继续#继续#继续#继续#继续#继续
        
        print(self.grad_avg_log_expectation, "grad_avg_log_expectation")
        # +0.77 when batch is 1000. in_features is 100, out_features is 100.
        # batch incr 10x, this +0.03
        # in_features incr 10x, this -0.04
        # out_features doesn't affect this.
        
        self.gramo_for_weight_extra_factor = math.pow(10., self.param_avg_log_expectation-self.grad_avg_log_expectation)
        self.drag_avg_log_of_scaling_factor_for_out_gramo_to = torch.nn.Parameter(torch.tensor([self.param_avg_log_expectation]), requires_grad=False)
        
        self.gramo_for_weight = GradientModification()
        self.out_gramo = GradientModification()
        self.reset_scaling_factor_for_weight()
        
        
        
        
        #self.out_gramo.set_scaling_factor(123.45)
        
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
    
    def reset_scaling_factor_for_weight(self):
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
    def set_scaling_factor_for_weight(self, scaling_factor:float):
        '''simply sets the inner'''
        self.gramo_for_weight.set_scaling_factor(scaling_factor)
        pass

    # def set_batch_size(self, batch_size:int):
    #     self.batch_size = torch.nn.Parameter(torch.tensor([batch_size]), requires_grad=False)
    #     temp = self.batch_size.to(torch.float32)
    #     temp2 = 
    #     self.out_gramo.set_scaling_factor()
        
        
        

    def forward(self, input_b_i:torch.Tensor)->torch.Tensor:
        w_after_gramo_o_i:torch.Tensor = self.gramo_for_weight(self.weight_o_i.view(1, -1)).view(self.out_features, self.in_features)
        x_after_linear_b_o = torch.nn.functional.linear(input_b_i,w_after_gramo_o_i,self.bias_o)
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
    
# '''dim check. The forward should be the same as traditional fcnn.'''
# batch = 2
# in_features = 3
# out_features = 5
# layer = FCL_from_yagaodirac(in_features, out_features, True)
# ref_layer = torch.nn.Linear(in_features, out_features, True)
# layer.weight_o_i.data = ref_layer.weight.detach().clone()
# layer.bias_o.data = ref_layer.bias.detach().clone()
# input = torch.randn([batch, in_features])
# pred = layer(input)
# pred_ref = ref_layer(input)
# print(pred, "pred")
# print(pred_ref, "pred_ref")
# fds=432

'''grad strength test.'''

in_features = 100
# b=1000        in
#              10     100     1000    10000  100000
# out  100   0.99     0.5     0.25     0.2      0.2
#     1000    0.5    0.02    -0.22   -0.28              
#    10000      0   -0.46    -0.71   -0.77             
#   log10(out)*-0.5
#
# in=100        batch
#               100     1000   10000   100000
# out  100     -0.2      0.5    1.48    -0.12
#     1000    -0.72     0.02   -1.14    -1.14卡到gramo那个epi了。。。
#

继续。卡到epi了。
batch = 10000
out_features = 1000
layer = FCL_from_yagaodirac(in_features, out_features, True)
input = torch.randn([batch, in_features], requires_grad=True)
target = torch.randn([batch, out_features])
pred:torch.Tensor = layer(input)
loss_function = torch.nn.MSELoss()
loss = loss_function(pred, target)
loss.backward()
print(batch, "batch", in_features, "in_features", out_features, "out_features")
# print(debug_avg_log(layer.weight_o_i), "weight_o_i") the first 3 are already aligned.
# print(debug_avg_log(layer.bias_o), "bias_o")
# print(debug_avg_log(layer.weight_o_i.grad), "weight_o_i.grad")
print(debug_avg_log(layer.bias_o.grad), "bias_o.grad")
#print(debug_avg_log(input), "input")
input
fds=432

# '''single layer random teacher test'''
# batch = 100
# in_features = 20
# mid_width = 50
# out_features = 10
# teacher = torch.nn.Linear(in_features, out_features, True).eval()
# student = FCL_from_yagaodirac(in_features, out_features, True)
# input = torch.randn([batch, in_features])
# target = data_gen_from_random_teacher(teacher,input)
# loss_function = torch.nn.MSELoss()
# optim = torch.optim.SGD(student.parameters(), lr=0.01)
# for _ in range(30):
#     pred = student(input)
#     loss:torch.Tensor = loss_function(pred, target)
#     print(loss, "loss")
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
    
# fds=432




class MLP_from_yagaodirac(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, num_layers = 2, \
        mid_width =Optional[int], relu_offset = 0.1, \
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


 
batch = 2
in_features = 3
out_features = 5
mid_width = 20
num_layers = 3
lr = 0.001
iter_per_print = 1#111
print_count = 155555
input = torch.randn([batch, in_features])
target = torch.randn([batch, out_features])
model = MLP_from_yagaodirac(in_features, out_features, True, num_layers=num_layers, mid_width=mid_width, relu_offset=0.1)
    
# if True and "print parameters":
#     if True and "only the training params":
#         for p in model.parameters():
#             if p.requires_grad:
#                 print(p)
#                 pass
#             pass
#         pass
#     else:# prints all the params.
#         for p in model.parameters():
#             print(p)
#             pass
#         pass
    
    
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model.cuda().half()
input = input.cuda().to(torch.float16)
target = target.cuda().to(torch.float16)

for epoch in range(iter_per_print*print_count):
    model.train()
    pred = model(input)
    #print(pred, "pred", __line__str())
    if False and "shape":
        print(pred.shape, "pred.shape")
        print(target.shape, "target.shape")
        fds=423
    if False and "print pred":
        if epoch%iter_per_print == iter_per_print-1:
            print(pred[:5], "pred")
            print(target[:5], "target")
            pass
        pass
    loss = loss_function(pred, target)
    optimizer.zero_grad()
    loss.backward()
    if True and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            layer:FCL_from_yagaodirac = model.layers[0]
            print(layer.debug_get_all_avg_log(), "<<<<<<<   0 layer")
            layer:FCL_from_yagaodirac = model.layers[1]
            print(layer.debug_get_all_avg_log(), "<<<<<<<   1 layer")
            layer:FCL_from_yagaodirac = model.layers[-1]
            print(layer.debug_get_all_avg_log(), "<<<<<<<   -1 layer")
            pass
        pass
    if True and "print the weight":
        if epoch%iter_per_print == iter_per_print-1:
            #layer = model.out_mapper
            layer = model.first_layer.in_mapper
            print(layer.raw_weight, "first_layer.in_mapper   before update")
            optimizer.step()
            print(layer.raw_weight, "first_layer.in_mapper   after update")
            
            layer = model.model.second_to_last_layers[0]
            print(layer.raw_weight, "second_to_last_layers[0]   before update")
            optimizer.step()
            print(layer.raw_weight, "second_to_last_layers[0]   after update")
            
            layer = model.out_mapper
            print(layer.raw_weight, "out_mapper   before update")
            optimizer.step()
            print(layer.raw_weight, "out_mapper   after update")
            
            pass    
        pass    
    if True and "print zero grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            result = model.get_zero_grad_ratio()
            print("print zero grad ratio: ", result)
            pass
        pass
    if True and "print strong grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            result = model.get_strong_grad_ratio()
            print("print strong grad ratio: ", result)
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    optimizer.step()
    if True and "print param overlap":
        every = 100
        if epoch%every == every-1:
            model.print_param_overlap_ratio()
            pass
        pass
    if True and "print acc":
        if epoch%iter_per_print == iter_per_print-1:
            with torch.inference_mode():
                model.eval()
                pred = model(input)
                #print(pred, "pred", __line__str())
                #print(target, "target")
                acc = DigitalMapper_V1_1.bitwise_acc(pred, target)
                model.set_acc(acc)
                if 1. != acc:
                    print(epoch+1, "    ep/acc    ", acc)
                else:
                    #print(epoch+1, "    ep/acc    ", acc)
                    finished = model.can_convert_into_eval_only_mode()
                    print(finished, "is param hard enough __line 1273")
                    if finished[0]:
                        print(pred[:5].T, "pred", __line__str())
                        print(target[:5].T, "target")
                        break
                        pass
                    pass
                pass
            pass
        pass

fds=432




