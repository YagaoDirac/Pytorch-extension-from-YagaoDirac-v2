
from typing import Any, List, Tuple, Optional, Self
import torch
import math
from ParamMo import GradientModification
from util import debug_strong_grad_ratio, init_weight_style_1, debug_zero_grad_ratio
import util




#考虑一下 linear, gramo, xmo, leakyrelu, xmo

class test_FCNN_with_doumo_stack_test(torch.nn.Module): 
    def __init__(self, in_features: int, out_features: int, \
        mid_width:int, num_layers:int, needs_in_xmos:bool, \
            protect_param_every____epoch = 20, leaky = 0.01, \
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if num_layers<2:
            raise Exception("emmmm")
        self.in_features = in_features
        self.out_features = out_features
        self.mid_width = mid_width
        self.num_layers = num_layers
        self.needs_in_xmos = needs_in_xmos
        
        self.linears = torch.nn.ParameterList([])        
        temp_linear = torch.nn.Linear(in_features, mid_width, bias=False)
        temp_linear.weight.data = init_weight_style_1(in_features, mid_width)
        self.linears.append(temp_linear)
        for _ in range(num_layers-2):
            temp_linear = torch.nn.Linear(mid_width, mid_width, bias=False)
            temp_linear.weight.data = init_weight_style_1(mid_width, mid_width)
            self.linears.append(temp_linear)
            pass
        temp_linear = torch.nn.Linear(mid_width, out_features, bias=False)
        temp_linear.weight.data = init_weight_style_1(mid_width, out_features)
        self.linears.append(temp_linear)
        
        self.gramos = torch.nn.ParameterList([GradientModification() for _ in range(num_layers)])      
        self.reset_scaling_ratio_for_gramos()
        self.in_xmos = XModification()
        self.xmos_1 = torch.nn.ParameterList([XModification() for _ in range(num_layers-1)])      
        self.leakyrelus = torch.nn.ParameterList([torch.nn.LeakyReLU(leaky) for _ in range(num_layers-1)])       
        self.xmos_2 = torch.nn.ParameterList([XModification() for _ in range(num_layers-1)])      
        
        self.protect_param_every____epoch = protect_param_every____epoch
        self.protect_param_training_count = 0
        pass
    
    
    def reset_scaling_ratio_for_gramos(self):
        #the_factor = 3./math.sqrt(self.in_features)
        the_factor = math.sqrt(3.)#*in_features)
        
        self.gramos[0].set_scaling_ratio(the_factor)
        
        the_factor = 3./math.sqrt(self.mid_width)
        gramo:GradientModification
        for gramo in self.gramos[1:]:
            gramo.set_scaling_ratio(the_factor)
            pass
    # def set_scaling_ratio_for_gramos(self, set_to:float):
    #     gramo:GradientModification
    #     for gramo in self.gramos:
    #         gramo.set_scaling_ratio(set_to)
    #         pass
    #     pass
    def scale_the_scaling_ratio_for_gramos(self, by:float):
        gramo:GradientModification
        for gramo in self.gramos:
            gramo.set_scaling_ratio((gramo.scaling_ratio*by).item())
            pass
        pass
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        protects_param = False
        if self.protect_param_training_count >= self.protect_param_every____epoch:
            self.protect_param_training_count = 1
            protects_param = True
            pass
            
        if self.needs_in_xmos:
            x = self.in_xmos(x)
            pass
        for i in range(self.num_layers-1):
            #debug_the_linear_data = self.linears[i].weight.data
            #debug_length_sqr = x.mul(x).sum(dim=1)
            x = self.linears[i](x)
            if protects_param:
                with torch.no_grad():
                    length_sqr = x.mul(x).sum(dim=1)
                    big_count = length_sqr.gt(1.25).sum()
                    if big_count> self.out_features*0.9:
                        self.linears[i].weight.data *=0.9
                        pass
                    small_count = length_sqr.lt(0.8).sum()
                    if small_count> self.out_features*0.9:
                        self.linears[i].weight.data *=1.1
                        pass
                    pass
                pass
            #debug_length_sqr = x.mul(x).sum(dim=1)
            x = self.gramos[i](x)
            x = self.xmos_1[i](x)
            x = self.leakyrelus[i](x)
            x = self.xmos_2[i](x)
            pass
        x = self.linears[-1](x)
        debug_length_sqr = x.mul(x).sum(dim=1)
        x = self.gramos[-1](x)
        return x
    
    def print_zero_grad_ratio(self, log10_diff: float = 0, \
                epi_for_w: float = 0.01, epi_for_g: float = 0.01,):
        result_list:List[float] = []
        linear:torch.nn.Linear
        for linear in self.linears:
            result_list.append(debug_zero_grad_ratio(linear.weight))
            pass
        print("zero grad: {}".format(result_list))
        pass
    def print_strong_grad_ratio(self, log10_diff: float = 0, \
                epi_for_w: float = 0.01, epi_for_g: float = 0.01,):
        result_list:List[float] = []
        linear:torch.nn.Linear
        for linear in self.linears:
            result_list.append(debug_strong_grad_ratio(linear.weight, log10_diff, epi_for_w, epi_for_g))
            pass
        print("strong grad: [", end="")
        for item in result_list:
            print(f"{item:.3f}, ", end="" )
            pass
        print("]")
        pass
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, num_layers={}'.format(
            self.in_features, self.out_features, self.num_layers
        )
    pass #end of function.

fast_travel____end_of__test_FCNN_with_doumo_stack_test__class = 432
# '''test the param protection.'''
# model = test_FCNN_with_doumo_stack_test(3,1,2,2,False, protect_param_every____epoch=0)
# #print(model.linears[0].weight.data.shape)
# model.linears[0].weight.data = torch.tensor([[0.1,0.1,0.1,],[1,1,1,],])
# input = torch.ones([10,3])
# model(input)

# model.linears[0].weight.data = torch.tensor([[0.1,0.1,0.1,],[0.,0.,0.,],])
# input = torch.ones([10,3])
# model(input)
# fds=432

# '''does the rand init output x close to 1.?'''
# in_features = 300
# out_features = 400
# mid_width = 500
# input_temp = torch.rand([1,in_features])
# length_of_input_temp = input_temp.mul(input_temp).sum().sqrt()
# input = input_temp/length_of_input_temp
# debug_checks_the_length = input.mul(input).sum()
# model = test_FCNN_with_doumo_stack_test(in_features,out_features,mid_width,5,False, )
# model(input)
# fds=432




fast_travel____test_FCNN_with_doumo_stack_test = 432
batch = 10000
in_feature = 100
out_feature = 50
mid_width = 512
num_layers = 3
lr = 1e-4#1e-3
leaky = 0.5
iter_per_print = 100#111
print_count = 111111

temp = torch.rand([batch, in_feature])*0.5+0.1
input = util.float_to_spherical(temp)
original_target = temp + 0.2
target_temp = util.float_to_spherical(original_target)
target = util.vector_length_norm(target_temp)
print(input)
print(target)


model = test_FCNN_with_doumo_stack_test(input.shape[1], target.shape[1], 
                                mid_width, num_layers,needs_in_xmos=True, leaky = leaky)
#model.scale_the_scaling_ratio_for_gramos(10.)
# fds=432

#model.set_scaling_ratio_for_gramos()

if False and "print parameters":
    if True and "only the training params":
        for p in model.parameters():
            if p.requires_grad:
                print(p)
                pass
            pass
        pass
    else:# prints all the params.
        for p in model.parameters():
            print(p)
            pass
        pass

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

if True and "Fake loss test":#0.0005
    for _ in range(5):
        random_fake_target = torch.rand([batch, in_feature])*0.5+0.3
        random_fake_target = util.float_to_spherical(random_fake_target)
        random_fake_target = util.vector_length_norm(random_fake_target)
        fake_loss = loss_function(random_fake_target, target)
        print(fake_loss, "fake_loss")
        pass
    for _ in range(5):
        random_fake_original = torch.rand([batch, in_feature])*0.5+0.3
        random_fake_original_ = torch.rand([batch, in_feature])*0.5+0.3
        fake_loss_in_original = loss_function(random_fake_original, random_fake_original_)
        print(fake_loss_in_original, "fake_loss_in_original ")
        pass
    pass



# model.half().cuda()
# input = input.to(torch.float16).cuda()
# target = target.to(torch.float16).cuda()
# original_target = original_target.to(torch.float16).cuda()
model.cuda()
input = input.cuda()
target = target.cuda()
original_target = original_target.cuda()
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
            print(pred[:1], "pred")
            print(target[:1], "target")
            pass
        pass
    loss = loss_function(pred, target)
    optimizer.zero_grad()
    loss.backward()
    if False and "make_grad_noisy":
        make_grad_noisy(model, noise_base)
        pass
    if False and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            #print(model.linears[0].weight.grad, "first_layer   grad")
            #print(model.linears[1].weight.grad, "first_layer   grad")
            print(model.linears[-1].weight.grad, "first_layer   grad")
            pass
        pass
    if False and "print the weight":
        if epoch%iter_per_print == iter_per_print-1:
            # layer = model.linears[0]
            # print(layer.weight[:1, :7], "first_layer.weight   before update")
            # optimizer.step()
            # print(layer.weight[:1, :7], "first_layer.weight   after update")
            
            # layer = model.linears[1]
            # print(layer.weight[:1, :7], "second layer.weight   before update")
            # optimizer.step()
            # print(layer.weight[:1, :7], "second layer.weight   after update")
            
            layer = model.linears[-1]
            print(layer.weight[:1, :7], "last layer.weight   before update")
            optimizer.step()
            print(layer.weight[:1, :7], "last layer.weight   after update")
            pass    
        pass    
    if False and "print zero grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            model.print_zero_grad_ratio()
            pass
        pass
    if True and "print strong grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            model.print_strong_grad_ratio(log10_diff=11.)
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    optimizer.step()
    if True and "print acc":
        if epoch%iter_per_print == iter_per_print-1:
            with torch.inference_mode():
                model.eval()
                pred = model(input)
                loss = loss_function(pred, target)
                
                pred_in_original_form = util.spherical_to_float(pred)
                loss_in_original_form = loss_function(pred_in_original_form,original_target)
                
                print("", epoch, "    ep/loss direct/original    ", f"{loss.item():.6f}", " / ", f"{loss_in_original_form.item():.6f}", "   __line 483")
                pass
            pass
        pass
    pass#end of main training loop.
with torch.inference_mode():
    model.eval()
    pred = model(input)
    loss = loss_function(pred, target)
    print(pred[:2, :7], "pred")
    print(target[:2, :7], "target")
    print("Epochs ran out...........", epoch, "    ep/loss    ", f"{loss.item():.6f}", "    __line 499")
    pred_in_original_form = util.spherical_to_float(pred)
    loss_in_original_form = loss_function(pred_in_original_form,original_target)
    print(loss_in_original_form, "loss_in_original_form")
    pass
fds=432