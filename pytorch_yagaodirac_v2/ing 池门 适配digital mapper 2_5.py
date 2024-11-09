from typing import List, Tuple, Any
import torch

import sys
ori_path = sys.path[0]
index = ori_path.rfind("\\")
upper_folder = ori_path[:index]
sys.path.append(upper_folder)
del ori_path
del index
del upper_folder

#from pytorch_yagaodirac_v2.Digital_mapper_v2_5 import DigitalMapper_v2_5

#笔记。
#这个版本适配的是目标回传（target/label/answer propagation）的digital mapper，具体版本是2.4和2.5，我忘了2.3是不是了。
#反正只有2.5是通过了最后那个测试的。所以这个门层现在只适配2.5.
#门层本身的原理和选线器层很类似，能清晰的找到回传通路的，直接传答案回去，不能清晰的找到的，乘以alpha缩小，但是方向依然是队的。
#考虑到xor的特殊性，暂时不写。
#这个版本在11月的答复补正当中不最终确定，在后续的主动补正里面再具体描述。
#门只有，非，且，或，3个。暂时不做且非和或非。所有多输入门只做2输入。

#基本原理是，不用之前的数学公式的版本了，那个适配的是误差回传（error/backward propagation)。
#原理是用类似池层的手感，但是不是池，而是完全自己写的一个结构。
#之前那个版本，且门是要通过一些计算，然后过二值化层。这个版本根本就没有二值化那些事情了，forward里面是纯数字化的，离散的。
#反向传播的部分依然是我最喜欢的“稠密”手感。



def calculate_fake_alpha(raw:float)->float:
    r'''formula is: 
    -1......0......1
    -1......0..a   (a means alpha)
    but because the result will go through a gramo, I can save some calculation.
    -k.........0...k*a
    a*k+k == 2
    (a+1)k == 2
    k == 2/(a+1)
    what I need is the distance between 1 and k*a
    -k.........0...k*a???1
    then, 1-a*k == k-1 == 2/(a+1)-1 == (1-a)/(1+a)
    '''
    return (1.-raw)/(1+raw)

class AND_Gate_pool_ver(torch.autograd.Function):
    r'''
    forward input list:
    >>> x = args[0]# shape must be [batch, in_features]
    >>> fake_alpha = args[1]# call calculate_fake_alpha to get this number.
    backward input list:
    >>> g_in #shape of g_in must be [batch, out_features]
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        input_b_i:torch.Tensor = args[0]# shape must be [batch, in_features]
        fake_alpha_s:torch.Tensor = args[1]# something for backward
        
        if input_b_i.shape.__len__() !=2:
            raise Exception()
        if input_b_i.shape[-1] %2!=0:
            raise Exception()
        
        gate_count = input_b_i.shape[-1]//2
        temp1:torch.Tensor = input[:,:gate_count].gt(0.)
        temp2 = input[:,gate_count:].gt(0.)
        temp3 = temp1.logical_and(temp2)
        temp4 = temp3*2.-1
        output = temp4.to(input_b_i.dtype)
        output.requires_grad_(input_b_i.requires_grad)
        
        input_requires_grad = torch.tensor([input_b_i.requires_grad], device=input_b_i.device)
        ctx.save_for_backward(input_requires_grad, fake_alpha_s)
        return output

    @staticmethod
    def backward(ctx, g_in_b_o):
        #shape of g_in must be [batch, out_features]
        input_requires_grad:torch.Tensor
        fake_alpha_s:torch.Tensor
        (input_requires_grad, fake_alpha_s) = ctx.saved_tensors
        
        grad_for_x_b_i:Tuple[torch.tensor|None] = None
        if input_requires_grad:
            g_in_abs_b_o = g_in_b_o.abs()
            grad_for_x_b_i = g_in_b_o - g_in_abs_b_o*fake_alpha_s
            pass
        return grad_for_x_b_i, None

    pass  # class

if 'basic test' and True:
    input = torch.tensor([[1.,1,-1,1,-1,-1]], requires_grad=True)
    pred = AND_Gate_pool_ver(input,0.1)
    print(pred, "pred")
    target = torch.tensor([[1.,1,1]])
    pred.backward(target)
    print(input.grad, "input.grad")
    print()
    input = torch.tensor([[1.,1,-1,1,-1,-1]], requires_grad=True)
    pred = AND_Gate_pool_ver(input,0.1)
    print(pred, "pred")
    target = torch.tensor([[-1.,-1,-1]])
    pred.backward(target)
    print(input.grad, "input.grad")
    pass


# '''Does this gate layer protect the grad?'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# print(input)
# print(input.shape)
# layer = AND_np(input_per_gate=2)
# pred = layer(input)
# g_in = torch.ones_like(pred)
# torch.autograd.backward(pred, g_in, inputs=[a,b])
# print(a.grad, b.grad)
# fds=432


# '''Not important. The shape of eval path'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b,a,b,a,b,a,b), dim=1) 
# print(input.shape)
# layer = AND_01(input_per_gate=2)
# layer.eval()
# result = layer(input)
# print(result.shape)
# fds=432


# '''a basic test'''
# a = torch.tensor([[-1.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[-1.],[-1.],[1.],], requires_grad=True)
# input = torch.concat((a,b), dim=1) 
# layer = AND_01()
# layer.Binarize1.set_big_number(3.)#So you can adjust the range of grad.
# result = layer(input)
# print(result, "should be 0 0 1")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad")
# fds=432


# '''This old test shows a trick. You can use slightly non standard input to help
# you track the behavior of the tested layer.'''
# '''These 2 tests show how to protect the gradient.
# Basically the big_number of the first binarize layer is the key.
# By making it smaller, you make the gradient of inputs closer over each elements.
# For AND gate, if it gets a lot false as input, the corresponding gradient is very small.
# '''
# a = torch.tensor([[0.001],[0.992],[0.993],], requires_grad=True)
# b = torch.tensor([[0.004],[0.005],[0.996],], requires_grad=True)
# input = torch.concat((a,b,b,b), dim=1) 
# layer = AND_01(input_per_gate=4, first_big_number=1.1)
# # layer.Binarize1.set_big_number(1., I_know_Im_setting_a_value_which_may_be_less_than_1=True)
# result = layer(input)
# print(result, "should be 0., 0., 1.")
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad, should be 3x of a's")
# # more~

# old test but tells something.
# '''If the input of a gate is too many, then, the greatest gradient is a lot times
# bigger than the smallest one. If the greatest-grad element needs n epochs to train,
# and its gradient is 100x bigger than the smallest, then the least-grad element needs 
# 100n epochs to train. This may harm the trainability of the least-grad element by a lot.
# Basically, the gates layers are designed to use as AND2, AND3, but never AND42 or AND69420.'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b,b,b,b,b,b,b,b,b,b), dim=1) 
# layer = AND_01(input_per_gate=11, first_big_number=0.35)
# result = layer(input)
# print(result)
# g_in = torch.ones_like(result)
# torch.autograd.backward(result, g_in, inputs = [a,b])
# print(a.grad, "a's grad")
# print(b.grad, "b's grad, should be 10x of a's")

# '''eval doesn't care the range. It only cares the threshold. 
# The threshold is inferred by the base layer.'''
# a = torch.tensor([[0.1],[0.9],[0.9],])
# b = torch.tensor([[0.2],[0.2],[0.8],])
# c = torch.tensor([[0.3],[0.3],[0.7],])
# input = torch.concat((a,b,c), 1) 
# layer = AND_01(input_per_gate=3)
# layer.eval()
# result = layer(input)
# print(layer(input), "should be 0., 0., 1.")

# '''output mode'''
# a = torch.tensor([[0.],[1.],[1.],], requires_grad=True)
# b = torch.tensor([[0.],[0.],[1.],], requires_grad=True)
# input = torch.concat((a,b), 1) 
# layer = AND_01()#first big number 5?
# print(layer(input), "should be 0., 0., 1.")
# layer.eval()
# print(layer(input), "should be 0., 0., 1.")
# layer = AND_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=1)
# print(layer(input), "should be 0., 0., 1., 1., 1., 0.")
# layer.eval()
# print(layer(input), "should be 0., 0., 1., 1., 1., 0.")
# layer = AND_01(output_mode_0_is_self_only__1_is_both__2_is_opposite_only=2)
# print(layer(input), "should be 1., 1., 0.")
# layer.eval()
# print(layer(input), "should be 1., 1., 0.")

# '''__str__'''
# layer = AND_01()
# print(layer)

# fds=432




































raise Exception()



class AND_Gate_pool_ver(torch.nn.Module):
                 #first_big_number:float = 3., 
    def __init__(self, input_per_gate:int = 2, \
                output_mode_0_is_self_only__1_is_both__2_is_opposite_only:int=0, \
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        if not output_mode_0_is_self_only__1_is_both__2_is_opposite_only in[0,1,2]:
            raise Exception("Param:output_mode_0_is_self_only__1_is_both__2_is_opposite_only can only be 0, 1 or 2.")
        self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only = output_mode_0_is_self_only__1_is_both__2_is_opposite_only
        
        if input_per_gate<2:
            raise Exception("Param:input_per_gate should >=2.")
        self.input_per_gate = input_per_gate
        # self.input_per_gate = torch.nn.Parameter(torch.tensor([input_per_gate]), requires_grad=False)
        # self.input_per_gate.requires_grad_(False)
        
        # The intermediate result will be binarized with following layers.
        #self.Binarize = Binarize.create_01_to_01()
        self.Binarize_doesnot_need_gramo = Binarize.create_analog_to_np(False)
        #self.Binarize1.set_big_number(first_big_number, I_know_Im_setting_a_value_which_may_be_less_than_1=True)
       
        # No matter what happens, this layer should be designed to output standard binarized result.
        # Even if the sigmoid is avoidable, you still need a final binarize layer.
        # But, Binarize_01_to_01 does this for you, so you don't need an individual final_binarize layer.
        # self.Final_Binarize = Binarize_01_Forward_only()
        pass
   
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x:torch.Tensor
        if self.training:
            # If you know how pytorch works, you can comment this checking out.
            # if not input.requires_grad:
            #     raise Exception("Set x.requires_grad to True. If you know what you are doing, you can comment this line.")
            if len(input.shape)!=2:
                raise Exception("Gates layers only accept rank-2 tensor. The shape should be[batch, gate_count * input_count]. If you have 5 data in a batch, you need 3 gates and each is AND2(which needs 2 inputs), the shape should be (5,6).")
            
            x = input
            # into rank-3
            x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
            x = x.sum(dim=2, keepdim=False)#dim=2
            #back to rank-2
            
            offset = torch.tensor([self.input_per_gate], dtype=x.dtype, device=x.device).neg()+1.
            x = x + offset
            
            # binarize 
            x = self.Binarize_doesnot_need_gramo(x)
            if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                return x
            else:
                opposite = x.neg()
                if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return torch.concat([x,opposite], dim=1)
                if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return opposite
                raise Exception("unreachable code.")
            
        else:#eval mode
            with torch.inference_mode():
                x = input.gt(0.)
                x = x.view([x.shape[0], x.shape[1]//self.input_per_gate, self.input_per_gate])
                x = x.all(dim=2, keepdim=False)
                x = x.to(torch.int8)
                x = x*2-1
                x = x.to(input.dtype)
                
                if 0 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                    return x
                else:
                    opposite = x.neg()
                    if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return torch.concat([x,opposite], dim=1)
                    if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
                        return opposite
                    raise Exception("unreachable code.")
        #end of function
        
    def extra_repr(self) -> str:
        output_mode = "Original only"
        if 1 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Both original and opposite"
        if 2 == self.output_mode_0_is_self_only__1_is_both__2_is_opposite_only:
            output_mode = "Opposite only"
        
        result = f'AND/NAND layer, output range is [0., 1.], output mode is {output_mode} mode'
        return result
        
    pass
