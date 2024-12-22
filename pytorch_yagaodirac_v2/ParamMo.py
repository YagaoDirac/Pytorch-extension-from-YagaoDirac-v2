# 后续计划。
# 简化版本的gramo
# x的归一化。
# 直接堆叠测试。
# 结合relu，tanh，sigmoid测试。


'''
Somethin you need to know, but Idk how to implememt as a param layer.
When some elements of a forward input of a layer(usually denoted by x)
are too close to 0, in most cases, the grad any weight element accumulates
is calculated with something from input of grad, multiplied by the input of forward.
When either is very close or equals to 0, the weight element doesn't accumulate useful grad.
The strength is too small.
When training speed(measured by amount of epochs) is more important than eventual precision, 
it's ok to dig a hole in both x and g. It's ok to offset them away from 0, 
usually by add some constant number according to the sign.
But, to make sure this offset only affects accumulated grad, I can only be done in a integrated
torch.autograd.function. 
I provided some code to help you with this, but don't use the code directly.
Copy paste the code to your customized torch.autograd.function.
And enjoy.
'''


'''
I use modification for all the names in this file. The purpose is to make it different from the 
normalization names being used since ever.
The difference is that, in most cases, normalization means some normal distribution based algorithm.
But in my experence, how the neural nets store n use info in numbers is more similar to linear algebra, not stats.
Over time, I found I need more tools to help protect all the passing param(x and g, input and grad).
Not only the linear algebra tools are needed. 
Then, the name of modification become a way to distinguish my code and others' code.
In this file, GradientModificaiton_xxx only affects grad, XModification only affects the x(forward passing param)
If you need to manipulate both x and g, use multiple from this file in a row. 
All the shape are design as [batch, dimention]. Data in different batches are absolutely isolated.
A trick. Some times some model param is in [out, in]shape. 
If you want all the elements to be treated as an entire vector, reshape to [1, -1], pass through some modification layer,
then reshape back to [out, in]. If you want it to be treated in a per output way, directly pass it to some layer in this file.
This trick is actually used in my digital mapper layer. In some cases it helps.


All the docs below are outdated and not checked again. Some are still referencable, some are very wrong.
All the docs below are outdated and not checked again. Some are still referencable, some are very wrong.
All the docs below are outdated and not checked again. Some are still referencable, some are very wrong.


All the main parts are probably corrected. But all the examples and 
integrated tests are nor validated after the latest update. 
Any of them may be wrong, or not work. But you still can use them as tutorials.

1 extra info I would like to provide. 
Sigmoid may have a chance to work in this project.
If:
x = sigmoid(x)
x = gramo(x)
x -= 0.5
x *= 5. (or 3. to 10.)
x = w*x+b ( or mirror style)
x = next sigmoid(x)
Notice that, 
x -= 0.5
x *= 5. (or 3. to 10.)
are not used in old tests.
I got new inspiration from my next project of DDNN.
I plan to come back and redo all the tests in late jul or aug.
'''


'''
Known issue.

Idk if this is a pytorch bug, or intentional.
If a nn.Parameter doesn't need grad, the code can only be:
p = torch.nn.Parameter(...)
p = p.requires_grad_(False)
If you directly specify the require_grad in the constructor function call, it's ignored.

When using customized backward function, 
2 ways are accepted, the 1st is :
loss = loss_func(some param, torch.tensor(some data, requires_grad = True))
The 2dn way is :
target = torch.tensor(some data)
loss = loss_func(some param, target)
By default, torch.tensor set the requires_grad to False. 
So, some black magic happens behind the scene.

To access the learning rate, the code looks like:
optimizer.param_groups[0]["lr"]
You can also set to this member to change it directly. I believe it's safe.
Most of the basic test are done with SGD for simplicity.

Search for convert_to_plain_Linear function.
This function provides a equivalence plain Linear layer( or the Dense layer in Tensorflow).
You basically want to convert the mirror (both w/ and w/o the gramo enhancement) to a plain Linear one before you do the ONNX part.
The reason is, a mirror is basically 2 Linear, but does the same thing.
Mirror is only good for training, but 1x slower in predicting while needs 1x more memory.
'''

'''
Known conclusions:

I'm not after the latest researchment for a while. 
At least several years ago, "assigning different lr for different layers" was an open question.
I believe we can close it now. It's very helpful, and powerful. 
Not only the lr should be modified for each layer, the relationship between the strength of grad of weight and bias should also
 be haddled carefully. If the updating strength of bias is greater then that one of the weight, the model doesn't train in a 
 proper way. So, it helps a lot to check all the inner param and their grad. If this issue happens, the output looks like a 
 torch.ones_like(...)*some value, for instance[0.123, 0.123, 0.123...](it's repeating). It's supposed to be something like 
 [0.123, 0.321, 0.6969, 0.4242...]

I don't have time for any further test. Here's an untested workflow.
1) Write the class (torch.nn.Module). Train it a bit.
2) Add a callback function to modify the bias.grad before optimizer.step().
3) Check if the output is repeating the same number for different input, if so, bias.grad*=0.1 with the callback function in the
 last step. Or maybe in some cases, it's *=0.01. Now the model should output differently for different input.
Get weight, weight.grad, bias, bias.grad. Calc absolute value, sort, remove the smallest 20%, calc the log, then avg.
Now you should get 4 numbers. Let's call them w,wg,b,bg. The relationship should be wg/w == bg/b, and for different layers, 
 the wg/w should be the same.
Now get back to the definition of model. With gramo or mig, you can use the scaling_factor param when create a gramo or mig to
 align the wg/w between layers. Usually, the layers in the mid are very similar at wg/w, so you can use them as reference 
 safely. Then the callback function you created in step 2 can help aligning the bg/b to the layer's wg/w.
End)
This can be seen in the last test. I did it manually. But I think it's time to write some code to automate it.




For all test:
Aligning the avg length of gradients for each layer is optional, it helps in some cases. 
In all the useful test, I only carefully aligned it for the last test.

When creating any gramo layer, set the scaling_factor param. You can use the result in a recent paper from XAI.


In most cases:
Sigmoid doesn't work when the test is scaled. It works for small test but it's too small to be any useful. 
With sigmoid, the loss value also drops, but the output doesn't look correct in some cases. 
With what I got from all the test, relu basically works better.

This file contains 3 building blocks for deep learning.
They are completely different from resnet. 
They guarantee the trainability in 2 ways, the 3rd one is a combination of previous 2.
According to the test, the combination doesn't look better than the second one. Idk if I messed with any test.


Gramo(GRA-dient MO-dificatino layer):
(Notice, by gramo, I mean the only gramo alone. The combination is explained below in another section.)
(Notice 2, this layer by itself doesn't learn anything. It feels like an activation function, but it doesn't do anything
 in the forward path.)
This is the first building block in this file.
It normalize the grad for each batch. When the length of grad is too small, it simply scale it a bit but if
 I set correctly, the result for super small input is always smaller than 1. Then right before the return
 directive, the grad is multiplied by scaling_factor.
This layer is supposed to be used right after any learnable layer, for inst Linear, Conv.
Since this layer protects the grad through the model while it doesn't touch the foreward path, it helps with
 training super deep models. I simply stacked 10 Linear/sigmoid and in another test 10 Linear/relu, and
 it trains very quickly. Which is already much better, if the tests were correct enough.

Linear Gramo Sigmoid/ReLU:
I tested this one with Linear(or Dense in tensorflow). The linear,gramo,sigmoid works fine.
Sigmoid softly mutes some neutrons behind it much more heavy than others, while gramo doesn't help in this case.
If I want to solve this problem, I probably need to pow(g,0.1), then normalize it, then pow(g,10) back.
Or maybe some new activation function like x/(1+x), or x*x/(1+x*x) to soften the grad. But this new
 activation is not widely tested. I don't plan to do this now.
Relu mutes neutrons hard. If too many are muted, the model doesn't train properly. Sometimes it needs the
 initialization to be good enough. But generally, according to my test, if the *input and model* are 
  both wide enough, it trains properly. The linear,gramo,relu doesn't work at width of 4.
Since the width of input is also counted as the width of the model, we really need a way to widen the input(duplicating may help)
 to make it work with relu.


Mirror(The magic Mirror):
Notice. Before this version, it works like 0.5*(w*(x+delta)+w*(x-delta))+b. But now, it works like half_w*(x+delta)+half_w*(x-delta)+b.
So, maybe I should name it in other ways.
The idea is that, the grad for w is g_in*x. If g_in is always in useful range, but there's absolutely
 no way to guaruntee all the x are in useful range. When the x is too close to 0, the w gets no useful grad.
To solve this problem, the simpliest way is to shift x a bit. When x is too close to 0(bad), x+delta is 
 basically at a distance of delta from 0(good). But this breaks the forward path. The formula shown above
  solved the problem. 2 w are different in code, 2 half_w are also different from each other in code. So
   at least one of them gets a useful grad.
When x is outside [-delta, delta], it's the same as a plain Linear. When x is in that range, it looks like
 the g is delta*2.. I use a 0.5 for delta so to align the learning strength to plain Linear.
This tool replaces the Linear(or Dense in Tensorflow). 

Mirror Sigmoid/ReLU:
Relu works. The output looks very correct.
(Sigmoid doesn't. The loss value decreases a bit but the outputs don't look correct.)
I remember the 10 layer test(with relu) only comsumed 3000 epochs, which is almost 10x faster than the combination.


The last one, the combination of the previous 2.
The MIG(MI-rror with G-ramo)(HELP ME NAME IT)
It looks slower than the previous one. 
But it trains almost the same speed with 10 layers as with 7 layers. 
Maybe we can stack much more layers directly and expect it to train at a decent speed.

'''

'''
About the test:
Notice: I only have 1 pcs of 1660(6G).
All the test are done with width 64 for most layers.

test_stacked_Linear_gramo_3:
It's a stacked Linear with gramo test.
I didn't push it to the extreme. I believe the result is good enough.
For no more than 6 layers, it's basically an instant training.
The test for <=6 layers was on cpu, then I move it to gpu for bigger test afterward.

test_multi_mirror__relu:
It's a stacked Mirror test.
For no more than 8 layers, it's basically within 1000 epochs.
Even for 10 layers, it's about 2000 epochs to train it.
The result is very crazy.
All the test from 3 layer to 10 layers are all on cpu.

test_multi_MIG__relu:
It's the final test of the combination.
It takes more epochs than the 2nd one. 

Maybe the Mirror withOUT gramo is the best.. I expect the last one to be the best, but the test almost drove me crazy.
'''


from typing import Any, Optional
#import math
import torch


# __all__ = [
#     'make_grad_noisy',
#     'GradientModification',
#     'MirrorLayer',
#     'MirrorWithGramo',
#     'GradientModificationFunction', #Should I expose this?
#     'Linear_gramo', #Should I rename this one? Or somebody help me with the naming?
#     ]






def assert_param_shape__batch_dim(param:torch.Tensor):
    assert param.shape.__len__() == 2, "Only accept rank-2 tensor. The shape should be[batch, something]"
    pass



def assert_param_for_ParamMo_make_holo_direct_offset(holo:torch.Tensor, epi:torch.Tensor):
    #assert param.shape.__len__() == 2, "Only accept rank-2 tensor. The shape should be[batch, something]"
    assert epi>0.
    assert holo>0.
    pass

def ParamMo_make_holo_direct_offset(param:torch.Tensor, holo:torch.Tensor, epi:torch.Tensor)->torch.Tensor:
    flag_pos = param.gt(epi)
    flag_neg = param.lt(-epi)
    flag_zero = flag_pos.logical_not().logical_and(flag_neg.logical_not())

    part_pos = (param+holo)*flag_pos
    part_neg = (param-holo)*flag_neg
    
    some_rand_sign = torch.randint(low=0,high=2,size=[1])*2.-1.
    part_zero = (param+some_rand_sign*holo)*flag_zero
    
    result = part_pos+part_neg+part_zero
    return result

if 'how to make param holo. Style 1, linear offset.' and False:
    the_input = torch.tensor([[-1,-0.1,0,0.1,1]])
    epi = torch.tensor(0.01)
    holo = torch.tensor(0.2)
    
    assert_param_shape__batch_dim(the_input)
    assert_param_for_ParamMo_make_holo_direct_offset(holo, epi)
    result = ParamMo_make_holo_direct_offset(the_input, holo, epi)
    
    print(the_input)
    print(result)
    pass



def assert_param_for_ParamMo_make_holo_keep_the_max_abs_as_1(holo:torch.Tensor, epi:torch.Tensor):
    #assert param.shape.__len__() == 2, "Only accept rank-2 tensor. The shape should be[batch, something]"
    assert epi>0.
    assert holo>0.
    pass

def ParamMo_make_holo_keep_the_max_abs_as_1(param:torch.Tensor, holo:torch.Tensor, epi:torch.Tensor)->torch.Tensor:
    r'''
    Detail: This method assumes the max abs is already 1, and does NOT modify this boundary.
    It pushes elements close at 0 away from 0, but doesn't touch elements close to +-1 very much.
    '''
    one_minus_holo = 1-holo
    flag_pos = param.gt(epi)
    flag_neg = param.lt(-epi)
    flag_zero = flag_pos.logical_not().logical_and(flag_neg.logical_not())

    part_pos = (param*one_minus_holo+holo)*flag_pos
    part_neg = (param*one_minus_holo-holo)*flag_neg
    
    some_rand_sign = torch.randint(low=0,high=2,size=[1])*2.-1.
    part_zero = (param+some_rand_sign*holo)*flag_zero
    
    result = part_pos+part_neg+part_zero
    return result

if 'how to make param holo. Style 2, keeps the max abs as 1' and False:
    the_input = torch.tensor([[-1,-0.1,0,0.1,1]])
    epi = torch.tensor(0.01)
    holo = torch.tensor(0.2)
    
    assert_param_shape__batch_dim(the_input)
    assert_param_for_ParamMo_make_holo_keep_the_max_abs_as_1(holo, epi)
    result = ParamMo_make_holo_keep_the_max_abs_as_1(the_input, holo, epi)
    
    print(the_input)
    print(result)
    pass
    






class XModificationFunction_sign_balance_abs_to_less_than_1(torch.autograd.Function):
    r'''This autograd function scale grad to have a mean(abs(square)) to specified number(1 by default).
    It's designed mainly to help analogy signal handling with error propagation.
    
    input param:
    
    >>> x (shape must be [batch, dim])
    >>> div_me_when_g_too_small (basically the epi)

    return type: torch.Tensor
    
    to get a correct result of retain_grad, modify this class.
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, scaling_factor:float = torch.tensor([1.]), \
        #               epi=torch.tensor([1e-5]), \
        #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
        #but python grammar punched me.
        x:torch.Tensor = args[0]
        div_me_when_g_too_small = args[1]
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")
        
        flag_pos_b_i = x.gt(0.)
        temp_pos_elements_b_1 = x*flag_pos_b_i
        abs_of_sum_of_pos__raw_b_1 = temp_pos_elements_b_1.sum(dim=1,keepdim=True)
        abs_of_sum_of_pos_b_1 = abs_of_sum_of_pos__raw_b_1.maximum(div_me_when_g_too_small)
        top_element_of_pos__raw_b_1 = temp_pos_elements_b_1.amax(dim=1,keepdim=True)
        top_element_of_pos_b_1 = top_element_of_pos__raw_b_1.maximum(div_me_when_g_too_small)
        
        temp_neg_elements_b_1 = x*(flag_pos_b_i.logical_not())
        abs_of_sum_of_neg__raw_b_1 = temp_neg_elements_b_1.sum(dim=1,keepdim=True).abs()
        abs_of_sum_of_neg_b_1 = abs_of_sum_of_neg__raw_b_1.maximum(div_me_when_g_too_small)
        top_element_of_neg__raw_b_1 = temp_neg_elements_b_1.amin(dim=1,keepdim=True).abs()
        top_element_of_neg_b_1 = top_element_of_neg__raw_b_1.maximum(div_me_when_g_too_small)
        
        temp_for_pos = abs_of_sum_of_pos_b_1/top_element_of_pos_b_1
        temp_for_neg = abs_of_sum_of_neg_b_1/top_element_of_neg_b_1
        
        flag_idk_how_to_name_it = temp_for_pos.lt(temp_for_neg)
        pos_div_me_part_1__b_1 = flag_idk_how_to_name_it*top_element_of_pos_b_1
        neg_div_me_part_1__b_1 = pos_div_me_part_1__b_1*abs_of_sum_of_neg_b_1/abs_of_sum_of_pos_b_1

        not_flag_idk_how_to_name_it = flag_idk_how_to_name_it.logical_not()
        neg_div_me_part_2__b_1 = not_flag_idk_how_to_name_it*top_element_of_neg_b_1
        pos_div_me_part_2__b_1 = neg_div_me_part_2__b_1*abs_of_sum_of_pos_b_1/abs_of_sum_of_neg_b_1
        
        pos_div_me_b_1 = pos_div_me_part_1__b_1+pos_div_me_part_2__b_1
        neg_div_me_b_1 = neg_div_me_part_1__b_1+neg_div_me_part_2__b_1
        
        final_pos_elements = temp_pos_elements_b_1/pos_div_me_b_1
        final_neg_elements = temp_neg_elements_b_1/neg_div_me_b_1
        
        result = final_pos_elements+final_neg_elements
        return result

    @staticmethod
    def backward(ctx, g_in_b_o):
        return g_in_b_o, None
    pass  # class

if '''basic test''' and False:
    div_me_when_g_too_small = torch.tensor(1e-2)
    a = torch.tensor([[2.,0,0,0],[2,1,0,0],[2,1,-4,0],[2,1,-4,-1],], dtype=torch.float16)
    b = XModificationFunction_sign_balance_abs_to_less_than_1.apply(a,div_me_when_g_too_small)
    print(b)
    print(b.sum(dim=1))
    print()
    
    a = torch.tensor([[1.,1,1,-1],[1,1,1,-2],[2,2,2,-1],[1,1,2,-1],], dtype=torch.float16)
    aa = a*torch.rand([1])*-1
    b = XModificationFunction_sign_balance_abs_to_less_than_1.apply(aa,div_me_when_g_too_small)
    print(b)
    print(b.sum(dim=1))
    print()
    pass

if '''basic test with random number''' and False:
    div_me_when_g_too_small = torch.tensor(1e-2)
    a = torch.randn([3,5])
    b = XModificationFunction_sign_balance_abs_to_less_than_1.apply(a,div_me_when_g_too_small)
    print(b)
    print(b.sum(dim=1))
    print(b.sum(dim=1).abs().lt(0.0001))
    print(b/a)
    print()
    pass

if '''dtype adaption.''' and False:
    div_me_when_g_too_small = torch.tensor(1e-2, dtype=torch.float64)
    
    a = torch.tensor([[0.]], requires_grad=True, dtype=torch.float16)
    original_dtype = a.dtype
    b = XModificationFunction_sign_balance_abs_to_less_than_1.apply(a, div_me_when_g_too_small)
    ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
    g_in = torch.tensor([[1.]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad.dtype, "should be ", original_dtype)
    pass

if '''device adaption''' and False:
    div_me_when_g_too_small = torch.tensor(1e-2).cuda()
    
    a = torch.tensor([[0.]], requires_grad=True).cuda()
    b = XModificationFunction_sign_balance_abs_to_less_than_1.apply(a,div_me_when_g_too_small)
    g_in = torch.tensor([[1.]]).cuda()
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad.device, "should be cuda")
    pass



class XModification_sign_balance_abs_to_less_than_1(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """

    def __init__(self, div_me_when_g_too_small=1e-2, \
                    *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.div_me_when_g_too_small=torch.nn.Parameter(torch.tensor(div_me_when_g_too_small), requires_grad=False)
        pass
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        result = XModificationFunction_sign_balance_abs_to_less_than_1.apply(
            x,self.div_me_when_g_too_small)
        
        return result
   
    def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
        the_device = self.div_me_when_g_too_small.device
        the_dtype = self.div_me_when_g_too_small.dtype
        self.div_me_when_g_too_small.data = torch.tensor(div_me_when_g_too_small, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def extra_repr(self) -> str:
        return f'div_me_when_g_too_small={self.div_me_when_g_too_small.item():.2e}'

if '''all the setters''' and False:
    layer = XModification_sign_balance_abs_to_less_than_1(0.5)
    print(layer.div_me_when_g_too_small.requires_grad, "should be False")
    layer.set_div_me_when_g_too_small(0.234)
    print(layer.div_me_when_g_too_small.item(), "should be 0.234")
    print(layer.div_me_when_g_too_small.requires_grad, "should be False")
    pass

if '''dtype adaption.''' and False:
    input = torch.tensor([[1.]], requires_grad=True)
    target = torch.tensor([[0.]])
    model = XModification_sign_balance_abs_to_less_than_1(0.5)
    model.to(torch.float64)
    #model.to(torch.float16)

    loss_function = torch.nn.L1Loss()# the L1Loss function only provides the direction. It's the dirivitive of abs.
    optimizer = torch.optim.SGD([input], lr=0.1)
    for epoch in range(1):
        model.train()
        pred = model(input)
        print(pred.dtype, "pred.dtype should be f32")
        loss = loss_function(pred, target)
        print(loss.dtype, "loss.dtype should be f32")
        optimizer.zero_grad()
        loss.backward()
        #optimizer.param_groups[0]["lr"] = 0.01
        print(input.grad, "should be 1.")
        print(input.grad.dtype, "input.grad.dtype should be f32")

        optimizer.step()
        print(input, "should be 0.9")
        
        model.eval()
        pass
    pass







class XModificationFunction_abs_to_less_than_1_then_expansion(torch.autograd.Function):
    r'''This autograd function scale grad to have a mean(abs(square)) to specified number(1 by default).
    It's designed mainly to help analogy signal handling with error propagation.
    
    input param:
    
    >>> x (shape must be [batch, dim])
    >>> forward_expansion_factor (if this is 1., the layer doesn't do anything.)
    >>> div_me_when_g_too_small (basically the epi)

    return type: torch.Tensor
    
    to get a correct result of retain_grad, modify this class.
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, scaling_factor:float = torch.tensor([1.]), \
        #               epi=torch.tensor([1e-5]), \
        #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
        #but python grammar punched me.
        x:torch.Tensor = args[0]
        forward_expansion_factor = args[1]
        div_me_when_g_too_small = args[2]
        #flag_grad_expansion_back = args[3]
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")
        x_needs_grad = torch.tensor([x.requires_grad])
        x_needs_grad_bool = x.requires_grad
        
        #this helps????? maybe...
        max_of_each_batch_b_1 = x.abs().amax(dim=1,keepdim=True)
        max_of_each_batch__safe__b_1 = max_of_each_batch_b_1.maximum(div_me_when_g_too_small)
        reverse_of_max_of_each_batch__safe__b_1 = (1./max_of_each_batch__safe__b_1).to(x.dtype)
        x_bofore_expansion_b_o = x*reverse_of_max_of_each_batch__safe__b_1
        
        if forward_expansion_factor==1.:
            x = x_bofore_expansion_b_o
        else:
            sign__result_b_o = x_bofore_expansion_b_o.gt(0.)*2.-1.
            abs_of_result_b_o = x_bofore_expansion_b_o.abs()
            powered_abs_of_result_b_o = abs_of_result_b_o.pow(forward_expansion_factor)
            x = powered_abs_of_result_b_o*sign__result_b_o
            pass
        
        #ctx.save_for_backward(forward_expansion_factor,div_me_when_g_too_small,x_needs_grad)
        #ctx.save_for_backward(forward_expansion_factor,div_me_when_g_too_small,flag_grad_expansion_back,x_needs_grad)
        x.requires_grad_(x_needs_grad_bool)
        return x

    @staticmethod
    def backward(ctx, g_in_b_o):
        return g_in_b_o, None, None
        #super().backward()
        forward_expansion_factor:torch.Tensor
        div_me_when_g_too_small:torch.Tensor
        flag_grad_expansion_back:torch.Tensor
        x_needs_grad:torch.Tensor
        
        (forward_expansion_factor,div_me_when_g_too_small,flag_grad_expansion_back,x_needs_grad) = ctx.saved_tensors
        grad_for_x_b_o:Optional[torch.Tensor] = None
        
        if x_needs_grad:
                
            max_of_each_batch_b_1 = g_in_b_o.abs().amax(dim=1,keepdim=True)
            max_of_each_batch__safe__b_1 = max_of_each_batch_b_1.maximum(div_me_when_g_too_small)
            reverse_of_max_of_each_batch__safe__b_1 = (1./max_of_each_batch__safe__b_1).to(g_in_b_o.dtype)
            grad_for_x_bofore_scale_step2_b_o = g_in_b_o*reverse_of_max_of_each_batch__safe__b_1
            
            if forward_expansion_factor==1. or flag_grad_expansion_back.logical_not():
                grad_for_x_b_o = grad_for_x_bofore_scale_step2_b_o
            else:
                sign__grad_for_x_pos_b_o = grad_for_x_bofore_scale_step2_b_o.gt(0.)*2.-1.
                abs_of_grad_for_x_b_o = grad_for_x_bofore_scale_step2_b_o.abs()
                powered_abs_of_grad_for_x_b_o = abs_of_grad_for_x_b_o.pow(1./forward_expansion_factor)
                grad_for_x_b_o = powered_abs_of_grad_for_x_b_o*sign__grad_for_x_pos_b_o
                pass
            pass

        return grad_for_x_b_o, None, None, None

    pass  # class

if '''dim irrelated ???''' and False:
    forward_expansion_factor = torch.tensor(0.5)
    div_me_when_g_too_small = torch.tensor(2e-3)
    #flag_grad_expansion_back = torch.tensor(True)
    
    a = torch.tensor([[0.1,0.2],[0.01,0.02,],[0.001,0.002],[1e-4,2e-4],[1e-5,2e-5]], requires_grad=True, dtype=torch.float16)
    b = XModificationFunction_abs_to_less_than_1_then_expansion.apply(a,forward_expansion_factor,div_me_when_g_too_small)
    #b = XModificationFunction_abs_to_less_than_1_for_both_path.apply(a,forward_expansion_factor,div_me_when_g_too_small,flag_grad_expansion_back)
    print(b)
    g_in = torch.tensor([[0.1,0.2],[0.01,0.02,],[0.001,0.002],[1e-4,2e-4],[1e-5,2e-5]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad)
    
    a = torch.tensor([[0.1],[0.01],[0.001],[1e-4],[1e-5]], requires_grad=True, dtype=torch.float16)
    b = XModificationFunction_abs_to_less_than_1_then_expansion.apply(a,forward_expansion_factor,div_me_when_g_too_small)
    print(b)
    g_in = torch.tensor([[0.1],[0.01],[0.001],[1e-4],[1e-5]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad)
    pass

if '''dtype adaption.''' and False:
    forward_expansion_factor = torch.tensor(1., dtype=torch.float64)
    div_me_when_g_too_small = torch.tensor(2e-3, dtype=torch.float16)
    
    a = torch.tensor([[0.]], requires_grad=True, dtype=torch.float16)
    original_dtype = a.dtype
    b = XModificationFunction_abs_to_less_than_1_then_expansion.apply(a,forward_expansion_factor,div_me_when_g_too_small)
    ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
    g_in = torch.tensor([[1.]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad.dtype, "should be ", original_dtype)
    pass

if '''device adaption''' and False:
    forward_expansion_factor = torch.tensor(1.).cuda()
    div_me_when_g_too_small = torch.tensor(2e-3).cuda()
    
    a = torch.tensor([[0.]], requires_grad=True).cuda()
    b = XModificationFunction_abs_to_less_than_1_then_expansion.apply(a,forward_expansion_factor,div_me_when_g_too_small)
    g_in = torch.tensor([[1.]]).cuda()
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad.device, "should be cuda")
    pass



class XModification_abs_to_less_than_1_then_expansion(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """

    def __init__(self, forward_expansion_factor:float,div_me_when_g_too_small=1e-2, \
                    *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forward_expansion_factor = torch.nn.Parameter(torch.tensor(forward_expansion_factor), requires_grad=False)
        self.div_me_when_g_too_small=torch.nn.Parameter(torch.tensor(div_me_when_g_too_small), requires_grad=False)
        pass
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        result = XModificationFunction_abs_to_less_than_1_then_expansion.apply(
            x,self.forward_expansion_factor,self.div_me_when_g_too_small)
        
        return result
        
    def set_forward_expansion_factor(self, forward_expansion_factor:float)->None:
        the_device = self.forward_expansion_factor.device
        the_dtype = self.forward_expansion_factor.dtype
        self.forward_expansion_factor.data = torch.tensor(forward_expansion_factor, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
        the_device = self.div_me_when_g_too_small.device
        the_dtype = self.div_me_when_g_too_small.dtype
        self.div_me_when_g_too_small.data = torch.tensor(div_me_when_g_too_small, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def extra_repr(self) -> str:
        return f'grad_expansion_factor={self.grad_expansion_factor.item():.3e}, div_me_when_g_too_small={self.div_me_when_g_too_small.item():.2e}'

if '''all the setters''' and False:
    layer = XModification_abs_to_less_than_1_then_expansion(0.5)
    print(layer.forward_expansion_factor.requires_grad, "should be False")
    print(layer.div_me_when_g_too_small.requires_grad, "should be False")
    layer.set_forward_expansion_factor(0.123)
    print(layer.forward_expansion_factor.item(), "should be 0.123")
    print(layer.forward_expansion_factor.requires_grad, "should be False")
    layer.set_div_me_when_g_too_small(0.234)
    print(layer.div_me_when_g_too_small.item(), "should be 0.234")
    print(layer.div_me_when_g_too_small.requires_grad, "should be False")
    pass

if '''dtype adaption.''' and False:
    input = torch.tensor([[1.]], requires_grad=True)
    target = torch.tensor([[0.]])
    model = XModification_abs_to_less_than_1_then_expansion(0.5)
    model.to(torch.float64)
    #model.to(torch.float16)

    loss_function = torch.nn.L1Loss()# the L1Loss function only provides the direction. It's the dirivitive of abs.
    optimizer = torch.optim.SGD([input], lr=0.1)
    for epoch in range(1):
        model.train()
        pred = model(input)
        print(pred.dtype, "pred.dtype should be f32")
        loss = loss_function(pred, target)
        print(loss.dtype, "loss.dtype should be f32")
        optimizer.zero_grad()
        loss.backward()
        #optimizer.param_groups[0]["lr"] = 0.01
        print(input.grad, "should be 1.")
        print(input.grad.dtype, "input.grad.dtype should be f32")

        optimizer.step()
        print(input, "should be 0.9")
        
        model.eval()
        pass
    pass







class GradientModificationFunction_v2_mean_abs_to_1(torch.autograd.Function):
    r'''This autograd function scale grad to have a mean(abs(square)) to specified number(1 by default).
    It's designed mainly to help analogy signal handling with error propagation.
    
    input param:
    >>> x:torch.Tensor (must be set as require_grad = True)
    >>> scaling_factor = torch.tensor([1.])
    >>> epi = torch.tensor([1e-5])
    >>> div_me_when_g_too_small = torch.tensor([1e-3])

    retur type: torch.Tensor
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, scaling_factor:float = torch.tensor([1.]), \
        #               epi=torch.tensor([1e-5]), \
        #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
        #but python grammar punched me.
        x:torch.Tensor = args[0]
        scaling_factor = args[1]
        epi = args[2]
        mul_me_when_g_too_small = args[3]
        # the default values:
        # scaling_factor = torch.tensor([1.])
        # epi = torch.tensor([0.00001])
        # div_me_when_g_too_small = torch.tensor([0.001])
        # the definition of the 3 param are different from the previous version
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")
        
        x_needs_grad = torch.tensor([x.requires_grad])
        ctx.save_for_backward(scaling_factor, epi, mul_me_when_g_too_small, x_needs_grad)
        return x

    @staticmethod
    def backward(ctx, g_in_b_o):
        #super().backward()
        scaling_factor:torch.Tensor
        epi:torch.Tensor
        epi:torch.Tensor
        mul_me_when_g_too_small:torch.Tensor
        (scaling_factor, epi, mul_me_when_g_too_small, x_needs_grad) = ctx.saved_tensors
        
        grad_for_x_b_o:Optional[torch.Tensor] = None
        
        if x_needs_grad:
            out_features_as_float:torch.Tensor = torch.tensor([g_in_b_o.shape[-1]], dtype=torch.float64, device=g_in_b_o.device)
            #mul_me_when_g_too_small = mul_me_when_g_too_small_per_element#*out_features_as_float

            avg_length_per_element_b_1:torch.Tensor = (g_in_b_o.mul(g_in_b_o).sum(dim=1,keepdim = True)/out_features_as_float).sqrt()
            mul_me_when_g_is_ok_raw_b_1 = scaling_factor/avg_length_per_element_b_1
            
            mul_me_when_g_is_ok_raw_b_1.nan_to_num_(mul_me_when_g_too_small.item())
            not_too_big_flag = mul_me_when_g_is_ok_raw_b_1.lt(mul_me_when_g_too_small*1000)#is this needed?
            mul_me_when_g_is_ok_b_1 = not_too_big_flag*mul_me_when_g_is_ok_raw_b_1+not_too_big_flag.logical_not()*mul_me_when_g_too_small
            
            too_small_b_1:torch.Tensor = avg_length_per_element_b_1.le(epi)#*out_features_as_float)
            
            mul_me_b_1 = too_small_b_1.logical_not()*mul_me_when_g_is_ok_b_1+ too_small_b_1*mul_me_when_g_too_small
            mul_me_b_1 = mul_me_b_1.to(g_in_b_o.dtype)
            grad_for_x_b_o:torch.Tensor = g_in_b_o*mul_me_b_1
            pass

        return grad_for_x_b_o, None, None, None

    pass  # class

if '''dim irrelated gramo''' and False:
    scaling_factor = torch.tensor([1.], dtype=torch.float64)
    epi=torch.tensor([1e-3], dtype=torch.float32)
    mul_me_when_g_too_small = torch.tensor([10], dtype=torch.float16)
    a = torch.zeros([5,2], requires_grad=True, dtype=torch.float16)
    b = GradientModificationFunction_v2_mean_abs_to_1.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.1,0.2],[0.01,0.02,],[0.001,0.002],[1e-4,2e-4],[1e-5,2e-5]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad)
    
    a = torch.zeros([5,1], requires_grad=True, dtype=torch.float16)
    b = GradientModificationFunction_v2_mean_abs_to_1.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[0.1],[0.01],[0.001],[1e-4],[1e-5]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad)
    pass

if '''dtype adaption.''' and False:
    scaling_factor = torch.tensor([1.], dtype=torch.float64)
    epi=torch.tensor([1e-5], dtype=torch.float32)
    mul_me_when_g_too_small = torch.tensor([1e3], dtype=torch.float16)
    a = torch.tensor([[0.]], requires_grad=True, dtype=torch.float16)
    original_dtype = a.dtype
    b = GradientModificationFunction_v2_mean_abs_to_1.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
    g_in = torch.tensor([[1.]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    #print(g[0])
    print(a.grad.dtype, "should be ", original_dtype)
    pass

if '''device adaption''' and False:
    scaling_factor = torch.tensor([1.]).cuda()
    epi=torch.tensor([1e-5]).cuda()
    mul_me_when_g_too_small = torch.tensor([1e3]).cuda()
    a = torch.tensor([[0.]], requires_grad=True).cuda()
    b = GradientModificationFunction_v2_mean_abs_to_1.apply(a,scaling_factor,epi,mul_me_when_g_too_small)
    g_in = torch.tensor([[1.]]).cuda()
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad.device, "should be cuda")
    pass



class GradientModification_v2_mean_abs_to_1(torch.nn.Module):
    r"""This autograd function scale grad to have a mean(abs(square)) to specified number(1 by default).
    It's designed mainly to help analogy signal handling with error propagation.
    
    Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """

    def __init__(self, scaling_factor:float = 1., \
                       epi=1e-5, \
                       mul_me_when_g_too_small = 1e3, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scaling_factor = torch.nn.Parameter(torch.tensor(scaling_factor), requires_grad=False)
        self.epi=torch.nn.Parameter(torch.tensor(epi), requires_grad=False)
        self.mul_me_when_g_too_small = torch.nn.Parameter(torch.tensor(mul_me_when_g_too_small), requires_grad=False)
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_factor:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction_v2_mean_abs_to_1.apply(x, self.scaling_factor, self.epi, \
                                                   self.mul_me_when_g_too_small)
    def set_scaling_factor(self, scaling_factor:float)->None:
        the_device = self.scaling_factor.device
        the_dtype = self.scaling_factor.dtype
        self.scaling_factor.data = torch.tensor(scaling_factor, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def scale_scaling_factor(self, by:float)->None:
        self.set_scaling_factor((self.scaling_factor*by).item())
        pass
    def set_epi(self, epi:float)->None:
        the_device = self.epi.device
        the_dtype = self.epi.dtype
        self.epi.data = torch.tensor(epi, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def set_mul_me_when_g_too_small(self, mul_me_when_g_too_small:float)->None:
        the_device = self.mul_me_when_g_too_small.device
        the_dtype = self.mul_me_when_g_too_small.dtype
        self.mul_me_when_g_too_small.data = torch.tensor(mul_me_when_g_too_small, device=the_device, dtype=the_dtype, requires_grad=False)
        pass

    def extra_repr(self) -> str:
        return f'scaling_factor={self.scaling_factor.item():.4e}, epi={self.epi.item():.4e}, mul_me_when_g_too_small={self.mul_me_when_g_too_small.item():.4e}'

if '''all the setters''' and False:
    model = GradientModification_v2_mean_abs_to_1()
    print(model.scaling_factor.requires_grad, "should be False")
    print(model.epi.requires_grad, "should be False")
    print(model.mul_me_when_g_too_small.requires_grad, "should be False")
    model.set_scaling_factor(0.123)
    print(model.scaling_factor, "should be 0.123")
    print(model.scaling_factor.requires_grad, "should be False")
    model.set_epi(0.234)
    print(model.epi, "should be 0.234")
    print(model.epi.requires_grad, "should be False")
    model.set_mul_me_when_g_too_small(0.345)
    print(model.mul_me_when_g_too_small, "should be 0.345")
    print(model.mul_me_when_g_too_small.requires_grad, "should be False")
    pass

if '''dtype adaption.''' and False:
    input = torch.tensor([[1.]], requires_grad=True)
    target = torch.tensor([[0.]])
    model = GradientModification_v2_mean_abs_to_1()
    model.to(torch.float64)
    #model.to(torch.float16)

    loss_function = torch.nn.L1Loss()# the L1Loss function only provides the direction. It's the dirivitive of abs.
    optimizer = torch.optim.SGD([input], lr=0.1)
    for epoch in range(1):
        model.train()
        pred = model(input)
        print(pred.dtype, "pred.dtype should be f32")
        loss = loss_function(pred, target)
        print(loss.dtype, "loss.dtype should be f32")
        optimizer.zero_grad()
        loss.backward()
        #optimizer.param_groups[0]["lr"] = 0.01
        print(input.grad, "should be 1.")
        print(input.grad.dtype, "input.grad.dtype should be f32")

        optimizer.step()
        print(input, "should be 0.9")
        
        model.eval()
        pass
    pass





class GradientModificationFunction_v2_abs_to_less_than_1(torch.autograd.Function):
    r'''This autograd function scale grad to have a max(abs) to 1.
    With a known lr, the maximum updating strength is limited to a known range, 
    which is eazier to use when the param protection is heavily applied.
    
    input param:
    >>> x:torch.Tensor (must be set as require_grad = True)
    >>> grad_expansion_factor =??? torch.tensor([1.])
    >>> div_me_when_g_too_small =??? torch.tensor([1e-5])

    retur type: torch.Tensor
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        #I tried to write like:
        #def forward(ctx, x:torch.Tensor, scaling_factor:float = torch.tensor([1.]), \
        #               epi=torch.tensor([1e-5]), \
        #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
        #but python grammar punched me.
        x:torch.Tensor = args[0]
        grad_expansion_factor = args[1]
        div_me_when_g_too_small = args[2]
        # the default values:
        # scaling_factor = torch.tensor([1.])
        # epi = torch.tensor([0.00001])
        # div_me_when_g_too_small = torch.tensor([0.001])
        # the definition of the 3 param are different from the previous version
        if len(x.shape)!=2:
            raise Exception("GradientModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")
        
        x_needs_grad = torch.tensor([x.requires_grad])
        ctx.save_for_backward(grad_expansion_factor, x_needs_grad,div_me_when_g_too_small)
        return x

    @staticmethod
    def backward(ctx, g_in_b_o):
        #super().backward()
        grad_expansion_factor:torch.Tensor
        x_needs_grad:torch.Tensor
        div_me_when_g_too_small:torch.Tensor
        (grad_expansion_factor, x_needs_grad,div_me_when_g_too_small) = ctx.saved_tensors
        
        grad_for_x_b_o:Optional[torch.Tensor] = None
        
        if x_needs_grad:
                
            #this helps????? maybe...
            max_of_each_batch_b_1 = g_in_b_o.abs().amax(dim=1,keepdim=True)
            max_of_each_batch__safe__b_1 = max_of_each_batch_b_1.maximum(div_me_when_g_too_small)
            reverse_of_max_of_each_batch__safe__b_1 = (1./max_of_each_batch__safe__b_1).to(g_in_b_o.dtype)
            grad_for_x_bofore_scale_step2_b_o = g_in_b_o*reverse_of_max_of_each_batch__safe__b_1
            
            sign__grad_for_x_pos_b_o = grad_for_x_bofore_scale_step2_b_o.gt(0.)*2.-1.
            abs_of_grad_for_x_b_o = grad_for_x_bofore_scale_step2_b_o.abs()
            powered_abs_of_grad_for_x_b_o = abs_of_grad_for_x_b_o.pow(grad_expansion_factor)
            grad_for_x_b_o = powered_abs_of_grad_for_x_b_o*sign__grad_for_x_pos_b_o
            
            pass

        return grad_for_x_b_o, None, None

    pass  # class

if '''dim irrelated gramo''' and False:
    grad_expansion_factor = torch.tensor([0.3], dtype=torch.float64)
    div_me_when_g_too_small = torch.tensor([2e-3], dtype=torch.float64)
    a = torch.zeros([5,2], requires_grad=True, dtype=torch.float16)
    b = GradientModificationFunction_v2_abs_to_less_than_1.apply(a,grad_expansion_factor,div_me_when_g_too_small)
    g_in = torch.tensor([[0.1,0.2],[0.01,0.02,],[0.001,0.002],[1e-4,2e-4],[1e-5,2e-5]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad)
    
    a = torch.zeros([5,1], requires_grad=True, dtype=torch.float16)
    b = GradientModificationFunction_v2_abs_to_less_than_1.apply(a,grad_expansion_factor,div_me_when_g_too_small)
    g_in = torch.tensor([[0.1],[0.01],[0.001],[1e-4],[1e-5]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad)
    
    grad_expansion_factor = torch.tensor([1.], dtype=torch.float64)
    a = torch.zeros([3,2], requires_grad=True, dtype=torch.float16)
    b = GradientModificationFunction_v2_abs_to_less_than_1.apply(a,grad_expansion_factor,div_me_when_g_too_small)
    g_in = torch.tensor([[0.1,0.2],[-0.1,0.2],[-0.1,-0.2],], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    print(a.grad)
    pass

if '''dtype adaption.''' and False:
    grad_expansion_factor = torch.tensor([1.], dtype=torch.float64)
    div_me_when_g_too_small=torch.tensor([1e-5], dtype=torch.float32)
    
    a = torch.tensor([[0.]], requires_grad=True, dtype=torch.float16)
    original_dtype = a.dtype
    
    b = GradientModificationFunction_v2_abs_to_less_than_1.apply(a,grad_expansion_factor,div_me_when_g_too_small)
    ### g = torch.autograd.grad(b, a, retain_graph= True)#this one doesn't help.
    g_in = torch.tensor([[1.]], dtype=torch.float16)
    torch.autograd.backward(b, g_in,inputs= a)
    #print(g[0])
    print(a.grad.dtype, "should be ", original_dtype)
    pass

if '''device adaption''' and False:
    grad_expansion_factor = torch.tensor([1.]).cuda()
    div_me_when_g_too_small=torch.tensor([1e-5]).cuda()
    a = torch.tensor([[0.]], requires_grad=True).cuda()
    b = GradientModificationFunction_v2_abs_to_less_than_1.apply(a,grad_expansion_factor,div_me_when_g_too_small)
    g_in = torch.tensor([[1.]]).cuda()
    torch.autograd.backward(b, g_in,inputs= a)
    #print(g[0])
    print(a.grad.device, "should be cuda")
    pass



class GradientModification_v2_abs_to_less_than_1(torch.nn.Module):
    r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
    To access the learning rate, you usually need some thing like:
    lr:float = optimizer.param_groups[0]["lr"]
    """

    def __init__(self, grad_expansion_factor:float = 0.3, \
                       div_me_when_g_too_small=1e-2, \
                        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grad_expansion_factor = torch.nn.Parameter(torch.tensor(grad_expansion_factor), requires_grad=False)
        self.div_me_when_g_too_small=torch.nn.Parameter(torch.tensor(div_me_when_g_too_small), requires_grad=False)
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.

        if len(x.shape)!=2:
            raise Exception("GradientModification only accept rank-2 tensor. The shape should be[batch, something]")

        #forward(ctx, x:torch.Tensor, scaling_factor:torch.Tensor, epi=torch.Tensor, \
        #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
        return GradientModificationFunction_v2_abs_to_less_than_1.apply(x, self.grad_expansion_factor, self.div_me_when_g_too_small)
    def set_grad_expansion_factor(self, grad_expansion_factor:float)->None:
        the_device = self.grad_expansion_factor.device
        the_dtype = self.grad_expansion_factor.dtype
        self.grad_expansion_factor.data = torch.tensor(grad_expansion_factor, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
        the_device = self.div_me_when_g_too_small.device
        the_dtype = self.div_me_when_g_too_small.dtype
        self.div_me_when_g_too_small.data = torch.tensor(div_me_when_g_too_small, device=the_device, dtype=the_dtype, requires_grad=False)
        pass
    def extra_repr(self) -> str:
        return f'grad_expansion_factor={self.grad_expansion_factor.item():.3e}, div_me_when_g_too_small={self.div_me_when_g_too_small.item():.2e}'

if '''all the setters''' and False:
    model = GradientModification_v2_abs_to_less_than_1()
    print(model.grad_expansion_factor.requires_grad, "should be False")
    print(model.div_me_when_g_too_small.requires_grad, "should be False")
    model.set_grad_expansion_factor(0.123)
    print(model.grad_expansion_factor, "should be 0.123")
    print(model.grad_expansion_factor.requires_grad, "should be False")
    model.set_div_me_when_g_too_small(0.234)
    print(model.div_me_when_g_too_small, "should be 0.234")
    print(model.div_me_when_g_too_small.requires_grad, "should be False")
    pass

if '''dtype adaption.''' and False:
    input = torch.tensor([[1.]], requires_grad=True)
    target = torch.tensor([[0.]])
    model = GradientModification_v2_abs_to_less_than_1()
    model.to(torch.float64)
    #model.to(torch.float16)

    loss_function = torch.nn.L1Loss()# the L1Loss function only provides the direction. It's the dirivitive of abs.
    optimizer = torch.optim.SGD([input], lr=0.1)
    for epoch in range(1):
        model.train()
        pred = model(input)
        print(pred.dtype, "pred.dtype should be f32")
        loss = loss_function(pred, target)
        print(loss.dtype, "loss.dtype should be f32")
        optimizer.zero_grad()
        loss.backward()
        #optimizer.param_groups[0]["lr"] = 0.01
        print(input.grad, "should be 1.")
        print(input.grad.dtype, "input.grad.dtype should be f32")

        optimizer.step()
        print(input, "should be 0.9")
        
        model.eval()
        pass
    pass























# class XModificationFunction(torch.autograd.Function):
#     r'''input param:
#     >>> x:torch.Tensor (must be set as require_grad = True)
#     >>> scaling_factor = torch.tensor([1.])
#     >>> epi = torch.tensor([1e-5])
#     >>> div_me_when_g_too_small = torch.tensor([1e-3])

#     retur type: torch.Tensor
#     '''
#     @staticmethod
#     def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
#         #I tried to write like:
#         #def forward(ctx, x:torch.Tensor, scaling_factor:float = torch.tensor([1.]), \
#         #               epi=torch.tensor([1e-5]), \
#         #               div_me_when_g_too_small = torch.tensor([1e-3]))->torch.Tensor:
#         #but python grammar punched me.
#         x_in:torch.Tensor = args[0]
#         scaling_factor = args[1]
#         epi = args[2]
#         div_me_when_g_too_small = args[3]
#         # the default values:
#         # scaling_factor = torch.tensor([1.])
#         # epi = torch.tensor([0.00001])
#         # div_me_when_g_too_small = torch.tensor([0.001])
#         # the definition of the 3 param are different from the previous version
        
#         if len(x_in.shape)!=2:
#             raise Exception("XModificationFunction only accept rank-2 tensor. The shape should be[batch, something]")

#         length:torch.Tensor = x_in.mul(x_in).sum(dim=1,).sqrt()
#         too_small:torch.Tensor = length.le(epi)
#         div_me = too_small.logical_not()*length + too_small*div_me_when_g_too_small
#         div_me = div_me.unsqueeze(dim=1)
#         div_me = div_me.to(x_in.dtype)
#         x_out:torch.Tensor = x_in/div_me

#         scaling_factor = scaling_factor.to(x_in.dtype)
#         if 1.!=scaling_factor.item():
#             x_out *= scaling_factor
#             pass
        
#         return x_out

#     @staticmethod
#     def backward(ctx, g):
#         #super().backward()
#         return g, None, None, None

#     pass  # class



# # '''dtype adaption.'''
# # scaling_factor = torch.tensor([1.], dtype=torch.float64)
# # epi=torch.tensor([1e-5], dtype=torch.float32)
# # div_me_when_g_too_small = torch.tensor([1e-3], dtype=torch.float16)
# # a = torch.tensor([[0.]], dtype=torch.float16)
# # original_dtype = a.dtype
# # print(XModificationFunction.apply(a,scaling_factor,epi,div_me_when_g_too_small))
# # print("should be ", original_dtype)
# # fds=432

# # '''when x_in is too small.'''
# # scaling_factor = torch.tensor([1.])
# # epi=torch.tensor([1e-5])
# # div_me_when_g_too_small = torch.tensor([1e-3])
# # input = torch.tensor([[0.0000012]])
# # print(XModificationFunction.apply(input,scaling_factor,epi,div_me_when_g_too_small))
# # print("should be ", input/div_me_when_g_too_small)

# # '''when x_in is NOT too small.'''
# # scaling_factor = torch.tensor([1.])
# # epi=torch.tensor([1e-5])
# # div_me_when_g_too_small = torch.tensor([1e-3])
# # input = torch.tensor([[0.12]])
# # print(XModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small))
# # print("should be 1.")
# # fds=432

# # '''The shape is [batches, inside a batch]. Computation is limited inside each batch.'''
# # scaling_factor = torch.tensor([1.])
# # epi=torch.tensor([1e-5])
# # div_me_when_g_too_small = torch.tensor([1e-3])
# # input = torch.tensor([[0.12, 0.12]])
# # print(XModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small))
# # print("should be 0.7, 0.7")

# # input = torch.tensor([[0.12], [0.12]])
# # print(XModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small))
# # print("should be 1., 1.")


# # input = torch.tensor([[0.1, 0.173], [0.12, 0.12]])
# # print(XModificationFunction.apply(input, scaling_factor,epi,div_me_when_g_too_small))
# # print("should be 0.5, 0.86, 0.7, 0.7")
# # fds=432

# # '''Other 3 input besides the main input x.'''
# # input = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# # print(XModificationFunction.apply(input, torch.tensor([1.]), torch.tensor([0.001]),torch.tensor([0.1])))
# # print("should be 0.001, 0.001, 0.7, 0.7")

# # input = torch.tensor([[0.0001, 0.0001], [0.12, 0.12]])
# # print(XModificationFunction.apply(input, torch.tensor([2.]), torch.tensor([0.001]),torch.tensor([0.1])))
# # print("should be 0.002, 0.002, 1.4, 1.4")
# # fds=432




# class XModification(torch.nn.Module):
#     r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
#     To access the learning rate, you usually need some thing like:
#     lr:float = optimizer.param_groups[0]["lr"]
#     """
#     def __init__(self, scaling_factor:float = 1., \
#                        epi=1e-5, \
#                        div_me_when_g_too_small = 1e-3, \
#                         *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.scaling_factor = torch.nn.Parameter(torch.tensor([scaling_factor]), requires_grad=False)
#         self.scaling_factor.requires_grad_(False)
#         self.epi=torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
#         self.epi.requires_grad_(False)
#         self.div_me_when_g_too_small = torch.nn.Parameter(torch.tensor([div_me_when_g_too_small]), requires_grad=False)
#         self.div_me_when_g_too_small.requires_grad_(False)
#         #raise Exception("untested")
#         pass
#     def forward(self, x:torch.Tensor)->torch.Tensor:
#         # If you know how pytorch works, you can comment this checking out.

#         if len(x.shape)!=2:
#             raise Exception("XModification only accept rank-2 tensor. The shape should be[batch, something]")

#         #forward(ctx, x:torch.Tensor, scaling_factor:torch.Tensor, epi=torch.Tensor, \
#         #div_me_when_g_too_small:torch.Tensor)->torch.Tensor:
#         return XModificationFunction.apply(x, self.scaling_factor, self.epi, \
#                                                    self.div_me_when_g_too_small)
#     def set_scaling_factor(self, scaling_factor:float)->None:
#         the_device = self.scaling_factor.device
#         the_dtype = self.scaling_factor.dtype
#         self.scaling_factor.data = torch.tensor([scaling_factor], device=the_device, dtype=the_dtype)
#         self.scaling_factor.requires_grad_(False)
#         pass
#     def set_epi(self, epi:float)->None:
#         the_device = self.epi.device
#         the_dtype = self.epi.dtype
#         self.epi.data = torch.tensor([epi], device=the_device, dtype=the_dtype)
#         self.epi.requires_grad_(False)
#         pass
#     def set_div_me_when_g_too_small(self, div_me_when_g_too_small:float)->None:
#         the_device = self.div_me_when_g_too_small.device
#         the_dtype = self.div_me_when_g_too_small.dtype
#         self.div_me_when_g_too_small.data = torch.tensor([div_me_when_g_too_small], device=the_device, dtype=the_dtype)
#         self.div_me_when_g_too_small.requires_grad_(False)
#         pass

#     def extra_repr(self) -> str:
#         return f'scaling_factor={self.scaling_factor.item():.4e}, epi={self.epi.item():.4e}, div_me_when_g_too_small={self.div_me_when_g_too_small.item():.4e}'

#     pass#end of class
# #No tests currently.



# class DoubleModification(torch.nn.Module):
#     r"""Remember to set learning rate every iteration(or at least when learning rate is changed.)
#     To access the learning rate, you usually need some thing like:
#     lr:float = optimizer.param_groups[0]["lr"]
#     """
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.xmo = XModification()
#         self.gramo = GradientModification()
#         pass
#     def forward(self, x:torch.Tensor)->torch.Tensor:
#         x = self.xmo(x)
#         x = self.gramo(x)
#         return x
#     pass #end of class.




class ReLU_with_offset(torch.nn.Module):
    r"""y = max(1, x)
    """
    def __init__(self, offset:float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.offset = torch.nn.Parameter(torch.tensor(offset), requires_grad=False)
        #raise Exception ('untested.')
        pass
    def forward(self, x:torch.Tensor)->torch.Tensor:
        #tensor_one = torch.tensor([1.], dtype=x.dtype, device=x.device)
        result = torch.maximum(x, self.offset)
        return result
    pass #end of class.

if 'basic test' and False:
    layer = ReLU_with_offset(0.5123)
    input = torch.linspace(0.,1.,10)
    output = layer(input)
    pass





