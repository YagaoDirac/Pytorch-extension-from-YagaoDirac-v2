from typing import Any
import torch


        # 两个事情，
        # 测试。l1loss
        # 安全log。
        
        
        
a = torch.nn.ParameterList([torch.nn.Linear(n,n) for n in range(5)])      
print(a[-1].weight.shape)
fds=432
        

new_top_raw_weight_o = torch.tensor([0.,1, -1.])
new_top_raw_weight_o = new_top_raw_weight_o.maximum(torch.zeros([1,], dtype= new_top_raw_weight_o.dtype, device=new_top_raw_weight_o.device ))


a = torch.tensor([torch.nan, torch.inf, torch.inf*-1, 0.,1, -1.])
a.nan_to_num_(42)
print(a)
fds=432

if True:
    a = torch.tensor([torch.nan, torch.inf, torch.inf*-1, 0.,1, -1.])
    print(a.log(), "log")
    print(a.exp(), "exp")
    print(a/0., "a/0")
    print(0./a, "0/a")
    print(a+1, "+1")
    print(a+torch.inf, "+inf")
    print(a+torch.inf*-1, "-inf")
    print(a*torch.inf, "*inf")
    print(a/torch.inf, "a/inf")
    print(torch.inf/a, "inf/a")
    print(a.gt(torch.inf), "a>inf")
    print(a.eq(torch.inf), "a==inf")
    print(a.lt(torch.inf), "a<inf")
    print(a+torch.nan, "+nan")
    print(a*torch.nan, "*nan")
    print(a/torch.nan, "/nan")
    print(torch.nan/a, "nan/")
    print(a.gt(torch.nan), "a>nan")
    print(a.eq(torch.nan), "a==nan")
    print(a.lt(torch.nan), "a<nan")
    print(a)
    print(a.isnan())
    print(a.isinf())
    print(a.isposinf())
    print(a.isneginf())
    print(a.nan_to_num(42))
    print(a.nan_to_num(42)*0.)
    print(a.nan_to_num(42)*0.+1)
    flag_nan = a.isnan().logical_or(a.isinf())
    print(flag_nan)
    b = a.nan_to_num(111)
    print(b)
    c = flag_nan*222.+flag_nan.logical_not()*b
    print(c)
    
    pass











a = torch.tensor([torch.inf])
b = a/a
print(b)
fds=432


a = torch.tensor([42., torch.nan])
print(a)
flag_nan = torch.isnan(a)
print(flag_nan)
a.nan_to_num_(0.)
print(a)
a = flag_nan*123+flag_nan.logical_not()*a
print(a)
fds=432



a = torch.rand([5])*torch.tensor([-20.]).abs()*0.1+torch.tensor([-20.])
print(a)
fds=432




a = torch.tensor([torch.nan],dtype=torch.float16).cuda()
print(a)
a = a.nan_to_num(1.)
print(a)
fds=432


a = torch.linspace(-10.,-1000.,10, dtype=torch.float16, requires_grad=True).cuda()
print(a)
b = a.exp()
print(b)
c = a.softmax(dim=0)
print(c)
g_in = torch.ones_like(c)
torch.autograd.backward(c, g_in, inputs=a)
print(a.grad)

fds=432


a = torch.tensor([True, False])
b = a.all()
print(b)
fds=432














# @staticmethod
#     def apply_ghost_weight(the_tensor : torch.tensor, \
#             ghost_weight_length:torch.nn.parameter.Parameter, \
#             out_iota: torch.tensor, the_max_index: torch.tensor)->torch.tensor:
#         with torch.no_grad():
#             the_tensor[out_iota, the_max_index] = ghost_weight_length
#             return the_tensor











a = torch.linspace(0,11,12).view([3,4])
print(a)
b = a[[0,1,2], [2,3,3]]
print(b)
c = a[None,[2,3,3]]
print(c)

fds=432




# a = torch.tensor([[1.,2],[1.,3],])
# mean = a.mean(dim=1,keepdim=True)
# print(mean)
# a = a-mean
# print(a)
# fds=432

# a = torch.tensor([False])
# if not a:
#     print("inside")
# fds=432


a = torch.nn.Parameter(torch.tensor([1.], requires_grad=False))#True
print(a, 11)
a.requires_grad_(False)
print(a, 12)
b = torch.nn.Parameter(torch.tensor([1.], requires_grad=True))
print(b, 21)
b.requires_grad_(False)
print(b, 22)
c = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
print(c, 3)
d = torch.nn.Parameter(torch.tensor([1.], requires_grad=True), requires_grad=False)
print(d, 4)
fds=43






a = torch.nn.Parameter(torch.tensor([1.,], dtype=torch.float16))
a = a.cuda()
print(a)
a.data = torch.tensor([2.,], dtype=a.dtype, device=a.device)
print(a)

fds=432









a = torch.tensor([1.])
a = a.to(torch.float16)
print(a.dtype)
fds=432



# import sys

# def LINE():
#     return sys._getframe(1).f_lineno

# print('This is line', LINE())  


 

from inspect import currentframe, getframeinfo
print(getframeinfo(currentframe()).lineno)

