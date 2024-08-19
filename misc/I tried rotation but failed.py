from typing import Any
import math
import torch
#from ../pytorch_yagaodirac_v2 import util
import sys
sys.path.append(sys.path[0][:-5])
from pytorch_yagaodirac_v2.util import vector_length_norm
#vector_length_norm







# def protect_rotation_matrix(input:torch.Tensor, epi = 0.000001):#->torch.Tensor:
#     if len(input.shape)!=2:
#         raise Exception("send matrix here.")
#     dim = input.shape[0]
#     if dim!=input.shape[1]:
#         raise Exception("It must be square.")
    
#     with torch.no_grad():
        
        
#         two_triagles = (input-input.T)*0.5
#         diagonal = input.mul(torch.eye(dim))
#         output_raw = two_triagles+diagonal
        
        
#         length_of_output_raw_b = output_raw.mul(output_raw).sum(dim=1,keepdim=False).sqrt()
#         epi_tensor = torch.tensor([epi], device=length_of_output_raw_b.device, dtype=length_of_output_raw_b.dtype)
#         length_of_output_raw_safe_b = length_of_output_raw_b.maximum(epi_tensor)
#         sqrt_of_length_b = length_of_output_raw_safe_b.sqrt()
#         #result = input/length_of_input_safe_b#.unsqueeze(dim=1)
#         output = output_raw/sqrt_of_length_b.unsqueeze(dim=1)/sqrt_of_length_b.unsqueeze(dim=0)
        
#         raise Exception("test not passed..")
#         fds=432
    
#     #output = vector_length_norm(output_raw)#shape is intentional.
    
#     return output
# raw_from_randn = torch.tensor([[0.5,2],[3.,4]])#randn([2,2])
# rotation_matrix = protect_rotation_matrix(raw_from_randn)
# print(rotation_matrix[0].mul(rotation_matrix[0]).sum())
# print(rotation_matrix[1].mul(rotation_matrix[1]).sum())
# print(rotation_matrix.T[0].mul(rotation_matrix.T[0]).sum())
# print(rotation_matrix.T[1].mul(rotation_matrix.T[1]).sum())
# unit_length_vec = vector_length_norm(torch.randn([1,2])).unsqueeze(dim=2)
# print(unit_length_vec.mul(unit_length_vec).sum(), "unit_length_vec")
# after_rotation = rotation_matrix.matmul(unit_length_vec).squeeze(dim=2)
# print(after_rotation.mul(after_rotation).sum())
# length_after_rotation = after_rotation.mul(after_rotation).sum(dim=1)

# fds=432

#no, this is too hard
# dim = 3
# #raw_matrix = torch.tensor([[1.,2,3],[4,5,6],[7,8,9]])
# raw_matrix_d_d = torch.tensor([[1.,2,0],[1,0,0],[0,1,0]])
# #result_of_matmul_without_eye = result_of_matmul-result_of_matmul*torch.eye(3)
# length_of_raw_row_d_1 = raw_matrix_d_d.mul(raw_matrix_d_d).sum(dim=1, keepdim=True).sqrt()#unprotected.
# length_norm_ed_raw_d_d = raw_matrix_d_d/length_of_raw_row_d_1#unprotected.
# print(length_norm_ed_raw_d_d)

# result_of_matmul_d_d = length_norm_ed_raw_d_d.matmul(length_norm_ed_raw_d_d.T)
# result_of_matmul_without_diagonal_d_d = result_of_matmul_d_d-result_of_matmul_d_d*torch.eye(dim)
# result_of_matmul_without_diagonal_d_d_1 = result_of_matmul_without_diagonal_d_d.unsqueeze(dim=2)
# print(result_of_matmul_without_diagonal_d_d_1)
# a = result_of_matmul_without_diagonal_d_d_1.mul(length_norm_ed_raw_d_d)
# print(a)
# fds=432





dim = 3
#raw_matrix_d_d = vector_length_norm(torch.tensor([[1.,2,3],[4,5,6],[7,8,9]]))
#raw_matrix_d_d = vector_length_norm(torch.tensor([[1.,2,0],[1,0,0],[0,1,0]]))
raw_matrix_d_d = vector_length_norm(torch.tensor([[1.,0.1,0],[0,1,0],[0,0,1]]))
temp_core_raw_d = raw_matrix_d_d.sum(dim=0,keepdim=False)
core_1_d = vector_length_norm(temp_core_raw_d[None,:])
print(temp_core_raw_d)
print(core_1_d)

debug_factor_d = raw_matrix_d_d.mul(core_1_d).sum(dim=1,keepdim=True)
debug_1 = raw_matrix_d_d[0].mul(core_1_d.squeeze(dim=0)).sum()

the_projection_d_d = raw_matrix_d_d.mul(core_1_d).sum(dim=1,keepdim=True)*core_1_d
#the_projection_d = the_projection_1_d.squeeze(dim=0)
minus_the_projection_d_d = raw_matrix_d_d-the_projection_1_d
minus_the_projection__len_norm_ed__d_d = vector_length_norm(minus_the_projection_d_d)

temp_axis = torch.zeros([dim])
temp_axis[0] = 1.
temp_core = torch.ones_like(temp_axis)/torch.sqrt(torch.tensor([dim]))
the_cos = torch.sqrt(torch.tensor([1./float(dim)]))
the_sin = (the_cos.mul(the_cos)*-1.+1).sqrt()



fds=432













a1 = torch.rand([1,])*torch.pi*2
r1 = torch.tensor([
    [a1.cos().item(), a1.sin().item(), 0],
    [-a1.sin().item(), a1.cos().item(), 0],[0,0,1],])
print(r1.mul(r1).sum(dim=1))
print(r1.mul(r1).sum(dim=0))
print(r1)

v_before_rot = vector_length_norm(torch.randn([1,3]))
print(v_before_rot.mul(v_before_rot).sum(), v_before_rot, "v_before_rot")
v1_after_rot = v_before_rot.matmul(r1)
print(v1_after_rot.mul(v1_after_rot).sum(), v1_after_rot, "v1_after_rot")


a2 = torch.rand([1,])*torch.pi*2
r2 = torch.tensor([
    [a2.cos().item(), 0,a2.sin().item()],
    [0,1,0,],
    [-a2.sin().item(),0, a2.cos().item()],])
print(r2.mul(r2).sum(dim=1))
print(r2.mul(r2).sum(dim=0))
print(r2)

v_before_rot = vector_length_norm(torch.randn([1,3]))
print(v_before_rot.mul(v_before_rot).sum(), v_before_rot, "v_before_rot")
v1_after_rot = v_before_rot.matmul(r2)
print(v1_after_rot.mul(v1_after_rot).sum(), v1_after_rot, "v1_after_rot")


r_combined = r1.matmul(r2)
print(r_combined.mul(r_combined).sum(dim=1))
print(r_combined.mul(r_combined).sum(dim=0))
print(r_combined)

v_before_rot = vector_length_norm(torch.randn([1,3]))
print(v_before_rot.mul(v_before_rot).sum(), v_before_rot, "v_before_rot")
v1_after_rot = v_before_rot.matmul(r_combined)
print(v1_after_rot.mul(v1_after_rot).sum(), v1_after_rot, "v1_after_rot")


fds=432












batch = 3
in_features = 2
out_features = 10
weight_o_i = vector_length_norm(torch.randn([out_features, in_features]))
debug_temp_length_weight_o_i = weight_o_i.mul(weight_o_i).sum(dim=1).sqrt()

x_b_i_1 = vector_length_norm(torch.randn([batch, in_features])).unsqueeze(dim=2)
debug_temp_length_x_b_i_1 = x_b_i_1.mul(x_b_i_1).sum(dim=1).sqrt()

output_b_o = weight_o_i.matmul(x_b_i_1).squeeze(dim=2)
debug_temp_length_output_b_o = output_b_o.mul(output_b_o).sum(dim=1).sqrt()


fds=432














# loss_function = torch.nn.MSELoss()
# # pred = torch.tensor([2., 0.])
# # target = torch.tensor([0., 0.])
# batch = 100
# out_feature = 20
# pred = torch.rand([batch, out_feature])-0.5
# target = torch.rand([batch, out_feature])-0.5
# print(loss_function(pred, target))
# fds=432





in_features = 3000
out_features = 400
for _ in range(5):
    #input_temp = torch.rand([1,in_features, 1])
    input_temp = torch.rand([1,in_features])
    length_of_input_temp = input_temp.mul(input_temp).sum().sqrt()
    input = input_temp/length_of_input_temp
    debug_checks_the_length = input.mul(input).sum()
    
    layer = torch.nn.Linear(in_features, out_features, False)
    the_factor = math.sqrt(3.)/math.sqrt(out_features)#*in_features)
    layer.weight.data = (torch.rand([out_features, in_features])*2.-1.)*the_factor
    #w = (torch.rand([out_features, in_features])*2.-1.)*the_factor
    output:torch.Tensor = layer(input)
    print(output.mul(output).sum())
fds=432



# w = torch.empty(6, 600)
# torch.nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
# print(w.abs().amax(), w.abs().amin())
# w = torch.empty(6, 600)
# torch.nn.init.kaiming_uniform_(w, mode='fan_out', nonlinearity='relu')
# print(w.abs().amax(), w.abs().amin())
# w = torch.empty(24, 2400)
# torch.nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
# print(w.abs().amax(), w.abs().amin())
# w = torch.empty(24, 2400)
# torch.nn.init.kaiming_uniform_(w, mode='fan_out', nonlinearity='relu')
# print(w.abs().amax(), w.abs().amin())
# w = torch.empty(24, 2400)
# torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
# print(w.abs().amax(), w.abs().amin())
# torch.nn.Linear
# fds=423
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

