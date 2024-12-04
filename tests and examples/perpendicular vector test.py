from typing import Union
import sys
from pathlib import Path

import math
import torch

sys.path.append(str(Path(__file__).parent.parent))
#print(sys.path[-1])
from pytorch_yagaodirac_v2.torch_vec import torch_vec
from pytorch_yagaodirac_v2.Util import vector_length_norm
from pytorch_yagaodirac_v2.training_ended_sound import play_noise
from pytorch_yagaodirac_v2.log_print import Log


# def get_rand(DIM:int)->torch.Tensor:
#     temp1 = torch.rand([1,DIM])-0.5
#     temp2 = temp1*torch.pi
#     temp3 = temp2.cos()
#     #flag = temp3.gt(0.)
#     #the_abs = temp3.abs()
#     #temp4 = the_abs.pow(0.5)
#     #temp5 = temp4*(flag*2.-1.)
#     temp6 = vector_length_norm(temp3).squeeze(dim=0)
#     angle = torch.atan2(temp6[0],temp6[1])*180/torch.pi
#     return temp6

# #table = torch.tensor([0.,0.1,0.2,0.4,0.66,1.])#less150?
# table = torch.tensor([0.,0.2,0.71,0.89,1.])#less150?
# #table = torch.tensor([0.,0.1,0.2,0.5,0.66,1.])
# #table = torch.tensor([0.,0.1,0.2,0.5,0.66,1.])
# temp1 = torch.rand([360,2])
# temp2 = table.quantile(temp1.reshape(-1)).reshape(temp1.shape)
# flag = torch.randint_like(temp2,0,2)*2-1
# temp3 = temp2*flag
# temp6 = vector_length_norm(temp3).squeeze(dim=0)
# angle = torch.atan2(temp6[:,0],temp6[:,1])*180/torch.pi
# sorted_angles = angle.sort().values
# reshaped_sorted_angles = sorted_angles.reshape([36,10])
# aaaadiff = reshaped_sorted_angles[1:,0]-reshaped_sorted_angles[:-1,0]
# aaaadiff_mirrored = aaaadiff+aaaadiff.flip(dims=(0,))
# print(table)
# print(aaaadiff_mirrored)
# fds=432


# n = 360
# DIM = 111
# buff = torch_vec(DIM,n)

# for _ in range(n):
#     already_pushed = False
#     for _ in range(1000):
#         temp1 = torch.rand([DIM])
#         length = (temp1*temp1).sum()
#         if length<=1. and length>0.2:
#             buff.pushback(temp1)
#             already_pushed = True
#             break
#     if not already_pushed:
#         temp1 = torch.rand([DIM])
#         buff.pushback(temp1)
# normalized = vector_length_norm(buff.get_useful())
# angle = torch.atan2(normalized[:,2],normalized[:,33])*180/torch.pi
# sorted_angles = angle.sort().values
# reshaped_sorted_angles = sorted_angles.reshape([n//10,10])
# aaaadiff = reshaped_sorted_angles[1:,0]-reshaped_sorted_angles[:-1,0]
# aaaadiff_mirrored = aaaadiff+aaaadiff.flip(dims=(0,))
# print(aaaadiff_mirrored)
# fds=432
    



# def debug__get_rand(DIM:int)->torch.Tensor:
#     temp1 = torch.rand([1,DIM])-0.5
#     temp2 = temp1*torch.pi
#     temp3 = temp2.cos()
#     #flag = temp3.gt(0.)
#     #the_abs = temp3.abs()
#     #temp4 = the_abs.pow(0.5)
#     #temp5 = temp4*(flag*2.-1.)
#     temp6 = vector_length_norm(temp3).squeeze(dim=0)
#     angle = torch.atan2(temp6[0],temp6[1])*180/torch.pi
#     return angle
# len = 10
# DIM = 2
# buff = torch.empty([len], dtype=torch.float32)
# for i in range(len):
#     buff[i] = debug__get_rand(DIM)
#     pass
# temp = buff.sort()
# print(temp)
# fds=432


def get_rand_point_on_hyper_sphere(n:int, dim:int, device = None)->torch.Tensor:
        
    buff = torch.rand([n, DIM], device=device)
    for _ in range(0):
        length = (buff*buff).sum(dim=1,keepdim=True)
        #print(length)
        flag_useful = length.lt(1.).logical_and(length.gt(0.2))
        if flag_useful.all():
            break
        extra_rand = torch.rand([n, DIM], device=device)
        buff = flag_useful*buff+flag_useful.logical_not()*extra_rand
        pass
    
    for i in range(buff.shape[0]):
        for _ in range(1000):
            length = (buff[i]*buff[i]).sum()
            flag_useful = length.lt(1.).logical_and(length.gt(0.2))
            if flag_useful:
                break
            buff[i] = torch.rand([DIM], device=device)
            pass
        pass
    normalized = vector_length_norm(buff)
    flag = torch.randint_like(normalized,0,2, device=device)*2-1
    signed = normalized*flag
    return signed
if 'basic test' and False:
    n = 360
    DIM = 22
    signed = get_rand_point_on_hyper_sphere(n,DIM)
    angle = torch.atan2(signed[:,0],signed[:,1])*180/torch.pi
    sorted_angles = angle.sort().values
    reshaped_sorted_angles = sorted_angles.reshape([n//10,10])
    aaaadiff = reshaped_sorted_angles[1:,0]-reshaped_sorted_angles[:-1,0]
    aaaadiff_mirrored = aaaadiff+aaaadiff.flip(dims=(0,))
    print(aaaadiff_mirrored)
    pass
    









is_cuda = True
device:Union[str|None]
if is_cuda:
    device = "cuda"
else:
    device = None
    pass
#tolerence = math.cos(math.radians(60.))#0.5 for 60 degrees.
tolerence = math.cos(math.radians(75.))#0.25 for 75 degrees.
#tolerence = math.cos(math.radians(88.))#0.0348 for 88 degrees.
DIM = 20
max_n_ng_in_a_row = 5000

log = Log(f"vector perpendicular test max_n_ng_in_a_row {max_n_ng_in_a_row}  tolerence {tolerence:.4f}  DIM {DIM}")
log.log_print(f"is_cuda: {is_cuda}, max_n_ng_in_a_row: {max_n_ng_in_a_row}, tolerence: {tolerence:.4f}, DIM: {DIM}")
log.log_print(f"is_cuda: {is_cuda}, max_n_ng_in_a_row: {max_n_ng_in_a_row}, tolerence: {tolerence:.4f}, DIM: {DIM}")

'''
tolerence:DIM
60 deg, 0.5:2(2)3(4)6(14)10(39)15(111)20(266)30() (5k failures before stop)
75 deg, 0.25:5(5)10(10)15(15 3x)20()30(40)50(99 needs 3 test.) 100(>700) (5k failures before stop)
88 deg, 0.0349:2(2)3(3)6(4)10(4)20(5?needs more test.)30(3?) (1k or 5k failures before stop)
'''
for _ in range(6):
    #print(tolerence, "tolerence")
    cont = torch_vec(DIM)
    if is_cuda:
        cont = cont.cuda()
        pass
    cont.pushback(vector_length_norm(torch.randn([1,DIM], device=device)).squeeze(dim=0))
    #print(cont)

    n_ng_in_a_row = 0
    while True:
        test_this = vector_length_norm(get_rand_point_on_hyper_sphere(1,DIM, device=device)).squeeze(dim=0)
        #print(test_this, "test_this")
        useful = cont.get_useful()
        #print(useful, "useful")
        temp1 = useful*test_this
        #print(temp1, "temp1")
        temp2 = temp1.sum(dim=1,keepdim=True)
        #print(temp2, "temp2")
        flag_perpendicular_enough = temp2.abs().lt(tolerence)
        #print(flag_perpendicular_enough, "flag_perpendicular_enough")
        if flag_perpendicular_enough.all():
            cont.pushback(test_this)
            if cont.__len__()>40 or cont.__len__()%5 == 0:
                log.log_print(f"{cont.__len__()}:{n_ng_in_a_row}  ", end="")
                #print(f"{cont.__len__()}:{n_ng_in_a_row}  ", end="")
                pass
            n_ng_in_a_row = 0
        else:
            n_ng_in_a_row+=1
            if n_ng_in_a_row>max_n_ng_in_a_row:
                log.log_print(f"tolerence: {tolerence:.4f}, DIM: {DIM},  final result  {cont.__len__()}")
                #print(f"tolerence: {tolerence:.4f}, DIM: {DIM},  final result  {cont.__len__()}")
                break
        pass
    pass
log.log_print("------------------------FINISH---------------------------")
log.log_print("------------------------FINISH---------------------------")
log.log_print("------------------------FINISH---------------------------")
play_noise()
    