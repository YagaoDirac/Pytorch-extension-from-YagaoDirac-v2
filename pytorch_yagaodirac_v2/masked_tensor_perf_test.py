import torch

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from timeit_yagaodirac import timeit


size=(100,100)
device='cuda'
mask = torch.randint(size=size,low=0, high=2, dtype=torch.bool, device = device)
def make_masked()->torch.masked.MaskedTensor:
    data = torch.rand(size=size,device=device)
    masked_tensor = torch.masked.as_masked_tensor(data,mask)
    return masked_tensor



# t1 = make_masked()
# print(t1.data.shape)
# print(t1.get_mask().shape)
# t2 = make_masked()
# print(t2.data.shape)
# print(t2.get_mask().shape)
# a = t1+t2
# print(a.data.shape)
# print(a.get_mask().shape)





if "main test" and False:

    def func_empty():
        t1 = make_masked()
        t2 = make_masked()
        for _ in range(10):
            pass
        pass

    time_of_empty,_ = timeit(func_empty, time_at_most=2.)




    def func_masked():
        t1 = make_masked()
        t2 = make_masked()
        for _ in range(10):
            t1+t2
            t1+t2
            t1+t2
            t1+t2
            t1+t2
            t1+t2
            t1+t2
            t1+t2
            t1+t2
            t1+t2
            pass
        pass

    time_of_masked,_ = timeit(func_masked, time_at_most=2.)







    def func_normal():
        t1 = make_masked()
        t2 = make_masked()
        for _ in range(10):
            t1.data+t2.data
            t1.data+t2.data
            t1.data+t2.data
            t1.data+t2.data
            t1.data+t2.data
            t1.data+t2.data
            t1.data+t2.data
            t1.data+t2.data
            t1.data+t2.data
            t1.data+t2.data
            pass
        pass

    time_of_normal,_ = timeit(func_normal, time_at_most=2.)


    print("time_of_masked", time_of_masked-time_of_empty)
    print("time_of_normal", time_of_normal-time_of_empty)









mask = torch.randint(size=(3,),low=0, high=2, dtype=torch.bool, device = device)
data = torch.rand(size=(3,2),device=device)
masked_tensor = torch.masked.as_masked_tensor(data,mask)
print((masked_tensor+masked_tensor).shape)
print((masked_tensor+masked_tensor).get_mask().shape)




