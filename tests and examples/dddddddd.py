import torch






a = torch.tensor([1,2,3])
print(torch.rand_like(a,dtype=torch.float16))
#print(torch.rand_like(a,dtype=torch.int8))#"check_uniform_bounds" not implemented for 'Char'
#print(torch.rand_like(a,dtype=torch.uint8))#"check_uniform_bounds" not implemented for 'Byte'
print(torch.randn_like(a,dtype=torch.float16))
#print(torch.randn_like(a,dtype=torch.int8))#"normal_kernel_cpu" not implemented for 'Char'
#print(torch.randn_like(a,dtype=torch.int8,device="cuda"))#"normal_kernel_cuda" not implemented for 'Char'
#print(torch.randn_like(a,dtype=torch.uint8))#"normal_kernel_cpu" not implemented for 'Byte'
#print(torch.randn_like(a,dtype=torch.uint8,device="cuda"))#"normal_kernel_cuda" not implemented for 'Byte'
print(torch.randint_like(a,-10,10,dtype=torch.float16))
print(torch.randint_like(a,-10,10,dtype=torch.int8))
print(torch.randint_like(a,0,256,dtype=torch.uint8))
fds=432







def float_precision_test(big:float, small:float):
    print(f"{big} + {small} is {big+small}")
    pass
if 'basic test' and True:
    #float_precision_test(1,1e-15)
    #float_precision_test(1,1e-16)
    pass

def float_precision_test_torch_tensor(big:float, small:float, dtype = torch.float32):
    t_big = torch.tensor(big,dtype=dtype)
    t_small = torch.tensor(small,dtype=dtype)
    print(f"{t_big.item()} + {t_small.item()} is {(t_big+t_small).item()}")
    pass
if 'fp32' and False:
    dtype = torch.float32
    float_precision_test_torch_tensor(1,1e-7,dtype=dtype)
    float_precision_test_torch_tensor(1,1e-8,dtype=dtype)
    print('-------------------------------')
    float_precision_test_torch_tensor(1,torch.rand([1])*1e-6,dtype=dtype)
    float_precision_test_torch_tensor(1,torch.rand([1])*1e-6,dtype=dtype)
    float_precision_test_torch_tensor(1,torch.rand([1])*1e-6,dtype=dtype)
    float_precision_test_torch_tensor(1,torch.rand([1])*1e-6,dtype=dtype)
    float_precision_test_torch_tensor(1,torch.rand([1])*1e-6,dtype=dtype)
    #1e-7*sqrt(dim)


if 'fp16' and False:
    dtype = torch.float16
    float_precision_test_torch_tensor(1,1e-3,dtype=dtype)
    float_precision_test_torch_tensor(1,1e-4,dtype=dtype)
    print('-------------------------------')
    float_precision_test_torch_tensor(1,torch.rand([1])*2e-2,dtype=dtype)
    float_precision_test_torch_tensor(1,torch.rand([1])*2e-2,dtype=dtype)
    float_precision_test_torch_tensor(1,torch.rand([1])*2e-2,dtype=dtype)
    float_precision_test_torch_tensor(1,torch.rand([1])*2e-2,dtype=dtype)
    float_precision_test_torch_tensor(1,torch.rand([1])*2e-2,dtype=dtype)
    pass


fds=432
    
    

a = torch.tensor([1.],dtype=torch.float32)
a += 0.0000003
print(a.item())
a = torch.tensor([1.],dtype=torch.float16)
a += 0.01
print(a.item())

fds=432


a = torch.tensor([1.,1,1])
b = torch.tensor([1.,1,3])
loss_function = torch.nn.L1Loss()
loss = loss_function(a,b)
print(loss)


a = None
b = a 
c = b is a
pass

a = torch.tensor([1.1])
a.requires_grad_()
print(a.requires_grad)


a = torch.tensor([1.1])
b = torch.tensor(1.1)
print(a.shape)
print(b.shape)


a = torch.tensor([1.1,2.2,3.3])
b = torch.tensor([1])
c = a[b]
print(c)
fds=432


a = torch.empty((0))
print(a.nelement())

fds=43

w = torch.randint(0,3,[11,6])
print(w)
temp =  w.max(dim=1,keepdim=False).indices
print(temp)
previous_index = temp.unique()
print(previous_index)







w = torch.randint(0,3,[11,6])
index:torch.Tensor = w.max(dim=1,keepdim=False).indices
print(index)
temp2 = index.tolist()
print(temp2)
temp3 = list(set(temp2))
print(temp3)
print(temp3.__len__())
previous_index = torch.tensor(temp3)
print(previous_index)

fds=432