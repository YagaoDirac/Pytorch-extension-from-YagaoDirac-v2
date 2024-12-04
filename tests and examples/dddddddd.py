import torch




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