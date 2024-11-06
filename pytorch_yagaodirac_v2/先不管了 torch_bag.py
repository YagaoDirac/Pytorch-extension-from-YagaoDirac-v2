from typing import Any, List, Tuple, Optional
import torch

#class Torch_Bag(torch.nn.Module):
class Torch_Bag():
    '''
    # @staticmethod
    # def from_list(data:List)->'Torch_Bag':
    #     result = Torch_Bag()
    #     result.mem.data = torch.tensor(data)
    #     result.len.data = torch.tensor(data.__len__())
    #     return result        
    # @staticmethod
    # def from_tensor(data:torch.Tensor)->'Torch_Bag':
    #     if data.shape.__len__()>1:
    #         raise Exception()
    #     result = Torch_Bag()
    #     result.mem.data = torch.tensor(data.detach())
    #     result.len.data = torch.tensor(data.nelement())
    #     return result        
    '''
        
    def __init__(self, data, device=None, dtype=None) -> None:
           
        factory_kwargs = {'device': device, 'dtype': dtype}
        #super().__init__()
        if data is None:
            self.mem = torch.nn.Parameter(torch.tensor([0], **factory_kwargs), requires_grad=False)
            self.len = torch.nn.Parameter(torch.tensor(0, **factory_kwargs), requires_grad=False)
        elif data is List:
            self.mem = torch.nn.Parameter(torch.tensor(data, **factory_kwargs), requires_grad=False)
            self.len = torch.nn.Parameter(torch.tensor(data.__len__(), **factory_kwargs), requires_grad=False)
        elif data is torch.Tensor:
            if data.shape.__len__()>1:
                raise Exception()
            pass
            self.mem = torch.nn.Parameter(torch.tensor(data, **factory_kwargs), requires_grad=False)
            self.len = torch.nn.Parameter(torch.tensor(data.nelement(), **factory_kwargs), requires_grad=False)
        else:
            raise Exception("data must be None, List or torch.Tensor")
        pass
    def set_cap(self, set_to:int):
        if set_to == self.mem.nelement():
            return
        new_mem = torch.tensor(set_to, device=self.mem.device, dtype=self.mem.dtype)
        min_shape = torch.min(torch.tensor(set_to, dtype=self.len.dtype),torch.tensor(self.mem.nelement()))#.item()
        new_mem[:min_shape] = self.mem[:min_shape]
        if set_to<self.len:
            self.len.data = torch.tensor(set_to, device=self.len.device)
            #self.len.data[()] = set_to
            #self.len.data[...] = set_to
            pass
        pass
    
    def cap(self)->int:
        return self.mem.nelement()
    
    
    继续。
    
    def __check_cap(self, extra = 1, allocate_more_mem_when_needed = True)->bool:
        if index>self.len:
            return False
        if self.cap()<=self.len:
            if not allocate_more_mem_when_needed:
                return False
            else:
                self.set_cap(self.cap()*2)
                pass
            pass
    
    def insert(self, element:Any, index:int, allocate_more_mem_when_needed = True)->bool:
        if index>self.len:
            return False
        if self.cap()<=self.len:
            if not allocate_more_mem_when_needed:
                return False
            else:
                self.set_cap(self.cap()*2)
                pass
            pass
        
        self.mem.data[self.len] = self.mem.data[index]
        self.mem.data[index] = element
        self.len.data+=1
        return True
        pass
    def append(self, element, allocate_more_mem_when_needed = True)->bool:
        if self.mem.nelement()<=self.len:
            return False
        
        self.mem.data[self.len] = element
        self.len.data+=1
        return True
        pass
    
    #to do list:insert_multiple append_multiple
    def remove(self, index:int)->Optional[Any]:
        if index>=self.len:
            return None
        result = self.mem[index]
        self.mem[index] = self.mem[self.len]
        self.len.data-=1
        return result
    def pop(self, index:int)->Optional[Any]:
        if 0 == self.len:
            return None
        result = self.mem[self.len-1]
        self.len.data-=1
        return result
    def pop_front(self, index:int)->Optional[Any]:
        if 0 == self.len:
            return None
        result = self.mem[0]
        self.mem[0] = self.mem[self.len-1]
        self.len.data-=1
        return result
    def extra_repr(self):
        s = f'len:{self.len.data.item()}, {str(self.mem.data)}'
        return s
    pass#end of class


a:torch.nn.parameter.Parameter = torch.nn.Parameter(torch.tensor(123,device='cuda'), requires_grad=False)
print(a.device)
a.data = torch.tensor([321])
print(a.device)


if 'create test' and True:
    a = Torch_Bag([3,4,3,2,1])
    a.set_cap(9)
    b = Torch_Bag(torch.tensor([3,4,3,2,1]))
    b.set_cap(3)
    c = Torch_Bag()
    c.set_cap(4)
    pass
    
if 'add elements test' and True:
    a:Torch_Bag = Torch_Bag.from_list([3,4,3,2,1])
    a.insert(11,1)
    a.insert(33,3)
    a.append
    
    
