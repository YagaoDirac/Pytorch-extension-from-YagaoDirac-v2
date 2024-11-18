import torch

#加一个get last，好像又不用了。


class Torch_Ring_buffer_1D(torch.nn.Module):
    def __init__(self, cap:int, dimention:int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.start_included = torch.nn.Parameter(torch.tensor(0, **factory_kwargs), requires_grad=False)
        self.end_excluded = torch.nn.Parameter(torch.tensor(0, **factory_kwargs), requires_grad=False)
        self.length = torch.nn.Parameter(torch.tensor(0, **factory_kwargs), requires_grad=False)
        self.data = torch.nn.Parameter(torch.empty([cap, dimention], **factory_kwargs), requires_grad=False)
        pass
    
    def __len__(self)->int:
        return self.length.item()
    
    def get_rearranged(self)->torch.Tensor:
        with torch.no_grad():
            if 0 == self.length:
                return torch.empty_like(self.data)
            if self.start_included<self.end_excluded:
                #the simple case
                result = torch.empty_like(self.data)
                result[:self.length] = self.data[self.start_included:self.end_excluded]
                return result
            else:
                #complex case.
                result = torch.empty_like(self.data)
                start_to_tail = self.get_max_cap_as_tensor() - self.start_included
                result[:start_to_tail] = self.data[self.start_included:]
                result[start_to_tail:self.length] = self.data[:self.end_excluded]
                return result
        #end of function.
    
    def rotate_to_preferred(self):
        #to do:shrink?
        self.data.data = self.get_rearranged()
        self.start_included.data[...]= 0
        self.end_excluded.data = self.length.detach().clone()
        self.end_excluded.data = self.end_excluded%self.get_max_cap_as_tensor()#optimizable.
        pass
        
    def full(self)->torch.Tensor:
        return self.length.eq(self.data.shape[0])
    def get_max_cap_as_tensor(self)->torch.Tensor:
        result = torch.tensor(self.data.shape[0],device=self.data.device)
        return result
    def cap(self)->int:
        return self.data.shape[0]
        
    def pushback(self, data:torch.Tensor, overwrite = False)->bool:
        with torch.no_grad():
            if self.full():
                if not overwrite:
                    return False
                self.data[self.end_excluded] = data.clone()
                self.start_included.data+=1
                self.start_included.data = self.start_included%self.get_max_cap_as_tensor()
                self.end_excluded.data = self.start_included.detach().clone()
                return True
            self.length.data+=1
            self.data[self.end_excluded] = data.clone()
            self.end_excluded.data += 1
            self.end_excluded.data = self.end_excluded%self.get_max_cap_as_tensor()
            return True
        
           
    # def _pushback_partly(self, data:torch.Tensor, overwrite = False)->bool:
    #     r'''data is a shorter tensor than what is needed. This function only overwrites the provided part and left the tail unchanged. 
    #     Element count is increased.
    #     If the provided data is really shorter than the dim of the container, some useless elements are considered as real data.
    #     Make sure you really know the behavior of this function.
    #     '''
    #     with torch.no_grad():
    #         if self.full():
    #             if not overwrite:
    #                 return False
    #             self.data[self.end_excluded, :data.shape[0]] = data.clone()
    #             self.start_included.data+=1
    #             self.start_included.data = self.start_included%self.get_max_cap_as_tensor()
    #             self.end_excluded.data = self.start_included.detach().clone()
    #             return True
    #         self.length.data+=1
    #         self.data[self.end_excluded, :data.shape[0]] = data.clone()
    #         self.end_excluded.data += 1
    #         self.end_excluded.data = self.end_excluded%self.get_max_cap_as_tensor()
    #         return True
        
    def extra_repr(self):
        return f"len:{self.__len__()}, data:{self.get_rearranged()[:self.__len__()]}"
        #return super().extra_repr()
        
    def ______a():
        '''
        empty,full
        to do list:
        re cap
        shrink
        push at any position.
        
        '''
    
    pass#end of class

if '_pushback_partly test' and False:
    a = Torch_Ring_buffer_1D(3,3,dtype=torch.int32)
    a.pushback(torch.tensor([11,11,11]),overwrite=True)
    a.pushback(torch.tensor([22,22,22]),overwrite=True)
    a.pushback(torch.tensor([33,33,33]),overwrite=True)
    a.pushback(torch.tensor([44,44,44]),overwrite=True)
    #a._pushback_partly(torch.tensor([55,55]),overwrite=True)
    print(a.data)
    print(a)
    
    # a = Torch_Ring_buffer_1D(3,2,dtype=torch.int32)
    # a._pushback_partly(torch.tensor([11]),overwrite=True)
    # print(a)
    # a._pushback_partly(torch.tensor([22]),overwrite=True)
    # print(a)
    # a._pushback_partly(torch.tensor([33]),overwrite=True)
    # print(a)
    # a._pushback_partly(torch.tensor([44]),overwrite=True)
    # print(a)
    pass


if 'create test' and False:
    a = Torch_Ring_buffer_1D(2,1,dtype=torch.int32)
    b = a.get_max_cap_as_tensor()
    c = a.__len__()
    a.pushback(torch.tensor(3),overwrite=True)
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    a.rotate_to_preferred()
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    a.pushback(torch.tensor(4),overwrite=True)
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    a.rotate_to_preferred()
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    a.pushback(torch.tensor(5),overwrite=True)
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    a.rotate_to_preferred()
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    
    a = Torch_Ring_buffer_1D(2,1,dtype=torch.int32)
    a.pushback(torch.tensor(3),overwrite=True)
    a.pushback(torch.tensor(4),overwrite=True)
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    a.rotate_to_preferred()
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    
    
    a = Torch_Ring_buffer_1D(2,1,dtype=torch.int32)
    a.pushback(torch.tensor(3),overwrite=True)
    a.pushback(torch.tensor(4),overwrite=True)
    a.pushback(torch.tensor(5),overwrite=True)
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    a.rotate_to_preferred()
    print(a)
    print(a.get_max_cap_as_tensor())
    print(a.__len__())
    
    pass
   
    
    
