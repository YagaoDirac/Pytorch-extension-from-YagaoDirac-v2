from typing import Callable
import torch

a = torch.tensor([1,2,3])
b = a.expand([2,-1])
c = b.reshape(-1)
print(c)



# class Test_Config:
#     def __init__(self, log_file:str = "out.txt"):
#         self.filename = log_file
#         pass
    
#     def add_config(self, config_name)
    
#     def run(self, test_func:Callable):
#         test_func