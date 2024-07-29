import torch

torch.nn.modules.container.ModuleList

class MyLayer(torch.nn.Module):
    def __init__(self, index:int):
        super().__init__()
        self.index = index

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        print(self.index)
        return x
    
    
class MyModuleList(torch.nn.Module):
    def __init__(self, index:int):
        super().__init__()
        self.layers = torch.nn.modules.container.ModuleList([MyLayer(i) for i in range(index)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for layer in self.layers:
            x = layer(x)
        return x
    
model = MyModuleList(3)
input = torch.ones([42])
model(input)



        
        
        
