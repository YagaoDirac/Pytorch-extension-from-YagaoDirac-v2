import torch
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal


if "shape" and False:
    conv_1d = torch.nn.Conv1d(in_channels = 2, out_channels = 3,
                        kernel_size = 5)
    assert conv_1d.weight.shape == torch.Size([3,2,5])
    assert conv_1d.bias
    assert conv_1d.bias.shape == torch.Size([3])
    
    conv_2d = torch.nn.Conv2d(in_channels = 2, out_channels = 3,
                        kernel_size = (5,7))
    assert conv_2d.weight.shape == torch.Size([3,2,5,7])
    assert conv_2d.bias
    assert conv_2d.bias.shape == torch.Size([3])
    pass

if "basic" and False:
    #basic
    input = torch.tensor([[1.,2,5]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1, bias= False)
    assert conv.weight.shape == torch.Size([1,1,1])
    conv.weight.data = torch.tensor([[[2.]]])
    assert conv.weight.shape == torch.Size([1,1,1])
    output:torch.Tensor = conv(input)
    assert _tensor_equal(output, [[2.,4,10]])
    
    #in channel
    input = torch.tensor([[1.,2,5],[1.,3,6]])
    conv = torch.nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 1, bias= False)
    assert conv.weight.shape == torch.Size([1,2,1])
    conv.weight.data = torch.tensor([[[2.],[3.]]])
    assert conv.weight.shape == torch.Size([1,2,1])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.,2,5]])*2. + torch.tensor([[1.,3,6]])*3.)
    
    #out channel
    input = torch.tensor([[1.,2,5]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 2, kernel_size = 1, bias= False)
    assert conv.weight.shape == torch.Size([2,1,1])
    conv.weight.data = torch.tensor([[[2.]],[[3.]]])
    assert conv.weight.shape == torch.Size([2,1,1])
    output = conv(input)
    assert _tensor_equal(output, [[2.,4,10],[3.,6,15]])
    
    #kernal size
    input = torch.tensor([[1.,2,5]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[2.,3.]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.,2]])*2. + torch.tensor([[2.,5]])*3.)
    #assert isinstance(output, torch.Tensor)
    assert input.shape[1]-1 == output.shape[1]
    pass

if "stride " and False:
    #stride= 1
    #.....
    #1133
    # 2244
    input = torch.tensor([[1.,2,3,4,5]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, stride = 1, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[11.,22]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*11+2*22, 2*11+3*22, 3*11+4*22, 4*11+5*22]]))
    
    #stride= 2
    #.....
    #1122
    input = torch.tensor([[1.,2,3,4,5]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, stride = 2, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[11.,22]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*11+2*22, 3*11+4*22]]))
    
    #stride= 3
    #.....
    #11 22
    input = torch.tensor([[1.,2,3,4,5]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, stride = 3, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[11.,22]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*11+2*22, 4*11+5*22]]))
    
    #stride= 4
    #.....
    #11
    input = torch.tensor([[1.,2,3,4,5]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, stride = 4, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[11.,22]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*11+2*22]]))
    pass

if "stride " and False:
    # valid looks the same as 0.
    #valid
    #...
    #11
    # 22
    input = torch.tensor([[1.,2,4]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 'valid', bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*1.01+2*1.0001, 2*1.01+4*1.0001]]), epsilon=1e-7)

    #same
    #...
    #113
    # 22
    input = torch.tensor([[1.,2,4]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 'same', bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*1.01+2*1.0001, 2*1.01+4*1.0001, 4*1.01]]), epsilon=1e-7)
    
    # 0, the default.
    #...
    #11
    # 22
    input = torch.tensor([[1.,2,4]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 0, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*1.01+2*1.0001, 2*1.01+4*1.0001]]), epsilon=1e-7)
    
    # 1
    #0...0
    #1133
    # 2244
    input = torch.tensor([[1.,2,4]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 1, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    #assert _tensor_equal(output, torch.tensor([[1.*11+2*22, 2*11+4*22, 4*11+8*22, 8*11+16*22, 16*11]]))
    assert _tensor_equal(output, torch.tensor([[1*1.0001, 1.*1.01+2*1.0001, 2*1.01+4*1.0001, 4*1.01]])
                                                                                    , epsilon=1e-7)
    
    # 2
    #00...00
    #113355
    # 224466
    input = torch.tensor([[1.,2,4]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 2, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    #assert _tensor_equal(output, torch.tensor([[1.*11+2*22, 2*11+4*22, 4*11+8*22, 8*11+16*22, 16*11]]))
    assert _tensor_equal(output, torch.tensor([[0, 1*1.0001, 1.*1.01+2*1.0001, 2*1.01+4*1.0001, 4*1.01, 0]])
                                                                                    , epsilon=1e-7)
    #00....00
    #11335577
    # 224466
    input = torch.tensor([[1.,2,4,8]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 2, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    #assert _tensor_equal(output, torch.tensor([[1.*11+2*22, 2*11+4*22, 4*11+8*22, 8*11+16*22, 16*11]]))
    assert _tensor_equal(output, torch.tensor([[0, 1*1.0001, 1.*1.01+2*1.0001, 2*1.01+4*1.0001,
                                                        4*1.01+8*1.0001, 8*1.01, 0]]), epsilon=1e-7)
    #00...00
    #111444
    # 222555
    #  333
    input = torch.tensor([[1.,2,4]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 3, padding = 2, bias= False)
    assert conv.weight.shape == torch.Size([1,1,3])
    conv.weight.data = torch.tensor([[[1.1,1.01,1.001]]])
    assert conv.weight.shape == torch.Size([1,1,3])
    output = conv(input)
    #assert _tensor_equal(output, torch.tensor([[1.*11+2*22, 2*11+4*22, 4*11+8*22, 8*11+16*22, 16*11]]))
    assert _tensor_equal(output, torch.tensor([[1*1.001, 1*1.01+2*1.001, 1.*1.1+2*1.01+4*1.001, 
                                                            2*1.1+4*1.01, 4*1.1]]), epsilon=1e-7)
    pass

if "dilation " and False:
    # 1, the default.
    #....
    #1133
    # 22
    input = torch.tensor([[1.,2,4,8]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 'valid',dilation = 1, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*1.01+2*1.0001, 2*1.01+4*1.0001, 4*1.01+8*1.0001,]]), epsilon=1e-7)
    
    # 2
    #....
    #1 1
    # 2 2
    input = torch.tensor([[1.,2,4,8]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 'valid',dilation = 2, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*1.01+4*1.0001, 2*1.01+8*1.0001]]), epsilon=1e-7)
    
    # 3
    #....
    #1  1
    input = torch.tensor([[1.,2,4,8]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 'valid',dilation = 3, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*1.01+8*1.0001]]), epsilon=1e-7)
    
    # 3, padding=1
    #0....0
    #1  1
    # 2  2
    #  3  3
    input = torch.tensor([[1.,2,4,8]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 1, dilation = 3, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[4*1.0001, 1*1.01+8*1.0001, 2*1.01]]), epsilon=1e-7)
    
    pass

if "groups" and True:
    # 1, the default.
    input = torch.tensor([[1.,2,4,8]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 'valid',
                            dilation = 1, groups = 1, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*1.01+2*1.0001, 2*1.01+4*1.0001, 4*1.01+8*1.0001,]]), epsilon=1e-7)
    
    # groups doesn't affect the result.
    for _ in range(11):
        1w
        DIM = 21
        input = torch.randn(size=[2,DIM])    
        conv_group_1 = torch.nn.Conv1d(in_channels = 2, out_channels = 2, kernel_size = 2, padding = 'valid',
                            dilation = 1, groups = 1, bias= False)
        conv_group_2 = torch.nn.Conv1d(in_channels = 2, out_channels = 2, kernel_size = 2, padding = 'valid',
                            dilation = 1, groups = 2, bias= False)
        assert conv_group_1.weight.shape == conv_group_2.weight.shape
        conv_group_2.weight.data == conv_group_1.weight.data.detach().clone()
        output_group_1 = conv_group_1(input)
        output_group_2 = conv_group_2(input)
        assert _tensor_equal(output_group_1, output_group_2)
        pass
    
    
    pass













# 1w 还要padding没测？
# padding_mode   padding=0, dilation=1, groups=1,
for _ in range(55):#
    batch = random.randint(5,15)
    in_channels = random.randint(5,15)
    out_channels = random.randint(5,15)
    kernel_size = random.randint(2,5)
    stride=random.randint(1,3)
    resolution_in = random.randint(50,100)
    
    conv = torch.nn.Conv1d(in_channels = in_channels, out_channels = out_channels,
                        kernel_size = kernel_size, stride = stride)
    input = torch.randn(size=[batch, in_channels, resolution_in])
    output = conv(input)

    #resolution_out = (resolution_in-kernel_size+1)//stride
    resolution_out = (resolution_in-kernel_size+1-1)//stride+1
    assert output.shape == torch.Size([batch, out_channels, resolution_out])
    pass

fds=432

# a = torch.tensor([0.,0,1,0,0])
# conv = torch.nn.Conv1d(,)
