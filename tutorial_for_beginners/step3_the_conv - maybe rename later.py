import torch
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _float_equal, _tensor_equal

# aaa_out = 123
# def __test_func():
#     aaa_in = 456
#     print(aaa_in)
#     print(aaa_out)
#     pass
# __test_func()

# def __test_func():
#     #  v ctrl space here to check out the auto completion.
#     aaa
#     #  ^ ctrl space here to check out the auto completion.
#     pass
# __test_func()



def ____shape():
    conv_1d = torch.nn.Conv1d(in_channels = 2, out_channels = 3,
                        kernel_size = 5)
    assert conv_1d.weight.shape == torch.Size([3,2,5])
    assert conv_1d.bias is not None
    assert conv_1d.bias.shape == torch.Size([3])
    
    conv_2d = torch.nn.Conv2d(in_channels = 2, out_channels = 3,
                        kernel_size = (5,7))
    assert conv_2d.weight.shape == torch.Size([3,2,5,7])
    assert conv_2d.bias is not None
    assert conv_2d.bias.shape == torch.Size([3])
    pass
____shape()

def ____basic():
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
____basic()


def ____stride():
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
____stride()


def ____padding():
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
____padding()


def ____dilation():
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
____dilation()


def ____groups():
    # 1, the default.
    input = torch.tensor([[1.,2,4,8]])
    conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 2, padding = 'valid',
                            dilation = 1, groups = 1, bias= False)
    assert conv.weight.shape == torch.Size([1,1,2])
    conv.weight.data = torch.tensor([[[1.01,1.0001]]])
    assert conv.weight.shape == torch.Size([1,1,2])
    output = conv(input)
    assert _tensor_equal(output, torch.tensor([[1.*1.01+2*1.0001, 2*1.01+4*1.0001, 4*1.01+8*1.0001,]]), epsilon=1e-7)
    
    # 2
    #let's say input consists of in1 and in2.
    #weight consists of w1 and w2
    # output is out1 and out2
    # the behavior of group is :
    #out1 = in1 @ w1, out2 = in2 @ w2. Everything stays in group.
    #but a normal conv is 
    #out1 = in @ w1, out2 = in @ w2. The only input goes to every output
    #so if you have 2 conv with the exactly same shape and same behavior, you can merge them into 
    #a single conv with groups=2.
    input = torch.tensor([[1.,2,4],[1.,3,9]])
    conv_group_1 = torch.nn.Conv1d(in_channels = 1, out_channels = 2, kernel_size = 2, padding = 'valid',
                        dilation = 1, groups = 1, bias= False)
    conv_group_2 = torch.nn.Conv1d(in_channels = 2, out_channels = 2, kernel_size = 2, padding = 'valid',
                        dilation = 1, groups = 2, bias= False)
    
    assert conv_group_1.weight.shape == torch.Size([2,1,2])
    conv_group_1.weight.data = torch.tensor([[[1.101,3.101]],[[1.11,2.11]]])
    assert conv_group_1.weight.shape == torch.Size([2,1,2])
    assert conv_group_2.weight.shape == torch.Size([2,1,2])
    conv_group_2.weight.data = conv_group_1.weight.data.detach().clone()
    assert conv_group_2.weight.shape == torch.Size([2,1,2])
    
    output_group_1 = torch.empty(size=[2,2])
    _g1_conv_the_input_0 = conv_group_1(input[0].reshape([1,-1]))
    _g1_conv_the_input_1 = conv_group_1(input[1].reshape([1,-1]))
    output_group_1[0] = conv_group_1(input[0].reshape([1,-1]))[0]
    output_group_1[1] = conv_group_1(input[1].reshape([1,-1]))[1]
    output_group_2 = conv_group_2(input)
    assert _tensor_equal(output_group_1, output_group_2)
    
    for _ in range(11):
        DIM = random.randint(5,15)
        input = torch.randn(size=[2, DIM])
        conv_group_1 = torch.nn.Conv1d(in_channels = 1, out_channels = 2, kernel_size = 2, padding = 'valid',
                            dilation = 1, groups = 1, bias= False)
        conv_group_2 = torch.nn.Conv1d(in_channels = 2, out_channels = 2, kernel_size = 2, padding = 'valid',
                            dilation = 1, groups = 2, bias= False)
        
        assert conv_group_1.weight.shape == torch.Size([2,1,2])
        assert conv_group_2.weight.shape == torch.Size([2,1,2])
        conv_group_2.weight.data = conv_group_1.weight.data.detach().clone()
        
        output_group_1 = torch.empty(size=[2,DIM-1])
        output_group_1[0] = conv_group_1(input[0].reshape([1,-1]))[0]
        output_group_1[1] = conv_group_1(input[1].reshape([1,-1]))[1]
        output_group_2 = conv_group_2(input)
        assert _tensor_equal(output_group_1, output_group_2)
        pass
    
    # 3
    for _ in range(11):
        DIM = random.randint(5,15)
        input = torch.randn(size=[3, DIM])
        conv_group_1 = torch.nn.Conv1d(in_channels = 1, out_channels = 3, kernel_size = 2, padding = 'valid',
                            dilation = 1, groups = 1, bias= False)
        conv_group_3 = torch.nn.Conv1d(in_channels = 3, out_channels = 3, kernel_size = 2, padding = 'valid',
                            dilation = 1, groups = 3, bias= False)
        
        assert conv_group_1.weight.shape == torch.Size([3,1,2])
        assert conv_group_3.weight.shape == torch.Size([3,1,2])
        conv_group_3.weight.data = conv_group_1.weight.data.detach().clone()
        
        output_group_1 = torch.empty(size=[3,DIM-1])
        output_group_1[0] = conv_group_1(input[0].reshape([1,-1]))[0]
        output_group_1[1] = conv_group_1(input[1].reshape([1,-1]))[1]
        output_group_1[2] = conv_group_1(input[2].reshape([1,-1]))[2]
        output_group_3 = conv_group_3(input)
        assert _tensor_equal(output_group_1, output_group_3)
        pass
    
    # group on multiple channels.
    
    input = torch.tensor([[1.],[2],[4],[8],[16],[32]])
    conv_group_1 = torch.nn.Conv1d(in_channels = 3,   out_channels = 2, kernel_size = 1, padding = 'valid',
                        dilation = 1, groups = 1, bias= False)
    conv_group_2 = torch.nn.Conv1d(in_channels = 3*2, out_channels = 2, kernel_size = 1, padding = 'valid',
                        dilation = 1, groups = 2, bias= False)
    
    assert conv_group_1.weight.shape == torch.Size([2,3,1])
    conv_group_1.weight.data = torch.tensor([[[1.1],[1.01],[1.001]],[[10.1],[10.01],[10.001]]])
    assert conv_group_1.weight.shape == torch.Size([2,3,1])
    assert conv_group_2.weight.shape == torch.Size([2,3,1])
    conv_group_2.weight.data = conv_group_1.weight.data.detach().clone()
    assert conv_group_2.weight.shape == torch.Size([2,3,1])
    
    output_group_2 = conv_group_2(input)
    assert _tensor_equal(output_group_2,    [[1*1.1 +  2*1.01 +   4*1.001],
                                            [8*10.1 + 16*10.01 + 32*10.001]])
    output_group_1 = torch.empty(size=[2,1])
    _g1_conv_the_input_0 = conv_group_1(input[:3])
    assert _tensor_equal(_g1_conv_the_input_0,  [[1*1.1 +  2*1.01 +  4*1.001],
                                    [1*10.1 + 2*10.01 + 4*10.001]])
    _g1_conv_the_input_1 = conv_group_1(input[3:])
    assert _tensor_equal(_g1_conv_the_input_1,  [[8*1.1 +  16*1.01 +  32*1.001],
                                    [8*10.1 + 16*10.01 + 32*10.001]])
    
    output_group_1[0] = conv_group_1(input[:3])[0]
    output_group_1[1] = conv_group_1(input[3:])[1]
    assert _tensor_equal(output_group_1, output_group_2)
    pass
____groups()








def dimention_test():
    for _ in range(123):
        batch = random.randint(1,9)
        in_channels_per_group = random.randint(1,5)
        out_channels = random.randint(1,7)
        kernel_size = random.randint(2,5)
        stride = random.randint(1,3)
        padding = random.randint(0,3)
        dilation = random.randint(1,3)
        groups = random.randint(1,3)
        in_channels = in_channels_per_group*groups
        
        resolution_in = random.randint(15,200)
        
        conv = torch.nn.Conv1d(in_channels = in_channels, out_channels = out_channels,
                            kernel_size = kernel_size, stride = stride, padding = padding,
                            dilation = dilation)
        #groups is a bit special. Not included here.
        
        
        input = torch.randn(size=[batch, in_channels, resolution_in])
        assert input.nelement()<=9*5*3*200
        output = conv(input)

        #now manually calc the dimention of output
        dim_of__input_with_padding = resolution_in+padding*2# ppiiiiiiipp, 7i(resolution_in 7), 2p each side(padding 2)
        dim_of__effective_kernal = (kernel_size-1)*dilation+1#o..o..o..o, 4o(kernal_size 4), dilation is 3
        dim_of_non_first_output_element = dim_of__input_with_padding-dim_of__effective_kernal
        #1111111????????????, 7 1s, idk how many ?s. This line calc the amount of ?s.
        #1(used for the first element in output), ?(only for the rest)
        if dim_of_non_first_output_element<0:
            #not enough for a single output element. So nothing is outputed.
            resolution_out = 0
            assert output.nelement() == 0
            pass
        else:#at least 1 output element, dim_of_non_first_output_element>=0
            extra_output_dim = dim_of_non_first_output_element//stride
            # the amount of output dimention excluded the first one.
            
            resolution_out = extra_output_dim+1
            assert output.shape == torch.Size([batch, out_channels, resolution_out])
        pass#/for _
    return
dimention_test()




